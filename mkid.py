"""
mkid_simulator.py

Simulates MKID IQ timestream data including:
- Detector signal (photon pulses with energy-dependent amplitude)
- Two-level system (TLS) noise (1/f in frequency domain)
- HEMT amplifier noise (white, in IQ)
- LO phase noise (common-mode, multiplicative)
- Cleaning tone timestreams (same common-mode, no detector signal)

Usage:
    from mkid_simulator import MKIDSimulator

    sim = MKIDSimulator(fr=5.0e9, Qi=200000, Qc=50000)
    data = sim.generate_timestream(duration=10.0, photon_rate=50.0)
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Tuple, List
import scipy.special as spec

Planck_h = 6.626e-34
Boltz_k = 1.381e-23
eV_to_J = 1.602e-19
N_0 = 1.72e10 * eV_to_J  # m^-3 J^-1

 
@dataclass
class ResonatorParams:
    """Parameters defining the superconducting resonator."""
    fr: float = 5.0e9          # Resonance frequency (Hz)
    Qi: float = 200000         # Internal quality factor
    Qc: float = 50000          # Coupling quality factor
    phi_c: float = 0.1         # Asymmetry angle (rad)
    a: float = 5.0+2j             # Overall gain
    tau: float = 50e-9         # Cable delay (s)

    @property
    def Qr(self):
        """Total (loaded) quality factor."""
        return 1.0 / (1.0 / self.Qi + np.real(1.0 / (self.Qc * np.exp(1j * self.phi_c))))

    @property
    def linewidth(self):
        """Resonator linewidth (Hz)."""
        return self.fr / self.Qr





# ============================================================
# ADD these module-level physics functions (before the MKID class)
# ============================================================

def _n_qp(T, Delta0):
    """Thermal quasiparticle density (m^-3)."""
    T = np.asarray(T, dtype=float)
    x = np.clip(Delta0 / (Boltz_k * T), 0, 500)
    return 2.0 * N_0 * np.sqrt(2 * np.pi * Boltz_k * T * Delta0) * np.exp(-x)


def _kappa_1(T, f0, Delta0):
    """Mattis-Bardeen kernel κ₁ (dissipation, m³)."""
    T = np.asarray(T, dtype=float)
    xi = np.clip(0.5 * Planck_h * f0 / (Boltz_k * T), 1e-10, 500)
    return (1.0 / (np.pi * Delta0 * N_0)) * \
           np.sqrt(2.0 * Delta0 / (np.pi * Boltz_k * T)) * \
           np.sinh(xi) * spec.k0(xi)


def _kappa_2(T, f0, Delta0):
    """Mattis-Bardeen kernel κ₂ (reactive, m³)."""
    T = np.asarray(T, dtype=float)
    xi = np.clip(0.5 * Planck_h * f0 / (Boltz_k * T), 1e-10, 500)
    return (1.0 / (2.0 * Delta0 * N_0)) * \
           (1.0 + np.sqrt(2.0 * Delta0 / (np.pi * Boltz_k * T)) * \
            np.exp(-xi) * spec.i0(xi))


def _fr_of_T(T, f0, Delta0, alpha_f):
    """Resonance frequency vs temperature (Hz)."""
    T = np.asarray(T, dtype=float)
    return f0 * (1.0 - 0.5 * alpha_f * _kappa_2(T, f0, Delta0) * _n_qp(T, Delta0))


def _Qi_of_T(T, f0, Qi0, Delta0, alpha_Q):
    """Internal quality factor vs temperature."""
    T = np.asarray(T, dtype=float)
    return 1.0 / (alpha_Q * _kappa_1(T, f0, Delta0) * _n_qp(T, Delta0) + 1.0 / Qi0)


def _heat_capacity(T, Delta0, volume):
    """
    Total heat capacity: BCS electronic + Debye phonon.
    Prevents division by zero at very low T.
    """
    a_BCS = 2.0 * np.sqrt(2 * np.pi)
    x = np.clip(Delta0 / (Boltz_k * T), 0, 500)
    C_el = N_0 * volume * Boltz_k * a_BCS * x**2 * np.exp(-x)

    # Debye phonon floor (Al: Θ_D = 428 K, n_atoms = 6e28 m^-3)
    Theta_D = 428.0
    n_atoms = 6.0e28
    b_ph = (12.0 / 5.0) * np.pi**4 * n_atoms * Boltz_k / Theta_D**3
    C_ph = b_ph * volume * T**3

    return C_el + C_ph



@dataclass
class PulseParams:
    """Parameters defining photon pulse response via thermal model."""
    tau_rise: float = 1.0e-6       # Quasiparticle thermalization time (s)
    tau_fall: float = 50.0e-6      # Quasiparticle recombination time (s)

    # Material parameters
    Tc: float = 1.2                # Critical temperature (K)
    alpha_f: float = 0.05          # Kinetic inductance fraction (freq shift)
    alpha_Q: float = 0.05          # Kinetic inductance fraction (dissipation)
    volume: float = 1.2e-10       # Absorber volume (m³)
    T_bath: float = 0.1            # Bath temperature (K)

    # Heat capacity: set directly for tutorial dynamic range
    # If None, uses physical BCS + phonon model
    C_override: Optional[float] = None  # J/K — set this to bypass exponential C
    @property
    def Delta0(self):
        """BCS gap energy (J)."""
        return 1.764 * Boltz_k * self.Tc

    

@dataclass
class NoiseParams:
    """Parameters defining noise sources."""
    # TLS noise (1/f frequency noise on the resonator)
    tls_Sf0: float = 1.0e4         # TLS frequency noise PSD at 1 Hz (Hz^2/Hz)
    tls_alpha: float = 0.5         # TLS noise exponent (PSD ∝ 1/f^alpha)

    # HEMT amplifier noise (white, additive in IQ)
    hemt_noise_temp: float = 4.0   # HEMT noise temperature (K)
    hemt_sigma: float = 0.005      # Std dev of white noise in IQ (fractional units of S21)

    # LO phase noise (multiplicative, common-mode)
    lo_Sphi0: float = 1e-4         # LO phase noise PSD at 1 Hz (rad^2/Hz)
    lo_alpha: float = 1.0          # LO phase noise exponent (PSD ∝ 1/f^alpha)
    lo_white: float = 1e-7         # LO white phase noise floor (rad^2/Hz)

    # Readout photon noise (shot noise from readout power)
    photon_noise_sigma: float = 0.002  # Readout photon noise (fractional IQ units)


@dataclass
class ReadoutParams:
    """Parameters defining the readout configuration."""
    f_tone: Optional[float] = None   # Tone frequency (Hz); None = place at fr
    f_clean: List[float] = field(default_factory=lambda: [])  # Cleaning tone freqs
    fs: float = 200.0                # Output sample rate after decimation (Hz)
    tone_power_dBm: float = -70      # Readout power at device


@dataclass
class TimeStreamData:
    """Container for output timestream data."""
    t: np.ndarray                          # Time array (s)
    I_det: np.ndarray                      # Detector tone I(t)
    Q_det: np.ndarray                      # Detector tone Q(t)
    I_clean: List[np.ndarray]              # Cleaning tone I(t) for each clean tone
    Q_clean: List[np.ndarray]              # Cleaning tone Q(t) for each clean tone

    # Truth information (for answer key)
    fr_t: np.ndarray                       # True resonance frequency vs time
    Qi_t: np.ndarray                       # True Qi vs time
    pulse_times: np.ndarray                # Photon arrival times
    pulse_energies: np.ndarray             # Photon energies (eV)
    delta_fr_true: np.ndarray              # True frequency shift signal

    # Noise decomposition (for answer key)
    noise_tls: np.ndarray                  # TLS noise contribution (in fr, Hz)
    noise_lo_phase: np.ndarray             # LO phase noise (rad)
    noise_hemt_I: np.ndarray               # HEMT noise I component
    noise_hemt_Q: np.ndarray               # HEMT noise Q component


# ============================================================
# MAIN SIMULATOR CLASS
# ============================================================

class MKID:
    """
    Simulate MKID IQ timestream data with realistic noise.

    Example:
        sim = MKID(fr=5.0e9, Qi=200000, Qc=50000)
        data = sim.timestream(duration=10.0, photon_rate=50.0)

        # Plot raw IQ
        plt.plot(data.I_det, data.Q_det, ',')

        # Access truth
        plt.plot(data.t, data.delta_fr_true)
    """

    def __init__(
        self,
        fr: float = 5.0e9,
        Qi: float = 200000,
        Qc: float = 50000,
        phi_c: float = 0.02,
        a: float = 5+2j,
        tau: float = 50e-9,
        resonator_params: Optional[ResonatorParams] = None,
        pulse_params: Optional[PulseParams] = None,
        noise_params: Optional[NoiseParams] = None,
        readout_params: Optional[ReadoutParams] = None,
        seed: Optional[int] = None,
    ):
        """
        Initialize the MKID simulator.

        Can pass individual resonator parameters or full dataclass objects.
        """
        if resonator_params is not None:
            self.resonator = resonator_params
        else:
            self.resonator = ResonatorParams(fr=fr, Qi=Qi, Qc=Qc,
                                             phi_c=phi_c, a=a, tau=tau)

        self.pulse = pulse_params or PulseParams()
        self.noise = noise_params or NoiseParams()
        self.readout = readout_params or ReadoutParams()

        # Default: tone on resonance, cleaning tones ±10 linewidths away
        if self.readout.f_tone is None:
            self.readout.f_tone = self.resonator.fr
        if not self.readout.f_clean:
            lw = self.resonator.linewidth
            self.readout.f_clean = [
                self.resonator.fr - 10 * lw,
                self.resonator.fr + 10 * lw,
            ]

        self.rng = np.random.default_rng(seed)

    # --------------------------------------------------------
    # RESONATOR MODEL
    # --------------------------------------------------------

    def S21(self, f: np.ndarray, fr: float = None, Qi: float = None, 
            fr_cable: float = None) -> np.ndarray:
        """
        Compute S21(f) for the resonator.

        S21(f) = a * exp(-2πi τ (f - fr_cable)) * (1 - Qr/Qc_hat / (1 + 2i Qr (f-fr)/fr))

        Args:
            f: frequency array (Hz)
            fr: resonance frequency (overrides self.resonator.fr if given)
            Qi: internal Q (overrides self.resonator.Qi if given)
            fr_cable: reference frequency for cable delay phase. 
                      If None, uses fr (standard VNA sweep behavior).
                      Set to a FIXED value for timestream simulations where
                      fr moves but the cable doesn't.

        Returns:
            Complex S21
        """
        r = self.resonator
        fr = fr if fr is not None else r.fr
        Qi = Qi if Qi is not None else r.Qi
        Qc_hat = r.Qc * np.exp(1j * r.phi_c)
        Qr = 1.0 / (1.0 / Qi + np.real(1.0 / Qc_hat))
        x = (f - fr) / fr

        # Cable delay: use fixed reference if provided, otherwise fr
        # For a VNA sweep: fr_cable = fr (cable phase is referenced to resonance)
        # For a timestream: fr_cable = fixed (cable doesn't move when resonance shifts)
        fr_ref = fr_cable if fr_cable is not None else fr
        cable = r.a * np.exp(-2j * np.pi * r.tau * (f - fr_ref))

        resonance = 1.0 - (Qr / Qc_hat) / (1.0 + 2j * Qr * x)
        return cable * resonance

    def dS21_dfr(self, f: float, fr: float = None, Qi: float = None) -> complex:
        """Numerical derivative dS21/dfr at a single frequency point."""
        fr = fr if fr is not None else self.resonator.fr
        df = 1.0  # 1 Hz perturbation
        s21_plus = self.S21(np.array([f]), fr=fr + df, Qi=Qi)[0]
        s21_minus = self.S21(np.array([f]), fr=fr - df, Qi=Qi)[0]
        return (s21_plus - s21_minus) / (2 * df)

    # --------------------------------------------------------
    # PULSE GENERATION
    # --------------------------------------------------------

    def _pulse_template(self, t: np.ndarray, t0: float) -> np.ndarray:
        """
        Normalized double-exponential pulse template, peak = 1.

        Args:
            t: time array
            t0: pulse start time
        """
        tau_r = self.pulse.tau_rise
        tau_f = self.pulse.tau_fall

        pulse = np.zeros_like(t)
        mask = t >= t0
        dt = t[mask] - t0
        raw = np.exp(-dt / tau_f) - np.exp(-dt / tau_r)

        # Normalize peak to 1
        peak_time = (tau_f * tau_r / (tau_f - tau_r)) * np.log(tau_f / tau_r)
        norm = np.exp(-peak_time / tau_f) - np.exp(-peak_time / tau_r)
        pulse[mask] = raw / norm

        return pulse

    def generate_photon_events(
        self,
        duration: float,
        photon_rate: float = 50.0,
        energy_distribution: str = 'exponential',
        mean_energy: float = 1.5,
        fixed_energy: Optional[float] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate random photon arrival times and energies.

        Args:
            duration: total time (s)
            photon_rate: mean photon rate (Hz)
            energy_distribution: 'exponential', 'fixed', or 'uniform'
            mean_energy: mean photon energy for exponential dist (eV)
            fixed_energy: if set, all photons have this energy (eV)

        Returns:
            times: arrival times (s)
            energies: photon energies (eV)
        """
        # Poisson arrival times
        n_expected = int(photon_rate * duration * 1.5)  # oversample
        inter_arrival = self.rng.exponential(1.0 / photon_rate, size=n_expected)
        times = np.cumsum(inter_arrival)
        times = times[times < duration]

        # Energies
        n_photons = len(times)
        if fixed_energy is not None:
            energies = np.full(n_photons, fixed_energy)
        elif energy_distribution == 'exponential':
            energies = self.rng.exponential(mean_energy, size=n_photons)
        elif energy_distribution == 'uniform':
            energies = self.rng.uniform(0.5, 3.0 * mean_energy, size=n_photons)
        elif energy_distribution == 'fixed':
            energies = np.full(n_photons, mean_energy)
        else:
            raise ValueError(f"Unknown energy distribution: {energy_distribution}")

        return times, energies

    def _build_signal_timestream(
            self,
            t: np.ndarray,
            photon_times: np.ndarray,
            photon_energies: np.ndarray,
        ) -> Tuple[np.ndarray, np.ndarray]:
            """
            Build fr(t) and Qi(t) from photon events using physical model:
                photon energy → ΔT → T(t) → fr(T), Qi(T)
    
            Uses Mattis-Bardeen conductivity to compute the resonance
            frequency and quality factor shifts from quasiparticle generation.
    
            Returns:
                delta_fr: frequency shift vs time (Hz)
                delta_Qi_arr: Qi(t) - Qi0 vs time
            """
            p = self.pulse
            f0 = self.resonator.fr
            Qi0 = self.resonator.Qi
            Delta0 = p.Delta0
            T_bath = p.T_bath
    
            # Quiescent state
            fr_bath = _fr_of_T(T_bath, f0, Delta0, p.alpha_f)
            Qi_bath = _Qi_of_T(T_bath, f0, Qi0, Delta0, p.alpha_Q)
    
            # Build temperature timestream from all photon events
            T_t = np.full_like(t, T_bath)
            for t0, E in zip(photon_times, photon_energies):
                # Heat capacity: use override if set, otherwise physical model
                if p.C_override is not None:
                    C = p.C_override
                else:
                    C = _heat_capacity(T_bath, Delta0, p.volume)
                
                E_joules = E * eV_to_J
                delta_T_peak = E_joules / C

                # Clip to stay below Tc
                if delta_T_peak > 0.9 * (p.Tc - T_bath):
                    delta_T_peak = 0.9 * (p.Tc - T_bath)

    
                # Double-exponential pulse shape
                mask = t >= t0
                dt = t[mask] - t0
                raw = np.exp(-dt / p.tau_fall) - np.exp(-dt / p.tau_rise)
                peak_time = (p.tau_fall * p.tau_rise / (p.tau_fall - p.tau_rise)) * \
                            np.log(p.tau_fall / p.tau_rise)
                norm = np.exp(-peak_time / p.tau_fall) - np.exp(-peak_time / p.tau_rise)
    
                T_t[mask] += delta_T_peak * (raw / norm)
    
            # Convert T(t) → fr(t), Qi(t) via Mattis-Bardeen
            fr_t = _fr_of_T(T_t, f0, Delta0, p.alpha_f)
            Qi_t = _Qi_of_T(T_t, f0, Qi0, Delta0, p.alpha_Q)
    
            # Return as deltas from quiescent
            delta_fr = fr_t - fr_bath
            delta_Qi = Qi_t - Qi_bath
    
            return delta_fr, delta_Qi

    # --------------------------------------------------------
    # NOISE GENERATION
    # --------------------------------------------------------

    def _generate_colored_noise(
        self,
        n_samples: int,
        fs: float,
        S0: float,
        alpha: float,
        white_floor: float = 0.0,
    ) -> np.ndarray:
        """
        Generate colored noise with PSD: S(f) = S0 / f^alpha + white_floor

        Uses Fourier filtering of white noise.

        Args:
            n_samples: number of time samples
            fs: sample rate (Hz)
            S0: PSD amplitude at 1 Hz
            alpha: spectral exponent
            white_floor: white noise PSD floor

        Returns:
            Noise time series
        """
        freqs = np.fft.rfftfreq(n_samples, d=1.0 / fs)
        freqs[0] = freqs[1]  # Avoid division by zero at DC

        # Target amplitude spectrum: sqrt(PSD * df)
        psd = S0 / freqs**alpha + white_floor
        amplitude = np.sqrt(psd * fs / 2)  # Scale for rfft normalization

        # Generate white noise in Fourier domain
        phases = self.rng.uniform(0, 2 * np.pi, size=len(freqs))
        noise_fft = amplitude * np.exp(1j * phases)
        noise_fft[0] = 0  # No DC component

        # Inverse FFT to time domain
        noise = np.fft.irfft(noise_fft, n=n_samples)

        return noise

    def _generate_tls_noise(self, n_samples: int, fs: float) -> np.ndarray:
        """
        Generate TLS frequency noise.

        TLS noise causes fractional frequency fluctuations with PSD ∝ 1/f^alpha.
        Returns noise in units of Hz (frequency shift).
        """
        return self._generate_colored_noise(
            n_samples, fs,
            S0=self.noise.tls_Sf0,
            alpha=self.noise.tls_alpha,
        )

    def _generate_lo_phase_noise(self, n_samples: int, fs: float) -> np.ndarray:
        """
        Generate LO phase noise.

        LO phase noise is multiplicative and common to all tones.
        Returns noise in units of radians.
        """
        return self._generate_colored_noise(
            n_samples, fs,
            S0=self.noise.lo_Sphi0,
            alpha=self.noise.lo_alpha,
            white_floor=self.noise.lo_white,
        )

    def _generate_hemt_noise(self, n_samples: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate HEMT amplifier noise (white, additive in IQ).

        Returns:
            noise_I, noise_Q: independent white noise in I and Q
        """
        sigma = self.noise.hemt_sigma
        noise_I = self.rng.normal(0, sigma, size=n_samples)
        noise_Q = self.rng.normal(0, sigma, size=n_samples)
        return noise_I, noise_Q

    # --------------------------------------------------------
    # MAIN TIMESTREAM GENERATION
    # --------------------------------------------------------

    def timestream(
        self,
        duration: float = 10.0,
        photon_rate: float = 50.0,
        energy_distribution: str = 'exponential',
        mean_energy: float = 1.5,
        fixed_energy: Optional[float] = None,
        photon_times: Optional[np.ndarray] = None,
        photon_energies: Optional[np.ndarray] = None,
        include_tls: bool = True,
        include_hemt: bool = True,
        include_lo: bool = True,
    ) -> TimeStreamData:
        """
        Generate a complete simulated MKID IQ timestream.

        Args:
            duration: total observation time (s)
            photon_rate: mean photon event rate (Hz)
            energy_distribution: 'exponential', 'fixed', or 'uniform'
            mean_energy: mean photon energy (eV)
            fixed_energy: override all energies to this value (eV)
            photon_times: explicitly provide photon times (overrides rate)
            photon_energies: explicitly provide energies (must match times)
            include_tls: include TLS noise
            include_hemt: include HEMT noise
            include_lo: include LO phase noise

        Returns:
            TimeStreamData with all timestreams and truth information
        """
        fs = self.readout.fs
        n_samples = int(duration * fs)
        t = np.arange(n_samples) / fs

        # ---- Generate photon events ----
        if photon_times is not None:
            assert photon_energies is not None, "Must provide energies with times"
            p_times = photon_times
            p_energies = photon_energies
        else:
            p_times, p_energies = self.generate_photon_events(
                duration, photon_rate, energy_distribution, mean_energy, fixed_energy
            )

        # ---- Build true detector signal: delta_fr(t), delta_Qi(t) ----
        delta_fr, delta_Qi = self._build_signal_timestream(t, p_times, p_energies)
        fr_t = self.resonator.fr + delta_fr
        Qi_t = self.resonator.Qi + delta_Qi
        Qi_t = np.clip(Qi_t, 1000, None)  # Keep physical

        # ---- Generate noise ----
        # TLS noise (frequency noise intrinsic to resonator)
        noise_tls = self._generate_tls_noise(n_samples, fs) if include_tls else np.zeros(n_samples)

        # LO phase noise (common-mode, affects all tones)
        noise_lo = self._generate_lo_phase_noise(n_samples, fs) if include_lo else np.zeros(n_samples)

        # HEMT noise (white, independent per tone)
        if include_hemt:
            hemt_I_det, hemt_Q_det = self._generate_hemt_noise(n_samples)
        else:
            hemt_I_det = np.zeros(n_samples)
            hemt_Q_det = np.zeros(n_samples)

        # ---- Compute detector tone IQ ----
        f_tone = self.readout.f_tone
        fr_fixed = self.resonator.fr  # Fixed cable delay reference

        # S21 at readout tone with time-varying resonance + TLS noise
        S21_t = np.array([
            self.S21(np.array([f_tone]), fr=fr_t[i] + noise_tls[i], Qi=Qi_t[i],
                     fr_cable=fr_fixed)[0]
            for i in range(n_samples)
        ])

        # Apply LO phase noise (multiplicative: rotates IQ)
        S21_t *= np.exp(1j * noise_lo)

        # Add HEMT noise (additive white in IQ)
        I_det = S21_t.real + hemt_I_det
        Q_det = S21_t.imag + hemt_Q_det


        # ---- Compute cleaning tone IQ ----
        I_clean_list = []
        Q_clean_list = []

        for f_c in self.readout.f_clean:
            # Cleaning tone: S21 far from resonance (nearly constant)
            S21_clean = self.S21(np.array([f_c]))[0]  # Static (no detector signal)

            # Same LO phase noise (common-mode!)
            S21_clean_t = S21_clean * np.exp(1j * noise_lo)

            # Independent HEMT noise per cleaning tone
            if include_hemt:
                hemt_I_c, hemt_Q_c = self._generate_hemt_noise(n_samples)
            else:
                hemt_I_c = np.zeros(n_samples)
                hemt_Q_c = np.zeros(n_samples)

            I_clean_list.append(S21_clean_t.real + hemt_I_c)
            Q_clean_list.append(S21_clean_t.imag + hemt_Q_c)

        # ---- Package results ----
        return TimeStreamData(
            t=t,
            I_det=I_det,
            Q_det=Q_det,
            I_clean=I_clean_list,
            Q_clean=Q_clean_list,
            fr_t=fr_t,
            Qi_t=Qi_t,
            pulse_times=p_times,
            pulse_energies=p_energies,
            delta_fr_true=delta_fr,
            noise_tls=noise_tls,
            noise_lo_phase=noise_lo,
            noise_hemt_I=hemt_I_det,
            noise_hemt_Q=hemt_Q_det,
        )

    # --------------------------------------------------------
    # CONVENIENCE: FREQUENCY SWEEP
    # --------------------------------------------------------

    def frequency_sweep(
        self,
        f_start: float,
        f_stop: float,
        n_points: int = 1000,
        ifbw: float = 1e3,
        readout_power_dBm: float = -70,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate a simulated VNA/SMuRF frequency sweep.

        Mimics what you'd get from setting up a network analyzer sweep
        across your resonator.

        Args:
            f_start: sweep start frequency (Hz)
            f_stop: sweep stop frequency (Hz)
            n_points: number of frequency points in sweep
            ifbw: IF bandwidth of the measurement (Hz). Lower IFBW = longer
                integration per point = less noise, but slower sweep.
                Typical VNA values: 100 Hz – 10 kHz.
            readout_power_dBm: power at the device input (dBm). Sets SNR
                relative to amplifier noise. More negative = less power = 
                more noise, but avoids driving the resonator nonlinear.

        Returns:
            f: frequency array (Hz)
            S21_meas: measured (noisy) complex S21

        Notes:
            The noise level is set by the combination of IFBW and readout power:
            - Higher IFBW → less averaging → more noise per point
            - Lower readout power → signal closer to amplifier noise floor

            Noise sigma ≈ sqrt(kT * IFBW / P_readout) (simplified model)
            In practice this maps to a noise floor relative to |S21| = 1.
        """
        f = np.linspace(f_start, f_stop, n_points)

        # True S21
        S21_true = self.S21(f)

        # Noise model:
        # The VNA measures S21 = V_out / V_in. Noise on this measurement
        # comes from the amplifier (HEMT) noise integrated over the IFBW.
        #
        # SNR per point ∝ P_readout / (kT_noise * IFBW)
        # → noise sigma ∝ sqrt(kT_noise * IFBW / P_readout)
        #
        # We use a simplified model where:
        #   sigma = noise_scale * sqrt(IFBW / IFBW_ref) * sqrt(P_ref / P_readout)

        T_noise = self.noise.hemt_noise_temp  # Amplifier noise temperature (K)
        k_B = 1.381e-23  # Boltzmann constant (J/K)
        P_readout = 1e-3 * 10**(readout_power_dBm / 10)  # Convert dBm to Watts

        # Noise power per measurement point
        noise_power_per_point = k_B * T_noise * ifbw  # Watts

        # Noise sigma in S21 units (relative to input amplitude)
        # S21 = V_out/V_in, noise on V_out ~ sqrt(noise_power * Z0)
        # Normalized: sigma ~ sqrt(P_noise / P_signal)
        noise_sigma = np.sqrt(noise_power_per_point / P_readout)

        # Complex Gaussian noise (independent I and Q)
        noise = (
            self.rng.normal(0, noise_sigma, n_points)
            + 1j * self.rng.normal(0, noise_sigma, n_points)
        )

        S21_meas = S21_true + noise

        return f, S21_meas

    def timestream_fast(
        self,
        duration: float = 10.0,
        photon_rate: float = 50.0,
        energy_distribution: str = 'exponential',
        mean_energy: float = 1.5,
        fixed_energy: Optional[float] = None,
        photon_times: Optional[np.ndarray] = None,
        photon_energies: Optional[np.ndarray] = None,
        include_tls: bool = True,
        include_hemt: bool = True,
        include_lo: bool = True,
    ) -> TimeStreamData:
        """
        Vectorized version of generate_timestream (much faster for long durations).
        Uses broadcasting instead of per-sample S21 evaluation.
        """
        fs = self.readout.fs
        n_samples = int(duration * fs)
        t = np.arange(n_samples) / fs

        # ---- Photon events ----
        if photon_times is not None:
            p_times = photon_times
            p_energies = photon_energies
        else:
            p_times, p_energies = self.generate_photon_events(
                duration, photon_rate, energy_distribution, mean_energy, fixed_energy
            )

        # ---- True signal ----
        delta_fr, delta_Qi = self._build_signal_timestream(t, p_times, p_energies)
        fr_t = self.resonator.fr + delta_fr
        Qi_t = np.clip(self.resonator.Qi + delta_Qi, 1000, None)

        # ---- Noise ----
        noise_tls = self._generate_tls_noise(n_samples, fs) if include_tls else np.zeros(n_samples)
        noise_lo = self._generate_lo_phase_noise(n_samples, fs) if include_lo else np.zeros(n_samples)
        hemt_I_det, hemt_Q_det = self._generate_hemt_noise(n_samples) if include_hemt else (np.zeros(n_samples), np.zeros(n_samples))

        # ---- Vectorized S21 at detector tone ----
        f_tone = self.readout.f_tone
        fr_effective = fr_t + noise_tls  # TLS shifts resonance
        Qr_t = 1.0 / (1.0 / Qi_t + np.real(1.0 / (self.resonator.Qc * np.exp(1j * self.resonator.phi_c))))
        Qc_hat = self.resonator.Qc * np.exp(1j * self.resonator.phi_c)

        # Cable delay: CONSTANT for a fixed tone frequency
        # This is the gain and phase of the system at f_tone — it doesn't change
        # when the resonance moves.
        cable = self.resonator.a * np.exp(-2j * np.pi * self.resonator.tau * (f_tone - self.resonator.fr))
        
        x = (f_tone - fr_effective) / fr_effective
        resonance = 1.0 - (Qr_t / Qc_hat) / (1.0 + 2j * Qr_t * x)
        S21_t = cable * resonance

        # Apply LO phase noise and HEMT
        S21_t *= np.exp(1j * noise_lo)
        I_det = S21_t.real + hemt_I_det
        Q_det = S21_t.imag + hemt_Q_det

        # ---- Cleaning tones (vectorized) ----
        I_clean_list = []
        Q_clean_list = []
        for f_c in self.readout.f_clean:
            # Far off resonance: S21 is ~constant, just cable delay
            S21_clean_static = self.S21(np.array([f_c]))[0]
            S21_clean_t = S21_clean_static * np.exp(1j * noise_lo)  # Same LO noise

            if include_hemt:
                hc_I, hc_Q = self._generate_hemt_noise(n_samples)
            else:
                hc_I, hc_Q = np.zeros(n_samples), np.zeros(n_samples)

            I_clean_list.append(S21_clean_t.real + hc_I)
            Q_clean_list.append(S21_clean_t.imag + hc_Q)

        return TimeStreamData(
            t=t,
            I_det=I_det,
            Q_det=Q_det,
            I_clean=I_clean_list,
            Q_clean=Q_clean_list,
            fr_t=fr_t,
            Qi_t=Qi_t,
            pulse_times=p_times,
            pulse_energies=p_energies,
            delta_fr_true=delta_fr,
            noise_tls=noise_tls,
            noise_lo_phase=noise_lo,
            noise_hemt_I=hemt_I_det,
            noise_hemt_Q=hemt_Q_det,
        )


# ============================================================
# QUICK DEMO / VALIDATION
# ============================================================

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # Create simulator with default parameters
    sim = MKID(
        fr=5.0e9, Qi=200000, Qc=50000, phi_c=0.1, tau=50e-9,
        noise_params=NoiseParams(
            tls_Sf0=1e4,
            hemt_sigma=0.005,
            lo_Sphi0=1e-4,
        ),
        seed=42,
    )

    # Generate timestream
    print("Generating timestream...")
    data = sim.timestream_fast(
        duration=5.0,
        photon_rate=30.0,
        mean_energy=1.5,
    )
    print(f"  {len(data.pulse_times)} photon events")
    print(f"  {len(data.t)} time samples at {sim.readout.fs} Hz")

    # ---- Plot ----
    fig, axes = plt.subplots(3, 2, figsize=(14, 10))

    # IQ plane
    axes[0, 0].plot(data.I_det, data.Q_det, ',', alpha=0.3, color='steelblue')
    axes[0, 0].set_xlabel('I')
    axes[0, 0].set_ylabel('Q')
    axes[0, 0].set_title('Detector Tone: IQ Plane')
    axes[0, 0].set_aspect('equal')

    # Cleaning tone IQ
    axes[0, 1].plot(data.I_clean[0], data.Q_clean[0], ',', alpha=0.3, color='orange')
    axes[0, 1].set_xlabel('I')
    axes[0, 1].set_ylabel('Q')
    axes[0, 1].set_title('Cleaning Tone 1: IQ Plane')
    axes[0, 1].set_aspect('equal')

    # Time series I
    axes[1, 0].plot(data.t, data.I_det, lw=0.5)
    axes[1, 0].set_xlabel('Time (s)')
    axes[1, 0].set_ylabel('I')
    axes[1, 0].set_title('Detector I(t)')

    # Time series Q
    axes[1, 1].plot(data.t, data.Q_det, lw=0.5)
    axes[1, 1].set_xlabel('Time (s)')
    axes[1, 1].set_ylabel('Q')
    axes[1, 1].set_title('Detector Q(t)')

    # True signal
    axes[2, 0].plot(data.t, data.delta_fr_true / 1e3, lw=0.5, color='green')
    axes[2, 0].set_xlabel('Time (s)')
    axes[2, 0].set_ylabel('Δfr (kHz)')
    axes[2, 0].set_title('True Frequency Shift (Answer Key)')

    # Photon energy histogram
    axes[2, 1].hist(data.pulse_energies, bins=30, color='purple', alpha=0.7)
    axes[2, 1].set_xlabel('Photon Energy (eV)')
    axes[2, 1].set_ylabel('Count')
    axes[2, 1].set_title(f'Photon Energies ({len(data.pulse_energies)} events)')

    plt.tight_layout()
    plt.savefig('mkid_simulator_demo.png', dpi=150, bbox_inches='tight')
    plt.show()

    # ---- Frequency sweep demo ----
    f_sweep, S21_sweep = sim.frequency_sweep(4.9e9, 5.1e9, n_points=500, ifbw=1000, readout_power_dBm=-70)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(S21_sweep.real, S21_sweep.imag, '.-', markersize=2)
    axes[0].set_xlabel('Re(S21)')
    axes[0].set_ylabel('Im(S21)')
    axes[0].set_title('Frequency Sweep: IQ')
    axes[0].set_aspect('equal')

    axes[1].plot((f_sweep - sim.resonator.fr) / 1e3, 20 * np.log10(np.abs(S21_sweep)))
    axes[1].set_xlabel('f - fr (kHz)')
    axes[1].set_ylabel('|S21| (dB)')
    axes[1].set_title('Frequency Sweep: Magnitude')

    plt.tight_layout()
    plt.savefig('mkid_sweep_demo.png', dpi=150, bbox_inches='tight')
    plt.show()
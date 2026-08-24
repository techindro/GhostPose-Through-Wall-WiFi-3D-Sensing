"""
rf_signal_processor.py
=============================================================================
High-Throughput RF & Wi-Fi CSI Signal Preprocessing Pipeline:
- Linear Phase Sanitization (CFO & SFO Untwisting across subcarriers)
- MIMO Static Clutter & Ambient Multipath Cancellation
- Dual-Band Butterworth Filtering (Locomotion vs. Cardiopulmonary Micro-Doppler)
- Doppler-Time STFT Spectrogram & Micro-Movement Extraction
- Tensor Packaging for PyTorch Deep Neural Networks
=============================================================================
"""

import numpy as np
from scipy import signal
from typing import Tuple, Dict, Optional


class RFSignalProcessor:
    """
    Hardware-agnostic signal processor for multi-antenna, multi-subcarrier
    raw CSI (Channel State Information) streams.
    """
    def __init__(
        self,
        num_subcarriers: int = 64,
        sampling_rate_hz: float = 100.0,
        subcarrier_indices: Optional[np.ndarray] = None
    ):
        self.num_subcarriers = num_subcarriers
        self.fs = sampling_rate_hz
        
        # 802.11n/ac standard subcarrier index mapping (excluding guard/pilot bands if needed)
        if subcarrier_indices is None:
            self.subcarrier_idx = np.arange(-num_subcarriers // 2, num_subcarriers // 2)
        else:
            self.subcarrier_idx = subcarrier_indices
            
        # Design Butterworth Filter Banks using Second-Order Sections (SOS)
        self._init_filter_banks()

    def _init_filter_banks(self) -> None:
        """Instantiates SOS (Second-Order Sections) filters for numerical stability."""
        nyq = 0.5 * self.fs
        
        # 1. Locomotion / Kinematics Bandpass (0.5 Hz - 5.0 Hz)
        self.sos_locomotion = signal.butter(
            N=4, Wn=[0.5 / nyq, min(5.0 / nyq, 0.99)], btype='bandpass', output='sos'
        )
        
        # 2. Respiration Bandpass (0.1 Hz - 0.45 Hz: ~6 - 27 breaths/min)
        self.sos_respiration = signal.butter(
            N=3, Wn=[0.1 / nyq, 0.45 / nyq], btype='bandpass', output='sos'
        )
        
        # 3. Heart Rate / Cardiac Bandpass (0.8 Hz - 2.5 Hz: ~48 - 150 BPM)
        self.sos_cardiac = signal.butter(
            N=4, Wn=[0.8 / nyq, min(2.5 / nyq, 0.99)], btype='bandpass', output='sos'
        )

    def sanitize_phase(self, csi_raw: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Applies linear transformation to untwist phase across subcarriers:
        Eliminates SFO (Sampling Frequency Offset) linear slope and CFO (Carrier Frequency Offset) intercept.
        
        Args:
            csi_raw: Complex array of shape [Antennas, Subcarriers, Time]
        Returns:
            amplitude: [Antennas, Subcarriers, Time]
            sanitized_phase: [Antennas, Subcarriers, Time] (unwrapped & calibrated)
        """
        n_ant, n_sub, n_time = csi_raw.shape
        amplitude = np.abs(csi_raw)
        raw_phase = np.angle(csi_raw)
        
        # Unwrap phase along the subcarrier axis (axis=1)
        unwrapped_phase = np.unwrap(raw_phase, axis=1)
        
        k = self.subcarrier_idx.reshape(1, n_sub, 1)  # Broadcast shape
        k_1, k_n = self.subcarrier_idx[0], self.subcarrier_idx[-1]
        
        # Linear slope: a = (phi_N - phi_1) / (k_N - k_1)
        slope = (unwrapped_phase[:, -1:, :] - unwrapped_phase[:, 0:1, :]) / (k_n - k_1)
        
        # Offset: b = (1 / N) * sum(phi_i)
        offset = np.mean(unwrapped_phase, axis=1, keepdims=True)
        
        # Calibrated Phase: phi_hat = phi_i - a * k_i - b
        sanitized_phase = unwrapped_phase - (slope * k) - offset
        return amplitude, sanitized_phase

    def remove_static_clutter(self, amplitude: np.ndarray, alpha: float = 0.98) -> np.ndarray:
        """
        Removes stationary background multipath reflections using recursive EMA subtraction.
        
        Args:
            amplitude: [Antennas, Subcarriers, Time]
            alpha: IIR smoothing coefficient for static channel baseline
        Returns:
            dynamic_csi: [Antennas, Subcarriers, Time]
        """
        dynamic_csi = np.zeros_like(amplitude)
        for a in range(amplitude.shape[0]):
            static_est = amplitude[a, :, 0].copy()
            for t in range(amplitude.shape[2]):
                static_est = alpha * static_est + (1.0 - alpha) * amplitude[a, :, t]
                dynamic_csi[a, :, t] = amplitude[a, :, t] - static_est
        return dynamic_csi

    def compute_conjugate_phase_diff(self, csi_raw: np.ndarray) -> np.ndarray:
        """
        Computes conjugate cross-antenna phase difference between Rx1 and other Rx antennas.
        Cancels common oscillator jitter and SFO when antennas share the RF front-end.
        
        Args:
            csi_raw: [Antennas, Subcarriers, Time], complex
        Returns:
            cross_phase: [Antennas-1, Subcarriers, Time]
        """
        ref_antenna = csi_raw[0:1, :, :]  # Rx1 as spatial reference
        cross_csi = csi_raw[1:, :, :] * np.conj(ref_antenna)
        return np.angle(cross_csi)

    def extract_doppler_spectrogram(
        self,
        signal_1d: np.ndarray,
        nperseg: int = 64,
        noverlap: int = 48
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Extracts micro-Doppler frequency spectrum via Short-Time Fourier Transform (STFT).
        
        Args:
            signal_1d: Time series [Time] from dominant dynamic subcarrier
        Returns:
            frequencies, time_bins, spectrogram (Power Spectral Density)
        """
        f, t_bins, zxx = signal.stft(
            signal_1d,
            fs=self.fs,
            window='hann',
            nperseg=nperseg,
            noverlap=noverlap,
            boundary='zeros'
        )
        spectrogram = np.abs(zxx) ** 2
        return f, t_bins, spectrogram

    def extract_vital_signs(
        self,
        sanitized_phase: np.ndarray
    ) -> Dict[str, np.ndarray]:
        """
        Isolates cardiopulmonary chest displacements and computes respiration and cardiac rates.
        
        Args:
            sanitized_phase: [Antennas, Subcarriers, Time]
        Returns:
            Dictionary containing breathing waveform, cardiac waveform, and estimated rates.
        """
        # Select subcarrier with highest phase variance (dynamic sensitivity)
        var_per_sub = np.var(sanitized_phase, axis=(0, 2))
        best_sub_idx = int(np.argmax(var_per_sub))
        vital_phase_trace = np.mean(sanitized_phase[:, best_sub_idx, :], axis=0)
        
        # 1. Filter Respiration (0.1 - 0.45 Hz)
        if len(vital_phase_trace) > 18:
            resp_waveform = signal.sosfiltfilt(self.sos_respiration, vital_phase_trace)
            cardiac_waveform = signal.sosfiltfilt(self.sos_cardiac, vital_phase_trace)
        else:
            resp_waveform = vital_phase_trace
            cardiac_waveform = vital_phase_trace
        
        # Spectral rate extraction via FFT peak analysis
        n_samples = len(vital_phase_trace)
        freqs = np.fft.rfftfreq(n_samples, d=1.0 / self.fs)
        
        # Respiration Peak (0.1 - 0.45 Hz)
        resp_fft = np.abs(np.fft.rfft(resp_waveform))
        resp_mask = (freqs >= 0.1) & (freqs <= 0.45)
        if np.any(resp_mask) and len(freqs[resp_mask]) > 0:
            resp_bpm = freqs[resp_mask][np.argmax(resp_fft[resp_mask])] * 60.0
        else:
            resp_bpm = 16.0
        
        # Heart Rate Peak (0.8 - 2.5 Hz)
        cardiac_fft = np.abs(np.fft.rfft(cardiac_waveform))
        cardiac_mask = (freqs >= 0.8) & (freqs <= 2.5)
        if np.any(cardiac_mask) and len(freqs[cardiac_mask]) > 0:
            cardiac_bpm = freqs[cardiac_mask][np.argmax(cardiac_fft[cardiac_mask])] * 60.0
        else:
            cardiac_bpm = 72.0
        
        return {
            "respiration_waveform": resp_waveform,
            "cardiac_waveform": cardiac_waveform,
            "respiration_rate_brpm": float(resp_bpm),
            "heart_rate_bpm": float(cardiac_bpm)
        }

    def process_frame_tensor(self, csi_raw: np.ndarray) -> np.ndarray:
        """
        End-to-End tensor transformation preparing input for the PyTorch RFStudentNetwork.
        
        Input:
            csi_raw: Complex array [Antennas, Subcarriers, Time]
        Output:
            preprocessed_tensor: Float32 array [Channels, Subcarriers, Time]
            where Channels = [Dynamic Amp, Sanitized Phase, In-Phase (I), Quadrature (Q)]
        """
        amp, phase = self.sanitize_phase(csi_raw)
        dynamic_amp = self.remove_static_clutter(amp)
        
        i_comp = dynamic_amp * np.cos(phase)
        q_comp = dynamic_amp * np.sin(phase)
        
        # Concatenate along channel dimension: [4 * Antennas, Subcarriers, Time]
        feature_stack = np.concatenate([dynamic_amp, phase, i_comp, q_comp], axis=0)
        return feature_stack.astype(np.float32)

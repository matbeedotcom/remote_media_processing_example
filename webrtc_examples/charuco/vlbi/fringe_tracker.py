"""
Fringe Tracking for Real-time VLBI

Real-time fringe tracking system that maintains coherence across interferometric
baselines by tracking and compensating for atmospheric and instrumental phase
variations. Essential for long-exposure interferometry.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Deque
from dataclasses import dataclass, field
from collections import deque
import logging
import time
from threading import Lock

logger = logging.getLogger(__name__)


@dataclass
class FringeState:
    """Current fringe tracking state for a baseline."""
    
    baseline_id: str
    is_locked: bool = False
    phase_offset_rad: float = 0.0
    phase_rate_rad_per_sec: float = 0.0
    coherence_amplitude: float = 0.0
    lock_timestamp: float = 0.0
    tracking_error_rad: float = 0.0
    snr: float = 0.0
    
    # History for trend analysis
    phase_history: Deque[float] = field(default_factory=lambda: deque(maxlen=100))
    amplitude_history: Deque[float] = field(default_factory=lambda: deque(maxlen=100))
    timestamp_history: Deque[float] = field(default_factory=lambda: deque(maxlen=100))


@dataclass
class FringeCorrection:
    """Phase correction to apply to visibility data."""
    
    baseline_id: str
    phase_correction_rad: float
    correction_timestamp: float
    confidence: float = 1.0
    prediction_horizon_sec: float = 0.0


class PhaseTrackingLoop:
    """Phase-locked loop for tracking atmospheric phase variations."""
    
    def __init__(self, baseline_id: str, loop_bandwidth_hz: float = 0.1):
        """
        Initialize phase tracking loop.
        
        Args:
            baseline_id: Identifier for the baseline being tracked
            loop_bandwidth_hz: Tracking loop bandwidth in Hz
        """
        self.baseline_id = baseline_id
        self.bandwidth_hz = loop_bandwidth_hz
        
        # PLL parameters
        self.kp = 2 * loop_bandwidth_hz  # Proportional gain
        self.ki = (loop_bandwidth_hz ** 2)  # Integral gain
        
        # State variables
        self.phase_estimate_rad = 0.0
        self.frequency_estimate_hz = 0.0
        self.integrator_state = 0.0
        self.last_update_time = 0.0
        
        # Lock detection
        self.lock_threshold = 0.7  # Coherence threshold for lock
        self.unlock_threshold = 0.3  # Coherence threshold for unlock
        self.lock_time_constant = 10.0  # Seconds to average for lock detection
        
        logger.debug(f"📡 Initialized PLL for {baseline_id}: BW={loop_bandwidth_hz:.3f} Hz")
    
    def update(self, measured_phase: float, coherence: float, timestamp: float) -> Tuple[float, bool]:
        """
        Update phase tracking loop with new measurement.
        
        Args:
            measured_phase: Measured phase in radians
            coherence: Fringe coherence amplitude (0-1)
            timestamp: Measurement timestamp
            
        Returns:
            (predicted_phase, is_locked)
        """
        
        current_time = timestamp if timestamp > 0 else time.time()
        
        if self.last_update_time == 0:
            self.last_update_time = current_time
            return measured_phase, False
        
        dt = current_time - self.last_update_time
        if dt <= 0:
            return self.phase_estimate_rad, coherence > self.lock_threshold
        
        # Phase error (unwrapped)
        phase_error = self._unwrap_phase_error(measured_phase - self.phase_estimate_rad)
        
        # Update frequency estimate (integral term)
        self.integrator_state += self.ki * phase_error * dt
        self.frequency_estimate_hz = self.integrator_state
        
        # Update phase estimate
        phase_increment = 2 * np.pi * self.frequency_estimate_hz * dt
        self.phase_estimate_rad += phase_increment + self.kp * phase_error
        
        # Wrap phase to [-π, π]
        self.phase_estimate_rad = self._wrap_phase(self.phase_estimate_rad)
        
        # Lock detection
        is_locked = coherence > self.lock_threshold
        
        self.last_update_time = current_time
        
        return self.phase_estimate_rad, is_locked
    
    def predict_phase(self, prediction_time: float) -> float:
        """Predict phase at future time."""
        
        dt = prediction_time - self.last_update_time
        predicted_phase = self.phase_estimate_rad + 2 * np.pi * self.frequency_estimate_hz * dt
        
        return self._wrap_phase(predicted_phase)
    
    def _unwrap_phase_error(self, phase_error: float) -> float:
        """Unwrap phase error to handle 2π discontinuities."""
        
        while phase_error > np.pi:
            phase_error -= 2 * np.pi
        while phase_error < -np.pi:
            phase_error += 2 * np.pi
            
        return phase_error
    
    def _wrap_phase(self, phase: float) -> float:
        """Wrap phase to [-π, π] range."""
        
        return ((phase + np.pi) % (2 * np.pi)) - np.pi


class FringeTracker:
    """Real-time fringe tracking system for maintaining interferometric coherence."""
    
    def __init__(self, baseline_ids: List[str]):
        """
        Initialize fringe tracker.
        
        Args:
            baseline_ids: List of baseline identifiers to track
        """
        self.baseline_ids = baseline_ids
        
        # Tracking parameters
        self.coherence_integration_time = 1.0  # Seconds
        self.phase_prediction_horizon = 0.1    # Seconds ahead to predict
        self.min_snr_for_tracking = 5.0        # Minimum SNR to maintain lock
        
        # State for each baseline
        self.fringe_states: Dict[str, FringeState] = {}
        self.phase_loops: Dict[str, PhaseTrackingLoop] = {}
        
        # Initialize tracking loops
        for baseline_id in baseline_ids:
            self.fringe_states[baseline_id] = FringeState(baseline_id=baseline_id)
            self.phase_loops[baseline_id] = PhaseTrackingLoop(baseline_id)
        
        # Thread safety
        self.state_lock = Lock()
        
        logger.info(f"🎯 Initialized fringe tracker for {len(baseline_ids)} baselines")
    
    def track_fringes(self, visibility_data: List) -> List[FringeCorrection]:
        """
        Track fringes and generate phase corrections.
        
        Args:
            visibility_data: List of complex visibility measurements
            
        Returns:
            List of fringe corrections to apply
        """
        
        corrections = []
        current_time = time.time()
        
        with self.state_lock:
            # Group visibilities by baseline
            baseline_visibilities = self._group_by_baseline(visibility_data)
            
            # Process each baseline
            for baseline_id in self.baseline_ids:
                if baseline_id not in baseline_visibilities:
                    continue
                
                visibilities = baseline_visibilities[baseline_id]
                if not visibilities:
                    continue
                
                # Compute coherent average for this baseline
                coherence_amplitude, average_phase = self._compute_coherent_average(visibilities)
                
                # Update phase tracking loop
                state = self.fringe_states[baseline_id]
                pll = self.phase_loops[baseline_id]
                
                predicted_phase, is_locked = pll.update(
                    average_phase, coherence_amplitude, current_time
                )
                
                # Update fringe state
                self._update_fringe_state(state, coherence_amplitude, predicted_phase, 
                                        is_locked, current_time, visibilities)
                
                # Generate correction
                correction = self._generate_correction(baseline_id, state, current_time)
                if correction:
                    corrections.append(correction)
        
        # Log tracking status
        locked_count = sum(1 for state in self.fringe_states.values() if state.is_locked)
        logger.debug(f"📊 Fringe tracking: {locked_count}/{len(self.baseline_ids)} baselines locked")
        
        return corrections
    
    def _group_by_baseline(self, visibility_data: List) -> Dict[str, List]:
        """Group visibility measurements by baseline."""
        
        baseline_groups = {}
        
        for visibility in visibility_data:
            baseline_id = f"{visibility.baseline.camera1_id}_{visibility.baseline.camera2_id}"
            
            if baseline_id not in baseline_groups:
                baseline_groups[baseline_id] = []
            
            baseline_groups[baseline_id].append(visibility)
        
        return baseline_groups
    
    def _compute_coherent_average(self, visibilities: List) -> Tuple[float, float]:
        """Compute coherent average amplitude and phase from visibility measurements."""
        
        if not visibilities:
            return 0.0, 0.0
        
        # Vector average of complex visibilities
        complex_sum = sum(vis.complex_value for vis in visibilities)
        coherent_average = complex_sum / len(visibilities)
        
        coherence_amplitude = abs(coherent_average)
        average_phase = np.angle(coherent_average)
        
        return coherence_amplitude, average_phase
    
    def _update_fringe_state(self, state: FringeState, coherence_amplitude: float,
                           predicted_phase: float, is_locked: bool, timestamp: float,
                           visibilities: List):
        """Update fringe state with new measurements."""
        
        # Update basic state
        state.is_locked = is_locked
        state.coherence_amplitude = coherence_amplitude
        state.phase_offset_rad = predicted_phase
        
        # Compute SNR estimate
        if visibilities:
            snr_estimates = [vis.snr for vis in visibilities if vis.snr > 0]
            state.snr = np.mean(snr_estimates) if snr_estimates else 0.0
        
        # Update history
        state.phase_history.append(predicted_phase)
        state.amplitude_history.append(coherence_amplitude)
        state.timestamp_history.append(timestamp)
        
        # Compute phase rate if we have enough history
        if len(state.phase_history) >= 3:
            times = list(state.timestamp_history)[-3:]
            phases = list(state.phase_history)[-3:]
            
            # Simple linear fit for phase rate
            dt_total = times[-1] - times[0]
            if dt_total > 0:
                # Unwrap phases for rate calculation
                unwrapped_phases = self._unwrap_phase_sequence(phases)
                dphase = unwrapped_phases[-1] - unwrapped_phases[0]
                state.phase_rate_rad_per_sec = dphase / dt_total
        
        # Compute tracking error (RMS phase deviation)
        if len(state.phase_history) >= 5:
            recent_phases = list(state.phase_history)[-5:]
            phase_std = np.std(self._unwrap_phase_sequence(recent_phases))
            state.tracking_error_rad = phase_std
        
        # Update lock timestamp
        if is_locked and not hasattr(state, '_was_locked'):
            state.lock_timestamp = timestamp
        
        state._was_locked = is_locked
    
    def _unwrap_phase_sequence(self, phases: List[float]) -> List[float]:
        """Unwrap a sequence of phase measurements."""
        
        if not phases:
            return []
        
        unwrapped = [phases[0]]
        
        for i in range(1, len(phases)):
            diff = phases[i] - phases[i-1]
            
            # Unwrap 2π jumps
            while diff > np.pi:
                diff -= 2 * np.pi
            while diff < -np.pi:
                diff += 2 * np.pi
            
            unwrapped.append(unwrapped[-1] + diff)
        
        return unwrapped
    
    def _generate_correction(self, baseline_id: str, state: FringeState, 
                           current_time: float) -> Optional[FringeCorrection]:
        """Generate fringe correction for a baseline."""
        
        if not state.is_locked or state.snr < self.min_snr_for_tracking:
            return None
        
        # Predict phase at correction application time
        application_time = current_time + self.phase_prediction_horizon
        pll = self.phase_loops[baseline_id]
        predicted_phase = pll.predict_phase(application_time)
        
        # Phase correction is negative of predicted phase error
        phase_correction = -predicted_phase
        
        # Confidence based on tracking error and SNR
        confidence = min(1.0, state.snr / 10.0) * np.exp(-state.tracking_error_rad)
        
        correction = FringeCorrection(
            baseline_id=baseline_id,
            phase_correction_rad=phase_correction,
            correction_timestamp=current_time,
            confidence=confidence,
            prediction_horizon_sec=self.phase_prediction_horizon
        )
        
        return correction
    
    def get_tracking_status(self) -> Dict:
        """Get current fringe tracking status for all baselines."""
        
        with self.state_lock:
            status = {}
            
            for baseline_id, state in self.fringe_states.items():
                status[baseline_id] = {
                    'is_locked': state.is_locked,
                    'coherence_amplitude': state.coherence_amplitude,
                    'phase_offset_rad': state.phase_offset_rad,
                    'phase_rate_rad_per_sec': state.phase_rate_rad_per_sec,
                    'tracking_error_rad': state.tracking_error_rad,
                    'snr': state.snr,
                    'lock_duration_sec': time.time() - state.lock_timestamp if state.is_locked else 0.0
                }
        
        return status
    
    def reset_tracking(self, baseline_id: Optional[str] = None):
        """Reset tracking state for one or all baselines."""
        
        with self.state_lock:
            if baseline_id:
                if baseline_id in self.fringe_states:
                    self.fringe_states[baseline_id] = FringeState(baseline_id=baseline_id)
                    self.phase_loops[baseline_id] = PhaseTrackingLoop(baseline_id)
            else:
                # Reset all baselines
                for bid in self.baseline_ids:
                    self.fringe_states[bid] = FringeState(baseline_id=bid)
                    self.phase_loops[bid] = PhaseTrackingLoop(bid)
        
        logger.info(f"🔄 Reset fringe tracking for {baseline_id or 'all baselines'}")


class AdaptiveFringeTracker(FringeTracker):
    """Advanced fringe tracker with adaptive parameters."""
    
    def __init__(self, baseline_ids: List[str]):
        super().__init__(baseline_ids)
        
        # Adaptive parameters
        self.adaptive_bandwidth = True
        self.min_bandwidth_hz = 0.01
        self.max_bandwidth_hz = 1.0
        self.bandwidth_adaptation_rate = 0.1
        
        # Coherence-based thresholds
        self.high_coherence_threshold = 0.8
        self.low_coherence_threshold = 0.3
    
    def adapt_tracking_parameters(self, baseline_id: str, state: FringeState):
        """Adapt tracking parameters based on conditions."""
        
        if not self.adaptive_bandwidth:
            return
        
        pll = self.phase_loops[baseline_id]
        
        # Adapt loop bandwidth based on coherence and tracking error
        if state.coherence_amplitude > self.high_coherence_threshold:
            # High coherence: can use tighter tracking
            target_bandwidth = self.min_bandwidth_hz
        elif state.coherence_amplitude < self.low_coherence_threshold:
            # Low coherence: need wider bandwidth
            target_bandwidth = self.max_bandwidth_hz
        else:
            # Intermediate coherence: scale bandwidth
            coherence_factor = (state.coherence_amplitude - self.low_coherence_threshold) / \
                             (self.high_coherence_threshold - self.low_coherence_threshold)
            target_bandwidth = self.max_bandwidth_hz - coherence_factor * \
                             (self.max_bandwidth_hz - self.min_bandwidth_hz)
        
        # Smooth adaptation
        current_bandwidth = pll.bandwidth_hz
        new_bandwidth = current_bandwidth + self.bandwidth_adaptation_rate * \
                       (target_bandwidth - current_bandwidth)
        
        # Update PLL parameters
        pll.bandwidth_hz = new_bandwidth
        pll.kp = 2 * new_bandwidth
        pll.ki = new_bandwidth ** 2
        
        logger.debug(f"🎛️ Adapted {baseline_id} bandwidth: {current_bandwidth:.3f} → {new_bandwidth:.3f} Hz")


def apply_fringe_corrections(visibility_data: List, corrections: List[FringeCorrection]) -> List:
    """Apply fringe corrections to visibility measurements."""
    
    # Create correction lookup
    correction_map = {corr.baseline_id: corr for corr in corrections}
    
    corrected_visibilities = []
    
    for visibility in visibility_data:
        baseline_id = f"{visibility.baseline.camera1_id}_{visibility.baseline.camera2_id}"
        
        correction = correction_map.get(baseline_id)
        if correction and correction.confidence > 0.5:
            # Apply phase correction
            correction_factor = np.exp(1j * correction.phase_correction_rad)
            corrected_complex = visibility.complex_value * correction_factor
            
            # Update visibility
            corrected_visibility = visibility
            corrected_visibility.amplitude = abs(corrected_complex)
            corrected_visibility.phase = np.angle(corrected_complex)
            
            corrected_visibilities.append(corrected_visibility)
        else:
            # No correction or low confidence
            corrected_visibilities.append(visibility)
    
    applied_count = sum(1 for corr in corrections if corr.confidence > 0.5)
    logger.debug(f"✅ Applied {applied_count}/{len(corrections)} fringe corrections")
    
    return corrected_visibilities


def main():
    """Test fringe tracking system."""
    
    # Create test baseline IDs
    baseline_ids = ["0_1", "0_2", "1_3", "2_3", "0_3", "1_2"]
    
    # Initialize fringe tracker
    tracker = AdaptiveFringeTracker(baseline_ids)
    
    # Simulate tracking over time
    print("🎯 Testing fringe tracking system...")
    
    for i in range(50):
        # Simulate visibility data with drift and noise
        simulated_visibilities = []
        
        for baseline_id in baseline_ids:
            # Simulate atmospheric phase drift
            time_sec = i * 0.1  # 100ms updates
            atmospheric_phase = 0.5 * np.sin(0.1 * time_sec) + 0.1 * np.random.randn()
            
            # Simulate visibility with noise
            true_amplitude = 0.8
            noise_amplitude = 0.1 * np.random.randn()
            noise_phase = 0.2 * np.random.randn()
            
            complex_vis = (true_amplitude + noise_amplitude) * np.exp(1j * (atmospheric_phase + noise_phase))
            
            # Create mock visibility object
            class MockVisibility:
                def __init__(self, baseline_id, complex_val):
                    self.baseline = type('', (), {
                        'camera1_id': int(baseline_id.split('_')[0]),
                        'camera2_id': int(baseline_id.split('_')[1])
                    })()
                    self.complex_value = complex_val
                    self.amplitude = abs(complex_val)
                    self.phase = np.angle(complex_val)
                    self.snr = 10.0  # Fixed SNR for test
                    self.timestamp = time.time()
            
            simulated_visibilities.append(MockVisibility(baseline_id, complex_vis))
        
        # Track fringes
        corrections = tracker.track_fringes(simulated_visibilities)
        
        # Show status every 10 iterations
        if i % 10 == 0:
            status = tracker.get_tracking_status()
            locked_count = sum(1 for s in status.values() if s['is_locked'])
            avg_coherence = np.mean([s['coherence_amplitude'] for s in status.values()])
            
            print(f"  Iteration {i:2d}: {locked_count}/6 locked, avg coherence = {avg_coherence:.3f}")
    
    # Final status
    final_status = tracker.get_tracking_status()
    print(f"\n📊 Final tracking status:")
    for baseline_id, status in final_status.items():
        print(f"  {baseline_id}: {'🔒' if status['is_locked'] else '🔓'} "
              f"coherence={status['coherence_amplitude']:.3f}, "
              f"error={status['tracking_error_rad']:.3f} rad")
    
    print(f"\n✅ Fringe tracking test complete!")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
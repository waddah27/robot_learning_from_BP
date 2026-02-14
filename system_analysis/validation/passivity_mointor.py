import numpy as np

__all__ = ["PassivityMonitor"]


class PassivityMonitor:
    """Real-time passivity and stability monitoring"""
    def __init__(self, window_size=100):
        self.window_size = window_size
        self.energy_flow = []  # ∫vᵀF dt
        self.lyapunov_values = []
        self.passivity_violations = 0

    def update(self, K, D, velocity, force):
        """Update passivity monitor with current data"""
        # Compute instantaneous power
        instantaneous_power = np.dot(velocity, force)

        # Update energy flow (integral of power)
        if len(self.energy_flow) == 0:
            self.energy_flow.append(instantaneous_power)
        else:
            self.energy_flow.append(self.energy_flow[-1] + instantaneous_power)

        # Keep window
        if len(self.energy_flow) > self.window_size:
            self.energy_flow.pop(0)

        # Check passivity condition: energy flow should not become too negative
        if len(self.energy_flow) > 10:
            min_energy = min(self.energy_flow)
            if min_energy < -10.0:  # Threshold for passivity violation
                self.passivity_violations += 1
                return False

        return True

    def check_stability_criteria(self, K_history, D_history, errors):
        """Check stability criteria over a window"""
        criteria = {
            'lyapunov_decreasing': self.check_lyapunov_decrease(errors),
            'impedance_bounded': self.check_impedance_bounds(K_history, D_history),
            'energy_bounded': self.check_energy_bounded(),
            'passivity_maintained': self.passivity_violations == 0
        }

        return criteria

    def check_lyapunov_decrease(self, errors):
        """Check if Lyapunov-like function is decreasing"""
        if len(errors) < 2:
            return True

        # Simple check: position error should decrease on average
        errors_norm = [np.linalg.norm(e) for e in errors]
        if len(errors_norm) > 10:
            # Check last 10 errors
            recent = errors_norm[-10:]
            return np.mean(recent) < np.mean(errors_norm[-20:-10])

        return True
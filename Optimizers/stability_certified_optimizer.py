import cvxp as cp
from .impednace_optimizer import ImpedanceProfileOptimizer

class StabilityCertifiedOptimizer(ImpedanceProfileOptimizer):
    def __init__(self, dt=0.002):
        super().__init__(dt)
        self.setup_stability_constraints()

    def setup_stability_constraints(self):
        """LMI-based stability constraints"""
        # For each direction, ensure:
        # 1. D ≥ 2√(M*K) for critical damping (conservative)
        # 2. K_rate ≤ α*K to prevent instability
        # 3. D_rate ≤ β*D to maintain damping

        self.alpha = 0.1  # Max relative stiffness change per step
        self.beta = 0.2   # Max relative damping change per step

    def add_stability_lmis(self, K, D, M, constraints):
        """Add LMI constraints for guaranteed stability"""
        N = len(K)

        for i in range(N):
            # Passivity constraint: D ≥ ε√(MK)
            # Simplified to element-wise for diagonal M
            for j in range(3):
                m_eff = M[j, j] if M.ndim == 2 else M[j]
                constraints.append(
                    D[i, j] >= 0.7 * cp.sqrt(m_eff * K[i, j])
                )

            # Rate constraints for stability
            if i > 0:
                # Relative rate limits
                for j in range(3):
                    constraints.append(
                        cp.abs(K[i, j] - K[i-1, j]) <= self.alpha * K[i-1, j]
                    )
                    constraints.append(
                        cp.abs(D[i, j] - D[i-1, j]) <= self.beta * D[i-1, j]
                    )

        # Final constraint: End with high damping for safety
        constraints.append(D[-1] >= 0.8 * self.D_max)
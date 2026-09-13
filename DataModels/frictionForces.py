import numpy as np
class FrictionForce:
    def __init__(self, f_s: float, f_k: float):
        self.f_s = f_s
        self.f_k = f_k

    def get_force(self, velocity: float, required_force: float, weight: float) -> float:
        max_static = self.f_s * weight
        kinetic = self.f_k * weight

        # Static friction: cancels out external/required lateral force
        if abs(velocity) <= 0.001 and abs(required_force) <= max_static:
            return -required_force  # Equal and opposite

        # Kinetic friction: opposes direction of lateral velocity (or required force if stationary)
        if abs(velocity) > 0.001:
            return -np.sign(velocity) * kinetic
        else:
            return -np.sign(required_force) * kinetic
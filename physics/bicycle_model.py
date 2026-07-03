import numpy as np
class BicycleModel:
    def __init__(self, wheelbase):
        self.wheelbase = wheelbase

    def update(self, x, y, theta, v, delta, dt):
        """
        Update the state of the vehicle using the bicycle model equations.

        Parameters:
        - x: Current x position
        - y: Current y position
        - theta: Current orientation (in radians)
        - v: Current velocity
        - delta: Steering angle (in radians)
        - dt: Time step for the update

        Returns:
        - new_x: Updated x position
        - new_y: Updated y position
        - new_theta: Updated orientation
        """
        # Update equations based on the bicycle model
        new_x = x + v * np.cos(theta) * dt
        new_y = y + v * np.sin(theta) * dt
        new_theta = theta + (v / self.wheelbase) * np.tan(delta) * dt

        return new_x, new_y, new_theta
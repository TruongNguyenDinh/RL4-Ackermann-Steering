import math


class Differential:
    """
    Compute wheel speeds using Ackermann geometry.

    Returned speeds are linear speeds (pixels/s hoặc m/s,
    tùy theo đơn vị bạn dùng trong project).
    """

    def __init__(self, wheelbase: float, track_width: float):
        self.wheelbase = wheelbase
        self.track_width = track_width

    def compute(self, velocity: float, steering_angle: float):
        """
        Parameters
        ----------
        velocity : float
            Vehicle center velocity.

        steering_angle : float
            Bicycle steering angle (radian)

        Returns
        -------
        {
            "front_left": ...,
            "front_right": ...,
            "rear_left": ...,
            "rear_right": ...
        }
        """

        # Đi thẳng
        if abs(steering_angle) < 1e-6:
            return {
                "front_left": velocity,
                "front_right": velocity,
                "rear_left": velocity,
                "rear_right": velocity,
            }

        # Turning radius of vehicle center
        R = self.wheelbase / math.tan(abs(steering_angle))

        omega = velocity / R

        half_track = self.track_width / 2

        rear_inner = omega * (R - half_track)
        rear_outer = omega * (R + half_track)

        front_inner = omega * math.sqrt(
            self.wheelbase**2 +
            (R - half_track)**2
        )

        front_outer = omega * math.sqrt(
            self.wheelbase**2 +
            (R + half_track)**2
        )

        if steering_angle > 0:
            # Rẽ trái
            return {
                "front_left": front_inner,
                "front_right": front_outer,
                "rear_left": rear_inner,
                "rear_right": rear_outer,
            }

        else:
            # Rẽ phải
            return {
                "front_left": front_outer,
                "front_right": front_inner,
                "rear_left": rear_outer,
                "rear_right": rear_inner,
            }
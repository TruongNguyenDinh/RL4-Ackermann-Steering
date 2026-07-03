import math


class AckermannSteering:
    """
    Ackermann steering geometry.

    Input:
        steering_angle : Góc lái của mô hình Bicycle (rad)

    Output:
        left_angle  : Góc bánh trước trái (rad)
        right_angle : Góc bánh trước phải (rad)
    """

    def __init__(self, wheelbase: float, track_width: float):
        self.wheelbase = wheelbase
        self.track_width = track_width

    def compute(self, steering_angle):
        """
        Parameters
        ----------
        steering_angle : float
            Steering angle của Bicycle Model (radian)

        Returns
        -------
        left_angle : float
        right_angle : float
        """

        # Đi thẳng
        if abs(steering_angle) < 1e-6:
            return 0.0, 0.0

        # Bán kính quay của Bicycle Model
        radius = self.wheelbase / math.tan(abs(steering_angle))

        # Ackermann
        inner = math.atan(
            self.wheelbase /
            (radius - self.track_width / 2)
        )

        outer = math.atan(
            self.wheelbase /
            (radius + self.track_width / 2)
        )

        # Rẽ trái
        if steering_angle > 0:
            left = inner
            right = outer

        # Rẽ phải
        else:
            left = -outer
            right = -inner

        return left, right
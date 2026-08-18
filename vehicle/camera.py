import math
import pygame


class VehicleCamera:
    """
    Camera sensor phía trước xe.

    Camera tracking của World hoàn toàn độc lập.
    VehicleCamera chỉ lấy một ô nhỏ phía trước xe.
    """

    def __init__(
        self,
        width=480,
        height=270,
        fov=90,
        view_distance=400,
    ):
        # ==================================================
        # Output
        # ==================================================

        self.width = width
        self.height = height

        # ==================================================
        # Camera parameters
        # ==================================================

        self.fov = math.radians(fov)

        # Độ sâu vùng nhìn
        self.view_distance = view_distance

        # Camera nằm trước xe
        self.forward_offset = 0

        # ==================================================
        # Surface
        # ==================================================

        self.surface = pygame.Surface(
            (
                self.width,
                self.height
            )
        )

    # ==================================================
    # Camera position
    # ==================================================

    def get_position(self, car):

        forward = pygame.Vector2(
            math.cos(car.heading),
            math.sin(car.heading)
        )

        return (
            pygame.Vector2(
                car.x,
                car.y
            )
            + forward * self.forward_offset
        )

    # ==================================================
    # Camera direction
    # ==================================================

    def get_direction(self, car):

        return pygame.Vector2(
            math.cos(car.heading),
            math.sin(car.heading)
        )

    # ==================================================
    # Render
    # ==================================================
    def render(
        self,
        world_surface,
        car,
        render_camera,
    ):
        """
        Rotate-then-crop: xoay trước, crop sau,
        để tránh mất pixel / méo hình khi xe quay.
        """

        # --------------------------------------------------
        # Vị trí xe trên screen (KHÔNG dùng camera_pos đã offset,
        # để tâm xoay chính xác là tâm xe)
        # --------------------------------------------------

        car_screen_pos = render_camera.world_to_screen(
            pygame.Vector2(car.x, car.y)
        )

        # --------------------------------------------------
        # Kích thước vùng nhìn — GIỮ ĐÚNG TỈ LỆ output
        # để không bị bóp méo khi resize cuối cùng
        # --------------------------------------------------

        crop_height = int(self.view_distance)
        crop_width = int(
            crop_height * (self.width / self.height)
        )

        # --------------------------------------------------
        # Cạnh của vùng vuông cần crop TRƯỚC khi xoay.
        # Phải đủ lớn để chứa toàn bộ vùng nhìn ở MỌI góc xoay.
        # --------------------------------------------------

        max_extent = self.forward_offset + crop_height

        half_diag = math.hypot(
            max_extent,
            crop_width / 2
        )

        square_size = int(half_diag * 2) + 4

        square_rect = pygame.Rect(0, 0, square_size, square_size)
        square_rect.center = (
            int(car_screen_pos.x),
            int(car_screen_pos.y)
        )

        square_image = pygame.Surface((square_size, square_size))

        square_image.fill((35, 120, 35)) 

        square_image.blit(world_surface, (-square_rect.left, -square_rect.top))


        angle = math.degrees(car.heading) + 90   # option B

        rotated = pygame.transform.rotate(square_image, angle)

        rotated_center = pygame.Vector2(
            rotated.get_width() / 2,
            rotated.get_height() / 2,
        )

        # --------------------------------------------------
        # Crop vùng camera thật sự: một khoảng forward_offset
        # phía trên tâm (gần xe), kéo dài crop_height về phía xa.
        # --------------------------------------------------

        view_rect = pygame.Rect(0, 0, crop_width, crop_height)
        view_rect.centerx = int(rotated_center.x)
        view_rect.bottom = int(rotated_center.y - self.forward_offset)

        view_rect = view_rect.clip(rotated.get_rect())

        if view_rect.width < 10 or view_rect.height < 10:
            self.surface.fill((0, 0, 0))
            return self.surface

        final_image = rotated.subsurface(view_rect).copy()

        # --------------------------------------------------
        # Resize về kích thước output (không méo vì đã đúng tỉ lệ)
        # --------------------------------------------------

        final_image = pygame.transform.smoothscale(
            final_image,
            (self.width, self.height)
        )

        self.surface.blit(final_image, (0, 0))

        return self.surface
    # ==================================================
    # FOV DEBUG
    # ==================================================

    def get_fov_points(self, car):

        position = self.get_position(car)

        half_fov = self.fov / 2

        left_angle = (
            car.heading - half_fov
        )

        right_angle = (
            car.heading + half_fov
        )

        left_direction = pygame.Vector2(
            math.cos(left_angle),
            math.sin(left_angle)
        )

        right_direction = pygame.Vector2(
            math.cos(right_angle),
            math.sin(right_angle)
        )

        left_point = (
            position
            + left_direction
            * self.view_distance
        )

        right_point = (
            position
            + right_direction
            * self.view_distance
        )

        return (
            position,
            left_point,
            right_point
        )

    # ==================================================
    # Draw FOV
    # ==================================================

    def draw_fov(
        self,
        screen,
        car,
        world_to_screen,
    ):

        (
            position,
            left_point,
            right_point
        ) = self.get_fov_points(car)

        position_screen = world_to_screen(
            position
        )

        left_screen = world_to_screen(
            left_point
        )

        right_screen = world_to_screen(
            right_point
        )

        overlay = pygame.Surface(
            screen.get_size(),
            pygame.SRCALPHA
        )

        pygame.draw.polygon(
            overlay,
            (80, 180, 255, 35),
            [
                position_screen,
                left_screen,
                right_screen
            ]
        )

        screen.blit(
            overlay,
            (0, 0)
        )

        pygame.draw.line(
            screen,
            (80, 180, 255),
            position_screen,
            left_screen,
            2
        )

        pygame.draw.line(
            screen,
            (80, 180, 255),
            position_screen,
            right_screen,
            2
        )
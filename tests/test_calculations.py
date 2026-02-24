
import pytest
from PySide6.QtCore import QPoint

from fidmaa_gui.calculations import findParalellPoint, findPoint


class TestFindPoint:
    def test_direction_must_be_1_or_minus_1(self):
        with pytest.raises(ValueError):
            findPoint(240, 320, direction=0)

    def test_must_pass_angle_or_linear_coefficient(self):
        with pytest.raises(ValueError, match="Pass either angle or linear_coefficient"):
            findPoint(240, 320, direction=1)

    def test_horizontal_line_right(self):
        # angle=0 means horizontal line (tan(0) = 0)
        result = findPoint(240, 320, direction=1, angle=0, maxWidth=480, maxHeight=640)
        assert result.x() == 480
        assert result.y() == 320

    def test_horizontal_line_left(self):
        result = findPoint(240, 320, direction=-1, angle=0, maxWidth=480, maxHeight=640)
        assert result.x() == 0
        assert result.y() == 320

    def test_vertical_line_up(self):
        # angle=90 -> tan(90) is very large, so line goes nearly vertical
        # At 90 degrees the line should reach top or bottom edge
        result = findPoint(240, 320, direction=1, angle=89.99, maxWidth=480, maxHeight=640)
        # Should reach near the top (y=0) or bottom (y=640) boundary
        assert result.y() == 640 or result.y() == 0

    def test_45_degree_angle_right(self):
        # tan(45) = 1, so the line has slope 1
        result = findPoint(240, 320, direction=1, angle=45, maxWidth=480, maxHeight=640)
        # Going right from (240,320) with slope 1: y = x - 240 + 320
        # At x=480: y = 560 (within bounds)
        # At y=640: x = 560 (out of bounds)
        assert result.x() == 480
        assert result.y() == 560

    def test_45_degree_angle_left(self):
        result = findPoint(240, 320, direction=-1, angle=45, maxWidth=480, maxHeight=640)
        # Going left: at x=0, y = 320 - 240 = 80
        assert result.x() == 0
        assert result.y() == 80

    def test_negative_angle(self):
        result = findPoint(240, 320, direction=1, angle=-45, maxWidth=480, maxHeight=640)
        # tan(-45) = -1, going right: at x=480, y = -240 + 320 = 80
        assert result.x() == 480
        assert result.y() == 80

    def test_linear_coefficient_instead_of_angle(self):
        # slope = 1 is same as 45 degrees
        result = findPoint(
            240, 320, direction=1, linear_coefficient=1.0, maxWidth=480, maxHeight=640
        )
        assert result.x() == 480
        assert result.y() == 560

    def test_point_at_origin(self):
        result = findPoint(0, 0, direction=1, angle=45, maxWidth=480, maxHeight=640)
        # From (0,0) going right with slope 1: at x=480, y~=480
        assert result.x() == 480
        # int() truncation of floating point may give 479 or 480
        assert result.y() in (479, 480)

    def test_point_at_corner(self):
        result = findPoint(480, 640, direction=-1, angle=45, maxWidth=480, maxHeight=640)
        # From bottom-right going left with slope 1: at x=0, y = 640 - 480 = 160
        assert result.x() == 0
        assert result.y() == 160

    def test_steep_angle_hits_top_boundary(self):
        # Very steep angle going right should hit y=0 before x=maxWidth
        result = findPoint(240, 100, direction=1, angle=80, maxWidth=480, maxHeight=640)
        # tan(80) ~= 5.67, going right with steep slope hits y=640 quickly
        assert result.y() == 640 or result.y() == 0

    def test_returns_qpoint(self):
        result = findPoint(240, 320, direction=1, angle=0)
        assert isinstance(result, QPoint)

    def test_default_max_dimensions(self):
        # Should use MAIN_IMAGE_WIDTH/HEIGHT from const if not specified
        result = findPoint(240, 320, direction=1, angle=0)
        assert result.x() == 480  # default maxWidth

    def test_no_infinite_loop_on_extreme_angles(self):
        """Verify the direct calculation doesn't hang on extreme angles."""
        # These should all complete near-instantly
        for angle in [0, 1, 45, 89, 89.99, 90.01, 135, 179, 180, -45, -89]:
            findPoint(240, 320, direction=1, angle=angle)
            findPoint(240, 320, direction=-1, angle=angle)


class TestFindParalellPoint:
    def test_basic_perpendicular(self):
        # At angle=0, sin(0)=0 and cos(0)=1
        # So result should be (midX, midY + distance)
        x, y = findParalellPoint(100, 100, 0, 50, direction=1)
        assert abs(x - 100) < 1e-9
        assert abs(y - 150) < 1e-9

    def test_direction_minus_1(self):
        x, y = findParalellPoint(100, 100, 0, 50, direction=-1)
        assert abs(x - 100) < 1e-9
        assert abs(y - 50) < 1e-9

    def test_90_degree_angle(self):
        # At angle=90: sin(-90)=-1, cos(-90)~=0
        x, y = findParalellPoint(100, 100, 90, 50, direction=1)
        assert abs(x - 50) < 1e-9
        assert abs(y - 100) < 1e-6

    def test_distance_zero(self):
        x, y = findParalellPoint(100, 200, 45, 0)
        assert abs(x - 100) < 1e-9
        assert abs(y - 200) < 1e-9

    def test_returns_tuple(self):
        result = findParalellPoint(100, 200, 45, 10)
        assert isinstance(result, tuple)
        assert len(result) == 2

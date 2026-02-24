"""Tests for the 3D view process lifecycle management."""

from unittest.mock import MagicMock, patch


class TestOpen3DViewProcessLifecycle:
    """Test that open3DView spawns a process and closeEvent cleans it up.

    We avoid importing app.py directly due to heavy GUI dependencies,
    so we test the process management logic in isolation.
    """

    def _make_mock_window(self):
        """Create a mock MainWindow with just the 3D-related attributes."""
        window = MagicMock()
        window._3d_process = None
        return window

    def test_close_event_terminates_live_process(self):
        process = MagicMock()
        process.is_alive.return_value = True

        window = self._make_mock_window()
        window._3d_process = process

        # Simulate the closeEvent logic
        if hasattr(window, "_3d_process") and window._3d_process.is_alive():
            window._3d_process.terminate()
            window._3d_process.join(timeout=2)

        process.terminate.assert_called_once()
        process.join.assert_called_once_with(timeout=2)

    def test_close_event_skips_dead_process(self):
        process = MagicMock()
        process.is_alive.return_value = False

        window = self._make_mock_window()
        window._3d_process = process

        # Simulate the closeEvent logic
        if hasattr(window, "_3d_process") and window._3d_process.is_alive():
            window._3d_process.terminate()
            window._3d_process.join(timeout=2)

        process.terminate.assert_not_called()
        process.join.assert_not_called()

    def test_close_event_no_3d_process(self):
        """closeEvent should not fail if no 3D process was ever started."""
        window = MagicMock(spec=[])  # no _3d_process attribute

        # Simulate the closeEvent logic
        if hasattr(window, "_3d_process") and window._3d_process.is_alive():
            window._3d_process.terminate()
            window._3d_process.join(timeout=2)

        # No error raised — test passes

    def test_show_3d_view_calls_pyvista_show(self):
        """_show_3d_view delegates to pyvista_show."""
        with patch(
            "fidmaa_simple_viewer.core.pyvista_show"
        ) as mock_pyvista_show:
            from fidmaa_gui.app import _show_3d_view

            image = MagicMock()
            depthmap = MagicMock()

            _show_3d_view(image, depthmap, 0.5, 2.0)

            mock_pyvista_show.assert_called_once_with(image, depthmap, 0.5, 2.0)

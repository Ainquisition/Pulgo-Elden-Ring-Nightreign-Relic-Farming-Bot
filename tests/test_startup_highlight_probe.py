import ast
import pathlib
import unittest


def _load_probe_function():
    """Load the pure helper without importing the Tk/pynput application."""
    app_path = pathlib.Path(__file__).parents[1] / "ui" / "app.py"
    tree = ast.parse(app_path.read_text(encoding="utf-8"), filename=str(app_path))
    node = next(
        item for item in tree.body
        if isinstance(item, ast.FunctionDef)
        and item.name == "_menu_highlight_probe"
    )
    namespace = {}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(app_path), "exec"), namespace)
    return namespace["_menu_highlight_probe"]


class StartupHighlightProbeTests(unittest.TestCase):
    def setUp(self):
        self.probe = _load_probe_function()
        self.order = ["expeditions", "visual_codex", "journal", "sparring_grounds"]

    def test_middle_item_is_probed_down_one_row(self):
        self.assertEqual(
            self.probe(self.order, "visual_codex"),
            ("Key.down", "journal"),
        )

    def test_last_item_is_probed_up_one_row(self):
        self.assertEqual(
            self.probe(self.order, "sparring_grounds"),
            ("Key.up", "journal"),
        )

    def test_unknown_pixel_candidate_cannot_be_validated(self):
        self.assertIsNone(self.probe(self.order, "title_logo"))


if __name__ == "__main__":
    unittest.main()

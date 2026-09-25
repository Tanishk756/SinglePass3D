from pathlib import Path

from streamlit.testing.v1 import AppTest


def test_app_initial_screen_renders():
    app = Path(__file__).parents[1] / "streamlit_app.py"
    result = AppTest.from_file(str(app), default_timeout=10).run()
    assert not result.exception
    assert any("SinglePass3D" in item.value for item in result.markdown)
    assert result.file_uploader[0].label == "Video"
    assert result.button(key="FormSubmitter:reconstruction-Build reconstruction").disabled

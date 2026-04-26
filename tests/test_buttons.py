from lenet1_physical.control.buttons import MockButton, ButtonRig


def test_mock_buttons_invoke_callbacks():
    presses = []
    rig = ButtonRig(
        sample_button=MockButton(),
        step_button=MockButton(),
        on_sample=lambda: presses.append("sample"),
        on_step=lambda: presses.append("step"),
    )
    rig.start()
    rig.sample_button.press()
    rig.step_button.press()
    assert presses == ["sample", "step"]
    rig.stop()


def test_double_press_is_debounced_when_within_window(monkeypatch):
    times = iter([0.0, 0.005, 0.500])
    monkeypatch.setattr("time.monotonic", lambda: next(times))
    presses = []
    rig = ButtonRig(MockButton(), MockButton(), lambda: presses.append("x"), lambda: None)
    rig.start()
    rig.sample_button.press()
    rig.sample_button.press()  # debounced out
    rig.sample_button.press()
    assert presses == ["x", "x"]

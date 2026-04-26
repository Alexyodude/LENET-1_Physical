from lenet1_physical.control.state import StateMachine, State


def test_initial_state_is_idle():
    sm = StateMachine()
    assert sm.state is State.IDLE


def test_select_then_step_through_layers():
    sm = StateMachine()
    sm.on_sample()
    assert sm.state is State.ANIMATING
    assert sm.layer == "L1"
    for expected in ["L2", "L3", "L4", "L5", "L6"]:
        sm.on_step()
        assert sm.layer == expected
    sm.on_step()
    assert sm.state is State.DONE


def test_step_in_idle_is_a_no_op():
    sm = StateMachine()
    sm.on_step()
    assert sm.state is State.IDLE


def test_sample_during_animating_resets_to_l1():
    sm = StateMachine()
    sm.on_sample()
    sm.on_step()
    assert sm.layer == "L2"
    sm.on_sample()
    assert sm.layer == "L1"

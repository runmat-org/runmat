use runmat_node_agent::allocation::DrainState;

#[test]
fn drain_rejects_completion_until_every_allocation_is_idle() {
    let mut state = DrainState::Accepting;
    state.begin();
    assert_eq!(state, DrainState::Draining);
    assert!(!state.complete_if_idle(1));
    assert!(state.complete_if_idle(0));
    assert_eq!(state, DrainState::Complete);
    assert!(!state.complete_if_idle(0));
}

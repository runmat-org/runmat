use super::State;

pub(super) fn state_mut<'a>(
    state: &'a mut Option<State>,
    pc: usize,
    action: &str,
) -> Result<&'a mut State, String> {
    state
        .as_mut()
        .ok_or_else(|| format!("{action} at instruction {pc} without a plan"))
}

pub(super) fn increment_prepared(state: &mut State, pc: usize) -> Result<(), String> {
    state.prepared = state
        .prepared
        .checked_add(1)
        .ok_or_else(|| format!("prepared output target count overflows at instruction {pc}"))?;
    if state.prepared > state.expected {
        return Err(format!(
            "too many output targets are prepared at instruction {pc}: expected {}",
            state.expected
        ));
    }
    Ok(())
}

pub(super) fn prepare_target(state: &mut Option<State>, pc: usize) -> Result<(), String> {
    let state = state_mut(state, pc, "output target is prepared")?;
    if state.cardinality_loaded {
        return Err(format!(
            "output target is prepared at instruction {pc} after cardinality was loaded"
        ));
    }
    increment_prepared(state, pc)
}

pub(super) fn finish_target(state: &mut Option<State>, pc: usize) -> Result<(), String> {
    let state = state_mut(state, pc, "sequence destination finishes")?;
    if !state.destination_live || state.selector_components.is_some() {
        return Err(format!(
            "sequence destination finishes in an invalid state at instruction {pc}"
        ));
    }
    state.destination_live = false;
    increment_prepared(state, pc)
}

pub(super) fn load_cardinality(state: &mut Option<State>, pc: usize) -> Result<(), String> {
    let state = state_mut(state, pc, "output cardinality is loaded")?;
    if state.prepared != state.expected {
        return Err(format!(
            "output cardinality is loaded at instruction {pc} with {}/{} targets prepared",
            state.prepared, state.expected
        ));
    }
    if state.destination_live || state.selector_components.is_some() {
        return Err(format!(
            "output cardinality is loaded with an incomplete destination path at instruction {pc}"
        ));
    }
    if state.cardinality_loaded {
        return Err(format!(
            "output cardinality is loaded more than once at instruction {pc}"
        ));
    }
    state.cardinality_loaded = true;
    Ok(())
}

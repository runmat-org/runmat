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

pub(super) fn begin_output_assignment(
    outgoing: &mut Option<State>,
    target_count: usize,
    pc: usize,
) -> Result<(), String> {
    if outgoing.is_some() {
        return Err(format!(
            "prepared output-assignment state is overwritten at instruction {pc}"
        ));
    }
    *outgoing = Some(State {
        expected: target_count,
        prepared: 0,
        cardinality_loaded: false,
        destination_live: false,
        selector_components: None,
        fixed_owner: None,
    });
    Ok(())
}

pub(super) fn begin_fixed_assignment(
    outgoing: &mut Option<State>,
    source_statement: runmat_types::StmtId,
    root_slot: usize,
    pc: usize,
) -> Result<(), String> {
    if outgoing.is_some() {
        return Err(format!(
            "fixed destination overwrites live prepared state at instruction {pc}"
        ));
    }
    *outgoing = Some(State {
        expected: 1,
        prepared: 0,
        cardinality_loaded: false,
        destination_live: false,
        selector_components: None,
        fixed_owner: Some((source_statement, root_slot)),
    });
    Ok(())
}

pub(super) fn commit_output_assignment(
    outgoing: &mut Option<State>,
    pc: usize,
) -> Result<(), String> {
    let state = outgoing.ok_or_else(|| {
        format!("prepared outputs are committed at instruction {pc} without a plan")
    })?;
    if state.fixed_owner.is_some() || !state.cardinality_loaded || state.prepared != state.expected
    {
        return Err(format!(
            "incomplete output-assignment plan is committed at instruction {pc}"
        ));
    }
    *outgoing = None;
    Ok(())
}

pub(super) fn commit_fixed_assignment(
    outgoing: &mut Option<State>,
    source_statement: runmat_types::StmtId,
    root_slot: usize,
    pc: usize,
) -> Result<(), String> {
    let state = outgoing.ok_or_else(|| {
        format!("fixed destination consumed without preparation at instruction {pc}")
    })?;
    if state.fixed_owner != Some((source_statement, root_slot))
        || !state.cardinality_loaded
        || state.prepared != 1
        || state.expected != 1
        || state.destination_live
        || state.selector_components.is_some()
    {
        return Err(format!("fixed destination consumer differs from its completed source occurrence at instruction {pc}"));
    }
    *outgoing = None;
    Ok(())
}

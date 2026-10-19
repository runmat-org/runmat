use super::{helpers::*, Instr, State};

pub(super) fn apply(
    incoming: Option<State>,
    instruction: &Instr,
    pc: usize,
) -> Result<Option<State>, String> {
    let mut outgoing = incoming;
    match instruction {
        Instr::BeginOutputAssignment { target_count } => {
            if outgoing.is_some() {
                return Err(format!(
                    "prepared output-assignment state is overwritten at instruction {pc}"
                ));
            }
            outgoing = Some(State {
                expected: *target_count,
                prepared: 0,
                cardinality_loaded: false,
                destination_live: false,
                selector_components: None,
            });
        }
        Instr::PrepareFixedOutputTarget
        | Instr::PrepareDiscardOutputTarget
        | Instr::PrepareMemberSequenceOutputTarget { .. }
        | Instr::PrepareMemberDynamicSequenceOutputTarget { .. }
        | Instr::PrepareIndexedMemberSequenceOutputTarget { .. }
        | Instr::PrepareIndexedMemberDynamicSequenceOutputTarget { .. }
        | Instr::PrepareCellContentsSequenceOutputTarget { .. } => {
            prepare_target(&mut outgoing, pc)?
        }
        Instr::BeginSequenceOutputTarget { .. } => {
            let state = state_mut(&mut outgoing, pc, "sequence destination begins")?;
            if state.cardinality_loaded || state.destination_live {
                return Err(format!(
                    "sequence destination begins in an invalid state at instruction {pc}"
                ));
            }
            state.destination_live = true;
        }
        Instr::PrepareMemberPathStep(_) | Instr::PrepareDynamicMemberPathStep => {
            let state = outgoing.as_ref().ok_or_else(|| {
                format!("destination path step appears at instruction {pc} without a plan")
            })?;
            if !state.destination_live || state.selector_components.is_some() {
                return Err(format!(
                    "destination path step appears outside a live path at instruction {pc}"
                ));
            }
        }
        Instr::BeginPreparedIndexSelectors { component_count } => {
            let state = state_mut(&mut outgoing, pc, "index selector context begins")?;
            if !state.destination_live || state.selector_components.is_some() {
                return Err(format!(
                    "index selector context begins in an invalid state at instruction {pc}"
                ));
            }
            state.selector_components = Some(*component_count);
        }
        Instr::LoadPreparedIndexEnd { component } => {
            let state = outgoing.as_ref().ok_or_else(|| {
                format!("contextual end appears at instruction {pc} without a plan")
            })?;
            let count = state.selector_components.ok_or_else(|| {
                format!("contextual end appears at instruction {pc} without selector context")
            })?;
            if *component >= count {
                return Err(format!(
                    "contextual end component {component} exceeds selector count {count} at instruction {pc}"
                ));
            }
        }
        Instr::PrepareParenthesesPathStep {
            component_count,
            selectors,
        }
        | Instr::PrepareBracesPathStep {
            component_count,
            selectors,
            ..
        } => {
            let state = state_mut(&mut outgoing, pc, "indexed destination step appears")?;
            if selectors.len() != *component_count
                || state.selector_components.take() != Some(*component_count)
            {
                return Err(format!(
                    "indexed destination step has inconsistent selector context at instruction {pc}"
                ));
            }
        }
        Instr::FinishMemberSequenceOutputTarget(_)
        | Instr::FinishDynamicMemberSequenceOutputTarget => finish_target(&mut outgoing, pc)?,
        Instr::FinishCellContentsSequenceOutputTarget {
            component_count,
            selectors,
            expand_all,
        } => {
            let state = state_mut(&mut outgoing, pc, "sequence destination finishes")?;
            if selectors.len() != *component_count
                || (*expand_all && !selectors.is_empty())
                || !state.destination_live
                || state.selector_components.take() != Some(*component_count)
            {
                return Err(format!(
                    "cell-content destination finishes in an invalid selector state at instruction {pc}"
                ));
            }
            state.destination_live = false;
            increment_prepared(state, pc)?;
        }
        Instr::LoadPreparedOutputCardinality => load_cardinality(&mut outgoing, pc)?,
        Instr::CommitPreparedOutputTargets { .. } => {
            let state = outgoing.ok_or_else(|| {
                format!("prepared outputs are committed at instruction {pc} without a plan")
            })?;
            if !state.cardinality_loaded || state.prepared != state.expected {
                return Err(format!(
                    "incomplete output-assignment plan is committed at instruction {pc}"
                ));
            }
            outgoing = None;
        }
        _ => {}
    }
    Ok(outgoing)
}

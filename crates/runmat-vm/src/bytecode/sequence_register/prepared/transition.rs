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
                fixed_owner: None,
            });
        }
        Instr::BeginFixedAssignment {
            source_statement,
            root_slot,
        } => {
            if outgoing.is_some() {
                return Err(format!(
                    "fixed destination overwrites live prepared state at instruction {pc}"
                ));
            }
            outgoing = Some(State {
                expected: 1,
                prepared: 0,
                cardinality_loaded: false,
                destination_live: false,
                selector_components: None,
                fixed_owner: Some((*source_statement, *root_slot)),
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
        Instr::BeginSequenceOutputTarget {
            root_slot,
            root_creation,
        } => {
            let state = state_mut(&mut outgoing, pc, "sequence destination begins")?;
            if let Some((_, expected_root)) = state.fixed_owner {
                if *root_slot != expected_root || !root_creation.retains_original_receiver() {
                    return Err(format!(
                        "fixed destination root disagrees with its owner at instruction {pc}"
                    ));
                }
            } else if root_creation.retains_original_receiver() {
                return Err(format!(
                    "ordinary fixed root has no fixed owner at instruction {pc}"
                ));
            }
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
        Instr::FinishFixedAssignmentTarget {
            endpoint,
            selector_context,
            mutation,
        } => {
            if outgoing
                .as_ref()
                .and_then(|state| state.fixed_owner)
                .is_none()
            {
                return Err(format!(
                    "fixed destination finishes without its fixed owner at instruction {pc}"
                ));
            }
            use crate::bytecode::BytecodeSubscriptStep as Step;
            use runmat_types::PlaceMutationKind;
            if !matches!(
                (endpoint, mutation),
                (
                    Step::Member(_) | Step::DynamicMember,
                    PlaceMutationKind::MemberAssign
                ) | (Step::Braces { .. }, PlaceMutationKind::CellAssign)
                    | (
                        Step::Parentheses { .. },
                        PlaceMutationKind::IndexedAssign | PlaceMutationKind::Delete
                    )
            ) {
                return Err(format!(
                    "fixed target mutation disagrees with its endpoint at instruction {pc}"
                ));
            }
            let selectors = match endpoint {
                Step::Parentheses { selectors } | Step::Braces { selectors } => Some(selectors),
                Step::Member(_) | Step::DynamicMember => None,
                Step::DottedInvoke { .. } => {
                    return Err(format!(
                        "invocation is not an assignment target at instruction {pc}"
                    ))
                }
            };
            if *mutation == runmat_types::PlaceMutationKind::Delete
                && !matches!(endpoint, Step::Parentheses { .. })
            {
                return Err(format!(
                    "deletion requires a parentheses endpoint at instruction {pc}"
                ));
            }
            if let Some(selectors) = selectors {
                let count = selector_context.ok_or_else(|| {
                    format!("indexed fixed target has no selector context at instruction {pc}")
                })?;
                if selectors.len() != count
                    && !(count == 0
                        && matches!(endpoint, Step::Braces { .. })
                        && matches!(
                            selectors.as_slice(),
                            [crate::bytecode::BytecodeSubscriptSelector::Colon]
                        ))
                {
                    return Err(format!(
                        "fixed target selector context mismatches at instruction {pc}"
                    ));
                }
                if matches!(endpoint, Step::Braces { .. })
                    && selectors.iter().any(|selector| {
                        matches!(
                            selector,
                            crate::bytecode::BytecodeSubscriptSelector::Expansion(_)
                        )
                    })
                {
                    return Err(format!(
                        "brace target contains application expansion at instruction {pc}"
                    ));
                }
                let state = state_mut(&mut outgoing, pc, "fixed destination finishes")?;
                if state.selector_components.take() != Some(count) {
                    return Err(format!(
                        "fixed destination has inconsistent selector state at instruction {pc}"
                    ));
                }
            } else if selector_context.is_some() {
                return Err(format!(
                    "member target has an index context at instruction {pc}"
                ));
            }
            finish_target(&mut outgoing, pc)?;
        }
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
            if state.fixed_owner.is_some()
                || !state.cardinality_loaded
                || state.prepared != state.expected
            {
                return Err(format!(
                    "incomplete output-assignment plan is committed at instruction {pc}"
                ));
            }
            outgoing = None;
        }
        Instr::CommitFixedAssignment {
            source_statement,
            root_slot,
        } => {
            let state = outgoing.ok_or_else(|| {
                format!("fixed destination consumed without preparation at instruction {pc}")
            })?;
            if state.fixed_owner != Some((*source_statement, *root_slot))
                || !state.cardinality_loaded
                || state.prepared != 1
                || state.expected != 1
                || state.destination_live
                || state.selector_components.is_some()
            {
                return Err(format!("fixed destination consumer differs from its completed source occurrence at instruction {pc}"));
            }
            outgoing = None;
        }
        _ => {}
    }
    Ok(outgoing)
}

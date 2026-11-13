use std::collections::{BTreeMap, VecDeque};

use super::{normal_successors, Instr};

mod helpers;
mod transition;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct State {
    pub(super) expected: usize,
    pub(super) prepared: usize,
    pub(super) cardinality_loaded: bool,
    pub(super) destination_live: bool,
    pub(super) selector_components: Option<usize>,
    pub(super) fixed_owner: Option<(runmat_types::StmtId, usize)>,
}

/// Proves that a runtime-cardinality destination plan is constructed once,
/// completed before its count is observed, and consumed on every CFG path.
pub(super) fn validate(instructions: &[Instr]) -> Result<(), String> {
    if instructions.is_empty() {
        return Ok(());
    }
    let mut entries = BTreeMap::<usize, Option<State>>::new();
    let mut pending = VecDeque::from([(0usize, None)]);
    while let Some((pc, incoming)) = pending.pop_front() {
        if pc >= instructions.len() {
            if incoming.is_none() {
                continue;
            }
            return Err(
                "prepared output-assignment state remains live at the function boundary".into(),
            );
        }
        if let Some(existing) = entries.get(&pc) {
            if existing != &incoming {
                return Err(format!(
                    "prepared output-assignment state disagrees at control-flow join instruction {pc}"
                ));
            }
            continue;
        }
        entries.insert(pc, incoming);
        if let Instr::EnterTry { catch_pc, .. } = &instructions[pc] {
            // Routing an exception abandons the invocation's prepared
            // destination. Verify the handler with that same empty state,
            // including handlers not reached by normal successors.
            pending.push_back((*catch_pc, None));
        }
        let outgoing = transition::apply(incoming, &instructions[pc], pc)?;
        let successors = normal_successors(instructions, pc)?;
        if successors.is_empty()
            && !super::control_flow::is_exceptional_exit(&instructions[pc])
            && outgoing.is_some()
        {
            return Err(format!(
                "prepared output-assignment state remains live at function exit instruction {pc}"
            ));
        }
        pending.extend(successors.into_iter().map(|next| (next, outgoing)));
    }
    Ok(())
}

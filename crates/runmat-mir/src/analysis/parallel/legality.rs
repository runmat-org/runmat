use std::collections::{BTreeMap, BTreeSet, VecDeque};

use crate::{
    BasicBlockId, MirBody, MirDiagnostic, MirDiagnosticSeverity, MirLocalId, MirTerminatorKind,
};

pub(super) fn validate_control_flow(
    body: &MirBody,
    blocks: &BTreeSet<BasicBlockId>,
    header: BasicBlockId,
    exit: BasicBlockId,
    construct: &'static str,
    diagnostics: &mut Vec<MirDiagnostic>,
) -> bool {
    let mut valid = true;
    let category = if construct == "spmd" {
        "spmd-legality"
    } else {
        "parfor-legality"
    };
    for block in body
        .blocks
        .iter()
        .filter(|block| blocks.contains(&block.id))
    {
        let diagnostic = match &block.terminator.kind {
            MirTerminatorKind::Return(_) => Some((
                "RM-MIR0015",
                format!("return cannot leave a {construct} region"),
                "return would escape the independently scheduled region",
                format!("move the return outside the {construct} region"),
            )),
            MirTerminatorKind::Await { .. } => Some((
                "RM-MIR0016",
                format!("await is not permitted inside {construct}"),
                "a parallel region cannot suspend into an independently owned continuation",
                format!("await the work outside the {construct} region"),
            )),
            MirTerminatorKind::Goto(target) if *target == exit && construct == "parfor" => Some((
                "RM-MIR0017",
                format!("break cannot leave a {construct} region"),
                "break would make parallel completion depend on worker execution order",
                format!("express the condition inside the {construct} region"),
            )),
            kind if super::super::regions::successors(kind)
                .into_iter()
                .any(|target| target != header && target != exit && !blocks.contains(&target)) =>
            {
                Some((
                    "RM-MIR0018",
                    format!("{construct} control flow leaves its compiled region"),
                    "this edge does not target the region header or a block owned by the region",
                    format!("keep all control flow inside the {construct} body"),
                ))
            }
            _ => None,
        };
        if let Some((code, message, label, help)) = diagnostic {
            diagnostics.push(
                MirDiagnostic::new(
                    code,
                    MirDiagnosticSeverity::Error,
                    message,
                    block.terminator.span,
                )
                .with_primary_label(label)
                .with_help(help)
                .with_category(category),
            );
            valid = false;
        }
    }
    valid
}

/// A private or temporary value is iteration-local only if every read is
/// dominated by an assignment within that iteration. The loop binding is the
/// sole value initialized on entry. This is a forward must-analysis: joins use
/// intersection, so assignment on only one branch is not accepted.
pub(super) fn read_before_iteration_assignment(
    body: &MirBody,
    blocks: &BTreeSet<BasicBlockId>,
    first: BasicBlockId,
    binding: MirLocalId,
    local: MirLocalId,
) -> bool {
    let mut in_states = BTreeMap::from([(first, BTreeSet::from([binding]))]);
    let mut pending = VecDeque::from([first]);
    while let Some(block_id) = pending.pop_front() {
        let Some(block) = body.blocks.iter().find(|block| block.id == block_id) else {
            continue;
        };
        let mut assigned = in_states.get(&block_id).cloned().unwrap_or_default();
        for statement in &block.statements {
            let (_, writes) = super::super::regions::statement_uses_defs(statement);
            assigned.extend(writes);
        }
        let (_, terminator_writes) =
            super::super::regions::terminator_uses_defs(&block.terminator.kind);
        assigned.extend(terminator_writes);
        for successor in super::super::regions::successors(&block.terminator.kind)
            .into_iter()
            .filter(|successor| blocks.contains(successor))
        {
            let changed = match in_states.get_mut(&successor) {
                Some(existing) => {
                    let intersection = existing
                        .intersection(&assigned)
                        .copied()
                        .collect::<BTreeSet<_>>();
                    if *existing == intersection {
                        false
                    } else {
                        *existing = intersection;
                        true
                    }
                }
                None => {
                    in_states.insert(successor, assigned.clone());
                    true
                }
            };
            if changed {
                pending.push_back(successor);
            }
        }
    }

    body.blocks
        .iter()
        .filter(|block| blocks.contains(&block.id))
        .any(|block| {
            let Some(mut assigned) = in_states.get(&block.id).cloned() else {
                return false;
            };
            for statement in &block.statements {
                let (reads, writes) = super::super::regions::statement_uses_defs(statement);
                if reads.contains(&local) && !assigned.contains(&local) {
                    return true;
                }
                assigned.extend(writes);
            }
            let (reads, _) = super::super::regions::terminator_uses_defs(&block.terminator.kind);
            reads.contains(&local) && !assigned.contains(&local)
        })
}

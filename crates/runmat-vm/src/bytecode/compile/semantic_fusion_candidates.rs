use super::{merge_stmt_run_span, rvalue_has_fusion_signal};
use runmat_hir::FunctionId;
use runmat_mir::{MirAssembly, MirRvalue, MirStmtKind};

pub(super) fn derive_semantic_fusion_candidate_groups(
    mir: &MirAssembly,
    entrypoint_target: Option<FunctionId>,
) -> (usize, Vec<crate::bytecode::FusionCandidateGroup>) {
    let mut signal_count = 0usize;
    let mut groups = Vec::new();
    let mut function_ids: Vec<_> = if let Some(function) = entrypoint_target {
        vec![function]
    } else {
        mir.bodies.keys().copied().collect()
    };
    function_ids.sort_by_key(|id| id.0);
    for function_id in function_ids {
        let Some(body) = mir.bodies.get(&function_id) else {
            continue;
        };
        for block in &body.blocks {
            let mut run_start = None;
            let mut run_end = 0usize;
            let mut run_signals = 0usize;
            for (stmt_index, stmt) in block.statements.iter().enumerate() {
                let value = match &stmt.kind {
                    MirStmtKind::Assign { value, .. }
                    | MirStmtKind::MultiAssign { value, .. }
                    | MirStmtKind::SequenceAssign { value, .. }
                    | MirStmtKind::Expr(value) => value,
                    MirStmtKind::PrepareFixedAssignment(_)
                    | MirStmtKind::PreparedAssign { .. }
                    | MirStmtKind::PlaceMutation(_)
                    | MirStmtKind::CaptureSequence { .. }
                    | MirStmtKind::WorkspaceEffect { .. }
                    | MirStmtKind::StatementOutput(_)
                    | MirStmtKind::EnvironmentEffect(_) => {
                        finish_semantic_candidate_group(
                            &mut groups,
                            body,
                            block,
                            run_start,
                            run_end,
                            run_signals,
                        );
                        run_start = None;
                        run_signals = 0;
                        continue;
                    }
                };
                if rvalue_has_fusion_signal(value) {
                    signal_count += 1;
                    run_signals += 1;
                    run_start.get_or_insert(stmt_index);
                    run_end = stmt_index + 1;
                } else if run_start.is_some() && empty_private_tensor_control(stmt, body) {
                    // The literal control is evaluated in order but is not a
                    // signal. A following signal must close the candidate.
                } else {
                    finish_semantic_candidate_group(
                        &mut groups,
                        body,
                        block,
                        run_start,
                        run_end,
                        run_signals,
                    );
                    run_start = None;
                    run_signals = 0;
                }
            }
            finish_semantic_candidate_group(
                &mut groups,
                body,
                block,
                run_start,
                run_end,
                run_signals,
            );
        }
    }
    (signal_count, groups)
}

fn finish_semantic_candidate_group(
    groups: &mut Vec<crate::bytecode::FusionCandidateGroup>,
    body: &runmat_mir::MirBody,
    block: &runmat_mir::BasicBlock,
    start: Option<usize>,
    end: usize,
    signals: usize,
) {
    let Some(stmt_start) = start.filter(|_| signals >= 2) else {
        return;
    };
    groups.push(crate::bytecode::FusionCandidateGroup {
        id: groups.len(),
        signal_count: signals,
        function: body.function,
        block: block.id,
        stmt_start,
        stmt_end: end,
        source_span: merge_stmt_run_span(block, stmt_start, end),
    });
}

fn empty_private_tensor_control(stmt: &runmat_mir::MirStmt, body: &runmat_mir::MirBody) -> bool {
    let MirStmtKind::Assign {
        place: runmat_mir::MirPlace::Local(local),
        value:
            MirRvalue::Aggregate {
                kind: runmat_mir::MirAggregateKind::Tensor,
                row_lengths,
                elements,
            },
    } = &stmt.kind
    else {
        return false;
    };
    elements.is_empty()
        && row_lengths.iter().all(|length| *length == 0)
        && body.locals.get(local.0).is_some_and(|entry| {
            entry.id == *local
                && entry.binding.is_none()
                && matches!(&entry.kind, runmat_mir::MirLocalKind::Temporary)
        })
}

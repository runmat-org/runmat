use crate::{
    BasicBlockId, MirBody, MirDiagnostic, MirDiagnosticSeverity, MirOperand, MirPlace, MirRvalue,
    MirStmtKind, MirTerminatorKind, SpawnBoundary,
};
use runmat_hir::{FunctionId, Span, SpawnSafetyFact, SpawnSafetyReason};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeSet, HashMap};

use super::FunctionSummary;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SpawnSafetySummary {
    pub function: FunctionId,
    pub safety: SpawnSafetyFact,
}

pub fn analyze_spawn_boundaries(body: &MirBody) -> Vec<SpawnBoundary> {
    let mut boundaries = Vec::new();
    for block in &body.blocks {
        for stmt in &block.statements {
            match &stmt.kind {
                MirStmtKind::Assign { value, .. }
                | MirStmtKind::MultiAssign { value, .. }
                | MirStmtKind::Expr(value) => {
                    collect_spawn_rvalue(value, stmt.span, &mut boundaries)
                }
                MirStmtKind::PlaceMutation(_)
                | MirStmtKind::WorkspaceEffect { .. }
                | MirStmtKind::EnvironmentEffect(_) => {}
            }
        }
    }
    boundaries
}

pub fn analyze_spawn_boundaries_with_summaries(
    body: &MirBody,
    summaries: &HashMap<FunctionId, FunctionSummary>,
) -> Vec<SpawnBoundary> {
    let in_states = future_target_in_states(body);
    let block_index: HashMap<_, _> = body
        .blocks
        .iter()
        .enumerate()
        .map(|(idx, block)| (block.id, idx))
        .collect();
    let mut boundaries = Vec::new();
    for block in &body.blocks {
        let mut state = block_index
            .get(&block.id)
            .and_then(|idx| in_states.get(*idx))
            .cloned()
            .unwrap_or_else(|| vec![BTreeSet::new(); body.locals.len()]);
        for stmt in &block.statements {
            match &stmt.kind {
                MirStmtKind::Assign { place, value } => {
                    collect_classified_spawn_rvalue(
                        value,
                        stmt.span,
                        summaries,
                        &state,
                        &mut boundaries,
                    );
                    if let MirPlace::Local(local) = place {
                        state[local.0].clear();
                        if let Some(target) = future_target(value) {
                            state[local.0].insert(target);
                        }
                    }
                }
                MirStmtKind::MultiAssign { value, .. } | MirStmtKind::Expr(value) => {
                    collect_classified_spawn_rvalue(
                        value,
                        stmt.span,
                        summaries,
                        &state,
                        &mut boundaries,
                    );
                }
                MirStmtKind::PlaceMutation(_)
                | MirStmtKind::WorkspaceEffect { .. }
                | MirStmtKind::EnvironmentEffect(_) => {}
            }
        }
    }
    boundaries
}

fn future_target_in_states(body: &MirBody) -> Vec<Vec<BTreeSet<FunctionId>>> {
    let local_count = body.locals.len();
    let block_index: HashMap<BasicBlockId, usize> = body
        .blocks
        .iter()
        .enumerate()
        .map(|(idx, block)| (block.id, idx))
        .collect();
    let mut in_states = vec![vec![BTreeSet::new(); local_count]; body.blocks.len()];
    let mut out_states = in_states.clone();
    let mut changed = true;
    while changed {
        changed = false;
        for (idx, block) in body.blocks.iter().enumerate() {
            let mut state = in_states[idx].clone();
            transfer_future_targets(block, &mut state);
            if state != out_states[idx] {
                out_states[idx] = state.clone();
                changed = true;
            }
            for successor in successors(&block.terminator.kind) {
                let Some(&successor_idx) = block_index.get(&successor) else {
                    continue;
                };
                for (target, incoming) in in_states[successor_idx].iter_mut().zip(&state) {
                    let before = target.len();
                    target.extend(incoming.iter().copied());
                    if target.len() != before {
                        changed = true;
                    }
                }
            }
        }
    }
    in_states
}

fn transfer_future_targets(block: &crate::BasicBlock, state: &mut [BTreeSet<FunctionId>]) {
    for stmt in &block.statements {
        let MirStmtKind::Assign { place, value } = &stmt.kind else {
            continue;
        };
        if let MirPlace::Local(local) = place {
            state[local.0].clear();
            if let Some(target) = future_target(value) {
                state[local.0].insert(target);
            }
        }
    }
}

pub fn diagnose_spawn_safety(boundaries: &[SpawnBoundary]) -> Vec<MirDiagnostic> {
    boundaries
        .iter()
        .filter_map(|boundary| match &boundary.safety {
            SpawnSafetyFact::NotSpawnSafe { reason } => {
                Some(spawn_safety_diagnostic(reason.clone(), boundary.span))
            }
            SpawnSafetyFact::SpawnSafe | SpawnSafetyFact::RequiresIsolation => None,
        })
        .collect()
}

pub fn summarize_spawn_safety(body: &MirBody) -> SpawnSafetySummary {
    let safety = if analyze_spawn_boundaries(body).is_empty() {
        SpawnSafetyFact::SpawnSafe
    } else {
        SpawnSafetyFact::RequiresIsolation
    };
    SpawnSafetySummary {
        function: body.function,
        safety,
    }
}

fn collect_spawn_rvalue(value: &MirRvalue, span: Span, boundaries: &mut Vec<SpawnBoundary>) {
    match value {
        MirRvalue::Spawn(future) => boundaries.push(SpawnBoundary {
            future: future.clone(),
            safety: SpawnSafetyFact::RequiresIsolation,
            span,
        }),
        MirRvalue::Use(_)
        | MirRvalue::Unary(_, _)
        | MirRvalue::Binary(_, _, _)
        | MirRvalue::Range { .. }
        | MirRvalue::Call(_)
        | MirRvalue::Aggregate { .. }
        | MirRvalue::Index { .. }
        | MirRvalue::Member { .. }
        | MirRvalue::DynamicMember { .. }
        | MirRvalue::MetaClass(_)
        | MirRvalue::Colon
        | MirRvalue::End
        | MirRvalue::Future { .. } => {}
    }
}

fn collect_classified_spawn_rvalue(
    value: &MirRvalue,
    span: Span,
    summaries: &HashMap<FunctionId, FunctionSummary>,
    future_targets: &[BTreeSet<FunctionId>],
    boundaries: &mut Vec<SpawnBoundary>,
) {
    let MirRvalue::Spawn(future) = value else {
        return;
    };
    let mut boundary = SpawnBoundary {
        future: future.clone(),
        safety: SpawnSafetyFact::RequiresIsolation,
        span,
    };
    boundary.safety = classify_spawn_boundary(&boundary, summaries, future_targets);
    boundaries.push(boundary);
}

fn future_target(value: &MirRvalue) -> Option<FunctionId> {
    match value {
        MirRvalue::Future { function, .. } => Some(*function),
        _ => None,
    }
}

fn classify_spawn_boundary(
    boundary: &SpawnBoundary,
    summaries: &HashMap<FunctionId, FunctionSummary>,
    future_targets: &[BTreeSet<FunctionId>],
) -> SpawnSafetyFact {
    let Some(targets) = spawn_targets(&boundary.future, future_targets) else {
        return SpawnSafetyFact::NotSpawnSafe {
            reason: SpawnSafetyReason::UnknownDynamicCapture,
        };
    };
    let mut requires_isolation = false;
    for target in targets {
        let Some(summary) = summaries.get(&target) else {
            requires_isolation = true;
            continue;
        };
        if !summary.reads_captures.is_empty() || !summary.writes_captures.is_empty() {
            return SpawnSafetyFact::NotSpawnSafe {
                reason: SpawnSafetyReason::MutableLexicalCapture,
            };
        }
    }
    if requires_isolation {
        SpawnSafetyFact::RequiresIsolation
    } else {
        SpawnSafetyFact::SpawnSafe
    }
}

fn spawn_targets(
    future: &MirOperand,
    future_targets: &[BTreeSet<FunctionId>],
) -> Option<BTreeSet<FunctionId>> {
    match future {
        MirOperand::Local(local) => future_targets
            .get(local.0)
            .filter(|targets| !targets.is_empty())
            .cloned(),
        MirOperand::FunctionHandle(_) | MirOperand::Temp(_) | MirOperand::Constant(_) => None,
    }
}

fn successors(kind: &MirTerminatorKind) -> Vec<BasicBlockId> {
    match kind {
        MirTerminatorKind::Goto(target) => vec![*target],
        MirTerminatorKind::Branch {
            then_block,
            else_block,
            ..
        } => vec![*then_block, *else_block],
        MirTerminatorKind::Switch {
            cases, otherwise, ..
        } => cases
            .iter()
            .map(|(_, block)| *block)
            .chain(std::iter::once(*otherwise))
            .collect(),
        MirTerminatorKind::For {
            body_block,
            exit_block,
            ..
        } => vec![*body_block, *exit_block],
        MirTerminatorKind::TryCatch {
            try_block,
            catch_block,
        } => vec![*try_block, *catch_block],
        MirTerminatorKind::Await {
            resume, cleanup, ..
        } => cleanup.map_or_else(|| vec![*resume], |cleanup| vec![*resume, cleanup]),
        MirTerminatorKind::Return(_) | MirTerminatorKind::Unreachable => Vec::new(),
    }
}

fn spawn_safety_diagnostic(reason: SpawnSafetyReason, span: Span) -> MirDiagnostic {
    let message = match reason {
        SpawnSafetyReason::MutableLexicalCapture => {
            "spawned future mutates a lexical capture from its parent frame"
        }
        SpawnSafetyReason::NonSendableRuntimeHandle => {
            "spawned future captures a non-sendable runtime handle"
        }
        SpawnSafetyReason::UnsynchronizedSharedMutation => {
            "spawned future performs unsynchronized shared mutation"
        }
        SpawnSafetyReason::UnknownDynamicCapture => {
            "spawned future has dynamically captured state with unknown spawn safety"
        }
    };
    MirDiagnostic::new("RM-MIR0003", MirDiagnosticSeverity::Error, message, span)
        .with_primary_label("this spawn crosses an unsafe task boundary")
        .with_help("avoid mutating parent-frame lexical captures from spawned futures")
        .with_category("spawn-safety")
}

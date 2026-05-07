use crate::{
    MirBody, MirDiagnostic, MirDiagnosticSeverity, MirOperand, MirPlace, MirRvalue, MirStmtKind,
    SpawnBoundary,
};
use runmat_hir::{FunctionId, Span, SpawnSafetyFact, SpawnSafetyReason};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

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
    let future_targets = collect_future_targets(body);
    analyze_spawn_boundaries(body)
        .into_iter()
        .map(|mut boundary| {
            boundary.safety = classify_spawn_boundary(&boundary, summaries, &future_targets);
            boundary
        })
        .collect()
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

fn collect_future_targets(body: &MirBody) -> HashMap<crate::MirLocalId, FunctionId> {
    let mut targets = HashMap::new();
    for block in &body.blocks {
        for stmt in &block.statements {
            let MirStmtKind::Assign { place, value } = &stmt.kind else {
                continue;
            };
            let MirPlace::Local(local) = place else {
                continue;
            };
            if let Some(target) = future_target(value) {
                targets.insert(*local, target);
            }
        }
    }
    targets
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
    future_targets: &HashMap<crate::MirLocalId, FunctionId>,
) -> SpawnSafetyFact {
    let Some(target) = spawn_target(&boundary.future, future_targets) else {
        return SpawnSafetyFact::NotSpawnSafe {
            reason: SpawnSafetyReason::UnknownDynamicCapture,
        };
    };
    let Some(summary) = summaries.get(&target) else {
        return SpawnSafetyFact::RequiresIsolation;
    };
    if !summary.reads_captures.is_empty() || !summary.writes_captures.is_empty() {
        return SpawnSafetyFact::NotSpawnSafe {
            reason: SpawnSafetyReason::MutableLexicalCapture,
        };
    }
    SpawnSafetyFact::SpawnSafe
}

fn spawn_target(
    future: &MirOperand,
    future_targets: &HashMap<crate::MirLocalId, FunctionId>,
) -> Option<FunctionId> {
    match future {
        MirOperand::Local(local) => future_targets.get(local).copied(),
        MirOperand::FunctionHandle(_) | MirOperand::Temp(_) | MirOperand::Constant(_) => None,
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

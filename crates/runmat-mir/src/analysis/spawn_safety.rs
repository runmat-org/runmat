use crate::{MirBody, MirRvalue, MirStmtKind, SpawnBoundary};
use runmat_hir::{FunctionId, Span, SpawnSafetyFact};
use serde::{Deserialize, Serialize};

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
        | MirRvalue::Future(_) => {}
    }
}

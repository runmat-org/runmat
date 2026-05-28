use crate::{MirLocalId, MirPlace, MirRvalue};
use runmat_hir::{
    AssignmentCreationPolicy, AssignmentShapePolicy, EnvironmentEffect, PlaceMutationKind,
    RequestedOutputCount, Span, WorkspaceEffect,
};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MirStmt {
    pub kind: MirStmtKind,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum MirStmtKind {
    Assign {
        place: MirPlace,
        value: MirRvalue,
    },
    MultiAssign {
        targets: MirOutputTargetList,
        value: MirRvalue,
    },
    Expr(MirRvalue),
    PlaceMutation(MirPlaceMutation),
    WorkspaceEffect {
        effect: WorkspaceEffect,
        bindings: Vec<MirLocalId>,
    },
    EnvironmentEffect(EnvironmentEffect),
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MirPlaceMutation {
    pub place: MirPlace,
    pub kind: PlaceMutationKind,
    pub creation_policy: AssignmentCreationPolicy,
    pub shape_policy: AssignmentShapePolicy,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MirOutputTargetList {
    pub targets: Vec<MirOutputTarget>,
    pub requested_outputs: RequestedOutputCount,
}

impl MirOutputTargetList {
    pub fn validate_fixed_arity(&self, context: &str) -> Result<usize, String> {
        let expected = self.targets.len();
        let count = self.requested_outputs.fixed_count();
        if count != expected {
            return Err(format!(
                "{context} output target count mismatch: requested {count}, targets {expected}"
            ));
        }
        Ok(count)
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum MirOutputTarget {
    Place(MirPlace),
    Discard,
}

use crate::{MirLocalId, MirPlace, MirRvalue};
use runmat_hir::{
    AssignmentCreationPolicy, AssignmentShapePolicy, EnvironmentEffect, PlaceMutationKind, Span,
    WorkspaceEffect,
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
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum MirOutputTarget {
    Place(MirPlace),
    Discard,
}

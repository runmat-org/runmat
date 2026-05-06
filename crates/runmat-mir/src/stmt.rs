use crate::{MirPlace, MirRvalue};
use runmat_hir::{PlaceMutation, Span};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MirStmt {
    pub kind: MirStmtKind,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum MirStmtKind {
    Assign { place: MirPlace, value: MirRvalue },
    Expr(MirRvalue),
    PlaceMutation(PlaceMutation),
}

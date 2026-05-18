use crate::MirOperand;
use runmat_hir::{IndexKind, IndexResultContext};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MirIndexing {
    pub kind: IndexKind,
    pub components: Vec<MirIndexComponent>,
    pub result_context: IndexResultContext,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum MirIndexComponent {
    Colon,
    End { dim: Option<usize>, offset: isize },
    Expr(MirOperand),
}

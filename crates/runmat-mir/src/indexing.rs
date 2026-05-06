use runmat_hir::{IndexComponent, IndexKind, IndexResultContext};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MirIndexing {
    pub kind: IndexKind,
    pub components: Vec<IndexComponent>,
    pub result_context: IndexResultContext,
}

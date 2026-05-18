use crate::MirOperand;
use runmat_hir::{Span, SpawnSafetyFact};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub enum AsyncBehaviorFact {
    #[default]
    NeverSuspends,
    MaySuspend,
    RequiresAsyncRuntime,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SpawnBoundary {
    pub future: MirOperand,
    pub safety: SpawnSafetyFact,
    pub span: Span,
}

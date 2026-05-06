use crate::{BasicBlockId, MirOperand};
use runmat_hir::{AsyncValueFact, BindingId, Span, SpawnSafetyFact};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum AsyncBehaviorFact {
    NeverSuspends,
    MaySuspend,
    RequiresAsyncRuntime,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AwaitPoint {
    pub future: MirOperand,
    pub resume: BasicBlockId,
    pub live_bindings: Vec<BindingId>,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SpawnBoundary {
    pub future: MirOperand,
    pub safety: SpawnSafetyFact,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AsyncFact {
    pub behavior: AsyncBehaviorFact,
    pub value: Option<AsyncValueFact>,
}

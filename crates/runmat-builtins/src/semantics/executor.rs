//! Execution operations declared by canonical builtin contracts. Ordinary
//! runtime calls do not acquire executor semantics from their names.
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum BuiltinExecutorIntrinsic {
    Parallel(ParallelExecutorIntrinsic),
}

impl BuiltinExecutorIntrinsic {
    pub const fn requires_executor_lowering(self) -> bool {
        match self {
            Self::Parallel(operation) => operation.requires_executor_lowering(),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ParallelExecutorIntrinsic {
    EnsurePool,
    CurrentPool,
    ScheduleFeval,
    ScheduleFevalOnAll,
    FetchOutputs,
    FetchNext,
}

impl ParallelExecutorIntrinsic {
    /// Pool operations have ordinary runtime implementations. Future
    /// submission/waits require the executor to lower the operation.
    pub const fn requires_executor_lowering(self) -> bool {
        match self {
            Self::EnsurePool | Self::CurrentPool => false,
            Self::ScheduleFeval
            | Self::ScheduleFevalOnAll
            | Self::FetchOutputs
            | Self::FetchNext => true,
        }
    }
}

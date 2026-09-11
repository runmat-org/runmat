use super::ValueSequenceFact;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ExecutionFact {
    Future {
        output: ValueSequenceFact,
        state: FutureStateFact,
    },
    Task {
        output: ValueSequenceFact,
        spawn_safety: SpawnSafetyFact,
    },
    Pool,
    Job {
        output: ValueSequenceFact,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum FutureStateFact {
    Lazy,
    Awaited,
    Unknown,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum SpawnSafetyFact {
    SpawnSafe,
    RequiresIsolation,
    NotSpawnSafe { reason: SpawnSafetyReason },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SpawnSafetyReason {
    MutableLexicalCapture,
    NonSendableRuntimeHandle,
    UnsynchronizedSharedMutation,
    UnknownDynamicCapture,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ExceptionFact {
    pub identifier: Option<String>,
}

use serde::{Deserialize, Serialize};

/// Deterministic function ordinal within one immutable program revision.
/// Product schemas always pair this identity with that revision.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct ProgramFunctionId(pub u32);

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct ProgramPointId {
    pub function: ProgramFunctionId,
    pub block: u32,
    /// Zero is block entry; `n + 1` is immediately after statement `n`.
    pub position: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct RegionId {
    pub function: ProgramFunctionId,
    pub ordinal: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct RegionValueId {
    pub function: ProgramFunctionId,
    pub local: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct RegionGuardId {
    pub region: RegionId,
    pub ordinal: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct DeoptimizationPointId {
    pub function: ProgramFunctionId,
    pub ordinal: u32,
}

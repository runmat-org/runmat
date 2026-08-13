use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ForeignFact {
    pub family: String,
    pub type_name: Option<String>,
    pub type_version: Option<u32>,
    pub ownership: ForeignOwnershipFact,
    pub affinity: ForeignAffinityFact,
    pub lifetime: ForeignLifetimeFact,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ForeignOwnershipFact {
    Unknown,
    Borrowed,
    Owned,
    Shared,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ForeignAffinityFact {
    Unknown,
    AnyThread,
    OriginThread,
    OriginProcess,
    RemoteHost,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ForeignLifetimeFact {
    Unknown,
    Call,
    Session,
    Persistent,
    External,
}

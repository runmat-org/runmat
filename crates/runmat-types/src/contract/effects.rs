use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum EffectKind {
    WorkspaceRead,
    WorkspaceWrite,
    EnvironmentRead,
    EnvironmentWrite,
    FilesystemRead,
    FilesystemWrite,
    Network,
    UserInterface,
    Randomness,
    Clock,
    HostCallback,
    MaySuspend,
    MayThrow,
    Unknown,
}

#[derive(Debug, Clone, PartialEq, Eq, Default, Serialize, Deserialize)]
pub struct EffectSet(pub BTreeSet<EffectKind>);

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum CapabilityRequirement {
    HostRuntime,
    Filesystem,
    Network,
    UserInterface,
    Accelerator,
    NativeCode,
    ForeignRuntime,
    ParallelRuntime,
    DistributedRuntime,
}

#[derive(Debug, Clone, PartialEq, Eq, Default, Serialize, Deserialize)]
pub struct CapabilitySet(pub BTreeSet<CapabilityRequirement>);

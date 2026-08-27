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

/// Stack contract for code that crosses a host execution boundary.
///
/// `Process` is required by in-process runtimes that attach to, inspect, or
/// unwind through the operating-system thread stack. Portable RunMat code has
/// no such restriction and may use a VM-managed segmented stack.
#[derive(
    Debug, Clone, Copy, Default, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize,
)]
pub enum ExecutionStackRequirement {
    #[default]
    Any,
    Process,
}

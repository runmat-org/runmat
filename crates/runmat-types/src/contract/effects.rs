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

impl EffectKind {
    /// A deterministic local evaluation may reject its inputs without reading or
    /// changing external state. Unknown effects and suspension are not admitted.
    pub const fn is_side_effect_free(self) -> bool {
        matches!(self, Self::MayThrow)
    }
}

impl EffectSet {
    pub fn is_side_effect_free(&self) -> bool {
        self.0.iter().all(|effect| effect.is_side_effect_free())
    }

    /// A deterministic function of its inputs: it may reject them or suspend while awaiting
    /// its own computation (for example an accelerator readback), but it neither observes nor
    /// changes external state. Analysis generators (signals, spectra, geometry) require this.
    pub fn is_deterministic_value_function(&self) -> bool {
        self.0
            .iter()
            .all(|effect| matches!(effect, EffectKind::MayThrow | EffectKind::MaySuspend))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn deterministic_value_functions_may_suspend_but_not_observe_external_state() {
        assert!(EffectSet(BTreeSet::from([
            EffectKind::MaySuspend,
            EffectKind::MayThrow
        ]))
        .is_deterministic_value_function());
        for forbidden in [
            EffectKind::WorkspaceRead,
            EffectKind::FilesystemRead,
            EffectKind::Randomness,
            EffectKind::Clock,
            EffectKind::HostCallback,
            EffectKind::Unknown,
        ] {
            assert!(
                !EffectSet(BTreeSet::from([EffectKind::MaySuspend, forbidden]))
                    .is_deterministic_value_function()
            );
        }
    }

    #[test]
    fn pure_evaluation_may_throw_but_cannot_observe_or_change_external_state() {
        assert!(EffectSet::default().is_side_effect_free());
        assert!(EffectSet(BTreeSet::from([EffectKind::MayThrow])).is_side_effect_free());
        for forbidden in [
            EffectKind::WorkspaceRead,
            EffectKind::WorkspaceWrite,
            EffectKind::EnvironmentRead,
            EffectKind::EnvironmentWrite,
            EffectKind::FilesystemRead,
            EffectKind::FilesystemWrite,
            EffectKind::Network,
            EffectKind::UserInterface,
            EffectKind::Randomness,
            EffectKind::Clock,
            EffectKind::HostCallback,
            EffectKind::MaySuspend,
            EffectKind::Unknown,
        ] {
            assert!(!EffectSet(BTreeSet::from([forbidden])).is_side_effect_free());
            assert!(
                !EffectSet(BTreeSet::from([EffectKind::MayThrow, forbidden])).is_side_effect_free()
            );
        }
    }
}

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

use super::*;
use runmat_value::ExactHostAllocationClaim;

pub(super) type NamedMap =
    HashMap<WorkspaceName, WorkspaceValueLease, RandomState, OperationCollectionAllocator>;
pub(super) type SlotMap =
    HashMap<usize, WorkspaceValueLease, RandomState, OperationCollectionAllocator>;

#[derive(Debug)]
pub(super) struct GlobalGeneration {
    pub(super) named: NamedMap,
    pub(super) slots: SlotMap,
    pub(super) gate: OperationPermitGate,
    // Both maps retire their exact bucket claims before this owner refunds.
    pub(super) _claim: Option<ExactHostAllocationClaim>,
}

#[derive(Clone, Debug, Default)]
pub(in crate::workspace::session) struct SessionGlobals {
    pub(super) generation: Option<ExactShared<GlobalGeneration>>,
}

/// Stable borrowed view of one session-global generation. Later writes detach
/// the canonical COW table; this owner retains the original name/value/root
/// claims without cloning a language Value or resolving ambient session state.
#[derive(Clone, Debug)]
pub struct SessionGlobalGeneration {
    generation: Option<ExactShared<GlobalGeneration>>,
    revision: super::super::GlobalRevision,
}

impl SessionGlobalGeneration {
    pub fn revision(&self) -> super::super::GlobalRevision {
        self.revision
    }
    pub fn named(&self, name: &str) -> Option<&runmat_value::Value> {
        self.generation
            .as_ref()?
            .get()
            .named
            .get(name)
            .map(WorkspaceValueLease::value)
    }

    pub fn slot(&self, slot: usize) -> Option<&runmat_value::Value> {
        self.generation
            .as_ref()?
            .get()
            .slots
            .get(&slot)
            .map(WorkspaceValueLease::value)
    }

    pub(crate) fn named_lease(
        &self,
        name: &str,
    ) -> Option<crate::workspace::WorkspaceValueReadLease> {
        self.generation
            .as_ref()?
            .get()
            .named
            .get(name)
            .cloned()
            .map(crate::workspace::retained::WorkspaceValueReadLease::new)
    }

    pub(crate) fn slot_lease(
        &self,
        slot: usize,
    ) -> Option<crate::workspace::WorkspaceValueReadLease> {
        self.generation
            .as_ref()?
            .get()
            .slots
            .get(&slot)
            .cloned()
            .map(crate::workspace::retained::WorkspaceValueReadLease::new)
    }

    pub fn named_values(&self) -> impl Iterator<Item = (&str, &runmat_value::Value)> {
        self.generation.as_ref().into_iter().flat_map(|generation| {
            generation
                .get()
                .named
                .iter()
                .map(|(name, value)| (name.as_str(), value.value()))
        })
    }
}

impl SessionGlobals {
    pub(in crate::workspace::session) fn capture_generation(
        &self,
        revision: super::super::GlobalRevision,
    ) -> SessionGlobalGeneration {
        SessionGlobalGeneration {
            generation: self.generation.clone(),
            revision,
        }
    }
}

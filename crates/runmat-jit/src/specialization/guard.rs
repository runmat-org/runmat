use std::collections::{BTreeMap, BTreeSet};

use runmat_types::CapabilityRequirement;
#[cfg(not(target_arch = "wasm32"))]
use runmat_types::{FactSatisfaction, RegionGuardCondition, RegionGuardContract};
#[cfg(not(target_arch = "wasm32"))]
use runmat_value::Value;

/// Invocation-owned authorities used to evaluate portable region guards.
#[derive(Clone, Debug, Default)]
pub struct GuardEnvironment {
    runtime_revisions: BTreeMap<String, String>,
    capabilities: BTreeSet<CapabilityRequirement>,
}

impl GuardEnvironment {
    pub fn with_runtime_revision(
        mut self,
        identity: impl Into<String>,
        revision: impl Into<String>,
    ) -> Self {
        self.runtime_revisions
            .insert(identity.into(), revision.into());
        self
    }

    pub fn with_capability(mut self, capability: CapabilityRequirement) -> Self {
        self.capabilities.insert(capability);
        self
    }

    #[cfg(not(target_arch = "wasm32"))]
    pub(crate) fn evaluate(
        &self,
        guard: &RegionGuardContract,
        value: Option<&Value>,
    ) -> Result<(), GuardFailure> {
        let actual = value.map(runmat_runtime::value_fact::value_fact);
        let satisfied = match &guard.condition {
            RegionGuardCondition::ValueFact { expected, .. } => actual
                .as_ref()
                .is_some_and(|actual| actual.satisfies(expected)),
            RegionGuardCondition::Shape { expected, .. } => actual
                .as_ref()
                .is_some_and(|actual| actual.shape.satisfies(expected)),
            RegionGuardCondition::Class { expected, .. } => actual
                .as_ref()
                .is_some_and(|actual| actual.kind.satisfies(expected)),
            RegionGuardCondition::Residency { expected, .. } => actual
                .as_ref()
                .is_some_and(|actual| actual.residency.satisfies(expected)),
            RegionGuardCondition::Alias { expected, .. } => actual.as_ref().is_some_and(|actual| {
                *expected == runmat_types::AliasFact::Unknown || actual.alias == *expected
            }),
            RegionGuardCondition::RuntimeState { identity, revision } => self
                .runtime_revisions
                .get(identity)
                .is_some_and(|actual| actual == revision),
            RegionGuardCondition::Capability { requirement } => {
                self.capabilities.contains(requirement)
            }
        };
        if satisfied {
            Ok(())
        } else {
            Err(GuardFailure {
                guard: guard.id,
                deopt: guard.deopt,
                kind: GuardFailureKind::for_condition(&guard.condition),
            })
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum GuardFailureKind {
    Representation,
    RuntimeState,
    Capability,
}

impl GuardFailureKind {
    #[cfg(not(target_arch = "wasm32"))]
    pub(crate) fn for_condition(condition: &RegionGuardCondition) -> Self {
        match condition {
            RegionGuardCondition::RuntimeState { .. } => Self::RuntimeState,
            RegionGuardCondition::Capability { .. } => Self::Capability,
            _ => Self::Representation,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct GuardFailure {
    pub guard: runmat_types::RegionGuardId,
    pub deopt: runmat_types::DeoptimizationPointId,
    pub kind: GuardFailureKind,
}

#[cfg(test)]
mod tests {
    use runmat_types::{
        CapabilityRequirement, DeoptimizationPointId, ProgramFunctionId, RegionGuardCondition,
        RegionGuardContract, RegionGuardId, RegionId,
    };

    use super::{GuardEnvironment, GuardFailureKind};

    fn guard(condition: RegionGuardCondition) -> RegionGuardContract {
        let function = ProgramFunctionId(1);
        let region = RegionId {
            function,
            ordinal: 2,
        };
        RegionGuardContract {
            id: RegionGuardId { region, ordinal: 3 },
            condition,
            deopt: DeoptimizationPointId {
                function,
                ordinal: 4,
            },
        }
    }

    #[test]
    fn runtime_and_capability_guards_use_explicit_invocation_authorities() {
        let runtime = guard(RegionGuardCondition::RuntimeState {
            identity: "catalog".into(),
            revision: "7".into(),
        });
        let capability = guard(RegionGuardCondition::Capability {
            requirement: CapabilityRequirement::NativeCode,
        });
        let empty = GuardEnvironment::default();
        assert_eq!(
            empty.evaluate(&runtime, None).unwrap_err().kind,
            GuardFailureKind::RuntimeState
        );
        assert_eq!(
            empty.evaluate(&capability, None).unwrap_err().kind,
            GuardFailureKind::Capability
        );

        let admitted = GuardEnvironment::default()
            .with_runtime_revision("catalog", "7")
            .with_capability(CapabilityRequirement::NativeCode);
        admitted.evaluate(&runtime, None).unwrap();
        admitted.evaluate(&capability, None).unwrap();
    }
}

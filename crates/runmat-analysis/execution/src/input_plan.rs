use runmat_execution::value::ValueRef;
use runmat_execution_artifact::{
    ArtifactAccess, ProgramExecutionDescriptor, ProgramExecutionObject,
};
use runmat_meshing_core::SolverMeshArtifact;

use crate::{
    AnalysisBridgeResult, FrozenDeferredSignal, InputMaterializationExpectation,
    PreparedInputMaterialization,
};

/// An input execution plan before the study's authenticated solver mesh is available.
/// Deferred intent is deliberately not a partially prepared materialization: it has no
/// cardinality, chunks, or materialization identity until resolution against that mesh.
#[derive(Clone, Debug, PartialEq)]
pub enum InputMaterializationPlan {
    Ready(Box<PreparedInputMaterialization>),
    DeferredSignal(Box<FrozenDeferredSignal>),
}

impl From<PreparedInputMaterialization> for InputMaterializationPlan {
    fn from(value: PreparedInputMaterialization) -> Self {
        Self::Ready(Box::new(value))
    }
}

impl InputMaterializationPlan {
    pub fn requires_mesh(&self) -> bool {
        match self {
            Self::DeferredSignal(_) => true,
            Self::Ready(value) => matches!(value.as_ref(),
                PreparedInputMaterialization::Signal(signal) if signal.topology_binding.is_some()),
        }
    }

    pub fn validate(&self) -> AnalysisBridgeResult<()> {
        match self {
            Self::Ready(value) => value.validate(),
            Self::DeferredSignal(value) => value.validate(),
        }
    }

    pub fn validate_closed(&self) -> AnalysisBridgeResult<()> {
        match self {
            Self::Ready(value) => value.validate_closed(),
            // Frozen deferred intent admits no artifact arguments or runtime captures.
            Self::DeferredSignal(value) => value.validate(),
        }
    }

    pub fn rebind(self, access: &ArtifactAccess) -> AnalysisBridgeResult<Self> {
        match self {
            Self::Ready(value) => value.rebind(access).map(Box::new).map(Self::Ready),
            Self::DeferredSignal(value) => {
                value.validate()?;
                Ok(Self::DeferredSignal(value))
            }
        }
    }

    pub fn expectation(&self) -> AnalysisBridgeResult<InputMaterializationPlanExpectation> {
        self.validate()?;
        match self {
            Self::Ready(value) => value
                .expectation()
                .map(Box::new)
                .map(InputMaterializationPlanExpectation::Ready),
            Self::DeferredSignal(value) => Ok(InputMaterializationPlanExpectation::DeferredSignal(
                value.clone(),
            )),
        }
    }

    pub fn input_objects(&self) -> &[ProgramExecutionObject] {
        match self {
            Self::Ready(value) => value.input_objects(),
            Self::DeferredSignal(_) => &[],
        }
    }

    pub fn input_references(&self) -> &[ValueRef] {
        match self {
            Self::Ready(value) => &value.program().input_references,
            Self::DeferredSignal(_) => &[],
        }
    }

    pub fn descriptor(&self) -> AnalysisBridgeResult<ProgramExecutionDescriptor> {
        self.validate()?;
        match self {
            Self::Ready(value) => Ok(value.program().descriptor.clone()),
            Self::DeferredSignal(value) => value.descriptor(),
        }
    }

    pub fn resolve(
        &self,
        mesh: &SolverMeshArtifact,
    ) -> AnalysisBridgeResult<PreparedInputMaterialization> {
        self.validate()?;
        match self {
            Self::Ready(value) => {
                if let PreparedInputMaterialization::Signal(signal) = value.as_ref() {
                    signal.validate_for_mesh(mesh)?;
                }
                Ok(value.as_ref().clone())
            }
            Self::DeferredSignal(value) => value.resolve(mesh).map(|resolved| {
                PreparedInputMaterialization::Signal(Box::new(resolved.into_prepared()))
            }),
        }
    }
}

/// Authority retained from preparation, before topology-dependent intent is resolved.
/// Cold imports must derive the resolved expectation from this original authority and
/// the verified returned mesh, rather than accepting a returned resolved declaration.
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum InputMaterializationPlanExpectation {
    Ready(Box<InputMaterializationExpectation>),
    DeferredSignal(Box<FrozenDeferredSignal>),
}

impl InputMaterializationPlanExpectation {
    pub fn validate(&self) -> AnalysisBridgeResult<()> {
        match self {
            Self::Ready(value) => value.validate(),
            Self::DeferredSignal(value) => value.validate(),
        }
    }

    pub fn resolve(
        &self,
        mesh: &SolverMeshArtifact,
    ) -> AnalysisBridgeResult<InputMaterializationExpectation> {
        self.validate()?;
        match self {
            Self::Ready(value) => {
                if let InputMaterializationExpectation::Signal(signal) = value.as_ref() {
                    signal.validate_for_mesh(mesh)?;
                }
                Ok(value.as_ref().clone())
            }
            Self::DeferredSignal(value) => value
                .resolve(mesh)?
                .prepared()
                .expectation()
                .map(Box::new)
                .map(InputMaterializationExpectation::Signal),
        }
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeSet;

    use super::*;
    use crate::callable::executable::tests as fixture;
    use runmat_analysis_core::{
        AuthoredSignalFieldLocation, AuthoredSignalTopologySelector, SignalShape,
    };
    use runmat_meshing_core::{fixtures::canonical_tetrahedron_solver_mesh, ElementOrder};

    fn deferred_plan() -> InputMaterializationPlan {
        let authored = fixture::signal().with_shape(SignalShape::DeferredTopologyField {
            selector: AuthoredSignalTopologySelector::StudySolverMesh {
                location: AuthoredSignalFieldLocation::Node,
            },
            component_count: 1,
        });
        InputMaterializationPlan::DeferredSignal(Box::new(
            FrozenDeferredSignal::prepare(
                authored,
                fixture::revision(),
                fixture::callable(),
                fixture::envelope(&fixture::analysis()),
                BTreeSet::new(),
                8192,
            )
            .unwrap(),
        ))
    }

    #[test]
    fn deferred_plan_and_retained_expectation_resolve_against_the_same_mesh_authority() {
        let plan = deferred_plan();
        plan.validate_closed().unwrap();
        assert!(plan.input_objects().is_empty());
        assert!(plan.input_references().is_empty());
        let descriptor = plan.descriptor().unwrap();
        let original_expectation = plan.expectation().unwrap();
        let reopened: InputMaterializationPlanExpectation =
            serde_json::from_slice(&serde_json::to_vec(&original_expectation).unwrap()).unwrap();
        assert_eq!(reopened, original_expectation);
        for order in [ElementOrder::Tet4, ElementOrder::Tet10] {
            let mesh = canonical_tetrahedron_solver_mesh(order);
            let prepared = plan.resolve(&mesh).unwrap();
            assert_eq!(prepared.program().descriptor, descriptor);
            assert_eq!(
                prepared.expectation().unwrap(),
                reopened.resolve(&mesh).unwrap()
            );
            let ready = InputMaterializationPlan::from(prepared.clone());
            assert_eq!(ready.resolve(&mesh).unwrap(), prepared);
            assert_eq!(
                ready.expectation().unwrap().resolve(&mesh).unwrap(),
                prepared.expectation().unwrap()
            );
            let foreign_order = match order {
                ElementOrder::Tet4 => ElementOrder::Tet10,
                ElementOrder::Tet10 => ElementOrder::Tet4,
            };
            let foreign = canonical_tetrahedron_solver_mesh(foreign_order);
            assert!(ready.resolve(&foreign).is_err());
            assert!(ready.expectation().unwrap().resolve(&foreign).is_err());
        }
        assert_eq!(plan.expectation().unwrap(), original_expectation);
    }
}

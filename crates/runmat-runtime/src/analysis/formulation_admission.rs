use std::fmt;

use runmat_analysis_core::{
    AnalysisProcedure, BoundaryTransferImplementation, FormulationRequirement, PhysicsFamily,
    SolutionSpace, SourceImplementation, SpatialOperatorImplementation, TopologyImplementation,
};
use runmat_analysis_fea::admission::{
    admit_formulation, registered_formulations, FormulationAdmissionRejection,
    FormulationAdmissionRequest,
};
use runmat_analysis_fea::AnalysisRunIdentity;
use runmat_meshing_core::{ElementOrder, SolverMeshArtifact};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum RuntimeProxyFormulation {
    AcousticHashedTargetGraphProjection,
    Domain {
        topology: RuntimeDomainTopology,
        source: RuntimeDomainSource,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum RuntimeDomainTopology {
    ImplicitChannel,
    CanonicalTet4,
    CanonicalTet10,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum RuntimeDomainSource {
    OrdinalProfile,
}

impl RuntimeDomainTopology {
    pub(super) fn from_solver_mesh(solver_mesh: Option<&SolverMeshArtifact>) -> Self {
        match solver_mesh.map(|mesh| mesh.resolved_request.element_order) {
            Some(ElementOrder::Tet4) => Self::CanonicalTet4,
            Some(ElementOrder::Tet10) => Self::CanonicalTet10,
            None => Self::ImplicitChannel,
        }
    }

    fn topology(self) -> TopologyImplementation {
        match self {
            Self::ImplicitChannel => TopologyImplementation::ImplicitChannel,
            Self::CanonicalTet4 => TopologyImplementation::CanonicalSolverMeshTet4,
            Self::CanonicalTet10 => TopologyImplementation::CanonicalSolverMeshTet10,
        }
    }
}

impl RuntimeDomainSource {
    fn source(self) -> SourceImplementation {
        match self {
            Self::OrdinalProfile => SourceImplementation::OrdinalDistribution,
        }
    }

    fn uncoupled_boundary_transfer(self) -> BoundaryTransferImplementation {
        match self {
            Self::OrdinalProfile => BoundaryTransferImplementation::GlobalCountProfile,
        }
    }
}

#[derive(Debug)]
pub(super) enum RuntimeProxyAdmissionError {
    UnregisteredActualBranch {
        family: PhysicsFamily,
        procedure: AnalysisProcedure,
        branch: RuntimeProxyFormulation,
    },
    Rejected(FormulationAdmissionRejection),
}

impl fmt::Display for RuntimeProxyAdmissionError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnregisteredActualBranch {
                family,
                procedure,
                branch,
            } => write!(
                formatter,
                "no registered exact formulation matches the runtime branch: family={family:?} procedure={procedure:?} branch={branch:?}"
            ),
            Self::Rejected(error) => error.fmt(formatter),
        }
    }
}

impl std::error::Error for RuntimeProxyAdmissionError {}

pub(super) fn admit_runtime_proxy(
    family: PhysicsFamily,
    procedure: AnalysisProcedure,
    requirement: FormulationRequirement,
    branch: RuntimeProxyFormulation,
) -> Result<AnalysisRunIdentity, RuntimeProxyAdmissionError> {
    let realization = registered_formulations(family, procedure)
        .into_iter()
        .find(|entry| matches_actual_branch(family, entry.realization(), branch))
        .map(|entry| entry.realization().clone())
        .ok_or(RuntimeProxyAdmissionError::UnregisteredActualBranch {
            family,
            procedure,
            branch,
        })?;
    admit_formulation(FormulationAdmissionRequest {
        family,
        procedure,
        solution_space: SolutionSpace::Direct,
        requirement,
        realization,
    })
    .map(|receipt| receipt.into_identity())
    .map_err(RuntimeProxyAdmissionError::Rejected)
}

fn matches_actual_branch(
    family: PhysicsFamily,
    realization: &runmat_analysis_core::ExactFormulationRealization,
    branch: RuntimeProxyFormulation,
) -> bool {
    match branch {
        RuntimeProxyFormulation::AcousticHashedTargetGraphProjection => {
            realization.components.len() == 1
                && realization.components[0].spatial_operator
                    == SpatialOperatorImplementation::ReducedAcousticHelmholtz
                && realization.components[0].topologies == [TopologyImplementation::GeneratedGraph]
                && realization.components[0].sources
                    == [SourceImplementation::AcousticHashedTargetGraphProjection]
                && realization.boundary_transfers
                    == [BoundaryTransferImplementation::GlobalCountProfile]
        }
        RuntimeProxyFormulation::Domain {
            topology: domain,
            source,
        } => {
            let topology = domain.topology();
            let source_implementation = source.source();
            let fluid_component = realization.components.iter().find(|component| {
                component.spatial_operator == SpatialOperatorImplementation::ReducedChannelFlow
            });
            let expected_boundary_transfer = match family {
                PhysicsFamily::Fluid => source.uncoupled_boundary_transfer(),
                PhysicsFamily::Coupled(_) => {
                    BoundaryTransferImplementation::HeuristicInterfaceTransfer
                }
                _ => return false,
            };
            fluid_component.is_some_and(|component| {
                component.topologies == [topology] && component.sources == [source_implementation]
            }) && realization.boundary_transfers == [expected_boundary_transfer]
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use runmat_analysis_core::CoupledPhysicsFamily;

    #[test]
    fn proxy_branch_selection_preserves_hashed_acoustic_and_implicit_flow_paths() {
        let acoustic = admit_runtime_proxy(
            PhysicsFamily::Acoustic,
            AnalysisProcedure::HarmonicResponse,
            FormulationRequirement::Experimental,
            RuntimeProxyFormulation::AcousticHashedTargetGraphProjection,
        )
        .expect("generated acoustic formulation must be registered");
        assert_eq!(
            acoustic.realization().components[0].topologies,
            [TopologyImplementation::GeneratedGraph]
        );
        assert_eq!(
            acoustic.realization().components[0].sources,
            [SourceImplementation::AcousticHashedTargetGraphProjection]
        );

        let fluid = admit_runtime_proxy(
            PhysicsFamily::Fluid,
            AnalysisProcedure::Static,
            FormulationRequirement::Experimental,
            RuntimeProxyFormulation::Domain {
                topology: RuntimeDomainTopology::ImplicitChannel,
                source: RuntimeDomainSource::OrdinalProfile,
            },
        )
        .expect("implicit-channel CFD formulation must be registered");
        assert_eq!(
            fluid.realization().components[0].topologies,
            [TopologyImplementation::ImplicitChannel]
        );

        let fsi = admit_runtime_proxy(
            PhysicsFamily::Coupled(CoupledPhysicsFamily::FluidStructureInteraction),
            AnalysisProcedure::TransientResponse,
            FormulationRequirement::Experimental,
            RuntimeProxyFormulation::Domain {
                topology: RuntimeDomainTopology::ImplicitChannel,
                source: RuntimeDomainSource::OrdinalProfile,
            },
        )
        .expect("implicit-channel FSI formulation must be registered");
        assert_eq!(
            fsi.realization().boundary_transfers,
            [BoundaryTransferImplementation::HeuristicInterfaceTransfer]
        );
    }

    #[test]
    fn production_requirement_rejects_unqualified_proxy_branch() {
        let error = admit_runtime_proxy(
            PhysicsFamily::Acoustic,
            AnalysisProcedure::HarmonicResponse,
            FormulationRequirement::Production,
            RuntimeProxyFormulation::AcousticHashedTargetGraphProjection,
        )
        .expect_err("current acoustic proxy is not production-qualified");
        assert!(matches!(error, RuntimeProxyAdmissionError::Rejected(_)));
    }
}

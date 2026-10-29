use std::fmt;

use runmat_analysis_core::{
    AnalysisProcedure, BoundaryTransferImplementation, PhysicsFamily, SolutionSpace,
    SourceImplementation, SpatialOperatorImplementation, TopologyImplementation,
};
use runmat_analysis_fea::admission::{
    admit_formulation, registered_formulations, FormulationAdmissionRejection,
    FormulationAdmissionRequest,
};
use runmat_analysis_fea::AnalysisRunIdentity;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum RuntimeProxyFormulation {
    AcousticHashedTargetGraphProjection,
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
        realization,
    })
    .map(|receipt| receipt.into_identity())
    .map_err(RuntimeProxyAdmissionError::Rejected)
}

fn matches_actual_branch(
    _family: PhysicsFamily,
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
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn proxy_branch_selection_preserves_hashed_acoustic_path() {
        let acoustic = admit_runtime_proxy(
            PhysicsFamily::Acoustic,
            AnalysisProcedure::HarmonicResponse,
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
    }

    #[test]
    fn proxy_branch_cannot_be_reused_for_another_physics_family() {
        let error = admit_runtime_proxy(
            PhysicsFamily::Structural,
            AnalysisProcedure::HarmonicResponse,
            RuntimeProxyFormulation::AcousticHashedTargetGraphProjection,
        )
        .expect_err("an acoustic realization must not qualify a structural request");
        assert!(matches!(
            error,
            RuntimeProxyAdmissionError::UnregisteredActualBranch {
                family: PhysicsFamily::Structural,
                ..
            }
        ));
    }
}

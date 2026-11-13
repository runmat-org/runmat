//! Exact positive-frequency electric-field curl-curl and complex-mass realization.
//! An algorithm identity is not proof of authored PEC coverage, affine geometry, source
//! realization, worker admission, or production physics qualification.
use runmat_analysis_core::{
    AnalysisProcedure, BoundaryTransferImplementation, ConstitutiveImplementation,
    CouplingImplementation, ExactFormulationComponent, ExactFormulationRealization, PhysicsFamily,
    RecoveryImplementation, SolutionSpace, SourceImplementation, SpatialOperatorImplementation,
    TemporalImplementation, TopologyImplementation,
};
use runmat_meshing_core::{ElementOrder, SolverMeshArtifact};

use super::{admit_formulation, AnalysisRunIdentity, FormulationAdmissionRequest};
use crate::contracts::FeaRunError;

/// Admission of the explicitly implemented full exterior electric perfect conductor envelope.
/// The physical input owner separately proves exact boundary intent and geometric eligibility.
pub fn admit_canonical_maxwell(
    mesh: &SolverMeshArtifact,
) -> Result<AnalysisRunIdentity, FeaRunError> {
    admit_formulation(FormulationAdmissionRequest {
        family: PhysicsFamily::Electromagnetic,
        procedure: AnalysisProcedure::HarmonicResponse,
        solution_space: SolutionSpace::Direct,
        realization: canonical_maxwell_realization(mesh.resolved_request.element_order),
    })
    .map(|receipt| receipt.into_identity())
    .map_err(Into::into)
}

pub(super) fn registered_realizations() -> Vec<ExactFormulationRealization> {
    [ElementOrder::Tet4, ElementOrder::Tet10]
        .into_iter()
        .map(canonical_maxwell_realization)
        .collect()
}

/// Lowest-order edge approximation on either supported canonical mesh order; Tet10 geometry
/// does not imply a higher-order Maxwell field basis.
pub fn canonical_maxwell_realization(order: ElementOrder) -> ExactFormulationRealization {
    let topology = match order {
        ElementOrder::Tet4 => TopologyImplementation::CanonicalSolverMeshTet4,
        ElementOrder::Tet10 => TopologyImplementation::CanonicalSolverMeshTet10,
    };
    ExactFormulationRealization {
        components: vec![ExactFormulationComponent {
            spatial_operator:
                SpatialOperatorImplementation::MaxwellLowestOrderEdgeCurlCurlComplexMass,
            constitutive:
                ConstitutiveImplementation::ElementOwnedIsotropicHarmonicElectromagneticLaws,
            recovery: RecoveryImplementation::MaxwellSolverMeshElectricFluxFieldPower,
            topologies: vec![topology, TopologyImplementation::OrientedEdgeIncidence],
            sources: vec![SourceImplementation::CanonicalVolumeProjection],
        }],
        temporals: vec![TemporalImplementation::MaxwellComplexElectricFieldGeneralSolve],
        coupling: CouplingImplementation::Uncoupled,
        boundary_transfers: vec![
            BoundaryTransferImplementation::FullExteriorElectricPerfectConductor,
        ],
    }
}

#[cfg(test)]
mod tests;

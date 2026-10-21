//! Exact identity for affine lowest-order edge vector-potential magnetostatics.
//! Admission declares numerical algorithms, not a native resource/restart contract.
use runmat_analysis_core::{
    AnalysisProcedure, BoundaryTransferImplementation, ConstitutiveImplementation,
    CouplingImplementation, ExactFormulationComponent, ExactFormulationRealization, PhysicsFamily,
    RecoveryImplementation, SolutionSpace, SourceImplementation, SpatialOperatorImplementation,
    TemporalImplementation, TopologyImplementation,
};
use runmat_meshing_core::{ElementOrder, SolverMeshArtifact};

use super::{admit_formulation, AnalysisRunIdentity, FormulationAdmissionRequest};
use crate::contracts::FeaRunError;

/// Admit the explicit full-exterior tangential ground and weak Coulomb gauge
/// envelope. Geometry affinity and authored policy must be proved by input
/// compilation; a mesh order alone does not establish either fact.
pub fn admit_canonical_magnetostatic(
    mesh: &SolverMeshArtifact,
) -> Result<AnalysisRunIdentity, FeaRunError> {
    admit_formulation(FormulationAdmissionRequest {
        family: PhysicsFamily::Electromagnetic,
        procedure: AnalysisProcedure::Static,
        solution_space: SolutionSpace::Direct,
        realization: realization(mesh.resolved_request.element_order),
    })
    .map(|receipt| receipt.into_identity())
    .map_err(Into::into)
}

pub(super) fn registered_realizations() -> Vec<ExactFormulationRealization> {
    [ElementOrder::Tet4, ElementOrder::Tet10]
        .into_iter()
        .map(realization)
        .collect()
}

pub(crate) fn realization(order: ElementOrder) -> ExactFormulationRealization {
    let topology = match order {
        ElementOrder::Tet4 => TopologyImplementation::CanonicalSolverMeshTet4,
        ElementOrder::Tet10 => TopologyImplementation::CanonicalSolverMeshTet10,
    };
    ExactFormulationRealization {
        components: vec![ExactFormulationComponent {
            spatial_operator: SpatialOperatorImplementation::MagneticGalerkinLowestOrderEdgeCurlCurl,
            constitutive: ConstitutiveImplementation::ElementOwnedIsotropicMagneticPermeability,
            recovery: RecoveryImplementation::MagneticSolverMeshVectorPotentialFluxFieldEnergy,
            topologies: vec![topology, TopologyImplementation::OrientedEdgeIncidence],
            sources: vec![SourceImplementation::CanonicalVolumeProjection],
        }],
        temporals: vec![TemporalImplementation::MagnetostaticWeakCoulombSaddleSolve],
        coupling: CouplingImplementation::Uncoupled,
        boundary_transfers: vec![
            BoundaryTransferImplementation::FullExteriorTangentialPotentialGroundWithWeakCoulombGauge,
        ],
    }
}

#[cfg(test)]
mod tests;

//! Exact canonical steady finite-volume realization, never the reduced-channel identity.
use super::{admit_formulation, AnalysisRunIdentity, FormulationAdmissionRequest};
use crate::contracts::FeaRunError;
use runmat_analysis_core::{
    AnalysisProcedure, BoundaryTransferImplementation, ConstitutiveImplementation,
    CouplingImplementation, ExactFormulationComponent, ExactFormulationRealization, PhysicsFamily,
    RecoveryImplementation, SolutionSpace, SourceImplementation, SpatialOperatorImplementation,
    TemporalImplementation, TopologyImplementation,
};
use runmat_meshing_core::{ElementOrder, SolverMeshArtifact};

pub fn admit_canonical_fluid_steady(
    mesh: &SolverMeshArtifact,
) -> Result<AnalysisRunIdentity, FeaRunError> {
    admit_canonical_fluid(
        mesh,
        AnalysisProcedure::Static,
        canonical_fluid_steady_realization(),
    )
}

pub fn admit_canonical_fluid_transient(
    mesh: &SolverMeshArtifact,
) -> Result<AnalysisRunIdentity, FeaRunError> {
    admit_canonical_fluid(
        mesh,
        AnalysisProcedure::TransientResponse,
        canonical_fluid_transient_realization(),
    )
}
fn admit_canonical_fluid(
    mesh: &SolverMeshArtifact,
    procedure: AnalysisProcedure,
    realization: ExactFormulationRealization,
) -> Result<AnalysisRunIdentity, FeaRunError> {
    if mesh.resolved_request.element_order != ElementOrder::Tet4
        || mesh
            .topology
            .volume_elements
            .iter()
            .any(|element| element.order != ElementOrder::Tet4)
    {
        return Err(FeaRunError::InvalidModel(
            "canonical fluid admits only Tet4 cell geometry".into(),
        ));
    }
    admit_formulation(FormulationAdmissionRequest {
        family: PhysicsFamily::Fluid,
        procedure,
        solution_space: SolutionSpace::Direct,
        realization,
    })
    .map(|receipt| receipt.into_identity())
    .map_err(Into::into)
}

pub fn canonical_fluid_steady_realization() -> ExactFormulationRealization {
    ExactFormulationRealization {
        components: vec![ExactFormulationComponent {
            spatial_operator:
                SpatialOperatorImplementation::FluidCellCenteredConservativeMomentumPressure,
            constitutive: ConstitutiveImplementation::CellOwnedIncompressibleNewtonianFluid,
            recovery: RecoveryImplementation::FluidOriginalCellVelocityPressureOwnerFaceFlux,
            topologies: vec![
                TopologyImplementation::CanonicalSolverMeshTet4,
                TopologyImplementation::OwnerOrientedFiniteVolumeFaceIncidence,
            ],
            sources: vec![
                SourceImplementation::CanonicalVolumeProjection,
                SourceImplementation::CanonicalBoundaryProjection,
            ],
        }],
        temporals: vec![TemporalImplementation::FluidSteadyNonorthogonalPressureCorrection],
        coupling: CouplingImplementation::Uncoupled,
        boundary_transfers: vec![BoundaryTransferImplementation::CanonicalBoundaryProjection],
    }
}

pub fn canonical_fluid_transient_realization() -> ExactFormulationRealization {
    let mut realization = canonical_fluid_steady_realization();
    realization.temporals =
        vec![TemporalImplementation::FluidBackwardEulerConservativeMomentumPressure];
    realization
}

#[cfg(test)]
mod tests {
    use super::*;
    use runmat_meshing_core::fixtures::canonical_tetrahedron_solver_mesh;

    #[test]
    fn canonical_fluid_has_one_exact_identity_not_channel_or_tet10() {
        let mesh = canonical_tetrahedron_solver_mesh(ElementOrder::Tet4);
        let identity = admit_canonical_fluid_steady(&mesh).unwrap();
        assert_eq!(
            identity.realization(),
            &canonical_fluid_steady_realization()
        );
        let entries =
            super::super::registered_formulations(PhysicsFamily::Fluid, AnalysisProcedure::Static);
        assert_eq!(
            entries
                .iter()
                .filter(|entry| entry.realization() == identity.realization())
                .count(),
            1
        );
        assert!(admit_canonical_fluid_steady(&mesh).is_err());
        assert!(
            admit_canonical_fluid_steady(&canonical_tetrahedron_solver_mesh(ElementOrder::Tet10))
                .is_err()
        );
        for axis in 0..4 {
            let mut realization = canonical_fluid_steady_realization();
            match axis {
                0 => {
                    realization.components[0].spatial_operator =
                        SpatialOperatorImplementation::ReducedChannelFlow
                }
                1 => {
                    realization.components[0].recovery =
                        RecoveryImplementation::CfdReducedChannelProfile
                }
                2 => {
                    realization.components[0].topologies =
                        vec![TopologyImplementation::ImplicitChannel]
                }
                _ => realization.temporals = vec![TemporalImplementation::ThermalSteadyLinearSolve],
            }
            assert!(admit_formulation(FormulationAdmissionRequest {
                family: PhysicsFamily::Fluid,
                procedure: AnalysisProcedure::Static,
                solution_space: SolutionSpace::Direct,
                realization,
            })
            .is_err());
        }
    }
}

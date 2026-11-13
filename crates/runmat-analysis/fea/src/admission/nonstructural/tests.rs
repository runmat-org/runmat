use super::*;
use runmat_meshing_core::fixtures::canonical_tetrahedron_solver_mesh;

#[test]
fn thermal_mesh_and_procedure_select_the_registry_issued_identity() {
    for (order, topology) in [
        (
            ElementOrder::Tet4,
            TopologyImplementation::CanonicalSolverMeshTet4,
        ),
        (
            ElementOrder::Tet10,
            TopologyImplementation::CanonicalSolverMeshTet10,
        ),
    ] {
        let mesh = canonical_tetrahedron_solver_mesh(order);
        for procedure in [
            AnalysisProcedure::Static,
            AnalysisProcedure::TransientResponse,
        ] {
            let identity = admit_nonstructural(
                PhysicsFamily::Thermal,
                procedure,
                Some(&mesh),
                SourceImplementation::CanonicalVolumeProjection,
                BoundaryTransferImplementation::CanonicalBoundaryProjection,
            )
            .unwrap();
            assert_eq!(identity.family(), PhysicsFamily::Thermal);
            assert_eq!(identity.procedure(), procedure);
            assert_eq!(identity.solution_space(), SolutionSpace::Direct);
            assert!(identity.realization().components.iter().any(|component| {
                component.topologies.contains(&topology)
                    && component
                        .sources
                        .contains(&SourceImplementation::CanonicalVolumeProjection)
            }));
            let receipt = admit_formulation(FormulationAdmissionRequest {
                family: PhysicsFamily::Thermal,
                procedure,
                solution_space: SolutionSpace::Direct,
                realization: identity.realization().clone(),
            })
            .unwrap();
            assert_eq!(identity, receipt.into_identity());
        }
    }
}

#[test]
fn thermal_admission_preserves_missing_mesh_error() {
    let error = admit_nonstructural(
        PhysicsFamily::Thermal,
        AnalysisProcedure::Static,
        None,
        SourceImplementation::CanonicalVolumeProjection,
        BoundaryTransferImplementation::CanonicalBoundaryProjection,
    )
    .unwrap_err();
    assert!(matches!(error, FeaRunError::InvalidModel(message)
        if message == "no meshless FEA pipeline admission branch is defined for Thermal"));
}

#[test]
fn nonlinear_thermal_identity_is_exact() {
    for order in [ElementOrder::Tet4, ElementOrder::Tet10] {
        let mesh = canonical_tetrahedron_solver_mesh(order);
        for (procedure, temporal) in [
            (
                AnalysisProcedure::Static,
                TemporalImplementation::ThermalSteadyDampedNewton,
            ),
            (
                AnalysisProcedure::TransientResponse,
                TemporalImplementation::ThermalBackwardEulerDampedNewton,
            ),
        ] {
            let admit = || {
                admit_nonstructural_with_temporal(
                    PhysicsFamily::Thermal,
                    procedure,
                    Some(&mesh),
                    SourceImplementation::CanonicalVolumeProjection,
                    BoundaryTransferImplementation::CanonicalBoundaryProjection,
                    Some(temporal),
                )
            };
            let identity = admit().unwrap();
            assert_eq!(identity.realization().temporals, vec![temporal]);
            assert!(admit_nonstructural_with_temporal(
                PhysicsFamily::Thermal,
                procedure,
                Some(&mesh),
                SourceImplementation::CanonicalVolumeProjection,
                BoundaryTransferImplementation::CanonicalBoundaryProjection,
                Some(TemporalImplementation::AcousticCanonicalHarmonicSolve),
            )
            .is_err());
        }
    }
}

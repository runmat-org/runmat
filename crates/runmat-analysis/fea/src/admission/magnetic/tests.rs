use super::*;
use crate::admission::{admit_nonstructural, registered_formulations};
use runmat_meshing_core::fixtures::canonical_tetrahedron_solver_mesh;

#[test]
fn magnetic_identity_is_distinct_and_topology_specific() {
    let mut identities = Vec::new();
    for order in [ElementOrder::Tet4, ElementOrder::Tet10] {
        let mesh = canonical_tetrahedron_solver_mesh(order);
        let identity = admit_canonical_magnetostatic(&mesh).unwrap();
        assert_eq!(identity.realization(), &realization(order));
        assert!(
            registered_formulations(PhysicsFamily::Electromagnetic, AnalysisProcedure::Static)
                .iter()
                .any(|entry| entry.realization() == identity.realization())
        );
        let wire = serde_json::to_vec(&identity).unwrap();
        assert_eq!(
            serde_json::from_slice::<AnalysisRunIdentity>(&wire).unwrap(),
            identity
        );
        assert!(!identities.contains(&identity));
        identities.push(identity);
        // Re-admission of the same mesh is deterministic.
        assert_eq!(
            &admit_canonical_magnetostatic(&mesh).unwrap(),
            identities.last().unwrap()
        );
        assert!(admit_nonstructural(
            PhysicsFamily::Electromagnetic, AnalysisProcedure::Static, Some(&mesh),
            SourceImplementation::CanonicalVolumeProjection,
            BoundaryTransferImplementation::FullExteriorTangentialPotentialGroundWithWeakCoulombGauge,
        ).is_err());
    }
}

#[test]
fn magnetic_identity_axis_splicing_and_production_forgery_fail_closed() {
    let exact = realization(ElementOrder::Tet4);
    for axis in 0..8 {
        let mut forged = exact.clone();
        match axis {
            0 => {
                forged.components[0].spatial_operator =
                    SpatialOperatorImplementation::MaxwellLowestOrderEdgeCurlCurlComplexMass
            }
            1 => {
                forged.components[0].constitutive =
                    ConstitutiveImplementation::ElementOwnedIsotropicPermittivity
            }
            2 => {
                forged.components[0].recovery =
                    RecoveryImplementation::MaxwellSolverMeshElectricFluxFieldPower
            }
            3 => {
                forged.components[0].topologies.remove(1);
            }
            4 => {
                forged.components[0].sources =
                    vec![SourceImplementation::CanonicalBoundaryProjection]
            }
            5 => {
                forged.boundary_transfers =
                    vec![BoundaryTransferImplementation::ImplicitNaturalBoundaryAndGaugeAnchor]
            }
            6 => forged.temporals = vec![TemporalImplementation::ElectrostaticPotentialLinearSolve],
            7 => forged.coupling = CouplingImplementation::StaggeredElectrothermalConductanceLine,
            _ => unreachable!(),
        }
        assert!(admit_formulation(FormulationAdmissionRequest {
            family: PhysicsFamily::Electromagnetic,
            procedure: AnalysisProcedure::Static,
            solution_space: SolutionSpace::Direct,
            realization: forged,
        })
        .is_err());
    }
    admit_formulation(FormulationAdmissionRequest {
        family: PhysicsFamily::Electromagnetic,
        procedure: AnalysisProcedure::Static,
        solution_space: SolutionSpace::Direct,
        realization: exact,
    })
    .expect("exact magnetostatic realization is admitted");
    let mesh = canonical_tetrahedron_solver_mesh(ElementOrder::Tet4);
    let identity = admit_canonical_magnetostatic(&mesh).unwrap();
    let mut wire = serde_json::to_value(identity).unwrap();
    wire["qualification"] = serde_json::json!("production");
    assert!(serde_json::from_value::<AnalysisRunIdentity>(wire).is_err());
}

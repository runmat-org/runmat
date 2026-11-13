use super::*;
use crate::{
    admission::{admit_nonstructural, registered_formulations},
    contracts::{validate_solver_formulation_compatibility, SolverMethod, SolverPreconditioner},
};
use runmat_meshing_core::fixtures::canonical_tetrahedron_solver_mesh;

#[test]
fn volume_charge_has_distinct_identity_and_no_conduction_admission() {
    for order in [ElementOrder::Tet4, ElementOrder::Tet10] {
        let mesh = canonical_tetrahedron_solver_mesh(order);
        let charged = admit_canonical_electrical_static(
            StaticElectricalEquation::Electrostatics,
            CanonicalElectricalSource::VolumeCharge,
            &mesh,
        )
        .unwrap();
        let free = admit_canonical_electrical_static(
            StaticElectricalEquation::Electrostatics,
            CanonicalElectricalSource::SourceFree,
            &mesh,
        )
        .unwrap();
        assert_ne!(charged, free);
        assert!(admit_canonical_electrical_static(
            StaticElectricalEquation::SteadyConduction,
            CanonicalElectricalSource::VolumeCharge,
            &mesh,
        )
        .is_err());
    }
}

#[test]
fn equation_and_topology_select_four_distinct_registry_identities() {
    let mut identities = Vec::new();
    for equation in [
        StaticElectricalEquation::Electrostatics,
        StaticElectricalEquation::SteadyConduction,
    ] {
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
            let identity = admit_canonical_electrical_static(
                equation,
                CanonicalElectricalSource::SourceFree,
                &mesh,
            )
            .unwrap();
            assert_eq!(identity.family(), PhysicsFamily::Electromagnetic);
            assert_eq!(identity.procedure(), AnalysisProcedure::Static);
            assert_eq!(identity.solution_space(), SolutionSpace::Direct);
            assert_eq!(
                identity.realization(),
                &realization(equation, order, CanonicalElectricalSource::SourceFree)
            );
            assert_eq!(
                identity.realization().components[0].topologies,
                vec![topology]
            );
            assert!(registered_formulations(
                PhysicsFamily::Electromagnetic,
                AnalysisProcedure::Static
            )
            .iter()
            .any(|registered| registered.realization() == identity.realization()));
            let wire = serde_json::to_vec(&identity).unwrap();
            assert_eq!(
                serde_json::from_slice::<AnalysisRunIdentity>(&wire).unwrap(),
                identity
            );
            validate_solver_formulation_compatibility(
                &identity,
                SolverMethod::AssembledPcg,
                SolverPreconditioner::Jacobi,
            )
            .unwrap();
            assert!(validate_solver_formulation_compatibility(
                &identity,
                SolverMethod::GeneralGmres,
                SolverPreconditioner::BlockJacobi
            )
            .is_err());
            assert!(!identities.contains(&identity));
            identities.push(identity);
        }
    }
    assert_eq!(identities.len(), 4);
}

#[test]
fn cross_equation_or_harmonic_identity_splicing_fails_closed() {
    for equation in [
        StaticElectricalEquation::Electrostatics,
        StaticElectricalEquation::SteadyConduction,
    ] {
        for order in [ElementOrder::Tet4, ElementOrder::Tet10] {
            let mesh = canonical_tetrahedron_solver_mesh(order);
            admit_canonical_electrical_static(
                equation,
                CanonicalElectricalSource::SourceFree,
                &mesh,
            )
            .expect("canonical source-free electrical identity is admitted");
            let mut mixed = realization(equation, order, CanonicalElectricalSource::SourceFree);
            mixed.temporals = vec![TemporalImplementation::MaxwellComplexElectricFieldGeneralSolve];
            assert!(admit_formulation(FormulationAdmissionRequest {
                family: PhysicsFamily::Electromagnetic,
                procedure: AnalysisProcedure::Static,
                solution_space: SolutionSpace::Direct,
                realization: mixed,
            })
            .is_err());
            let mut crossed = realization(equation, order, CanonicalElectricalSource::SourceFree);
            crossed.components[0].recovery = match equation {
                StaticElectricalEquation::Electrostatics => {
                    RecoveryImplementation::ElectricalSolverMeshFieldCurrentJoulePower
                }
                StaticElectricalEquation::SteadyConduction => {
                    RecoveryImplementation::ElectricalSolverMeshFieldDisplacementEnergy
                }
            };
            assert!(admit_formulation(FormulationAdmissionRequest {
                family: PhysicsFamily::Electromagnetic,
                procedure: AnalysisProcedure::Static,
                solution_space: SolutionSpace::Direct,
                realization: crossed,
            })
            .is_err());
        }
    }
}

#[test]
fn generic_nonstructural_selector_cannot_guess_a_static_electrical_equation() {
    let mesh = canonical_tetrahedron_solver_mesh(ElementOrder::Tet4);
    assert!(admit_nonstructural(
        PhysicsFamily::Electromagnetic,
        AnalysisProcedure::Static,
        Some(&mesh),
        SourceImplementation::NotApplicable,
        BoundaryTransferImplementation::CanonicalBoundaryProjection,
    )
    .is_err());
}

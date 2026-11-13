use super::*;
use crate::admission::{admit_canonical_magnetostatic, registered_formulations};
use runmat_meshing_core::fixtures::canonical_tetrahedron_solver_mesh;

#[test]
fn retired_proxy_contract_tags_cannot_deserialize_into_live_formulation_authority() {
    for tag in [
        "reduced_electromagnetic_potential",
        "reduced_edge_graph_curl_curl",
    ] {
        assert!(
            serde_json::from_value::<SpatialOperatorImplementation>(serde_json::json!(tag))
                .is_err()
        );
    }
    assert!(
        serde_json::from_value::<TemporalImplementation>(serde_json::json!(
            "electromagnetic_reduced_edge_graph_harmonic_solve"
        ))
        .is_err()
    );
    assert!(
        serde_json::from_value::<ConstitutiveImplementation>(serde_json::json!(
            "reduced_frequency_material_response"
        ))
        .is_err()
    );
    for tag in [
        "electromagnetic_potential_profile",
        "reduced_edge_graph_flux",
    ] {
        assert!(serde_json::from_value::<RecoveryImplementation>(serde_json::json!(tag)).is_err());
    }
    assert!(
        serde_json::from_value::<SourceImplementation>(serde_json::json!(
            "canonical_edge_projection"
        ))
        .is_err()
    );
    assert!(
        serde_json::from_value::<BoundaryTransferImplementation>(serde_json::json!(
            "canonical_ground_with_ordinal_insulation_profile"
        ))
        .is_err()
    );
}

#[test]
fn canonical_harmonic_maxwell_is_exact_order_bound_and_distinct_from_static_and_graph() {
    let mut identities = Vec::new();
    for order in [ElementOrder::Tet4, ElementOrder::Tet10] {
        let mesh = canonical_tetrahedron_solver_mesh(order);
        let identity = admit_canonical_maxwell(&mesh).unwrap();
        assert_eq!(
            identity.realization(),
            &canonical_maxwell_realization(order)
        );
        let registered = registered_formulations(
            PhysicsFamily::Electromagnetic,
            AnalysisProcedure::HarmonicResponse,
        );
        assert_eq!(
            registered
                .iter()
                .filter(|entry| entry.realization() == identity.realization())
                .count(),
            1
        );
        assert_eq!(registered.len(), 2);
        for entry in &registered {
            assert_eq!(
                entry.realization().temporals,
                vec![TemporalImplementation::MaxwellComplexElectricFieldGeneralSolve]
            );
        }
        assert_ne!(admit_canonical_magnetostatic(&mesh).unwrap(), identity);
        assert!(!identities.contains(&identity));
        let wire = serde_json::to_vec(&identity).unwrap();
        assert_eq!(
            serde_json::from_slice::<AnalysisRunIdentity>(&wire).unwrap(),
            identity
        );
        identities.push(identity);
        // Re-admission of the same mesh is deterministic.
        assert_eq!(
            &admit_canonical_maxwell(&mesh).unwrap(),
            identities.last().unwrap()
        );
    }
}

#[test]
fn maxwell_axis_splicing_and_foreign_procedure_fail_closed() {
    for order in [ElementOrder::Tet4, ElementOrder::Tet10] {
        let exact = canonical_maxwell_realization(order);
        for axis in 0..8 {
            let mut forged = exact.clone();
            match axis {
                0 => forged.components[0].spatial_operator = SpatialOperatorImplementation::MagneticGalerkinLowestOrderEdgeCurlCurl,
                1 => forged.components[0].constitutive = ConstitutiveImplementation::ElementOwnedIsotropicMagneticPermeability,
                2 => forged.components[0].recovery = RecoveryImplementation::MagneticSolverMeshVectorPotentialFluxFieldEnergy,
                3 => { forged.components[0].topologies.remove(1); }
                4 => forged.components[0].sources = vec![SourceImplementation::CanonicalBoundaryProjection],
                5 => forged.boundary_transfers = vec![BoundaryTransferImplementation::FullExteriorTangentialPotentialGroundWithWeakCoulombGauge],
                6 => forged.temporals = vec![TemporalImplementation::MagnetostaticWeakCoulombSaddleSolve],
                7 => forged.coupling = CouplingImplementation::StaggeredElectrothermalConductanceLine,
                _ => unreachable!(),
            }
            assert!(admit_formulation(FormulationAdmissionRequest {
                family: PhysicsFamily::Electromagnetic,
                procedure: AnalysisProcedure::HarmonicResponse,
                solution_space: SolutionSpace::Direct,
                realization: forged,
            })
            .is_err());
        }
        assert!(admit_formulation(FormulationAdmissionRequest {
            family: PhysicsFamily::Electromagnetic,
            procedure: AnalysisProcedure::Static,
            solution_space: SolutionSpace::Direct,
            realization: exact.clone(),
        })
        .is_err());
        admit_formulation(FormulationAdmissionRequest {
            family: PhysicsFamily::Electromagnetic,
            procedure: AnalysisProcedure::HarmonicResponse,
            solution_space: SolutionSpace::Direct,
            realization: exact,
        })
        .expect("exact Maxwell realization is admitted");
    }
}

#[test]
fn serialized_identity_cannot_upgrade_qualification_or_replace_one_axis() {
    let mesh = canonical_tetrahedron_solver_mesh(ElementOrder::Tet4);
    let identity = admit_canonical_maxwell(&mesh).unwrap();
    let mut wire = serde_json::to_value(&identity).unwrap();
    wire["qualification"] = serde_json::json!("production");
    assert!(serde_json::from_value::<AnalysisRunIdentity>(wire).is_err());
    let mut wire = serde_json::to_value(&identity).unwrap();
    wire["realization"]["components"][0]["spatial_operator"] =
        serde_json::json!("reduced_edge_graph_curl_curl");
    assert!(serde_json::from_value::<AnalysisRunIdentity>(wire).is_err());
}

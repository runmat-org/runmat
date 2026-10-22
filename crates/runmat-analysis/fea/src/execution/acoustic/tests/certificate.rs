use super::*;

// witness: FEA-ACOUSTIC-NUMERICS-001/cold_claim @ acoustic_harmonic/tet4/numerical_kernel;
//   boundary: Retained certificate/receipt provenance and tamper rejection; no cold refactorization.
#[test]
fn retained_certificate_survives_solve_free_cold_lineage_and_tampering_fails() {
    let mesh = slab_mesh(1);
    let workload = workload();
    let result = workload.execute(&mesh).unwrap();
    let certificate = result.certificate();
    assert!(certificate.dimension > 0);
    assert!(certificate.contraction_bound < certificate.limit);
    assert_eq!(certificate.limit, workload.solver.contraction_limit);
    let receipt = AcousticNumericalReceiptArtifact::new(&workload, &result, &mesh).unwrap();
    assert_eq!(receipt.receipt().certificate, certificate);
    let bytes = result.canonical_bytes().unwrap();
    let receipt_bytes = receipt.canonical_bytes().unwrap();
    // Import checks retained lineage, not a reconstructed factorization.
    let cold = AcousticHarmonicSolverResult::from_canonical_bytes(&bytes).unwrap();
    let cold_receipt =
        AcousticNumericalReceiptArtifact::from_canonical_bytes(&receipt_bytes).unwrap();
    assert_eq!(cold, result);
    cold_receipt
        .validate_against(&workload, &cold, &mesh)
        .unwrap();
    for mutation in 0..4 {
        let mut bad = serde_json::to_value(&cold).unwrap();
        match mutation {
            0 => bad["certificate"] = serde_json::Value::Null,
            1 => bad["certificate"]["contraction_bound"] = 0.75.into(),
            2 => bad["certificate"]["limit"] = 0.25.into(),
            3 => bad["certificate"]["dimension"] = (mesh.topology.nodes.len() * 2 + 1).into(),
            _ => unreachable!(),
        }
        assert!(
            serde_json::from_value::<AcousticHarmonicSolverResult>(bad).is_err(),
            "mutation={mutation}"
        );
    }
    // A well-shaped foreign operator identity changes exact result identity and its receipt
    // no longer binds.
    let mut changed = serde_json::to_value(&cold).unwrap();
    changed["certificate"]["operator_identity"] =
        serde_json::to_value(Digest::sha256(b"foreign operator")).unwrap();
    let foreign: AcousticHarmonicSolverResult = serde_json::from_value(changed).unwrap();
    assert_ne!(
        cold.source_result_identity().unwrap(),
        foreign.source_result_identity().unwrap()
    );
    assert!(cold_receipt
        .validate_against(&workload, &foreign, &mesh)
        .is_err());
}

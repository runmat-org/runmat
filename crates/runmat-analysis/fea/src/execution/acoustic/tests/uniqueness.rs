use super::*;

// witness: FEA-ACOUSTIC-NUMERICS-001/cold_claim @ acoustic_harmonic/tet4/numerical_kernel;
//   boundary: Retained producer claim/receipt provenance and tamper rejection; no cold mathematical
//   uniqueness reproof.
#[test]
fn required_diagnostic_survives_solve_free_cold_lineage_and_tampering_fails() {
    let mesh = slab_mesh(1);
    let workload = workload();
    let result = workload.execute(&mesh).unwrap();
    let record = result.uniqueness().unwrap();
    assert!(record.dimension() > 0);
    assert!(record.defect_infinity_upper().unwrap() < 1.0);
    let receipt = AcousticNumericalReceiptArtifact::new(&workload, &result, &mesh).unwrap();
    assert_eq!(receipt.receipt().uniqueness, Some(record));
    let bytes = result.canonical_bytes().unwrap();
    let receipt_bytes = receipt.canonical_bytes().unwrap();
    let token = crate::progress::FeaCancellationToken::new();
    token.cancel();
    let guard =
        crate::progress::replace_fea_progress_context(crate::progress::FeaProgressContext {
            handler: None,
            cancellation: token,
            attempt_id: None,
        });
    // Import checks retained lineage, not reconstructed mathematical proof.
    let cold = AcousticHarmonicSolverResult::from_canonical_bytes(&bytes).unwrap();
    let cold_receipt =
        AcousticNumericalReceiptArtifact::from_canonical_bytes(&receipt_bytes).unwrap();
    assert_eq!(cold, result);
    // Decoding retained evidence needs neither geometry nor a solve. Mesh-grounded
    // provenance validation constructs the original projection and must honor cancellation.
    assert!(cold_receipt
        .validate_against(&workload, &cold, &mesh)
        .is_err());
    assert!(workload.execute(&mesh).is_err());
    drop(guard);
    cold_receipt
        .validate_against(&workload, &cold, &mesh)
        .unwrap();
    for mutation in 0..5 {
        let mut bad = serde_json::to_value(&cold).unwrap();
        match mutation {
            0 => bad["uniqueness"] = serde_json::Value::Null,
            1 => {
                bad["uniqueness"]["claim"]["Verified"]["defect_infinity_upper_bits"] =
                    1.0_f64.to_bits().into()
            }
            2 => bad["uniqueness"]["policy"]["candidate_relative_tolerance"] = 0.5.into(),
            3 => bad["uniqueness"]["dimension"] = (mesh.topology.nodes.len() * 2 + 1).into(),
            4 => bad["solver"]["uniqueness"] = serde_json::Value::Null,
            _ => unreachable!(),
        }
        assert!(
            serde_json::from_value::<AcousticHarmonicSolverResult>(bad).is_err(),
            "mutation={mutation}"
        );
    }
    // Changing a well-shaped claim changes exact result identity. With columns
    // discarded, cold import cannot independently disprove a fabricated hash.
    let mut changed = serde_json::to_value(&cold).unwrap();
    changed["uniqueness"]["operator_identity"] =
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

#[test]
fn explicit_uniqueness_controls_change_workload_identity_without_changing_physical_targets() {
    let workload = workload();
    let mut other = workload.clone();
    other
        .solver
        .uniqueness
        .as_mut()
        .unwrap()
        .maximum_iterations_per_column += 1;
    assert_ne!(workload.identity().unwrap(), other.identity().unwrap());
    assert_eq!(
        workload.solver.relative_tolerance,
        other.solver.relative_tolerance
    );
    assert_eq!(
        workload.power_balance_relative_tolerance,
        other.power_balance_relative_tolerance
    );
    assert_eq!(
        workload.solver.maximum_iterations,
        other.solver.maximum_iterations
    );
    other.solver.uniqueness.as_mut().unwrap().restart = 0;
    assert!(other.validate().is_err());
    let mut missing = serde_json::to_value(workload.solver).unwrap();
    missing.as_object_mut().unwrap().remove("uniqueness");
    assert!(serde_json::from_value::<AcousticHarmonicPolicy>(missing).is_err());
}

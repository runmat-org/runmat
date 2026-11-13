use super::*;
use runmat_analysis_core::{AnalysisConstraint, FluidConstraintKind, PrescribedValue, Velocity};

fn moving() -> (FluidSteadyWorkload, SolverMeshArtifact) {
    let (mut w, m) = super::super::tests::authored();
    w.policy.maximum_outer_iterations = 200;
    for c in &mut w.model.constraints {
        if let AnalysisConstraint::Fluid(c) = c {
            match &mut c.kind {
                FluidConstraintKind::NoSlipWall { target } => {
                    c.kind = FluidConstraintKind::MovingWall {
                        target: target.clone(),
                        velocity: PrescribedValue::Constant([
                            Velocity::new(1.0),
                            Velocity::new(0.0),
                            Velocity::new(0.0),
                        ]),
                    };
                }
                FluidConstraintKind::PrescribedVelocity { value, .. } => {
                    *value = PrescribedValue::Constant([
                        Velocity::new(1.0),
                        Velocity::new(0.0),
                        Velocity::new(0.0),
                    ]);
                }
                _ => unreachable!(),
            }
        }
    }
    (w, m)
}

#[test]
fn actual_nonzero_solve_cold_receipt_preserves_original_authority_and_schema_bound() {
    let (w, m) = moving();
    let result = w.execute(&m).unwrap();
    assert!(result
        .original()
        .velocity_m_per_s
        .canonical_values()
        .iter()
        .any(|u| u[0] > 0.9));
    assert!(result
        .original()
        .face_volume_flux_m3_per_s
        .canonical_values()
        .iter()
        .any(|q| q.abs() > 0.0));
    let original_bytes = result.canonical_bytes().unwrap();
    let receipt = FluidNumericalReceiptArtifact::new(&w, &result, &m).unwrap();
    let bytes = receipt.canonical_bytes().unwrap();
    // The projection reserves the widest scalar for every field: an upper bound.
    let actual = runmat_canonical_codec::profile_serialized_shape(&receipt, LIMITS).unwrap();
    let bound = fluid_receipt_encoding_profile().unwrap();
    assert!(actual.encoded_bytes_upper_bound <= bound.encoded_bytes_upper_bound);
    assert!(
        actual.materialization_resident_bytes_estimate
            <= bound.materialization_resident_bytes_estimate
    );
    assert!(bytes.len() as u64 <= FLUID_RECEIPT_MAX_ENCODED_BYTES);
    // The wire ceiling planners admit covers the projection and the actual receipt.
    assert!(bound.encoded_bytes_upper_bound <= FLUID_RECEIPT_MAX_ENCODED_BYTES);
    assert!(
        receipt
            .encoding_profile()
            .unwrap()
            .encoded_bytes_upper_bound
            <= FLUID_RECEIPT_MAX_ENCODED_BYTES
    );
    let cold = FluidNumericalReceiptArtifact::from_canonical_bytes_for(
        &bytes,
        Digest::sha256(&bytes),
        &w,
        &result,
        &m,
    )
    .unwrap();
    assert_eq!(cold, receipt);
    assert_eq!(result.canonical_bytes().unwrap(), original_bytes);
    assert_eq!(
        receipt.receipt().linear_iterations,
        result.original().linear_iterations
    );
    assert_eq!(receipt.receipt().reservation, result.reservation());
    assert_eq!(
        receipt.receipt().faces,
        result
            .original()
            .face_volume_flux_m3_per_s
            .canonical_values()
            .len() as u64
    );
    assert!(FluidNumericalReceiptArtifact::from_canonical_bytes_for(
        &bytes,
        Digest::sha256(b"wrong"),
        &w,
        &result,
        &m
    )
    .is_err());
}

#[test]
fn self_consistent_receipt_substitutions_never_gain_original_source_authority() {
    let (w, m) = moving();
    let result = w.execute(&m).unwrap();
    let receipt = FluidNumericalReceiptArtifact::new(&w, &result, &m).unwrap();
    let mut substituted = receipt.clone();
    substituted.receipt.global_boundary_flux_m3_per_s = substituted
        .receipt
        .global_boundary_flux_m3_per_s
        .clone()
        .try_map_canonical_storage(|value| value + (1.0))
        .unwrap();
    substituted.identity = receipt_identity(
        substituted.source_workload,
        substituted.source_result,
        &substituted.receipt,
    )
    .unwrap();
    let bytes = substituted.canonical_bytes().unwrap();
    // Wire self-hash is deliberately not a scientific gate.
    FluidNumericalReceiptArtifact::from_canonical_bytes(&bytes).unwrap();
    assert!(substituted.validate_against(&w, &result, &m).is_err());
    assert!(FluidNumericalReceiptArtifact::from_canonical_bytes_for(
        &bytes,
        Digest::sha256(&bytes),
        &w,
        &result,
        &m
    )
    .is_err());
    let mut changed = w.clone();
    changed.policy.maximum_outer_iterations += 1;
    assert!(receipt.validate_against(&changed, &result, &m).is_err());
    let mut changed_mesh = m.clone();
    {
        let quantity_change = 0.01;
        let quantity_index = 0;
        let quantity_slot = &mut changed_mesh.topology.nodes[0].coordinates_m;
        let mut quantity_coordinates = quantity_slot.canonical_value();
        quantity_coordinates[quantity_index] += quantity_change;
        *quantity_slot = runmat_types::physical::Quantity::<
            runmat_types::physical::kind::Position,
            _,
        >::new(quantity_coordinates);
    };
    changed_mesh.seal_canonical_digest().unwrap();
    assert!(receipt
        .validate_against(&w, &result, &changed_mesh)
        .is_err());
}

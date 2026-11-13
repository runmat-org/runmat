use super::*;
use crate::{
    fixtures::canonical_tetrahedron_solver_mesh, ElementOrder, SolverMeshValidation,
    SOLVER_MESH_PROJECTION_SCHEMA_VERSION,
};
use std::sync::atomic::{AtomicUsize, Ordering};

struct Probe {
    calls: AtomicUsize,
    cancel_after: usize,
}
impl MeshingCancellationSignal for Probe {
    fn is_cancelled(&self) -> bool {
        self.calls.fetch_add(1, Ordering::Relaxed) >= self.cancel_after
    }
}
fn projection(order: ElementOrder) -> SolverMeshProjection {
    let artifact = canonical_tetrahedron_solver_mesh(order);
    SolverMeshProjection {
        schema_version: SOLVER_MESH_PROJECTION_SCHEMA_VERSION,
        geometry: artifact.geometry,
        resolved_request: artifact.resolved_request,
        topology: artifact.topology.into_topology(),
    }
}

#[test]
fn cooperative_solver_projection_preserves_bytes_receipt_and_sealed_artifact() {
    for order in [ElementOrder::Tet4, ElementOrder::Tet10] {
        let projection = projection(order);
        let probe = Probe {
            calls: AtomicUsize::new(0),
            cancel_after: usize::MAX,
        };
        let bytes = projection.canonical_encode().unwrap();
        assert_eq!(
            projection
                .canonical_encode_with_cancellation(&probe)
                .unwrap(),
            bytes
        );
        assert_eq!(
            SolverMeshProjection::canonical_decode_with_cancellation(&bytes, &probe).unwrap(),
            projection
        );
        assert_eq!(
            projection
                .canonical_digest_with_cancellation(&probe)
                .unwrap(),
            projection.canonical_digest().unwrap()
        );
        let receipt =
            SolverMeshValidation::from_validated_projection_with_cancellation(&projection, &probe)
                .unwrap();
        assert_eq!(
            receipt,
            SolverMeshValidation::from_validated_projection(&projection).unwrap()
        );
        receipt
            .validate_against_with_cancellation(&projection, &probe)
            .unwrap();
        let digest = StableDigest::from_bytes([71; 32]);
        let artifact = projection
            .clone()
            .into_artifact_with_cancellation(digest, &probe)
            .unwrap();
        assert_eq!(artifact, projection.into_artifact(digest).unwrap());
        assert_eq!(
            artifact.canonical_encode_with_cancellation(&probe).unwrap(),
            artifact.canonical_encode().unwrap()
        );
        assert!(probe.calls.load(Ordering::Relaxed) > 20);
    }
}

#[test]
fn solver_terminal_operations_observe_mid_operation_cancellation_and_tampering() {
    for order in [ElementOrder::Tet4, ElementOrder::Tet10] {
        let projection = projection(order);
        let bytes = projection.canonical_encode().unwrap();
        for operation in 0..6 {
            let probe = Probe {
                calls: AtomicUsize::new(0),
                cancel_after: 3,
            };
            let digest = StableDigest::from_bytes([71; 32]);
            let result = match operation {
                0 => projection
                    .canonical_encode_with_cancellation(&probe)
                    .map(|_| ()),
                1 => SolverMeshProjection::canonical_decode_with_cancellation(&bytes, &probe)
                    .map(|_| ()),
                2 => projection
                    .canonical_digest_with_cancellation(&probe)
                    .map(|_| ()),
                3 => SolverMeshValidation::from_validated_projection_with_cancellation(
                    &projection,
                    &probe,
                )
                .map(|_| ()),
                4 => projection
                    .clone()
                    .into_artifact_with_cancellation(digest, &probe)
                    .map(|_| ()),
                _ => projection
                    .clone()
                    .into_artifact(digest)
                    .unwrap()
                    .canonical_encode_with_cancellation(&probe)
                    .map(|_| ()),
            };
            assert!(result.is_err(), "operation {operation} must cancel");
        }
        let receipt = SolverMeshValidation::from_validated_projection(&projection).unwrap();
        let mut tampered = projection;
        {
            let quantity_change = f64::NAN;
            let quantity_index = 0;
            let quantity_slot = &mut tampered.topology.nodes[0].coordinates_m;
            let mut quantity_coordinates = quantity_slot.canonical_value();
            quantity_coordinates[quantity_index] = quantity_change;
            *quantity_slot = runmat_types::physical::Quantity::<
                runmat_types::physical::kind::Position,
                _,
            >::new(quantity_coordinates);
        };
        assert!(tampered
            .validate_with_cancellation(&NeverCancelled)
            .is_err());
        assert!(receipt
            .validate_against_with_cancellation(&tampered, &NeverCancelled)
            .is_err());
    }
}

#[test]
fn fresh_authority_preserves_wire_and_keeps_raw_and_cold_checks_independent() {
    for order in [ElementOrder::Tet4, ElementOrder::Tet10] {
        let raw = projection(order);
        let bytes = raw.canonical_encode().unwrap();
        let fresh = crate::ValidatedSolverMeshTopology::new(
            raw.topology.clone(),
            raw.resolved_request.clone(),
            &NeverCancelled,
        )
        .unwrap()
        .into_projection(raw.geometry.clone())
        .unwrap();
        assert_eq!(fresh.encode(&NeverCancelled).unwrap(), bytes);
        assert_eq!(SolverMeshProjection::canonical_decode(&bytes).unwrap(), raw);
        let mut invalid = fresh.into_projection();
        {
            let quantity_change = f64::NAN;
            let quantity_index = 0;
            let quantity_slot = &mut invalid.topology.nodes[0].coordinates_m;
            let mut quantity_coordinates = quantity_slot.canonical_value();
            quantity_coordinates[quantity_index] = quantity_change;
            *quantity_slot = runmat_types::physical::Quantity::<
                runmat_types::physical::kind::Position,
                _,
            >::new(quantity_coordinates);
        };
        assert!(invalid.canonical_encode().is_err());
        assert!(crate::ValidatedSolverMeshTopology::new(
            invalid.topology,
            invalid.resolved_request,
            &NeverCancelled
        )
        .is_err());
        let fresh = crate::ValidatedSolverMeshTopology::new(
            raw.topology,
            raw.resolved_request,
            &NeverCancelled,
        )
        .unwrap()
        .into_projection(raw.geometry)
        .unwrap();
        let cancelled = Probe {
            calls: AtomicUsize::new(0),
            cancel_after: 0,
        };
        assert!(fresh.encode(&cancelled).is_err());
    }
}

// witness: FEA-SHARED-RESOURCES-001/phase_lifetimes @
//   shared_infrastructure/canonical_mixed_topology/numerical_kernel;
//   boundary: Both element orders retain original cold projection values, separate typed
//   materialization and topology scratch admission, preserve the exact denied callback, and reject
//   canonical but forged topology. This witness does not establish whole-pipeline storage or model
//   performance.
#[test]
fn cold_projection_admission_precedes_materialization_and_independent_validation() {
    for order in [ElementOrder::Tet4, ElementOrder::Tet10] {
        let original = projection(order);
        let bytes = original.canonical_encode().unwrap();
        let control = crate::MeshingCanonicalCodecControl::new(&NeverCancelled);
        let mut inventory = Vec::new();
        let decoded = SolverMeshProjection::canonical_decode_with_resource_admission(
            &bytes,
            &control,
            &NeverCancelled,
            |bytes| {
                inventory.push(bytes);
                Ok(())
            },
        )
        .unwrap();
        assert_eq!(decoded, original);
        assert_eq!(
            inventory.len(),
            2,
            "typed decode and topology scratch have distinct admission boundaries"
        );
        assert!(
            inventory[0]
                >= original
                    .retained_storage_bytes(u64::MAX, &mut || Ok(()))
                    .unwrap()
        );
        assert_eq!(
            inventory[1],
            original.topology.validation_scratch_bytes(order).unwrap()
        );
        for deny_at in [1, 2] {
            let mut calls = 0;
            let failure = SolverMeshProjection::canonical_decode_with_resource_admission(
                &bytes,
                &control,
                &NeverCancelled,
                |_| {
                    calls += 1;
                    if calls == deny_at {
                        Err(MeshingContractError::invalid("fixture admission", "denied"))
                    } else {
                        Ok(())
                    }
                },
            )
            .unwrap_err();
            assert_eq!(failure.field, "fixture admission");
            assert_eq!(calls, deny_at);
        }
        let mut changed = original;
        changed.topology.nodes[0].node_id = changed.topology.nodes[1].node_id;
        let hostile = super::super::encode_contract(
            SolverMeshProjection::DOMAIN,
            &changed,
            SolverMeshProjection::LIMITS,
        )
        .unwrap();
        assert!(
            SolverMeshProjection::canonical_decode_with_resource_admission(
                &hostile,
                &control,
                &NeverCancelled,
                |_| Ok(())
            )
            .is_err(),
            "admission must not grant semantic authority to forged topology"
        );
    }
}

#[test]
fn sealing_inventory_profiles_the_exact_borrowed_identity_without_copying_topology() {
    for order in [ElementOrder::Tet4, ElementOrder::Tet10] {
        let projection = projection(order);
        let control = crate::MeshingCanonicalCodecControl::new(&NeverCancelled);
        let requested = projection
            .artifact_sealing_workspace_bytes(&control)
            .unwrap();
        let mut identity = projection
            .clone()
            .into_artifact(StableDigest::from_bytes([71; 32]))
            .unwrap();
        identity.canonical_digest = StableDigest::ZERO;
        identity.root_stage_manifest_digest = StableDigest::ZERO;
        let expected = runmat_canonical_codec::profile_canonical_encoding_with_control(
            &identity,
            <crate::SolverMeshArtifact as CanonicalMeshingContract>::LIMITS.into(),
            &control,
        )
        .unwrap()
        .resident_workspace_bytes_estimate
        .max(identity.topology.validation_scratch_bytes(order).unwrap());
        assert_eq!(requested, expected);
        let cancelled = Probe {
            calls: AtomicUsize::new(0),
            cancel_after: 3,
        };
        let control = crate::MeshingCanonicalCodecControl::new(&cancelled);
        assert!(projection
            .artifact_sealing_workspace_bytes(&control)
            .is_err());
    }
}

#[test]
fn run_data_limits_are_platform_bounded_and_control_documents_stay_small() {
    let artifact = SolverMeshProjection::LIMITS;
    assert_eq!(
        artifact.maximum_encoded_bytes,
        runmat_canonical_codec::PLATFORM_MAXIMUM_ENCODED_BYTES
    );
    // Every collection item takes at least one byte, so the byte ceiling bounds collections.
    assert_eq!(
        artifact.maximum_collection_items,
        artifact.maximum_encoded_bytes
    );
    assert_eq!(
        <crate::SolverMeshArtifact as CanonicalMeshingContract>::LIMITS,
        artifact
    );
    #[cfg(target_pointer_width = "64")]
    assert!(artifact.maximum_encoded_bytes as u64 > 2 * 1024 * 1024 * 1024);
    #[cfg(target_pointer_width = "32")]
    assert_eq!(artifact.maximum_encoded_bytes, i32::MAX as usize);
    assert_eq!(
        MeshingCanonicalLimits::MANIFEST.maximum_encoded_bytes,
        64 * 1024 * 1024
    );
    assert_eq!(
        MeshingCanonicalLimits::REQUEST.maximum_encoded_bytes,
        32 * 1024 * 1024
    );
    assert_eq!(
        MeshingCanonicalLimits::IDENTITY.maximum_encoded_bytes,
        4 * 1024 * 1024
    );
}

/// A budget the run admitted: each allocation request draws from what remains, and a refusal
/// names the requirement and the budget.
fn budget(remaining: &mut u64) -> impl FnMut(u64) -> Result<(), MeshingContractError> + '_ {
    move |requested| {
        if requested > *remaining {
            return Err(MeshingContractError::invalid(
                "run budget",
                format!("requires {requested} bytes, {} bytes admitted", *remaining),
            ));
        }
        *remaining -= requested;
        Ok(())
    }
}

#[test]
fn run_budget_admits_the_artifact_when_it_fits_and_refuses_it_with_both_sizes_when_not() {
    let original = projection(ElementOrder::Tet10);
    let bytes = original.canonical_encode().unwrap();
    let control = crate::MeshingCanonicalCodecControl::new(&NeverCancelled);
    let mut requirement = 0_u64;
    SolverMeshProjection::canonical_decode_with_resource_admission(
        &bytes,
        &control,
        &NeverCancelled,
        |requested| {
            requirement += requested;
            Ok(())
        },
    )
    .unwrap();
    let mut remaining = requirement;
    let decoded = SolverMeshProjection::canonical_decode_with_resource_admission(
        &bytes,
        &control,
        &NeverCancelled,
        budget(&mut remaining),
    )
    .unwrap();
    assert_eq!(decoded, original);
    assert_eq!(remaining, 0);

    let mut remaining = requirement - 1;
    let refused = SolverMeshProjection::canonical_decode_with_resource_admission(
        &bytes,
        &control,
        &NeverCancelled,
        budget(&mut remaining),
    )
    .unwrap_err();
    assert_eq!(refused.field, "run budget");
    assert!(
        refused.reason.contains("bytes admitted"),
        "{}",
        refused.reason
    );
}

#[test]
fn platform_bounded_run_data_still_refuses_malformed_and_noncanonical_input() {
    let bytes = projection(ElementOrder::Tet4).canonical_encode().unwrap();
    let control = crate::MeshingCanonicalCodecControl::new(&NeverCancelled);
    let mut trailing = bytes.clone();
    trailing.push(0);
    let truncated = &bytes[..bytes.len() - 1];
    let mut wrong_domain = bytes.clone();
    let at = wrong_domain.len() / 2;
    wrong_domain[at] ^= 0xff;
    for hostile in [trailing.as_slice(), truncated, wrong_domain.as_slice()] {
        assert!(SolverMeshProjection::canonical_decode(hostile).is_err());
        assert!(
            SolverMeshProjection::canonical_decode_with_resource_admission(
                hostile,
                &control,
                &NeverCancelled,
                |_| Ok(()),
            )
            .is_err()
        );
    }
}

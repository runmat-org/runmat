use super::super::{tests::preparation, ThermalStudyCoordinator, ThermalStudyPhase};
use super::*;

mod retained;

fn roundtrip(wire: &PreparationWire) -> PreparationWire {
    let bytes = encode_contract(
        PREFIX,
        "thermal-preparation-test",
        wire,
        completion_codec_limits(ObjectInventoryLimits::default()),
    )
    .unwrap();
    decode_contract(
        PREFIX,
        "thermal-preparation-test",
        &bytes,
        completion_codec_limits(ObjectInventoryLimits::default()),
    )
    .unwrap()
}

#[test]
fn preparation_codec_preserves_authored_and_execution_inputs() {
    let mut input = preparation();
    input.maximum_resident_bytes = Some(128 * 1024 * 1024);
    let access = input.artifact_access.clone();
    let compiled = input.product.compiled.clone();
    let workload = input.product.workload.clone();
    let revision = input.program_revision.clone();
    let expected_digest = compiled.definition_digest().unwrap();
    let expected_workload = workload.identity().unwrap();
    let wire = PreparationWire::new(input);
    let decoded = roundtrip(&wire).into_preparation(access.clone()).unwrap();
    assert_eq!(decoded.maximum_resident_bytes, Some(128 * 1024 * 1024));
    assert_eq!(decoded.product.compiled, compiled);
    assert_eq!(decoded.product.workload, workload);
    assert_eq!(
        decoded.product.workload.identity().unwrap(),
        expected_workload
    );
    assert_eq!(decoded.program_revision, revision);
    assert_eq!(decoded.artifact_access, access);
    assert_eq!(decoded.output_policy.chunk_bytes(), wire.chunk_bytes);
    assert_eq!(
        decoded.output_policy.inventory_limits().max_objects,
        wire.max_objects
    );
    assert_eq!(
        decoded.output_policy.inventory_limits().max_object_bytes,
        wire.max_object_bytes
    );
    assert_eq!(
        decoded.output_policy.inventory_limits().max_total_bytes,
        wire.max_total_bytes
    );
    let coordinator = ThermalStudyCoordinator::new(decoded).unwrap();
    assert_eq!(coordinator.phase(), ThermalStudyPhase::ReadyForSolve);
    assert_eq!(
        coordinator.compiled_study().definition_digest().unwrap(),
        expected_digest
    );
}

#[test]
fn preparation_codec_preserves_selected_scope_and_full_model() {
    let input = super::super::tests::selected_preparation();
    let access = input.artifact_access.clone();
    let compiled = input.product.compiled.clone();
    let wire = PreparationWire::new(input);
    let decoded = roundtrip(&wire).into_preparation(access.clone()).unwrap();
    assert_eq!(decoded.scope, ThermalStudyScope::SelectedStep);
    assert_eq!(decoded.product.compiled, compiled);
    let coordinator = ThermalStudyCoordinator::new(decoded).unwrap();
    assert_eq!(coordinator.selected_step().step_id.0, "heat");
    let mut wrong_scope = roundtrip(&wire);
    wrong_scope.scope = ThermalStudyScope::Standalone;
    assert!(ThermalStudyCoordinator::new(wrong_scope.into_preparation(access).unwrap()).is_err());
}

#[test]
fn preparation_wire_rejects_invalid_or_silently_clamped_policy() {
    for chunk_bytes in [0, usize::MAX] {
        let input = preparation();
        let access = input.artifact_access.clone();
        let mut wire = PreparationWire::new(input);
        wire.chunk_bytes = chunk_bytes;
        assert!(roundtrip(&wire).into_preparation(access).is_err());
    }
    for zero_limit in 0..3 {
        let input = preparation();
        let access = input.artifact_access.clone();
        let mut wire = PreparationWire::new(input);
        match zero_limit {
            0 => wire.max_objects = 0,
            1 => wire.max_object_bytes = 0,
            _ => wire.max_total_bytes = 0,
        }
        assert!(roundtrip(&wire).into_preparation(access).is_err());
    }
}

#[test]
fn decoded_preparation_cannot_authorize_forged_numerical_inputs() {
    for forgery in 0..3 {
        let input = preparation();
        let access = input.artifact_access.clone();
        let mut wire = PreparationWire::new(input);
        match forgery {
            0 => wire.compiled.model.geometry_revision += 1,
            1 => wire.workload.residual_target *= 0.5,
            _ => wire.mesh.canonical_digest = runmat_meshing_core::StableDigest::ZERO,
        }
        let decoded = roundtrip(&wire).into_preparation(access).unwrap();
        assert!(ThermalStudyCoordinator::new(decoded).is_err());
    }
}

fn references() -> (ArtifactAccess, Vec<ValueRef>) {
    let access = preparation().artifact_access;
    let references = [b"first".as_slice(), b"second".as_slice()]
        .into_iter()
        .enumerate()
        .map(|(ordinal, bytes)| {
            let object = LogicalObject::new(
                ObjectNamespace::ResultValue,
                format!("thermal-test/{ordinal}"),
                "application/vnd.runmat.test",
                bytes.to_vec(),
            )
            .unwrap();
            access
                .reference(&object, ValueRefKind::ResultObject, "runmat.test.v1")
                .unwrap()
        })
        .collect();
    (access, references)
}

#[test]
fn inventory_deduplicates_equal_references_but_rejects_conflicts() {
    let (_, references) = references();
    assert_eq!(
        unique(vec![references[0].clone(), references[0].clone()]).unwrap(),
        vec![references[0].clone()]
    );
    let mut conflicting = references[0].clone();
    conflicting.encoded_length += 1;
    assert!(unique(vec![references[0].clone(), conflicting]).is_err());
}

#[test]
fn inventory_enforces_access_and_caller_resource_limits() {
    let (access, references) = references();
    let total = references
        .iter()
        .map(|value| value.encoded_length)
        .sum::<u64>();
    let maximum = references
        .iter()
        .map(|value| value.encoded_length)
        .max()
        .unwrap();
    let exact = ObjectInventoryLimits {
        max_objects: 2,
        max_object_bytes: maximum,
        max_total_bytes: total,
    };
    check_inventory(&references, &access, exact).unwrap();
    for limits in [
        ObjectInventoryLimits {
            max_objects: 1,
            ..exact
        },
        ObjectInventoryLimits {
            max_object_bytes: maximum - 1,
            ..exact
        },
        ObjectInventoryLimits {
            max_total_bytes: total - 1,
            ..exact
        },
    ] {
        assert!(check_inventory(&references, &access, limits).is_err());
    }
    let mut foreign = access.clone();
    foreign.authorization_scope.push_str("-foreign");
    assert!(check_inventory(&references, &foreign, exact).is_err());
    let mut foreign = access;
    foreign.encryption_context = Digest::sha256("foreign-key");
    assert!(check_inventory(&references, &foreign, exact).is_err());
}

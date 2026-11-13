use super::super::{tests::preparation, AcousticStudyCoordinator, AcousticStudyPhase};
use super::*;

fn roundtrip(wire: &PreparationWire) -> PreparationWire {
    let bytes = encode_contract(
        PREFIX,
        "acoustic-preparation-test",
        wire,
        completion_codec_limits(ObjectInventoryLimits::default()),
    )
    .unwrap();
    decode_contract(
        PREFIX,
        "acoustic-preparation-test",
        &bytes,
        completion_codec_limits(ObjectInventoryLimits::default()),
    )
    .unwrap()
}

#[test]
fn preparation_codec_preserves_authored_and_execution_inputs() {
    let input = preparation();
    let access = input.artifact_access.clone();
    let compiled = input.product.compiled.clone();
    let workload = input.frequency_workload().unwrap();
    let revision = input.program_revision.clone();
    let expected_digest = compiled.definition_digest().unwrap();
    let expected_workload = workload.identity().unwrap();
    let wire = PreparationWire::new(input).unwrap();
    let decoded = roundtrip(&wire).into_preparation(access.clone()).unwrap();
    assert_eq!(decoded.product.compiled, compiled);
    assert_eq!(decoded.frequency_workload().unwrap(), workload);
    assert_eq!(
        decoded.frequency_workload().unwrap().identity().unwrap(),
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
    let coordinator = AcousticStudyCoordinator::new(decoded).unwrap();
    assert_eq!(
        coordinator.entries()[0].phase(),
        AcousticStudyPhase::ReadyForSolve
    );
    assert_eq!(
        coordinator.compiled_study().definition_digest().unwrap(),
        expected_digest
    );
}

#[test]
fn preparation_codec_retains_entire_multi_frequency_plan() {
    let mut input = preparation();
    input.product.compiled.model.sequence.steps[0].independent_domain =
        runmat_analysis_core::IndependentDomain::FrequencyHertz(vec![0.1, 0.2, 0.3]);
    input.product.plan = plan_acoustic_study(&input.product.compiled, input.policy).unwrap();
    let workload = input.frequency_workload().unwrap();
    let access = input.artifact_access.clone();
    let wire = PreparationWire::new(input).unwrap();
    let decoded = roundtrip(&wire).into_preparation(access).unwrap();
    assert_eq!(decoded.frequency_workload().unwrap(), workload);
    assert_eq!(decoded.product.plan.frequencies().len(), 3);
    assert_eq!(
        decoded
            .product
            .plan
            .frequency_plan()
            .reference_frequency_hz(),
        0.1
    );
    let coordinator = AcousticStudyCoordinator::new(decoded).unwrap();
    assert_eq!(coordinator.entries().len(), 3);
}

#[test]
fn preparation_codec_does_not_normalize_invalid_authored_frequency_order() {
    let mut input = preparation();
    input.product.compiled.model.sequence.steps[0].independent_domain =
        runmat_analysis_core::IndependentDomain::FrequencyHertz(vec![0.3, 0.1, 0.2]);
    assert!(plan_acoustic_study(&input.product.compiled, input.policy).is_err());
    assert!(AcousticStudyCoordinator::new(input).is_err());
}

#[test]
fn preparation_wire_rejects_invalid_or_silently_clamped_policy() {
    for chunk_bytes in [0, usize::MAX] {
        let input = preparation();
        let access = input.artifact_access.clone();
        let mut wire = PreparationWire::new(input).unwrap();
        wire.chunk_bytes = chunk_bytes;
        assert!(roundtrip(&wire).into_preparation(access).is_err());
    }
    for zero_limit in 0..3 {
        let input = preparation();
        let access = input.artifact_access.clone();
        let mut wire = PreparationWire::new(input).unwrap();
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
        let mut wire = PreparationWire::new(input).unwrap();
        match forgery {
            0 => wire.compiled.model.geometry_revision += 1,
            1 => wire.policy.contraction_limit *= 0.5,
            _ => wire.mesh.canonical_digest = runmat_meshing_core::StableDigest::ZERO,
        }
        let decoded = roundtrip(&wire).into_preparation(access);
        assert!(decoded.is_err() || AcousticStudyCoordinator::new(decoded.unwrap()).is_err());
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
                format!("acoustic-test/{ordinal}"),
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

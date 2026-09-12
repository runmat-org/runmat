use std::collections::BTreeMap;

use runmat_analysis_fea::execution::{
    StudyIdentity, SweepCompositionPolicy, SweepDefinition, SweepIdentity,
};
use runmat_execution::value::{ValuePayload, ValueRefKind};
use runmat_execution::Digest;
use runmat_execution_artifact::cache::CacheImport;
use runmat_execution_artifact::{LogicalObject, ObjectNamespace, ProgramExecutionResponse};

use super::*;

struct TestCache(BTreeMap<Digest, Vec<u8>>);

impl CacheImport for TestCache {
    fn read_verified(
        &self,
        digest: Digest,
    ) -> runmat_execution_artifact::ArtifactResult<Option<Vec<u8>>> {
        Ok(self.0.get(&digest).cloned())
    }
}

fn definition(first: Digest, second: Digest) -> SweepDefinition {
    SweepDefinition::new(
        Digest::sha256(b"sweep"),
        vec![
            StudyIdentity::from_authored_digest(first, None).unwrap(),
            StudyIdentity::from_authored_digest(second, None).unwrap(),
        ],
        SweepCompositionPolicy::FailFast,
    )
    .unwrap()
}

fn identity(first: Digest, second: Digest) -> SweepIdentity {
    definition(first, second).identity().unwrap()
}

#[test]
fn sweep_root_round_trips_ordered_studies_and_deduplicates_programs() {
    let authority = "sweep-run";
    let source = Digest::sha256(b"portable exact CAD source");
    let first_digest = Digest::sha256(b"study-a");
    let second_digest = Digest::sha256(b"study-b");
    let first =
        crate::exact_structural_study_root::tests::prepared(authority, source, first_digest);
    let second =
        crate::exact_structural_study_root::tests::prepared(authority, source, second_digest);
    let prepared = prepare_exact_structural_sweep_root(ExactStructuralSweepRootPreparation {
        program_revision: crate::exact_structural_study_root::tests::revision(),
        artifact_access: crate::exact_structural_study_root::tests::access(authority),
        definition: definition(first_digest, second_digest),
        studies: vec![first, second],
    })
    .unwrap();
    let expectation = prepared.result_expectation();
    assert_eq!(prepared.required_programs().len(), 5);

    let store = TestCache(
        prepared
            .transport_objects()
            .into_iter()
            .map(|object| (object.reference.logical_digest, object.bytes))
            .collect(),
    );
    let decoded = decode_exact_structural_sweep_root(prepared.request(), &store).unwrap();
    assert_eq!(decoded.identity, identity(first_digest, second_digest));
    assert_eq!(decoded.composition, SweepCompositionPolicy::FailFast);
    assert_eq!(decoded.studies.len(), 2);
    assert_ne!(
        decoded.studies[0].arguments[0],
        decoded.studies[1].arguments[0]
    );

    let (response, result_objects) = package_exact_structural_sweep_result(
        prepared.request(),
        expectation.identity().unwrap(),
        &expectation.artifact_access,
        vec![
            ExactStructuralSweepRemoteOutcome::Failed("solver failed".into()),
            ExactStructuralSweepRemoteOutcome::Cancelled,
        ],
    )
    .unwrap();
    let results = TestCache(
        result_objects
            .into_iter()
            .map(|object| (object.reference.logical_digest, object.bytes))
            .collect(),
    );
    let imported = import_exact_structural_sweep_result(
        &results,
        response,
        &expectation,
        runmat_execution_artifact::object::ObjectInventoryLimits::default(),
    )
    .unwrap();
    assert!(matches!(
        &imported.entries[0].terminal,
        ExactStructuralSweepTerminal::Failed(message) if message == "solver failed"
    ));
    assert!(matches!(
        imported.entries[1].terminal,
        ExactStructuralSweepTerminal::Cancelled
    ));

    let (response, result_objects) = package_exact_structural_sweep_result(
        prepared.request(),
        expectation.identity().unwrap(),
        &expectation.artifact_access,
        vec![
            ExactStructuralSweepRemoteOutcome::Failed("solver failed".into()),
            ExactStructuralSweepRemoteOutcome::Indeterminate("worker ownership lost".into()),
        ],
    )
    .unwrap();
    let results = TestCache(
        result_objects
            .into_iter()
            .map(|object| (object.reference.logical_digest, object.bytes))
            .collect(),
    );
    let imported = import_exact_structural_sweep_result(
        &results,
        response,
        &expectation,
        runmat_execution_artifact::object::ObjectInventoryLimits::default(),
    )
    .unwrap();
    assert!(matches!(
        &imported.entries[1].terminal,
        ExactStructuralSweepTerminal::Indeterminate(message)
            if message == "worker ownership lost"
    ));
}

#[test]
fn sweep_root_rejects_a_child_from_another_artifact_authority() {
    let authority = "sweep-run";
    let source = Digest::sha256(b"portable exact CAD source");
    let first_digest = Digest::sha256(b"study-a");
    let second_digest = Digest::sha256(b"study-b");
    let first =
        crate::exact_structural_study_root::tests::prepared(authority, source, first_digest);
    let second =
        crate::exact_structural_study_root::tests::prepared("another-run", source, second_digest);

    assert!(
        prepare_exact_structural_sweep_root(ExactStructuralSweepRootPreparation {
            program_revision: crate::exact_structural_study_root::tests::revision(),
            artifact_access: crate::exact_structural_study_root::tests::access(authority),
            definition: definition(first_digest, second_digest),
            studies: vec![first, second],
        })
        .is_err()
    );
}

#[test]
fn sweep_root_rejects_a_definition_for_another_authored_order() {
    let authority = "sweep-run";
    let source = Digest::sha256(b"portable exact CAD source");
    let first_digest = Digest::sha256(b"study-a");
    let second_digest = Digest::sha256(b"study-b");
    let first =
        crate::exact_structural_study_root::tests::prepared(authority, source, first_digest);
    let second =
        crate::exact_structural_study_root::tests::prepared(authority, source, second_digest);

    assert!(
        prepare_exact_structural_sweep_root(ExactStructuralSweepRootPreparation {
            program_revision: crate::exact_structural_study_root::tests::revision(),
            artifact_access: crate::exact_structural_study_root::tests::access(authority),
            definition: definition(second_digest, first_digest),
            studies: vec![first, second],
        })
        .is_err()
    );
}

#[test]
fn sweep_catalog_does_not_read_unselected_study_closures() {
    let authority = "sweep-run";
    let source = Digest::sha256(b"portable exact CAD source");
    let first_digest = Digest::sha256(b"study-a");
    let second_digest = Digest::sha256(b"study-b");
    let first =
        crate::exact_structural_study_root::tests::prepared(authority, source, first_digest);
    let second =
        crate::exact_structural_study_root::tests::prepared(authority, source, second_digest);
    let prepared = prepare_exact_structural_sweep_root(ExactStructuralSweepRootPreparation {
        program_revision: crate::exact_structural_study_root::tests::revision(),
        artifact_access: crate::exact_structural_study_root::tests::access(authority),
        definition: definition(first_digest, second_digest),
        studies: vec![first, second],
    })
    .unwrap();
    let expectation = prepared.result_expectation();
    let access = &expectation.artifact_access;
    let child_response = |label: &str| {
        let object = LogicalObject::new(
            ObjectNamespace::ResultValue,
            "analysis/test-structural-study-results",
            "application/vnd.runmat.test-structural-study-result",
            label.as_bytes().to_vec(),
        )
        .unwrap();
        let reference = access
            .reference(
                &object,
                ValueRefKind::ResultObject,
                "runmat.test.study-result",
            )
            .unwrap();
        ProgramExecutionResponse::ExternalizedSuccess {
            outputs: vec![ValuePayload::Object(Box::new(reference.clone()))],
            result_objects: vec![reference],
        }
    };
    let (response, publication_objects) = package_exact_structural_sweep_result(
        prepared.request(),
        expectation.identity().unwrap(),
        access,
        vec![
            ExactStructuralSweepRemoteOutcome::Succeeded(child_response("first")),
            ExactStructuralSweepRemoteOutcome::Succeeded(child_response("second")),
        ],
    )
    .unwrap();
    let publication_only = TestCache(
        publication_objects
            .into_iter()
            .map(|object| (object.reference.logical_digest, object.bytes))
            .collect(),
    );

    let catalog =
        import_exact_structural_sweep_result_catalog(&publication_only, response, &expectation)
            .unwrap();
    assert_eq!(
        catalog.statuses(),
        &[
            ExactStructuralSweepEntryStatus::Succeeded,
            ExactStructuralSweepEntryStatus::Succeeded,
        ]
    );
    assert!(catalog
        .import_entry(
            &publication_only,
            0,
            runmat_execution_artifact::object::ObjectInventoryLimits::default(),
        )
        .is_err());
}

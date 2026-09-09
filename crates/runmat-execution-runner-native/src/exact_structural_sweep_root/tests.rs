use std::collections::BTreeMap;

use runmat_analysis_fea::execution::{StudyIdentity, SweepIdentity};
use runmat_execution::Digest;
use runmat_execution_artifact::cache::CacheImport;

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

fn identity(first: Digest, second: Digest) -> SweepIdentity {
    SweepIdentity::new(
        Digest::sha256(b"sweep"),
        vec![
            StudyIdentity::from_authored_digest(first, Some(Digest::sha256(b"project"))).unwrap(),
            StudyIdentity::from_authored_digest(second, Some(Digest::sha256(b"project"))).unwrap(),
        ],
        Digest::sha256(b"fail-fast"),
    )
    .unwrap()
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
        identity: identity(first_digest, second_digest),
        fail_fast: true,
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
    assert!(decoded.fail_fast);
    assert_eq!(decoded.studies.len(), 2);
    assert_ne!(
        decoded.studies[0].arguments[0],
        decoded.studies[1].arguments[0]
    );

    let (response, result_objects) = package_exact_structural_sweep_result(
        prepared.request(),
        expectation.identity,
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
        expectation.identity,
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
            identity: identity(first_digest, second_digest),
            fail_fast: true,
            studies: vec![first, second],
        })
        .is_err()
    );
}

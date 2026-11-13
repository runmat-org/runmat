//! Only the catalog is copied: cold import must not depend on the producer store.
use runmat_analysis_execution::{
    import_nonlinear_solid_study_root, prepare_nonlinear_solid_study_root, AnalysisOutputPolicy,
    CompletedNonlinearSolidStudy,
};
use runmat_execution::Digest;
use runmat_execution_artifact::{
    cache::{CacheExport, CacheImport, FilesystemObjectStore},
    ArtifactAccess, ArtifactResult, LogicalObject, ObjectNamespace,
};

mod host_workspace;
pub(super) mod runtime_publication;

struct Missing<'a> {
    source: &'a FilesystemObjectStore,
    digest: Digest,
}
impl CacheImport for Missing<'_> {
    fn read_verified(&self, digest: Digest) -> ArtifactResult<Option<Vec<u8>>> {
        if digest == self.digest {
            Ok(None)
        } else {
            self.source.read_verified(digest)
        }
    }
}

pub(super) fn verify_original_root(
    original: &CompletedNonlinearSolidStudy,
    source: &FilesystemObjectStore,
    policy: AnalysisOutputPolicy,
) {
    verify_original_root_with(
        original,
        source,
        policy,
        runtime_publication::history::Reference::Uniaxial,
    );
}

pub(super) fn verify_original_root_with(
    original: &CompletedNonlinearSolidStudy,
    source: &FilesystemObjectStore,
    policy: AnalysisOutputPolicy,
    reference: runtime_publication::history::Reference,
) {
    let prepared = prepare_nonlinear_solid_study_root(original).unwrap();
    let limits = policy.inventory_limits();
    let maximum_resident_bytes = original.workload.model.sequence.steps[0]
        .accuracy
        .resources
        .resident_byte_budget();
    let directory = tempfile::tempdir().unwrap();
    let mut cold =
        FilesystemObjectStore::open(directory.path().join("retained"), limits.max_object_bytes)
            .unwrap();
    for reference in &prepared.catalog.objects {
        let bytes = prepared
            .objects
            .iter()
            .find(|o| o.descriptor.digest == reference.logical_digest)
            .map(|o| o.bytes.clone())
            .unwrap_or_else(|| {
                source
                    .read_verified(reference.logical_digest)
                    .unwrap()
                    .unwrap()
            });
        let object = LogicalObject::new(
            ObjectNamespace::ResultValue,
            "original-j2-cold",
            "application/octet-stream",
            bytes,
        )
        .unwrap();
        assert_eq!(object.descriptor.digest, reference.logical_digest);
        cold.write_verified(&object).unwrap();
    }
    drop(cold);
    let cold =
        FilesystemObjectStore::open(directory.path().join("retained"), limits.max_object_bytes)
            .unwrap();
    let root = &prepared.root;
    let access = ArtifactAccess {
        authorization_scope: root.authorization_scope.clone(),
        encryption_context: root.encryption_context,
    };
    assert!(import_nonlinear_solid_study_root(
        &cold,
        root,
        root.logical_digest,
        &access,
        limits,
        1024
    )
    .is_err());
    assert!(import_nonlinear_solid_study_root(
        &cold,
        root,
        Digest::sha256(b"foreign-root"),
        &access,
        limits,
        maximum_resident_bytes
    )
    .is_err());
    let missing = prepared
        .catalog
        .objects
        .iter()
        .find(|o| o.logical_digest != root.logical_digest)
        .unwrap();
    assert!(import_nonlinear_solid_study_root(
        &Missing {
            source: &cold,
            digest: missing.logical_digest
        },
        root,
        root.logical_digest,
        &access,
        limits,
        maximum_resident_bytes
    )
    .is_err());
    let mut bounded = limits;
    bounded.max_objects = 1;
    assert!(import_nonlinear_solid_study_root(
        &cold,
        root,
        root.logical_digest,
        &access,
        bounded,
        maximum_resident_bytes
    )
    .is_err());
    let mut foreign = access.clone();
    foreign.authorization_scope.push_str("-foreign");
    assert!(import_nonlinear_solid_study_root(
        &cold,
        root,
        root.logical_digest,
        &foreign,
        limits,
        maximum_resident_bytes
    )
    .is_err());
    let reopened = import_nonlinear_solid_study_root(
        &cold,
        root,
        root.logical_digest,
        &access,
        limits,
        maximum_resident_bytes,
    )
    .unwrap();
    assert_eq!(reopened.compiled, original.compiled);
    assert_eq!(reopened.workload, original.workload);
    assert_eq!(reopened.terminal_block.block, original.terminal_block.block);
    assert_eq!(reopened.states, original.states);
    assert_eq!(reopened.accuracy, original.accuracy);
    assert_eq!(reopened.publication, original.publication);
    assert_eq!(
        reopened.baseline_usage.evidence,
        original.baseline_usage.evidence
    );
    assert_eq!(
        reopened.terminal_usage.evidence,
        original.terminal_usage.evidence
    );
    assert_eq!(reopened.result_closures(), original.result_closures());
    host_workspace::verify(
        &directory.path().join("retained"),
        &prepared,
        policy,
        maximum_resident_bytes,
    );
    let prefix = reopened.import_original_block(&cold, 0).unwrap();
    let terminal = reopened.import_original_block(&cold, 1).unwrap();
    let expected_factors = prefix
        .block
        .result
        .load_factors
        .iter()
        .chain(&terminal.block.result.load_factors)
        .copied()
        .collect::<Vec<_>>();
    assert_eq!(
        expected_factors.len(),
        original
            .terminal_block
            .block
            .checkpoint
            .work()
            .accepted_increments()
    );
    runtime_publication::verify(
        &directory.path().join("retained"),
        &prepared,
        original.mesh(),
        policy,
        maximum_resident_bytes,
        &expected_factors,
        reference,
    );
    assert!(!prefix.block.complete);
    assert!((prefix.block.result.accepted_state.load_factor() - 0.9).abs() < 1e-14);
    assert_eq!(prefix.block.result.load_factors.len(), 9);
    match reference {
        runtime_publication::history::Reference::Uniaxial => {
            let alpha = (5.6 * 0.9 - 5.0) / 30.0;
            assert!(prefix
                .block
                .result
                .accepted_state
                .point_history()
                .all(|(_, _, point)| (point.equivalent_plastic_strain - alpha).abs() < 2e-9));
        }
        runtime_publication::history::Reference::Layered => {
            super::layered::reference::verify(original.mesh(), &prefix.block, 0.9)
        }
    }
    assert_eq!(terminal.block, original.terminal_block.block);
    assert!(reopened.import_original_block(&cold, 2).is_err());
    super::durable_tamper::verify_rehashed_metadata(
        &cold,
        &prepared,
        &access,
        policy,
        maximum_resident_bytes,
    );
}

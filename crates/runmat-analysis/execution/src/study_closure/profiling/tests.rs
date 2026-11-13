use super::*;
use runmat_execution_artifact::{
    cache::{CacheExport, FilesystemObjectStore},
    ObjectNamespace,
};
use std::sync::atomic::{AtomicUsize, Ordering};

struct Counted {
    store: FilesystemObjectStore,
    reads: AtomicUsize,
}
impl CacheImport for Counted {
    fn read_verified(&self, _: Digest) -> ArtifactResult<Option<Vec<u8>>> {
        panic!("preflight must never issue an unbounded read");
    }
    fn read_verified_bounded(
        &self,
        digest: Digest,
        maximum: u64,
    ) -> ArtifactResult<Option<Vec<u8>>> {
        self.reads.fetch_add(1, Ordering::SeqCst);
        self.store.read_verified_bounded(digest, maximum)
    }
}

#[test]
fn actual_shape_fits_without_weakening_projection_and_tiny_scope_reads_nothing() {
    let directory = tempfile::tempdir().unwrap();
    let limits = ObjectInventoryLimits::default();
    let mut store =
        FilesystemObjectStore::open(directory.path().join("store"), limits.max_object_bytes)
            .unwrap();
    let payload = runmat_canonical_codec::encode_contract(
        b"bounded-profile\0",
        "profile/v1",
        &vec![1.25_f64; 4096],
        runmat_canonical_codec::CanonicalLimits::new(8 * 1024 * 1024, 1_000_000, 65536, 64),
    )
    .unwrap();
    let prepared = crate::prepare_analysis_objects(
        runmat_analysis_fea::execution::AnalysisNumericalObjectKind::History,
        Digest::sha256(&payload),
        &payload,
        4096,
        ObjectNamespace::ResultValue,
        limits,
    )
    .unwrap();
    for object in &prepared.objects {
        store.write_verified(object).unwrap();
    }
    let access = ArtifactAccess {
        authorization_scope: "bounded-profile".into(),
        encryption_context: Digest::sha256(b"scope"),
    };
    let references = prepared
        .unique_references(&access, ValueRefKind::ResultObject)
        .unwrap();
    let source = Counted {
        store,
        reads: AtomicUsize::new(0),
    };
    assert!(profile_import_workspace_phases(&source, &[&references], &access, limits, 1).is_err());
    assert_eq!(source.reads.load(Ordering::SeqCst), 0);
    // The actual typed-decode profile never exceeds the projection, and it is exactly the
    // ceiling the phases need: one byte less is rejected.
    let ceiling =
        profile_import_workspace_phases(&source, &[&references], &access, limits, u64::MAX)
            .unwrap();
    assert!(ceiling <= import_workspace(&references).unwrap());
    let actual =
        profile_import_workspace_phases(&source, &[&references], &access, limits, ceiling).unwrap();
    assert_eq!(actual, ceiling);
    assert!(
        profile_import_workspace_phases(&source, &[&references], &access, limits, ceiling - 1)
            .is_err()
    );
    assert!(source.reads.load(Ordering::SeqCst) > 0);
    let mut missing = references.clone();
    let index = missing
        .iter()
        .position(|r| r.media_type == ANALYSIS_CHUNK_MEDIA_TYPE)
        .unwrap();
    missing.remove(index);
    assert!(
        profile_import_workspace_phases(&source, &[&missing], &access, limits, ceiling).is_err()
    );
    assert_eq!(
        profile_import_workspace_phases(&source, &[&references], &access, limits, ceiling).unwrap(),
        actual
    );
}

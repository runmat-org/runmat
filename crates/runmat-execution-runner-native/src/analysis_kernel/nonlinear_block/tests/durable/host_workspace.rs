//! Actual original source must honor host allowance before authenticating/decoding.
use super::*;
use runmat_analysis_execution::PreparedNonlinearSolidStudyRoot;
use runmat_runtime::analysis::storage::AnalysisPublicationError;
use std::sync::{
    atomic::{AtomicUsize, Ordering},
    Arc,
};

struct CountReads {
    store: FilesystemObjectStore,
    reads: Arc<AtomicUsize>,
}
impl CacheImport for CountReads {
    fn read_verified_bounded(
        &self,
        digest: Digest,
        maximum_bytes: u64,
    ) -> ArtifactResult<Option<Vec<u8>>> {
        self.reads.fetch_add(1, Ordering::SeqCst);
        self.store.read_verified_bounded(digest, maximum_bytes)
    }
    fn read_verified(&self, digest: Digest) -> ArtifactResult<Option<Vec<u8>>> {
        self.reads.fetch_add(1, Ordering::SeqCst);
        self.store.read_verified(digest)
    }
}

pub(super) fn verify(
    root: &std::path::Path,
    prepared: &PreparedNonlinearSolidStudyRoot,
    policy: AnalysisOutputPolicy,
    original_ceiling: u64,
) {
    let access = ArtifactAccess {
        authorization_scope: prepared.root.authorization_scope.clone(),
        encryption_context: prepared.root.encryption_context,
    };
    let reads = Arc::new(AtomicUsize::new(0));
    let store = CountReads {
        store: FilesystemObjectStore::open(root, policy.inventory_limits().max_object_bytes)
            .unwrap(),
        reads: reads.clone(),
    };
    let (original, source) =
        crate::analysis_nonlinear::source::nonlinear_solid_study_completion_from_source(
            store,
            prepared.root.clone(),
            prepared.root.logical_digest,
            access,
            policy.inventory_limits(),
            original_ceiling,
        )
        .unwrap();
    let before = reads.load(Ordering::SeqCst);
    assert!(before > 0);
    let retained = original.retained_storage_bytes(original_ceiling).unwrap();
    assert!(matches!(
        original.retained_storage_bytes(retained - 1),
        Err(AnalysisPublicationError::ResourceLimit { .. })
    ));
    let mut separate = original.clone();
    for state in &mut separate.states {
        *state = runmat_analysis_fea::execution::CanonicalSemanticState::from_canonical_bytes(
            &state.canonical_bytes().unwrap(),
        )
        .unwrap();
    }
    assert_eq!(separate, original);
    assert!(separate.retained_storage_bytes(original_ceiling).unwrap() > retained);
    drop(separate);
    for allowance in [0, 1] {
        assert!(matches!(
            source.authenticate(&original, allowance),
            Err(AnalysisPublicationError::ResourceLimit { .. })
        ));
        assert_eq!(reads.load(Ordering::SeqCst), before);
    }
    assert!(matches!(
        source.materialize_block(&original, 0, 0, 1),
        Err(AnalysisPublicationError::ResourceLimit { .. })
    ));
    assert_eq!(reads.load(Ordering::SeqCst), before);
    // Retained earlier output must coexist during authentication, not only recovery.
    assert!(matches!(
        source.materialize_block(&original, 0, original_ceiling, original_ceiling),
        Err(AnalysisPublicationError::ResourceLimit { .. })
    ));
    assert_eq!(reads.load(Ordering::SeqCst), before);
    source.authenticate(&original, original_ceiling).unwrap();
    let prefix = source
        .materialize_block(&original, 0, 0, original_ceiling)
        .unwrap();
    assert_eq!(prefix.load_factors.len(), 9);
    assert!((prefix.accepted_state.load_factor() - 0.9).abs() < 1e-14);
    // Authentication still rejects semantic substitution under sufficient admission.
    let mut substituted = original.clone();
    substituted.block_count += 1;
    assert!(matches!(
        source.authenticate(&substituted, original_ceiling),
        Err(AnalysisPublicationError::Rejected { .. })
    ));
    assert_eq!(original.block_count, 2);
}

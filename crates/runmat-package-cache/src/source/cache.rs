use super::{GitSnapshot, SnapshotBlob};
use crate::{
    AccessRecord, BlobMetadata, CacheBackend, CacheError, CacheObject, CacheState,
    CacheTransaction, ObjectWrite,
};
use runmat_package::GitSourceId;

pub fn cache_git_snapshot(
    expected_revision: u64,
    state: CacheState,
    snapshot: &GitSnapshot,
    now_ms: u64,
) -> Result<CacheTransaction, CacheError> {
    snapshot.validate()?;
    let current = state.clone();
    let mut transaction = CacheTransaction::metadata_only(expected_revision, state.clone());
    for blob in &snapshot.blobs {
        let metadata = BlobMetadata {
            digest: blob.digest.clone(),
            byte_len: blob.bytes.len() as u64,
        };
        transaction
            .next_state
            .objects
            .insert(blob.digest.clone(), CacheObject::Blob(metadata.clone()));
        transaction
            .next_state
            .access
            .entry(blob.digest.clone())
            .and_modify(|access| access.touch(now_ms))
            .or_insert_with(|| AccessRecord::new(now_ms));
        transaction.writes.insert(
            blob.digest.clone(),
            ObjectWrite::new(CacheObject::Blob(metadata), Some(blob.bytes.clone()))?,
        );
    }
    transaction.next_state.objects.insert(
        snapshot.tree.digest.clone(),
        CacheObject::Tree(snapshot.tree.clone()),
    );
    transaction
        .next_state
        .access
        .entry(snapshot.tree.digest.clone())
        .and_modify(|access| access.touch(now_ms))
        .or_insert_with(|| AccessRecord::new(now_ms));
    transaction.validate_transition(&current)?;
    Ok(transaction)
}

pub async fn load_git_snapshot<B: CacheBackend>(
    backend: &B,
    source: GitSourceId,
) -> Result<GitSnapshot, CacheError> {
    let snapshot = backend.snapshot().await?;
    let tree = match snapshot.state.objects.get(&source.tree_digest) {
        Some(CacheObject::Tree(tree)) => tree.clone(),
        Some(_) => {
            return Err(CacheError::InvalidState(format!(
                "{} is not a cached tree",
                source.tree_digest
            )));
        }
        None => return Err(CacheError::Miss(source.tree_digest.clone())),
    };
    let mut blobs = Vec::new();
    for digest in tree.referenced_blobs() {
        let bytes = backend
            .read_object_bytes(&digest)
            .await?
            .ok_or_else(|| CacheError::Miss(digest.clone()))?;
        blobs.push(SnapshotBlob { digest, bytes });
    }
    GitSnapshot::new(source, tree, blobs)
}

use crate::{
    AccessRecord, CacheBackend, CacheError, CacheObject, CacheState, CacheTransaction,
    CommitOutcome, ObjectWrite, SourceIndexMetadata,
};

pub async fn publish_source_inventory<B: CacheBackend>(
    backend: &B,
    inventory: &SourceInventory,
    now_ms: u64,
    retries: usize,
) -> Result<(), CacheError> {
    for _ in 0..retries {
        let current = backend.snapshot().await?;
        let transaction =
            cache_source_inventory(current.revision, current.state, inventory, now_ms)?;
        match backend.commit(transaction).await? {
            CommitOutcome::Committed(_) => return Ok(()),
            CommitOutcome::Conflict { .. } => continue,
        }
    }
    Err(CacheError::ConflictExhausted { attempts: retries })
}
use runmat_package::{ContentDigest, SourceInventory};

pub fn cache_source_inventory(
    expected_revision: u64,
    state: CacheState,
    inventory: &SourceInventory,
    now_ms: u64,
) -> Result<CacheTransaction, CacheError> {
    inventory
        .validate()
        .map_err(|reason| CacheError::InvalidObject(reason.to_string()))?;
    if !state.objects.contains_key(&inventory.tree_digest) {
        return Err(CacheError::InvalidState(format!(
            "source inventory references uncached tree {}",
            inventory.tree_digest
        )));
    }
    let bytes = serde_json::to_vec(inventory)
        .map_err(|error| CacheError::InvalidObject(error.to_string()))?;
    let metadata = SourceIndexMetadata {
        digest: ContentDigest::sha256(&bytes),
        tree_digest: inventory.tree_digest.clone(),
        schema_version: inventory.schema_version,
        byte_len: bytes.len() as u64,
    };
    let current = state.clone();
    let mut transaction = CacheTransaction::metadata_only(expected_revision, state);
    transaction.next_state.objects.insert(
        metadata.digest.clone(),
        CacheObject::SourceIndex(metadata.clone()),
    );
    transaction
        .next_state
        .access
        .entry(metadata.digest.clone())
        .and_modify(|access| access.touch(now_ms))
        .or_insert_with(|| AccessRecord::new(now_ms));
    transaction.writes.insert(
        metadata.digest.clone(),
        ObjectWrite::new(CacheObject::SourceIndex(metadata), Some(bytes))?,
    );
    transaction.validate_transition(&current)?;
    Ok(transaction)
}

pub async fn load_source_inventory<B: CacheBackend>(
    backend: &B,
    tree_digest: &ContentDigest,
    schema_version: u32,
) -> Result<SourceInventory, CacheError> {
    let snapshot = backend.snapshot().await?;
    let matches = snapshot
        .state
        .objects
        .values()
        .filter_map(|object| match object {
            CacheObject::SourceIndex(metadata)
                if &metadata.tree_digest == tree_digest
                    && metadata.schema_version == schema_version =>
            {
                Some(metadata)
            }
            _ => None,
        })
        .collect::<Vec<_>>();
    let [metadata] = matches.as_slice() else {
        return if matches.is_empty() {
            Err(CacheError::Miss(tree_digest.clone()))
        } else {
            Err(CacheError::InvalidState(format!(
                "tree {tree_digest} has multiple source inventories for schema {schema_version}"
            )))
        };
    };
    let bytes = backend
        .read_object_bytes(&metadata.digest)
        .await?
        .ok_or_else(|| CacheError::Miss(metadata.digest.clone()))?;
    if bytes.len() as u64 != metadata.byte_len || ContentDigest::sha256(&bytes) != metadata.digest {
        return Err(CacheError::DigestMismatch(metadata.digest.clone()));
    }
    let inventory: SourceInventory = serde_json::from_slice(&bytes)
        .map_err(|error| CacheError::InvalidObject(error.to_string()))?;
    inventory
        .validate()
        .map_err(|reason| CacheError::InvalidObject(reason.to_string()))?;
    if inventory.tree_digest != *tree_digest || inventory.schema_version != schema_version {
        return Err(CacheError::InvalidObject(
            "source inventory payload does not match its cache metadata".to_string(),
        ));
    }
    Ok(inventory)
}

use super::{acquire, release, renew, Lease, LeaseId, LeaseOwner};
use crate::{CacheBackend, CacheError, CacheTransaction, CommitOutcome};
use runmat_package::ContentDigest;
use std::collections::BTreeSet;

pub async fn acquire_lease<B: CacheBackend>(
    backend: &B,
    id: LeaseId,
    owner: LeaseOwner,
    objects: BTreeSet<ContentDigest>,
    now_ms: u64,
    ttl_ms: u64,
    retries: usize,
) -> Result<Lease, CacheError> {
    for _ in 0..retries {
        let snapshot = backend.snapshot().await?;
        let mut next = snapshot.state;
        let lease = acquire(
            &mut next,
            id.clone(),
            owner.clone(),
            objects.clone(),
            now_ms,
            ttl_ms,
        )?;
        match backend
            .commit(CacheTransaction::metadata_only(snapshot.revision, next))
            .await?
        {
            CommitOutcome::Committed(_) => return Ok(lease),
            CommitOutcome::Conflict { .. } => continue,
        }
    }
    Err(CacheError::ConflictExhausted { attempts: retries })
}

pub async fn renew_lease<B: CacheBackend>(
    backend: &B,
    lease: &Lease,
    now_ms: u64,
    ttl_ms: u64,
    retries: usize,
) -> Result<Lease, CacheError> {
    for _ in 0..retries {
        let snapshot = backend.snapshot().await?;
        let mut next = snapshot.state;
        let renewed = renew(
            &mut next,
            &lease.id,
            &lease.owner,
            lease.generation,
            now_ms,
            ttl_ms,
        )?;
        match backend
            .commit(CacheTransaction::metadata_only(snapshot.revision, next))
            .await?
        {
            CommitOutcome::Committed(_) => return Ok(renewed),
            CommitOutcome::Conflict { .. } => continue,
        }
    }
    Err(CacheError::ConflictExhausted { attempts: retries })
}

pub async fn release_lease<B: CacheBackend>(
    backend: &B,
    lease: &Lease,
    retries: usize,
) -> Result<(), CacheError> {
    for _ in 0..retries {
        let snapshot = backend.snapshot().await?;
        let mut next = snapshot.state;
        release(&mut next, &lease.id, &lease.owner, lease.generation)?;
        match backend
            .commit(CacheTransaction::metadata_only(snapshot.revision, next))
            .await?
        {
            CommitOutcome::Committed(_) => return Ok(()),
            CommitOutcome::Conflict { .. } => continue,
        }
    }
    Err(CacheError::ConflictExhausted { attempts: retries })
}

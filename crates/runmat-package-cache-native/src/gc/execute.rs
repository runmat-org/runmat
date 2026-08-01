use runmat_package_cache::gc;
use runmat_package_cache::{
    CacheBackend, CacheError, CacheTransaction, CommitOutcome, GcPlan, GcPolicy,
};

pub async fn execute<B: CacheBackend>(
    backend: &B,
    policy: GcPolicy,
    retries: usize,
) -> Result<GcPlan, CacheError> {
    for _ in 0..retries {
        let snapshot = backend.snapshot().await?;
        let plan = GcPlan::build(&snapshot.state, policy);
        if plan.delete.is_empty() {
            return Ok(plan);
        }
        let mut next = snapshot.state;
        gc::apply_plan(&mut next, &plan);
        let mut transaction = CacheTransaction::metadata_only(snapshot.revision, next);
        transaction.deletes.clone_from(&plan.delete);
        match backend.commit(transaction).await? {
            CommitOutcome::Committed(_) => return Ok(plan),
            CommitOutcome::Conflict { .. } => continue,
        }
    }
    Err(CacheError::ConflictExhausted { attempts: retries })
}

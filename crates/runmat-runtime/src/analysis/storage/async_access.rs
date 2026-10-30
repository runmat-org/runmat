//! Suspend only at provider capture/commit boundaries. The canonical artifact
//! algorithms and file formats operate unchanged inside one private transaction.
use super::*;
use std::cell::RefCell;

thread_local! {
    static SCOPED_STORE: RefCell<Option<Arc<dyn AnalysisArtifactStore>>> = const { RefCell::new(None) };
}
pub(super) fn scoped_store() -> Option<Arc<dyn AnalysisArtifactStore>> {
    SCOPED_STORE.with(|slot| slot.borrow().clone())
}
pub(in crate::analysis) fn scoped_artifact_provider(
) -> Option<Arc<dyn runmat_filesystem::FsProvider>> {
    scoped_store()?
        .filesystem_scope()
        .map(|(_, provider)| provider)
}
struct Restore(Option<Arc<dyn AnalysisArtifactStore>>);
impl Drop for Restore {
    fn drop(&mut self) {
        SCOPED_STORE.with(|slot| *slot.borrow_mut() = self.0.take());
    }
}

/// The inner operation error is preserved intact. A successful inner result is
/// returned only after the original provider confirms atomic publication.
/// There is no await while an artifact scope is installed. The process-wide
/// filesystem registry is never changed by this adapter.
pub async fn with_artifact_access<T, E>(
    writable: bool,
    operation: impl FnOnce(&dyn AnalysisArtifactStore) -> Result<T, E>,
) -> Result<Result<T, E>, AnalysisPublicationError> {
    let store = capture_artifact_store()?;
    let Some((root, provider)) = store.filesystem_scope() else {
        return Ok(operation(store.as_ref()));
    };
    let maximum = crate::analysis::configuration::publication_staging_budget()?;
    match provider.workspace_access(&root, writable, maximum) {
        Ok(scope) => {
            drop(scope);
            return Ok(operation(store.as_ref()));
        }
        Err(error) if error.kind() == ErrorKind::Unsupported => {}
        Err(error) => return Err(AnalysisPublicationError::from_io(error)),
    }
    if !provider.supports_workspace_transactions() {
        return Err(
            "This storage provider cannot save or read simulation results atomically.".into(),
        );
    }
    // Evidence is part of the same result tree and the same commit. A separately
    // configured root requires its own provider-coordinated multi-root contract.
    if !crate::analysis::structural_study::publication::evidence::study_evidence_root()
        .starts_with(&root)
    {
        return Err("Simulation evidence must share the configured result storage root.".into());
    }
    let transaction = runmat_filesystem::workspace_access::WorkspaceTransaction::capture(
        provider,
        root.clone(),
        writable,
        maximum,
    )
    .await
    .map_err(AnalysisPublicationError::from_io)?;
    let view = transaction.view();
    let scoped: Arc<dyn AnalysisArtifactStore> = Arc::new(FilesystemAnalysisArtifactStore {
        root,
        provider: view.clone(),
        manifests: RwLock::new(FilesystemManifestCache::default()),
    });
    let output = {
        let _restore = Restore(SCOPED_STORE.with(|slot| slot.replace(Some(scoped.clone()))));
        operation(scoped.as_ref())
    };
    if output.is_err() {
        return Ok(output);
    }
    if writable && crate::interrupt::is_cancelled() {
        return Err("Simulation result publication was cancelled.".into());
    }
    use runmat_filesystem::workspace::WorkspacePublicationOutcome::*;
    match transaction
        .commit()
        .await
        .map_err(AnalysisPublicationError::from_io)?
    {
        None | Some(Published { .. } | CommittedPendingRecovery { .. }) => Ok(output),
        Some(DurabilityIndeterminate { receipt }) => {
            Err(AnalysisPublicationError::DurabilityIndeterminate {
                workspace: receipt,
                publication: None,
            })
        }
        Some(Conflict { .. }) => Err(
            "Simulation result storage changed during publication. Retry to save the results."
                .into(),
        ),
        Some(IdReused) => Err("The simulation publication identity was reused.".into()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn public_attempt_queries_use_the_captured_store_and_restore_the_global_owner() {
        let _host = crate::analysis::tests::analysis_test_guard();
        // An uncaptured memory filesystem refuses coherent access. It must
        // remain the global owner, not replace the scoped transaction below.
        let global: Arc<dyn AnalysisArtifactStore> = Arc::new(FilesystemAnalysisArtifactStore {
            provider: Arc::new(runmat_filesystem::MemoryFsProvider::new()),
            root: "/results".into(),
            manifests: RwLock::new(FilesystemManifestCache::default()),
        });
        replace_artifact_store(Arc::clone(&global)).unwrap();
        assert!(super::super::run_attempts("definition").is_err());
        {
            let scoped: Arc<dyn AnalysisArtifactStore> =
                Arc::new(InMemoryAnalysisArtifactStore::new());
            let _restore = Restore(SCOPED_STORE.with(|slot| slot.replace(Some(scoped))));
            assert!(super::super::run_attempts("definition").unwrap().is_empty());
            assert_eq!(
                super::super::resolve_run_attempt("definition").unwrap(),
                "definition"
            );
            assert!(Arc::ptr_eq(&*global_store().read().unwrap(), &global));
        }
        assert!(super::super::resolve_run_attempt("definition").is_err());
    }
}

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
/// Where result-store files (meshes and study evidence) are read and written.
#[derive(Clone)]
pub(in crate::analysis) enum ResultStoreProvider {
    /// The captured transaction view of an open [`with_artifact_access`] scope.
    Scoped(Arc<dyn runmat_filesystem::FsProvider>),
    /// The result store's own provider, outside any scope. A provider that
    /// allows writes only through a captured transaction refuses direct access.
    Store(Arc<dyn runmat_filesystem::FsProvider>),
    /// The result store keeps results in memory and has no filesystem. Its
    /// meshes and evidence are files of the host filesystem installed when
    /// the provider was captured.
    HostOfMemoryStore(Arc<dyn runmat_filesystem::FsProvider>),
}

impl ResultStoreProvider {
    pub(in crate::analysis) fn capture() -> Result<Self, String> {
        if let Some(provider) = scoped_artifact_provider() {
            return Ok(Self::Scoped(provider));
        }
        Ok(match capture_artifact_store()?.filesystem_scope() {
            Some((_, provider)) => Self::Store(provider),
            None => Self::HostOfMemoryStore(runmat_filesystem::current_provider()),
        })
    }

    /// The provider to use. The host provider is chosen only for an in-memory store.
    pub(in crate::analysis) fn provider(&self) -> Arc<dyn runmat_filesystem::FsProvider> {
        match self {
            Self::Scoped(provider) | Self::Store(provider) | Self::HostOfMemoryStore(provider) => {
                provider.clone()
            }
        }
    }

    /// Open coherent access to `root`. Outside a scope, a provider that allows
    /// writes only through a captured transaction refuses here with a clear error.
    pub(in crate::analysis) fn workspace_access(
        &self,
        root: &Path,
        writable: bool,
        maximum: u64,
    ) -> Result<
        Box<dyn runmat_filesystem::workspace_access::WorkspaceAccess>,
        AnalysisPublicationError,
    > {
        match self.provider().workspace_access(root, writable, maximum) {
            Ok(access) => Ok(access),
            Err(error)
                if error.kind() == ErrorKind::Unsupported
                    && matches!(self, Self::Store(_)) =>
            {
                Err(format!(
                    "Simulation results at {} can be saved only inside a result transaction, and none is open. Run the save through with_artifact_access. ({error})",
                    root.display()
                )
                .into())
            }
            Err(error) => Err(AnalysisPublicationError::from_io(error)),
        }
    }
}

/// The result store refuses direct coherent access, so every read and write
/// goes through one captured transaction of its provider.
fn admit_transaction(
    provider: &dyn runmat_filesystem::FsProvider,
    root: &Path,
) -> Result<(), AnalysisPublicationError> {
    admit_read_snapshot(provider)?;
    // Evidence is part of the same result tree and the same commit. A separately
    // configured root requires its own provider-coordinated multi-root contract.
    if !crate::analysis::structural_study::publication::evidence::study_evidence_root()
        .starts_with(root)
    {
        return Err("Simulation evidence must share the configured result storage root.".into());
    }
    Ok(())
}

/// A read snapshot needs only a provider that captures transactions. It reads
/// the result tree and commits nothing, so the rule that evidence shares the
/// write commit does not apply.
fn admit_read_snapshot(
    provider: &dyn runmat_filesystem::FsProvider,
) -> Result<(), AnalysisPublicationError> {
    if !provider.supports_workspace_transactions() {
        return Err(
            "This storage provider cannot save or read simulation results atomically.".into(),
        );
    }
    Ok(())
}

/// A read-only result store over one consistent snapshot of `provider`.
pub(super) fn capture_read_snapshot(
    root: PathBuf,
    provider: Arc<dyn runmat_filesystem::FsProvider>,
    maximum: u64,
) -> Result<FilesystemAnalysisArtifactStore, AnalysisPublicationError> {
    admit_read_snapshot(provider.as_ref())?;
    let view =
        runmat_filesystem::workspace_access::WorkspaceTransaction::capture_read_view_blocking(
            provider,
            root.clone(),
            maximum,
        )
        .map_err(AnalysisPublicationError::from_io)?;
    Ok(FilesystemAnalysisArtifactStore {
        root,
        provider: view,
        manifests: RwLock::new(FilesystemManifestCache::default()),
    })
}

/// Run synchronous result reads against one consistent snapshot of the result store.
///
/// Inside an open artifact scope, or when the store's provider grants direct
/// coherent access, the operation runs unchanged. When the provider allows
/// access only through a captured transaction (a cloud project), one
/// read-only snapshot is captured and installed for the whole operation, so a
/// run attempt, its records, field values and meshes all come from the same
/// committed publication. Nothing is committed afterwards.
pub fn with_result_reads<R>(operation: impl FnOnce() -> R) -> Result<R, AnalysisPublicationError> {
    if scoped_store().is_some() {
        return Ok(operation());
    }
    let store = capture_artifact_store()?;
    let Some((root, provider)) = store.filesystem_scope() else {
        return Ok(operation());
    };
    let maximum = crate::analysis::configuration::publication_staging_budget()?;
    match provider.workspace_access(&root, false, maximum) {
        Ok(scope) => {
            drop(scope);
            return Ok(operation());
        }
        Err(error) if error.kind() == ErrorKind::Unsupported => {}
        // A result store that does not exist yet has no results to read; the
        // operation reports that in its own terms.
        Err(error) if error.kind() == ErrorKind::NotFound => return Ok(operation()),
        Err(error) => return Err(AnalysisPublicationError::from_io(error)),
    }
    let snapshot: Arc<dyn AnalysisArtifactStore> =
        Arc::new(capture_read_snapshot(root, provider, maximum)?);
    let _restore = Restore(SCOPED_STORE.with(|slot| slot.replace(Some(snapshot))));
    Ok(operation())
}

/// Read one file of the result tree (a mesh or study evidence) through the
/// result store's provider, as it was written. Returns `None` for a path
/// outside the result tree, or for an in-memory store whose meshes live on
/// the host filesystem; the caller reads those through its own provider.
pub(in crate::analysis) fn read_result_file(
    path: &Path,
    maximum: u64,
) -> Option<std::io::Result<Vec<u8>>> {
    let store = match capture_artifact_store() {
        Ok(store) => store,
        Err(error) => return Some(Err(std::io::Error::other(error))),
    };
    let (root, provider) = store.filesystem_scope()?;
    let relative = path.strip_prefix(&root).ok()?;
    if relative
        .components()
        .any(|part| !matches!(part, std::path::Component::Normal(_)))
    {
        return None;
    }
    let access = match provider.workspace_access(&root, false, maximum) {
        Ok(access) => Ok(access),
        Err(error)
            if error.kind() == ErrorKind::Unsupported
                && provider.supports_workspace_transactions() =>
        {
            runmat_filesystem::workspace_access::WorkspaceTransaction::capture_read_view_blocking(
                provider,
                root.clone(),
                maximum,
            )
            .and_then(|view| view.workspace_access(&root, false, maximum))
        }
        Err(error) => Err(error),
    };
    Some(access.and_then(|access| access.read(relative, maximum)))
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
    let captured = CapturedArtifactAccess::capture(writable).await?;
    let output = captured.scope().enter(operation);
    captured.finish(writable, output).await
}

/// Run an async operation against one captured artifact transaction.
///
/// The operation receives an [`ArtifactAccessScope`]. Each call to
/// [`ArtifactAccessScope::enter`] runs synchronous work with the captured store
/// installed, and every write staged by those calls lands in the same
/// transaction. The scope is never installed across an await, so work between
/// the `enter` calls (for example, assessment functions running on a task
/// pool) sees no artifact scope. The transaction commits once, after the
/// operation returns `Ok`. An `Err` discards every staged write.
pub async fn with_artifact_access_async<T, E, F, Fut>(
    writable: bool,
    operation: F,
) -> Result<Result<T, E>, AnalysisPublicationError>
where
    F: FnOnce(ArtifactAccessScope) -> Fut,
    Fut: std::future::Future<Output = Result<T, E>>,
{
    let captured = CapturedArtifactAccess::capture(writable).await?;
    let output = operation(captured.scope()).await;
    captured.finish(writable, output).await
}

/// The captured result store of one [`with_artifact_access_async`] call.
#[derive(Clone)]
pub struct ArtifactAccessScope {
    store: Arc<dyn AnalysisArtifactStore>,
    install: bool,
}

impl ArtifactAccessScope {
    /// Run synchronous artifact work inside the captured transaction.
    pub fn enter<R>(&self, operation: impl FnOnce(&dyn AnalysisArtifactStore) -> R) -> R {
        if !self.install {
            return operation(self.store.as_ref());
        }
        let _restore = Restore(SCOPED_STORE.with(|slot| slot.replace(Some(self.store.clone()))));
        operation(self.store.as_ref())
    }
}

enum CapturedArtifactAccess {
    /// The store's provider grants coherent access directly, or the store has no filesystem.
    Direct(Arc<dyn AnalysisArtifactStore>),
    /// The provider requires one captured transaction, committed in `finish`.
    Transaction {
        transaction: runmat_filesystem::workspace_access::WorkspaceTransaction,
        scoped: Arc<dyn AnalysisArtifactStore>,
    },
}

impl CapturedArtifactAccess {
    async fn capture(writable: bool) -> Result<Self, AnalysisPublicationError> {
        let store = capture_artifact_store()?;
        let Some((root, provider)) = store.filesystem_scope() else {
            return Ok(Self::Direct(store));
        };
        let maximum = crate::analysis::configuration::publication_staging_budget()?;
        match provider.workspace_access(&root, writable, maximum) {
            Ok(scope) => {
                drop(scope);
                return Ok(Self::Direct(store));
            }
            Err(error) if error.kind() == ErrorKind::Unsupported => {}
            Err(error) => return Err(AnalysisPublicationError::from_io(error)),
        }
        admit_transaction(provider.as_ref(), &root)?;
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
        Ok(Self::Transaction {
            transaction,
            scoped,
        })
    }

    fn scope(&self) -> ArtifactAccessScope {
        match self {
            Self::Direct(store) => ArtifactAccessScope {
                store: store.clone(),
                install: false,
            },
            Self::Transaction { scoped, .. } => ArtifactAccessScope {
                store: scoped.clone(),
                install: true,
            },
        }
    }

    async fn finish<T, E>(
        self,
        writable: bool,
        output: Result<T, E>,
    ) -> Result<Result<T, E>, AnalysisPublicationError> {
        let Self::Transaction { transaction, .. } = self else {
            return Ok(output);
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
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn public_attempt_queries_use_the_captured_store_and_restore_the_global_owner() {
        let _host = crate::analysis::tests::analysis_test_guard();
        // An uncaptured memory filesystem refuses direct coherent access, so
        // its reads come from a read-only snapshot. It must remain the global
        // owner, not replace the scoped transaction below.
        let global: Arc<dyn AnalysisArtifactStore> = Arc::new(FilesystemAnalysisArtifactStore {
            provider: Arc::new(runmat_filesystem::MemoryFsProvider::new()),
            root: "/results".into(),
            manifests: RwLock::new(FilesystemManifestCache::default()),
        });
        replace_artifact_store(Arc::clone(&global)).unwrap();
        assert!(super::super::run_attempts("definition").unwrap().is_empty());
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
        assert_eq!(
            super::super::resolve_run_attempt("definition").unwrap(),
            "definition"
        );
        assert!(Arc::ptr_eq(&*global_store().read().unwrap(), &global));
    }
}

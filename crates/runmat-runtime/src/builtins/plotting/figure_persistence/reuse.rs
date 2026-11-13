//! Session-bound identity for `openfig(...,"reuse")`.

use runmat_filesystem::FsProvider;
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::{Mutex, OnceLock, Weak};

use super::*;

#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
enum ReuseSession {
    Runtime(crate::context::RuntimeContextLocalId),
    Ambient,
}

#[derive(Clone, Debug, Hash, PartialEq, Eq)]
pub(super) struct FigureArchiveIdentity {
    session: ReuseSession,
    provider: usize,
    canonical_path: PathBuf,
}

struct ReuseEntry {
    session_lease: Option<Weak<()>>,
    provider_lease: Weak<dyn FsProvider>,
    handles: Vec<FigureHandle>,
}

fn registry() -> &'static Mutex<HashMap<FigureArchiveIdentity, ReuseEntry>> {
    static REGISTRY: OnceLock<Mutex<HashMap<FigureArchiveIdentity, ReuseEntry>>> = OnceLock::new();
    REGISTRY.get_or_init(|| Mutex::new(HashMap::new()))
}

pub(super) async fn archive_identity(
    runtime: Option<&crate::context::RuntimeContext>,
    selected: &FigureFsPath,
    builtin: &'static str,
) -> BuiltinResult<FigureArchiveIdentity> {
    let canonical_path = crate::execution::staged_file::await_cancellable(
        runtime,
        builtin,
        selected.canonical_path(),
    )
    .await?
    .map_err(|source| {
        persistence_error_with_source(
            &ERROR_IO,
            builtin,
            format!("{builtin}: unable to identify figure archive"),
            source,
        )
    })?;
    Ok(FigureArchiveIdentity {
        session: runtime
            .map(|runtime| ReuseSession::Runtime(runtime.local_identity()))
            .unwrap_or(ReuseSession::Ambient),
        provider: selected.provider_identity(),
        canonical_path,
    })
}

pub(super) fn reusable_figures(identity: &FigureArchiveIdentity) -> Option<Vec<FigureHandle>> {
    let mut entries = registry()
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    entries.retain(|_, entry| {
        entry
            .session_lease
            .as_ref()
            .is_none_or(|lease| lease.strong_count() > 0)
            && entry.provider_lease.strong_count() > 0
    });
    let handles = entries.get(identity)?.handles.clone();
    if handles.is_empty()
        || handles
            .iter()
            .any(|handle| !super::super::state::figure_handle_exists(*handle))
    {
        entries.remove(identity);
        return None;
    }
    Some(handles)
}

pub(super) fn remember_figures(
    runtime: Option<&crate::context::RuntimeContext>,
    identity: FigureArchiveIdentity,
    provider_lease: Weak<dyn FsProvider>,
    handles: &[FigureHandle],
) {
    registry()
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
        .insert(
            identity,
            ReuseEntry {
                session_lease: runtime.map(crate::context::RuntimeContext::session_lease),
                provider_lease,
                handles: handles.to_vec(),
            },
        );
}

#[cfg(test)]
mod tests {
    use super::*;
    use runmat_filesystem::MemoryFsProvider;
    use std::sync::Arc;

    #[test]
    fn retired_provider_entries_release_their_handles() {
        let provider = Arc::new(MemoryFsProvider::with_current_dir("/reuse"));
        let provider_dyn: Arc<dyn FsProvider> = provider;
        let provider_lease = Arc::downgrade(&provider_dyn);
        let identity = FigureArchiveIdentity {
            session: ReuseSession::Ambient,
            provider: Arc::as_ptr(&provider_dyn) as *const () as usize,
            canonical_path: PathBuf::from("/reuse/figure.fig"),
        };
        remember_figures(
            None,
            identity.clone(),
            provider_lease,
            &[FigureHandle::from(1)],
        );

        drop(provider_dyn);

        assert!(reusable_figures(&identity).is_none());
        assert!(!registry()
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .contains_key(&identity));
    }
}

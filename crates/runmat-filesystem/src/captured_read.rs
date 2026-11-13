//! Poll/thread-local read selection for host tooling. Never swaps the process
//! provider, never masks a selected Runtime invocation, and never holds a lock
//! or local selection across suspension. Read attenuation reuses the existing
//! provider wrapper; no new filesystem authority is issued.
use crate::{
    pinned::{InputManifest, PinnedFsProvider},
    FsProvider, SelectedFsProvider,
};
use std::{
    cell::RefCell,
    future::{poll_fn, Future},
    io,
    marker::PhantomData,
    path::{Path, PathBuf},
    rc::Rc,
    sync::Arc,
};

#[derive(Clone)]
struct Selection {
    provider: Arc<dyn FsProvider>,
    invocation_provider: Arc<dyn FsProvider>,
    invocation_owner: Option<crate::invocation::InvocationFilesystem>,
    directory: Option<crate::invocation::InvocationFilesystem>,
    readonly: bool,
}
thread_local! { static READ_SCOPE: RefCell<Option<Selection>> = const { RefCell::new(None) }; }

#[derive(Clone)]
pub struct CapturedReadScope {
    original: Arc<dyn FsProvider>,
    selected: Selection,
    original_selection: bool,
}

impl CapturedReadScope {
    /// Capture original provider and only the CWD required by known relative
    /// paths, atomically under the existing admission port. Empty paths are valid
    /// but do not grant an ambient CWD. Inputs attenuate reads before discovery.
    pub fn capture(
        paths: &[&Path],
        inputs: Option<InputManifest>,
    ) -> io::Result<(Self, Vec<PathBuf>)> {
        let (original, directory) = crate::capture_provider_for_paths(paths)?;
        let local = READ_SCOPE.with(|slot| slot.borrow().clone());
        let invocation_owner = match crate::invocation::selected() {
            Some(crate::invocation::InvocationFilesystemSelection::Selected(owner)) => Some(owner),
            Some(crate::invocation::InvocationFilesystemSelection::Denied { .. }) => {
                return Err(crate::invocation::denied_environment())
            }
            None => None,
        };
        let directory_owner = match &invocation_owner {
            Some(owner) => Some(owner.clone()),
            None if local.is_some() => local.as_ref().and_then(|scope| scope.directory.clone()),
            None => directory
                .as_ref()
                .map(|directory| {
                    crate::invocation::InvocationFilesystem::new(
                        original.clone(),
                        directory.clone(),
                        true,
                        true,
                    )
                })
                .transpose()?,
        };
        let namespace = original.path_namespace();
        let resolved = paths
            .iter()
            .map(|path| {
                if namespace.is_absolute(path) {
                    Ok(path.to_path_buf())
                } else {
                    Ok(directory.as_ref().ok_or_else(missing_directory)?.join(path))
                }
            })
            .collect::<io::Result<Vec<_>>>()?;
        let original_selection =
            inputs.is_none() && !local.as_ref().is_some_and(|scope| scope.readonly);
        let provider: Arc<dyn FsProvider> = match inputs {
            Some(inputs) => Arc::new(PinnedFsProvider::new(original.clone(), inputs)?),
            None => original.clone(),
        };
        let invocation_provider = provider.clone();
        let provider = Arc::new(SelectedFsProvider::new(provider, true, false)?);
        Ok((
            Self {
                original,
                selected: Selection {
                    provider,
                    invocation_provider,
                    invocation_owner,
                    directory: directory_owner,
                    readonly: true,
                },
                original_selection,
            },
            resolved,
        ))
    }

    /// Construct the canonical input owner over this exact captured effective
    /// provider, without exposing or re-capturing its underlying authority.
    pub fn pin_inputs(&self, inputs: InputManifest) -> io::Result<Arc<PinnedFsProvider>> {
        if !self.original_selection {
            return Err(derivation_denied());
        }
        PinnedFsProvider::new(self.original.clone(), inputs).map(Arc::new)
    }

    /// Explicit original execution/publication view, never an unwrapped provider.
    pub fn execution(&self) -> io::Result<CapturedExecutionScope> {
        if !self.original_selection {
            return Err(derivation_denied());
        }
        let directory = self
            .selected
            .directory
            .clone()
            .ok_or_else(missing_directory)?;
        Ok(CapturedExecutionScope {
            scope: Self {
                original: self.original.clone(),
                original_selection: false,
                selected: Selection {
                    provider: self.original.clone(),
                    invocation_provider: self.original.clone(),
                    invocation_owner: self.selected.invocation_owner.clone(),
                    directory: Some(directory),
                    readonly: false,
                },
            },
        })
    }

    /// Reuse this exact existing immutable-input/owned-output owner. A read
    /// overlay cannot nominate another provider or create a new output registry.
    /// Include a relative path (e.g. ".") at original capture to retain CWD.
    pub fn pinned(&self, provider: Arc<PinnedFsProvider>) -> io::Result<CapturedPinnedScope> {
        if !self.original_selection || !provider.base_matches(&self.original) {
            return Err(io::Error::new(
                io::ErrorKind::PermissionDenied,
                "pinned execution provider does not match the captured base owner",
            ));
        }
        let directory = self
            .selected
            .directory
            .clone()
            .ok_or_else(missing_directory)?;
        let provider: Arc<dyn FsProvider> = provider;
        Ok(CapturedPinnedScope {
            scope: Self {
                original: self.original.clone(),
                selected: Selection {
                    invocation_provider: provider.clone(),
                    invocation_owner: self.selected.invocation_owner.clone(),
                    provider,
                    directory: Some(directory),
                    readonly: false,
                },
                original_selection: false,
            },
        })
    }

    /// The private !Send guard exists only inside a poll. It is dropped before
    /// Pending/Ready returns, restoring ONLY the current thread's previous view.
    pub async fn run<F: Future>(&self, future: F) -> io::Result<F::Output> {
        let mut future = std::pin::pin!(future);
        poll_fn(|cx| {
            let _guard = match self.enter() {
                Ok(guard) => guard,
                Err(error) => return std::task::Poll::Ready(Err(error)),
            };
            future.as_mut().poll(cx).map(Ok)
        })
        .await
    }

    fn check_invocation(
        &self,
        selected: Option<(Arc<dyn FsProvider>, Option<PathBuf>)>,
    ) -> io::Result<()> {
        if let Some((provider, _)) = selected {
            // An overlay is not permission to replace a live invocation's exact
            // filesystem. The caller must admit that snapshot through its owner.
            if !Arc::ptr_eq(&provider, &self.selected.invocation_provider) {
                return Err(io::Error::new(
                    io::ErrorKind::PermissionDenied,
                    "captured read scope does not match the selected invocation filesystem",
                ));
            }
        }
        Ok(())
    }
    fn enter(&self) -> io::Result<Guard> {
        self.check_invocation(crate::invocation::snapshot())?;
        check_owner(&self.selected)?;
        let previous = READ_SCOPE.with(|slot| slot.replace(Some(self.selected.clone())));
        Ok(Guard {
            previous,
            _not_send: PhantomData,
        })
    }
    fn attenuated(&self) -> io::Result<Self> {
        let provider = Arc::new(SelectedFsProvider::new(
            self.selected.provider.clone(),
            true,
            false,
        )?);
        Ok(Self {
            original: self.original.clone(),
            original_selection: false,
            selected: Selection {
                provider,
                readonly: true,
                ..self.selected.clone()
            },
        })
    }
}

fn derivation_denied() -> io::Error {
    io::Error::new(
        io::ErrorKind::PermissionDenied,
        "an attenuated read view cannot derive execution authority",
    )
}

pub struct CapturedExecutionScope {
    scope: CapturedReadScope,
}
impl CapturedExecutionScope {
    pub async fn run<F: Future>(&self, future: F) -> io::Result<F::Output> {
        self.scope.run(future).await
    }
    pub fn read_scope(&self) -> io::Result<CapturedReadScope> {
        self.scope.attenuated()
    }
}

/// Only the actual existing PinnedFsProvider may supply this execution view.
/// Its canonical immutable inputs and admitted owned-output registry remain
/// authoritative; unlike CapturedReadScope, outputs are not attenuated away.
pub struct CapturedPinnedScope {
    scope: CapturedReadScope,
}
impl CapturedPinnedScope {
    pub async fn run<F: Future>(&self, future: F) -> io::Result<F::Output> {
        self.scope.run(future).await
    }
    pub fn read_scope(&self) -> io::Result<CapturedReadScope> {
        self.scope.attenuated()
    }
}

struct Guard {
    previous: Option<Selection>,
    _not_send: PhantomData<Rc<()>>,
}
impl Drop for Guard {
    fn drop(&mut self) {
        READ_SCOPE.with(|slot| {
            let _ = slot.replace(self.previous.take());
        });
    }
}
// A RuntimeContext may be entered inside the polled future. Check captures too,
// not only poll entry: never let that change silently bypass pinned selection.
pub(super) fn check_selected(provider: &Arc<dyn FsProvider>) -> io::Result<()> {
    READ_SCOPE.with(|slot| {
        if slot
            .borrow()
            .as_ref()
            .is_some_and(|scope| !Arc::ptr_eq(provider, &scope.invocation_provider))
        {
            Err(io::Error::new(
                io::ErrorKind::PermissionDenied,
                "captured filesystem differs from the selected invocation",
            ))
        } else {
            slot.borrow().as_ref().map_or(Ok(()), check_owner)
        }
    })
}
fn check_owner(scope: &Selection) -> io::Result<()> {
    if let Some(expected) = &scope.invocation_owner {
        if !matches!(crate::invocation::selected(), Some(crate::invocation::InvocationFilesystemSelection::Selected(ref current)) if current.same_owner(expected))
        {
            return Err(io::Error::new(
                io::ErrorKind::PermissionDenied,
                "captured invocation filesystem owner changed",
            ));
        }
    }
    Ok(())
}
pub(super) fn snapshot() -> Option<(Arc<dyn FsProvider>, Option<PathBuf>)> {
    READ_SCOPE.with(|slot| {
        slot.borrow().as_ref().map(|selected| {
            if check_owner(selected).is_err() {
                return (
                    Arc::new(crate::UnavailableFsProvider) as Arc<dyn FsProvider>,
                    None,
                );
            }
            (
                selected.provider.clone(),
                selected
                    .directory
                    .as_ref()
                    .map(|directory| directory.captured_dir()),
            )
        })
    })
}
pub(super) fn selected_view(provider: &Arc<dyn FsProvider>) -> io::Result<Arc<dyn FsProvider>> {
    check_selected(provider)?;
    Ok(READ_SCOPE
        .with(|slot| slot.borrow().as_ref().map(|scope| scope.provider.clone()))
        .unwrap_or_else(|| provider.clone()))
}
pub(super) fn require_directory_write() -> io::Result<()> {
    if READ_SCOPE.with(|slot| slot.borrow().as_ref().is_some_and(|scope| scope.readonly)) {
        Err(io::Error::new(
            io::ErrorKind::PermissionDenied,
            "captured read scope cannot publish a working directory",
        ))
    } else {
        Ok(())
    }
}
pub(super) async fn set_local_directory(path: &Path) -> io::Result<bool> {
    let Some(selected) = READ_SCOPE.with(|slot| slot.borrow().clone()) else {
        return Ok(false);
    };
    require_directory_write()?;
    let directory = selected.directory.as_ref().ok_or_else(missing_directory)?;
    let target = if selected.provider.path_namespace().is_absolute(path) {
        path.to_path_buf()
    } else {
        directory.current_dir()?.join(path)
    };
    let canonical = selected
        .provider
        .canonicalize(&target)
        .await
        .unwrap_or_else(|_| target.clone());
    if !selected.provider.metadata(&canonical).await?.is_dir() {
        return Err(io::Error::new(
            io::ErrorKind::NotFound,
            "selected working directory is not a directory",
        ));
    }
    let same = crate::invocation::selected().is_none()
        && READ_SCOPE.with(|slot| {
            slot.borrow().as_ref().is_some_and(|current| {
                !current.readonly
                    && Arc::ptr_eq(&current.provider, &selected.provider)
                    && current
                        .directory
                        .as_ref()
                        .is_some_and(|current| current.same_owner(directory))
            })
        });
    if !same {
        return Err(io::Error::new(
            io::ErrorKind::Interrupted,
            "filesystem owner changed before local working-directory publication",
        ));
    }
    directory.publish_current_dir(canonical)?;
    Ok(true)
}
pub(super) fn missing_directory() -> io::Error {
    io::Error::new(
        io::ErrorKind::PermissionDenied,
        "captured read scope has no admitted working directory",
    )
}

#[cfg(all(test, not(target_arch = "wasm32")))]
mod tests {
    use super::*;
    use crate::{provider_override_lock, replace_provider, MemoryFsProvider};
    #[test]
    fn local_read_scope_keeps_poll_thread_process_and_invocation_ownership_separate() {
        let _serial = provider_override_lock();
        let first = Arc::new(MemoryFsProvider::with_current_dir("/first"));
        first.write_project_path("/first/a.m", b"first").unwrap();
        let first: Arc<dyn FsProvider> = first;
        let _restore = replace_provider(first.clone());
        let (scope, paths) = CapturedReadScope::capture(&[Path::new("a.m")], None).unwrap();
        assert_eq!(paths, vec![PathBuf::from("/first/a.m")]);
        let second: Arc<dyn FsProvider> = Arc::new(MemoryFsProvider::with_current_dir("/second"));
        crate::set_provider(second.clone());
        // Provider replacement retains an existing virtual CWD. The local
        // execution scope must preserve that exact ambient state, not install
        // the replacement provider's suggested default.
        let ambient_directory = crate::current_dir().unwrap();
        let mut ready = false;
        let mut future = std::pin::pin!(scope.run(poll_fn(|cx| {
            assert_eq!(crate::current_dir().unwrap(), PathBuf::from("/first"));
            if !ready {
                ready = true;
                cx.waker().wake_by_ref();
                return std::task::Poll::Pending;
            }
            std::task::Poll::Ready(())
        })));
        let waker = futures::task::noop_waker();
        let mut context = std::task::Context::from_waker(&waker);
        assert!(future.as_mut().poll(&mut context).is_pending());
        assert!(Arc::ptr_eq(&crate::current_provider(), &second));
        assert!(future.as_mut().poll(&mut context).is_ready());
        assert!(Arc::ptr_eq(&crate::current_provider(), &second));
        let value = futures::executor::block_on(scope.run(async {
            assert!(check_selected(&first).is_ok());
            assert!(check_selected(&second).is_err());
            assert!(selected_view(&first)
                .unwrap()
                .write(Path::new("/first/attenuation"), b"denied")
                .await
                .is_err());
            std::thread::scope(|threads| {
                threads
                    .spawn(|| {
                        assert!(Arc::ptr_eq(&crate::current_provider(), &second));
                    })
                    .join()
                    .unwrap();
            });
            assert!(crate::write_async(Path::new("/first/no-write"), b"denied")
                .await
                .is_err());
            assert!(crate::set_current_dir_async("/").await.is_err());
            crate::read_async(Path::new("/first/a.m")).await.unwrap()
        }))
        .unwrap();
        assert_eq!(value, b"first");
        let original_execution = scope.execution().unwrap();
        futures::executor::block_on(original_execution.run(async {
            crate::set_current_dir_async("/").await.unwrap();
            let mut pending = true;
            poll_fn(|cx| {
                if pending {
                    pending = false;
                    cx.waker().wake_by_ref();
                    std::task::Poll::Pending
                } else {
                    std::task::Poll::Ready(())
                }
            })
            .await;
            assert_eq!(crate::current_dir().unwrap(), PathBuf::from("/"));
            crate::write_async("local.txt", b"local").await.unwrap();
        }))
        .unwrap();
        assert_eq!(crate::current_dir().unwrap(), ambient_directory);
        assert!(Arc::ptr_eq(&crate::current_provider(), &second));
        assert!(original_execution
            .read_scope()
            .unwrap()
            .execution()
            .is_err());
        futures::executor::block_on(scope.run(async {
            let (nested, _) = CapturedReadScope::capture(&[Path::new(".")], None).unwrap();
            assert!(nested.execution().is_err());
        }))
        .unwrap();
        let pinned = Arc::new(
            PinnedFsProvider::new(
                first.clone(),
                InputManifest {
                    snapshot_id: "retained".into(),
                    entries: Vec::new(),
                },
            )
            .unwrap(),
        );
        let execution = scope.pinned(pinned.clone()).unwrap();
        futures::executor::block_on(execution.run(async {
            crate::write_async("/first/owned.txt", b"one")
                .await
                .unwrap();
        }))
        .unwrap();
        let same_execution = scope.pinned(pinned.clone()).unwrap();
        let output =
            futures::executor::block_on(same_execution.run(crate::read_async("/first/owned.txt")))
                .unwrap()
                .unwrap();
        assert_eq!(output, b"one");
        let read = same_execution.read_scope().unwrap();
        assert!(futures::executor::block_on(
            read.run(crate::write_async("/first/owned.txt", b"denied"))
        )
        .unwrap()
        .is_err());
        let selected: Arc<dyn FsProvider> = pinned;
        assert!(execution
            .scope
            .check_invocation(Some((selected, Some("/first".into()))))
            .is_ok());
        assert!(execution
            .scope
            .check_invocation(Some((first.clone(), Some("/first".into()))))
            .is_err());
        let foreign = Arc::new(
            PinnedFsProvider::new(
                second.clone(),
                InputManifest {
                    snapshot_id: "foreign".into(),
                    entries: Vec::new(),
                },
            )
            .unwrap(),
        );
        assert!(scope.pinned(foreign).is_err());
        assert!(scope
            .check_invocation(Some((first.clone(), Some("/first".into()))))
            .is_ok());
        assert!(scope
            .check_invocation(Some((second, Some("/second".into()))))
            .is_err());
        // The compatible selected invocation keeps ownership of its own CWD.
        assert!(scope
            .check_invocation(Some((first, Some("/changed".into()))))
            .is_ok());
        assert!(!Arc::ptr_eq(&crate::current_provider(), &scope.original));
    }
}

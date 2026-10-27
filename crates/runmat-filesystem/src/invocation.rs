//! Poll-scoped selection of an invocation's filesystem authority.
//!
//! The callback is installed once by the Runtime owner. It is consulted only
//! when an operation captures provider and CWD; the returned Arc remains the
//! operation's authority through suspension. No process-global provider guard
//! spans an await or serializes unrelated invocations.

use crate::FsProvider;
use std::io;
use std::path::PathBuf;
use std::sync::{Arc, Mutex, OnceLock};

#[derive(Clone)]
pub struct InvocationFilesystem {
    provider: Arc<dyn FsProvider>,
    directory: Arc<Mutex<PathBuf>>,
    environment_read: bool,
    environment_write: bool,
}

impl InvocationFilesystem {
    pub fn new(
        provider: Arc<dyn FsProvider>,
        directory: PathBuf,
        environment_read: bool,
        environment_write: bool,
    ) -> io::Result<Self> {
        if !provider.path_namespace().is_absolute(&directory) {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "selected filesystem directory must be absolute in its provider namespace",
            ));
        }
        Ok(Self {
            provider,
            directory: Arc::new(Mutex::new(directory)),
            environment_read,
            environment_write,
        })
    }

    pub fn provider(&self) -> &Arc<dyn FsProvider> {
        &self.provider
    }

    pub(crate) fn environment_write_allowed(&self) -> bool {
        self.environment_write
    }

    pub fn current_dir(&self) -> io::Result<PathBuf> {
        if !self.environment_read {
            return Err(denied_environment());
        }
        Ok(self.captured_dir())
    }

    pub(crate) fn captured_dir(&self) -> PathBuf {
        self.directory
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .clone()
    }

    pub(crate) fn same_owner(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.provider, &other.provider)
            && Arc::ptr_eq(&self.directory, &other.directory)
    }

    pub(crate) fn publish_current_dir(&self, directory: PathBuf) -> io::Result<()> {
        if !self.environment_write {
            return Err(denied_environment());
        }
        *self
            .directory
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner()) = directory;
        Ok(())
    }
}

/// `None` from the resolver means there is no active RuntimeContext. `Denied`
/// means there is an active selected invocation but no filesystem authority;
/// it must never fall through to the process-global registry.
#[derive(Clone)]
pub enum InvocationFilesystemSelection {
    Selected(InvocationFilesystem),
    Denied {
        directory: Option<PathBuf>,
        environment_read: bool,
    },
}

type Resolver = fn() -> Option<InvocationFilesystemSelection>;
static RESOLVER: OnceLock<Resolver> = OnceLock::new();

pub fn register_invocation_filesystem_resolver(
    resolver: fn() -> Option<InvocationFilesystemSelection>,
) -> io::Result<()> {
    install_resolver(&RESOLVER, resolver)
}

fn install_resolver(slot: &OnceLock<Resolver>, resolver: Resolver) -> io::Result<()> {
    // Concurrent contexts can install the same owner. Compare against the
    // winning registration rather than treating every losing set as a change.
    let current = slot.get_or_init(|| resolver);
    if std::ptr::fn_addr_eq(*current, resolver) {
        Ok(())
    } else {
        Err(io::Error::new(
            io::ErrorKind::AlreadyExists,
            "a different invocation filesystem resolver is already installed",
        ))
    }
}

pub(crate) fn selected() -> Option<InvocationFilesystemSelection> {
    RESOLVER.get().and_then(|resolver| resolver())
}

pub(crate) fn snapshot() -> Option<(Arc<dyn FsProvider>, Option<PathBuf>)> {
    selected().map(|selection| match selection {
        InvocationFilesystemSelection::Selected(owner) => {
            (Arc::clone(owner.provider()), Some(owner.captured_dir()))
        }
        InvocationFilesystemSelection::Denied { directory, .. } => (
            Arc::new(crate::UnavailableFsProvider) as Arc<dyn FsProvider>,
            Some(directory.unwrap_or_else(|| PathBuf::from("/"))),
        ),
    })
}

pub(crate) fn denied_environment() -> io::Error {
    io::Error::new(
        io::ErrorKind::PermissionDenied,
        "selected invocation has no environment service for working directory",
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn concurrent_resolver_registration_is_idempotent_but_rejects_replacement() {
        fn selected() -> Option<InvocationFilesystemSelection> {
            None
        }
        fn denied() -> Option<InvocationFilesystemSelection> {
            Some(InvocationFilesystemSelection::Denied {
                directory: None,
                environment_read: false,
            })
        }
        let slot = OnceLock::new();
        let start = std::sync::Barrier::new(8);
        std::thread::scope(|threads| {
            for _ in 0..8 {
                threads.spawn(|| {
                    start.wait();
                    install_resolver(&slot, selected).unwrap();
                });
            }
        });
        assert_eq!(
            install_resolver(&slot, denied).unwrap_err().kind(),
            io::ErrorKind::AlreadyExists
        );
        assert!(std::ptr::fn_addr_eq(
            *slot.get().unwrap(),
            selected as Resolver
        ));
    }

    #[test]
    fn cloned_invocation_shares_only_its_own_directory_and_checks_environment_issuers() {
        let provider: Arc<dyn FsProvider> = Arc::new(crate::MemoryFsProvider::new());
        let first =
            InvocationFilesystem::new(Arc::clone(&provider), PathBuf::from("/first"), true, true)
                .unwrap();
        let first_clone = first.clone();
        let second =
            InvocationFilesystem::new(provider, PathBuf::from("/second"), true, true).unwrap();
        first
            .publish_current_dir(PathBuf::from("/changed"))
            .unwrap();
        assert_eq!(
            first_clone.current_dir().unwrap(),
            PathBuf::from("/changed")
        );
        assert_eq!(second.current_dir().unwrap(), PathBuf::from("/second"));
        assert!(!first.same_owner(&second));

        let denied = InvocationFilesystem::new(
            Arc::clone(first.provider()),
            PathBuf::from("/private"),
            false,
            false,
        )
        .unwrap();
        assert_eq!(
            denied.current_dir().unwrap_err().kind(),
            io::ErrorKind::PermissionDenied
        );
        assert_eq!(
            denied
                .publish_current_dir(PathBuf::from("/escape"))
                .unwrap_err()
                .kind(),
            io::ErrorKind::PermissionDenied
        );
        assert_eq!(denied.captured_dir(), PathBuf::from("/private"));
    }
}

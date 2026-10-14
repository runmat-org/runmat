use std::path::{Path, PathBuf};

use runmat_process_host::private_bytes::PrivateByteLease;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum LoaderError {
    #[error("could not load native library {path}: {source}")]
    Open {
        path: PathBuf,
        #[source]
        source: libloading::Error,
    },
    #[error("could not resolve native symbol {symbol} in {path}: {source}")]
    Symbol {
        path: PathBuf,
        symbol: String,
        #[source]
        source: libloading::Error,
    },
    #[error("native library {path} does not belong to private byte root {root}")]
    PrivateByteRoot { path: PathBuf, root: PathBuf },
    #[error("native library {path} already has a private byte owner")]
    PrivateByteOwnerAlreadySet { path: PathBuf },
}

pub struct LoadedLibrary {
    path: PathBuf,
    library: libloading::Library,
    // Drop the loaded handle before releasing its private on-disk bytes.
    _private_byte_lease: Option<PrivateByteLease>,
}

#[derive(Debug)]
pub struct LoadedSymbol<'library> {
    _library: &'library LoadedLibrary,
    code: libffi::middle::CodePtr,
}

impl std::fmt::Debug for LoadedLibrary {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("LoadedLibrary")
            .field("path", &self.path)
            .finish_non_exhaustive()
    }
}

impl LoadedLibrary {
    pub fn open(path: impl AsRef<Path>) -> Result<Self, LoaderError> {
        let path = path.as_ref().to_path_buf();
        // SAFETY: Loading a library may run platform constructors. This API is
        // native-only and retains the library for the complete symbol lifetime;
        // higher layers apply trust and isolation policy before calling it.
        let library =
            unsafe { libloading::Library::new(&path) }.map_err(|source| LoaderError::Open {
                path: path.clone(),
                source,
            })?;
        Ok(Self {
            path,
            library,
            _private_byte_lease: None,
        })
    }

    /// Bind exact private bytes before this library can escape into a pointer.
    /// Raw user libraries outside a private root remain caller-owned.
    pub fn attach_private_byte_lease(
        &mut self,
        lease: PrivateByteLease,
    ) -> Result<(), LoaderError> {
        if !self.path.starts_with(lease.path()) {
            return Err(LoaderError::PrivateByteRoot {
                path: self.path.clone(),
                root: lease.path().to_path_buf(),
            });
        }
        if let Some(existing) = self._private_byte_lease.as_ref() {
            return if existing.path() == lease.path() {
                Ok(())
            } else {
                Err(LoaderError::PrivateByteOwnerAlreadySet {
                    path: self.path.clone(),
                })
            };
        }
        self._private_byte_lease = Some(lease);
        Ok(())
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    pub fn symbol(&self, name: &str) -> Result<LoadedSymbol<'_>, LoaderError> {
        let name_bytes = name.as_bytes();
        // SAFETY: The symbol is treated only as an untyped code address here.
        // Its ABI and argument contract are checked against normalized metadata
        // before invocation, and the returned handle borrows this library.
        let symbol = unsafe { self.library.get::<unsafe extern "C" fn()>(name_bytes) }.map_err(
            |source| LoaderError::Symbol {
                path: self.path.clone(),
                symbol: name.into(),
                source,
            },
        )?;
        Ok(LoadedSymbol {
            _library: self,
            code: libffi::middle::CodePtr::from_fun(*symbol),
        })
    }
}

impl LoadedSymbol<'_> {
    pub fn code_ptr(&self) -> libffi::middle::CodePtr {
        self.code
    }
}

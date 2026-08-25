use std::path::{Path, PathBuf};

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
}

pub struct LoadedLibrary {
    path: PathBuf,
    library: libloading::Library,
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
        Ok(Self { path, library })
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

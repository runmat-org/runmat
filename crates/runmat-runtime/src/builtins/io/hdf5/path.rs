#[cfg(not(target_arch = "wasm32"))]
use super::{hdf5_error, ERR_IO, ERR_PROVIDER};
#[cfg(target_arch = "wasm32")]
use super::{hdf5_error, ERR_WASM};
#[cfg(not(target_arch = "wasm32"))]
use crate::build_runtime_error;
use crate::BuiltinResult;
use runmat_filesystem::NativePathIntent;
use std::path::PathBuf;

#[cfg(not(target_arch = "wasm32"))]
pub(super) type Hdf5PathLease = runmat_filesystem::CapturedNativePath;
#[cfg(target_arch = "wasm32")]
pub(super) enum Hdf5PathLease {}

pub(super) fn capture(
    path: PathBuf,
    intent: NativePathIntent,
    builtin: &'static str,
) -> BuiltinResult<Hdf5PathLease> {
    #[cfg(not(target_arch = "wasm32"))]
    {
        let captured = runmat_filesystem::capture_native_path(&path, intent).map_err(|source| {
            build_runtime_error(format!("{builtin}: unable to capture native HDF5 path"))
                .with_builtin(builtin)
                .with_identifier(ERR_IO.identifier.expect("declared HDF5 I/O identifier"))
                .with_source(source)
                .build()
        })?;
        captured.ok_or_else(|| hdf5_error(builtin, &ERR_PROVIDER, ERR_PROVIDER.message))
    }
    #[cfg(target_arch = "wasm32")]
    {
        let _ = (path, intent);
        Err(hdf5_error(builtin, &ERR_WASM, ERR_WASM.message))
    }
}

pub(super) fn check(path: &Hdf5PathLease, builtin: &'static str) -> BuiltinResult<()> {
    #[cfg(not(target_arch = "wasm32"))]
    {
        path.check_current().map_err(|source| {
            build_runtime_error(format!("{builtin}: captured native HDF5 path was revoked"))
                .with_builtin(builtin)
                .with_identifier(ERR_IO.identifier.expect("declared HDF5 I/O identifier"))
                .with_source(source)
                .build()
        })
    }
    #[cfg(target_arch = "wasm32")]
    {
        let _ = builtin;
        match *path {}
    }
}

//! Exact selected-session dialog routing.

use super::{RuntimeContext, RuntimeServiceFuture};
use crate::RuntimeError;

pub(crate) fn select_file_open(
    runtime: Option<&RuntimeContext>,
    request: runmat_filesystem::OpenFileDialogRequest,
) -> RuntimeServiceFuture<Result<Option<runmat_filesystem::OpenFileDialogSelection>, RuntimeError>>
{
    let Some(host) = runtime.and_then(|runtime| runtime.service_ports().host().cloned()) else {
        return Box::pin(async { Ok(None) });
    };
    host.select_file_open(request)
}

pub(crate) fn select_file_save(
    runtime: Option<&RuntimeContext>,
    request: runmat_filesystem::SaveFileDialogRequest,
) -> RuntimeServiceFuture<Result<Option<runmat_filesystem::SaveFileDialogSelection>, RuntimeError>>
{
    let Some(host) = runtime.and_then(|runtime| runtime.service_ports().host().cloned()) else {
        return Box::pin(async { Ok(None) });
    };
    host.select_file_save(request)
}

pub(crate) fn select_directory(
    runtime: Option<&RuntimeContext>,
    request: runmat_filesystem::DirectoryDialogRequest,
) -> RuntimeServiceFuture<Result<Option<runmat_filesystem::DirectoryDialogSelection>, RuntimeError>>
{
    let Some(host) = runtime.and_then(|runtime| runtime.service_ports().host().cloned()) else {
        return Box::pin(async { Ok(None) });
    };
    host.select_directory(request)
}

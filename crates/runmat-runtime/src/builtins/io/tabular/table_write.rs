//! Table serializers use the existing selected tabular path/staged owner.
//! This bridge adds no filesystem or publication authority.
use super::{
    cooperation::{await_io, TabularCooperation},
    path::{TabularFsOwner, TabularFsPath},
    write_owner,
};
use crate::{context::RuntimeContext, BuiltinResult, RuntimeError};
use runmat_builtins::TabularWriteKind;
use std::path::{Path, PathBuf};

fn io(
    kind: TabularWriteKind,
    source: impl crate::runtime_error::RuntimeErrorSourceBound,
) -> RuntimeError {
    crate::build_runtime_error(format!("{}: selected table file write failed", kind.name()))
        .with_builtin(kind.name())
        .with_identifier(
            kind.io_error()
                .identifier
                .expect("canonical writer IO identity"),
        )
        .with_source(source)
        .with_gpu_gather_retry(crate::GpuGatherRetry::Never)
        .build()
}
pub(in crate::builtins) fn capture(
    kind: TabularWriteKind,
    path: Option<&Path>,
) -> BuiltinResult<TabularFsOwner> {
    TabularFsOwner::capture(path).map_err(|source| io(kind, source))
}
pub(in crate::builtins) fn resolve(
    kind: TabularWriteKind,
    owner: TabularFsOwner,
    path: PathBuf,
) -> BuiltinResult<TabularFsPath> {
    owner.path(path).map_err(|source| io(kind, source))
}
pub(in crate::builtins) async fn publish(
    runtime: Option<&RuntimeContext>,
    kind: TabularWriteKind,
    selected: &TabularFsPath,
    bytes: &[u8],
) -> BuiltinResult<()> {
    let lock = await_io(
        runtime,
        kind.name(),
        write_owner::write_lock_for_path(selected),
    )
    .await?
    .map_err(|source| io(kind, source))?;
    let _guard = await_io(runtime, kind.name(), lock.lock()).await?;
    let mut cooperation = TabularCooperation::new(runtime, kind.name());
    let result =
        write_owner::safe_replace_file(runtime, &mut cooperation, kind.name(), selected, bytes)
            .await;
    result.map_err(|source| {
        let mut error = match source {
            write_owner::StagedWriteError::Control(error) => error,
            write_owner::StagedWriteError::PublicationInterrupted(error, disposition) => {
                write_owner::publication_interruption_error(error, disposition)
            }
            write_owner::StagedWriteError::Publication(source) => io(kind, source),
            write_owner::StagedWriteError::Io(source) => io(kind, source),
        };
        error.gpu_gather_retry = crate::GpuGatherRetry::Never;
        error
    })
}

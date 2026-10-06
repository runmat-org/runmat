use super::*;
use crate::context::RuntimeContext;
use crate::execution::staged_file::{self, StagedCpuCooperation, StagedFileError};
use runmat_filesystem::FsPublicationFailure;

pub(super) async fn write_bytes(
    runtime: Option<&RuntimeContext>,
    selected: &FigureFsPath,
    payload: &[u8],
    builtin: &'static str,
) -> BuiltinResult<()> {
    let mut cooperation = StagedCpuCooperation::new(runtime, builtin);
    let mut stage = staged_file::begin_stage(
        &mut cooperation,
        || selected.check(),
        || selected.begin_staged_replace(),
        "selected filesystem provider cannot stage figure-file replacement",
    )
    .await
    .map_err(|error| map_staged_error(error, builtin))?;
    staged_file::write_stage_bytes(
        runtime,
        &mut cooperation,
        builtin,
        || selected.check(),
        stage.as_mut(),
        payload,
        staged_file::StagedWriteMessages {
            no_progress: "staged figure-file write made no progress",
            excess_progress: "staged figure-file write reported excess progress",
        },
    )
    .await
    .map_err(|error| map_staged_error(error, builtin))?;
    staged_file::finish_stage(
        runtime,
        &mut cooperation,
        builtin,
        || selected.check(),
        stage,
    )
    .await
    .map_err(|error| map_staged_error(error, builtin))
}

fn map_staged_error(error: StagedFileError, builtin: &'static str) -> RuntimeError {
    match error {
        StagedFileError::Io(source) => persistence_error_with_source(
            &ERROR_IO,
            builtin,
            format!("{builtin}: figure-file I/O failed"),
            source,
        ),
        StagedFileError::Control(error) => error,
        StagedFileError::Publication(source) => map_publication(source, builtin),
        StagedFileError::PublicationInterrupted(error, disposition) => {
            staged_file::interruption_error(error, disposition)
        }
    }
}

fn map_publication(source: FsPublicationFailure, builtin: &'static str) -> RuntimeError {
    persistence_error_with_source(
        &ERROR_IO,
        builtin,
        format!("{builtin}: figure-file publication failed"),
        source,
    )
}

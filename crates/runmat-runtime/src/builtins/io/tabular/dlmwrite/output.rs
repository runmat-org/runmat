//! Append streams and exclusive overwrite stages have distinct publication owners.
use super::writer::check_dlm_owner;
use super::*;
use crate::builtins::io::tabular::write_owner;
use crate::context::RuntimeContext;
use runmat_filesystem::FsStagedOutput;

enum Destination {
    Append(File),
    Replace(Box<dyn FsStagedOutput>),
}

pub(super) struct DlmOutput<'a> {
    runtime: Option<&'a RuntimeContext>,
    destination: Destination,
}

impl<'a> DlmOutput<'a> {
    pub(super) async fn begin(
        runtime: Option<&'a RuntimeContext>,
        cooperation: &mut TabularCooperation<'_>,
        selected: &TabularFsPath,
        append: bool,
    ) -> BuiltinResult<Self> {
        let destination = if append {
            let mut options = OpenOptions::new();
            options.create(true).append(true);
            let file = await_io(runtime, BUILTIN_NAME, selected.open_with(&options))
                .await?
                .map_err(|source| {
                    dlmwrite_error_with_source(
                        &DLMWRITE_ERROR_IO,
                        "dlmwrite: unable to open append output",
                        source,
                    )
                })?;
            Destination::Append(file)
        } else {
            Destination::Replace(
                write_owner::begin_stage(cooperation, selected)
                    .await
                    .map_err(staged_error)?,
            )
        };
        Ok(Self {
            runtime,
            destination,
        })
    }

    pub(super) async fn write_bytes(
        &mut self,
        cooperation: &mut TabularCooperation<'_>,
        selected: &TabularFsPath,
        bytes: &[u8],
        total: &mut usize,
    ) -> BuiltinResult<()> {
        match &mut self.destination {
            Destination::Append(file) => {
                write_append_bytes(file, cooperation, selected, bytes, total).await
            }
            Destination::Replace(stage) => {
                let next = total.checked_add(bytes.len()).ok_or_else(|| {
                    dlmwrite_error_with(
                        &DLMWRITE_ERROR_SIZE,
                        "dlmwrite: byte count overflow while writing output",
                    )
                })?;
                write_owner::write_stage_bytes(
                    self.runtime,
                    cooperation,
                    BUILTIN_NAME,
                    selected,
                    stage.as_mut(),
                    bytes,
                )
                .await
                .map_err(staged_error)?;
                *total = next;
                Ok(())
            }
        }
    }

    pub(super) async fn finish(
        self,
        cooperation: &mut TabularCooperation<'_>,
        selected: &TabularFsPath,
    ) -> BuiltinResult<()> {
        match self.destination {
            Destination::Append(mut file) => {
                check_dlm_owner(selected)?;
                await_io(self.runtime, BUILTIN_NAME, file.flush_async())
                    .await?
                    .map_err(|source| {
                        dlmwrite_error_with_source(
                            &DLMWRITE_ERROR_IO,
                            "dlmwrite: failed to flush append output",
                            source,
                        )
                    })?;
                check_dlm_owner(selected)
            }
            Destination::Replace(stage) => {
                write_owner::finish_stage(self.runtime, cooperation, BUILTIN_NAME, selected, stage)
                    .await
                    .map_err(staged_error)
            }
        }
    }
}

fn staged_error(source: write_owner::StagedWriteError) -> RuntimeError {
    match source {
        write_owner::StagedWriteError::Control(error) => error,
        write_owner::StagedWriteError::PublicationInterrupted(error, disposition) => {
            write_owner::publication_interruption_error(error, disposition)
        }
        write_owner::StagedWriteError::Publication(source) => dlmwrite_error_with_source(
            &DLMWRITE_ERROR_IO,
            "dlmwrite: staged publication failed or could not be confirmed",
            source,
        ),
        write_owner::StagedWriteError::Io(source) => dlmwrite_error_with_source(
            &DLMWRITE_ERROR_IO,
            "dlmwrite: unable to stage replacement output",
            source,
        ),
    }
}

async fn write_append_bytes(
    file: &mut File,
    cooperation: &mut TabularCooperation<'_>,
    selected: &TabularFsPath,
    bytes: &[u8],
    total: &mut usize,
) -> BuiltinResult<()> {
    for chunk in bytes.chunks(4096) {
        (*total).checked_add(chunk.len()).ok_or_else(|| {
            dlmwrite_error_with(
                &DLMWRITE_ERROR_SIZE,
                "dlmwrite: byte count overflow while writing output",
            )
        })?;
        let mut remaining = chunk;
        while !remaining.is_empty() {
            check_dlm_owner(selected)?;
            let count = match file.write(remaining) {
                Ok(count) => count,
                Err(source) if source.kind() == io::ErrorKind::Interrupted => {
                    cooperation.completed(1).await?;
                    continue;
                }
                Err(source) => {
                    return Err(dlmwrite_error_with_source(
                        &DLMWRITE_ERROR_IO,
                        format!("dlmwrite: failed to write output ({source})"),
                        source,
                    ));
                }
            };
            if count > remaining.len() {
                let source = io::Error::new(
                    io::ErrorKind::InvalidData,
                    "dlmwrite: provider reported more bytes than requested",
                );
                return Err(dlmwrite_error_with_source(
                    &DLMWRITE_ERROR_IO,
                    source.to_string(),
                    source,
                ));
            }
            if count == 0 {
                let source = io::Error::new(
                    io::ErrorKind::WriteZero,
                    "dlmwrite: provider made no progress writing output",
                );
                return Err(dlmwrite_error_with_source(
                    &DLMWRITE_ERROR_IO,
                    source.to_string(),
                    source,
                ));
            }
            add_written_bytes(total, count)?;
            remaining = &remaining[count..];
            cooperation.completed(count).await?;
        }
    }
    Ok(())
}

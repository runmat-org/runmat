use super::super::{authority::SharedBacking, format_error, MemmapFormat};
use crate::context::RuntimeContext;
use crate::execution::CpuCooperationBudget;
use crate::{build_runtime_error, BuiltinResult, RuntimeError};
use runmat_builtins::{MEMMAPFILE_ERROR_READ_FAILED, MEMMAPFILE_ERROR_WRITE_FAILED};
use runmat_filesystem::FsMappedWriteFailure;
use runmat_value::{MemmapReceiverState, Value};
use std::io::{self, ErrorKind};

const CHUNK: usize = 4096;

fn read_error(source: impl crate::runtime_error::RuntimeErrorSourceBound) -> RuntimeError {
    build_runtime_error("memmapfile: mapped read failed")
        .with_builtin("memmapfile")
        .with_identifier(
            MEMMAPFILE_ERROR_READ_FAILED
                .identifier
                .expect("typed read error"),
        )
        .with_source(source)
        .build()
}

fn write_error(
    source: impl crate::runtime_error::RuntimeErrorSourceBound,
    committed_before_failure: usize,
) -> RuntimeError {
    build_runtime_error(format!(
        "memmapfile: mapped write failed after {committed_before_failure} prior bytes committed"
    ))
    .with_builtin("memmapfile")
    .with_identifier(
        MEMMAPFILE_ERROR_WRITE_FAILED
            .identifier
            .expect("typed write error"),
    )
    .with_source(source)
    .build()
}

async fn checkpoint(budget: &mut Option<CpuCooperationBudget>, bytes: usize) -> BuiltinResult<()> {
    if let Some(budget) = budget {
        budget
            .checkpoint(bytes as u64)
            .await
            .map_err(|source| source.into_runtime_error())?;
    }
    Ok(())
}

pub(super) async fn read_data(
    runtime: Option<&RuntimeContext>,
    state: &MemmapReceiverState,
    backing: &SharedBacking,
) -> BuiltinResult<Value> {
    let format = MemmapFormat::from_receiver(state)?;
    let record_bytes = format.validate_layout(state.repeat)?;
    let mut budget = runtime.map(RuntimeContext::cpu_cooperation);
    checkpoint(&mut budget, 0).await?;
    let mut opened = if let Some(budget) = &budget {
        budget
            .await_cancellable(backing.lock())
            .await
            .map_err(|source| source.into_runtime_error())?
    } else {
        backing.lock().await
    };
    opened.check_current().map_err(read_error)?;
    let offset = u64::try_from(state.offset).map_err(read_error)?;
    let available = opened
        .opened_extent()
        .checked_sub(offset)
        .ok_or_else(|| format_error("memmapfile: Offset exceeds opened extent"))?;
    let available = usize::try_from(available).map_err(read_error)?;
    let records = state
        .repeat
        .unwrap_or(available / record_bytes)
        .min(available / record_bytes);
    let required = records
        .checked_mul(record_bytes)
        .ok_or_else(|| format_error("memmapfile: mapped Data is too large"))?;
    let mut bytes = Vec::new();
    bytes.try_reserve_exact(required).map_err(read_error)?;
    while bytes.len() < required {
        let mut chunk = [0u8; CHUNK];
        let count = (required - bytes.len()).min(CHUNK);
        let position = offset
            .checked_add(bytes.len() as u64)
            .ok_or_else(|| format_error("memmapfile: mapped offset overflow"))?;
        let read = if let Some(budget) = &budget {
            budget
                .await_cancellable(opened.read_at(position, &mut chunk[..count]))
                .await
                .map_err(|source| source.into_runtime_error())?
        } else {
            opened.read_at(position, &mut chunk[..count]).await
        }
        .map_err(read_error)?;
        if read == 0 {
            return Err(read_error(io::Error::new(
                ErrorKind::UnexpectedEof,
                "mapped source truncated after open",
            )));
        }
        bytes.extend_from_slice(&chunk[..read]);
        checkpoint(&mut budget, read).await?;
    }
    opened.check_current().map_err(read_error)?;
    drop(opened);
    let value = format.decode(&bytes, state.repeat)?;
    checkpoint(&mut budget, 0).await?;
    Ok(value)
}

pub(super) async fn write_data(
    runtime: Option<&RuntimeContext>,
    state: &MemmapReceiverState,
    backing: &SharedBacking,
    rhs: &Value,
) -> BuiltinResult<()> {
    if !state.writable {
        return Err(format_error("memmapfile: receiver is read-only"));
    }
    let format = MemmapFormat::from_receiver(state)?;
    let record_bytes = format.validate_layout(state.repeat)?;
    let mut budget = runtime.map(RuntimeContext::cpu_cooperation);
    checkpoint(&mut budget, 0).await?;
    let mut opened = if let Some(budget) = &budget {
        budget
            .await_cancellable(backing.lock())
            .await
            .map_err(|source| source.into_runtime_error())?
    } else {
        backing.lock().await
    };
    opened
        .check_current()
        .map_err(|source| write_error(source, 0))?;
    let offset = u64::try_from(state.offset).map_err(|source| write_error(source, 0))?;
    let available = opened
        .opened_extent()
        .checked_sub(offset)
        .ok_or_else(|| format_error("memmapfile: Offset exceeds opened extent"))?;
    let available = usize::try_from(available).map_err(|source| write_error(source, 0))?;
    let records = state
        .repeat
        .unwrap_or(available / record_bytes)
        .min(available / record_bytes);
    let bytes = format.encode(rhs, records, state.repeat)?;
    let mut committed = 0usize;
    while committed < bytes.len() {
        let end = (committed + CHUNK).min(bytes.len());
        let position = offset
            .checked_add(committed as u64)
            .ok_or_else(|| format_error("memmapfile: mapped offset overflow"))?;
        let outcome = if let Some(budget) = &budget {
            budget
                .await_cancellable(opened.write_at(position, &bytes[committed..end]))
                .await
                .map_err(|source| {
                    write_error(
                        FsMappedWriteFailure::indeterminate(end - committed, source),
                        committed,
                    )
                })?
        } else {
            opened.write_at(position, &bytes[committed..end]).await
        }
        .map_err(|source| write_error(source, committed))?;
        if outcome.accepted_bytes == 0 {
            return Err(write_error(
                io::Error::new(ErrorKind::WriteZero, "mapped provider accepted zero bytes"),
                committed,
            ));
        }
        committed += outcome.accepted_bytes;
        if let Some(budget) = &mut budget {
            budget
                .checkpoint(outcome.accepted_bytes as u64)
                .await
                .map_err(|source| {
                    write_error(
                        FsMappedWriteFailure::committed(committed, source),
                        committed,
                    )
                })?;
        }
    }
    opened
        .check_current()
        .map_err(|source| write_error(source, committed))?;
    if let Some(budget) = &mut budget {
        budget.checkpoint(0).await.map_err(|source| {
            write_error(
                FsMappedWriteFailure::committed(committed, source),
                committed,
            )
        })?;
    }
    Ok(())
}

//! Selected reads and writes preserve the opened backing and cooperation owner.

use super::super::super::format_error;
use super::super::{member, Property};
use super::plan::plan;
use super::{checkpoint, empty_value, mapped_error, receiver, scalar_bytes};
use crate::context::RuntimeContext;
use crate::execution::{RootedValue, RootedValueLease};
use crate::object::indexing::ObjectSubscriptPath;
use crate::BuiltinResult;
use runmat_builtins::{MEMMAPFILE_ERROR_READ_FAILED, MEMMAPFILE_ERROR_WRITE_FAILED};
use runmat_filesystem::FsMappedWriteFailure;
use runmat_value::{MemmapReceiverAuthority, Value};
use std::io::{self, ErrorKind};

const CHUNK: usize = 4096;

pub(in crate::builtins::io::repl_fs::compat::memmapfile::live) async fn read(
    runtime: Option<&RuntimeContext>,
    receiver_value: Value,
    path: ObjectSubscriptPath,
) -> BuiltinResult<RootedValue> {
    let _receiver_root = RootedValueLease::for_values(std::iter::once(&receiver_value))
        .map_err(|error| error.into_runtime_error())?;
    if member(&path.steps()[0])? != Property::Data {
        return Err(format_error("memmapfile: expected Data path"));
    }
    let state = receiver(&receiver_value)?;
    if let MemmapReceiverAuthority::Detached(snapshot) = &state.authority {
        // A detached, read-only receiver has an explicit snapshot authority.
        // Keep its ordinary indexed-read contract without claiming a live map.
        let suffix = ObjectSubscriptPath::new(path.steps()[1..].to_vec())?;
        return match runtime {
            Some(runtime) => {
                crate::object::protocol::read_subscript_path_in_context(
                    runtime,
                    (**snapshot).clone(),
                    suffix,
                    None,
                )
                .await
            }
            None => {
                crate::object::protocol::read_subscript_path((**snapshot).clone(), suffix, None)
                    .await
            }
        };
    }
    let mut budget = runtime.map(RuntimeContext::cpu_cooperation);
    checkpoint(&mut budget, 0).await?;
    let (backing, selected) = plan(runtime, &mut budget, state, &path, false).await?;
    let mut opened = if let Some(budget) = &budget {
        budget
            .await_cancellable(backing.lock())
            .await
            .map_err(|source| source.into_runtime_error())?
    } else {
        backing.lock().await
    };
    let capacity = selected
        .offsets
        .len()
        .checked_mul(selected.unit_bytes)
        .ok_or_else(|| format_error("memmapfile: selected range is too large"))?;
    let mut bytes = Vec::new();
    bytes
        .try_reserve_exact(capacity)
        .map_err(|source| mapped_error(&MEMMAPFILE_ERROR_READ_FAILED, source, 0))?;
    for offset in selected.offsets {
        let mut remaining = selected.unit_bytes;
        let mut position = offset;
        while remaining > 0 {
            let mut chunk = [0u8; CHUNK];
            let size = remaining.min(CHUNK);
            let result = if let Some(budget) = &budget {
                budget
                    .await_cancellable(opened.read_at(position, &mut chunk[..size]))
                    .await
                    .map_err(|source| source.into_runtime_error())?
            } else {
                opened.read_at(position, &mut chunk[..size]).await
            };
            let count =
                result.map_err(|source| mapped_error(&MEMMAPFILE_ERROR_READ_FAILED, source, 0))?;
            if count == 0 {
                return Err(mapped_error(
                    &MEMMAPFILE_ERROR_READ_FAILED,
                    io::Error::new(ErrorKind::UnexpectedEof, "mapped source truncated"),
                    0,
                ));
            }
            bytes.extend_from_slice(&chunk[..count]);
            position += count as u64;
            remaining -= count;
            checkpoint(&mut budget, count).await?;
        }
    }
    opened
        .check_current()
        .map_err(|source| mapped_error(&MEMMAPFILE_ERROR_READ_FAILED, source, 0))?;
    drop(opened);
    let value = if bytes.is_empty() && !selected.records {
        empty_value(selected.format.dtype, selected.shape.clone())?
    } else {
        selected.format.decode(
            &bytes,
            Some(if selected.records {
                capacity / selected.unit_bytes
            } else {
                1
            }),
        )?
    };
    let value = if selected.records {
        match value {
            Value::StructArray(array) if array.shape() != selected.shape.as_slice() => {
                let names =
                    std::iter::once(selected.format.field.clone().expect("structured format"))
                        .chain(
                            selected
                                .format
                                .additional_fields
                                .iter()
                                .map(|field| field.name.clone()),
                        )
                        .collect();
                runmat_value::StructArray::normalize(names, array.into_elements(), selected.shape)
                    .map_err(format_error)?
            }
            value => value,
        }
    } else {
        value
    };
    let value = RootedValue::new(value).map_err(|error| error.into_runtime_error())?;
    checkpoint(&mut budget, 0).await?;
    Ok(value)
}

pub(in crate::builtins::io::repl_fs::compat::memmapfile::live) async fn write(
    runtime: Option<&RuntimeContext>,
    receiver_value: Value,
    path: ObjectSubscriptPath,
    rhs: Value,
) -> BuiltinResult<RootedValue> {
    let _input_roots = RootedValueLease::for_values([&receiver_value, &rhs].into_iter())
        .map_err(|error| error.into_runtime_error())?;
    if member(&path.steps()[0])? != Property::Data {
        return Err(format_error("memmapfile: expected Data path"));
    }
    let state = receiver(&receiver_value)?;
    if !state.writable {
        return Err(format_error("memmapfile: receiver is read-only"));
    }
    let MemmapReceiverAuthority::Opened(_) = &state.authority else {
        return Err(format_error("memmapfile: detached snapshot is read-only"));
    };
    let mut budget = runtime.map(RuntimeContext::cpu_cooperation);
    checkpoint(&mut budget, 0).await?;
    let (backing, selected) = plan(runtime, &mut budget, state, &path, true).await?;
    let rhs = if let Some(budget) = &budget {
        budget
            .await_cancellable(crate::gather_if_needed_async(&rhs))
            .await
            .map_err(|source| source.into_runtime_error())??
    } else {
        crate::gather_if_needed_async(&rhs).await?
    };
    let _gathered_root = RootedValueLease::for_values(std::iter::once(&rhs))
        .map_err(|error| error.into_runtime_error())?;
    let bytes = if !selected.records && selected.unit_bytes == selected.format.dtype.byte_size() {
        if let Some(element) = scalar_bytes(selected.format.dtype, &rhs) {
            let capacity = element
                .len()
                .checked_mul(selected.offsets.len())
                .ok_or_else(|| format_error("memmapfile: selected assignment is too large"))?;
            let mut bytes = Vec::new();
            bytes
                .try_reserve_exact(capacity)
                .map_err(|source| mapped_error(&MEMMAPFILE_ERROR_WRITE_FAILED, source, 0))?;
            for _ in &selected.offsets {
                bytes.extend_from_slice(&element);
                checkpoint(&mut budget, element.len()).await?;
            }
            bytes
        } else {
            selected.format.encode(&rhs, 1, Some(1))?
        }
    } else if selected.records {
        let value = if let Value::StructArray(array) = &rhs {
            if array.shape() != selected.shape.as_slice() {
                return Err(format_error(
                    "memmapfile: assigned record shape does not match selected indices",
                ));
            }
            let names = array.field_names().map(ToOwned::to_owned).collect();
            runmat_value::StructArray::normalize(
                names,
                array.elements().map(|element| element.to_owned()).collect(),
                vec![selected.offsets.len(), 1],
            )
            .map_err(format_error)?
        } else {
            rhs.clone()
        };
        selected
            .format
            .encode(&value, selected.offsets.len(), Some(selected.offsets.len()))?
    } else {
        selected.format.encode(&rhs, 1, Some(1))?
    };
    let expected = selected
        .offsets
        .len()
        .checked_mul(selected.unit_bytes)
        .ok_or_else(|| format_error("memmapfile: selected range is too large"))?;
    if bytes.len() != expected {
        return Err(format_error(
            "memmapfile: assigned Data does not fit selected ranges",
        ));
    }
    let mut opened = if let Some(budget) = &budget {
        budget
            .await_cancellable(backing.lock())
            .await
            .map_err(|source| source.into_runtime_error())?
    } else {
        backing.lock().await
    };
    let mut committed = 0usize;
    for offset in selected.offsets {
        let mut unit_written = 0usize;
        while unit_written < selected.unit_bytes {
            let end = (unit_written + CHUNK).min(selected.unit_bytes);
            let slice = &bytes[committed + unit_written..committed + end];
            let position = offset
                .checked_add(unit_written as u64)
                .ok_or_else(|| format_error("memmapfile: selected offset overflow"))?;
            let result = if let Some(budget) = &budget {
                budget
                    .await_cancellable(opened.write_at(position, slice))
                    .await
                    .map_err(|source| {
                        mapped_error(
                            &MEMMAPFILE_ERROR_WRITE_FAILED,
                            FsMappedWriteFailure::indeterminate(slice.len(), source),
                            committed + unit_written,
                        )
                    })?
            } else {
                opened.write_at(position, slice).await
            };
            let outcome = result.map_err(|source| {
                mapped_error(
                    &MEMMAPFILE_ERROR_WRITE_FAILED,
                    source,
                    committed + unit_written,
                )
            })?;
            if outcome.accepted_bytes == 0 {
                return Err(mapped_error(
                    &MEMMAPFILE_ERROR_WRITE_FAILED,
                    io::Error::new(ErrorKind::WriteZero, "mapped provider accepted zero bytes"),
                    committed + unit_written,
                ));
            }
            unit_written += outcome.accepted_bytes;
            if let Some(budget) = &mut budget {
                budget
                    .checkpoint(outcome.accepted_bytes as u64)
                    .await
                    .map_err(|source| {
                        mapped_error(
                            &MEMMAPFILE_ERROR_WRITE_FAILED,
                            FsMappedWriteFailure::committed(committed + unit_written, source),
                            committed + unit_written,
                        )
                    })?;
            }
        }
        committed += selected.unit_bytes;
    }
    opened
        .check_current()
        .map_err(|source| mapped_error(&MEMMAPFILE_ERROR_WRITE_FAILED, source, committed))?;
    if let Some(budget) = &mut budget {
        budget.checkpoint(0).await.map_err(|source| {
            mapped_error(
                &MEMMAPFILE_ERROR_WRITE_FAILED,
                FsMappedWriteFailure::committed(committed, source),
                committed,
            )
        })?;
    }
    RootedValue::new(receiver_value).map_err(|error| error.into_runtime_error())
}

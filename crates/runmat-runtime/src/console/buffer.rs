//! Canonical thread-buffer ownership, admission, and stable batch publication.

use super::{
    console_admission, console_text_error, with_console_state, CapturedConsoleEntries,
    ConsoleEntry, ConsoleText,
};
use crate::context::host_allocation::{
    HostAllocationError, HostAllocationPlan, HostAllocationPolicy, HostAllocationReceipt,
    HostTypedWorkspace,
};
use crate::context::overlay_storage::AdmittedOverlayVec;

pub(super) fn append_buffer_entry(
    entry: ConsoleEntry,
    runtime: Option<&crate::context::RuntimeContext>,
) -> Result<(), crate::RuntimeError> {
    let Some(runtime) = runtime else {
        return with_console_state(|state| state.buffer.push_unbounded(entry)).map_err(|_| {
            crate::runtime_error::semantic_error(
                "RunMat:console:Buffer",
                "console buffer generation exhausted",
            )
        });
    };
    ensure_buffer_policy(runtime)?;
    let snapshot = with_console_state(|state| state.buffer.snapshot_push())
        .map_err(|error| console_admission(Some(runtime), error))?;
    let prepared = AdmittedOverlayVec::<ConsoleEntry>::prepare_push(runtime, snapshot)
        .map_err(|error| console_admission(Some(runtime), error))?;
    let prepared = prepared
        .checkpoint(runtime)
        .map_err(|(error, _)| console_admission(Some(runtime), error))?;
    let result = with_console_state(|state| state.buffer.install_push(prepared, entry));
    match result {
        Ok(retirement) => {
            drop(retirement);
            Ok(())
        }
        Err(failure) => {
            let error = failure.cause;
            drop(failure);
            Err(console_admission(Some(runtime), error))
        }
    }
}

fn ensure_buffer_policy(
    runtime: &crate::context::RuntimeContext,
) -> Result<(), crate::RuntimeError> {
    let policy = HostAllocationReceipt::policy(Some(runtime))
        .map_err(|error| console_admission(Some(runtime), error))?;
    if policy != HostAllocationPolicy::Managed {
        return Ok(());
    }
    let needs_managed =
        with_console_state(|state| matches!(&state.buffer, AdmittedOverlayVec::Unbounded { .. }));
    if !needs_managed {
        return Ok(());
    }
    if !with_console_state(|state| state.buffer.is_empty()) {
        return Err(console_admission(
            Some(runtime),
            HostAllocationError::MissingManagedOwner,
        ));
    }
    let receipt = HostAllocationReceipt::admit(
        Some(runtime),
        HostAllocationPlan::new(0, 0).map_err(|error| console_admission(Some(runtime), error))?,
    )
    .map_err(|error| console_admission(Some(runtime), error))?
    .ok_or_else(|| console_admission(Some(runtime), HostAllocationError::MissingManagedOwner))?;
    let managed = AdmittedOverlayVec::managed(runtime, &receipt)
        .map_err(|error| console_admission(Some(runtime), error))?;
    receipt
        .checkpoint(runtime)
        .map_err(|error| console_admission(Some(runtime), error))?;
    let mut managed = Some(managed);
    let retired = with_console_state(|state| {
        if matches!(&state.buffer, AdmittedOverlayVec::Unbounded { .. }) && state.buffer.is_empty()
        {
            Some(std::mem::replace(
                &mut state.buffer,
                managed.take().expect("prepared owner"),
            ))
        } else {
            None
        }
    });
    drop(managed);
    drop(retired);
    Ok(())
}

/// Clears the per-thread console buffer. Call this before execution begins so
/// each run only returns fresh output.
pub fn reset_thread_buffer() {
    let retired = with_console_state(|state| {
        (
            std::mem::replace(&mut state.buffer, AdmittedOverlayVec::unbounded()),
            state.last_value_output.take(),
        )
    });
    drop(retired);
}

/// Drain (and return) the buffered console entries for the current thread.
pub fn take_thread_buffer() -> CapturedConsoleEntries {
    CapturedConsoleEntries::from_storage(with_console_state(|state| {
        std::mem::replace(&mut state.buffer, AdmittedOverlayVec::unbounded())
    }))
}

/// Append console entries captured on another execution thread without
/// forwarding them again to live stream listeners.
pub fn append_thread_buffer(
    entries: impl Into<CapturedConsoleEntries>,
) -> Result<(), crate::RuntimeError> {
    let mut source = entries.into().into_storage();
    if source.is_empty() {
        return Ok(());
    }
    let runtime = crate::context::legacy::active();
    let Some(runtime) = runtime.as_ref() else {
        with_console_state(|state| {
            let AdmittedOverlayVec::Unbounded { values, .. } = &mut state.buffer else {
                return Err(());
            };
            values.extend(source);
            values.sort_by_key(|entry| entry.timestamp_ms);
            Ok(())
        })
        .map_err(|_| {
            crate::runtime_error::semantic_error(
                "RunMat:console:Buffer",
                "managed buffer has no active context",
            )
        })?;
        return Ok(());
    };
    ensure_buffer_policy(runtime)?;
    let policy = HostAllocationReceipt::policy(Some(runtime))
        .map_err(|error| console_admission(Some(runtime), error))?;
    let adoption_bytes = if policy == HostAllocationPolicy::Managed {
        source
            .as_slice()
            .iter()
            .try_fold(0_u64, |total, entry| {
                let bytes = entry.text.adoption_retained_bytes()?;
                total
                    .checked_add(bytes)
                    .ok_or(HostAllocationError::Overflow)
            })
            .map_err(|error| console_admission(Some(runtime), error))?
    } else {
        0
    };
    let adoption_receipt = if adoption_bytes == 0 {
        None
    } else {
        let receipt = HostAllocationReceipt::admit(
            Some(runtime),
            HostAllocationPlan::new(0, adoption_bytes)
                .map_err(|error| console_admission(Some(runtime), error))?,
        )
        .map_err(|error| console_admission(Some(runtime), error))?
        .ok_or_else(|| {
            console_admission(Some(runtime), HostAllocationError::MissingManagedOwner)
        })?;
        let permit = receipt
            .output_permit(runtime)
            .map_err(|error| console_admission(Some(runtime), error))?;
        for entry in source.as_mut_slice() {
            let text = std::mem::replace(&mut entry.text, ConsoleText::from(String::new()));
            let adopted = text.try_adopt(&permit);
            receipt
                .checkpoint(runtime)
                .map_err(|error| console_admission(Some(runtime), error))?;
            entry.text = adopted.map_err(|error| console_text_error(runtime, error))?;
        }
        receipt
            .checkpoint(runtime)
            .map_err(|error| console_admission(Some(runtime), error))?;
        Some(receipt)
    };
    let snapshot = with_console_state(|state| state.buffer.snapshot_push())
        .map_err(|error| console_admission(Some(runtime), error))?;
    let current_len = snapshot.len();
    let total = current_len
        .checked_add(source.len())
        .ok_or_else(|| console_admission(Some(runtime), HostAllocationError::Overflow))?;
    let mut ordering = if policy == HostAllocationPolicy::Managed {
        let quote = HostTypedWorkspace::<usize>::quoted_bytes(total)
            .and_then(|left| {
                HostTypedWorkspace::<u64>::quoted_bytes(total)
                    .and_then(|right| left.checked_add(right).ok_or(HostAllocationError::Overflow))
            })
            .map_err(|error| console_admission(Some(runtime), error))?;
        let receipt = HostAllocationReceipt::admit(
            Some(runtime),
            HostAllocationPlan::new(quote, 0)
                .map_err(|error| console_admission(Some(runtime), error))?,
        )
        .map_err(|error| console_admission(Some(runtime), error))?
        .ok_or_else(|| {
            console_admission(Some(runtime), HostAllocationError::MissingManagedOwner)
        })?;
        let mut timestamps = receipt
            .temporary_typed::<u64>(runtime, total)
            .map_err(|error| console_admission(Some(runtime), error))?;
        let mut indices = receipt
            .temporary_typed::<usize>(runtime, total)
            .map_err(|error| console_admission(Some(runtime), error))?;
        with_console_state(|state| {
            for (slot, entry) in timestamps.as_mut_slice()[..current_len]
                .iter_mut()
                .zip(state.buffer.as_slice())
            {
                *slot = entry.timestamp_ms;
            }
        });
        for (slot, entry) in timestamps.as_mut_slice()[current_len..]
            .iter_mut()
            .zip(source.as_slice())
        {
            *slot = entry.timestamp_ms;
        }
        for (index, slot) in indices.as_mut_slice().iter_mut().enumerate() {
            *slot = index;
        }
        indices.as_mut_slice().sort_unstable_by(|left, right| {
            timestamps.as_slice()[*left]
                .cmp(&timestamps.as_slice()[*right])
                .then(left.cmp(right))
        });
        receipt
            .checkpoint(runtime)
            .map_err(|error| console_admission(Some(runtime), error))?;
        Some((receipt, timestamps, indices))
    } else {
        None
    };
    let prepared =
        AdmittedOverlayVec::<ConsoleEntry>::prepare_append(runtime, snapshot, source.len())
            .map_err(|error| console_admission(Some(runtime), error))?;
    let prepared = prepared
        .checkpoint(runtime)
        .map_err(|(error, _)| console_admission(Some(runtime), error))?;
    let result = with_console_state(|state| {
        let result = state.buffer.install_append(prepared, source);
        if result.is_ok() {
            let values = state.buffer.as_mut_slice();
            if let Some((_, _, indices)) = ordering.as_mut() {
                let indices = indices.as_mut_slice();
                for start in 0..values.len() {
                    let mut current = start;
                    while indices[current] != start {
                        let next = indices[current];
                        values.swap(current, next);
                        indices[current] = current;
                        current = next;
                    }
                    indices[current] = current;
                }
            } else {
                values.sort_by_key(|entry| entry.timestamp_ms);
            }
        }
        result
    });
    match result {
        Ok(retirement) => {
            drop(retirement);
            drop(adoption_receipt);
            Ok(())
        }
        Err((error, prepared, source)) => {
            drop((prepared, source));
            drop(adoption_receipt);
            Err(console_admission(Some(runtime), error))
        }
    }
}

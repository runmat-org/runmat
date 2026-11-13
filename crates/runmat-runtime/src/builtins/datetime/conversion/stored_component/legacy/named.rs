//! Exact named month/day Cell output with cooperative admitted construction.
use super::super::{allocation_error, input::SerialView, never_retry, Work};
use crate::builtins::datetime::{
    component_builtins::weekday_name, conversion::month_name, naive_from_datenum,
};
use crate::context::host_allocation::{
    HostAllocationError, HostAllocationPlan, HostAllocationReceipt,
};
use crate::context::RuntimeContext;
use crate::BuiltinResult;
use runmat_value::{
    CellArray, CharArray, ExactArrayInitializationError, ExactArrayStorage, OutputListStorage,
    SharedShape, Value,
};

#[derive(Clone, Copy)]
pub(super) enum NamedMode {
    Month { short: bool },
    Day { short: bool },
}

fn name_at(source: &SerialView<'_>, index: usize, mode: NamedMode) -> BuiltinResult<&'static str> {
    let serial = source.at(index);
    if !serial.is_finite() {
        return Ok("");
    }
    let naive = naive_from_datenum(serial).map_err(never_retry)?;
    Ok(match mode {
        NamedMode::Month { short } => month_name(chrono::Datelike::month(&naive), short),
        NamedMode::Day { short } => weekday_name(chrono::Datelike::weekday(&naive), short),
    })
}

fn numeric(
    runtime: &RuntimeContext,
    work: &Work<'_>,
    source: runmat_value::ExactHostNumericAllocationError,
) -> crate::RuntimeError {
    allocation_error(
        Some(runtime),
        work.label,
        HostAllocationError::Numeric(source),
    )
}

fn add_bytes(
    runtime: &RuntimeContext,
    work: &Work<'_>,
    physical: &mut u64,
    claims: &mut usize,
    bytes: usize,
    count: usize,
) -> BuiltinResult<()> {
    *physical = physical
        .checked_add(u64::try_from(bytes).map_err(|_| {
            allocation_error(Some(runtime), work.label, HostAllocationError::Overflow)
        })?)
        .ok_or_else(|| {
            allocation_error(Some(runtime), work.label, HostAllocationError::Overflow)
        })?;
    *claims = claims.checked_add(count).ok_or_else(|| {
        allocation_error(Some(runtime), work.label, HostAllocationError::Overflow)
    })?;
    Ok(())
}

pub(super) async fn map_named(
    runtime: &RuntimeContext,
    work: &mut Work<'_>,
    source: &SerialView<'_>,
    shape: &[usize],
    mode: NamedMode,
) -> BuiltinResult<Value> {
    let len = source.len();
    let mut physical = 0_u64;
    let mut claims = 0_usize;
    add_bytes(
        runtime,
        work,
        &mut physical,
        &mut claims,
        OutputListStorage::requested_bytes(len).map_err(|source| numeric(runtime, work, source))?,
        usize::from(len != 0),
    )?;
    add_bytes(
        runtime,
        work,
        &mut physical,
        &mut claims,
        SharedShape::requested_bytes_for_rank(shape.len())
            .map_err(|source| numeric(runtime, work, source))?,
        1,
    )?;
    for index in 0..len {
        let chars = name_at(source, index, mode)?.encode_utf16().count();
        add_bytes(
            runtime,
            work,
            &mut physical,
            &mut claims,
            ExactArrayStorage::<runmat_value::Utf16CodeUnit>::requested_bytes(chars)
                .map_err(|source| numeric(runtime, work, source))?,
            usize::from(chars != 0),
        )?;
        add_bytes(
            runtime,
            work,
            &mut physical,
            &mut claims,
            SharedShape::requested_bytes_for_rank(2)
                .map_err(|source| numeric(runtime, work, source))?,
            1,
        )?;
        work.charge(1).await?;
    }
    work.flush().await?;
    let retained = HostAllocationPlan::retained_exact_output_bytes(physical, claims)
        .map_err(|source| allocation_error(Some(runtime), work.label, source))?;
    let plan = HostAllocationPlan::new(0, retained)
        .map_err(|source| allocation_error(Some(runtime), work.label, source))?;
    let receipt = HostAllocationReceipt::admit(Some(runtime), plan)
        .map_err(|source| allocation_error(Some(runtime), work.label, source))?
        .ok_or_else(|| {
            allocation_error(
                Some(runtime),
                work.label,
                HostAllocationError::StaleAuthority,
            )
        })?;
    work.check()?;
    let permit = receipt
        .output_permit(runtime)
        .map_err(|source| allocation_error(Some(runtime), work.label, source))?;
    let output_shape = SharedShape::new_admitted_from_slice(shape, &permit)
        .map_err(|source| numeric(runtime, work, source))?;
    let mut outputs = OutputListStorage::try_begin(len, &permit)
        .map_err(|source| numeric(runtime, work, source))?;
    for index in 0..len {
        let name = name_at(source, index, mode)?;
        let count = name.encode_utf16().count();
        let mut units = name.encode_utf16();
        let data = ExactArrayStorage::<runmat_value::Utf16CodeUnit>::try_initialize(
            count,
            &permit,
            |_slot| {
                Ok::<runmat_value::Utf16CodeUnit, crate::RuntimeError>(
                    runmat_value::Utf16CodeUnit::from_raw(
                        units.next().expect("name slot in bounds"),
                    ),
                )
            },
        )
        .map_err(|source| match source {
            ExactArrayInitializationError::Allocation(source) => numeric(runtime, work, source),
            ExactArrayInitializationError::Element(source) => source,
        })?;
        let char_shape = SharedShape::new_admitted_from_slice(&[1, count], &permit)
            .map_err(|source| numeric(runtime, work, source))?;
        let child = CharArray::from_admitted(data, char_shape).map_err(|source| {
            never_retry(crate::builtins::datetime::datetime_error(format!(
                "{}: {source}",
                work.label,
            )))
        })?;
        outputs.try_push(Value::CharArray(child)).map_err(|_| {
            allocation_error(Some(runtime), work.label, HostAllocationError::Overflow)
        })?;
        work.charge(1).await?;
        receipt
            .checkpoint(runtime)
            .map_err(|source| allocation_error(Some(runtime), work.label, source))?;
    }
    work.flush().await?;
    work.check()?;
    let data = outputs
        .try_finish()
        .map_err(|_| allocation_error(Some(runtime), work.label, HostAllocationError::Overflow))?;
    receipt
        .checkpoint(runtime)
        .map_err(|source| allocation_error(Some(runtime), work.label, source))?;
    let cell = CellArray::from_admitted(data, output_shape).map_err(|source| {
        never_retry(crate::builtins::datetime::datetime_error(format!(
            "{}: {source}",
            work.label,
        )))
    })?;
    work.check()?;
    Ok(Value::Cell(cell))
}

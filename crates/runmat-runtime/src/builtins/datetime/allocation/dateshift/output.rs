//! Preserve the original datetime format without an unclaimed String clone.
use crate::builtins::datetime::{
    capabilities::{DATETIME_CLASS, DEFAULT_DATETIME_FORMAT, FORMAT_FIELD, SERIAL_FIELD},
    conversion::{CalendarWork, PreparedControl},
    datetime_error,
};
use crate::context::host_allocation::{
    host_allocation_runtime_error, HostAllocationError, HostAllocationPlan, HostAllocationReceipt,
};
use crate::context::RuntimeContext;
use crate::{BuiltinResult, GpuGatherRetry, RuntimeError};
use runmat_value::{Tensor, Utf8Text, Value};
use std::num::NonZeroUsize;

enum FormatOwner<'a> {
    Borrowed(&'a Value),
    Owned(Value),
}

impl FormatOwner<'_> {
    fn value(&self) -> &Value {
        match self {
            Self::Borrowed(value) => value,
            Self::Owned(value) => value,
        }
    }
}

fn error(runtime: &RuntimeContext, source: HostAllocationError) -> RuntimeError {
    let mut error = host_allocation_runtime_error(
        Some(runtime),
        "RunMat:dateshift:CalendarWorkspace",
        "dateshift: format output allocation failed",
        source,
    );
    error.gpu_gather_retry = GpuGatherRetry::Never;
    error
}

fn numeric(
    runtime: &RuntimeContext,
    source: runmat_value::ExactHostNumericAllocationError,
) -> RuntimeError {
    error(runtime, HostAllocationError::Numeric(source))
}

async fn exact_format<'a>(
    runtime: &RuntimeContext,
    work: &mut CalendarWork<'_>,
    input: &'a Value,
) -> BuiltinResult<FormatOwner<'a>> {
    work.check()?;
    let Value::Object(object) = input else {
        return Err(datetime_error("datetime: expected a datetime value"));
    };
    let property = object.property(FORMAT_FIELD);
    if let Some(value @ Value::String(text)) = property {
        super::super::super::nonmissing_text(text, "Format")?;
        return Ok(FormatOwner::Borrowed(value));
    }
    // Character-format conversion uses the existing admitted control owner;
    // both Unicode encoding and the final UTF-8 copy can suspend cooperatively.
    let control = if matches!(property, Some(Value::CharArray(array)) if array.rows == 1) {
        Some(PreparedControl::new(runtime, work, property).await?)
    } else {
        None
    };
    let source = match property {
        Some(Value::StringArray(array)) if array.data().len() == 1 => {
            super::super::super::nonmissing_text(&array.data()[0], "Format")?
        }
        Some(Value::CharArray(array)) if array.rows == 1 => control
            .as_ref()
            .and_then(PreparedControl::text)
            .expect("prepared character format"),
        _ => DEFAULT_DATETIME_FORMAT,
    };
    let bytes = Utf8Text::requested_bytes_for_capacity(source.len())
        .map_err(|source| numeric(runtime, source))?;
    let retained = HostAllocationPlan::retained_exact_output_bytes(
        u64::try_from(bytes).map_err(|_| error(runtime, HostAllocationError::Overflow))?,
        usize::from(bytes != 0),
    )
    .map_err(|source| error(runtime, source))?;
    let plan = HostAllocationPlan::new(0, retained).map_err(|source| error(runtime, source))?;
    let receipt = HostAllocationReceipt::admit(Some(runtime), plan)
        .map_err(|source| error(runtime, source))?
        .ok_or_else(|| error(runtime, HostAllocationError::StaleAuthority))?;
    work.check()?;
    let permit = receipt
        .output_permit(runtime)
        .map_err(|source| error(runtime, source))?;
    let mut copy = Utf8Text::try_begin_copy_exact(source, &permit)
        .map_err(|source| numeric(runtime, source))?;
    while !copy.is_complete() {
        let completed = copy.advance(NonZeroUsize::new(4096).unwrap());
        work.cooperate(completed as u64).await?;
        receipt
            .checkpoint(runtime)
            .map_err(|source| error(runtime, source))?;
    }
    let text = copy.try_finish().expect("completed exact datetime format");
    receipt
        .checkpoint(runtime)
        .map_err(|source| error(runtime, source))?;
    work.check()?;
    // The completed text retains its physical claim; unused stage headroom
    // need not outlive the exact copy or inflate the returned format owner.
    Ok(FormatOwner::Owned(Value::String(text.into())))
}

pub(super) async fn publish(
    runtime: &RuntimeContext,
    work: &mut CalendarWork<'_>,
    input: &Value,
    serials: Tensor,
) -> BuiltinResult<Value> {
    let format = exact_format(runtime, work, input).await?;
    crate::builtins::datetime::ensure_datetime_class_registered();
    let serial = Value::Tensor(serials);
    let fields = [(SERIAL_FIELD, &serial), (FORMAT_FIELD, format.value())];
    let object = crate::execution::object::construct_object_from_borrowed_fields(
        Some(runtime),
        DATETIME_CLASS.display_name(),
        fields.into_iter(),
    )?;
    work.check()?;
    Ok(object.into_value_and_lease().0)
}

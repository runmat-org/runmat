//! Selected numeric component mapping over borrowed native serial storage.
//! Text and named-cell forms require their own exact text/collection owners.
use super::{allocation_error, input::SerialView, never_retry, output, Work};
use crate::builtins::datetime::{
    allocation::numeric_view::NumericView,
    capabilities::{
        DATETIME_GPU_INPUT_EXTENSION, DATETIME_LOGICAL_INPUT_EXTENSION,
        HOUR_TYPED_LEGACY_NUMERIC_EXTENSION, MINUTE_TYPED_LEGACY_NUMERIC_EXTENSION,
        MONTH_TYPED_LEGACY_NUMERIC_EXTENSION, YEAR_TYPED_LEGACY_NUMERIC_EXTENSION,
    },
    datetime_error, datetime_gather_error, datetime_utf16_error, is_datetime_object,
    is_typed_legacy_numeric, naive_from_datenum,
    text_parsing::{encode_utf16_chunk, parse_legacy_component_text_borrowed, utf16_chunk_end},
};
use crate::call::FunctionArgumentView;
use crate::context::host_allocation::{
    HostAllocationError, HostAllocationPlan, HostAllocationReceipt, HostTypedWorkspace,
};
use crate::context::RuntimeContext;
use crate::execution::RootedValueLease;
use crate::BuiltinResult;
use chrono::{Datelike, NaiveDateTime, Timelike};
use runmat_value::Value;
#[path = "legacy/control.rs"]
mod control;
#[path = "legacy/named.rs"]
mod named;
pub(in crate::builtins::datetime) use control::PreparedControl;
use named::NamedMode;

#[derive(Clone, Copy)]
pub(in crate::builtins::datetime) enum ComponentKind {
    Year,
    Month,
    Day,
    Hour,
    Minute,
}

impl ComponentKind {
    fn label(self) -> &'static str {
        match self {
            Self::Year => "year",
            Self::Month => "month",
            Self::Day => "day",
            Self::Hour => "hour",
            Self::Minute => "minute",
        }
    }

    fn typed_extension(self) -> Option<&'static runmat_builtins::BuiltinExtensionDescriptor> {
        match self {
            Self::Year => Some(&YEAR_TYPED_LEGACY_NUMERIC_EXTENSION),
            Self::Month => Some(&MONTH_TYPED_LEGACY_NUMERIC_EXTENSION),
            Self::Day => None,
            Self::Hour => Some(&HOUR_TYPED_LEGACY_NUMERIC_EXTENSION),
            Self::Minute => Some(&MINUTE_TYPED_LEGACY_NUMERIC_EXTENSION),
        }
    }

    fn invalid_stored_option(self, option: &str) -> crate::RuntimeError {
        let message = match self {
            Self::Year => format!("year: unsupported year type '{option}'"),
            Self::Month => format!("month: unsupported month type '{option}'"),
            Self::Day => format!("day: unsupported day type '{option}'"),
            Self::Hour | Self::Minute => format!(
                "{}: legacy date format is not supported for datetime input",
                self.label(),
            ),
        };
        never_retry(datetime_error(message))
    }
}

#[derive(Clone, Copy)]
enum NumericMode {
    YearIso,
    YearGregorian,
    Month,
    DayOfMonth,
    DayOfWeek,
    IsoDayOfWeek,
    DayOfYear,
    Hour,
    Minute,
}

enum SelectedMode {
    Numeric(NumericMode),
    Named(NamedMode),
}

impl NumericMode {
    fn date_only_format(self) -> bool {
        match self {
            Self::DayOfMonth | Self::DayOfWeek | Self::IsoDayOfWeek | Self::DayOfYear => true,
            Self::YearIso | Self::YearGregorian | Self::Month | Self::Hour | Self::Minute => false,
        }
    }

    fn extract(self, naive: &NaiveDateTime) -> f64 {
        match self {
            Self::YearIso => f64::from(naive.year()),
            Self::YearGregorian => {
                let year = naive.year();
                if year > 0 {
                    f64::from(year)
                } else {
                    f64::from(1 - year)
                }
            }
            Self::Month => f64::from(naive.month()),
            Self::DayOfMonth => f64::from(naive.day()),
            Self::DayOfWeek => f64::from(naive.weekday().num_days_from_sunday() + 1),
            Self::IsoDayOfWeek => f64::from(naive.weekday().number_from_monday()),
            Self::DayOfYear => f64::from(naive.ordinal()),
            Self::Hour => f64::from(naive.hour()),
            Self::Minute => f64::from(naive.minute()),
        }
    }
}

enum Source<'a> {
    Stored(SerialView<'a>),
    Numeric(NumericView<'a>),
    Text(&'a Value),
}

impl Source<'_> {
    fn len(&self) -> usize {
        match self {
            Self::Stored(source) => source.len(),
            Self::Numeric(source) => source.len(),
            Self::Text(Value::String(_)) => 1,
            Self::Text(Value::StringArray(array)) => array.data().len(),
            Self::Text(Value::CharArray(array)) => array.rows,
            Self::Text(_) => unreachable!("text source classified at admission"),
        }
    }

    fn at(&self, index: usize) -> f64 {
        match self {
            Self::Stored(source) => source.at(index),
            Self::Numeric(source) => source.at(index),
            Self::Text(_) => unreachable!("text source is parsed from borrowed UTF-8"),
        }
    }

    fn stored(&self) -> bool {
        matches!(self, Self::Stored(_))
    }
}

fn text_scratch(value: &Value, label: &str) -> BuiltinResult<usize> {
    match value {
        Value::String(_) => Ok(0),
        Value::StringArray(array) => {
            let extent = array
                .shape
                .iter()
                .try_fold(1usize, |n, dim| n.checked_mul(*dim));
            if extent != Some(array.data().len()) {
                return Err(never_retry(datetime_error(format!(
                    "{label}: string shape and payload disagree"
                ))));
            }
            Ok(0)
        }
        Value::CharArray(array) => {
            array
                .validate_layout()
                .map_err(|source| never_retry(datetime_error(format!("{label}: {source}"))))?;
            if array.rows == 0 {
                Ok(0)
            } else {
                array
                    .cols
                    .checked_mul(4)
                    .map(|bytes| bytes.max(1))
                    .ok_or_else(|| {
                        never_retry(datetime_error(format!(
                            "{label}: character row extent exceeds platform limits"
                        )))
                    })
            }
        }
        _ => unreachable!("text source classified at admission"),
    }
}

async fn component_at(
    source: &Source<'_>,
    mode: NumericMode,
    index: usize,
    runtime: &RuntimeContext,
    work: &mut Work<'_>,
    format: Option<&str>,
    row_scratch: Option<&mut [u8]>,
) -> BuiltinResult<f64> {
    let Source::Text(value) = source else {
        let serial = source.at(index);
        return if !source.stored() && !serial.is_finite() {
            Ok(f64::NAN)
        } else {
            naive_from_datenum(serial)
                .map(|naive| mode.extract(&naive))
                .map_err(never_retry)
        };
    };
    let text = match value {
        Value::String(text) => {
            crate::builtins::datetime::nonmissing_text(text, work.label).map_err(never_retry)?
        }
        Value::StringArray(array) => {
            crate::builtins::datetime::nonmissing_text(&array.data()[index], work.label)
                .map_err(never_retry)?
        }
        Value::CharArray(array) => {
            let scratch = row_scratch.expect("nonempty character rows have admitted workspace");
            let row = &array.data()[index * array.cols..(index + 1) * array.cols];
            let mut end = 0;
            let mut source = 0;
            while source < row.len() {
                let next = utf16_chunk_end(row, source);
                end += encode_utf16_chunk(&row[source..next], source, &mut scratch[end..])
                    .map_err(|cause| never_retry(datetime_utf16_error(cause)))?;
                if next < row.len() {
                    work.cooperate(u64::try_from(next - source).unwrap_or(u64::MAX))
                        .await?;
                }
                source = next;
            }
            std::str::from_utf8(&scratch[..end])
                .expect("UTF-8 character row was encoded in the admitted workspace")
        }
        _ => unreachable!("text source classified at admission"),
    };
    parse_legacy_component_text_borrowed(
        Some(runtime),
        text,
        format,
        work.label,
        mode.date_only_format(),
    )
    .map(|parsed| mode.extract(&parsed))
    .map_err(never_retry)
}

fn convert_legacy_format(source: &str, target: &mut [u8]) -> usize {
    let patterns: &[(&[u8], &[u8])] = &[
        (b".fff", b"%.3f"),
        (b"yyyy", b"%Y"),
        (b"mmmm", b"%B"),
        (b"mmm", b"%b"),
        (b"HH", b"%H"),
        (b"hh", b"%H"),
        (b"MM", b"%M"),
        (b"ss", b"%S"),
        (b"mm", b"%m"),
        (b"dd", b"%d"),
    ];
    let bytes = source.as_bytes();
    let mut read = 0;
    let mut write = 0;
    while read < bytes.len() {
        if let Some((from, to)) = patterns
            .iter()
            .find(|(from, _)| bytes[read..].starts_with(from))
        {
            target[write..write + to.len()].copy_from_slice(to);
            read += from.len();
            write += to.len();
        } else {
            target[write] = bytes[read];
            read += 1;
            write += 1;
        }
    }
    write
}

/// Capture the original owner before any input gather. Returns `None` only
/// for an initially unrestricted invocation.
pub(in crate::builtins::datetime) async fn map_legacy_component(
    runtime: Option<&RuntimeContext>,
    kind: ComponentKind,
    value: &Value,
    rest: FunctionArgumentView<'_>,
) -> BuiltinResult<Option<Value>> {
    let owner = HostAllocationReceipt::capture_owner(runtime)
        .map_err(|source| allocation_error(runtime, kind.label(), source))?;
    let Some(owner) = owner else {
        return Ok(None);
    };
    let runtime = runtime.ok_or_else(|| {
        allocation_error(runtime, kind.label(), HostAllocationError::StaleAuthority)
    })?;
    let mut work = Work::with_owner(Some(runtime), kind.label(), Some(owner))?;
    let label = kind.label();
    if rest.len() > 1 {
        return Err(never_retry(datetime_error(format!(
            "{label}: expected at most one component type or legacy date format"
        ))));
    }
    if matches!(value, Value::GpuTensor(_)) {
        crate::compatibility::ensure_builtin_extension_enabled(
            &DATETIME_GPU_INPUT_EXTENSION,
            label,
        )
        .map_err(never_retry)?;
    }
    if matches!(value, Value::Bool(_) | Value::LogicalArray(_)) {
        crate::compatibility::ensure_builtin_extension_enabled(
            &DATETIME_LOGICAL_INPUT_EXTENSION,
            label,
        )
        .map_err(never_retry)?;
    }
    if is_typed_legacy_numeric(value) {
        if let Some(extension) = kind.typed_extension() {
            crate::compatibility::ensure_builtin_extension_enabled(extension, label)
                .map_err(never_retry)?;
        }
    }
    if matches!(kind, ComponentKind::Day) {
        if rest
            .iter()
            .any(|value| matches!(value, Value::GpuTensor(_)))
        {
            crate::compatibility::ensure_builtin_extension_enabled(
                &DATETIME_GPU_INPUT_EXTENSION,
                label,
            )
            .map_err(never_retry)?;
        }
        if rest
            .iter()
            .any(|value| matches!(value, Value::Bool(_) | Value::LogicalArray(_)))
        {
            crate::compatibility::ensure_builtin_extension_enabled(
                &DATETIME_LOGICAL_INPUT_EXTENSION,
                label,
            )
            .map_err(never_retry)?;
        }
    } else if rest
        .iter()
        .any(|value| matches!(value, Value::GpuTensor(_)))
    {
        return Err(never_retry(datetime_error(format!(
            "{label}: component type or legacy date format must be host text"
        ))));
    }
    let _roots = RootedValueLease::for_values_iterator_in_context(
        std::iter::once(value).chain(rest.iter()),
        runtime,
    )
    .map_err(never_retry)?;
    work.check()?;
    let gathered = runtime
        .scope(crate::gather_if_needed_async(value))
        .await
        .map_err(|source| datetime_gather_error(Some(runtime), source))?;
    work.check()?;
    let gathered_control = if matches!(kind, ComponentKind::Day)
        && matches!(rest.first(), Some(Value::GpuTensor(_)))
    {
        let control = runtime
            .scope(crate::gather_if_needed_async(
                rest.first().expect("checked one optional control"),
            ))
            .await
            .map_err(|source| datetime_gather_error(Some(runtime), source))?;
        work.check()?;
        Some(control)
    } else {
        None
    };
    let control = PreparedControl::new(
        runtime,
        &mut work,
        gathered_control.as_ref().or_else(|| rest.first()),
    )
    .await?;
    let option = control.text();
    let stored = is_datetime_object(&gathered);
    let mode = if stored {
        let normalized = option.map(str::trim).unwrap_or("");
        match kind {
            ComponentKind::Year
                if normalized.is_empty() || normalized.eq_ignore_ascii_case("iso") =>
            {
                SelectedMode::Numeric(NumericMode::YearIso)
            }
            ComponentKind::Year if normalized.eq_ignore_ascii_case("gregorian") => {
                SelectedMode::Numeric(NumericMode::YearGregorian)
            }
            ComponentKind::Month
                if normalized.is_empty() || normalized.eq_ignore_ascii_case("monthofyear") =>
            {
                SelectedMode::Numeric(NumericMode::Month)
            }
            ComponentKind::Month if normalized.eq_ignore_ascii_case("name") => {
                SelectedMode::Named(NamedMode::Month { short: false })
            }
            ComponentKind::Month if normalized.eq_ignore_ascii_case("shortname") => {
                SelectedMode::Named(NamedMode::Month { short: true })
            }
            ComponentKind::Day
                if normalized.is_empty() || normalized.eq_ignore_ascii_case("dayofmonth") =>
            {
                SelectedMode::Numeric(NumericMode::DayOfMonth)
            }
            ComponentKind::Day if normalized.eq_ignore_ascii_case("dayofweek") => {
                SelectedMode::Numeric(NumericMode::DayOfWeek)
            }
            ComponentKind::Day if normalized.eq_ignore_ascii_case("iso-dayofweek") => {
                SelectedMode::Numeric(NumericMode::IsoDayOfWeek)
            }
            ComponentKind::Day if normalized.eq_ignore_ascii_case("dayofyear") => {
                SelectedMode::Numeric(NumericMode::DayOfYear)
            }
            ComponentKind::Day if normalized.eq_ignore_ascii_case("name") => {
                SelectedMode::Named(NamedMode::Day { short: false })
            }
            ComponentKind::Day if normalized.eq_ignore_ascii_case("shortname") => {
                SelectedMode::Named(NamedMode::Day { short: true })
            }
            ComponentKind::Hour if normalized.is_empty() => {
                SelectedMode::Numeric(NumericMode::Hour)
            }
            ComponentKind::Minute if normalized.is_empty() => {
                SelectedMode::Numeric(NumericMode::Minute)
            }
            _ => {
                return Err(kind.invalid_stored_option(normalized));
            }
        }
    } else {
        match kind {
            ComponentKind::Year => SelectedMode::Numeric(NumericMode::YearIso),
            ComponentKind::Month => SelectedMode::Numeric(NumericMode::Month),
            ComponentKind::Day => SelectedMode::Numeric(NumericMode::DayOfMonth),
            ComponentKind::Hour => SelectedMode::Numeric(NumericMode::Hour),
            ComponentKind::Minute => SelectedMode::Numeric(NumericMode::Minute),
        }
    };
    let source = if stored {
        Source::Stored(SerialView::borrow(&gathered).map_err(never_retry)?)
    } else if matches!(
        &gathered,
        Value::String(_) | Value::StringArray(_) | Value::CharArray(_)
    ) {
        Source::Text(&gathered)
    } else {
        Source::Numeric(
            NumericView::borrow(&gathered, label, true, &mut work)
                .await
                .map_err(never_retry)?,
        )
    };
    let scalar_shape = [1, 1];
    let string_empty_shape = if let Source::Text(Value::StringArray(array)) = &source {
        [array.data().len().min(1), 1]
    } else {
        scalar_shape
    };
    let char_shape = if let Source::Text(Value::CharArray(array)) = &source {
        [array.rows, 1]
    } else {
        scalar_shape
    };
    let shape = match &source {
        Source::Stored(SerialView::Scalar(_)) => &scalar_shape[..],
        Source::Stored(SerialView::Tensor(tensor)) => tensor.shape(),
        Source::Numeric(source) => source.shape(),
        Source::Text(Value::String(_)) => &scalar_shape[..],
        Source::Text(Value::StringArray(array)) if array.shape.is_empty() => {
            &string_empty_shape[..]
        }
        Source::Text(Value::StringArray(array)) => &array.shape,
        Source::Text(Value::CharArray(_)) => &char_shape[..],
        Source::Text(_) => unreachable!("text source classified at admission"),
    };
    let mode = match mode {
        SelectedMode::Numeric(mode) => mode,
        SelectedMode::Named(mode) => {
            let Source::Stored(serials) = &source else {
                unreachable!("named component mode requires stored datetime input");
            };
            return named::map_named(runtime, &mut work, serials, shape, mode)
                .await
                .map(Some);
        }
    };
    let len = source.len();
    let (format_len, row_len) = if let Source::Text(value) = &source {
        (option.map_or(0, str::len), text_scratch(value, label)?)
    } else {
        (0, 0)
    };
    let scratch_len = format_len
        .checked_add(row_len)
        .ok_or_else(|| allocation_error(Some(runtime), label, HostAllocationError::Overflow))?;
    let scratch_bytes = if scratch_len == 0 {
        0
    } else {
        HostTypedWorkspace::<u8>::quoted_bytes(scratch_len)
            .map_err(|source| allocation_error(Some(runtime), label, source))?
    };
    if len == 1 {
        let receipt = if scratch_len == 0 {
            None
        } else {
            let plan = HostAllocationPlan::new(scratch_bytes, 0)
                .map_err(|source| allocation_error(Some(runtime), label, source))?;
            HostAllocationReceipt::admit(Some(runtime), plan)
                .map_err(|source| allocation_error(Some(runtime), label, source))?
        };
        let mut scratch = receipt
            .as_ref()
            .map(|receipt| {
                receipt
                    .temporary_typed::<u8>(runtime, scratch_len)
                    .map_err(|source| allocation_error(Some(runtime), label, source))
            })
            .transpose()?;
        work.check()?;
        work.charge(1).await?;
        let result = if let Some(scratch) = scratch.as_mut() {
            let (format_storage, row_storage) = scratch.as_mut_slice().split_at_mut(format_len);
            let format = option.map(|option| {
                let len = convert_legacy_format(option, format_storage);
                std::str::from_utf8(&format_storage[..len]).expect("format bytes preserve UTF-8")
            });
            component_at(
                &source,
                mode,
                0,
                runtime,
                &mut work,
                format,
                Some(row_storage),
            )
            .await?
        } else {
            component_at(&source, mode, 0, runtime, &mut work, option, None).await?
        };
        work.flush().await?;
        work.check()?;
        if let Some(receipt) = &receipt {
            receipt
                .checkpoint(runtime)
                .map_err(|source| allocation_error(Some(runtime), label, source))?;
        }
        return Ok(Some(Value::Num(result)));
    }
    let receipt = output::admit(&work, shape, scratch_bytes)?;
    let mut scratch = receipt
        .as_ref()
        .filter(|_| scratch_len != 0)
        .map(|receipt| {
            receipt
                .temporary_typed::<u8>(runtime, scratch_len)
                .map_err(|source| allocation_error(Some(runtime), label, source))
        })
        .transpose()?;
    let converted_format_len = if let (Some(option), Some(scratch)) = (option, scratch.as_mut()) {
        convert_legacy_format(option, &mut scratch.as_mut_slice()[..format_len])
    } else {
        0
    };
    let mut output = output::Output::new(&work, shape, len, receipt.as_ref())?;
    let mut start = 0;
    while start < len {
        output.check(&work)?;
        let count = work.capacity().min(len - start);
        for index in start..start + count {
            let result = if let Some(scratch) = scratch.as_mut() {
                let (format_storage, row_storage) = scratch.as_mut_slice().split_at_mut(format_len);
                let format = option.map(|_| {
                    std::str::from_utf8(&format_storage[..converted_format_len])
                        .expect("format bytes preserve UTF-8")
                });
                component_at(
                    &source,
                    mode,
                    index,
                    runtime,
                    &mut work,
                    format,
                    Some(row_storage),
                )
                .await?
            } else {
                component_at(&source, mode, index, runtime, &mut work, option, None).await?
            };
            output.set(index, result);
        }
        work.charge(count).await?;
        output.check(&work)?;
        start += count;
    }
    work.flush().await?;
    output.check(&work)?;
    output.finish(&work).map(Some)
}

#[cfg(test)]
#[path = "legacy/tests.rs"]
mod tests;

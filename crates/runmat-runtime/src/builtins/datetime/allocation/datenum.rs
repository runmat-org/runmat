//! Borrowed calendar operands and exact retained output for managed `datenum`.
//! This route never constructs an intermediate datetime object or broadcast Vec.
use super::super::conversion::CalendarWork;
use super::super::{
    capabilities::{DATETIME_CLASS, SERIAL_FIELD},
    datetime_error, datetime_execution_error, datetime_gather_error, datetime_utf16_error,
    naive_from_components, serial_from_normalized_components,
    text_parsing::{encode_utf16_chunk, parse_datenum_text_borrowed, utf16_chunk_end},
};
use super::numeric_view::{validate_integer_prefix, NumericView};
use crate::builtins::common::tensor;
use crate::call::BuiltinArguments;
use crate::context::host_allocation::{
    host_allocation_runtime_error, HostAllocationCheckpoint, HostAllocationError,
    HostAllocationPlan, HostAllocationReceipt, HostTypedWorkspace,
};
use crate::context::RuntimeContext;
use crate::{BuiltinResult, GpuGatherRetry, RuntimeError};
use runmat_value::{NumericDType, Tensor, Value};

enum Shape<'a> {
    Borrowed(&'a [usize]),
    Pair([usize; 2]),
}
impl Shape<'_> {
    fn as_slice(&self) -> &[usize] {
        match self {
            Self::Borrowed(shape) => shape,
            Self::Pair(shape) => shape,
        }
    }
}

enum Source<'a> {
    Now,
    Serial(NumericView<'a>),
    Datevec {
        tensor: &'a Tensor,
        rows: usize,
        cols: usize,
    },
    Text(&'a Value),
    Components([NumericView<'a>; 6]),
}

struct Plan<'a> {
    source: Source<'a>,
    shape: Shape<'a>,
    len: usize,
    text_scratch: usize,
}

impl<'a> Plan<'a> {
    async fn new(args: &'a BuiltinArguments, work: &mut CalendarWork<'_>) -> BuiltinResult<Self> {
        match args.len() {
            0 => Ok(Self {
                source: Source::Now,
                shape: Shape::Pair([1, 1]),
                len: 1,
                text_scratch: 0,
            }),
            1 => Self::one(&args[0], work).await,
            3..=6 => Self::components(args, work).await,
            _ => Err(datetime_error(
                "datenum: expected datetime, text, date vector, or Y/M/D components",
            )),
        }
    }

    async fn one(value: &'a Value, work: &mut CalendarWork<'_>) -> BuiltinResult<Self> {
        if let Value::Object(object) = value {
            if object.is_class(DATETIME_CLASS) {
                let serial = match object.property(SERIAL_FIELD) {
                    Some(Value::Num(value)) => NumericView::Scalar(*value),
                    Some(Value::Tensor(tensor)) => NumericView::Tensor(tensor),
                    Some(other) => {
                        return Err(datetime_error(format!(
                            "datetime: invalid internal serial storage {other:?}"
                        )))
                    }
                    None => {
                        return Err(datetime_error("datetime: missing internal serial storage"))
                    }
                };
                return Ok(Self {
                    source: Source::Serial(serial),
                    shape: Shape::Borrowed(serial.shape()),
                    len: serial.len(),
                    text_scratch: 0,
                });
            }
        }
        if matches!(
            value,
            Value::String(_) | Value::StringArray(_) | Value::CharArray(_)
        ) {
            if let Value::CharArray(array) = value {
                array
                    .validate_layout()
                    .map_err(|source| datetime_error(format!("datenum: {source}")))?;
            }
            if let Value::StringArray(array) = value {
                let extent = array
                    .shape
                    .iter()
                    .try_fold(1usize, |len, axis| len.checked_mul(*axis));
                if extent != Some(array.data().len()) {
                    return Err(datetime_error(
                        "datenum: string array shape and payload disagree",
                    ));
                }
            }
            let (len, shape, text_scratch) = match value {
                Value::String(_) => (1, Shape::Pair([1, 1]), 0),
                Value::StringArray(array) => {
                    let shape = if array.shape.is_empty() {
                        Shape::Pair([array.data().len().min(1), 1])
                    } else {
                        Shape::Borrowed(&array.shape)
                    };
                    (array.data().len(), shape, 0)
                }
                Value::CharArray(array) => (
                    array.rows,
                    Shape::Pair([array.rows, 1]),
                    if array.rows == 0 {
                        0
                    } else {
                        array
                            .cols
                            .checked_mul(4)
                            .ok_or_else(|| {
                                datetime_error(
                                    "datenum: character row extent exceeds platform limits",
                                )
                            })?
                            .max(1)
                    },
                ),
                _ => unreachable!(),
            };
            return Ok(Self {
                source: Source::Text(value),
                shape,
                len,
                text_scratch,
            });
        }
        if let Value::Tensor(tensor) = value {
            // `tensor_from_datevec_like` first tries a 3/6-column matrix and
            // then a three/six-element row. Rejected geometry is a serial
            // array, not an invalid date-vector input.
            let shape = tensor.shape();
            let matrix = shape.len() >= 2
                && (shape[1] == 3
                    || (shape[1] == 6 && shape[0].checked_mul(shape[1]) == Some(tensor.len())));
            let row = tensor.len() == 3 || tensor.len() == 6;
            if matrix || row {
                let (rows, cols) = if matrix {
                    (shape[0], shape[1])
                } else {
                    (1, tensor.len())
                };
                // The legacy three-column normalization reads only its first
                // page. Validate exactly those authoritative integer lanes,
                // not ignored trailing pages, before any f64 conversion.
                let count = rows.checked_mul(cols).ok_or_else(|| {
                    datetime_error("datetime: date vector shape exceeds platform limits")
                })?;
                validate_integer_prefix(tensor, count, "date vector", false, work).await?;
                return Ok(Self {
                    source: Source::Datevec { tensor, rows, cols },
                    shape: Shape::Pair([rows, 1]),
                    len: rows,
                    text_scratch: 0,
                });
            }
        }
        let serial = NumericView::borrow(value, "datenum", true, work).await?;
        Ok(Self {
            source: Source::Serial(serial),
            shape: Shape::Borrowed(serial.shape()),
            len: serial.len(),
            text_scratch: 0,
        })
    }

    async fn components(
        args: &'a BuiltinArguments,
        work: &mut CalendarWork<'_>,
    ) -> BuiltinResult<Self> {
        let labels = ["year", "month", "day", "hour", "minute", "second"];
        let mut components = [NumericView::Scalar(0.0); 6];
        let mut target: Option<(usize, &'a [usize])> = None;
        for (index, value) in args.iter().enumerate() {
            let view = NumericView::borrow(value, labels[index], false, work).await?;
            components[index] = view;
            if view.len() != 1 {
                if let Some((length, shape)) = target {
                    if length != view.len() || shape != view.shape() {
                        return Err(datetime_error(
                            "datetime: non-scalar component inputs must have matching sizes",
                        ));
                    }
                } else {
                    target = Some((view.len(), view.shape()));
                }
            }
        }
        let (len, shape) = match target {
            Some((len, shape)) => (len, Shape::Borrowed(shape)),
            None => (1, Shape::Pair([1, 1])),
        };
        Ok(Self {
            source: Source::Components(components),
            shape,
            len,
            text_scratch: 0,
        })
    }

    fn at(&self, index: usize, runtime: &RuntimeContext, text: Option<&str>) -> BuiltinResult<f64> {
        match &self.source {
            Source::Now => Ok(super::super::datenum_from_naive(
                super::super::current_naive_local(Some(runtime))?,
            )),
            Source::Serial(values) => Ok(values.at(index)),
            Source::Datevec { tensor, rows, cols } => {
                let component = |column: usize| {
                    if column < *cols {
                        tensor::tensor_value_f64(tensor, column * *rows + index)
                    } else {
                        0.0
                    }
                };
                let naive = naive_from_components(
                    component(0),
                    component(1),
                    component(2),
                    component(3),
                    component(4),
                    component(5),
                )?;
                Ok(super::super::datenum_from_naive(naive))
            }
            Source::Text(value) => {
                let source = match value {
                    Value::String(text) => {
                        crate::builtins::datetime::nonmissing_text(text, "datenum input")?
                    }
                    Value::StringArray(array) => crate::builtins::datetime::nonmissing_text(
                        &array.data()[index],
                        "datenum input",
                    )?,
                    Value::CharArray(_) => text.expect("character row has admitted scratch"),
                    _ => unreachable!(),
                };
                parse_datenum_text_borrowed(source.trim_end(), Some(runtime))
            }
            Source::Components(values) => {
                let at = |column: usize| {
                    values[column].at(if values[column].len() == 1 { 0 } else { index })
                };
                serial_from_normalized_components(at(0), at(1), at(2), at(3), at(4), at(5))
            }
        }
    }
}

fn allocation_error(runtime: &RuntimeContext, source: HostAllocationError) -> RuntimeError {
    never(host_allocation_runtime_error(
        Some(runtime),
        "RunMat:datenum:CalendarWorkspace",
        "datenum: managed calendar allocation failed",
        source,
    ))
}

pub(in crate::builtins::datetime) fn never(mut error: RuntimeError) -> RuntimeError {
    error.gpu_gather_retry = GpuGatherRetry::Never;
    error
}

pub(in crate::builtins::datetime) fn capture(
    runtime: Option<&RuntimeContext>,
) -> BuiltinResult<Option<HostAllocationCheckpoint>> {
    HostAllocationReceipt::capture_owner(runtime).map_err(|source| {
        never(host_allocation_runtime_error(
            runtime,
            "RunMat:datenum:CalendarWorkspace",
            "datenum: calendar allocation owner unavailable",
            source,
        ))
    })
}

/// The original owner is captured before any GPU input gather. The returned
/// tensor is claimed against that owner after the last cooperative suspension.
pub(in crate::builtins::datetime) async fn managed(
    runtime: &RuntimeContext,
    original: HostAllocationCheckpoint,
    args: BuiltinArguments,
) -> BuiltinResult<Value> {
    let args = args
        .try_map_preserving_roots_async(args.len(), Some(runtime), async |index| {
            original
                .check(runtime)
                .map_err(|source| allocation_error(runtime, source))?;
            let value = runtime
                .scope(crate::gather_if_needed_async(&args[index]))
                .await
                .map_err(|source| never(datetime_gather_error(Some(runtime), source)))?;
            original
                .check(runtime)
                .map_err(|source| allocation_error(runtime, source))?;
            Ok(value)
        })
        .await
        .map_err(never)?;
    original
        .check(runtime)
        .map_err(|source| allocation_error(runtime, source))?;
    let mut preparation =
        CalendarWork::with_owner(Some(runtime), "datenum", Some(original.clone()))?;
    let plan = Plan::new(&args, &mut preparation).await.map_err(never)?;
    let scratch = if plan.text_scratch == 0 {
        0
    } else {
        HostTypedWorkspace::<u8>::quoted_bytes(plan.text_scratch)
            .map_err(|source| allocation_error(runtime, source))?
    };
    let output = if plan.len == 1 {
        0
    } else {
        HostAllocationPlan::retained_tensor_bytes(NumericDType::F64, plan.shape.as_slice())
            .map_err(|source| allocation_error(runtime, source))?
    };
    let quote = HostAllocationPlan::new(scratch, output)
        .map_err(|source| allocation_error(runtime, source))?;
    let receipt = HostAllocationReceipt::admit(Some(runtime), quote)
        .map_err(|source| allocation_error(runtime, source))?
        .ok_or_else(|| allocation_error(runtime, HostAllocationError::StaleAuthority))?;
    original
        .check(runtime)
        .map_err(|source| allocation_error(runtime, source))?;
    let mut text = if plan.text_scratch > 0 {
        Some(
            receipt
                .temporary_typed::<u8>(runtime, plan.text_scratch)
                .map_err(|source| allocation_error(runtime, source))?,
        )
    } else {
        None
    };
    let mut output = if plan.len == 1 {
        None
    } else {
        Some(
            receipt
                .exact_output_tensor(runtime, NumericDType::F64, plan.shape.as_slice())
                .map_err(|source| allocation_error(runtime, source))?,
        )
    };
    let mut budget = runtime.cpu_cooperation();
    let mut scalar = 0.0;
    for index in 0..plan.len {
        let char_text = if let (Source::Text(Value::CharArray(array)), Some(scratch)) =
            (&plan.source, text.as_mut())
        {
            let row = &array.data()[index * array.cols..(index + 1) * array.cols];
            let mut end = 0;
            let mut source = 0;
            while source < row.len() {
                let next = utf16_chunk_end(row, source);
                end += encode_utf16_chunk(
                    &row[source..next],
                    source,
                    &mut scratch.as_mut_slice()[end..],
                )
                .map_err(|cause| never(datetime_utf16_error(cause)))?;
                if next < row.len() {
                    budget
                        .checkpoint(u64::try_from(next - source).unwrap_or(u64::MAX))
                        .await
                        .map_err(|source| never(datetime_execution_error(Some(runtime), source)))?;
                    original
                        .check(runtime)
                        .map_err(|source| allocation_error(runtime, source))?;
                    receipt
                        .checkpoint(runtime)
                        .map_err(|source| allocation_error(runtime, source))?;
                }
                source = next;
            }
            Some(std::str::from_utf8(&scratch.as_slice()[..end]).expect("encoded Unicode row"))
        } else {
            None
        };
        let value = plan.at(index, runtime, char_text).map_err(never)?;
        if let Some(output) = output.as_mut() {
            output
                .as_f64_slice_mut()
                .expect("datenum exact output is double")[index] = value;
        } else {
            scalar = value;
        }
        if index % 256 == 255 || index + 1 == plan.len {
            let count = u64::try_from(index % 256 + 1).unwrap_or(u64::MAX);
            budget
                .checkpoint(count)
                .await
                .map_err(|source| never(datetime_execution_error(Some(runtime), source)))?;
            original
                .check(runtime)
                .map_err(|source| allocation_error(runtime, source))?;
            receipt
                .checkpoint(runtime)
                .map_err(|source| allocation_error(runtime, source))?;
        }
    }
    receipt
        .checkpoint(runtime)
        .map_err(|source| allocation_error(runtime, source))?;
    original
        .check(runtime)
        .map_err(|source| allocation_error(runtime, source))?;
    if let Some(output) = output {
        let tensor = output
            .into_tensor()
            .map_err(|source| allocation_error(runtime, source.into()))?;
        receipt
            .checkpoint(runtime)
            .map_err(|source| allocation_error(runtime, source))?;
        Ok(Value::Tensor(tensor))
    } else {
        Ok(Value::Num(scalar))
    }
}

#[cfg(test)]
#[path = "datenum/tests.rs"]
mod tests;

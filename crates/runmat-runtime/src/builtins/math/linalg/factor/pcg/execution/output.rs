use super::{
    invalid, map_operator_in_context, BuiltinResult, ExactHostAllocationPermit,
    ExactHostNumericAllocationError, ExactHostTensorAllocation, HostAllocationError,
    HostAllocationPlan, HostAllocationReceipt, NumericDType, PcgOutcome, PcgValue, RuntimeContext,
    SparseOperatorError, Tensor, Value,
};
use runmat_value::{OutputListInitializationError, OutputListStorage};

fn admission(context: Option<&RuntimeContext>, error: HostAllocationError) -> super::RuntimeError {
    map_operator_in_context(SparseOperatorError::Admission(error), context)
}

fn response(
    x: Value,
    residuals: Option<Value>,
    outcome: PcgOutcome,
    requested: Option<usize>,
) -> Value {
    match requested {
        None | Some(1) => x,
        Some(0) => Value::output_list(Vec::new()),
        Some(count) => {
            let mut output = Vec::with_capacity(count);
            output.push(x);
            if count > 1 {
                output.push(Value::Num(f64::from(outcome.flag)));
            }
            if count > 2 {
                output.push(Value::Num(outcome.relres));
            }
            if count > 3 {
                output.push(Value::Num(outcome.iterations as f64));
            }
            if count > 4 {
                output.push(residuals.expect("five outputs prepare residual history"));
            }
            Value::output_list(output)
        }
    }
}

fn unbounded_residual_output(dtype: NumericDType, values: &[f64]) -> BuiltinResult<Value> {
    let result = match dtype {
        NumericDType::F32 => Tensor::from_f32(
            values.iter().map(|&value| value as f32).collect(),
            vec![values.len(), 1],
        ),
        NumericDType::F64 => Tensor::new(values.to_vec(), vec![values.len(), 1]),
        _ => return Err(invalid("unsupported residual history class")),
    };
    result
        .map(Value::Tensor)
        .map_err(|_| invalid("unable to retain residual history"))
}

fn admitted_residual_output(
    dtype: NumericDType,
    values: &[f64],
    permit: &dyn ExactHostAllocationPermit,
) -> Result<Value, ExactHostNumericAllocationError> {
    let mut output = ExactHostTensorAllocation::new(dtype, &[values.len(), 1], permit)?;
    match dtype {
        NumericDType::F32 => {
            let target = output
                .as_f32_slice_mut()
                .ok_or(ExactHostNumericAllocationError::UnsupportedClass)?;
            for (destination, &value) in target.iter_mut().zip(values) {
                *destination = value as f32;
            }
        }
        NumericDType::F64 => output
            .as_f64_slice_mut()
            .ok_or(ExactHostNumericAllocationError::UnsupportedClass)?
            .copy_from_slice(values),
        _ => return Err(ExactHostNumericAllocationError::UnsupportedClass),
    }
    output.into_tensor().map(Value::Tensor)
}

pub(super) struct PcgQuote {
    pub(super) main_count: usize,
    pub(super) residual_count: usize,
    pub(super) plan: HostAllocationPlan,
}

pub(super) fn quote<T: PcgValue>(
    n: usize,
    maximum_iterations: usize,
    vector_count: usize,
    requested: Option<usize>,
    context: Option<&RuntimeContext>,
) -> BuiltinResult<PcgQuote> {
    let main_count = n
        .checked_mul(vector_count)
        .and_then(|count| count.checked_mul(T::LANES))
        .ok_or_else(|| invalid("workspace dimensions overflow"))?;
    let residual_count = maximum_iterations
        .checked_add(1)
        .ok_or_else(|| invalid("residual history dimensions overflow"))?;
    let scratch = HostAllocationPlan::numeric_workspace_bytes(T::DTYPE, main_count)
        .and_then(|bytes| {
            bytes
                .checked_add(HostAllocationPlan::numeric_workspace_bytes(
                    NumericDType::F64,
                    residual_count,
                )?)
                .ok_or(HostAllocationError::Overflow)
        })
        .map_err(|error| admission(context, error))?;
    let mut output = T::output_bytes(n).map_err(|error| admission(context, error))?;
    if requested.is_some_and(|count| count >= 5) {
        output = output
            .checked_add(
                HostAllocationPlan::retained_tensor_bytes_for_dimensions(
                    T::DTYPE,
                    residual_count,
                    2,
                )
                .map_err(|error| admission(context, error))?,
            )
            .ok_or_else(|| admission(context, HostAllocationError::Overflow))?;
    }
    if let Some(count @ 2..=5) = requested {
        let raw = OutputListStorage::requested_bytes(count)
            .map_err(|error| admission(context, HostAllocationError::Numeric(error)))?;
        output = output
            .checked_add(
                HostAllocationPlan::retained_exact_output_bytes(
                    u64::try_from(raw)
                        .map_err(|_| admission(context, HostAllocationError::Overflow))?,
                    1,
                )
                .map_err(|error| admission(context, error))?,
            )
            .ok_or_else(|| admission(context, HostAllocationError::Overflow))?;
    }
    Ok(PcgQuote {
        main_count,
        residual_count,
        plan: HostAllocationPlan::new(scratch, output)
            .map_err(|error| admission(context, error))?,
    })
}

pub(super) fn publish_admitted<T: PcgValue>(
    receipt: &HostAllocationReceipt,
    context: &RuntimeContext,
    x: &[T],
    norms: &[f64],
    outcome: PcgOutcome,
    requested: Option<usize>,
) -> BuiltinResult<Value> {
    receipt
        .checkpoint(context)
        .map_err(|error| admission(Some(context), error))?;
    let result = receipt
        .with_output_allocation(context, |permit| {
            let x = T::admitted_output(x, permit)?;
            let history = if requested.is_some_and(|count| count >= 5) {
                Some(admitted_residual_output(
                    T::DTYPE,
                    &norms[..outcome.residual_count],
                    permit,
                )?)
            } else {
                None
            };
            if let Some(count @ 2..=5) = requested {
                let mut x = Some(x);
                let mut history = history;
                let list = OutputListStorage::try_initialize(count, permit, |index| {
                    Ok::<Value, ExactHostNumericAllocationError>(match index {
                        0 => x.take().expect("solution initialized once"),
                        1 => Value::Num(f64::from(outcome.flag)),
                        2 => Value::Num(outcome.relres),
                        3 => Value::Num(outcome.iterations as f64),
                        4 => history
                            .take()
                            .expect("five outputs prepare residual history"),
                        _ => unreachable!("pcg validates output count"),
                    })
                })
                .map_err(|error| match error {
                    OutputListInitializationError::Allocation(error)
                    | OutputListInitializationError::Element(error) => error,
                })?;
                Ok(Value::OutputList(list))
            } else {
                Ok(response(x, history, outcome, requested))
            }
        })
        .map_err(|error| admission(Some(context), error))?;
    Ok(result)
}

pub(super) fn publish_unbounded<T: PcgValue>(
    mut main: Vec<T>,
    n: usize,
    norms: &[f64],
    outcome: PcgOutcome,
    requested: Option<usize>,
) -> BuiltinResult<Value> {
    main.truncate(n);
    let x = T::unbounded_output(main)?;
    let history = if requested.is_some_and(|count| count >= 5) {
        Some(unbounded_residual_output(
            T::DTYPE,
            &norms[..outcome.residual_count],
        )?)
    } else {
        None
    };
    Ok(response(x, history, outcome, requested))
}

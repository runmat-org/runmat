//! Runtime orchestration for integer division.

use runmat_builtins::BuiltinErrorDescriptor;
use runmat_types::IntegerClass;
use runmat_value::{NumericDType, Tensor, Value};

use crate::builtins::common::broadcast::BroadcastPlan;
use crate::builtins::common::integer_value::{value_from_exact_integers, IntegerClassValueExt};
use crate::builtins::common::{gpu_helpers, resident_output};
use crate::{build_runtime_error, BuiltinResult, RuntimeError};

use operand::{output_class, IdivideBuffer};
use rounding::RoundingMode;

const NAME: &str = "idivide";
const ERROR_INVALID_INPUT: BuiltinErrorDescriptor = runmat_builtins::IDIVIDE_ERROR_INVALID_INPUT;
const ERROR_SIZE_MISMATCH: BuiltinErrorDescriptor = runmat_builtins::IDIVIDE_ERROR_SIZE_MISMATCH;
const ERROR_DIVIDE_BY_ZERO: BuiltinErrorDescriptor = runmat_builtins::IDIVIDE_ERROR_DIVIDE_BY_ZERO;
const ERROR_OVERFLOW: BuiltinErrorDescriptor = runmat_builtins::IDIVIDE_ERROR_OVERFLOW;

mod operand;
mod rounding;

pub(super) async fn evaluate(args: Vec<Value>) -> BuiltinResult<Value> {
    if !(2..=3).contains(&args.len()) {
        return Err(error(
            &ERROR_INVALID_INPUT,
            "expected two integer inputs and an optional rounding mode",
        ));
    }
    let output_source = gpu_helpers::select_resident_output_source(
        args.iter().take(2).filter_map(|value| match value {
            Value::GpuTensor(handle) => Some(handle.clone()),
            _ => None,
        }),
        NAME,
    )?;
    let mut arguments = args.into_iter();
    let left = IdivideBuffer::from_value(arguments.next().expect("A")).await?;
    let right = IdivideBuffer::from_value(arguments.next().expect("B")).await?;
    let class = output_class(&left, &right)?;
    let rounding = match arguments.next() {
        Some(value) => RoundingMode::parse(&value)?,
        None => RoundingMode::Fix,
    };
    let plan = BroadcastPlan::new(&left.shape, &right.shape)
        .map_err(|detail| error(&ERROR_SIZE_MISMATCH, detail))?;
    let mut values = Vec::with_capacity(plan.len());
    for (_, left_index, right_index) in plan.iter() {
        let divisor = right.data[right_index];
        if divisor == 0 {
            return Err(error(&ERROR_DIVIDE_BY_ZERO, "divisor contains zero"));
        }
        let quotient = rounding.divide(left.data[left_index], divisor);
        let value = class.value_from_i128(quotient).ok_or_else(|| {
            error(
                &ERROR_OVERFLOW,
                format!("value {quotient} is outside output class range"),
            )
        })?;
        values.push(value);
    }
    let result = value_from_exact_integers(values, plan.output_shape().to_vec(), class)
        .map_err(|detail| error(&ERROR_INVALID_INPUT, detail))?;
    resident_output::restore_to_source_provider(result, output_source.as_ref())
        .map_err(|detail| error(&ERROR_INVALID_INPUT, detail))
}

fn error(
    descriptor: &'static BuiltinErrorDescriptor,
    detail: impl std::fmt::Display,
) -> RuntimeError {
    let message = format!("{}: {detail}", descriptor.message);
    let mut builder = build_runtime_error(message).with_builtin(NAME);
    if let Some(identifier) = descriptor.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

//! Runtime orchestration for integer division.

use runmat_builtins::BuiltinErrorDescriptor;
use runmat_types::IntegerClass;
use runmat_value::{NumericDType, Tensor, Value};

use crate::builtins::common::broadcast::BroadcastPlan;
use crate::builtins::common::integer_value::{value_from_exact_integers, IntegerClassValueExt};
use crate::builtins::common::{gpu_helpers, resident_output};
use crate::{build_runtime_error, BuiltinResult, RuntimeError};

use operand::{output_class, AdmittedOperand};
use rounding::RoundingMode;

const NAME: &str = "idivide";
const ERROR_INVALID_INPUT: BuiltinErrorDescriptor = runmat_builtins::IDIVIDE_ERROR_INVALID_INPUT;
const ERROR_SIZE_MISMATCH: BuiltinErrorDescriptor = runmat_builtins::IDIVIDE_ERROR_SIZE_MISMATCH;
const ERROR_DIVIDE_BY_ZERO: BuiltinErrorDescriptor = runmat_builtins::IDIVIDE_ERROR_DIVIDE_BY_ZERO;
const ERROR_OVERFLOW: BuiltinErrorDescriptor = runmat_builtins::IDIVIDE_ERROR_OVERFLOW;

mod operand;
mod rounding;

pub(super) async fn evaluate(args: crate::call::BuiltinArguments) -> BuiltinResult<Value> {
    crate::output_count::validate_fixed_outputs(
        runmat_builtins::IDIVIDE_CATALOG_ENTRY.descriptor,
        &ERROR_INVALID_INPUT,
        NAME,
    )?;
    if !(2..=3).contains(&args.len()) {
        return Err(error(
            &ERROR_INVALID_INPUT,
            "expected two integer inputs and an optional rounding mode",
        ));
    }
    let rounding = match args.get(2) {
        Some(value) => RoundingMode::parse(value)?,
        None => RoundingMode::Fix,
    };
    let borrowed: Vec<_> = args
        .iter()
        .take(2)
        .filter_map(|value| match value {
            Value::GpuTensor(handle) => Some(handle.clone()),
            _ => None,
        })
        .collect();
    let output_source = gpu_helpers::select_resident_output_source(borrowed.iter().cloned(), NAME)?;
    let mut arguments = args.into_iter();
    let left = AdmittedOperand::new(arguments.next().expect("A"))?;
    let right = AdmittedOperand::new(arguments.next().expect("B"))?;
    let class = output_class(&left, &right)?;
    let plan = BroadcastPlan::new(left.shape(), right.shape())
        .map_err(|detail| error(&ERROR_SIZE_MISMATCH, detail))?;
    let left = left.materialize().await?;
    let right = right.materialize().await?;
    let mut values = Vec::with_capacity(plan.len());
    for (_, left_index, right_index) in plan.iter() {
        let divisor = right.data[right_index];
        if divisor == 0 {
            return Err(error(&ERROR_DIVIDE_BY_ZERO, "divisor contains zero"));
        }
        let quotient = rounding
            .divide(left.data[left_index], divisor)
            .ok_or_else(|| {
                error(
                    &ERROR_OVERFLOW,
                    "integer quotient is outside the supported range",
                )
            })?;
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
    let result = match result {
        Value::Int(value) if output_source.is_some() => Value::Tensor(
            Tensor::new_integer(
                runmat_value::IntegerStorage::from_scalar(value),
                plan.output_shape().to_vec(),
            )
            .map_err(|detail| error(&ERROR_INVALID_INPUT, detail))?,
        ),
        other => other,
    };
    let protected: Vec<_> = borrowed.iter().collect();
    resident_output::restore_to_source_provider_protecting(
        result,
        output_source.as_ref(),
        &protected,
    )
    .map_err(|source| {
        let mut error = error(&ERROR_INVALID_INPUT, &source);
        error.source = Some(Box::new(source));
        error
    })
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

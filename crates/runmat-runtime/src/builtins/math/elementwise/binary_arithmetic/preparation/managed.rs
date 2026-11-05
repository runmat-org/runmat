//! The selected finite host route uses the same arithmetic recipe as the
//! ordinary native-storage kernels, writing its preallocated result in place.
use crate::context::preparation::PreparedHostNumericalOutput;
use crate::context::RuntimeContext;
use crate::execution::RootedValue;
use crate::{BuiltinResult, RuntimeError};
use runmat_builtins::BuiltinPreparationOperation;
use runmat_types::{NumericClass, OperatorKind, RealFloatingRecipe};
use runmat_value::{NumericDType, Value};

pub(in crate::builtins::math::elementwise::binary_arithmetic) fn require_unbounded_host_output(
    runtime: &RuntimeContext,
) -> BuiltinResult<()> {
    use crate::context::host_allocation::{
        host_allocation_runtime_error, HostAllocationError, HostAllocationPolicy,
        HostAllocationReceipt,
    };
    let error = match HostAllocationReceipt::policy(Some(runtime)) {
        Ok(HostAllocationPolicy::Unbounded) => return Ok(()),
        Ok(HostAllocationPolicy::Managed) => HostAllocationError::MissingManagedOwner,
        Err(error) => error,
    };
    Err(host_allocation_runtime_error(
        Some(runtime),
        "RunMat:Placement:HostNumericalOutput",
        "selected host result has no prepared output owner",
        error,
    ))
}

pub(in crate::builtins::math::elementwise::binary_arithmetic) enum FloatingSlice<'a> {
    Single(&'a [f32]),
    Double(&'a [f64]),
}

enum LogicalSlice<'a> {
    Scalar(u8),
    Array(&'a [u8]),
}

impl LogicalSlice<'_> {
    fn get(&self, index: usize) -> Option<f64> {
        match self {
            Self::Scalar(value) => (index == 0).then_some(f64::from(*value)),
            Self::Array(values) => values
                .get(index)
                .map(|&value| if value != 0 { 1.0 } else { 0.0 }),
        }
    }
}

enum ArithmeticSlice<'a> {
    Floating(FloatingSlice<'a>),
    Logical(LogicalSlice<'a>),
}

fn arithmetic_slice(value: &Value) -> BuiltinResult<ArithmeticSlice<'_>> {
    match value {
        Value::Bool(value) => Ok(ArithmeticSlice::Logical(LogicalSlice::Scalar(u8::from(
            *value,
        )))),
        Value::LogicalArray(value) => Ok(ArithmeticSlice::Logical(LogicalSlice::Array(
            value.data.as_slice(),
        ))),
        _ => floating_slice(value).map(ArithmeticSlice::Floating),
    }
}

fn floating_value(values: &FloatingSlice<'_>, index: usize) -> Option<f64> {
    match values {
        FloatingSlice::Single(values) => values.get(index).copied().map(f64::from),
        FloatingSlice::Double(values) => values.get(index).copied(),
    }
}

fn floating_class(values: &FloatingSlice<'_>) -> NumericClass {
    match values {
        FloatingSlice::Single(_) => NumericClass::Single,
        FloatingSlice::Double(_) => NumericClass::Double,
    }
}

pub(in crate::builtins::math::elementwise::binary_arithmetic) fn invalid() -> RuntimeError {
    crate::build_runtime_error(
        "selected host arithmetic differs from its admitted native operands or output",
    )
    .with_identifier("RunMat:Placement:InvalidHostNumericalOutput")
    .with_gpu_gather_retry(crate::GpuGatherRetry::Never)
    .build()
}

pub(in crate::builtins::math::elementwise::binary_arithmetic) fn floating_slice(
    value: &Value,
) -> BuiltinResult<FloatingSlice<'_>> {
    match value {
        Value::Num(value) => Ok(FloatingSlice::Double(std::slice::from_ref(value))),
        Value::Tensor(tensor) => match tensor.numeric_dtype() {
            NumericDType::F32 => tensor
                .as_f32_slice()
                .map(FloatingSlice::Single)
                .ok_or_else(invalid),
            NumericDType::F64 => tensor
                .as_f64_slice()
                .map(FloatingSlice::Double)
                .ok_or_else(invalid),
            _ => Err(invalid()),
        },
        _ => Err(invalid()),
    }
}

pub(super) fn execute(
    runtime: &RuntimeContext,
    output: PreparedHostNumericalOutput,
    operator: OperatorKind,
    left: &Value,
    right: &Value,
) -> BuiltinResult<RootedValue> {
    let value = execute_value(runtime, output, operator, left, right)?;
    RootedValue::new(value).map_err(|cause| cause.into_runtime_error())
}

/// The same preallocated real binary kernel serves intrinsic unary negation
/// (multiplication by -1) and scalar matrix multiplication.
pub(crate) fn execute_value(
    runtime: &RuntimeContext,
    output: PreparedHostNumericalOutput,
    operator: OperatorKind,
    left: &Value,
    right: &Value,
) -> BuiltinResult<Value> {
    if output.operation() != BuiltinPreparationOperation::ElementwiseBinary {
        return Err(invalid());
    }
    // Read retained native storage without materializing expanded inputs.
    // Floating-only pairs select their storage-specialized loop here.
    let value = match (arithmetic_slice(left)?, arithmetic_slice(right)?) {
        (
            ArithmeticSlice::Floating(FloatingSlice::Single(a)),
            ArithmeticSlice::Floating(FloatingSlice::Single(b)),
        ) => generate(
            runtime,
            output,
            operator,
            NumericClass::Single,
            NumericClass::Single,
            |i| a.get(i).copied().map(f64::from),
            |i| b.get(i).copied().map(f64::from),
        ),
        (
            ArithmeticSlice::Floating(FloatingSlice::Single(a)),
            ArithmeticSlice::Floating(FloatingSlice::Double(b)),
        ) => generate(
            runtime,
            output,
            operator,
            NumericClass::Single,
            NumericClass::Double,
            |i| a.get(i).copied().map(f64::from),
            |i| b.get(i).copied(),
        ),
        (
            ArithmeticSlice::Floating(FloatingSlice::Double(a)),
            ArithmeticSlice::Floating(FloatingSlice::Single(b)),
        ) => generate(
            runtime,
            output,
            operator,
            NumericClass::Double,
            NumericClass::Single,
            |i| a.get(i).copied(),
            |i| b.get(i).copied().map(f64::from),
        ),
        (
            ArithmeticSlice::Floating(FloatingSlice::Double(a)),
            ArithmeticSlice::Floating(FloatingSlice::Double(b)),
        ) => generate(
            runtime,
            output,
            operator,
            NumericClass::Double,
            NumericClass::Double,
            |i| a.get(i).copied(),
            |i| b.get(i).copied(),
        ),
        (ArithmeticSlice::Logical(a), ArithmeticSlice::Logical(b)) => generate(
            runtime,
            output,
            operator,
            NumericClass::Double,
            NumericClass::Double,
            |i| a.get(i),
            |i| b.get(i),
        ),
        (ArithmeticSlice::Logical(a), ArithmeticSlice::Floating(b)) => generate(
            runtime,
            output,
            operator,
            NumericClass::Double,
            floating_class(&b),
            |i| a.get(i),
            |i| floating_value(&b, i),
        ),
        (ArithmeticSlice::Floating(a), ArithmeticSlice::Logical(b)) => generate(
            runtime,
            output,
            operator,
            floating_class(&a),
            NumericClass::Double,
            |i| floating_value(&a, i),
            |i| b.get(i),
        ),
    }?;
    let Value::Tensor(tensor) = value else {
        return Err(invalid());
    };
    Ok(crate::builtins::common::tensor::tensor_into_value(tensor))
}

fn generate(
    runtime: &RuntimeContext,
    mut output: PreparedHostNumericalOutput,
    operator: OperatorKind,
    left_class: NumericClass,
    right_class: NumericClass,
    left: impl Fn(usize) -> Option<f64>,
    right: impl Fn(usize) -> Option<f64>,
) -> BuiltinResult<Value> {
    let recipe =
        RealFloatingRecipe::for_classes(operator, left_class, right_class).ok_or_else(invalid)?;
    let expected = match recipe.result_class() {
        NumericClass::Single => NumericDType::F32,
        NumericClass::Double => NumericDType::F64,
        _ => return Err(invalid()),
    };
    if output.dtype() != expected {
        return Err(invalid());
    }
    let mut plan = output.take_broadcast_plan().ok_or_else(invalid)?;
    if plan.output_shape() != output.shape() {
        return Err(invalid());
    }
    let mut indices = plan.iter();
    let mut next = |index| {
        let (actual, a, b) = indices.next().ok_or_else(invalid)?;
        if actual != index {
            return Err(invalid());
        }
        recipe
            .evaluate(left(a).ok_or_else(invalid)?, right(b).ok_or_else(invalid)?)
            .ok_or_else(invalid)
    };
    match expected {
        NumericDType::F32 => {
            output.generate_real_f32(runtime, |index| next(index).map(|value| value as f32))
        }
        NumericDType::F64 => output.generate_real_f64(runtime, next),
        _ => Err(invalid()),
    }
}

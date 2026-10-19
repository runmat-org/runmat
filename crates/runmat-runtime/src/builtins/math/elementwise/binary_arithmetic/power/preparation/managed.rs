//! Preallocated host power reuses the ordinary power domain/evaluation kernels.
use crate::builtins::math::elementwise::binary_arithmetic::preparation::managed::{
    floating_slice, invalid, FloatingSlice,
};
use crate::context::preparation::PreparedHostNumericalOutput;
use crate::context::RuntimeContext;
use crate::execution::RootedValue;
use crate::BuiltinResult;
use runmat_accelerate_api::{ExactSingleBasePower, PowerScalarControl};
use runmat_builtins::BuiltinPreparationOperation;
use runmat_types::NumericDomain;
use runmat_value::{NumericDType, Value};

use super::super::domain::{evaluate_f32, evaluate_f64, PowerResultDomain};

#[derive(Clone, Copy)]
enum NativePower {
    Single,
    Double,
    Mixed(Option<ExactSingleBasePower>),
}

impl NativePower {
    fn dtype(self) -> NumericDType {
        match self {
            Self::Single | Self::Mixed(_) => NumericDType::F32,
            Self::Double => NumericDType::F64,
        }
    }

    fn evaluate(self, base: f64, exponent: f64) -> (NumericDomain, (f64, f64)) {
        let (domain, pair) = match self {
            Self::Single => {
                let (domain, (real, imaginary)) = evaluate_f32(base as f32, exponent as f32);
                (domain, (f64::from(real), f64::from(imaginary)))
            }
            Self::Double => evaluate_f64(base, exponent),
            Self::Mixed(Some(operation)) => (
                PowerResultDomain::Real,
                (f64::from(operation.evaluate(base as f32)), 0.0),
            ),
            Self::Mixed(None) => {
                let (domain, (real, imaginary)) = evaluate_f64(base, exponent);
                (
                    domain,
                    (f64::from(real as f32), f64::from(imaginary as f32)),
                )
            }
        };
        (
            match domain {
                PowerResultDomain::Real => NumericDomain::Real,
                PowerResultDomain::Complex => NumericDomain::Complex,
            },
            pair,
        )
    }
}

pub(super) fn execute(
    runtime: &RuntimeContext,
    output: PreparedHostNumericalOutput,
    left: &Value,
    right: &Value,
) -> BuiltinResult<(RootedValue, PowerResultDomain)> {
    if output.operation() != BuiltinPreparationOperation::RealPower {
        return Err(invalid());
    }
    let value = match (floating_slice(left)?, floating_slice(right)?) {
        (FloatingSlice::Single(a), FloatingSlice::Single(b)) => generate(
            runtime,
            output,
            NativePower::Single,
            |i| a.get(i).copied().map(f64::from),
            |i| b.get(i).copied().map(f64::from),
        ),
        (FloatingSlice::Double(a), FloatingSlice::Double(b)) => generate(
            runtime,
            output,
            NativePower::Double,
            |i| a.get(i).copied(),
            |i| b.get(i).copied(),
        ),
        (FloatingSlice::Single(a), FloatingSlice::Double(b)) => {
            let exact = if let [exponent] = b {
                ExactSingleBasePower::for_control(PowerScalarControl::DoubleBits(
                    exponent.to_bits(),
                ))
            } else {
                None
            };
            generate(
                runtime,
                output,
                NativePower::Mixed(exact),
                |i| a.get(i).copied().map(f64::from),
                |i| b.get(i).copied(),
            )
        }
        (FloatingSlice::Double(a), FloatingSlice::Single(b)) => generate(
            runtime,
            output,
            NativePower::Mixed(None),
            |i| a.get(i).copied(),
            |i| b.get(i).copied().map(f64::from),
        ),
    }?;
    let (value, domain) = match value {
        Value::Tensor(tensor) => (
            crate::builtins::common::tensor::tensor_into_value(tensor),
            PowerResultDomain::Real,
        ),
        Value::ComplexTensor(tensor) => (
            crate::builtins::common::random_args::complex_tensor_into_value(tensor),
            PowerResultDomain::Complex,
        ),
        _ => return Err(invalid()),
    };
    RootedValue::new(value)
        .map(|value| (value, domain))
        .map_err(|cause| cause.into_runtime_error())
}

fn generate(
    runtime: &RuntimeContext,
    mut output: PreparedHostNumericalOutput,
    kernel: NativePower,
    left: impl Fn(usize) -> Option<f64>,
    right: impl Fn(usize) -> Option<f64>,
) -> BuiltinResult<Value> {
    if output.dtype() != kernel.dtype() {
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
        Ok(kernel.evaluate(left(a).ok_or_else(invalid)?, right(b).ok_or_else(invalid)?))
    };
    match kernel.dtype() {
        NumericDType::F32 => output.generate_promoting_f32(runtime, |index| {
            next(index).map(|(domain, (real, imaginary))| (domain, (real as f32, imaginary as f32)))
        }),
        NumericDType::F64 => output.generate_promoting_f64(runtime, next),
        _ => Err(invalid()),
    }
}

use super::invalid;
use crate::context::preparation::PreparedHostNumericalOutput;
use crate::context::RuntimeContext;
use crate::BuiltinResult;
use runmat_types::{NumericClass, OperatorKind, RealFloatingRecipe};
use runmat_value::{NumericDType, Value};

pub(super) fn generate(
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

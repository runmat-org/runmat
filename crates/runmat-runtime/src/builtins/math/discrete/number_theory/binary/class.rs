use runmat_types::NumericClass;

use super::{binary_error, BinaryContext, BinaryInput};
use crate::BuiltinResult;

#[derive(Clone, Copy)]
pub(in crate::builtins::math::discrete) struct BinaryOutput {
    pub(in crate::builtins::math::discrete) class: NumericClass,
}

pub(in crate::builtins::math::discrete) fn resolve_output(
    left: &BinaryInput,
    right: &BinaryInput,
    context: &'static BinaryContext,
) -> BuiltinResult<BinaryOutput> {
    let class = match (left.class, right.class) {
        (left, right) if left == right => left,
        (NumericClass::Double, NumericClass::Single)
        | (NumericClass::Single, NumericClass::Double) => NumericClass::Single,
        (integer, NumericClass::Double)
            if integer.integer_class().is_some() && right.is_scalar() =>
        {
            integer
        }
        (NumericClass::Double, integer)
            if integer.integer_class().is_some() && left.is_scalar() =>
        {
            integer
        }
        (left, right) if left.integer_class().is_some() && right.integer_class().is_some() => {
            return Err(binary_error(
                context,
                context.invalid,
                "integer inputs must have the same class",
            ));
        }
        _ => {
            return Err(binary_error(
                context,
                context.invalid,
                "integer inputs can only be paired with the same class or a double scalar",
            ));
        }
    };
    Ok(BinaryOutput { class })
}

use crate::catalog::inference::argument_error;
use runmat_types::{NumericClass, NumericDomain, ValueFact, ValueKindFact};

pub(super) fn resolve(
    left: &ValueFact,
    right: &ValueFact,
    diagnostics: &mut Vec<runmat_types::InferenceDiagnostic>,
) -> Option<NumericClass> {
    let (Some(left_numeric), Some(right_numeric)) = (left.numeric(), right.numeric()) else {
        if !matches!(left.kind, ValueKindFact::Unknown)
            || !matches!(right.kind, ValueKindFact::Unknown)
        {
            diagnostics.push(argument_error(
                "RM-CATALOG-INTEGER-BINARY-INPUT",
                "gcd and lcm require real numeric inputs",
                0,
            ));
        }
        return None;
    };
    if left_numeric.domain != NumericDomain::Real || right_numeric.domain != NumericDomain::Real {
        diagnostics.push(argument_error(
            "RM-CATALOG-INTEGER-BINARY-REAL",
            "gcd and lcm require real inputs",
            0,
        ));
        return None;
    }
    match (left_numeric.class, right_numeric.class) {
        (left, right) if left == right => Some(left),
        (NumericClass::Double, NumericClass::Single)
        | (NumericClass::Single, NumericClass::Double) => Some(NumericClass::Single),
        (integer, NumericClass::Double)
            if integer.integer_class().is_some() && right.is_scalar() =>
        {
            Some(integer)
        }
        (NumericClass::Double, integer)
            if integer.integer_class().is_some() && left.is_scalar() =>
        {
            Some(integer)
        }
        _ => {
            diagnostics.push(argument_error(
                "RM-CATALOG-INTEGER-BINARY-CLASS",
                "integer inputs must share a class or pair with a scalar double",
                0,
            ));
            None
        }
    }
}

//! Argument admission for the geometry builder. Every failure reports the
//! calling builtin's `RunMat:geometry:<name>:InvalidInput` identifier,
//! including arity and scalar conversion.

use runmat_geometry_core::GeometryScalarExpression;
use runmat_value::Value;

use super::invalid;
use crate::BuiltinResult;

/// Require exactly `count` arguments.
pub(super) fn exactly(
    builtin: &'static str,
    args: Vec<Value>,
    count: usize,
) -> BuiltinResult<Vec<Value>> {
    if args.len() != count {
        return Err(invalid(
            builtin,
            format!("expected {count} arguments, got {}", args.len()),
        ));
    }
    Ok(args)
}

/// Require at least `count` arguments.
pub(super) fn at_least(
    builtin: &'static str,
    args: Vec<Value>,
    count: usize,
    requirement: &str,
) -> BuiltinResult<Vec<Value>> {
    if args.len() < count {
        return Err(invalid(builtin, requirement.to_string()));
    }
    Ok(args)
}

pub(super) fn literal(value: f64) -> GeometryScalarExpression {
    GeometryScalarExpression::Literal { value }
}

/// Real numeric scalar: `double`, `single`, or an exactly representable
/// integer-class value. Logical, text, complex, and multi-element values are
/// rejected. Range checks belong to the feature-graph validator.
pub(super) fn scalar(
    value: &Value,
    builtin: &'static str,
    name: &str,
) -> BuiltinResult<GeometryScalarExpression> {
    let values = numeric_values(value, builtin, name)?;
    match values.as_slice() {
        [value] => Ok(literal(*value)),
        _ => Err(invalid(
            builtin,
            format!("{name} must be a real numeric scalar"),
        )),
    }
}

pub(super) fn vector3(
    value: &Value,
    builtin: &'static str,
    name: &str,
) -> BuiltinResult<[GeometryScalarExpression; 3]> {
    let values = numeric_values(value, builtin, name)?;
    let values: [f64; 3] = values
        .try_into()
        .map_err(|_| invalid(builtin, format!("{name} must contain exactly three values")))?;
    Ok(values.map(literal))
}

/// A MATLAB 4-by-4 homogeneous affine transform (`p' = T(1:3,1:3)*p +
/// T(1:3,4)`) converted to the row-major contract consumed by the exact-CAD
/// kernel. The final row must be `[0 0 0 1]`.
pub(super) fn transform(
    value: &Value,
    builtin: &'static str,
    name: &str,
) -> BuiltinResult<Vec<GeometryScalarExpression>> {
    let shape_ok = matches!(value, Value::Tensor(tensor) if tensor.shape() == [4, 4]);
    let values = numeric_values(value, builtin, name)?;
    if !shape_ok || values.len() != 16 {
        return Err(invalid(
            builtin,
            format!("{name} must be a 4-by-4 homogeneous affine transform"),
        ));
    }
    // Column-major storage: element (row, column) is values[column * 4 + row].
    let row_major = (0..4)
        .flat_map(|row| (0..4).map(move |column| (row, column)))
        .map(|(row, column)| values[column * 4 + row])
        .collect::<Vec<_>>();
    const TOLERANCE: f64 = 1.0e-12;
    let final_row = &row_major[12..16];
    if final_row[..3].iter().any(|value| value.abs() > TOLERANCE)
        || (final_row[3] - 1.0).abs() > TOLERANCE
    {
        return Err(invalid(
            builtin,
            format!("{name} must have final row [0 0 0 1]"),
        ));
    }
    Ok(row_major.into_iter().map(literal).collect())
}

pub(super) fn text(value: &Value, builtin: &'static str, name: &str) -> BuiltinResult<String> {
    match value {
        Value::String(text) => Ok(text.clone()),
        Value::CharArray(chars) if chars.rows == 1 => Ok(chars.data.iter().collect()),
        Value::StringArray(array) if array.data.len() == 1 => Ok(array.data[0].clone()),
        _ => Err(invalid(builtin, format!("{name} must be scalar text"))),
    }
}

fn numeric_values(value: &Value, builtin: &'static str, name: &str) -> BuiltinResult<Vec<f64>> {
    match value {
        Value::Num(value) if value.is_finite() => Ok(vec![*value]),
        Value::Num(_) => Err(invalid(builtin, format!("{name} must be finite"))),
        Value::Int(value) if value.to_f64().abs() <= 9_007_199_254_740_992.0 => {
            Ok(vec![value.to_f64()])
        }
        Value::Int(_) => Err(invalid(
            builtin,
            format!("{name} is not exactly representable as binary64"),
        )),
        Value::Tensor(tensor) if tensor.integer_storage().is_none() => {
            let values = tensor.materialize_f64();
            if values.iter().all(|value| value.is_finite()) {
                Ok(values)
            } else {
                Err(invalid(builtin, format!("{name} must be finite")))
            }
        }
        _ => Err(invalid(builtin, format!("{name} must be real numeric"))),
    }
}

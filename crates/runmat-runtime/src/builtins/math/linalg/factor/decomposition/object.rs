use super::*;

pub(super) fn public_property_value(object: &ObjectInstance, field: &str) -> BuiltinResult<Value> {
    match field {
        MATRIX_SIZE_FIELD
        | TYPE_FIELD
        | CHECK_CONDITION_FIELD
        | DATATYPE_FIELD
        | IS_CONJUGATE_TRANSPOSED_FIELD
        | IS_REAL_FIELD
        | IS_SPARSE_FIELD
        | SCALE_FACTOR_FIELD => object
            .properties()
            .get(field)
            .cloned()
            .ok_or_else(|| internal(format!("decomposition object is missing {field}"))),
        _ => Err(invalid(format!("unknown decomposition property '{field}'"))),
    }
}

pub(super) fn object_to_spec(value: &Value) -> BuiltinResult<DecompositionSpec> {
    let object = expect_object(value)?;
    let matrix = object
        .properties()
        .get(MATRIX_FIELD)
        .cloned()
        .ok_or_else(|| internal("decomposition object is missing coefficient matrix"))?;
    let decomp_type = object
        .properties()
        .get(TYPE_FIELD)
        .map(text_from_value)
        .transpose()?
        .flatten()
        .map(|value| parse_decomposition_type(&value))
        .transpose()?
        .ok_or_else(|| internal("decomposition object is missing Type"))?;
    let check_condition = object
        .properties()
        .get(CHECK_CONDITION_FIELD)
        .map(|value| bool_scalar(value, CHECK_CONDITION_FIELD))
        .transpose()?
        .unwrap_or(true);
    let is_conjugate_transposed = object
        .properties()
        .get(IS_CONJUGATE_TRANSPOSED_FIELD)
        .map(|value| bool_scalar(value, IS_CONJUGATE_TRANSPOSED_FIELD))
        .transpose()?
        .unwrap_or(false);
    let is_sparse = object
        .properties()
        .get(IS_SPARSE_FIELD)
        .map(|value| bool_scalar(value, IS_SPARSE_FIELD))
        .transpose()?
        .unwrap_or(false);
    let scale = object
        .properties()
        .get(SCALE_FACTOR_FIELD)
        .map(|value| complex_scalar(value, SCALE_FACTOR_FIELD))
        .transpose()?
        .unwrap_or((1.0, 0.0));
    let rank_tolerance = object
        .properties()
        .get(RANK_TOLERANCE_FIELD)
        .map(|value| real_scalar(value, RANK_TOLERANCE_FIELD))
        .transpose()?;
    Ok(DecompositionSpec {
        matrix,
        decomp_type,
        check_condition,
        rank_tolerance,
        is_conjugate_transposed,
        is_sparse,
        scale,
        factor_data: object.numeric_factor_data().cloned(),
        triangle: object
            .properties()
            .get(TRIANGLE_FIELD)
            .map(|v| {
                text_from_value(v)?
                    .ok_or_else(|| invalid("decomposition triangle must be text"))
                    .and_then(|s| {
                        if s.is_empty() {
                            Ok(None)
                        } else {
                            parse_triangular_flag(&s).map(Some)
                        }
                    })
            })
            .transpose()?
            .flatten(),
        source_is_real: object
            .properties()
            .get(SOURCE_REAL_FIELD)
            .map(|v| bool_scalar(v, SOURCE_REAL_FIELD))
            .transpose()?,
    })
}

pub(super) fn expect_object(value: &Value) -> BuiltinResult<&ObjectInstance> {
    match value {
        Value::Object(object) if object.is_class(CLASS_IDENTITY) => Ok(object),
        _ => Err(invalid("receiver must be a decomposition object")),
    }
}

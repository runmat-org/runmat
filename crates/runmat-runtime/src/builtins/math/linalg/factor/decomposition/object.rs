use super::*;

pub(super) fn spec_to_object(spec: &DecompositionSpec) -> BuiltinResult<ObjectInstance> {
    let mut object = ObjectInstance::new(CLASS_IDENTITY);
    let public_matrix = oriented_matrix(spec)?;
    object
        .properties
        .insert(MATRIX_FIELD.to_string(), spec.matrix.clone());
    object.properties.insert(
        TYPE_FIELD.to_string(),
        Value::String(decomposition_type_name(spec.decomp_type).to_string()),
    );
    object.properties.insert(
        MATRIX_SIZE_FIELD.to_string(),
        matrix_shape_value(&public_matrix)?,
    );
    object.properties.insert(
        CHECK_CONDITION_FIELD.to_string(),
        Value::Bool(spec.check_condition),
    );
    object.properties.insert(
        DATATYPE_FIELD.to_string(),
        Value::String(matrix_datatype(&spec.matrix).to_string()),
    );
    object.properties.insert(
        IS_CONJUGATE_TRANSPOSED_FIELD.to_string(),
        Value::Bool(spec.is_conjugate_transposed),
    );
    object.properties.insert(
        IS_REAL_FIELD.to_string(),
        Value::Bool(matrix_is_real(&public_matrix) && spec.scale.1 == 0.0),
    );
    object
        .properties
        .insert(IS_SPARSE_FIELD.to_string(), Value::Bool(spec.is_sparse));
    object.properties.insert(
        SCALE_FACTOR_FIELD.to_string(),
        complex_scalar_to_value(spec.scale),
    );
    if let Some(tol) = spec.rank_tolerance {
        object
            .properties
            .insert(RANK_TOLERANCE_FIELD.to_string(), Value::Num(tol));
    }
    Ok(object)
}

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
            .properties
            .get(field)
            .cloned()
            .ok_or_else(|| internal(format!("decomposition object is missing {field}"))),
        _ => Err(invalid(format!("unknown decomposition property '{field}'"))),
    }
}

pub(super) fn object_to_spec(value: &Value) -> BuiltinResult<DecompositionSpec> {
    let object = expect_object(value)?;
    let matrix = object
        .properties
        .get(MATRIX_FIELD)
        .cloned()
        .ok_or_else(|| internal("decomposition object is missing coefficient matrix"))?;
    let decomp_type = object
        .properties
        .get(TYPE_FIELD)
        .and_then(text_from_value)
        .map(|value| parse_decomposition_type(&value))
        .transpose()?
        .ok_or_else(|| internal("decomposition object is missing Type"))?;
    let check_condition = object
        .properties
        .get(CHECK_CONDITION_FIELD)
        .map(|value| bool_scalar(value, CHECK_CONDITION_FIELD))
        .transpose()?
        .unwrap_or(true);
    let is_conjugate_transposed = object
        .properties
        .get(IS_CONJUGATE_TRANSPOSED_FIELD)
        .map(|value| bool_scalar(value, IS_CONJUGATE_TRANSPOSED_FIELD))
        .transpose()?
        .unwrap_or(false);
    let is_sparse = object
        .properties
        .get(IS_SPARSE_FIELD)
        .map(|value| bool_scalar(value, IS_SPARSE_FIELD))
        .transpose()?
        .unwrap_or(false);
    let scale = object
        .properties
        .get(SCALE_FACTOR_FIELD)
        .map(|value| complex_scalar(value, SCALE_FACTOR_FIELD))
        .transpose()?
        .unwrap_or((1.0, 0.0));
    let rank_tolerance = object
        .properties
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
    })
}

pub(super) fn expect_object(value: &Value) -> BuiltinResult<&ObjectInstance> {
    match value {
        Value::Object(object) if object.is_class(CLASS_IDENTITY) => Ok(object),
        _ => Err(invalid("receiver must be a decomposition object")),
    }
}

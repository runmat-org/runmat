use super::*;
pub(super) fn parse_subscript_levels(value: &Value) -> crate::BuiltinResult<Vec<SubscriptLevel>> {
    match value {
        Value::Struct(struct_value) => Ok(vec![parse_subscript_level(struct_value)?]),
        Value::StructArray(array) => {
            let mut levels = Vec::with_capacity(array.len());
            for struct_value in array.elements() {
                levels.push(parse_subscript_level(&struct_value.to_owned())?);
            }
            if levels.is_empty() {
                return Err(num_args_error(
                    &NUM_ARGUMENTS_ERROR_SUBSTRUCT,
                    "substruct array must not be empty",
                ));
            }
            Ok(levels)
        }
        other => Err(num_args_error(
            &NUM_ARGUMENTS_ERROR_SUBSTRUCT,
            format!("S must be a struct or structure array, got {other:?}"),
        )),
    }
}

pub(super) fn parse_subscript_level(
    struct_value: &runmat_value::StructValue,
) -> crate::BuiltinResult<SubscriptLevel> {
    let kind_value = struct_value.fields.get("type").ok_or_else(|| {
        num_args_error(
            &NUM_ARGUMENTS_ERROR_SUBSTRUCT,
            "substruct level is missing field 'type'",
        )
    })?;
    let kind = String::try_from(kind_value).map_err(|err| {
        num_args_error(
            &NUM_ARGUMENTS_ERROR_SUBSTRUCT,
            format!("substruct type must be text: {err}"),
        )
    })?;
    let subs = struct_value.fields.get("subs").cloned().ok_or_else(|| {
        num_args_error(
            &NUM_ARGUMENTS_ERROR_SUBSTRUCT,
            "substruct level is missing field 'subs'",
        )
    })?;
    let kind = kind.trim().to_string();
    if !matches!(
        kind.as_str(),
        crate::OBJECT_INDEX_PAREN | crate::OBJECT_INDEX_BRACE | crate::OBJECT_INDEX_MEMBER
    ) {
        return Err(num_args_error(
            &NUM_ARGUMENTS_ERROR_SUBSTRUCT,
            format!("unsupported substruct type '{kind}'"),
        ));
    }
    Ok(SubscriptLevel { kind, subs })
}

pub(super) fn brace_selection_count(target: &Value, subs: &Value) -> crate::BuiltinResult<usize> {
    let Value::Cell(subscripts) = subs else {
        return Err(num_args_error(
            &NUM_ARGUMENTS_ERROR_SUBSTRUCT,
            "brace or parentheses subscripts must be stored in a cell array",
        ));
    };
    if subscripts.data.is_empty() {
        return Ok(0);
    }
    if subscripts.data.len() == 1 {
        return single_subscript_count(target, &subscripts.data[0], 0, true);
    }
    let mut count = 1usize;
    for (dim, subscript) in subscripts.data.iter().enumerate() {
        count = count
            .checked_mul(single_subscript_count(target, subscript, dim, false)?)
            .ok_or_else(|| {
                num_args_error(
                    &NUM_ARGUMENTS_ERROR_SUBSTRUCT,
                    "subscript selection count overflowed",
                )
            })?;
    }
    Ok(count)
}

pub(super) fn single_subscript_count(
    target: &Value,
    subscript: &Value,
    dim: usize,
    linear_indexing: bool,
) -> crate::BuiltinResult<usize> {
    match subscript {
        Value::String(value) if value.trim() == ":" => {
            target_colon_extent(target, dim, linear_indexing)
        }
        Value::CharArray(chars) if chars.data.iter().collect::<String>().trim() == ":" => {
            target_colon_extent(target, dim, linear_indexing)
        }
        Value::StringArray(array) if array.data.len() == 1 && array.data[0].trim() == ":" => {
            target_colon_extent(target, dim, linear_indexing)
        }
        Value::Bool(flag) => Ok(usize::from(*flag)),
        Value::Num(value) => {
            validate_positive_integer_subscript(*value)?;
            Ok(1)
        }
        Value::Int(value) => {
            if value.try_to_u64().is_some_and(|value| value >= 1) {
                Ok(1)
            } else {
                Err(num_args_error(
                    &NUM_ARGUMENTS_ERROR_SUBSTRUCT,
                    "numeric subscripts must be finite positive integers",
                ))
            }
        }
        Value::Tensor(tensor) => {
            for index in 0..tensor.len() {
                let value = tensor.numeric_value_at(index).ok_or_else(|| {
                    num_args_error(
                        &NUM_ARGUMENTS_ERROR_SUBSTRUCT,
                        "numeric subscript storage is invalid",
                    )
                })?;
                match value {
                    NumericScalar::F64(value) => validate_positive_integer_subscript(value)?,
                    NumericScalar::F32(value) => {
                        validate_positive_integer_subscript(f64::from(value))?
                    }
                    value => {
                        if value
                            .into_int_value()
                            .and_then(|value| value.try_to_u64())
                            .is_none_or(|value| value < 1)
                        {
                            return Err(num_args_error(
                                &NUM_ARGUMENTS_ERROR_SUBSTRUCT,
                                "numeric subscripts must be finite positive integers",
                            ));
                        }
                    }
                }
            }
            checked_shape_element_count(&tensor.shape)
        }
        Value::LogicalArray(array) => Ok(array.data.iter().filter(|&&bit| bit != 0).count()),
        Value::StringArray(_) => Err(num_args_error(
            &NUM_ARGUMENTS_ERROR_SUBSTRUCT,
            "string-array subscripts are only supported for ':'",
        )),
        Value::ComplexTensor(_) => Err(num_args_error(
            &NUM_ARGUMENTS_ERROR_SUBSTRUCT,
            "complex subscripts are not valid cell-index selectors",
        )),
        Value::Cell(_) => Err(num_args_error(
            &NUM_ARGUMENTS_ERROR_SUBSTRUCT,
            "nested cell subscripts are not valid cell-index selectors",
        )),
        Value::GpuTensor(_) => Err(num_args_error(
            &NUM_ARGUMENTS_ERROR_SUBSTRUCT,
            "gpuArray subscripts cannot be validated without gathering",
        )),
        other => Err(num_args_error(
            &NUM_ARGUMENTS_ERROR_SUBSTRUCT,
            format!("unsupported subscript value {other:?}"),
        )),
    }
}

pub(super) fn validate_positive_integer_subscript(value: f64) -> crate::BuiltinResult<()> {
    if value.is_finite() && value >= 1.0 && (value.round() - value).abs() <= f64::EPSILON {
        Ok(())
    } else {
        Err(num_args_error(
            &NUM_ARGUMENTS_ERROR_SUBSTRUCT,
            "numeric subscripts must be finite positive integers",
        ))
    }
}

pub(super) fn checked_shape_element_count(shape: &[usize]) -> crate::BuiltinResult<usize> {
    let mut count = 1usize;
    for &dim in shape {
        count = count.checked_mul(dim).ok_or_else(|| {
            num_args_error(
                &NUM_ARGUMENTS_ERROR_SUBSTRUCT,
                "shape element count overflowed",
            )
        })?;
    }
    Ok(count)
}

pub(super) fn target_colon_extent(
    target: &Value,
    dim: usize,
    linear_indexing: bool,
) -> crate::BuiltinResult<usize> {
    if linear_indexing {
        default_target_numel(target)
    } else {
        Ok(target_shape(target)
            .and_then(|shape| {
                if shape.is_empty() {
                    None
                } else {
                    Some(shape.get(dim).copied().unwrap_or(1))
                }
            })
            .unwrap_or(1))
    }
}

pub(super) fn target_shape(target: &Value) -> Option<Vec<usize>> {
    match target {
        Value::Cell(cell) => Some(cell.shape.clone()),
        Value::StructArray(array) => Some(array.shape().to_vec()),
        Value::ObjectArray(array) => Some(array.shape().to_vec()),
        Value::Tensor(tensor) => Some(tensor.shape.clone()),
        Value::ComplexTensor(tensor) => Some(tensor.shape.clone()),
        Value::LogicalArray(array) => Some(array.shape.clone()),
        Value::StringArray(array) => Some(array.shape.clone()),
        Value::SparseTensor(sparse) => Some(sparse.shape()),
        Value::CharArray(chars) => Some(vec![chars.rows, chars.cols]),
        Value::GpuTensor(handle) => Some(handle.shape.clone()),
        Value::Object(_) | Value::HandleObject(_) => Some(vec![1, 1]),
        _ => None,
    }
}

pub(super) fn default_target_numel(target: &Value) -> crate::BuiltinResult<usize> {
    match target_shape(target) {
        Some(shape) => checked_shape_element_count(&shape),
        None => Ok(1),
    }
}

pub(super) fn default_num_arguments(
    target: &Value,
    levels: &[SubscriptLevel],
    _context: IndexingContext,
) -> crate::BuiltinResult<usize> {
    let Some(first) = levels.first() else {
        return Err(num_args_error(
            &NUM_ARGUMENTS_ERROR_SUBSTRUCT,
            "substruct must include at least one indexing level",
        ));
    };
    if first.kind == crate::OBJECT_INDEX_BRACE {
        return brace_selection_count(target, &first.subs);
    }
    Ok(1)
}

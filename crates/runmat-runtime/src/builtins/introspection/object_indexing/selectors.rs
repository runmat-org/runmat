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
    let kind = crate::object::indexing::ObjectIndexKind::from_protocol_name(kind.trim())
        .ok_or_else(|| {
            num_args_error(
                &NUM_ARGUMENTS_ERROR_SUBSTRUCT,
                format!("unsupported substruct type '{}'", kind.trim()),
            )
        })?;
    Ok(SubscriptLevel { kind, subs })
}

pub(super) fn brace_selection_count(target: &Value, subs: &Value) -> crate::BuiltinResult<usize> {
    let Value::Cell(subscripts) = subs else {
        return Err(num_args_error(
            &NUM_ARGUMENTS_ERROR_SUBSTRUCT,
            "brace or parentheses subscripts must be stored in a cell array",
        ));
    };
    selection_count(subscripts.data().len(), |dim| {
        single_subscript_count(
            target,
            &subscripts.data()[dim],
            dim,
            subscripts.data().len(),
        )
    })
}

fn selection_count(
    width: usize,
    count_at: impl Fn(usize) -> crate::BuiltinResult<usize>,
) -> crate::BuiltinResult<usize> {
    if width == 0 {
        return Ok(0);
    }
    if width == 1 {
        return count_at(0);
    }
    let mut count = 1usize;
    for dim in 0..width {
        count = count.checked_mul(count_at(dim)?).ok_or_else(|| {
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
    selector_count: usize,
) -> crate::BuiltinResult<usize> {
    match subscript {
        Value::String(runmat_value::StringElement::Text(value)) if value.trim() == ":" => {
            target_colon_extent(target, dim, selector_count)
        }
        Value::CharArray(chars) => {
            // Compare the same decoded/trimmed text without materializing it.
            // Always decode the entire source: malformed UTF-16 has the same
            // precedence even if an earlier character already rules out ':'.
            let mut colons = 0usize;
            let mut other = false;
            let mut unit_index = 0usize;
            for decoded in char::decode_utf16(chars.data().iter().map(|unit| unit.raw())) {
                let character = decoded.map_err(|source| {
                    let mut error = num_args_error(
                        &NUM_ARGUMENTS_ERROR_SUBSTRUCT,
                        "index selector is not valid UTF-16 text",
                    );
                    error.source = Some(Box::new(runmat_value::Utf16DecodeError {
                        index: unit_index,
                        unpaired_surrogate: source.unpaired_surrogate(),
                    }));
                    error
                })?;
                unit_index += character.len_utf16();
                if character == ':' {
                    colons = colons.saturating_add(1);
                } else if !character.is_whitespace() {
                    other = true;
                }
            }
            if colons == 1 && !other {
                target_colon_extent(target, dim, selector_count)
            } else {
                Err(num_args_error(
                    &NUM_ARGUMENTS_ERROR_SUBSTRUCT,
                    "index selector must be ':' or numeric",
                ))
            }
        }
        Value::StringArray(array)
            if array.data().len() == 1
                && array.data()[0]
                    .as_str()
                    .is_some_and(|text| text.trim() == ":") =>
        {
            target_colon_extent(target, dim, selector_count)
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
            checked_shape_element_count(tensor.shape())
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
    selector_count: usize,
) -> crate::BuiltinResult<usize> {
    let shape = target_shape_borrowed(target).unwrap_or(
        crate::builtins::common::shape::ValueDimensions::Inline([1, 1]),
    );
    crate::indexing::shape::selector_extent(&shape, selector_count, dim)
        .map_err(|error| num_args_error(&NUM_ARGUMENTS_ERROR_SUBSTRUCT, error.message()))
}

pub(super) fn target_shape_borrowed(
    target: &Value,
) -> Option<crate::builtins::common::shape::ValueDimensions<'_>> {
    match target {
        Value::Cell(_)
        | Value::StructArray(_)
        | Value::ObjectArray(_)
        | Value::Tensor(_)
        | Value::ComplexTensor(_)
        | Value::LogicalArray(_)
        | Value::StringArray(_)
        | Value::SparseTensor(_)
        | Value::CharArray(_)
        | Value::GpuTensor(_)
        | Value::Object(_)
        | Value::HandleObject(_) => Some(
            crate::builtins::common::shape::stored_dimensions_borrowed(target),
        ),
        _ => None,
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
    default_level_cardinality(first.kind, || brace_selection_count(target, &first.subs))
}

pub(super) fn default_path_num_arguments(
    target: &Value,
    path: &crate::object::indexing::ObjectSubscriptPath,
) -> crate::BuiltinResult<usize> {
    use crate::object::indexing::{ObjectIndexComponent, ObjectIndexSelector};
    let first = path.steps().first().ok_or_else(|| {
        num_args_error(
            &NUM_ARGUMENTS_ERROR_SUBSTRUCT,
            "substruct must include at least one indexing level",
        )
    })?;
    default_level_cardinality(first.kind(), || match first.selector() {
        ObjectIndexSelector::ScalarIndices { indices } => selection_count(indices.len(), |dim| {
            // This is the existing public protocol's exact usize -> Num
            // projection, not a new integer selector policy.
            single_subscript_count(target, &Value::Num(indices[dim] as f64), dim, indices.len())
        }),
        ObjectIndexSelector::IndexValues { components } => {
            selection_count(components.len(), |dim| match &components[dim] {
                ObjectIndexComponent::Colon => target_colon_extent(target, dim, components.len()),
                ObjectIndexComponent::End => Err(crate::object::indexing::unresolved_end()),
                ObjectIndexComponent::Value(value) => {
                    single_subscript_count(target, value, dim, components.len())
                }
            })
        }
        ObjectIndexSelector::Member(_) => Err(num_args_error(
            &NUM_ARGUMENTS_ERROR_SUBSTRUCT,
            "brace or parentheses subscripts must be stored in a cell array",
        )),
    })
}

fn default_level_cardinality(
    kind: crate::object::indexing::ObjectIndexKind,
    brace: impl FnOnce() -> crate::BuiltinResult<usize>,
) -> crate::BuiltinResult<usize> {
    match kind {
        crate::object::indexing::ObjectIndexKind::Brace => brace(),
        crate::object::indexing::ObjectIndexKind::Paren
        | crate::object::indexing::ObjectIndexKind::Member => Ok(1),
    }
}

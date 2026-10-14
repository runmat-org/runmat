use super::*;
/// Resolve the number of brace-assignment destinations using the same
/// `numArgumentsFromSubscript` protocol as public object indexing.
pub async fn brace_assignment_cardinality(
    target: &Value,
    subscripts: Vec<Value>,
) -> crate::BuiltinResult<usize> {
    let subscript_count = subscripts.len();
    let subscript_values = Value::Cell(
        runmat_value::CellArray::new(subscripts, 1, subscript_count)
            .map_err(|error| num_args_error(&NUM_ARGUMENTS_ERROR_SUBSTRUCT, error))?,
    );
    assignment_cardinality_for_level(
        target,
        crate::object::indexing::ObjectIndexKind::Brace,
        subscript_values,
    )
    .await
}

/// Resolve the number of member-assignment destinations through the public
/// object indexing protocol. Plain structures use their aggregate cardinality;
/// class-defined objects may override it with `numArgumentsFromSubscript`.
pub async fn member_assignment_cardinality(
    target: &Value,
    member: String,
) -> crate::BuiltinResult<usize> {
    assignment_cardinality_for_level(
        target,
        crate::object::indexing::ObjectIndexKind::Member,
        Value::String(member),
    )
    .await
}

pub struct PreparedObjectAssignmentCardinality {
    pub count: usize,
    pub method: Option<crate::object::protocol::ResolvedObjectMethod>,
}

pub async fn object_path_assignment_cardinality(
    target: &Value,
    path: &crate::object::indexing::ObjectSubscriptPath,
) -> crate::BuiltinResult<PreparedObjectAssignmentCardinality> {
    object_path_cardinality(
        target,
        path,
        IndexingContext::Assignment,
        &crate::object::protocol::ObjectAccessContext::default(),
    )
    .await
}

pub async fn object_path_cardinality(
    target: &Value,
    path: &crate::object::indexing::ObjectSubscriptPath,
    indexing_context: IndexingContext,
    access: &crate::object::protocol::ObjectAccessContext,
) -> crate::BuiltinResult<PreparedObjectAssignmentCardinality> {
    object_path_cardinality_with_context(None, target, path, indexing_context, access).await
}

pub async fn object_path_cardinality_in_context(
    runtime: &crate::context::RuntimeContext,
    target: &Value,
    path: &crate::object::indexing::ObjectSubscriptPath,
    indexing_context: IndexingContext,
    access: &crate::object::protocol::ObjectAccessContext,
) -> crate::BuiltinResult<PreparedObjectAssignmentCardinality> {
    object_path_cardinality_with_context(Some(runtime), target, path, indexing_context, access)
        .await
}

async fn object_path_cardinality_with_context(
    runtime: Option<&crate::context::RuntimeContext>,
    target: &Value,
    path: &crate::object::indexing::ObjectSubscriptPath,
    indexing_context: IndexingContext,
    access: &crate::object::protocol::ObjectAccessContext,
) -> crate::BuiltinResult<PreparedObjectAssignmentCardinality> {
    ensure_indexing_context_classes_registered();
    let descriptor = path.to_standard_substruct_value()?;
    let context = indexing_context_value(indexing_context);
    let resolution = crate::object::protocol::resolve_object_protocol(
        target,
        crate::object::protocol::ObjectProtocol::NumArgumentsFromSubscript,
        access,
    )?;
    if let Some(method) = resolution {
        let arguments = vec![target.clone(), descriptor.clone(), context];
        let value = match runtime {
            Some(runtime) => {
                crate::object::protocol::invoke_resolved_object_method_in_context(
                    runtime, &method, arguments, 1,
                )
                .await?
            }
            None => {
                crate::object::protocol::invoke_resolved_object_method(&method, arguments, 1)
                    .await?
            }
        };
        let value =
            value.try_map_owned(require_single_sequence, |error| error.into_runtime_error())?;
        return Ok(PreparedObjectAssignmentCardinality {
            count: exact_output_count(value.value())?,
            method: Some(method),
        });
    }
    let levels = parse_subscript_levels(&descriptor)?;
    Ok(PreparedObjectAssignmentCardinality {
        count: default_num_arguments(target, &levels, indexing_context)?,
        method: None,
    })
}

pub(super) async fn assignment_cardinality_for_level(
    target: &Value,
    kind: crate::object::indexing::ObjectIndexKind,
    subs: Value,
) -> crate::BuiltinResult<usize> {
    ensure_indexing_context_classes_registered();
    let mut level = runmat_value::StructValue::new();
    level.insert("type", Value::String(kind.protocol_name().to_string()));
    level.insert("subs", subs.clone());
    let descriptor = Value::Struct(level);
    let context = indexing_context_value(IndexingContext::Assignment);
    if crate::object_receiver_class_name(target).is_some() {
        if let Some(value) =
            dispatch_num_arguments_overload(None, target.clone(), descriptor, context).await?
        {
            return exact_output_count(&value);
        }
    }
    if kind == crate::object::indexing::ObjectIndexKind::Member {
        return crate::object::resolve::member_sequence_cardinality(target);
    }
    let level = SubscriptLevel { kind, subs };
    default_num_arguments(target, &[level], IndexingContext::Assignment)
}

pub(super) fn exact_output_count(value: &Value) -> crate::BuiltinResult<usize> {
    let numeric = match value {
        Value::Num(value) => *value,
        Value::Int(value) => {
            return value
                .try_to_u64()
                .and_then(|value| usize::try_from(value).ok())
                .ok_or_else(|| {
                    num_args_error(
                        &NUM_ARGUMENTS_ERROR_COUNT,
                        "output count must be a nonnegative exact integer",
                    )
                });
        }
        Value::Tensor(tensor) if tensor.len() == 1 => {
            return tensor
                .numeric_value_at(0)
                .expect("validated scalar numeric tensor")
                .into_int_value()
                .map(|value| exact_output_count(&Value::Int(value)))
                .unwrap_or_else(|| {
                    exact_output_count(&Value::Num(
                        tensor
                            .numeric_value_at(0)
                            .expect("scalar")
                            .materialize_f64(),
                    ))
                })
        }
        _ => {
            return Err(num_args_error(
                &NUM_ARGUMENTS_ERROR_COUNT,
                "output count must be a nonnegative exact numeric scalar",
            ));
        }
    };
    if !numeric.is_finite() || numeric < 0.0 || numeric.fract() != 0.0 {
        return Err(num_args_error(
            &NUM_ARGUMENTS_ERROR_COUNT,
            "output count must be a nonnegative exact integer",
        ));
    }
    if numeric >= (usize::MAX as f64 + 1.0) {
        return Err(num_args_error(
            &NUM_ARGUMENTS_ERROR_COUNT,
            "output count exceeds the supported limit",
        ));
    }
    Ok(numeric as usize)
}

pub(super) fn require_single_sequence(
    sequence: runmat_value::ValueSequence,
) -> crate::BuiltinResult<Value> {
    let mut values = sequence.resolve(
        runmat_types::SequenceUse::RequireSingle,
        crate::sequence::SequenceResolutionContext::default(),
    )?;
    Ok(values.remove(0))
}

pub(super) fn exact_count_as_f64(count: usize) -> crate::BuiltinResult<f64> {
    if count != 0 {
        let significant_bits = usize::BITS - count.leading_zeros();
        let discarded_bits = significant_bits.saturating_sub(f64::MANTISSA_DIGITS);
        if count.trailing_zeros() < discarded_bits {
            return Err(num_args_error(
                &NUM_ARGUMENTS_ERROR_COUNT,
                format!("argument count {count} cannot be represented exactly as double"),
            ));
        }
    }
    Ok(count as f64)
}

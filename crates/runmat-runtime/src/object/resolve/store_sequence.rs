use super::*;

pub fn member_sequence_cardinality(base: &Value) -> Result<usize, RuntimeError> {
    match base {
        Value::Struct(_) | Value::Object(_) | Value::HandleObject(_) => Ok(1),
        Value::StructArray(array) => Ok(array.len()),
        Value::ObjectArray(array) => Ok(array.len()),
        _ => Err(mex(
            "RunMat:CommaSeparatedListDestination",
            "comma-separated member assignment requires a structure or object aggregate",
        )),
    }
}

pub async fn store_member_sequence_traced(
    base: Value,
    field: String,
    values: Vec<Value>,
    caller_function_name: Option<&str>,
) -> Result<Value, RuntimeError> {
    store_member_sequence_traced_with_context(None, base, field, values, caller_function_name).await
}

pub async fn store_member_sequence_traced_in_context(
    runtime: &crate::context::RuntimeContext,
    base: Value,
    field: String,
    values: Vec<Value>,
    caller_function_name: Option<&str>,
) -> Result<Value, RuntimeError> {
    store_member_sequence_traced_with_context(
        Some(runtime),
        base,
        field,
        values,
        caller_function_name,
    )
    .await
}

async fn store_member_sequence_traced_with_context(
    runtime: Option<&crate::context::RuntimeContext>,
    base: Value,
    field: String,
    values: Vec<Value>,
    caller_function_name: Option<&str>,
) -> Result<Value, RuntimeError> {
    let work =
        store_member_sequence_traced_in_scope(runtime, base, field, values, caller_function_name);
    match runtime {
        Some(runtime) => runtime.scope(work).await,
        None => work.await,
    }
}

async fn store_member_sequence_traced_in_scope(
    runtime: Option<&crate::context::RuntimeContext>,
    base: Value,
    field: String,
    values: Vec<Value>,
    caller_function_name: Option<&str>,
) -> Result<Value, RuntimeError> {
    match base {
        Value::StructArray(array) => crate::aggregate::structure::assign_member_values(
            array,
            field,
            values,
            runmat_gc::gc_record_write,
        ),
        Value::ObjectArray(array) => {
            if values.len() != array.len() {
                return Err(mex(
                    "RunMat:CommaSeparatedListAssignmentArity",
                    &format!(
                        "member assignment requires exactly one value per destination element (expected {}, received {})",
                        array.len(),
                        values.len()
                    ),
                ));
            }
            let aggregate = Value::ObjectArray(array);
            let resolution = crate::object::dispatch::resolve_object_index_protocol(
                &aggregate,
                ObjectIndexOp::Subsasgn,
                caller_function_name,
            )?;
            if let crate::object::protocol::ProtocolResolution::Method(method) = resolution {
                let path = crate::object::indexing::ObjectSubscriptPath::single(
                    crate::object::indexing::ObjectSubscript::member(field),
                );
                return match runtime {
                    Some(runtime) => {
                        crate::object::protocol::invoke_resolved_object_assignment_in_context(
                            runtime, &method, aggregate, path, values,
                        )
                        .await
                    }
                    None => {
                        crate::object::protocol::invoke_resolved_object_assignment(
                            &method, aggregate, path, values,
                        )
                        .await
                    }
                };
            }
            let Value::ObjectArray(array) = aggregate else {
                unreachable!("aggregate was constructed as an object array")
            };
            let class_name = array.class_name().clone();
            let shape = array.shape().to_vec();
            let mut updated = Vec::with_capacity(values.len());
            for (element, value) in array.into_data().into_iter().zip(values) {
                updated.push(match runtime {
                    Some(runtime) => {
                        store_member_traced_in_context(
                            runtime,
                            element,
                            field.clone(),
                            value,
                            false,
                            caller_function_name,
                        )
                        .await?
                    }
                    None => {
                        store_member_traced(
                            element,
                            field.clone(),
                            value,
                            false,
                            caller_function_name,
                        )
                        .await?
                    }
                });
            }
            runmat_value::ObjectArray::new(class_name, updated, shape)
                .map(Value::ObjectArray)
                .map_err(RuntimeError::from)
        }
        base if values.len() == 1 => {
            let value = values.into_iter().next().expect("validated singleton");
            match runtime {
                Some(runtime) => {
                    store_member_traced_in_context(
                        runtime,
                        base,
                        field,
                        value,
                        false,
                        caller_function_name,
                    )
                    .await
                }
                None => store_member_traced(base, field, value, false, caller_function_name).await,
            }
        }
        _ => Err(mex(
            "RunMat:CommaSeparatedListAssignmentArity",
            "member assignment requires exactly one value per destination element",
        )),
    }
}

/// Store a member while applying the canonical GC write barrier.
pub async fn store_member_traced(
    base: Value,
    field: String,
    rhs: Value,
    allow_init: bool,
    caller_function_name: Option<&str>,
) -> Result<Value, RuntimeError> {
    store_member(
        base,
        field,
        rhs,
        allow_init,
        caller_function_name,
        runmat_gc::gc_record_write,
    )
    .await
}

pub async fn store_member_traced_in_context(
    context: &crate::context::RuntimeContext,
    base: Value,
    field: String,
    rhs: Value,
    allow_init: bool,
    caller_function_name: Option<&str>,
) -> Result<Value, RuntimeError> {
    store_member_with_context(
        Some(context),
        base,
        field,
        rhs,
        allow_init,
        caller_function_name,
        runmat_gc::gc_record_write,
    )
    .await
}

/// Store a dynamically named member while applying the canonical GC barrier.
pub async fn store_member_dynamic_traced(
    base: Value,
    name: String,
    rhs: Value,
    allow_init: bool,
    caller_function_name: Option<&str>,
) -> Result<Value, RuntimeError> {
    store_member_dynamic(
        base,
        name,
        rhs,
        allow_init,
        caller_function_name,
        runmat_gc::gc_record_write,
    )
    .await
}

pub async fn store_member_dynamic_traced_in_context(
    context: &crate::context::RuntimeContext,
    base: Value,
    name: String,
    rhs: Value,
    allow_init: bool,
    caller_function_name: Option<&str>,
) -> Result<Value, RuntimeError> {
    store_member_with_context(
        Some(context),
        base,
        name,
        rhs,
        allow_init,
        caller_function_name,
        runmat_gc::gc_record_write,
    )
    .await
}

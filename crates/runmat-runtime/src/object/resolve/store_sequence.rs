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
) -> Result<RootedValue, RuntimeError> {
    store_member_sequence_traced_with_context(None, base, field, values, caller_function_name).await
}

pub async fn store_member_sequence_traced_in_context(
    runtime: &crate::context::RuntimeContext,
    base: Value,
    field: String,
    values: Vec<Value>,
    caller_function_name: Option<&str>,
) -> Result<RootedValue, RuntimeError> {
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
) -> Result<RootedValue, RuntimeError> {
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
) -> Result<RootedValue, RuntimeError> {
    let _inputs_root = root_assignment_inputs(&base, &values)?;
    match base {
        Value::StructArray(array) => {
            RootedValue::new(crate::aggregate::structure::assign_member_values(
                array,
                field,
                values,
                runmat_gc::gc_record_write,
            )?)
            .map_err(|error| error.into_runtime_error())
        }
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
            if let Some(method) = resolution {
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
            let selected_provenance = array.selected_provenance().to_vec();
            let unselected_provenance = array.has_unselected_provenance();
            let mut updated = Vec::with_capacity(values.len());
            let roots = crate::call::arguments::roots::ExpansionRoots::new()?;
            for (element, value) in array.into_data().into_iter().zip(values) {
                let selected = match runtime {
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
                };
                selected.with_owned(|value| {
                    roots.retain_values(std::slice::from_ref(&value));
                    updated.push(value);
                });
            }
            let mut result = runmat_value::ObjectArray::new(class_name, updated, shape)
                .map_err(RuntimeError::from)?;
            if result.is_empty() {
                result = result
                    .with_selected_provenance(selected_provenance)
                    .map_err(RuntimeError::from)?;
                if unselected_provenance {
                    result = result.with_unselected_provenance();
                }
            }
            RootedValue::new(Value::ObjectArray(result)).map_err(|error| error.into_runtime_error())
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

fn root_assignment_inputs(
    base: &Value,
    values: &[Value],
) -> Result<RootedValueLease, RuntimeError> {
    RootedValueLease::for_values(std::iter::once(base).chain(values.iter()))
        .map_err(|error| error.into_runtime_error())
}

#[cfg(all(test, not(target_arch = "wasm32")))]
mod root_tests {
    use super::*;
    use runmat_value::HandleRef;

    #[test]
    fn later_rhs_remains_live_while_earlier_assignment_can_collect() {
        runmat_gc::gc_test_context(|| {
            let first = runmat_gc::gc_allocate_rooted(Value::String("first".into()))
                .expect("first rooted target");
            let later = runmat_gc::gc_allocate_rooted(Value::String("later".into()))
                .expect("later rooted target");
            let later_target = later.handle();
            let rhs = [first.handle(), later_target]
                .into_iter()
                .map(|target| {
                    Value::HandleObject(HandleRef {
                        class_name: "StoreRhs".into(),
                        target,
                        valid: true,
                    })
                })
                .collect::<Vec<_>>();
            let inputs = root_assignment_inputs(&Value::Num(0.0), &rhs)
                .expect("admit every RHS before any callback");
            first.unroot().expect("release first producer");
            later.unroot().expect("release later producer");

            // Models a reentrant collection in the first element's store:
            // the second RHS has not yet been passed to its scalar store.
            runmat_gc::gc_collect_major().expect("collect during first store");
            assert_eq!(
                runmat_gc::gc_clone_value(&later_target).expect("later RHS is still rooted"),
                Value::String("later".into()),
            );

            drop(inputs);
            runmat_gc::gc_collect_major().expect("collect after assignment retires");
            assert!(runmat_gc::gc_clone_value(&later_target).is_err());
        });
    }
}

/// Store a member while applying the canonical GC write barrier.
pub async fn store_member_traced(
    base: Value,
    field: String,
    rhs: Value,
    allow_init: bool,
    caller_function_name: Option<&str>,
) -> Result<RootedValue, RuntimeError> {
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
) -> Result<RootedValue, RuntimeError> {
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
) -> Result<RootedValue, RuntimeError> {
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
) -> Result<RootedValue, RuntimeError> {
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

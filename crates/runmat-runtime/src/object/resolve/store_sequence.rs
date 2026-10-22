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
    let values = crate::sequence::ValueSequence::comma_separated(values)
        .map_err(crate::sequence::sequence_error_to_runtime)?;
    let roots = sequence_store_roots(runtime, &base, values.as_slice())?;
    let field = crate::call::argument_text::adopt_string(field, runtime)?;
    store_member_sequence_scoped(
        runtime,
        MemberSequenceStore {
            base,
            field,
            values,
            _roots: roots,
        },
        caller_function_name,
    )
    .await
}

pub async fn store_member_sequence_values(
    base: Value,
    field: runmat_value::Utf8Text,
    values: crate::sequence::ValueSequence,
    caller: Option<&str>,
) -> Result<RootedValue, RuntimeError> {
    store_member_sequence_values_with_context(None, base, field, values, caller).await
}

pub async fn store_member_sequence_values_in_context(
    runtime: &crate::context::RuntimeContext,
    base: Value,
    field: runmat_value::Utf8Text,
    values: crate::sequence::ValueSequence,
    caller: Option<&str>,
) -> Result<RootedValue, RuntimeError> {
    store_member_sequence_values_with_context(Some(runtime), base, field, values, caller).await
}

struct MemberSequenceStore {
    base: Value,
    field: runmat_value::Utf8Text,
    values: crate::sequence::ValueSequence,
    _roots: RootedValueLease,
}

async fn store_member_sequence_values_with_context(
    runtime: Option<&crate::context::RuntimeContext>,
    base: Value,
    field: runmat_value::Utf8Text,
    values: crate::sequence::ValueSequence,
    caller: Option<&str>,
) -> Result<RootedValue, RuntimeError> {
    let roots = sequence_store_roots(runtime, &base, values.as_slice())?;
    store_member_sequence_scoped(
        runtime,
        MemberSequenceStore {
            base,
            field,
            values,
            _roots: roots,
        },
        caller,
    )
    .await
}

async fn store_member_sequence_scoped(
    runtime: Option<&crate::context::RuntimeContext>,
    request: MemberSequenceStore,
    caller: Option<&str>,
) -> Result<RootedValue, RuntimeError> {
    let work = store_member_sequence_traced_in_scope(runtime, request, caller);
    match runtime {
        Some(runtime) => runtime.scope(work).await,
        None => work.await,
    }
}

async fn store_member_sequence_traced_in_scope(
    runtime: Option<&crate::context::RuntimeContext>,
    request: MemberSequenceStore,
    caller_function_name: Option<&str>,
) -> Result<RootedValue, RuntimeError> {
    let MemberSequenceStore {
        base,
        field,
        values,
        _roots,
    } = request;
    match base {
        Value::StructArray(array) => crate::aggregate::structure::assign_member_sequence(
            runtime,
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
            if let Some(method) = resolution {
                let path = crate::object::indexing::ObjectSubscriptPath::single_in_context(
                    runtime,
                    crate::object::indexing::ObjectSubscript::member_retained(field),
                )?;
                return match runtime {
                    Some(runtime) => {
                        crate::object::protocol::invoke_resolved_object_assignment_sequence_in_context(
                            runtime, &method, aggregate, path, values,
                        )
                        .await
                    }
                    None => {
                        crate::object::protocol::invoke_resolved_object_assignment_sequence(
                            &method, aggregate, path, values,
                        )
                        .await
                    }
                };
            }
            let Value::ObjectArray(array) = aggregate else {
                unreachable!("aggregate was constructed as an object array")
            };
            let mut child_roots = match runtime {
                Some(runtime) => crate::execution::RootedValueChildren::in_context(runtime)
                    .map_err(|source| {
                        crate::context::host_allocation::host_allocation_runtime_error(
                            Some(runtime),
                            "RunMat:ObjectArray:Allocation",
                            "object-array output roots admission failed",
                            source,
                        )
                    })?,
                None => crate::execution::RootedValueChildren::new(),
            };
            let mut output =
                crate::execution::object_array::ObjectArrayConstruction::new_retaining_shape(
                    runtime,
                    crate::execution::object_array::Class::Owner(array.class_name_owner()),
                    array.len(),
                    array.shape_owner(),
                    array.selected_provenance().iter(),
                    array.has_unselected_provenance(),
                )?;
            let values = crate::execution::RootedValuesIteration::from_sequence(values);
            for (element, value) in array.data().iter().cloned().zip(values) {
                let selected = store::store_member_text_with_context(
                    runtime,
                    element,
                    field.clone(),
                    value,
                    false,
                    caller_function_name,
                    runmat_gc::gc_record_write,
                )
                .await?;
                let child = crate::execution::RootedValueMember::from_rooted(selected)
                    .into_child(&mut child_roots)
                    .map_err(|source| {
                        crate::context::host_allocation::host_allocation_runtime_error(
                            runtime,
                            "RunMat:ObjectArray:Allocation",
                            "object-array output roots admission failed",
                            source,
                        )
                    })?;
                output.push(child)?;
            }
            output.finish()
        }
        base if values.len() == 1 => {
            let value = crate::execution::RootedValuesIteration::from_sequence(values)
                .next()
                .expect("validated singleton");
            store::store_member_text_with_context(
                runtime,
                base,
                field,
                value,
                false,
                caller_function_name,
                runmat_gc::gc_record_write,
            )
            .await
        }
        _ => Err(mex(
            "RunMat:CommaSeparatedListAssignmentArity",
            "member assignment requires exactly one value per destination element",
        )),
    }
}

fn sequence_store_roots(
    runtime: Option<&crate::context::RuntimeContext>,
    base: &Value,
    values: &[Value],
) -> Result<RootedValueLease, RuntimeError> {
    match runtime {
        Some(runtime) => RootedValueLease::for_value_slices_in_context(
            &[std::slice::from_ref(base), values],
            runtime,
        ),
        None => root_assignment_inputs(base, values),
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

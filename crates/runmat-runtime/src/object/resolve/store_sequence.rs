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

pub fn store_member_sequence_traced<'a>(
    base: Value,
    field: String,
    values: Vec<Value>,
    caller_function_name: Option<&'a str>,
) -> impl std::future::Future<Output = Result<RootedValue, RuntimeError>> + 'a {
    store_member_sequence_traced_with_scope(
        OrdinaryMemberScope,
        base,
        field,
        values,
        MemberCaller::named(caller_function_name),
    )
}

pub fn store_member_sequence_traced_in_context<'a>(
    runtime: &'a crate::context::RuntimeContext,
    base: Value,
    field: String,
    values: Vec<Value>,
    caller_function_name: Option<&'a str>,
) -> impl std::future::Future<Output = Result<RootedValue, RuntimeError>> + 'a {
    store_member_sequence_traced_with_scope(
        ContextualMemberScope(runtime),
        base,
        field,
        values,
        MemberCaller::named(caller_function_name),
    )
}

async fn store_member_sequence_traced_with_scope<'a, M: MemberStoreScope<'a> + 'a>(
    scope: M,
    base: Value,
    field: String,
    values: Vec<Value>,
    caller_function_name: MemberCaller<'a>,
) -> Result<RootedValue, RuntimeError> {
    let runtime = scope.runtime();
    let values = crate::sequence::ValueSequence::comma_separated(values)
        .map_err(crate::sequence::sequence_error_to_runtime)?;
    let roots = sequence_store_roots(runtime, &base, values.as_slice())?;
    let field = crate::call::argument_text::adopt_string(field, runtime)?;
    store_member_sequence_scoped(
        scope,
        MemberSequenceStore {
            base,
            field,
            values,
            dispatch: ProtocolDispatch::Resolve,
            _roots: roots,
        },
        caller_function_name,
    )
    .await
}

pub fn store_member_sequence_values<'a>(
    base: Value,
    field: runmat_value::Utf8Text,
    values: crate::sequence::ValueSequence,
    caller: Option<&'a str>,
) -> impl std::future::Future<Output = Result<RootedValue, RuntimeError>> + 'a {
    store_member_sequence_values_with_scope(OrdinaryMemberScope, base, field, values, caller)
}

pub fn store_member_sequence_values_in_context<'a>(
    runtime: &'a crate::context::RuntimeContext,
    base: Value,
    field: runmat_value::Utf8Text,
    values: crate::sequence::ValueSequence,
    caller: Option<&'a str>,
) -> impl std::future::Future<Output = Result<RootedValue, RuntimeError>> + 'a {
    store_member_sequence_values_with_scope(
        ContextualMemberScope(runtime),
        base,
        field,
        values,
        caller,
    )
}

struct MemberSequenceStore {
    base: Value,
    field: runmat_value::Utf8Text,
    values: crate::sequence::ValueSequence,
    dispatch: ProtocolDispatch,
    _roots: RootedValueLease,
}

fn store_member_sequence_values_with_caller_scope<'a, M: MemberStoreScope<'a> + 'a>(
    scope: M,
    base: Value,
    field: runmat_value::Utf8Text,
    values: crate::sequence::ValueSequence,
    caller: MemberCaller<'a>,
) -> impl std::future::Future<Output = Result<RootedValue, RuntimeError>> + 'a {
    store_member_sequence_values_with_dispatch_scope(
        scope,
        base,
        field,
        values,
        caller,
        ProtocolDispatch::Resolve,
    )
}

async fn store_member_sequence_values_with_dispatch_scope<'a, M: MemberStoreScope<'a> + 'a>(
    scope: M,
    base: Value,
    field: runmat_value::Utf8Text,
    values: crate::sequence::ValueSequence,
    caller: MemberCaller<'a>,
    dispatch: ProtocolDispatch,
) -> Result<RootedValue, RuntimeError> {
    let runtime = scope.runtime();
    let roots = sequence_store_roots(runtime, &base, values.as_slice())?;
    store_member_sequence_scoped(
        scope,
        MemberSequenceStore {
            base,
            field,
            values,
            dispatch,
            _roots: roots,
        },
        caller,
    )
    .await
}

fn store_member_sequence_scoped<'a, M: MemberStoreScope<'a> + 'a>(
    scope: M,
    request: MemberSequenceStore,
    caller: MemberCaller<'a>,
) -> impl std::future::Future<Output = Result<RootedValue, RuntimeError>> + 'a {
    scope.scope(store_member_sequence_traced_in_scope(
        scope, request, caller,
    ))
}

async fn store_member_sequence_traced_in_scope<'a, M: MemberStoreScope<'a> + 'a>(
    scope: M,
    request: MemberSequenceStore,
    caller_function_name: MemberCaller<'a>,
) -> Result<RootedValue, RuntimeError> {
    let runtime = scope.runtime();
    let MemberSequenceStore {
        base,
        field,
        values,
        dispatch,
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
            let resolution = dispatch.resolve(
                &aggregate,
                crate::object::protocol::ObjectProtocol::Subsasgn,
                &caller_function_name.protocol_access(),
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
                let selected = store::store_member_text_with_dispatch_scope(
                    scope,
                    element,
                    field.clone(),
                    value,
                    caller_function_name,
                    runmat_gc::gc_record_write,
                    dispatch,
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
            store::store_member_text_with_dispatch_scope(
                scope,
                base,
                field,
                value,
                caller_function_name,
                runmat_gc::gc_record_write,
                dispatch,
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
    caller_function_name: Option<&str>,
) -> Result<RootedValue, RuntimeError> {
    store_member(
        base,
        field,
        rhs,
        caller_function_name,
        runmat_gc::gc_record_write,
    )
    .await
}

pub(crate) fn store_member_sequence_values_with_scope<'a, M: MemberStoreScope<'a> + 'a>(
    scope: M,
    base: Value,
    field: runmat_value::Utf8Text,
    values: crate::sequence::ValueSequence,
    caller: Option<&'a str>,
) -> impl std::future::Future<Output = Result<RootedValue, RuntimeError>> + 'a {
    store_member_sequence_values_with_optional_access_scope(
        scope, base, field, values, caller, None,
    )
}

pub(crate) fn store_member_sequence_values_with_optional_access_scope<
    'a,
    M: MemberStoreScope<'a> + 'a,
>(
    scope: M,
    base: Value,
    field: runmat_value::Utf8Text,
    values: crate::sequence::ValueSequence,
    caller: Option<&'a str>,
    access: Option<&'a crate::object::protocol::ObjectAccessContext>,
) -> impl std::future::Future<Output = Result<RootedValue, RuntimeError>> + 'a {
    store_member_sequence_values_with_caller_scope(
        scope,
        base,
        field,
        values,
        match access {
            Some(access) => MemberCaller::frame(caller, access),
            None => MemberCaller::named(caller),
        },
    )
}

pub(crate) fn store_default_member_sequence_values<'a>(
    runtime: Option<&'a crate::context::RuntimeContext>,
    base: Value,
    field: runmat_value::Utf8Text,
    values: crate::sequence::ValueSequence,
    caller: Option<&'a str>,
    access: &'a crate::object::protocol::ObjectAccessContext,
) -> impl std::future::Future<Output = Result<RootedValue, RuntimeError>> + 'a {
    let caller = MemberCaller::frame(caller, access);
    match runtime {
        Some(runtime) => {
            futures::future::Either::Left(store_member_sequence_values_with_dispatch_scope(
                ContextualMemberScope(runtime),
                base,
                field,
                values,
                caller,
                ProtocolDispatch::Default,
            ))
        }
        None => futures::future::Either::Right(store_member_sequence_values_with_dispatch_scope(
            OrdinaryMemberScope,
            base,
            field,
            values,
            caller,
            ProtocolDispatch::Default,
        )),
    }
}
pub fn store_member_sequence_traced_with_access<'a>(
    context: &'a crate::context::RuntimeContext,
    base: Value,
    field: String,
    values: Vec<Value>,
    caller_function_name: Option<&'a str>,
    access: &'a crate::object::protocol::ObjectAccessContext,
) -> impl std::future::Future<Output = Result<RootedValue, RuntimeError>> + 'a {
    store_member_sequence_traced_with_scope(
        ContextualMemberScope(context),
        base,
        field,
        values,
        MemberCaller::frame(caller_function_name, access),
    )
}

pub async fn store_member_traced_in_context(
    context: &crate::context::RuntimeContext,
    base: Value,
    field: String,
    rhs: Value,
    caller_function_name: Option<&str>,
) -> Result<RootedValue, RuntimeError> {
    store_member_with_context(
        Some(context),
        base,
        field,
        rhs,
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
    caller_function_name: Option<&str>,
) -> Result<RootedValue, RuntimeError> {
    store_member_dynamic(
        base,
        name,
        rhs,
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
    caller_function_name: Option<&str>,
) -> Result<RootedValue, RuntimeError> {
    store_member_with_context(
        Some(context),
        base,
        name,
        rhs,
        caller_function_name,
        runmat_gc::gc_record_write,
    )
    .await
}

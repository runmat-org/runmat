//! Object save/load customization hooks.

use crate::execution::{RootedValue, RootedValueLease, RootedValuesBuilder};
use runmat_builtins::BuiltinErrorDescriptor;
use runmat_macros::runtime_builtin;
use runmat_value::{CellArray, StructValue, Value};
use std::ops::ControlFlow;

mod assembly;
#[cfg(test)]
#[path = "object_serialization/factor_tests.rs"]
mod factor_tests;
mod mat_load;
#[cfg(test)]
#[path = "object_serialization/pending_tests.rs"]
mod pending_tests;
mod restore;
pub(crate) use mat_load::{classify_mat_load_restoration, MatLoadRestoration};

pub(crate) const SAVEOBJ_METHOD: runmat_types::StaticMethodName =
    runmat_types::StaticMethodName::new("saveobj");
pub(crate) const LOADOBJ_METHOD: runmat_types::StaticMethodName =
    runmat_types::StaticMethodName::new("loadobj");

const MAX_SERIALIZATION_DEPTH: usize = 32;

pub use runmat_builtins::{
    LOADOBJ_DESCRIPTOR, LOADOBJ_EXTENSIONS, LOADOBJ_INTEGER_CAPABILITIES,
    LOADOBJ_PASSTHROUGH_EXTENSION, SAVEOBJ_DESCRIPTOR, SAVEOBJ_INTEGER_AUDIT,
    SERIALIZATION_ERROR_ARGUMENT, SERIALIZATION_ERROR_DISPATCH, SERIALIZATION_ERROR_RECURSION,
};

fn serialization_error(
    builtin: &'static str,
    descriptor: &'static BuiltinErrorDescriptor,
    detail: impl Into<String>,
) -> crate::RuntimeError {
    crate::runtime_descriptor_error_with_detail(builtin, descriptor, detail.into())
}

fn object_class_name(value: &Value) -> Option<runmat_types::ClassIdentity> {
    crate::object_receiver_class_name(value)
}

fn dispatch_registered_method<'a>(
    class_name: &'a runmat_types::ClassIdentity,
    method_name: runmat_types::StaticMethodName,
    args: Vec<Value>,
    builtin: &'static str,
    selected_runtime: Option<&crate::context::RuntimeContext>,
    builtin_issuance: Option<crate::class_registry::BuiltinClassIssuance>,
) -> crate::execution::RuntimeContinuationFuture<'a, ControlFlow<RootedValue, RootedValue>> {
    // The callback's inputs must remain rooted even if the admitted future is
    // retained unpolled, cancelled, or dropped by its caller.
    let runtime = selected_runtime
        .cloned()
        .or_else(crate::context::legacy::active);
    let args_root = match RootedValueLease::for_values(args.iter()) {
        Ok(root) => root,
        Err(error) => {
            return crate::execution::rooted_continuation::admit_rooted_future(
                runtime.as_ref(),
                builtin,
                crate::builtin::builtin_control_error,
                std::future::ready(Err(error.into_runtime_error())),
            );
        }
    };
    crate::execution::rooted_continuation::admit_rooted_future_with(
        runtime,
        builtin,
        crate::builtin::builtin_control_error,
        move |runtime| async move {
            let _args_root = args_root;
            let mut arguments = match runtime.as_ref() {
                Some(runtime) => crate::execution::RootedValues::in_context(args, runtime)?,
                None => crate::execution::RootedValues::new(args)
                    .map_err(|error| error.into_runtime_error())?,
            };
            if let Some(issuance) = builtin_issuance {
                // A typed MAT class is an exact current registration, not a
                // license to resolve a same-spelled callable later. Select
                // the registered method after future admission, then invoke
                // only its issued body or canonical builtin identity.
                let current = match runtime.as_ref() {
                    Some(runtime) => issuance.is_current_in_context(runtime),
                    None => issuance.is_current_ordinary(),
                };
                if !current || issuance.class() != class_name {
                    return Err(serialization_error(
                        builtin,
                        &SERIALIZATION_ERROR_DISPATCH,
                        "registered class changed during object restoration",
                    ));
                }
                let method = match runtime.as_ref() {
                    Some(runtime) => {
                        runtime
                            .scope(async {
                                crate::class_registry::lookup_bound_method(
                                    class_name,
                                    &method_name.owned(),
                                )
                            })
                            .await
                    }
                    None => {
                        crate::class_registry::lookup_bound_method(class_name, &method_name.owned())
                    }
                }
                .ok_or_else(|| {
                    serialization_error(
                        builtin,
                        &SERIALIZATION_ERROR_DISPATCH,
                        format!("registered {class_name}.{method_name} is no longer admitted"),
                    )
                })?;
                let current = match runtime.as_ref() {
                    Some(runtime) => issuance.is_current_in_context(runtime),
                    None => issuance.is_current_ordinary(),
                };
                if !current {
                    return Err(serialization_error(
                        builtin,
                        &SERIALIZATION_ERROR_DISPATCH,
                        "registered class changed during object restoration",
                    ));
                }
                if method.declaration.is_abstract {
                    return Err(serialization_error(
                        builtin,
                        &SERIALIZATION_ERROR_DISPATCH,
                        format!("registered {class_name}.{method_name} is abstract"),
                    ));
                }
                if let Some(binding) = method.bound_function {
                    let call = crate::call::RootedFunctionArguments::ordinary(arguments);
                    let result = match runtime.as_ref() {
                        Some(runtime) => {
                            crate::user_functions::try_call_rooted_bound_function_with_environment_in_context(
                                runtime, binding, call, 1, None,
                            )
                            .await
                        }
                        None => crate::user_functions::try_call_rooted_bound_function(
                            binding, call, 1,
                        )
                        .await,
                    }
                    .ok_or_else(|| {
                        serialization_error(
                            builtin,
                            &SERIALIZATION_ERROR_DISPATCH,
                            "issued loadobj body has no rooted invoker",
                        )
                    })??;
                    return crate::call::arguments::project_rooted_legacy_builtin_value_abi(result)
                        .map(ControlFlow::Break);
                }
                let runmat_types::CallableIdentity::Builtin(id) = method.callable else {
                    return Err(serialization_error(
                        builtin,
                        &SERIALIZATION_ERROR_DISPATCH,
                        "registered loadobj lacks an issued executable target",
                    ));
                };
                let sequence = crate::dispatcher::try_call_builtin_sequence_async_with_outputs(
                    &id.0,
                    arguments.values(),
                    1,
                    runtime.as_ref(),
                )
                .await?
                .ok_or_else(|| {
                    serialization_error(
                        builtin,
                        &SERIALIZATION_ERROR_DISPATCH,
                        "issued loadobj builtin is unavailable",
                    )
                })?;
                return crate::call::arguments::project_rooted_legacy_builtin_value_abi(sequence)
                    .map(ControlFlow::Break);
            }
            if let Some((method, owner)) =
                crate::class_registry::lookup_method(class_name, &method_name.owned())
            {
                let owner_member = format!("{owner}.{method_name}");
                let mut candidates = Vec::with_capacity(2);
                if !method.function_name.trim().is_empty() {
                    candidates.push(method.function_name);
                }
                if !candidates
                    .iter()
                    .any(|candidate| candidate == &owner_member)
                {
                    candidates.push(owner_member);
                }

                for candidate in candidates {
                    let (identity, fallback_policy) =
                        crate::callable_identity_for_handle_name(&candidate);
                    let request = crate::user_functions::CallableRequest::rooted_resolved(
                        identity,
                        fallback_policy,
                        crate::call::RootedFunctionArguments::ordinary(arguments),
                        1,
                    )
                    .with_selected_runtime(runtime.as_ref());
                    match crate::user_functions::try_call_semantic_descriptor_owned(request).await {
                        std::ops::ControlFlow::Break(result) => {
                            return result
                                .and_then(
                                    crate::call::arguments::project_rooted_legacy_builtin_value_abi,
                                )
                                .map(ControlFlow::Break);
                        }
                        std::ops::ControlFlow::Continue(returned) => {
                            arguments = returned.try_into_ordinary().map_err(|_| {
                                serialization_error(
                                    builtin,
                                    &SERIALIZATION_ERROR_DISPATCH,
                                    "save/load callback declined without its original arguments",
                                )
                            })?;
                        }
                    }
                    if runmat_builtins::builtin_name_is_known(&candidate) {
                        if let Some(sequence) =
                            crate::dispatcher::try_call_builtin_sequence_async_with_outputs(
                                &candidate,
                                arguments.values(),
                                1,
                                runtime.as_ref(),
                            )
                            .await?
                        {
                            return crate::call::arguments::project_rooted_legacy_builtin_value_abi(
                                sequence,
                            )
                            .map(ControlFlow::Break);
                        }
                    }
                }
                return Err(serialization_error(
                    builtin,
                    &SERIALIZATION_ERROR_DISPATCH,
                    format!("{class_name}.{method_name} did not resolve to a callable"),
                ));
            }

            if arguments
                .first()
                .is_some_and(|arg| matches!(arg, Value::Object(_) | Value::HandleObject(_)))
            {
                let request = crate::user_functions::CallableRequest::rooted_resolved(
                    crate::class_member_identity(class_name, method_name.display_name()),
                    runmat_types::CallableFallbackPolicy::ExternalBoundary,
                    crate::call::RootedFunctionArguments::ordinary(arguments),
                    1,
                )
                .with_selected_runtime(runtime.as_ref());
                match crate::user_functions::try_call_semantic_descriptor_owned(request).await {
                    std::ops::ControlFlow::Break(result) => result
                        .and_then(crate::call::arguments::project_rooted_legacy_builtin_value_abi)
                        .map(ControlFlow::Break),
                    std::ops::ControlFlow::Continue(returned) => {
                        let original = returned.try_into_ordinary().map_err(|_| {
                            serialization_error(
                                builtin,
                                &SERIALIZATION_ERROR_DISPATCH,
                                "external callback declined without its original arguments",
                            )
                        })?;
                        let mut values = original.try_into_values()?;
                        root_value(values.next().expect("one callback argument"))
                            .map(ControlFlow::Continue)
                    }
                }
            } else {
                let mut values = arguments.try_into_values()?;
                root_value(values.next().expect("one callback argument")).map(ControlFlow::Continue)
            }
        },
    )
}

async fn call_saveobj_if_available(
    value: Value,
    runtime: Option<&crate::context::RuntimeContext>,
) -> crate::BuiltinResult<ControlFlow<RootedValue, RootedValue>> {
    let Some(class_name) = object_class_name(&value) else {
        return root_value(value).map(ControlFlow::Continue);
    };
    dispatch_registered_method(
        &class_name,
        SAVEOBJ_METHOD,
        vec![value],
        SAVEOBJ_METHOD.display_name(),
        runtime,
        None,
    )
    .await
}

/// MAT's typed object node retains class/shape separately from this saved
/// element. The returned root stays live while its payload is encoded.
pub(crate) fn prepare_mat_object_element(
    value: Value,
    runtime: Option<&crate::context::RuntimeContext>,
) -> impl std::future::Future<Output = crate::BuiltinResult<RootedValue>> + '_ {
    // A queued MAT conversion may outlive its caller before its first poll.
    // Install the receiver's managed roots while constructing the request.
    let input_root =
        RootedValueLease::for_traced(&value).map_err(|error| error.into_runtime_error());
    async move {
        let _input_root = input_root?;
        if let Value::Object(object) = &value {
            if object.is_class(runmat_types::standard::DECOMPOSITION) {
                let selected = runtime.ok_or_else(|| {
                    serialization_error(
                        SAVEOBJ_METHOD.display_name(),
                        &SERIALIZATION_ERROR_DISPATCH,
                        "saving a decomposition factor requires its selected invocation",
                    )
                })?;
                crate::builtins::math::linalg::factor::decomposition::validate_factor_object_binding(
                    object, selected,
                )
                .await?;
            }
        }
        if let Some(facts) = crate::object::protocol::selected_graph_receiver_facts(&value)? {
            let method = facts.method(&SAVEOBJ_METHOD.owned());
            if let Some(method) = method {
                if method.is_abstract {
                    return Err(serialization_error(
                        SAVEOBJ_METHOD.display_name(),
                        &SERIALIZATION_ERROR_DISPATCH,
                        "selected saveobj is abstract",
                    ));
                }
                let binding = method.binding.clone().ok_or_else(|| {
                    serialization_error(
                        SAVEOBJ_METHOD.display_name(),
                        &SERIALIZATION_ERROR_DISPATCH,
                        "selected saveobj has no admitted body",
                    )
                })?;
                let selected = runtime
                    .cloned()
                    .ok_or_else(|| {
                        serialization_error(
                            SAVEOBJ_METHOD.display_name(),
                            &SERIALIZATION_ERROR_DISPATCH,
                            "selected saveobj requires its issuing runtime",
                        )
                    })?
                    .with_selected_object_class(std::rc::Rc::clone(&facts))
                    .with_selected_method_body(
                        std::rc::Rc::clone(&facts),
                        method.owner.clone(),
                        binding.clone(),
                    );
                let descriptor = crate::call::descriptor::CallableDescriptor::bound(
                    binding,
                    vec![value],
                    1,
                    crate::call::descriptor::CallableCallKind::Direct,
                );
                return crate::call::descriptor::execute_callable_descriptor_in_context(
                    &selected, descriptor,
                )
                .await
                .and_then(crate::call::arguments::project_rooted_legacy_builtin_value_abi);
            }
            return root_value(Value::Struct(into_object_properties_payload(value)?));
        }
        let class = object_class_name(&value).ok_or_else(|| {
            serialization_error(
                SAVEOBJ_METHOD.display_name(),
                &SERIALIZATION_ERROR_ARGUMENT,
                "MAT object element has no class identity",
            )
        })?;
        let has_saveobj = match runtime {
            Some(runtime) => {
                runtime
                    .scope(async {
                        crate::class_registry::lookup_method(&class, &SAVEOBJ_METHOD.owned())
                            .is_some()
                    })
                    .await
            }
            None => crate::class_registry::lookup_method(&class, &SAVEOBJ_METHOD.owned()).is_some(),
        };
        if has_saveobj {
            return match call_saveobj_if_available(value, runtime).await? {
                ControlFlow::Break(saved) => Ok(saved),
                ControlFlow::Continue(_) => Err(serialization_error(
                    SAVEOBJ_METHOD.display_name(),
                    &SERIALIZATION_ERROR_DISPATCH,
                    format!("registered {class}.saveobj did not resolve"),
                )),
            };
        }
        // An external member can still supply saveobj without a registry entry.
        // Its Decline path requires a property fallback at this output boundary.
        match call_saveobj_if_available(value, runtime).await? {
            ControlFlow::Break(saved) => Ok(saved),
            ControlFlow::Continue(original) => original.try_map_owned(
                |value| Ok(Value::Struct(into_object_properties_payload(value)?)),
                |error| error.into_runtime_error(),
            ),
        }
    }
}

fn into_object_properties_payload(value: Value) -> crate::BuiltinResult<StructValue> {
    match value {
        Value::Object(object) if !object.has_native_payload() => {
            let builder = object.into_unbounded_builder().map_err(|_| {
                serialization_error(
                    SAVEOBJ_METHOD.display_name(),
                    &SERIALIZATION_ERROR_ARGUMENT,
                    "admitted object properties require an admitted serialization destination",
                )
            })?;
            Ok(StructValue::from_fields_unbounded(
                builder.into_properties_unbounded(),
            ))
        }
        other => object_properties_payload(&other),
    }
}

async fn call_loadobj_if_available(
    class_name: &str,
    payload: Value,
    runtime: Option<&crate::context::RuntimeContext>,
    builtin_issuance: Option<&crate::class_registry::BuiltinClassIssuance>,
) -> crate::BuiltinResult<Option<RootedValue>> {
    let identity = runmat_types::ClassIdentity::new(class_name).map_err(|error| {
        serialization_error(
            LOADOBJ_METHOD.display_name(),
            &SERIALIZATION_ERROR_ARGUMENT,
            format!("invalid serialized class identity: {error}"),
        )
    })?;
    match dispatch_registered_method(
        &identity,
        LOADOBJ_METHOD,
        vec![payload],
        LOADOBJ_METHOD.display_name(),
        runtime,
        builtin_issuance.cloned(),
    )
    .await?
    {
        ControlFlow::Break(restored) => Ok(Some(restored)),
        ControlFlow::Continue(_) => Ok(None),
    }
}

/// Restore one mxOBJECT_CLASS element only after the file's class spelling has
/// been resolved against the selected current Runtime/project. The wire node
/// itself never constructs or carries a selected receiver authority.
pub(crate) async fn resolve_mat_object_class(
    class_name: &str,
    runtime: Option<&crate::context::RuntimeContext>,
) -> crate::BuiltinResult<(
    runmat_types::ClassIdentity,
    crate::context::CurrentClassProductResolution,
)> {
    let class = runmat_types::ClassIdentity::new(class_name).map_err(|error| {
        serialization_error(
            LOADOBJ_METHOD.display_name(),
            &SERIALIZATION_ERROR_ARGUMENT,
            format!("invalid serialized class identity: {error}"),
        )
    })?;
    if class.is(runmat_types::standard::INTEGER_CONSTRAINT) {
        let _entry = runtime.map(crate::context::RuntimeContext::enter);
        crate::builtins::math::optim::integer_constraint::ensure_registered();
    }
    let product = match runtime {
        Some(runtime) => runtime.resolve_current_object_class(&class).await?,
        None => crate::class_registry::issue_builtin_class_ordinary(&class)
            .map(crate::context::CurrentClassProductResolution::RegisteredBuiltin)
            .ok_or_else(|| serialization_error(
                LOADOBJ_METHOD.display_name(),
                &SERIALIZATION_ERROR_DISPATCH,
                format!("serialized class '{class}' requires a selected runtime for current-source admission"),
            ))?,
    };
    Ok((class, product))
}

pub(crate) fn validate_mat_object_product_current(
    product: &crate::context::CurrentClassProductResolution,
    runtime: Option<&crate::context::RuntimeContext>,
) -> crate::BuiltinResult<()> {
    if let crate::context::CurrentClassProductResolution::RegisteredBuiltin(issuance) = product {
        let current = match runtime {
            Some(runtime) => issuance.is_current_in_context(runtime),
            None => issuance.is_current_ordinary(),
        };
        if !current {
            return Err(serialization_error(
                LOADOBJ_METHOD.display_name(),
                &SERIALIZATION_ERROR_DISPATCH,
                "registered class changed during object restoration",
            ));
        }
    }
    Ok(())
}

pub(crate) async fn restore_mat_object_element(
    class: &runmat_types::ClassIdentity,
    product: &crate::context::CurrentClassProductResolution,
    handle: bool,
    payload: Value,
    runtime: Option<&crate::context::RuntimeContext>,
) -> crate::BuiltinResult<RootedValue> {
    let payload = RootedValue::in_optional_context(payload, runtime)?;
    validate_mat_object_product_current(product, runtime)?;
    if let crate::context::CurrentClassProductResolution::Selected(facts) = product {
        if let Some(method) = facts.method(&LOADOBJ_METHOD.owned()) {
            if method.is_abstract {
                return Err(serialization_error(
                    LOADOBJ_METHOD.display_name(),
                    &SERIALIZATION_ERROR_DISPATCH,
                    format!("selected {class}.loadobj is abstract"),
                ));
            }
            let binding = method.binding.clone().ok_or_else(|| {
                serialization_error(
                    LOADOBJ_METHOD.display_name(),
                    &SERIALIZATION_ERROR_DISPATCH,
                    format!("selected {class}.loadobj has no admitted body"),
                )
            })?;
            let selected = runtime
                .cloned()
                .ok_or_else(|| {
                    serialization_error(
                        LOADOBJ_METHOD.display_name(),
                        &SERIALIZATION_ERROR_DISPATCH,
                        "selected loadobj requires its issuing runtime",
                    )
                })?
                .with_selected_object_class(std::rc::Rc::clone(facts))
                .with_selected_method_body(
                    std::rc::Rc::clone(facts),
                    method.owner.clone(),
                    binding.clone(),
                );
            return payload
                .with_owned_async(|value| async move {
                    let descriptor = crate::call::descriptor::CallableDescriptor::bound(
                        binding,
                        vec![value],
                        1,
                        crate::call::descriptor::CallableCallKind::Direct,
                    );
                    crate::call::descriptor::execute_callable_descriptor_in_context(
                        &selected, descriptor,
                    )
                    .await
                    .and_then(crate::call::arguments::project_rooted_legacy_builtin_value_abi)
                })
                .await;
        }
    }
    let has_loadobj = matches!(
        product,
        crate::context::CurrentClassProductResolution::RegisteredBuiltin(_)
    ) && match runtime {
        Some(runtime) => {
            runtime
                .scope(async {
                    crate::class_registry::lookup_method(class, &LOADOBJ_METHOD.owned()).is_some()
                })
                .await
        }
        None => crate::class_registry::lookup_method(class, &LOADOBJ_METHOD.owned()).is_some(),
    };
    if has_loadobj {
        return payload
            .with_owned_async(|value| {
                call_loadobj_if_available(
                    class.display_name(),
                    value,
                    runtime,
                    match product {
                        crate::context::CurrentClassProductResolution::RegisteredBuiltin(
                            issuance,
                        ) => Some(issuance),
                        crate::context::CurrentClassProductResolution::Selected(_) => None,
                    },
                )
            })
            .await?
            .ok_or_else(|| {
                serialization_error(
                    LOADOBJ_METHOD.display_name(),
                    &SERIALIZATION_ERROR_DISPATCH,
                    format!("registered {class}.loadobj did not resolve"),
                )
            });
    }
    if !handle
        && matches!(
            product,
            crate::context::CurrentClassProductResolution::RegisteredBuiltin(_)
        )
        && class
            == &crate::builtins::math::linalg::factor::decomposition::DECOMPOSITION_CLASS_IDENTITY
    {
        let runtime = runtime.ok_or_else(|| {
            serialization_error(
                LOADOBJ_METHOD.display_name(),
                &SERIALIZATION_ERROR_DISPATCH,
                "restoring a decomposition factor requires its selected invocation",
            )
        })?;
        return payload.with_owned_async(|payload| async move {
            let Value::Struct(fields) = payload else {
                return Err(serialization_error(
                    LOADOBJ_METHOD.display_name(),
                    &SERIALIZATION_ERROR_ARGUMENT,
                    "decomposition factor requires its saved property recipe",
                ));
            };
            let object = crate::execution::object::construct_object_from_borrowed_fields(
                Some(runtime), class.display_name(), fields.fields.iter(),
            )?;
            object.with_owned_async(|value| async move {
                let Value::Object(object) = value else {
                    return Err(serialization_error(
                        LOADOBJ_METHOD.display_name(),
                        &SERIALIZATION_ERROR_DISPATCH,
                        "decomposition recipe did not construct an object",
                    ));
                };
                let restored = crate::builtins::math::linalg::factor::decomposition::restore_decomposition_factor_object(
                    object, runtime,
                ).await?;
                RootedValue::in_optional_context(Value::Object(restored), Some(runtime))
            }).await
        }).await;
    }
    payload.with_owned(|payload| {
        let Value::Struct(fields) = payload else {
            return Err(serialization_error(
                LOADOBJ_METHOD.display_name(),
                &SERIALIZATION_ERROR_ARGUMENT,
                "object without loadobj requires a saved property structure",
            ));
        };
        let object = crate::execution::object::construct_object_from_borrowed_fields(
            runtime,
            class.display_name(),
            fields.fields.iter(),
        )?;
        if handle {
            let Value::Object(instance) = object.value() else {
                unreachable!("constructed restored object")
            };
            let class_name = instance.class_name_owner().clone();
            object.with_owned(|value| {
                let target = runmat_gc::gc_allocate_rooted(value).map_err(|error| {
                    serialization_error(
                        LOADOBJ_METHOD.display_name(),
                        &SERIALIZATION_ERROR_ARGUMENT,
                        format!("failed to restore serialized handle object: {error}"),
                    )
                })?;
                RootedValue::in_optional_context(
                    Value::HandleObject(runmat_value::HandleRef {
                        class_name,
                        target: target.handle(),
                        valid: true,
                    }),
                    runtime,
                )
            })
        } else {
            Ok(object)
        }
    })
}

/// Rebind a decoded object's storage to the exact current product selected for
/// this file load. Neither the wire class name nor a later ambient registry
/// publication can confer receiver authority.
pub(crate) fn reconcile_mat_object_product(
    value: RootedValue,
    class: &runmat_types::ClassIdentity,
    product: &crate::context::CurrentClassProductResolution,
    handle: bool,
    runtime: Option<&crate::context::RuntimeContext>,
) -> crate::BuiltinResult<RootedValue> {
    let builtin_domain = class.is(runmat_types::standard::INTEGER_CONSTRAINT)
        && matches!(
            product,
            crate::context::CurrentClassProductResolution::RegisteredBuiltin(_)
        );
    if handle && builtin_domain {
        return Err(serialization_error(
            LOADOBJ_METHOD.display_name(),
            &SERIALIZATION_ERROR_ARGUMENT,
            "IntegerConstraint is a value class, not a handle class",
        ));
    }
    value.try_map_owned(
        |value| match (value, handle) {
            (Value::Object(mut object), false) if object.class_name() == class => {
                match product {
                    crate::context::CurrentClassProductResolution::Selected(facts) => {
                        if let Some(existing) = object.selected_class() {
                            if !std::rc::Rc::ptr_eq(existing, facts) {
                                return Err(serialization_error(
                                    LOADOBJ_METHOD.display_name(),
                                    &SERIALIZATION_ERROR_DISPATCH,
                                    "loadobj returned an object from a different selected product",
                                ));
                            }
                        } else if !object.try_attach_selected_class(std::rc::Rc::clone(facts)) {
                            return Err(serialization_error(
                                LOADOBJ_METHOD.display_name(),
                                &SERIALIZATION_ERROR_DISPATCH,
                                "loadobj returned an object incompatible with its selected product",
                            ));
                        }
                    }
                    crate::context::CurrentClassProductResolution::RegisteredBuiltin(_) => {
                        if object.selected_class().is_some() {
                            return Err(serialization_error(
                                LOADOBJ_METHOD.display_name(),
                                &SERIALIZATION_ERROR_DISPATCH,
                                "registered object carried unrelated selected authority",
                            ));
                        }
                    }
                }
                if builtin_domain {
                    crate::builtins::math::optim::integer_constraint::schema::validate(&object,runtime)
                        .map_err(|error|crate::builtins::math::optim::integer_constraint::validation_error(runtime,error))?;
                }
                Ok(Value::Object(object))
            }
            (Value::HandleObject(reference), true) if &reference.class_name == class => {
                runmat_gc::gc_with_value_mut(&reference.target, |target| {
                    let Value::Object(object) = target else {
                        return Err(serialization_error(
                            LOADOBJ_METHOD.display_name(),
                            &SERIALIZATION_ERROR_DISPATCH,
                            "loadobj returned a handle without an object target",
                        ));
                    };
                    if object.class_name() != class {
                        return Err(serialization_error(
                            LOADOBJ_METHOD.display_name(),
                            &SERIALIZATION_ERROR_DISPATCH,
                            "loadobj returned a handle target of another class",
                        ));
                    }
                    match product {
                        crate::context::CurrentClassProductResolution::Selected(facts) => {
                            if let Some(existing) = object.selected_class() {
                                if !std::rc::Rc::ptr_eq(existing, facts) {
                                    return Err(serialization_error(
                                        LOADOBJ_METHOD.display_name(),
                                        &SERIALIZATION_ERROR_DISPATCH,
                                        "loadobj returned a handle from a different selected product",
                                    ));
                                }
                            } else if !object.try_attach_selected_class(std::rc::Rc::clone(facts)) {
                                return Err(serialization_error(
                                    LOADOBJ_METHOD.display_name(),
                                    &SERIALIZATION_ERROR_DISPATCH,
                                    "loadobj returned a handle incompatible with its selected product",
                                ));
                            }
                        }
                        crate::context::CurrentClassProductResolution::RegisteredBuiltin(_) => {
                            if object.selected_class().is_some() {
                                return Err(serialization_error(
                                    LOADOBJ_METHOD.display_name(),
                                    &SERIALIZATION_ERROR_DISPATCH,
                                    "registered handle carried unrelated selected authority",
                                ));
                            }
                        }
                    }
                    Ok(())
                })
                .map_err(|error| {
                    serialization_error(
                        LOADOBJ_METHOD.display_name(),
                        &SERIALIZATION_ERROR_DISPATCH,
                        format!("cannot inspect restored handle target: {error}"),
                    )
                })??;
                Ok(Value::HandleObject(reference))
            }
            _ => Err(serialization_error(
                LOADOBJ_METHOD.display_name(),
                &SERIALIZATION_ERROR_DISPATCH,
                "loadobj result does not match the saved object class and kind",
            )),
        },
        |error| error.into_runtime_error(),
    )
}

fn object_properties_payload(value: &Value) -> crate::BuiltinResult<StructValue> {
    match value {
        Value::Object(object) => {
            if object.has_native_payload()
                && !(object.is_class(crate::builtins::math::linalg::factor::decomposition::DECOMPOSITION_CLASS_IDENTITY)
                    && object.numeric_factor_data().is_some())
            {
                return Err(serialization_error(
                    SAVEOBJ_METHOD.display_name(),
                    &SERIALIZATION_ERROR_ARGUMENT,
                    "GC-owned native object payloads require a typed serializer",
                ));
            }
            let mut payload = runmat_value::UnboundedStructValueBuilder::new();
            for (name, property) in object.properties().iter() {
                payload.insert(name.to_owned(), property.clone());
            }
            Ok(payload.finish())
        }
        Value::HandleObject(handle) => runmat_gc::gc_with_value(&handle.target, |target| {
            if let Value::Object(object) = target {
                if object.has_native_payload() {
                    return Err(serialization_error(
                        SAVEOBJ_METHOD.display_name(),
                        &SERIALIZATION_ERROR_ARGUMENT,
                        "GC-owned native object payloads require a typed serializer",
                    ));
                }
                let mut payload = runmat_value::UnboundedStructValueBuilder::new();
                for (name, property) in object.properties().iter() {
                    payload.insert(name.to_owned(), property.clone());
                }
                Ok(payload.finish())
            } else {
                Err(serialization_error(
                    SAVEOBJ_METHOD.display_name(),
                    &SERIALIZATION_ERROR_ARGUMENT,
                    "handle target is not an object",
                ))
            }
        })
        .map_err(|err| {
            serialization_error(
                SAVEOBJ_METHOD.display_name(),
                &SERIALIZATION_ERROR_ARGUMENT,
                format!("failed to read handle target: {err}"),
            )
        })?,
        other => Err(serialization_error(
            SAVEOBJ_METHOD.display_name(),
            &SERIALIZATION_ERROR_ARGUMENT,
            format!("expected object, got {other:?}"),
        )),
    }
}

fn root_value(value: Value) -> crate::BuiltinResult<RootedValue> {
    RootedValue::new(value).map_err(|error| error.into_runtime_error())
}

pub(crate) async fn restore_value_from_mat_load(value: Value) -> crate::BuiltinResult<RootedValue> {
    restore_value_from_mat_load_in_context(value, None).await
}

pub(crate) async fn restore_value_from_mat_load_in_context(
    value: Value,
    runtime: Option<&crate::context::RuntimeContext>,
) -> crate::BuiltinResult<RootedValue> {
    restore::restore_value_after_load_in_context(value, runtime).await
}

#[runtime_builtin(
    name = "saveobj",
    binding_variant = "default",
    builtin_path = "crate::builtins::introspection::object_serialization"
)]
pub async fn saveobj_builtin(value: Value) -> crate::BuiltinResult<RootedValue> {
    let _input_root =
        RootedValueLease::for_traced(&value).map_err(|error| error.into_runtime_error())?;
    match value {
        receiver @ (Value::Object(_) | Value::HandleObject(_)) => {
            match call_saveobj_if_available(receiver, None).await? {
                ControlFlow::Break(saved) => Ok(saved),
                ControlFlow::Continue(original) => original.try_map_owned(
                    |value| Ok(Value::Struct(into_object_properties_payload(value)?)),
                    |error| error.into_runtime_error(),
                ),
            }
        }
        other => Err(serialization_error(
            SAVEOBJ_METHOD.display_name(),
            &SERIALIZATION_ERROR_ARGUMENT,
            format!("expected object, got {other:?}"),
        )),
    }
}

#[runtime_builtin(
    name = "loadobj",
    binding_variant = "default",
    builtin_path = "crate::builtins::introspection::object_serialization"
)]
pub async fn loadobj_builtin(value: Value) -> crate::BuiltinResult<RootedValue> {
    if !matches!(
        value,
        Value::Object(_) | Value::HandleObject(_) | Value::Struct(_) | Value::StructArray(_)
    ) {
        crate::compatibility::ensure_builtin_extension_enabled(
            &LOADOBJ_PASSTHROUGH_EXTENSION,
            LOADOBJ_METHOD.display_name(),
        )?;
    }
    restore_value_from_mat_load(value).await
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures::executor::block_on;
    use runmat_value::{IntValue, IntegerStorage, StructArray, Tensor};

    use tempfile::tempdir;

    fn structure(field: &str, value: Value) -> StructValue {
        let mut structure = runmat_value::UnboundedStructValueBuilder::new();
        structure.insert(field, value);
        structure.finish()
    }

    #[test]
    fn loadobj_preserves_nested_integer_payloads_without_materialization() {
        let mut payload = runmat_value::UnboundedStructValueBuilder::new();
        payload.insert("scalar", Value::Int(IntValue::U64(u64::MAX)));
        payload.insert(
            "array",
            Value::Tensor(
                Tensor::new_integer(IntegerStorage::I64(vec![i64::MIN, i64::MAX]), vec![2, 1])
                    .expect("integer payload"),
            ),
        );
        let restored = block_on(loadobj_builtin(Value::Struct(payload.finish()))).expect("loadobj");
        let Value::Struct(restored) = restored.value() else {
            panic!("expected structure payload");
        };
        assert_eq!(
            restored.fields.get("scalar"),
            Some(&Value::Int(IntValue::U64(u64::MAX)))
        );
        let Value::Tensor(array) = restored.fields.get("array").expect("array") else {
            panic!("expected integer array");
        };
        assert_eq!(
            array.integer_storage(),
            Some(&IntegerStorage::I64(vec![i64::MIN, i64::MAX]))
        );
    }

    #[test]
    fn loadobj_preserves_empty_and_nd_structure_arrays() {
        let empty = StructArray::with_fields(vec!["payload".into()], Vec::new(), vec![0, 2, 3])
            .expect("empty structure array");
        let saved =
            block_on(loadobj_builtin(Value::StructArray(empty))).expect("restore empty array");
        let Value::StructArray(saved) = saved.value() else {
            panic!("expected typed empty structure array");
        };
        assert_eq!(saved.shape(), [0, 2, 3]);
        assert_eq!(
            saved
                .field_names()
                .map(ToOwned::to_owned)
                .collect::<Vec<_>>(),
            ["payload"]
        );

        let elements = (0..4)
            .map(|index| structure("payload", Value::Int(IntValue::U64(index))))
            .collect();
        let array = StructArray::with_fields(vec!["payload".into()], elements, vec![2, 1, 2])
            .expect("N-D structure array");
        let restored = block_on(loadobj_builtin(Value::StructArray(array))).expect("restore");
        let Value::StructArray(restored) = restored.value() else {
            panic!("expected typed N-D structure array");
        };
        assert_eq!(restored.shape(), [2, 1, 2]);
        for (index, element) in restored.elements().enumerate() {
            assert_eq!(
                element.fields.get("payload"),
                Some(&Value::Int(IntValue::U64(index as u64)))
            );
        }
    }

    #[test]
    fn loadobj_preserves_nested_value_and_handle_objects() {
        let mut value_object = runmat_value::UnboundedObjectBuilder::new("NestedValue");
        value_object.insert("payload".into(), Value::Int(IntValue::U64(u64::MAX)));

        let mut handle_object = runmat_value::UnboundedObjectBuilder::new("NestedHandle");
        handle_object.insert("payload".into(), Value::String("entry".into()));
        let target =
            runmat_gc::gc_allocate(Value::Object(handle_object.finish())).expect("handle target");
        let handle = Value::HandleObject(runmat_value::HandleRef {
            class_name: "NestedHandle".into(),
            target,
            valid: true,
        });

        let array = StructArray::with_fields(
            vec!["payload".into()],
            vec![
                structure("payload", Value::Object(value_object.finish())),
                structure("payload", handle),
            ],
            vec![2, 1],
        )
        .expect("structure array");
        let restored = block_on(loadobj_builtin(Value::StructArray(array))).expect("restore");
        let Value::StructArray(restored) = restored.value() else {
            panic!("expected typed structure array");
        };
        assert!(matches!(
            restored.get_linear(0).unwrap().fields.get("payload"),
            Some(Value::Object(object)) if object.class_name().display_name() == "NestedValue"
        ));
        let Some(Value::HandleObject(handle)) =
            restored.get_linear(1).unwrap().fields.get("payload")
        else {
            panic!("expected nested handle object");
        };
        assert_eq!(handle.class_name.display_name(), "NestedHandle");
        assert!(runmat_gc::gc_with_value(&handle.target, |target| {
            matches!(target, Value::Object(object)
                if object.property("payload") == Some(&Value::String("entry".into())))
        })
        .expect("live handle target"));
    }

    #[test]
    fn loadobj_reports_nested_depth() {
        let mut nested = Value::Num(1.0);
        for _ in 0..=MAX_SERIALIZATION_DEPTH {
            nested = Value::Struct(structure("next", nested));
        }
        let array = StructArray::with_fields(
            vec!["payload".into()],
            vec![
                structure("payload", nested),
                structure("payload", Value::Num(2.0)),
            ],
            vec![2, 1],
        )
        .expect("structure array");
        let error = block_on(loadobj_builtin(Value::StructArray(array)))
            .expect_err("nested depth must fail");
        assert_eq!(error.identifier(), SERIALIZATION_ERROR_RECURSION.identifier);
    }

    #[test]
    fn loadobj_plain_integer_passthrough_is_declared_and_mode_gated() {
        assert_eq!(LOADOBJ_INTEGER_CAPABILITIES.len(), 2);
        assert_eq!(LOADOBJ_EXTENSIONS.len(), 1);
        let value = Value::Int(IntValue::U64(u64::MAX));
        let runmat = crate::compatibility::push_runmat_extensions_enabled(true);
        assert_eq!(
            block_on(loadobj_builtin(value.clone()))
                .expect("RunMat passthrough")
                .value(),
            &value
        );
        drop(runmat);
        let strict = crate::compatibility::push_runmat_extensions_enabled(false);
        let error = block_on(loadobj_builtin(value)).expect_err("strict mode must reject");
        drop(strict);
        assert_eq!(
            error.identifier(),
            LOADOBJ_PASSTHROUGH_EXTENSION.error_identifier
        );
    }

    #[test]
    fn loadobj_accepts_typed_struct_arrays_in_compatibility_mode() {
        let array = StructArray::with_fields(
            vec!["payload".into()],
            vec![
                structure("payload", Value::Num(1.0)),
                structure("payload", Value::Num(2.0)),
            ],
            vec![2, 1],
        )
        .unwrap();
        let strict = crate::compatibility::push_runmat_extensions_enabled(false);
        let restored = block_on(loadobj_builtin(Value::StructArray(array.clone())))
            .expect("structure arrays are part of the compatible loadobj domain");
        drop(strict);
        assert_eq!(restored.value(), &Value::StructArray(array));
    }

    #[test]
    fn saveobj_without_overload_returns_property_struct() {
        let mut object = runmat_value::UnboundedObjectBuilder::new("NoIdx");
        object.insert("x".to_string(), Value::Num(4.0));
        let saved = block_on(saveobj_builtin(Value::Object(object.finish()))).expect("saveobj");
        let Value::Struct(st) = saved.value() else {
            panic!("expected struct payload");
        };
        assert_eq!(st.fields.get("x"), Some(&Value::Num(4.0)));
    }

    #[test]
    fn saveobj_dispatches_registered_method() {
        block_on(crate::register_test_classes_builtin()).expect("test classes");
        let mut object = runmat_value::UnboundedObjectBuilder::new("OverIdx");
        object.insert("k".to_string(), Value::Num(9.0));
        object.insert("nargs".to_string(), Value::Num(5.0));

        let saved = block_on(saveobj_builtin(Value::Object(object.finish()))).expect("saveobj");
        let Value::Struct(st) = saved.value() else {
            panic!("expected saveobj struct");
        };
        assert_eq!(st.fields.get("k"), Some(&Value::Num(9.0)));
        assert_eq!(
            st.fields.get("saved_by"),
            Some(&Value::String("OverIdx.saveobj".to_string().into()))
        );
    }

    #[test]
    fn former_marker_fields_roundtrip_as_an_ordinary_mat_struct() {
        // Level5 stores text as char arrays; isolate ordinary-struct
        // classification from the format's existing string conversion.
        let mut fields = runmat_value::UnboundedStructValueBuilder::new();
        fields.insert(
            "__runmat_serialized_object_class__",
            Value::CharArray(runmat_value::CharArray::new_row("NotAnObject")),
        );
        fields.insert(
            "__runmat_serialized_object_kind__",
            Value::CharArray(runmat_value::CharArray::new_row("value")),
        );
        fields.insert(
            "__runmat_serialized_object_had_saveobj__",
            Value::Bool(true),
        );
        fields.insert("__runmat_serialized_object_payload__", Value::Num(7.0));
        let expected = Value::Struct(fields.finish());
        assert_eq!(
            block_on(loadobj_builtin(expected.clone()))
                .expect("direct loadobj preserves ordinary struct")
                .value(),
            &expected
        );
        let bytes = block_on(
            crate::builtins::io::mat::save::encode_workspace_to_mat_bytes(&[(
                "marker_fields".to_string(),
                expected.clone(),
            )]),
        )
        .expect("encode ordinary struct");
        let entries = crate::builtins::io::mat::load::decode_workspace_from_mat_bytes(&bytes)
            .expect("decode ordinary struct");
        assert_eq!(
            entries.value().0[0],
            ("marker_fields".to_string(), expected)
        );
    }

    #[test]
    fn mat_roundtrip_calls_saveobj_and_loadobj() {
        block_on(crate::register_test_classes_builtin()).expect("test classes");
        // Typed MAT restoration admits only an actual compile-time builtin
        // issuance. The shared OverIdx registration is ordinary for other
        // tests; this fixture explicitly issues its builtin method catalog.
        let identity = runmat_types::StaticClassIdentity::new("OverIdx");
        let class = crate::class_registry::get_class(&identity.owned()).expect("OverIdx class");
        crate::class_registry::register_builtin_class(identity, class);
        let dir = tempdir().expect("tempdir");
        let path = dir.path().join("object_roundtrip.mat");

        let mut object = runmat_value::UnboundedObjectBuilder::new("OverIdx");
        object.insert("k".to_string(), Value::Num(13.0));
        object.insert("nargs".to_string(), Value::Num(2.0));

        let array_elements = [13.0, 22.0]
            .into_iter()
            .map(|number| {
                let mut element = runmat_value::UnboundedObjectBuilder::new("OverIdx");
                element.insert("k".to_owned(), Value::Num(number));
                Value::Object(element.finish())
            })
            .collect();
        let array = runmat_value::ObjectArray::new("OverIdx", array_elements, vec![2, 1])
            .expect("object array");
        let empty =
            runmat_value::ObjectArray::empty("OverIdx", vec![0, 2]).expect("empty object array");

        let bytes = block_on(
            crate::builtins::io::mat::save::encode_workspace_to_mat_bytes(&[
                ("obj".to_string(), Value::Object(object.finish())),
                ("array".to_string(), Value::ObjectArray(array)),
                ("empty".to_string(), Value::ObjectArray(empty)),
            ]),
        )
        .expect("encode");
        std::fs::write(&path, bytes).expect("write mat file");

        let entries =
            block_on(crate::builtins::io::mat::load::read_mat_file(&path)).expect("load mat file");
        let loaded = entries
            .value()
            .0
            .iter()
            .find(|(name, _)| name == "obj")
            .map(|(_, value)| value)
            .expect("obj entry");
        let Value::Object(restored_object) = loaded else {
            panic!("expected restored object");
        };
        assert_eq!(restored_object.class_name().display_name(), "OverIdx");
        assert_eq!(restored_object.property("k"), Some(&Value::Num(13.0)));
        assert_eq!(
            restored_object.property("loaded_by"),
            Some(&Value::String("OverIdx.loadobj".to_string().into()))
        );
        let loaded_array = entries
            .value()
            .0
            .iter()
            .find(|(name, _)| name == "array")
            .map(|(_, value)| value)
            .expect("array entry");
        let Value::ObjectArray(loaded_array) = loaded_array else {
            panic!("expected object array");
        };
        assert_eq!(loaded_array.shape(), &[2, 1]);
        let keys: Vec<_> = loaded_array
            .data()
            .iter()
            .map(|value| match value {
                Value::Object(object) => {
                    assert_eq!(
                        object.property("loaded_by"),
                        Some(&Value::String("OverIdx.loadobj".to_string().into()))
                    );
                    object.property("k").cloned()
                }
                _ => panic!("expected restored object element"),
            })
            .collect();
        assert_eq!(keys, vec![Some(Value::Num(13.0)), Some(Value::Num(22.0))]);
        let loaded_empty = entries
            .value()
            .0
            .iter()
            .find(|(name, _)| name == "empty")
            .map(|(_, value)| value)
            .expect("empty entry");
        let Value::ObjectArray(loaded_empty) = loaded_empty else {
            panic!("expected empty object array");
        };
        assert_eq!(loaded_empty.shape(), &[0, 2]);
        assert_eq!(loaded_empty.class_name().display_name(), "OverIdx");

        let decoded = crate::builtins::io::mat::load::decode_workspace_from_mat_bytes(
            &std::fs::read(&path).unwrap(),
        )
        .expect("decode workspace bytes");
        let decoded_value = decoded
            .value()
            .0
            .iter()
            .find(|(name, _)| name == "obj")
            .map(|(_, value)| value)
            .expect("decoded obj entry");
        let Value::Object(decoded_object) = decoded_value else {
            panic!("expected decoded object");
        };
        assert_eq!(decoded_object.class_name().display_name(), "OverIdx");
        assert_eq!(
            decoded_object.property("loaded_by"),
            Some(&Value::String("OverIdx.loadobj".to_string().into()))
        );
    }
}

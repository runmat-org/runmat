//! Bind one selected function's immutable superclass graph before its calls.

use super::{mex, Bytecode, FunctionRegistry, ProcessDriverHandle, VmResult};

#[inline(always)]
pub(super) fn invoke_selected_bound(
    runtime: runmat_runtime::context::RuntimeContext,
    binding: runmat_value::BoundFunctionBinding,
    args: runmat_runtime::call::RootedFunctionArguments,
    requested_outputs: usize,
    registry: FunctionRegistry,
    driver: Option<ProcessDriverHandle>,
    lexical_environment: Option<runmat_value::LexicalEnvironment>,
) -> impl std::future::Future<Output = VmResult<runmat_runtime::execution::RootedValueSequence>> {
    // Selection returns domain data, not a large concrete future. Execute only
    // after its construction frame has retired, inside the admitted poll.
    runmat_runtime::execution::after_result(
        async move {
            let (runtime, role) = select_bound(runtime, &binding, &registry)?;
            Ok(SelectedBound {
                runtime,
                role,
                invocation: BoundInvocation {
                    binding,
                    args,
                    requested_outputs,
                    registry,
                    driver,
                    lexical_environment,
                },
            })
        },
        execute_selected,
    )
}

struct SelectedBound {
    runtime: runmat_runtime::context::RuntimeContext,
    role: SelectedRole,
    invocation: BoundInvocation,
}

fn execute_selected(
    selected: SelectedBound,
) -> impl std::future::Future<Output = VmResult<runmat_runtime::execution::RootedValueSequence>> {
    use futures::future::Either;
    match selected.role {
        SelectedRole::Semantic => {
            Either::Left(invoke_semantic_bound(selected.runtime, selected.invocation))
        }
        SelectedRole::Constructor(constructor) => Either::Right(admit_constructor(
            selected.runtime,
            selected.invocation,
            constructor,
        )),
    }
}

async fn invoke_semantic_bound(
    runtime: runmat_runtime::context::RuntimeContext,
    invocation: BoundInvocation,
) -> VmResult<runmat_runtime::execution::RootedValueSequence> {
    let BoundInvocation {
        binding,
        args,
        requested_outputs,
        registry,
        driver,
        lexical_environment,
    } = invocation;
    let result = super::invoke_semantic_function_with_rooted_captures(
        binding.function,
        args,
        requested_outputs,
        registry,
        runtime,
        driver,
        lexical_environment,
    )
    .await
    .map(|(value, _)| value);
    // The original selected binding remains a source lease through completion,
    // including originating errors; selection never substitutes a later owner.
    drop(binding);
    result
}

enum SelectedRole {
    Semantic,
    Constructor(SelectedConstructor),
}

fn select_bound(
    runtime: runmat_runtime::context::RuntimeContext,
    binding: &runmat_value::BoundFunctionBinding,
    registry: &FunctionRegistry,
) -> VmResult<(runmat_runtime::context::RuntimeContext, SelectedRole)> {
    let function = runmat_hir::FunctionId(binding.function);
    let body = registry
        .get(function)
        .ok_or_else(|| mex("UnavailableBoundFunction", "selected bound body is absent"))?;
    let (runtime, selected_method) = registry.prepare_selected_receiver_method(runtime, binding)?;
    let (runtime, super_body) = consume_super_body(runtime, binding, body)?;
    // A selected graph constructor already carries unit-qualified method
    // bindings. Preserve only facts for this exact constructor; the product's
    // existing definition/seal checks still authenticate them before allocation.
    let selected_constructor = body.class_method_owner.as_ref().is_some_and(|owner| {
        owner.role == runmat_types::ClassMethodRole::Constructor
            && !owner.is_static
            && runtime.selected_object_class().is_some_and(|facts| {
                facts.constructor() == binding && facts.class() == &owner.declaring_class
            })
    });
    let runtime = if selected_method || super_body || selected_constructor {
        runtime
    } else {
        runtime.without_selected_object_class()
    };
    // A compiled direct constructor can enter its retained parent without
    // inherited receiver facts. Authenticate this entered local body's own
    // product instead of comparing it to a later registered numeric callable.
    // An existing Derived receiver product must stay the original owner.
    let runtime = if super_body
        && runtime.selected_object_class().is_none()
        && matches!(
            binding.authority,
            runmat_value::BoundFunctionAuthority::Local { .. }
        ) {
        select_constructor(binding, body)?
            .product
            .invocation_runtime_with_binding(runtime, registry, binding)?
    } else {
        runtime
    };
    if super_body
        || body
            .class_method_owner
            .as_ref()
            .is_none_or(|owner| owner.role != runmat_types::ClassMethodRole::Constructor)
    {
        return Ok((runtime, SelectedRole::Semantic));
    }

    let constructor = select_constructor(binding, body)?;
    Ok((runtime, SelectedRole::Constructor(constructor)))
}

async fn admit_constructor(
    runtime: runmat_runtime::context::RuntimeContext,
    invocation: BoundInvocation,
    constructor: SelectedConstructor,
) -> VmResult<runmat_runtime::execution::RootedValueSequence> {
    let selected_runtime = runtime.clone();
    // Keep effectful constructor admission on its original first-poll path.
    runmat_runtime::execution::admit_executable_sequence_with(&runtime, move || {
        invoke_constructor(selected_runtime, invocation, constructor)
    })?
    .await
}

struct BoundInvocation {
    binding: runmat_value::BoundFunctionBinding,
    args: runmat_runtime::call::RootedFunctionArguments,
    requested_outputs: usize,
    registry: FunctionRegistry,
    driver: Option<ProcessDriverHandle>,
    lexical_environment: Option<runmat_value::LexicalEnvironment>,
}

struct SelectedConstructor {
    owner: runmat_types::ClassMethodOwner,
    product: std::rc::Rc<crate::SelectedClassProduct>,
    portable: bool,
}

fn select_constructor(
    binding: &runmat_value::BoundFunctionBinding,
    body: &crate::FunctionBytecode,
) -> VmResult<SelectedConstructor> {
    let owner = body
        .class_method_owner
        .as_ref()
        .expect("constructor role checked above");
    if owner.is_static {
        return Err(mex(
            "ConstructorOwnerInvalid",
            "selected constructor has a static method owner",
        ));
    }
    let product = body.selected_class_product.as_ref().ok_or_else(|| {
        mex(
            "ConstructorOwnerInvalid",
            "selected constructor lacks its exact class product",
        )
    })?;
    if product.class() != &owner.declaring_class {
        return Err(mex(
            "ConstructorOwnerInvalid",
            "selected constructor disagrees with its class product",
        ));
    }
    let portable = if let Some(origin) = binding.authority.portable_origin() {
        if body.binding_origin != Some(origin) || body.selected_source_owner.is_none() {
            return Err(mex(
                "ConstructorOwnerInvalid",
                "portable constructor differs from its admitted source owner",
            ));
        }
        true
    } else {
        false
    };
    Ok(SelectedConstructor {
        owner: owner.clone(),
        product: std::rc::Rc::clone(product),
        portable,
    })
}

async fn invoke_constructor(
    runtime: runmat_runtime::context::RuntimeContext,
    invocation: BoundInvocation,
    constructor: SelectedConstructor,
) -> VmResult<runmat_runtime::execution::RootedValueSequence> {
    let BoundInvocation {
        binding,
        args,
        requested_outputs,
        registry,
        driver,
        lexical_environment,
    } = invocation;
    let SelectedConstructor {
        owner,
        product,
        portable,
    } = constructor;
    // A local constructor has the same exact leased receiver-facts owner as
    // Core's bound service. Projection alone supplies defaults, not the proof
    // that this executable is that constructor after registry republication.
    // Portable selection retains its existing admitted Graph/projection route;
    // never manufacture a local definition lease for that route.
    let runtime = if portable {
        product.invocation_runtime(runtime)?
    } else {
        product.invocation_runtime_with_binding(runtime, &registry, &binding)?
    };
    let argument_count = args.supplied_count();
    let invocation_owner = owner.clone();
    let invoked_function = binding.function;
    let invoke = runmat_runtime::dispatcher::PreparedConstructorInvoke::new(
        &runtime,
        move |selected| async move {
            let selected =
                selected.with_constructor_receiver_for_method(Some(&invocation_owner))?;
            super::invoke_semantic_function_with_rooted_captures(
                invoked_function,
                args,
                requested_outputs,
                registry,
                selected,
                driver,
                lexical_environment,
            )
            .await
            .map(|(value, _)| value)
        },
    )?;
    if portable {
        runmat_runtime::dispatcher::call_portable_prepared_constructor_sequence_with_argument_count_in_context(
            &runtime,
            owner,
            binding,
            argument_count,
            requested_outputs,
            invoke,
        )
        .await
    } else {
        runmat_runtime::dispatcher::call_prepared_constructor_sequence_with_argument_count_in_context(
            &runtime,
            owner,
            binding,
            argument_count,
            requested_outputs,
            invoke,
        )
        .await
    }
}

pub(super) fn consume_super_body(
    runtime: runmat_runtime::context::RuntimeContext,
    binding: &runmat_value::BoundFunctionBinding,
    body: &crate::FunctionBytecode,
) -> VmResult<(runmat_runtime::context::RuntimeContext, bool)> {
    let (runtime, expected) = runtime.take_retained_super_body_for(binding)?;
    let super_body = expected.is_some();
    if let Some(expected) = expected {
        if body.class_method_owner.as_ref() != Some(&expected)
            || expected.role != runmat_types::ClassMethodRole::Constructor
            || expected.is_static
        {
            return Err(mex(
                "SuperConstructorOwnerInvalid",
                "selected superclass body disagrees with its admitted owner",
            ));
        }
    }
    Ok((runtime, super_body))
}

pub(super) fn bind_selected(
    context: &mut crate::bytecode::program::ExecutionContext,
    bytecode: &Bytecode,
    registry: &FunctionRegistry,
    execution_sources: &runmat_runtime::user_functions::ExecutionSourceFacts,
) -> VmResult<()> {
    let Some(caller) = bytecode.active_function else {
        return Ok(());
    };
    if !bytecode
        .instructions
        .iter()
        .any(|instruction| crate::bytecode::program::instruction_classes(instruction).is_some())
    {
        if let Some(origin) = registry.get(caller).and_then(|body| body.binding_origin) {
            context.runtime = context
                .runtime
                .clone()
                .with_retained_super_caller_origin(origin);
        }
        return Ok(());
    }
    let targets = registry
        .retained_super_constructor_targets_for(caller)
        .map_err(|error| mex("SuperConstructorOwnerInvalid", &error.to_string()))?;
    // A declaration-only parent has no FunctionBytecode in this invocation's
    // registry. Its exact admitted bytes and source lease instead live on the
    // selected constructor's immutable class product, including after the
    // original dynamic publication retires. Never resolve it by class name.
    let linked = registry
        .get(caller)
        .and_then(|body| body.selected_class_product.as_ref())
        .map(|product| {
            product
                .declaration_products()
                .iter()
                .map(|declaration| {
                    let mut matching = declaration.declarations().iter().filter_map(
                        |instruction| match instruction {
                            crate::Instr::RegisterClass {
                                name, super_class, ..
                            } if name == declaration.class() => {
                                Some(runmat_runtime::context::RetainedClassDeclaration {
                                    class: name.clone(),
                                    parent: super_class.clone(),
                                    constructor: None,
                                })
                            }
                            _ => None,
                        },
                    );
                    let selected = matching.next().ok_or_else(|| {
                        mex(
                            "SuperConstructorOwnerInvalid",
                            "selected declaration-only superclass lacks its exact class metadata",
                        )
                    })?;
                    if matching.next().is_some() {
                        return Err(mex(
                            "SuperConstructorOwnerInvalid",
                            "selected declaration-only superclass has ambiguous class metadata",
                        ));
                    }
                    Ok(selected)
                })
                .collect::<VmResult<Vec<_>>>()
        })
        .transpose()?
        .unwrap_or_default();
    let implicit = registry
        .retained_implicit_super_constructor_targets_with_linked(caller, &linked)
        .map_err(|error| mex("SuperConstructorOwnerInvalid", &error.to_string()))?;
    let Some(caller_origin) = registry.get(caller).and_then(|body| body.binding_origin) else {
        if targets.is_empty() && implicit.is_empty() {
            return Ok(());
        }
        return Err(mex(
            "SuperConstructorOwnerInvalid",
            "superclass caller lacks immutable source identity",
        ));
    };
    if targets.is_empty() && implicit.is_empty() {
        context.runtime = context
            .runtime
            .clone()
            .with_retained_super_caller_origin(caller_origin);
        return Ok(());
    }
    let mut bindings = targets
        .into_iter()
        .map(|target| {
            let binding =
                runmat_runtime::user_functions::resolve_binding_origin_binding_in_context(
                    execution_sources,
                    target.origin,
                    &context.runtime,
                )
                .ok_or_else(|| {
                    mex(
                        "SuperConstructorOwnerInvalid",
                        "selected superclass lacks its exact executable definition",
                    )
                })?;
            runmat_runtime::context::RetainedSuperConstructorBinding::from_admitted_definition(
                &target.caller_declaration,
                &target.target_declaration,
                target.caller_origin,
                target.origin,
                target.owner,
                binding,
            )
        })
        .collect::<VmResult<Vec<_>>>()?;
    bindings.extend(
        implicit
            .into_iter()
            .map(|target| target.into_binding())
            .collect::<VmResult<Vec<_>>>()?,
    );
    context.runtime = context
        .runtime
        .clone()
        .with_retained_super_constructors(bindings, caller_origin)?;
    Ok(())
}

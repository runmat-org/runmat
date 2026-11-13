use crate::bytecode::instr::PropertyDefaultLiteral;
use crate::call::builtins as call_builtins;
use crate::call::builtins::ImportedBuiltinResolution;
use crate::call::closures as call_closures;
use crate::call::shared::build_expanded_args_from_specs;
use crate::call::{execute_callable_descriptor, execute_callable_descriptor_owned, root_values};
use crate::interpreter::debug;
use crate::interpreter::dispatch::exceptions::{redirect_exception_to_catch, ExceptionHandling};
use crate::interpreter::vars::InterpreterVars;
use crate::object::class_def as obj_class_def;
use runmat_hir::{CallableFallbackPolicy, CallableIdentity};
use runmat_runtime::call::arguments::ArgumentSpec;
use runmat_runtime::call::descriptor::{CallableCallKind, CallableDescriptor, CallableTarget};
use runmat_runtime::object::resolve as obj_resolve;
use runmat_runtime::RuntimeError;
use runmat_value::{MException, Value};

/// Select only a compiler-enumerated instruction reference. A selected graph
/// may contain the same numeric FunctionId in several admitted V3 units; the
/// site and current exact VM body, not that number, choose its bound target.
pub(super) fn selected_static_call_binding(
    runtime: &runmat_runtime::context::RuntimeContext,
    instruction: &crate::bytecode::Instr,
    pc: usize,
    function: runmat_hir::FunctionId,
) -> Result<Option<runmat_value::BoundFunctionBinding>, RuntimeError> {
    if !runtime.has_selected_static_site_resolver() {
        return Ok(None);
    }
    let sites = crate::bytecode::program::instruction_reference_sites(instruction, pc)
        .map_err(|message| crate::interpreter::errors::mex("StaticSiteInvalid", message))?;
    let mut matching = sites
        .into_iter()
        .filter(|(_, referenced, _)| *referenced == function);
    let Some((site, _, kind)) = matching.next() else {
        return Ok(None);
    };
    if matching.next().is_some() {
        return Err(crate::interpreter::errors::mex(
            "StaticSiteAmbiguous",
            "one instruction repeats a selected function reference",
        ));
    }
    let selected = runtime
        .selected_static_binding(site)
        .map_err(|message| crate::interpreter::errors::mex("StaticImportUnavailable", message))?;
    if selected.is_none() && kind == crate::bytecode::program::FunctionReferenceKind::External {
        return Err(crate::interpreter::errors::mex(
            "StaticImportUnavailable",
            "external call has no exact selected graph target",
        ));
    }
    Ok(selected)
}

#[path = "calls/dispatch.rs"]
mod dispatch;
pub(super) use dispatch::{dispatch_call, CallDispatchContext};
#[path = "calls/user_preparation.rs"]
mod user_preparation;

pub enum BuiltinHandling {
    Completed(runmat_runtime::execution::RootedValueSequence),
    Caught,
    Uncaught(Box<RuntimeError>),
}

/// `None` means the method handled publication itself; `Some` transfers its
/// existing rooted output without inflating a two-variant enum.
pub type MethodHandling = Option<runmat_runtime::execution::RootedValueSequence>;

pub enum UserCallHandling {
    Completed(runmat_runtime::execution::RootedValueSequence),
    Caught,
    Uncaught(Box<RuntimeError>),
}

pub(super) fn root_sequence(
    sequence: runmat_value::ValueSequence,
) -> Result<runmat_runtime::execution::RootedValueSequence, RuntimeError> {
    runmat_runtime::execution::RootedValueSequence::new(sequence)
        .map_err(|error| error.into_runtime_error())
}

pub(super) fn publish_fixed_outputs(
    stack: &mut crate::interpreter::stack::InterpreterStack,
    sequence: runmat_value::ValueSequence,
    requested_outputs: impl std::borrow::Borrow<usize>,
) -> Result<(), RuntimeError> {
    use runmat_runtime::sequence::ResolveValueSequence;

    let requested_outputs = *requested_outputs.borrow();
    let values = sequence.resolve(
        runmat_types::SequenceUse::SelectPrefix {
            count: requested_outputs,
        },
        runmat_runtime::sequence::SequenceResolutionContext::default(),
    )?;
    stack.try_extend_output(values)?;
    Ok(())
}

pub(super) fn publish_rooted_fixed_outputs(
    stack: &mut crate::interpreter::stack::InterpreterStack,
    sequence: runmat_runtime::execution::RootedValueSequence,
    requested_outputs: impl std::borrow::Borrow<usize>,
) -> Result<(), RuntimeError> {
    sequence
        .with_owned_sequence(|sequence| publish_fixed_outputs(stack, sequence, requested_outputs))
}

pub struct MethodExpandCallContext<'a> {
    pub identity: CallableIdentity,
    pub fallback_policy: CallableFallbackPolicy,
    pub specs: &'a [ArgumentSpec],
    pub requested_outputs: usize,
    pub current_function_name: &'a str,
    pub runtime: &'a runmat_runtime::context::RuntimeContext,
}

pub struct ExceptionRouteContext<'a> {
    pub try_stack: &'a mut Vec<crate::interpreter::state::ActiveTryHandler>,
    pub vars: &'a mut InterpreterVars,
    pub last_exception: &'a mut Option<MException>,
    pub pc: &'a mut usize,
}

pub struct BuiltinCallContext<'a> {
    pub runtime: &'a runmat_runtime::context::RuntimeContext,
    pub process_driver: Option<crate::interpreter::process_driver::ProcessDriverHandle>,
    pub stack: &'a mut crate::interpreter::stack::InterpreterStack,
    pub name: &'a str,
    pub arg_count: usize,
    pub source_id: Option<runmat_hir::SourceId>,
    pub call_arg_spans: Option<runmat_runtime::callsite::CallsiteSpans>,
    pub imports: &'a [(Vec<String>, bool)],
    pub invocation_arity: runmat_runtime::context::RuntimeInvocationArity,
    pub function_registry: &'a crate::bytecode::FunctionRegistry,
    pub current_function_name: &'a str,
    pub exception: ExceptionRouteContext<'a>,
}

pub struct UserCallContext<'a> {
    pub stack: &'a mut crate::interpreter::stack::InterpreterStack,
    pub runtime: &'a runmat_runtime::context::RuntimeContext,
    pub process_driver: Option<crate::interpreter::process_driver::ProcessDriverHandle>,
    pub identity: CallableIdentity,
    pub bound: Option<runmat_value::BoundFunctionBinding>,
    pub fallback_policy: CallableFallbackPolicy,
    pub out_count: usize,
    pub source_id: Option<runmat_hir::SourceId>,
    pub source_ref: Option<runmat_runtime::source_context::RuntimeSourceRef>,
    pub call_arg_spans: Option<runmat_runtime::callsite::CallsiteSpans>,
    pub current_function_name: &'a str,
    pub imports: &'a [(Vec<String>, bool)],
    pub exception: ExceptionRouteContext<'a>,
}

/// Only a catalog-proven non-callback builtin may stay on the caller's poll
/// stack. Semantic and lexical targets can re-enter the VM through their
/// selected invoker; unknown effects therefore take the sibling continuation.
fn descriptor_needs_continuation(descriptor: &CallableDescriptor) -> bool {
    let CallableTarget::Resolved { identity, .. } = &descriptor.target else {
        return true;
    };
    let CallableIdentity::Builtin(id) = identity.as_identity() else {
        return true;
    };
    let semantics = runmat_builtins::builtin_catalog_entry_by_name(&id.0)
        .map(runmat_builtins::BuiltinCatalogEntry::legacy_semantics)
        .or_else(|| runmat_builtins::builtin_semantics_for_name(&id.0));
    semantics.is_none_or(|semantics| semantics.effects.host_callback || semantics.effects.unknown)
}

pub struct WorkspaceFirstCallContext<'a> {
    pub stack: &'a mut crate::interpreter::stack::InterpreterStack,
    pub runtime: &'a runmat_runtime::context::RuntimeContext,
    pub process_driver: Option<crate::interpreter::process_driver::ProcessDriverHandle>,
    pub workspace_name: &'a str,
    pub identity: CallableIdentity,
    pub fallback_policy: CallableFallbackPolicy,
    pub bare_identifier: bool,
    pub out_count: usize,
    pub source_id: Option<runmat_hir::SourceId>,
    pub source_ref: Option<runmat_runtime::source_context::RuntimeSourceRef>,
    pub call_arg_spans: Option<runmat_runtime::callsite::CallsiteSpans>,
    pub current_function_name: &'a str,
    pub imports: &'a [(Vec<String>, bool)],
    pub function_registry: &'a crate::bytecode::FunctionRegistry,
    pub exception: ExceptionRouteContext<'a>,
}

fn imported_static_method_owner(
    imports: &[(Vec<String>, bool)],
    method_name: &runmat_types::MethodName,
) -> Result<Option<runmat_types::ClassIdentity>, RuntimeError> {
    let mut owners = Vec::new();
    for (path, wildcard) in imports {
        if *wildcard {
            if path.is_empty() {
                continue;
            }
            let class_name = runmat_types::ClassIdentity::new(path.join(".")).map_err(|error| {
                crate::interpreter::errors::mex("InvalidClassIdentity", &error.to_string())
            })?;
            if let Some((method, owner)) =
                runmat_runtime::class_registry::lookup_method(&class_name, method_name)
            {
                if method.is_static && !owners.iter().any(|existing| existing == &owner) {
                    owners.push(owner);
                }
            }
            continue;
        }
        if path.len() < 2 || path[path.len() - 1] != method_name.display_name() {
            continue;
        }
        let class_name = runmat_types::ClassIdentity::new(path[..path.len() - 1].join("."))
            .map_err(|error| {
                crate::interpreter::errors::mex("InvalidClassIdentity", &error.to_string())
            })?;
        if let Some((method, owner)) =
            runmat_runtime::class_registry::lookup_method(&class_name, method_name)
        {
            if method.is_static && !owners.iter().any(|existing| existing == &owner) {
                owners.push(owner);
            }
        }
    }
    if owners.len() > 1 {
        return Err(crate::interpreter::errors::mex(
            "AmbiguousImport",
            &format!(
                "ambiguous static method '{}' via imports: {}",
                method_name,
                owners
                    .iter()
                    .map(runmat_types::ClassIdentity::display_name)
                    .collect::<Vec<_>>()
                    .join(", ")
            ),
        ));
    }
    Ok(owners.into_iter().next())
}

pub async fn build_builtin_expand_multi_args(
    stack: &mut crate::interpreter::stack::InterpreterStack,
    specs: &[ArgumentSpec],
    sequence_state: &mut super::SequenceState,
    runtime: &runmat_runtime::context::RuntimeContext,
) -> Result<runmat_runtime::execution::RootedValues, RuntimeError> {
    build_expanded_args_from_specs(stack, specs, sequence_state, runtime).await
}

pub async fn build_feval_expand_multi_args(
    stack: &mut crate::interpreter::stack::InterpreterStack,
    specs: &[ArgumentSpec],
    sequence_state: &mut super::SequenceState,
    runtime: &runmat_runtime::context::RuntimeContext,
) -> Result<runmat_runtime::execution::RootedValues, RuntimeError> {
    build_expanded_args_from_specs(stack, specs, sequence_state, runtime).await
}

pub async fn build_user_function_expand_multi_args(
    stack: &mut crate::interpreter::stack::InterpreterStack,
    specs: &[ArgumentSpec],
    sequence_state: &mut super::SequenceState,
    runtime: &runmat_runtime::context::RuntimeContext,
) -> Result<runmat_runtime::execution::RootedValues, RuntimeError> {
    build_expanded_args_from_specs(stack, specs, sequence_state, runtime).await
}

pub fn handle_builtin_outcome(
    result: Result<runmat_runtime::execution::RootedValueSequence, RuntimeError>,
    imported: ImportedBuiltinResolution,
    _output_hint: usize,
    stack: &mut crate::interpreter::stack::InterpreterStack,
    ctx: ExceptionRouteContext<'_>,
) -> Result<BuiltinHandling, RuntimeError> {
    let ExceptionRouteContext {
        try_stack,
        vars,
        last_exception,
        pc,
    } = ctx;
    match result {
        Ok(sequence) => Ok(BuiltinHandling::Completed(sequence)),
        Err(err) => match imported {
            ImportedBuiltinResolution::Resolved(sequence) => {
                Ok(BuiltinHandling::Completed(sequence))
            }
            ImportedBuiltinResolution::Ambiguous(err) => Err(err),
            ImportedBuiltinResolution::NotFound => Ok(
                match redirect_exception_to_catch(err, try_stack, vars, last_exception, pc, stack) {
                    ExceptionHandling::Caught => BuiltinHandling::Caught,
                    ExceptionHandling::Uncaught(err) => BuiltinHandling::Uncaught(err),
                },
            ),
        },
    }
}

pub fn handle_builtin_call_multi<'a>(
    ctx: BuiltinCallContext<'a>,
    out_count: usize,
) -> impl std::future::Future<Output = Result<BuiltinHandling, RuntimeError>> + 'a {
    handle_builtin_call_inner(ctx, out_count)
}

async fn handle_builtin_call_inner(
    ctx: BuiltinCallContext<'_>,
    requested_outputs: usize,
) -> Result<BuiltinHandling, RuntimeError> {
    let BuiltinCallContext {
        runtime,
        process_driver,
        stack,
        name,
        arg_count,
        source_id,
        call_arg_spans,
        imports,
        invocation_arity,
        function_registry,
        current_function_name,
        exception,
    } = ctx;
    let ExceptionRouteContext {
        try_stack: _,
        vars: _,
        last_exception: _,
        pc,
    } = &exception;
    debug::trace_call_builtin(**pc, name, arg_count, stack);
    if call_builtins::is_vm_intrinsic_builtin(name) {
        let result = call_builtins::vm_intrinsic_builtin(stack, name, arg_count, invocation_arity);
        return handle_builtin_outcome(
            result
                .and_then(runmat_runtime::call::arguments::adapt_legacy_builtin_result)
                .and_then(root_sequence),
            ImportedBuiltinResolution::NotFound,
            requested_outputs,
            stack,
            exception,
        );
    }
    if call_builtins::is_vm_dynamic_workspace_builtin(name) {
        let result = call_builtins::vm_dynamic_workspace_builtin_with_driver(
            runtime,
            stack,
            name,
            arg_count,
            requested_outputs,
            function_registry,
            call_builtins::DynamicBuiltinSource {
                source_id,
                driver: process_driver,
                dynamic_admission: None,
            },
        )
        .await;
        return handle_builtin_outcome(
            result.and_then(|result| {
                result.try_map_owned(
                    runmat_runtime::call::arguments::adapt_legacy_builtin_result,
                    |error| error.into_runtime_error(),
                )
            }),
            ImportedBuiltinResolution::NotFound,
            requested_outputs,
            stack,
            exception,
        );
    }
    let args = call_builtins::collect_call_args(stack, arg_count)?;
    let _roots = root_values(&args)?;

    let _callsite_guard = runmat_runtime::callsite::push_callsite_spans(source_id, call_arg_spans)?;
    let _output_guard = runmat_runtime::output_context::push_output_count(requested_outputs);
    let current_class_context =
        runmat_runtime::class_registry::class_context_for_function(current_function_name);
    let _access_guard = current_class_context
        .map(|class_name| runmat_runtime::push_class_access_context(Some(class_name)))
        .transpose()?;

    let result =
        call_builtins::call_builtin_or_imported(runtime, name, imports, &args, requested_outputs)
            .await;
    let result = match result {
        Ok(sequence) => match runtime.vm_fusion_host_route() {
            Some(route) => route
                .retire_builtin_operands(runtime, args)
                .map(|()| sequence),
            None => Ok(sequence),
        },
        Err(error) => Err(error),
    };
    handle_builtin_outcome(
        result,
        ImportedBuiltinResolution::NotFound,
        requested_outputs,
        stack,
        exception,
    )
}

pub async fn handle_prepared_user_function_call(
    ctx: UserCallContext<'_>,
    args: Vec<Value>,
) -> Result<UserCallHandling, RuntimeError> {
    let _roots = root_values(&args)?;
    let UserCallContext {
        stack,
        runtime,
        process_driver,
        identity,
        bound,
        fallback_policy,
        out_count,
        source_id,
        source_ref,
        call_arg_spans,
        current_function_name,
        imports,
        exception,
    } = ctx;
    let current_class_context =
        runmat_runtime::class_registry::class_context_for_function(current_function_name);
    let ExceptionRouteContext {
        try_stack,
        vars,
        last_exception,
        pc,
    } = exception;

    let preparation_callsite_guard = runmat_runtime::callsite::push_function_input_callsite_spans(
        source_id,
        call_arg_spans.clone(),
    )?;
    let mut prepared = user_preparation::prepare(
        identity,
        bound,
        fallback_policy,
        out_count,
        args,
        current_class_context.as_ref(),
        imports,
    )?;
    prepared.descriptor.metadata.source_id = source_id;
    prepared.descriptor.metadata.source_ref = source_ref;
    prepared.descriptor.metadata.source_invocation = Some(
        runtime
            .invocation_arity()
            .unwrap_or(runmat_runtime::context::RuntimeInvocationArity::Script),
    );
    drop(preparation_callsite_guard);
    let result = if let Some(driver) =
        process_driver.filter(|_| descriptor_needs_continuation(&prepared.descriptor))
    {
        let callsite = call_arg_spans.map(|arg_spans| runmat_runtime::callsite::CallsiteInfo {
            source_id,
            arg_spans,
        });
        let class_context = (prepared.route != user_preparation::UserCallRoute::StaticMethod)
            .then_some(current_class_context)
            .flatten();
        let selected_runtime = runtime.clone();
        let prepared_scope = selected_runtime.prepare_poll_scope()?;
        // The descriptor owns its target and arguments. Polling it as a sibling
        // keeps a nested semantic callback off the caller's interpreter stack.
        match driver.enqueue_scoped_in_context(runtime, prepared_scope, move || async move {
            let descriptor_future =
                execute_callable_descriptor_owned(selected_runtime, prepared.descriptor);
            let call = async move {
                let output = descriptor_future.await?;
                let captures = runmat_runtime::execution::RootedValues::new(Vec::new())
                    .map_err(|error| error.into_runtime_error())?;
                Ok((output, captures))
            };
            let mut call = std::pin::pin!(runmat_runtime::callsite::with_function_input_callsite(
                callsite, call,
            ));
            std::future::poll_fn(move |context| {
                let access_guard = class_context
                    .clone()
                    .map(|class_name| runmat_runtime::push_class_access_context(Some(class_name)))
                    .transpose();
                let _access_guard = match access_guard {
                    Ok(guard) => guard,
                    Err(error) => return std::task::Poll::Ready(Err(error)),
                };
                std::future::Future::poll(call.as_mut(), context)
            })
            .await?
        }) {
            Ok(ticket) => ticket.await.map(|(output, _captures)| output),
            Err(error) => Err(error),
        }
    } else {
        let _function_input_callsite_guard =
            runmat_runtime::callsite::push_function_input_callsite_spans(
                source_id,
                call_arg_spans,
            )?;
        let _access_guard = if prepared.route == user_preparation::UserCallRoute::StaticMethod {
            None
        } else {
            current_class_context
                .map(|class_name| runmat_runtime::push_class_access_context(Some(class_name)))
                .transpose()?
        };
        execute_callable_descriptor(runtime, prepared.descriptor).await
    };
    if prepared.route == user_preparation::UserCallRoute::StaticMethod {
        return result.map(UserCallHandling::Completed);
    }
    match result {
        Ok(result) => Ok(UserCallHandling::Completed(result)),
        Err(err) => Ok(
            match redirect_exception_to_catch(err, try_stack, vars, last_exception, pc, stack) {
                ExceptionHandling::Caught => UserCallHandling::Caught,
                ExceptionHandling::Uncaught(err) => UserCallHandling::Uncaught(err),
            },
        ),
    }
}

pub async fn handle_user_function_call(
    ctx: UserCallContext<'_>,
    arg_count: usize,
) -> Result<UserCallHandling, RuntimeError> {
    let args = crate::call::builtins::collect_call_args(ctx.stack, arg_count)?;
    handle_prepared_user_function_call(ctx, args).await
}

pub async fn handle_workspace_first_prepared_call(
    ctx: WorkspaceFirstCallContext<'_>,
    args: Vec<Value>,
) -> Result<UserCallHandling, RuntimeError> {
    let _roots = root_values(&args)?;
    let WorkspaceFirstCallContext {
        stack,
        runtime,
        process_driver,
        workspace_name,
        identity,
        fallback_policy,
        bare_identifier,
        out_count,
        source_id,
        source_ref,
        call_arg_spans,
        current_function_name,
        imports,
        function_registry,
        exception,
    } = ctx;

    if let Some(base) = crate::runtime::workspace::workspace_lookup(workspace_name) {
        if bare_identifier && args.is_empty() {
            return Ok(UserCallHandling::Completed(root_sequence(
                runmat_value::ValueSequence::single(base)
                    .map_err(runmat_runtime::sequence::sequence_error_to_runtime)?,
            )?));
        }
        let _callsite_guard =
            runmat_runtime::callsite::push_callsite_spans(source_id, call_arg_spans)?;
        let _output_guard = runmat_runtime::output_context::push_output_count(out_count);
        let result = super::indexing::paren_index_sequence(
            runtime,
            base,
            args,
            out_count,
            function_registry,
        )
        .await;
        let ExceptionRouteContext {
            try_stack,
            vars,
            last_exception,
            pc,
        } = exception;
        return match result {
            Ok(sequence) => Ok(UserCallHandling::Completed(sequence)),
            Err(err) => Ok(
                match redirect_exception_to_catch(err, try_stack, vars, last_exception, pc, stack) {
                    ExceptionHandling::Caught => UserCallHandling::Caught,
                    ExceptionHandling::Uncaught(err) => UserCallHandling::Uncaught(err),
                },
            ),
        };
    }

    if let CallableIdentity::Builtin(id) = &identity {
        if call_builtins::is_vm_dynamic_workspace_builtin(&id.0) {
            let mut temp_stack = crate::interpreter::stack::InterpreterStack::from_unbounded(args);
            let arg_count = temp_stack.len();
            let result = call_builtins::vm_dynamic_workspace_builtin_with_driver(
                runtime,
                &mut temp_stack,
                &id.0,
                arg_count,
                out_count,
                function_registry,
                call_builtins::DynamicBuiltinSource {
                    source_id,
                    driver: process_driver,
                    dynamic_admission: None,
                },
            )
            .await;
            let ExceptionRouteContext {
                try_stack,
                vars,
                last_exception,
                pc,
            } = exception;
            return match result {
                Ok(value) => Ok(UserCallHandling::Completed(value.try_map_owned(
                    runmat_runtime::call::arguments::adapt_legacy_builtin_result,
                    |error| error.into_runtime_error(),
                )?)),
                Err(err) => Ok(
                    match redirect_exception_to_catch(
                        err,
                        try_stack,
                        vars,
                        last_exception,
                        pc,
                        stack,
                    ) {
                        ExceptionHandling::Caught => UserCallHandling::Caught,
                        ExceptionHandling::Uncaught(err) => UserCallHandling::Uncaught(err),
                    },
                ),
            };
        }
    }

    handle_prepared_user_function_call(
        UserCallContext {
            stack,
            runtime,
            process_driver,
            identity,
            bound: None,
            fallback_policy,
            out_count,
            source_id,
            source_ref,
            call_arg_spans,
            current_function_name,
            imports,
            exception,
        },
        args,
    )
    .await
}

pub async fn handle_method_or_member_index_multi_call(
    runtime: &runmat_runtime::context::RuntimeContext,
    stack: &mut crate::interpreter::stack::InterpreterStack,
    identity: CallableIdentity,
    fallback_policy: CallableFallbackPolicy,
    arg_count: usize,
    out_count: usize,
    current_function_name: &str,
) -> Result<MethodHandling, RuntimeError> {
    handle_method_or_member_index_call_inner(
        runtime,
        stack,
        identity,
        fallback_policy,
        arg_count,
        out_count,
        current_function_name,
    )
    .await
}

async fn handle_method_or_member_index_call_inner(
    runtime: &runmat_runtime::context::RuntimeContext,
    stack: &mut crate::interpreter::stack::InterpreterStack,
    identity: CallableIdentity,
    fallback_policy: CallableFallbackPolicy,
    arg_count: usize,
    requested_outputs: usize,
    current_function_name: &str,
) -> Result<MethodHandling, RuntimeError> {
    let (base, args) = call_closures::collect_method_args(stack, arg_count)?;
    let _roots = root_values(args.iter().chain(std::iter::once(&base)))?;
    let _output_guard = runmat_runtime::output_context::push_output_count(requested_outputs);
    let current_class_context =
        runmat_runtime::class_registry::class_context_for_function(current_function_name);
    let _access_guard = current_class_context
        .map(|class_name| runmat_runtime::push_class_access_context(Some(class_name)))
        .transpose()?;
    let outputs = runmat_runtime::execution::admit_executable_sequence_with(runtime, move || {
        runmat_runtime::object::dispatch::call_method_or_member_index_with_outputs_in_context(
            runtime,
            base,
            identity,
            args,
            requested_outputs,
            (!current_function_name.is_empty()).then_some(current_function_name),
            fallback_policy,
        )
    })?
    .await?;
    Ok(Some(outputs))
}

pub async fn handle_method_or_member_index_expand_multi_call(
    stack: &mut crate::interpreter::stack::InterpreterStack,
    sequence_state: &mut super::SequenceState,
    context: MethodExpandCallContext<'_>,
) -> Result<MethodHandling, RuntimeError> {
    let args = build_user_function_expand_multi_args(
        stack,
        context.specs,
        sequence_state,
        context.runtime,
    )
    .await?;
    if args.is_empty() {
        return Err(crate::interpreter::errors::mex(
            "MethodCallMissingReceiver",
            "method/member-index call requires a base receiver",
        ));
    }
    let _output_guard =
        runmat_runtime::output_context::push_output_count(context.requested_outputs);
    let current_class_context =
        runmat_runtime::class_registry::class_context_for_function(context.current_function_name);
    let _access_guard = current_class_context
        .map(|class_name| runmat_runtime::push_class_access_context(Some(class_name)))
        .transpose()?;
    let outputs = args
        .with_unbounded_values_async(|mut args| async move {
            let base = args.remove(0);
            runmat_runtime::execution::admit_executable_sequence_with(context.runtime, move || {
                runmat_runtime::object::dispatch::call_method_or_member_index_with_outputs_in_context(
                    context.runtime,
                    base,
                    context.identity,
                    args,
                    context.requested_outputs,
                    (!context.current_function_name.is_empty())
                        .then_some(context.current_function_name),
                    context.fallback_policy,
                )
            })?
            .await
        })
        .await??;
    Ok(Some(outputs))
}

pub fn handle_load_method(
    stack: &mut crate::interpreter::stack::InterpreterStack,
    name: String,
    current_function_name: &str,
) -> Result<MethodHandling, RuntimeError> {
    let base = crate::interpreter::stack::pop_value(stack)?;
    let value = runmat_runtime::call::closures::load_method_closure(
        base,
        name,
        (!current_function_name.is_empty()).then_some(current_function_name),
    )?;
    stack.try_push(value)?;
    Ok(None)
}

pub fn handle_create_closure(
    stack: &mut crate::interpreter::stack::InterpreterStack,
    func_name: String,
    capture_count: usize,
) -> Result<MethodHandling, RuntimeError> {
    call_closures::create_closure(stack, func_name, capture_count)?;
    Ok(None)
}

pub fn handle_create_semantic_closure(
    stack: &mut crate::interpreter::stack::InterpreterStack,
    function: runmat_hir::FunctionId,
    display_name: String,
    capture_count: usize,
    source_role: runmat_types::FunctionSourceRole,
    lexical_environment: Option<runmat_value::LexicalEnvironment>,
) -> Result<MethodHandling, RuntimeError> {
    call_closures::create_semantic_closure(
        stack,
        function,
        display_name,
        capture_count,
        source_role,
        lexical_environment,
    )?;
    Ok(None)
}

pub fn handle_load_static_property(
    stack: &mut crate::interpreter::stack::InterpreterStack,
    class_identity: &runmat_types::ClassIdentity,
    prop: &str,
) -> Result<MethodHandling, RuntimeError> {
    let value = obj_resolve::load_static_member(class_identity, prop, None)?;
    stack.try_push(value)?;
    Ok(None)
}

pub fn handle_register_class(
    registration: obj_class_def::ClassRegistration<crate::bytecode::instr::BytecodeClassProperty>,
) -> Result<MethodHandling, RuntimeError> {
    let checked = checked_class_registration(registration);
    runmat_runtime::class_registry::register_open_class_declarations(vec![checked])?;
    Ok(None)
}

pub(crate) fn checked_class_registration(
    registration: obj_class_def::ClassRegistration<crate::bytecode::instr::BytecodeClassProperty>,
) -> runmat_runtime::class_registry::CheckedClassRegistration {
    let registration = registration.map_properties(|property| {
        let default_value = property.default_literal.map(|literal| match literal {
            PropertyDefaultLiteral::Num(value) => Value::Num(value),
            PropertyDefaultLiteral::Int(value) => Value::Int(value),
            PropertyDefaultLiteral::Bool(value) => Value::Bool(value),
            PropertyDefaultLiteral::String(value) => Value::String(value.into()),
        });
        crate::object::class_def::RuntimeClassPropertyRegistration {
            name: property.name,
            is_static: property.is_static,
            is_constant: property.is_constant,
            is_dependent: property.is_dependent,
            default_value,
            get_access: property.get_access,
            set_access: property.set_access,
        }
    });
    obj_class_def::checked_class_registration(registration)
}

#[cfg(test)]
mod tests {
    use super::{
        descriptor_needs_continuation, handle_builtin_outcome, publish_fixed_outputs,
        ExceptionRouteContext,
    };
    use crate::call::builtins::ImportedBuiltinResolution;
    use crate::interpreter::errors::mex;
    use crate::interpreter::vars::InterpreterVars;
    use runmat_value::Value;

    #[test]
    fn only_catalog_proven_noncallback_builtins_stay_inline() {
        let builtin = |name: &str| {
            runmat_runtime::call::descriptor::CallableDescriptor::resolved(
                runmat_hir::CallableIdentity::Builtin(runmat_types::BuiltinId(name.into())),
                Vec::new(),
                1,
                runmat_hir::CallableFallbackPolicy::None,
                runmat_runtime::call::descriptor::CallableCallKind::Direct,
            )
        };
        assert!(!descriptor_needs_continuation(&builtin("zeros")));
        assert!(descriptor_needs_continuation(&builtin("feval")));
        assert!(descriptor_needs_continuation(&builtin("not_catalogued")));
    }

    #[test]
    fn fixed_outputs_publish_values_without_a_transient_carrier() {
        let mut stack = crate::interpreter::stack::InterpreterStack::unbounded();
        publish_fixed_outputs(
            &mut stack,
            runmat_value::ValueSequence::comma_separated(vec![Value::Num(7.0)]).unwrap(),
            1,
        )
        .unwrap();
        assert_eq!(stack.as_slice(), &[Value::Num(7.0)]);
    }

    #[test]
    fn fixed_zero_outputs_leave_no_stack_value() {
        let mut stack = crate::interpreter::stack::InterpreterStack::unbounded();
        publish_fixed_outputs(
            &mut stack,
            runmat_value::ValueSequence::single(Value::Num(7.0)).unwrap(),
            0,
        )
        .unwrap();
        assert!(stack.is_empty());
    }

    #[test]
    fn checked_sequence_rejects_nested_legacy_carriers_before_publication() {
        let error = runmat_value::ValueSequence::comma_separated(vec![Value::output_list(vec![
            Value::Num(7.0),
        ])])
        .unwrap_err();
        assert!(matches!(
            error,
            runmat_value::ValueSequenceError::TransientValue(_)
        ));
    }

    #[test]
    fn handle_builtin_outcome_preserves_ambiguous_import_identifier() {
        let mut try_stack: Vec<crate::interpreter::state::ActiveTryHandler> = Vec::new();
        let mut vars = InterpreterVars::from_owned(Vec::new());
        let mut last_exception = None;
        let mut pc = 0usize;
        let mut stack = crate::interpreter::stack::InterpreterStack::unbounded();

        let outcome = handle_builtin_outcome(
            Err("primary builtin failure".into()),
            ImportedBuiltinResolution::Ambiguous(mex(
                "AmbiguousBuiltinImport",
                "ambiguous builtin via imports",
            )),
            1,
            &mut stack,
            ExceptionRouteContext {
                try_stack: &mut try_stack,
                vars: &mut vars,
                last_exception: &mut last_exception,
                pc: &mut pc,
            },
        );
        let err = match outcome {
            Err(err) => err,
            Ok(_) => panic!("ambiguous imported builtin must surface explicit identifier"),
        };
        assert_eq!(err.identifier(), Some("RunMat:AmbiguousBuiltinImport"));
        assert!(err.message().contains("ambiguous builtin via imports"));
    }
}

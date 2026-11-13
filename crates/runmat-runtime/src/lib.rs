#![allow(
    clippy::await_holding_lock,
    clippy::enum_variant_names,
    clippy::get_first,
    clippy::io_other_error,
    clippy::needless_range_loop,
    clippy::redundant_closure,
    clippy::result_large_err,
    clippy::too_many_arguments,
    clippy::useless_conversion
)]
#![cfg_attr(target_arch = "wasm32", allow(dead_code))]

// The web runtime fixtures require browser services, including WebGPU.
#[cfg(all(test, target_arch = "wasm32", feature = "plot-web"))]
wasm_bindgen_test::wasm_bindgen_test_configure!(run_in_browser);

use runmat_types::MemberAccess;

use runmat_builtins::BuiltinErrorDescriptor;
use runmat_value::Value;
use std::future::Future;
use std::pin::Pin;
use std::task::{Context, Poll};

pub mod aggregate;
pub mod analysis;
pub mod builtin;
pub mod dispatcher;
pub mod geometry;
pub mod meshing;
pub mod operations;

pub mod callsite;
pub mod class_registry;
pub mod compatibility;
pub mod condition;
pub mod console;
pub mod context;
pub mod coverage;
pub mod data;
pub mod execution;
pub mod foreign;
pub mod http;
pub mod indexing;
pub mod interaction;
pub mod interrupt;
pub mod iteration;
pub mod native;
pub mod numeric_region;
pub mod object;
pub mod output_context;
pub mod output_count;
pub mod parallel;
pub mod source_context;
pub mod tcp;
pub mod testing;
pub mod third_party_notices;
pub mod value_fact;

pub mod builtins;
pub mod call;
pub mod comparison;
pub mod plotting_hooks;
pub mod replay;
pub mod runtime_error;
pub mod sequence;
pub mod unary_arithmetic;
pub mod user_functions;
pub mod warning_store;
pub(crate) mod warnings;
pub mod workspace;

#[cfg(test)]
mod architecture_tests;

/// Standard result type for runtime builtins.
pub type BuiltinResult<T> = Result<T, RuntimeError>;

pub const OBJECT_INDEX_PAREN: &str = "()";
pub const OBJECT_INDEX_BRACE: &str = "{}";
pub const OBJECT_INDEX_MEMBER: &str = ".";
pub const CALL_METHOD_BUILTIN_NAME: &str = "call_method";
pub const CALL_BOUND_METHOD_BUILTIN_NAME: &str = "__runmat_call_bound_method__";
pub const OBJECT_SUBSREF_METHOD: runmat_types::StaticMethodName =
    runmat_types::StaticMethodName::new("subsref");
pub const OBJECT_SUBSASGN_METHOD: runmat_types::StaticMethodName =
    runmat_types::StaticMethodName::new("subsasgn");
pub const OBJECT_NUM_ARGUMENTS_FROM_SUBSCRIPT_METHOD: runmat_types::StaticMethodName =
    runmat_types::StaticMethodName::new("numArgumentsFromSubscript");
pub const OBJECT_END_METHOD: runmat_types::StaticMethodName =
    runmat_types::StaticMethodName::new("end");
pub(crate) const IDENT_UNDEFINED_FUNCTION: &str = "RunMat:UndefinedFunction";
pub(crate) const HANDLE_VALID_FLAG_PROPERTY: &str = "__runmat_handle_valid__";

fn object_handle_flag_valid(obj: &runmat_value::ObjectInstance) -> bool {
    !matches!(
        obj.property(HANDLE_VALID_FLAG_PROPERTY),
        Some(Value::Bool(false))
    )
}

pub(crate) fn is_handle_valid(handle: &runmat_value::HandleRef) -> bool {
    if !handle.valid {
        return false;
    }
    is_handle_target_valid(handle)
}

pub(crate) fn is_handle_target_valid(handle: &runmat_value::HandleRef) -> bool {
    let target = runmat_gc::gc_with_value(&handle.target, |target| match target {
        Value::Object(obj) => Some((
            object_handle_flag_valid(obj),
            obj.class_name() == handle.class_name.as_identity(),
            obj.graphics_receiver().cloned(),
        )),
        _ => None,
    })
    .ok()
    .flatten();
    match target {
        Some((true, true, Some(receiver))) => {
            let admitted_class = match receiver.kind() {
                runmat_value::GraphicsReceiverKind::Figure { .. } => {
                    handle.class_name == runmat_types::standard::FIGURE
                        || runmat_types::standard::is_graphics_property_control(&handle.class_name)
                }
                runmat_value::GraphicsReceiverKind::Axes { .. } => {
                    handle.class_name == runmat_types::standard::AXES
                }
                runmat_value::GraphicsReceiverKind::PlotChild { .. } => {
                    plotting_hooks::graphics_receiver_class(&receiver)
                        .is_some_and(|class| handle.class_name == class)
                }
            };
            admitted_class && plotting_hooks::graphics_receiver_valid(&receiver)
        }
        Some((valid, exact_class, None)) => valid && exact_class,
        Some((_, _, Some(_))) => false,
        None => false,
    }
}

pub(crate) fn set_handle_valid(handle: &runmat_value::HandleRef, valid: bool) -> bool {
    let runtime = crate::context::legacy::active();
    crate::object::handle_validity::set(runtime.as_ref(), handle, valid).unwrap_or(false)
}

pub fn object_property_getter_name(field: &str) -> String {
    format!("get.{field}")
}

pub fn object_property_setter_name(field: &str) -> String {
    format!("set.{field}")
}

pub(crate) fn current_requested_outputs() -> usize {
    crate::output_count::current_output_count().unwrap_or(1)
}

thread_local! {
    static CONSTRUCTOR_RECEIVER_STACK: std::cell::RefCell<Vec<Value>> =
        const { std::cell::RefCell::new(Vec::new()) };
}

struct ConstructorReceiverPollGuard {
    context: Option<std::rc::Rc<crate::context::RuntimeContextState>>,
}

impl Drop for ConstructorReceiverPollGuard {
    fn drop(&mut self) {
        if let Some(context) = &self.context {
            context.constructor_receivers.borrow_mut().pop();
        } else {
            CONSTRUCTOR_RECEIVER_STACK.with(|stack| {
                stack.borrow_mut().pop();
            });
        }
    }
}

fn push_constructor_receiver_for_poll(receiver: Value) -> ConstructorReceiverPollGuard {
    let context =
        crate::context::legacy::active().map(|context| std::rc::Rc::clone(context.state()));
    if let Some(context) = &context {
        context.constructor_receivers.borrow_mut().push(receiver);
    } else {
        CONSTRUCTOR_RECEIVER_STACK.with(|stack| {
            stack.borrow_mut().push(receiver);
        });
    }
    ConstructorReceiverPollGuard { context }
}

pub(crate) struct ConstructorReceiverFuture<Fut> {
    receiver: Value,
    future: Pin<Box<Fut>>,
}

impl<Fut: Future> Future for ConstructorReceiverFuture<Fut> {
    type Output = Fut::Output;

    fn poll(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        let _guard = push_constructor_receiver_for_poll(self.receiver.clone());
        self.future.as_mut().poll(cx)
    }
}

pub(crate) fn with_constructor_receiver<Fut>(
    receiver: Value,
    future: Fut,
) -> ConstructorReceiverFuture<Fut>
where
    Fut: Future,
{
    ConstructorReceiverFuture {
        receiver,
        future: Box::pin(future),
    }
}

fn constructor_receiver_class_name(receiver: &Value) -> Option<&runmat_types::ClassIdentity> {
    match receiver {
        Value::Object(obj) => Some(obj.class_name()),
        Value::HandleObject(handle) => Some(handle.class_name.as_identity()),
        _ => None,
    }
}

pub(crate) fn constructor_receiver_matches_class(
    receiver: &Value,
    class_name: &runmat_types::ClassIdentity,
) -> bool {
    constructor_receiver_class_name(receiver).is_some_and(|receiver_class| {
        receiver_class == class_name
            || crate::class_registry::is_class_or_subclass(receiver_class, class_name)
    })
}

fn find_constructor_receiver(
    stack: &[Value],
    class_name: &runmat_types::ClassIdentity,
) -> Option<Value> {
    stack
        .iter()
        .rev()
        .find(|receiver| constructor_receiver_matches_class(receiver, class_name))
        .cloned()
}

/// Capture a constructor's exact poll-selected receiver before its executable
/// can become an independently polled child. A different session's selected
/// context cannot supply this invocation's receiver.
pub(crate) fn constructor_receiver_in_context(
    context: &crate::context::RuntimeContext,
    class_name: &runmat_types::ClassIdentity,
) -> Option<Value> {
    if let Some(selected) = crate::context::legacy::active() {
        return std::rc::Rc::ptr_eq(selected.state(), context.state())
            .then(|| {
                find_constructor_receiver(
                    &context.state().constructor_receivers.borrow(),
                    class_name,
                )
            })
            .flatten();
    }
    CONSTRUCTOR_RECEIVER_STACK.with(|stack| find_constructor_receiver(&stack.borrow(), class_name))
}

fn active_constructor_receiver_for(class_name: &runmat_types::ClassIdentity) -> Option<Value> {
    if let Some(context) = crate::context::legacy::active() {
        return find_constructor_receiver(
            &context.state().constructor_receivers.borrow(),
            class_name,
        )
        .or_else(|| {
            context
                .captured_constructor_receiver()
                .filter(|receiver| constructor_receiver_matches_class(receiver, class_name))
                .cloned()
        });
    }
    CONSTRUCTOR_RECEIVER_STACK.with(|stack| find_constructor_receiver(&stack.borrow(), class_name))
}

fn undefined_callable_error(identity: &runmat_types::CallableIdentity) -> RuntimeError {
    let detail = crate::call::identity::undefined_callable_message(identity);
    build_runtime_error(detail)
        .with_identifier(IDENT_UNDEFINED_FUNCTION)
        .build()
}

#[cfg(test)]
fn build_shape_checked_cell(
    values: Vec<Value>,
    rows: usize,
    cols: usize,
    context: &str,
) -> Result<runmat_value::CellArray, RuntimeError> {
    runmat_value::CellArray::new(values, rows, cols).map_err(|err| {
        build_runtime_error(format!("{context}: {err}"))
            .with_identifier("RunMat:ShapeMismatch")
            .build()
    })
}

pub(crate) fn runtime_descriptor_error(
    builtin: &'static str,
    error: &'static BuiltinErrorDescriptor,
) -> RuntimeError {
    runtime_descriptor_error_with_message(builtin, error.message, error)
}

pub(crate) fn runtime_descriptor_error_with_detail(
    builtin: &'static str,
    error: &'static BuiltinErrorDescriptor,
    detail: impl AsRef<str>,
) -> RuntimeError {
    runtime_descriptor_error_with_message(
        builtin,
        format!("{}: {}", error.message, detail.as_ref()),
        error,
    )
}

fn runtime_descriptor_error_with_message(
    builtin: &'static str,
    message: impl Into<String>,
    error: &'static BuiltinErrorDescriptor,
) -> RuntimeError {
    let mut builder = build_runtime_error(message).with_builtin(builtin);
    if let Some(identifier) = error.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

pub(crate) fn object_receiver_class_name(receiver: &Value) -> Option<runmat_types::ClassIdentity> {
    match receiver {
        Value::Object(obj) => Some(obj.class_name().clone()),
        Value::HandleObject(handle) => {
            let class_name = runmat_gc::gc_with_value(&handle.target, |target| match target {
                Value::Object(obj) => obj.class_name().clone(),
                _ => handle.class_name.as_identity().clone(),
            })
            .unwrap_or_else(|_| handle.class_name.as_identity().clone());
            Some(class_name)
        }
        Value::ObjectArray(array) => Some(array.class_name().clone()),
        _ => None,
    }
}

fn class_member_identity(
    class_name: &runmat_types::ClassIdentity,
    member: &str,
) -> runmat_types::CallableIdentity {
    let mut segments = class_name.qualified_name().0;
    segments.push(runmat_types::SymbolName(member.to_string()));
    runmat_types::CallableIdentity::ExternalName(runmat_types::QualifiedName(segments))
}

pub(crate) fn qualified_name_segments(name: &str) -> Vec<runmat_types::SymbolName> {
    name.split('.')
        .map(|segment| runmat_types::SymbolName(segment.to_string()))
        .collect()
}

pub(crate) fn is_well_formed_qualified_name(name: &str) -> bool {
    let segments = qualified_name_segments(name);
    segments.len() > 1 && segments.iter().all(|segment| !segment.0.is_empty())
}

pub(crate) fn callable_identity_for_handle_name(
    name: &str,
) -> (
    runmat_types::CallableIdentity,
    runmat_types::CallableFallbackPolicy,
) {
    if is_well_formed_qualified_name(name) {
        let segments = qualified_name_segments(name);
        (
            runmat_types::CallableIdentity::ExternalName(runmat_types::QualifiedName(segments)),
            runmat_types::CallableFallbackPolicy::ExternalBoundary,
        )
    } else {
        (
            runmat_types::CallableIdentity::DynamicName(runmat_types::SymbolName(name.to_string())),
            runmat_types::CallableFallbackPolicy::RuntimeNameResolution,
        )
    }
}

pub(crate) fn external_callable_identity_for_name(name: &str) -> runmat_types::CallableIdentity {
    if !is_well_formed_qualified_name(name) {
        runmat_types::CallableIdentity::ExternalName(runmat_types::QualifiedName(vec![
            runmat_types::SymbolName(name.to_string()),
        ]))
    } else {
        let segments = qualified_name_segments(name);
        runmat_types::CallableIdentity::ExternalName(runmat_types::QualifiedName(segments))
    }
}

pub(crate) async fn dispatch_object_external_member(
    class_name: runmat_types::ClassIdentity,
    member: &str,
    args: Vec<Value>,
    requested_outputs: usize,
) -> BuiltinResult<crate::execution::RootedValue> {
    dispatch_object_external_member_in_context(None, class_name, member, args, requested_outputs)
        .await
}

pub(crate) async fn dispatch_object_external_member_in_context(
    runtime: Option<&crate::context::RuntimeContext>,
    class_name: runmat_types::ClassIdentity,
    member: &str,
    args: Vec<Value>,
    requested_outputs: usize,
) -> BuiltinResult<crate::execution::RootedValue> {
    dispatch_callable_with_policy_in_context(
        runtime,
        class_member_identity(&class_name, member),
        runmat_types::CallableFallbackPolicy::ExternalBoundary,
        args,
        requested_outputs,
    )
    .await
}

fn require_single_sequence(sequence: runmat_value::ValueSequence) -> BuiltinResult<Value> {
    use crate::sequence::ResolveValueSequence;

    let values = sequence.resolve(
        runmat_types::SequenceUse::RequireSingle,
        crate::sequence::SequenceResolutionContext::default(),
    )?;
    Ok(crate::sequence::into_first_output(values)?.expect("one required output was resolved"))
}

#[cfg(test)]
pub(crate) async fn dispatch_callable_with_policy(
    identity: runmat_types::CallableIdentity,
    fallback_policy: runmat_types::CallableFallbackPolicy,
    args: Vec<Value>,
    requested_outputs: usize,
) -> BuiltinResult<crate::execution::RootedValue> {
    dispatch_callable_with_policy_in_context(
        None,
        identity,
        fallback_policy,
        args,
        requested_outputs,
    )
    .await
}

pub(crate) async fn dispatch_callable_with_policy_in_context(
    runtime: Option<&crate::context::RuntimeContext>,
    identity: runmat_types::CallableIdentity,
    fallback_policy: runmat_types::CallableFallbackPolicy,
    args: Vec<Value>,
    requested_outputs: usize,
) -> BuiltinResult<crate::execution::RootedValue> {
    dispatch_callable_with_policy_in_runtime(
        runtime,
        identity,
        fallback_policy,
        args,
        requested_outputs,
        None,
    )
    .await
    .and_then(crate::call::arguments::project_rooted_legacy_builtin_value_abi)
}

async fn dispatch_callable_with_policy_in_runtime(
    runtime: Option<&crate::context::RuntimeContext>,
    identity: runmat_types::CallableIdentity,
    fallback_policy: runmat_types::CallableFallbackPolicy,
    args: Vec<Value>,
    requested_outputs: usize,
    input_callsite: Option<crate::callsite::CallsiteInfo>,
) -> BuiltinResult<crate::execution::RootedValueSequence> {
    let args = match runtime {
        Some(runtime) => crate::execution::RootedValues::in_context(args, runtime)?,
        None => {
            crate::execution::RootedValues::new(args).map_err(|error| error.into_runtime_error())?
        }
    };
    dispatch_retained_callable_with_policy_in_runtime(
        runtime,
        identity,
        fallback_policy,
        args.into(),
        requested_outputs,
        input_callsite,
    )
    .await
}

pub(crate) async fn dispatch_retained_callable_with_policy_in_context(
    runtime: Option<&crate::context::RuntimeContext>,
    identity: runmat_types::CallableIdentity,
    fallback_policy: runmat_types::CallableFallbackPolicy,
    args: crate::call::BuiltinArguments,
    requested_outputs: usize,
) -> BuiltinResult<crate::execution::RootedValue> {
    dispatch_retained_callable_with_policy_in_runtime(
        runtime,
        identity,
        fallback_policy,
        args,
        requested_outputs,
        None,
    )
    .await
    .and_then(crate::call::arguments::project_rooted_legacy_builtin_value_abi)
}

pub(crate) async fn dispatch_retained_callable_with_policy_in_runtime(
    runtime: Option<&crate::context::RuntimeContext>,
    identity: runmat_types::CallableIdentity,
    fallback_policy: runmat_types::CallableFallbackPolicy,
    args: crate::call::BuiltinArguments,
    requested_outputs: usize,
    input_callsite: Option<crate::callsite::CallsiteInfo>,
) -> BuiltinResult<crate::execution::RootedValueSequence> {
    dispatch_prepared_callable_with_policy_in_runtime(
        runtime,
        identity,
        fallback_policy,
        args.into_supplied_function_arguments(runtime)?,
        requested_outputs,
        input_callsite,
    )
    .await
}

/// Internal continuation for the selected callee. Unlike the new-callee
/// adapter above, this must preserve its actual capture and routing roles.
pub(crate) async fn dispatch_prepared_callable_with_policy_in_runtime(
    runtime: Option<&crate::context::RuntimeContext>,
    identity: runmat_types::CallableIdentity,
    fallback_policy: runmat_types::CallableFallbackPolicy,
    args: crate::call::RootedFunctionArguments,
    requested_outputs: usize,
    input_callsite: Option<crate::callsite::CallsiteInfo>,
) -> BuiltinResult<crate::execution::RootedValueSequence> {
    let request = crate::user_functions::CallableRequest::rooted_resolved(
        identity.clone(),
        fallback_policy,
        args,
        requested_outputs,
    )
    .with_selected_runtime(runtime)
    .with_function_input_callsite(input_callsite.clone());
    let args = match crate::user_functions::try_call_semantic_descriptor_owned(request).await {
        std::ops::ControlFlow::Break(result) => return result,
        std::ops::ControlFlow::Continue(args) => crate::call::BuiltinArguments::new(args),
    };

    // An exact builtin identity is an executable target, not VM name fallback.
    // Keep its retained arguments on the same dispatcher used by direct calls.
    let name = match &identity {
        runmat_types::CallableIdentity::Builtin(id) => Some(id.0.clone()),
        _ => fallback_policy.vm_fallback_name_for(&identity),
    };
    if let Some(name) = name {
        return crate::callsite::with_callsite(
            input_callsite,
            crate::dispatcher::try_call_retained_builtin_sequence_with_outputs(
                &name,
                args,
                requested_outputs,
                runtime,
            ),
        )
        .await?
        .and_then(|result| result.ok_or_else(|| undefined_callable_error(&identity)));
    }

    Err(undefined_callable_error(&identity))
}

/// Legacy embedding result ABI; typed executors use the sequence entrypoints.
pub fn call_feval_async_with_outputs(
    func_value: Value,
    args: &[Value],
    requested_outputs: usize,
) -> impl std::future::Future<Output = Result<crate::execution::RootedValue, RuntimeError>> + '_ {
    let sequence = call_feval_sequence_async_with_outputs(func_value, args, requested_outputs);
    async move {
        sequence
            .await
            .and_then(crate::call::arguments::project_rooted_legacy_builtin_value_abi)
    }
}

pub fn call_feval_async_with_outputs_in_context<'a>(
    runtime: &'a crate::context::RuntimeContext,
    func_value: Value,
    args: &'a [Value],
    requested_outputs: usize,
) -> impl std::future::Future<Output = Result<crate::execution::RootedValue, RuntimeError>> + 'a {
    let sequence = call_feval_sequence_async_with_outputs_in_context(
        runtime,
        func_value,
        args,
        requested_outputs,
    );
    async move {
        sequence
            .await
            .and_then(crate::call::arguments::project_rooted_legacy_builtin_value_abi)
    }
}

pub fn call_feval_async_with_outputs_and_callsite_in_context<'a>(
    runtime: &'a crate::context::RuntimeContext,
    func_value: Value,
    args: &'a [Value],
    requested_outputs: usize,
    input_callsite: Option<crate::callsite::CallsiteInfo>,
) -> impl std::future::Future<Output = Result<crate::execution::RootedValue, RuntimeError>> + 'a {
    let sequence = call_feval_sequence_async_with_outputs_and_callsite_in_context(
        runtime,
        func_value,
        args,
        requested_outputs,
        input_callsite,
    );
    async move {
        sequence
            .await
            .and_then(crate::call::arguments::project_rooted_legacy_builtin_value_abi)
    }
}

/// Direct embedding callback call; no session binding authority is inferred
/// from an ambient poll scope.
pub fn call_feval_sequence_async_with_outputs(
    func_value: Value,
    args: &[Value],
    requested_outputs: usize,
) -> impl std::future::Future<Output = Result<crate::execution::RootedValueSequence, RuntimeError>> + '_
{
    let input_roots = runmat_value::validate_output_count(requested_outputs)
        .map_err(crate::sequence::sequence_error_to_runtime)
        .and_then(|_| crate::call::feval::admit_inputs(&func_value, args));
    async move {
        let _input_roots = input_roots?;
        let _guard = crate::output_count::push_output_count(Some(requested_outputs))?;
        Box::pin(crate::call::feval::invoke(
            None,
            Some(requested_outputs),
            func_value,
            args.to_vec(),
            None,
        ))
        .await
    }
}

pub fn call_feval_sequence_async_with_outputs_in_context<'a>(
    runtime: &'a crate::context::RuntimeContext,
    func_value: Value,
    args: &'a [Value],
    requested_outputs: usize,
) -> impl std::future::Future<Output = Result<crate::execution::RootedValueSequence, RuntimeError>> + 'a
{
    call_feval_sequence_async_with_outputs_and_callsite_in_context(
        runtime,
        func_value,
        args,
        requested_outputs,
        None,
    )
}

/// Invoke a callback without replacing the caller's admitted argument storage.
/// The target registration is retained when this future is constructed.
pub fn call_feval_retained_sequence_async_with_outputs_in_context<'a>(
    runtime: &'a crate::context::RuntimeContext,
    func_value: Value,
    args: crate::call::BuiltinArguments,
    requested_outputs: usize,
) -> impl std::future::Future<Output = Result<crate::execution::RootedValueSequence, RuntimeError>> + 'a
{
    call_feval_retained_sequence_async_with_outputs_and_callsite_in_context(
        runtime,
        func_value,
        args,
        requested_outputs,
        None,
    )
}

pub fn call_feval_retained_sequence_async_with_outputs_and_callsite_in_context<'a>(
    runtime: &'a crate::context::RuntimeContext,
    func_value: Value,
    args: crate::call::BuiltinArguments,
    requested_outputs: usize,
    input_callsite: Option<crate::callsite::CallsiteInfo>,
) -> impl std::future::Future<Output = Result<crate::execution::RootedValueSequence, RuntimeError>> + 'a
{
    let invocation = runmat_value::validate_output_count(requested_outputs)
        .map_err(crate::sequence::sequence_error_to_runtime)
        .map(|()| {
            crate::call::feval::invoke_retained(
                Some(runtime),
                func_value,
                args,
                requested_outputs,
                input_callsite,
            )
        });
    async move {
        let invocation = invocation?;
        runtime
            .scope(async {
                let _guard = crate::output_count::push_output_count(Some(requested_outputs))?;
                invocation.await
            })
            .await
    }
}

pub fn call_feval_sequence_async_with_outputs_and_callsite_in_context<'a>(
    runtime: &'a crate::context::RuntimeContext,
    func_value: Value,
    args: &'a [Value],
    requested_outputs: usize,
    input_callsite: Option<crate::callsite::CallsiteInfo>,
) -> impl std::future::Future<Output = Result<crate::execution::RootedValueSequence, RuntimeError>> + 'a
{
    let input_roots = runmat_value::validate_output_count(requested_outputs)
        .map_err(crate::sequence::sequence_error_to_runtime)
        .and_then(|_| crate::call::feval::admit_inputs(&func_value, args));
    async move {
        let _input_roots = input_roots?;
        runtime
            .scope(async {
                let _guard = crate::output_count::push_output_count(Some(requested_outputs))?;
                crate::call::feval::invoke(
                    Some(runtime),
                    Some(requested_outputs),
                    func_value,
                    args.to_vec(),
                    input_callsite,
                )
                .await
            })
            .await
    }
}

pub use runtime_error::{
    build_runtime_error, replay_error, replay_error_with_source, CallFrame,
    CallFrameDiagnosticSource, CallFrameSourceOwner, ErrorContext, GpuGatherRetry, ReplayErrorKind,
    RuntimeError, RuntimeErrorBuilder,
};

pub mod debug_context;

#[cfg(feature = "blas-lapack")]
pub mod blas;
#[cfg(feature = "blas-lapack")]
pub mod lapack;

// Link to Apple's Accelerate framework on macOS
#[cfg(all(feature = "blas-lapack", target_os = "macos"))]
#[link(name = "Accelerate", kind = "framework")]
extern "C" {}

// Ensure OpenBLAS is linked on non-macOS platforms when BLAS/LAPACK is enabled
#[cfg(all(feature = "blas-lapack", not(target_os = "macos")))]
extern crate openblas_src;

pub use dispatcher::{
    call_builtin, call_builtin_async, call_builtin_async_in_context,
    call_builtin_async_with_outputs, call_builtin_async_with_outputs_in_context,
    call_builtin_sequence_async, call_builtin_sequence_async_in_context,
    call_builtin_sequence_async_with_outputs, call_builtin_sequence_async_with_outputs_in_context,
    class_access_context, gather_if_needed, gather_if_needed_async, is_gpu_value,
    push_class_access_context, value_contains_gpu,
};

#[cfg(feature = "plot-core")]
pub use builtins::plotting::{
    export_figure_scene as runtime_plot_export_figure_scene,
    import_figure_scene_async as runtime_plot_import_figure_scene_async,
    import_figure_scene_from_path_async as runtime_plot_import_figure_scene_from_path_async,
};
pub use replay::{
    runtime_export_workspace_state, runtime_import_workspace_state, WorkspaceReplayMode,
};

pub use runmat_macros::{register_fusion_spec, register_gpu_spec};

// Pruned legacy re-exports; prefer builtins::* and explicit shims only
// Transitional root-level shims for widely used helpers
pub use builtins::common::concatenation::{
    create_host_matrix_from_values, create_matrix_from_values, host_matrix_can_run_inline,
};
pub use builtins::common::elementwise::{
    elementwise_div, elementwise_mul, elementwise_neg, elementwise_pow, power,
};
pub use builtins::common::indexing::perform_indexing;
pub use builtins::common::matrix::{value_matmul, value_matmul_in_context};
// Explicitly re-export for external users of the VM that build matrices from values
// (kept above)
// Note: constants and mathematics modules only contain #[runtime_builtin] functions
// and don't export public items, so they don't need to be re-exported

#[cfg(feature = "blas-lapack")]
pub use blas::*;
#[cfg(feature = "blas-lapack")]
pub use lapack::*;

pub fn make_cell_with_shape(values: Vec<Value>, shape: Vec<usize>) -> Result<Value, String> {
    let ca = runmat_value::CellArray::new_with_shape(values, shape)
        .map_err(|e| format!("Cell creation error: {e}"))?;
    Ok(Value::Cell(ca))
}

pub(crate) fn make_cell(values: Vec<Value>, rows: usize, cols: usize) -> Result<Value, String> {
    make_cell_with_shape(values, vec![rows, cols])
}

// Object/handle utilities used by interpreter lowering for OOP/func handles

pub(crate) async fn rethrow_builtin(e: Value) -> crate::BuiltinResult<Value> {
    match e {
        Value::MException(me) => Err(runtime_error::error_from_exception(me)),
        Value::String(s) => Err(build_runtime_error(s.display().to_string()).build()),
        other => Err(build_runtime_error(format!("RunMat:error: {other:?}")).build()),
    }
}

// -------- Handle classes & events --------

pub(crate) async fn new_handle_object_builtin(class_name: String) -> crate::BuiltinResult<Value> {
    // Create an underlying object instance and wrap it in a handle
    let obj = create_class_object(class_name.clone()).await?;
    if matches!(obj, Value::HandleObject(_)) {
        return Ok(obj);
    }
    let gc = runmat_gc::gc_allocate(obj).map_err(|e| format!("gc: {e}"))?;
    Ok(Value::HandleObject(runmat_value::HandleRef {
        class_name: class_name.into(),
        target: gc,
        valid: true,
    }))
}

pub(crate) async fn isvalid_builtin(v: Value) -> crate::BuiltinResult<Value> {
    match v {
        Value::HandleObject(h) => Ok(Value::Bool(crate::is_handle_valid(&h))),
        Value::Listener(l) => Ok(Value::Bool(l.valid && l.enabled)),
        _ => Ok(Value::Bool(false)),
    }
}

mod event_runtime;
#[cfg(test)]
use event_runtime::reset_event_registry_for_test;
pub(crate) use event_runtime::{
    addlistener_builtin, invalidate_listener_registration, notify_builtin, EventRegistry,
};

pub(crate) fn canonicalize_callback_handle_for_semantic_resolution(
    callback: Value,
) -> Result<Value, RuntimeError> {
    fn normalize_handle_name(text: &str) -> Option<String> {
        let trimmed = text.trim();
        let name = trimmed.strip_prefix('@').unwrap_or(trimmed).trim();
        (!name.is_empty()).then(|| name.to_string())
    }

    fn resolve_text_handle(text: &str) -> Option<Value> {
        let name = normalize_handle_name(text)?;
        let function = crate::user_functions::resolve_semantic_function_by_name(&name)?;
        Some(crate::user_functions::bound_function_handle(name, function))
    }

    Ok(match callback {
        Value::String(text) => text
            .as_str()
            .and_then(resolve_text_handle)
            .unwrap_or_else(|| {
                crate::builtins::introspection::function_handle_text::dispatch_str2func(
                    Value::String(text.clone()),
                )
                .unwrap_or(Value::String(text))
            }),
        Value::StringArray(array) if array.data().len() == 1 => {
            let text = &array.data()[0];
            text.as_str()
                .and_then(resolve_text_handle)
                .unwrap_or_else(|| {
                    crate::builtins::introspection::function_handle_text::dispatch_str2func(
                        Value::StringArray(array.clone()),
                    )
                    .unwrap_or(Value::StringArray(array))
                })
        }
        Value::CharArray(chars) if chars.rows == 1 => {
            let text = chars.try_row_string().map_err(|source| {
                crate::runtime_error::semantic_error_with_source(
                    "InvalidCallback",
                    "callback name contains malformed UTF-16",
                    source,
                )
            })?;
            text.and_then(|text| resolve_text_handle(&text))
                .unwrap_or_else(|| {
                    crate::builtins::introspection::function_handle_text::dispatch_str2func(
                        Value::CharArray(chars.clone()),
                    )
                    .unwrap_or(Value::CharArray(chars))
                })
        }
        Value::ScopedFunctionHandle(handle) => Value::ScopedFunctionHandle(handle),
        Value::FunctionHandle(name) => {
            if let Some(function) = crate::user_functions::resolve_semantic_function_by_name(&name)
            {
                crate::user_functions::bound_function_handle(name.into_name_unbounded(), function)
            } else {
                Value::FunctionHandle(name)
            }
        }
        Value::ExternalFunctionHandle(name) => {
            if is_well_formed_qualified_name(&name) {
                if let Some(function) =
                    crate::user_functions::resolve_semantic_function_by_name(&name)
                {
                    return Ok(crate::user_functions::bound_function_handle(
                        name.into_name_unbounded(),
                        function,
                    ));
                }
            }
            Value::ExternalFunctionHandle(name)
        }
        Value::MethodFunctionHandle(name) => {
            if let Some(function) = crate::user_functions::resolve_semantic_function_by_name(&name)
            {
                crate::user_functions::bound_function_handle(name.into_name_unbounded(), function)
            } else {
                Value::MethodFunctionHandle(name)
            }
        }
        Value::Closure(closure) => {
            if closure.binding.is_none() {
                if let Some(function) =
                    crate::user_functions::resolve_semantic_function_by_name(&closure.function_name)
                {
                    return crate::call::closure_binding::bind_selected_closure(closure, function)
                        .map(Value::Closure);
                }
            }
            Value::Closure(closure)
        }
        other => other,
    })
}

pub(crate) async fn make_anon_builtin(params: String, body: String) -> crate::BuiltinResult<Value> {
    Ok(Value::string(format!("@anon({params}) {body}")))
}

pub async fn create_class_object(class_name: String) -> crate::BuiltinResult<Value> {
    let class_identity = runmat_types::ClassIdentity::new(class_name.clone()).map_err(|error| {
        build_runtime_error(format!("Invalid class name '{class_name}': {error}"))
            .with_identifier("RunMat:InvalidClassName")
            .build()
    })?;
    create_class_object_by_identity(class_identity).await
}

async fn create_class_object_by_identity(
    class_identity: runmat_types::ClassIdentity,
) -> crate::BuiltinResult<Value> {
    if crate::class_registry::is_class_abstract(&class_identity) {
        return Err(build_runtime_error(format!(
            "Cannot instantiate abstract class '{}'.",
            class_identity
        ))
        .with_identifier("RunMat:AbstractMethodMissing")
        .build());
    }
    if let Some(def) = crate::class_registry::get_class(&class_identity) {
        // Collect class hierarchy from root to leaf for default initialization
        let mut chain: Vec<crate::class_registry::RuntimeClass> = Vec::new();
        let mut is_handle_class = false;
        let mut visited = std::collections::HashSet::new();
        // Walk up to root
        let mut cursor = Some(def.name.clone());
        while let Some(name) = cursor {
            if name.is(runmat_types::standard::HANDLE) {
                is_handle_class = true;
                break;
            }
            if !visited.insert(name.clone()) {
                break;
            }
            if let Some(cd) = crate::class_registry::get_class(&name) {
                if cd
                    .parent
                    .as_ref()
                    .is_some_and(|parent| parent.is(runmat_types::standard::HANDLE))
                {
                    is_handle_class = true;
                }
                chain.push(cd.clone());
                cursor = cd.parent.clone();
            } else {
                break;
            }
        }
        // Reverse to root-first
        chain.reverse();
        let runtime = crate::context::legacy::active();
        let selected = runtime
            .as_ref()
            .and_then(|context| context.selected_object_class().cloned())
            .filter(|selected| selected.class() == &def.name);
        if let Some(selected) = &selected {
            if selected.is_handle_class() != is_handle_class {
                return Err(build_runtime_error(
                    "selected object handle kind differs from its class ancestry",
                )
                .with_identifier("RunMat:ConstructorOwnerInvalid")
                .build());
            }
        }
        // Registry defaults remain rooted by the registry for this complete
        // construction. The empty fallback is one shared canonical Value.
        let empty_default =
            Value::Tensor(runmat_value::Tensor::new(vec![], vec![0, 0]).expect("empty tensor"));
        let defaults = || {
            chain
                .iter()
                .flat_map(|class| class.properties.iter())
                .filter_map(|(name, property)| {
                    (!property.is_static).then_some((
                        name.0.as_str(),
                        property.default_value.as_ref().unwrap_or(&empty_default),
                    ))
                })
        };
        let obj = if let Some(runtime) = &runtime {
            crate::execution::object::construct_registered_object_in_context(
                runtime,
                def.name.display_name(),
                defaults(),
                selected,
            )?
            .with_owned(|value| match value {
                Value::Object(object) => object,
                _ => unreachable!("registered object construction returns an object"),
            })
        } else {
            let mut builder = runmat_value::UnboundedObjectBuilder::new(def.name.clone());
            for (name, value) in defaults() {
                builder.insert(name.to_owned(), value.clone());
            }
            builder.finish()
        };
        if is_handle_class {
            let gc = runmat_gc::gc_allocate(Value::Object(obj)).map_err(|e| format!("gc: {e}"))?;
            Ok(Value::HandleObject(runmat_value::HandleRef {
                class_name: def.name.clone().into(),
                target: gc,
                valid: true,
            }))
        } else {
            Ok(Value::Object(obj))
        }
    } else if let Some(runtime) = crate::context::legacy::active() {
        crate::execution::object::construct_registered_object_in_context(
            &runtime,
            class_identity.display_name(),
            std::iter::empty(),
            None,
        )
        .map(|rooted| rooted.with_owned(|value| value))
    } else {
        Ok(Value::Object(runmat_value::ObjectInstance::new(
            class_identity,
        )))
    }
}

pub async fn call_super_constructor(
    runtime: &crate::context::RuntimeContext,
    class_identity: runmat_types::ClassIdentity,
    super_class_identity: runmat_types::ClassIdentity,
    args: Vec<Value>,
) -> crate::BuiltinResult<crate::execution::RootedValue> {
    let _argument_roots = crate::execution::RootedValueLease::for_values(args.iter())
        .map_err(|error| error.into_runtime_error())?;
    let selected = runtime
        .retained_super_constructor(&class_identity, &super_class_identity)?
        .cloned();
    if let Some(selected) = &selected {
        if selected.target.parent() != &super_class_identity {
            return Err(build_runtime_error(
                "retained super constructor disagrees with its admitted owner",
            )
            .with_identifier("RunMat:ConstructorOwnerInvalid")
            .build());
        }
        let accessible = match &selected.target {
            crate::context::RetainedSuperConstructorTarget::Implicit { .. } => true,
            crate::context::RetainedSuperConstructorTarget::Bound { access, .. } => match access {
                runmat_types::MemberAccess::Public => true,
                runmat_types::MemberAccess::Private => false,
                // The admitted caller body and immutable parent declaration
                // prove the lexical subclass for this exact call.
                runmat_types::MemberAccess::Protected => true,
            },
        };
        if !accessible {
            return Err(build_runtime_error("super constructor is not accessible")
                .with_identifier("RunMat:MethodPrivate")
                .build());
        }
    }
    let receiver = if let Some(active) = active_constructor_receiver_for(&class_identity) {
        active
    } else {
        create_class_object_by_identity(class_identity.clone()).await?
    };
    let _receiver_root = crate::execution::RootedValueLease::for_values(std::iter::once(&receiver))
        .map_err(|error| error.into_runtime_error())?;
    let ctor_result = with_constructor_receiver(receiver.clone(), async {
        if let Some(selected) = selected {
            match selected.target {
                crate::context::RetainedSuperConstructorTarget::Implicit { .. } => {
                    return Ok(None);
                }
                crate::context::RetainedSuperConstructorTarget::Bound {
                    binding, owner, ..
                } => {
                    let selected_runtime = runtime
                        .clone()
                        .with_constructor_receiver_for_method(Some(&owner))?
                        .with_retained_super_body(owner.clone(), binding.clone());
                    let result = crate::user_functions::try_call_bound_function_in_context(
                        &selected_runtime,
                        binding,
                        &args,
                        1,
                    )
                    .await
                    .ok_or_else(|| {
                        build_runtime_error("retained super constructor has no exact invoker")
                            .with_identifier("RunMat:BoundFunction:UnavailableInvoker")
                            .build()
                    })??;
                    return result
                        .try_map_owned(require_single_sequence, |error| error.into_runtime_error())
                        .map(Some);
                }
            }
        }
        let ctor_name = super_class_identity
            .display_name()
            .rsplit('.')
            .next()
            .filter(|name| !name.trim().is_empty())
            .unwrap_or(super_class_identity.display_name());
        let ctor_identity = runmat_types::MethodName::from(ctor_name);
        let qualified_ctor_identity =
            runmat_types::MethodName::from(super_class_identity.display_name());
        let ctor_lookup = crate::class_registry::lookup_method(
            &super_class_identity,
            &ctor_identity,
        )
        .or_else(|| {
            crate::class_registry::lookup_method(&super_class_identity, &qualified_ctor_identity)
        });
        let Some((ctor, registered_owner)) = ctor_lookup else {
            return Ok::<Option<crate::execution::RootedValue>, RuntimeError>(None);
        };
        let caller_class = crate::dispatcher::class_access_context();
        let accessible = match ctor.access {
            runmat_types::MemberAccess::Public => true,
            runmat_types::MemberAccess::Private => {
                caller_class.as_deref() == Some(&registered_owner)
            }
            runmat_types::MemberAccess::Protected => {
                caller_class.as_deref().is_some_and(|caller| {
                    crate::class_registry::is_class_or_subclass(caller, &registered_owner)
                })
            }
        };
        if !accessible {
            return Err(build_runtime_error("super constructor is not accessible")
                .with_identifier("RunMat:MethodPrivate")
                .build());
        }
        let Some(result) = crate::user_functions::try_call_semantic_function_by_name_in_context(
            Some(runtime),
            &ctor.function_name,
            &args,
            1,
        )
        .await
        else {
            return Err(build_runtime_error(
                "declared super constructor has no executable definition",
            )
            .with_identifier("RunMat:SuperConstructorUnavailable")
            .build());
        };
        result?
            .try_map_owned(require_single_sequence, |error| error.into_runtime_error())
            .map(Some)
    })
    .await?;
    let Some(ctor_result) = ctor_result else {
        return crate::execution::RootedValue::new(receiver)
            .map_err(|error| error.into_runtime_error());
    };
    fn merge_parent_props_into_object(
        runtime: &crate::context::RuntimeContext,
        receiver_obj: &mut runmat_value::ObjectInstance,
        ctor_result: Value,
        owner: Option<&runmat_gc::GcHandle>,
    ) -> Result<(), RuntimeError> {
        let mut insert = |name: &str, value: Value| -> Result<(), RuntimeError> {
            let displaced = crate::execution::object::replacement::replace_property(
                Some(runtime),
                receiver_obj,
                name,
                value,
                |message, source| {
                    crate::context::host_allocation::host_allocation_runtime_error(
                        Some(runtime),
                        "RunMat:SuperConstructorAllocation",
                        message,
                        source,
                    )
                },
                |message| crate::runtime_error::semantic_error("SuperConstructor", message),
            )?;
            if let Some(owner) = owner {
                let value = receiver_obj
                    .property(name)
                    .expect("published parent property remains present");
                runmat_gc::gc_record_handle_write(owner, value);
            }
            drop(displaced);
            Ok(())
        };
        match ctor_result {
            Value::Object(parent_obj) => {
                for (name, value) in parent_obj.properties().iter() {
                    insert(name, value.clone())?;
                }
            }
            Value::HandleObject(parent_handle) => {
                if let Some(owner) = owner {
                    if parent_handle.target == *owner {
                        if parent_handle.valid {
                            return Ok(());
                        }
                        return Err(build_runtime_error(
                            "super constructor returned invalid parent handle",
                        )
                        .build());
                    }
                }
                if !is_handle_valid(&parent_handle) {
                    return Err(build_runtime_error(
                        "super constructor returned invalid parent handle",
                    )
                    .build());
                }
                match runmat_gc::gc_clone_value(&parent_handle.target).map_err(|e| {
                    build_runtime_error(format!(
                        "super constructor returned stale parent handle: {e}"
                    ))
                    .build()
                })? {
                    Value::Object(parent_obj) => {
                        for (name, value) in parent_obj.properties().iter() {
                            insert(name, value.clone())?;
                        }
                    }
                    _ => {
                        return Err(build_runtime_error(
                            "super constructor returned non-object parent handle",
                        )
                        .build());
                    }
                }
            }
            Value::Struct(parent_fields) => {
                for (name, value) in parent_fields.fields.iter() {
                    insert(name, value.clone())?;
                }
            }
            _ => {}
        }
        Ok(())
    }
    ctor_result.with_owned(|ctor_result| {
        let merged = match receiver {
            Value::Object(mut receiver_obj) => {
                merge_parent_props_into_object(runtime, &mut receiver_obj, ctor_result, None)?;
                Ok::<Value, RuntimeError>(Value::Object(receiver_obj))
            }
            Value::HandleObject(handle) => {
                let merged = runmat_gc::gc_with_value_mut(&handle.target, |target| {
                    if let Value::Object(receiver_obj) = target {
                        if !object_handle_flag_valid(receiver_obj) {
                            return Err(build_runtime_error(
                                "super constructor receiver handle is invalid",
                            )
                            .build());
                        }
                        merge_parent_props_into_object(
                            runtime,
                            receiver_obj,
                            ctor_result,
                            Some(&handle.target),
                        )
                    } else {
                        Err(build_runtime_error(
                            "super constructor receiver target is not an object",
                        )
                        .build())
                    }
                })
                .map_err(|e| {
                    build_runtime_error(format!("super constructor receiver invalid: {e}")).build()
                })?;
                merged?;
                Ok(Value::HandleObject(handle))
            }
            _ => Ok(receiver),
        }?;
        crate::execution::RootedValue::new(merged).map_err(|error| error.into_runtime_error())
    })
}

pub async fn call_super_method(
    class_identity: runmat_types::ClassIdentity,
    super_class_identity: runmat_types::ClassIdentity,
    method_name: runmat_types::MethodName,
    args: Vec<Value>,
) -> crate::BuiltinResult<crate::execution::RootedValue> {
    call_super_method_with_selected_runtime(
        None,
        class_identity,
        super_class_identity,
        method_name,
        args,
    )
    .await
}

pub async fn call_super_method_in_context(
    runtime: &crate::context::RuntimeContext,
    class_identity: runmat_types::ClassIdentity,
    super_class_identity: runmat_types::ClassIdentity,
    method_name: runmat_types::MethodName,
    args: Vec<Value>,
) -> crate::BuiltinResult<crate::execution::RootedValue> {
    call_super_method_with_selected_runtime(
        Some(runtime),
        class_identity,
        super_class_identity,
        method_name,
        args,
    )
    .await
}

async fn call_super_method_with_selected_runtime(
    runtime: Option<&crate::context::RuntimeContext>,
    class_identity: runmat_types::ClassIdentity,
    super_class_identity: runmat_types::ClassIdentity,
    method_name: runmat_types::MethodName,
    args: Vec<Value>,
) -> crate::BuiltinResult<crate::execution::RootedValue> {
    let _argument_roots = crate::execution::RootedValueLease::for_values(args.iter())
        .map_err(|error| error.into_runtime_error())?;
    if let Some(facts) = args
        .first()
        .map(crate::object::protocol::selected_graph_receiver_facts)
        .transpose()?
        .flatten()
    {
        return crate::object::dispatch::call_selected_super_method(
            runtime,
            &class_identity,
            &super_class_identity,
            &method_name,
            args,
            facts,
        )
        .await;
    }
    let Some((method, owner)) =
        crate::class_registry::lookup_method(&super_class_identity, &method_name)
    else {
        return Err(build_runtime_error(format!(
            "Undefined superclass method '{}@{}'",
            method_name, super_class_identity
        ))
        .with_identifier("RunMat:UndefinedFunction")
        .build());
    };
    if method.is_static {
        return Err(build_runtime_error(format!(
            "Superclass method '{}@{}' is static and cannot be called with super method syntax.",
            method_name, super_class_identity
        ))
        .with_identifier("RunMat:MethodStaticAccess")
        .build());
    }
    let access_allowed = match method.access {
        runmat_types::MemberAccess::Public => true,
        runmat_types::MemberAccess::Protected => {
            crate::class_registry::is_class_or_subclass(&class_identity, &owner)
        }
        runmat_types::MemberAccess::Private => class_identity == owner,
    };
    if !access_allowed {
        return Err(build_runtime_error(format!(
            "Method '{}@{}' is not accessible from class '{}'.",
            method_name, super_class_identity, class_identity
        ))
        .with_identifier("RunMat:MethodPrivate")
        .build());
    }
    let Some(result) = crate::user_functions::try_call_semantic_function_by_name_in_context(
        runtime,
        &method.function_name,
        &args,
        1,
    )
    .await
    else {
        return Err(
            build_runtime_error(format!("Undefined function: {}", method.function_name))
                .with_identifier("RunMat:UndefinedFunction")
                .build(),
        );
    };
    result?.try_map_owned(require_single_sequence, |error| error.into_runtime_error())
}

// handle-object builtins removed for now

pub(crate) async fn classref_builtin(class_name: String) -> crate::BuiltinResult<Value> {
    let identity = runmat_types::ClassIdentity::new(class_name).map_err(|error| {
        build_runtime_error(format!("classref: {error}"))
            .with_identifier("RunMat:InvalidClassName")
            .build()
    })?;
    Ok(Value::ClassRef(identity.into()))
}

pub(crate) async fn register_test_classes_builtin() -> crate::BuiltinResult<Value> {
    let mut props = std::collections::HashMap::new();
    props.insert(
        "x".into(),
        crate::class_registry::RuntimeProperty {
            name: "x".into(),
            is_static: false,
            is_constant: false,
            is_dependent: false,
            get_access: MemberAccess::Public,
            set_access: MemberAccess::Public,
            default_value: Some(Value::Num(0.0)),
        },
    );
    props.insert(
        "y".into(),
        crate::class_registry::RuntimeProperty {
            name: "y".into(),
            is_static: false,
            is_constant: false,
            is_dependent: false,
            get_access: MemberAccess::Public,
            set_access: MemberAccess::Public,
            default_value: Some(Value::Num(0.0)),
        },
    );
    props.insert(
        "staticValue".into(),
        crate::class_registry::RuntimeProperty {
            name: "staticValue".into(),
            is_static: true,
            is_constant: false,
            is_dependent: false,
            get_access: MemberAccess::Public,
            set_access: MemberAccess::Public,
            default_value: Some(Value::Num(42.0)),
        },
    );
    props.insert(
        "secret".into(),
        crate::class_registry::RuntimeProperty {
            name: "secret".into(),
            is_static: false,
            is_constant: false,
            is_dependent: false,
            get_access: MemberAccess::Private,
            set_access: MemberAccess::Private,
            default_value: Some(Value::Num(99.0)),
        },
    );
    let mut methods = std::collections::HashMap::new();
    methods.insert(
        "move".into(),
        crate::class_registry::RuntimeMethod {
            name: "move".into(),
            is_static: false,
            is_abstract: false,
            is_sealed: false,
            access: MemberAccess::Public,
            function_name: "Point.move".into(),
            implicit_class_argument: None,
            role: runmat_types::ClassMethodRole::Ordinary,
        },
    );
    methods.insert(
        "origin".into(),
        crate::class_registry::RuntimeMethod {
            name: "origin".into(),
            is_static: true,
            is_abstract: false,
            is_sealed: false,
            access: MemberAccess::Public,
            function_name: "Point.origin".into(),
            implicit_class_argument: None,
            role: runmat_types::ClassMethodRole::Ordinary,
        },
    );
    crate::class_registry::register_class(crate::class_registry::RuntimeClass {
        events: Default::default(),
        name: "Point".into(),
        parent: None,
        properties: props,
        methods,
    });

    // Namespaced class example: pkg.PointNS with same shape as Point
    let mut ns_props = std::collections::HashMap::new();
    ns_props.insert(
        "x".into(),
        crate::class_registry::RuntimeProperty {
            name: "x".into(),
            is_static: false,
            is_constant: false,
            is_dependent: false,
            get_access: MemberAccess::Public,
            set_access: MemberAccess::Public,
            default_value: Some(Value::Num(1.0)),
        },
    );
    ns_props.insert(
        "y".into(),
        crate::class_registry::RuntimeProperty {
            name: "y".into(),
            is_static: false,
            is_constant: false,
            is_dependent: false,
            get_access: MemberAccess::Public,
            set_access: MemberAccess::Public,
            default_value: Some(Value::Num(2.0)),
        },
    );
    let ns_methods = std::collections::HashMap::new();
    crate::class_registry::register_class(crate::class_registry::RuntimeClass {
        events: Default::default(),
        name: "pkg.PointNS".into(),
        parent: None,
        properties: ns_props,
        methods: ns_methods,
    });

    // Inheritance: Shape (base) and Circle (derived)
    let shape_props = std::collections::HashMap::new();
    let mut shape_methods = std::collections::HashMap::new();
    shape_methods.insert(
        "area".into(),
        crate::class_registry::RuntimeMethod {
            name: "area".into(),
            is_static: false,
            is_abstract: false,
            is_sealed: false,
            access: MemberAccess::Public,
            function_name: "Shape.area".into(),
            implicit_class_argument: None,
            role: runmat_types::ClassMethodRole::Ordinary,
        },
    );
    crate::class_registry::register_class(crate::class_registry::RuntimeClass {
        events: Default::default(),
        name: "Shape".into(),
        parent: None,
        properties: shape_props,
        methods: shape_methods,
    });

    let mut circle_props = std::collections::HashMap::new();
    circle_props.insert(
        "r".into(),
        crate::class_registry::RuntimeProperty {
            name: "r".into(),
            is_static: false,
            is_constant: false,
            is_dependent: false,
            get_access: MemberAccess::Public,
            set_access: MemberAccess::Public,
            default_value: Some(Value::Num(0.0)),
        },
    );
    let mut circle_methods = std::collections::HashMap::new();
    circle_methods.insert(
        "area".into(),
        crate::class_registry::RuntimeMethod {
            name: "area".into(),
            is_static: false,
            is_abstract: false,
            is_sealed: false,
            access: MemberAccess::Public,
            function_name: "Circle.area".into(),
            implicit_class_argument: None,
            role: runmat_types::ClassMethodRole::Ordinary,
        },
    );
    crate::class_registry::register_class(crate::class_registry::RuntimeClass {
        events: Default::default(),
        name: "Circle".into(),
        parent: Some("Shape".into()),
        properties: circle_props,
        methods: circle_methods,
    });

    // Constructor demo class: Ctor with static constructor method Ctor
    let ctor_props = std::collections::HashMap::new();
    let mut ctor_methods = std::collections::HashMap::new();
    ctor_methods.insert(
        "Ctor".into(),
        crate::class_registry::RuntimeMethod {
            name: "Ctor".into(),
            is_static: true,
            is_abstract: false,
            is_sealed: false,
            access: MemberAccess::Public,
            function_name: "Ctor.Ctor".into(),
            implicit_class_argument: None,
            role: runmat_types::ClassMethodRole::Constructor,
        },
    );
    crate::class_registry::register_class(crate::class_registry::RuntimeClass {
        events: Default::default(),
        name: "Ctor".into(),
        parent: None,
        properties: ctor_props,
        methods: ctor_methods,
    });

    // Overloaded indexing demo class: OverIdx with subsref/subsasgn
    let overidx_props = std::collections::HashMap::new();
    let mut overidx_methods = std::collections::HashMap::new();
    overidx_methods.insert(
        OBJECT_SUBSREF_METHOD.into(),
        crate::class_registry::RuntimeMethod {
            name: OBJECT_SUBSREF_METHOD.into(),
            is_static: false,
            is_abstract: false,
            is_sealed: false,
            access: MemberAccess::Public,
            function_name: format!("OverIdx.{OBJECT_SUBSREF_METHOD}"),
            implicit_class_argument: None,
            role: runmat_types::ClassMethodRole::Ordinary,
        },
    );
    overidx_methods.insert(
        OBJECT_SUBSASGN_METHOD.into(),
        crate::class_registry::RuntimeMethod {
            name: OBJECT_SUBSASGN_METHOD.into(),
            is_static: false,
            is_abstract: false,
            is_sealed: false,
            access: MemberAccess::Public,
            function_name: format!("OverIdx.{OBJECT_SUBSASGN_METHOD}"),
            implicit_class_argument: None,
            role: runmat_types::ClassMethodRole::Ordinary,
        },
    );
    overidx_methods.insert(
        crate::builtins::introspection::object_indexing::NUM_ARGUMENTS_FROM_SUBSCRIPT_METHOD.into(),
        crate::class_registry::RuntimeMethod {
            name:
                crate::builtins::introspection::object_indexing::NUM_ARGUMENTS_FROM_SUBSCRIPT_METHOD
                    .into(),
            is_static: false,
            is_abstract: false,
            is_sealed: false,
            access: MemberAccess::Public,
            function_name: format!(
                "OverIdx.{}",
                crate::builtins::introspection::object_indexing::NUM_ARGUMENTS_FROM_SUBSCRIPT_METHOD
            ),
            implicit_class_argument: None,
            role: runmat_types::ClassMethodRole::Ordinary,
        },
    );
    for (name, is_static) in [
        (
            crate::builtins::introspection::object_serialization::SAVEOBJ_METHOD,
            false,
        ),
        (
            crate::builtins::introspection::object_serialization::LOADOBJ_METHOD,
            true,
        ),
    ] {
        overidx_methods.insert(
            name.into(),
            crate::class_registry::RuntimeMethod {
                name: name.into(),
                is_static,
                is_abstract: false,
                is_sealed: false,
                access: MemberAccess::Public,
                function_name: format!("OverIdx.{name}"),
                implicit_class_argument: None,
                role: runmat_types::ClassMethodRole::Ordinary,
            },
        );
    }
    for name in [
        "plus", "times", "mtimes", "lt", "gt", "eq", "uplus", "rdivide", "mrdivide", "ldivide",
        "mldivide", "and", "or", "xor",
    ] {
        overidx_methods.insert(
            name.into(),
            crate::class_registry::RuntimeMethod {
                name: name.into(),
                is_static: false,
                is_abstract: false,
                is_sealed: false,
                access: MemberAccess::Public,
                function_name: format!("OverIdx.{name}"),
                implicit_class_argument: None,
                role: runmat_types::ClassMethodRole::Ordinary,
            },
        );
    }
    crate::class_registry::register_class(crate::class_registry::RuntimeClass {
        events: Default::default(),
        name: "OverIdx".into(),
        parent: None,
        properties: overidx_props,
        methods: overidx_methods,
    });

    // Class without indexing protocol methods, used by negative subsref/subsasgn contracts.
    crate::class_registry::register_class(crate::class_registry::RuntimeClass {
        events: Default::default(),
        name: "NoIdx".into(),
        parent: None,
        properties: std::collections::HashMap::new(),
        methods: std::collections::HashMap::new(),
    });
    Ok(Value::Num(1.0))
}

#[cfg(feature = "test-classes")]
pub async fn test_register_classes() {
    let _ = register_test_classes_builtin().await;
}

#[cfg(test)]
mod tests {
    use super::*;

    async fn feval_builtin(
        f: Value,
        rest: Vec<Value>,
    ) -> crate::BuiltinResult<crate::execution::RootedValue> {
        let rest = crate::execution::RootedValues::new(rest)
            .map_err(|error| error.into_runtime_error())?
            .into();
        crate::builtins::introspection::feval::feval_builtin_registered(None, None, f, rest)
            .await
            .and_then(|result| {
                result.try_map_owned(
                    |sequence| {
                        Ok(crate::call::arguments::project_legacy_builtin_value_abi(
                            sequence,
                        ))
                    },
                    |error| error.into_runtime_error(),
                )
            })
    }
    use crate::builtins::introspection::test_methods::*;
    use futures::executor::block_on;
    use runmat_value::{HandleRef, IntegerStorage, Tensor};
    use std::collections::HashMap;
    use std::sync::{
        atomic::{AtomicU64, AtomicUsize, Ordering},
        Arc, Mutex,
    };

    static TEST_CLASS_COUNTER: AtomicU64 = AtomicU64::new(0);
    static LISTENER_TEST_LOCK: Mutex<()> = Mutex::new(());

    fn unique_class_name(prefix: &str) -> String {
        let id = TEST_CLASS_COUNTER.fetch_add(1, Ordering::Relaxed);
        format!("{}_{}", prefix, id)
    }

    fn standard_subscript(kind: &str, payload: Value) -> Value {
        crate::object::indexing::standard_substruct_fixture_from_parts(kind, payload)
            .expect("valid standard substruct fixture")
    }

    fn listener_gc_test(test: impl FnOnce()) {
        let _guard = LISTENER_TEST_LOCK
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        reset_event_registry_for_test();
        register_event_target_test_class();
        test();
        reset_event_registry_for_test();
    }

    fn register_event_target_test_class() {
        let events = ["Changed", "ChangedSoundness"]
            .into_iter()
            .map(|name| {
                let event = runmat_types::EventDeclaration {
                    name: name.into(),
                    attributes: Default::default(),
                    declared_attributes: Vec::new(),
                    span: Default::default(),
                };
                (event.name.clone(), event)
            })
            .collect();
        crate::class_registry::register_class(crate::class_registry::RuntimeClass {
            name: "EventTarget".into(),
            parent: Some(runmat_types::standard::HANDLE.owned()),
            properties: Default::default(),
            methods: Default::default(),
            events,
        });
    }

    #[test]
    fn descriptor_migration_covers_lib_runtime_builtins() {
        let cases = [
            ("deal", "[varargout] = deal(varargin)"),
            ("rethrow", "rethrow(err)"),
            ("call_method", "[out] = call_method(base, method, varargin)"),
            (
                "new_handle_object",
                "handle = new_handle_object(class_name)",
            ),
            (
                "addlistener",
                "listener = addlistener(target, event_name, callback)",
            ),
            ("notify", "status = notify(target, event_name, varargin)"),
            ("get.p", "value = get.p(obj)"),
            ("set.p", "obj = set.p(obj, value)"),
            ("make_anon", "handle_text = make_anon(params, body)"),
            ("classref", "ref = classref(class_name)"),
            (
                "__register_test_classes",
                "status = __register_test_classes()",
            ),
            ("Point.move", "obj = Point.move(obj, dx, dy)"),
            ("Circle.area", "area = Circle.area(obj)"),
            ("Ctor.Ctor", "obj = Ctor.Ctor(x)"),
            ("PkgF.foo", "value = PkgF.foo()"),
            ("OverIdx.plus", "out = OverIdx.plus(obj, rhs)"),
            ("OverIdx.subsref", "out = OverIdx.subsref(obj, S)"),
            ("feval", "[varargout] = feval(f, varargin)"),
            ("str2func", "fh = str2func(name)"),
            ("func2str", "name = func2str(fh)"),
            ("functions", "info = functions(fh)"),
            ("inputname", "name = inputname(argNumber)"),
            ("localfunctions", "handles = localfunctions()"),
            ("narginchk", "narginchk(minArgs, maxArgs)"),
            ("nargoutchk", "nargoutchk(minArgs, maxArgs)"),
            ("mfilename", "name = mfilename()"),
            ("getmethod", "fh = getmethod(obj_or_class, name)"),
        ];

        for (name, label) in cases {
            let descriptor = runmat_builtins::builtin_catalog_entry_by_name(name)
                .map(|entry| entry.descriptor)
                .or_else(|| {
                    runmat_builtins::builtin_function_by_name(name)
                        .and_then(|builtin| builtin.descriptor)
                })
                .unwrap_or_else(|| panic!("descriptor missing for {name}"));
            assert!(
                descriptor.signatures.iter().any(|sig| sig.label == label),
                "missing signature {label} for {name}"
            );
        }
    }

    #[test]
    fn non_object_handle_targets_are_invalid() {
        let target = runmat_gc::gc_allocate(Value::Num(1.0)).expect("gc allocate target");
        let handle = HandleRef {
            class_name: "MalformedHandle".into(),
            target,
            valid: true,
        };

        assert!(!is_handle_valid(&handle));
    }

    #[test]
    fn handle_validity_requires_exact_target_class_and_retains_derived_receivers() {
        let child: runmat_types::ClassIdentity = "HandleValidityDerivedSourceTest".into();
        crate::class_registry::register_class(crate::class_registry::RuntimeClass {
            name: child.clone(),
            parent: Some(runmat_types::standard::HANDLE.owned()),
            properties: Default::default(),
            methods: Default::default(),
            events: Default::default(),
        });
        let target = runmat_gc::gc_allocate(Value::Object(runmat_value::ObjectInstance::new(
            child.clone(),
        )))
        .expect("derived target");
        let derived = HandleRef {
            class_name: child.into(),
            target,
            valid: true,
        };
        assert!(is_handle_valid(&derived));
        let mismatched_alias = HandleRef {
            class_name: runmat_types::standard::HANDLE.owned().into(),
            target,
            valid: true,
        };
        assert!(!is_handle_valid(&mismatched_alias));
        assert!(set_handle_valid(&derived, false));
        assert!(!is_handle_valid(&derived));
    }

    #[test]
    fn feval_closure_uses_semantic_function_identity() {
        let runtime = crate::user_functions::test_support::exact_fixture_runtime();
        let _entry = runtime.enter();
        let binding = crate::user_functions::test_support::exact_local_binding(&runtime, 42);
        let _bound = crate::user_functions::test_support::install_bound_forwarder(binding.clone());
        let _guard = crate::user_functions::install_semantic_function_invoker(Some(Arc::new(
            |function, args, requested_outputs| {
                assert_eq!(function, 42);
                assert_eq!(requested_outputs, 1);
                assert_eq!(args, &[Value::Num(2.0)]);
                Box::pin(async {
                    crate::user_functions::test_support::rooted_single_value_sequence(Value::Num(
                        7.0,
                    ))
                })
            },
        )))
        .expect("call-port scope");
        let closure = Value::Closure(runmat_value::Closure::from_data(
            runmat_value::ClosureData {
                lexical_environment: None,
                function_name: "function_target".into(),
                binding: Some(binding),
                source_role: runmat_types::FunctionSourceRole::Unknown,
                source_file: None,
                captures: Vec::new().into(),
            },
        ));

        let result = block_on(feval_builtin(closure, vec![Value::Num(2.0)]))
            .expect("semantic closure feval succeeds");
        assert_eq!(result, Value::Num(7.0));
    }

    #[test]
    fn feval_semantic_function_handle_uses_semantic_identity() {
        let runtime = crate::user_functions::test_support::exact_fixture_runtime();
        let _entry = runtime.enter();
        let binding = crate::user_functions::test_support::exact_local_binding(&runtime, 43);
        let _bound = crate::user_functions::test_support::install_bound_forwarder(binding.clone());
        let _guard = crate::user_functions::install_semantic_function_invoker(Some(Arc::new(
            |function, args, requested_outputs| {
                assert_eq!(function, 43);
                assert_eq!(requested_outputs, 1);
                assert_eq!(args, &[Value::Num(3.0)]);
                Box::pin(async {
                    crate::user_functions::test_support::rooted_single_value_sequence(Value::Num(
                        9.0,
                    ))
                })
            },
        )))
        .expect("call-port scope");
        let handle =
            crate::user_functions::test_support::exact_local_value("function_target", binding);

        let result = block_on(feval_builtin(handle, vec![Value::Num(3.0)]))
            .expect("semantic function handle feval succeeds");
        assert_eq!(result, Value::Num(9.0));
    }

    #[test]
    fn feval_semantic_function_handle_errors_when_semantic_invoker_unavailable() {
        let _guard =
            crate::user_functions::clear_semantic_function_invoker().expect("call-port scope");
        let handle = Value::BoundFunctionHandle(runmat_value::BoundFunctionHandle::from_data(
            runmat_value::BoundFunctionHandleData {
                name: "function_target".into(),
                function: 9043,
                authority: runmat_value::BoundFunctionAuthority::unavailable(9043),
                source_role: runmat_types::FunctionSourceRole::Unknown,
                source_file: None,
            },
        ));

        let err = block_on(feval_builtin(handle, vec![Value::Num(3.0)])).expect_err(
            "semantic function handle should not fall back to name-based dispatch when unavailable",
        );
        assert_eq!(err.identifier(), Some("RunMat:UndefinedSemanticFunction"));
        assert!(
            err.message()
                .contains("'function_target' could not invoke semantic function 9043"),
            "unexpected error: {err:?}"
        );
    }

    #[test]
    fn feval_name_only_handle_uses_semantic_resolver() {
        let _resolver_guard =
            crate::user_functions::install_semantic_function_resolver(Some(Arc::new(|name| {
                (name == "resolved_target").then_some(45)
            })))
            .expect("call-port scope");
        let _invoker_guard = crate::user_functions::install_semantic_function_invoker(Some(
            Arc::new(|function, args, requested_outputs| {
                assert_eq!(function, 45);
                assert_eq!(requested_outputs, 1);
                assert_eq!(args, &[Value::Num(4.0)]);
                Box::pin(async {
                    crate::user_functions::test_support::rooted_single_value_sequence(Value::Num(
                        11.0,
                    ))
                })
            }),
        ))
        .expect("call-port scope");

        let result = block_on(feval_builtin(
            Value::FunctionHandle("resolved_target".into()),
            vec![Value::Num(4.0)],
        ))
        .expect("resolved name-only handle feval succeeds");
        assert_eq!(result, Value::Num(11.0));
    }

    #[test]
    fn feval_method_function_handle_uses_semantic_resolver() {
        let _resolver_guard =
            crate::user_functions::install_semantic_function_resolver(Some(Arc::new(|name| {
                (name == "resolved_method").then_some(5045)
            })))
            .expect("call-port scope");
        let _invoker_guard = crate::user_functions::install_semantic_function_invoker(Some(
            Arc::new(|function, args, requested_outputs| {
                assert_eq!(function, 5045);
                assert_eq!(requested_outputs, 1);
                assert_eq!(args, &[Value::Num(4.0)]);
                Box::pin(async {
                    crate::user_functions::test_support::rooted_single_value_sequence(Value::Num(
                        15.0,
                    ))
                })
            }),
        ))
        .expect("call-port scope");

        let result = block_on(feval_builtin(
            Value::MethodFunctionHandle("resolved_method".into()),
            vec![Value::Num(4.0)],
        ))
        .expect("resolved method handle feval succeeds");
        assert_eq!(result, Value::Num(15.0));
    }

    #[test]
    fn feval_method_function_handle_does_not_fallback_to_builtin_name() {
        let _resolver_guard = crate::user_functions::install_semantic_function_resolver(None)
            .expect("call-port scope");
        let err = block_on(feval_builtin(
            Value::MethodFunctionHandle("sqrt".into()),
            vec![Value::Num(9.0)],
        ))
        .expect_err("method function handle should not fallback to builtin name dispatch");
        assert_eq!(err.identifier(), Some("RunMat:UndefinedFunction"));
    }

    #[test]
    fn feval_name_only_closure_uses_semantic_resolver() {
        let _resolver_guard =
            crate::user_functions::install_semantic_function_resolver(Some(Arc::new(|name| {
                (name == "resolved_target").then_some(145)
            })))
            .expect("call-port scope");
        let _invoker_guard = crate::user_functions::install_semantic_function_invoker(Some(
            Arc::new(|function, args, requested_outputs| {
                assert_eq!(function, 145);
                assert_eq!(requested_outputs, 1);
                assert_eq!(args, &[Value::Num(9.0), Value::Num(4.0)]);
                Box::pin(async {
                    crate::user_functions::test_support::rooted_single_value_sequence(Value::Num(
                        13.0,
                    ))
                })
            }),
        ))
        .expect("call-port scope");

        let closure = Value::Closure(runmat_value::Closure::from_data(
            runmat_value::ClosureData {
                lexical_environment: None,
                function_name: "resolved_target".into(),
                binding: None,
                source_role: runmat_types::FunctionSourceRole::Unknown,
                source_file: None,
                captures: vec![Value::Num(9.0)].into(),
            },
        ));

        let result = block_on(feval_builtin(closure, vec![Value::Num(4.0)]))
            .expect("resolved name-only closure feval succeeds");
        assert_eq!(result, Value::Num(13.0));
    }

    #[test]
    fn feval_name_only_closure_falls_back_when_semantic_invoker_unavailable() {
        let _resolver_guard =
            crate::user_functions::install_semantic_function_resolver(Some(Arc::new(|name| {
                (name == "sin").then_some(245)
            })))
            .expect("call-port scope");
        let _invoker_guard =
            crate::user_functions::clear_semantic_function_invoker().expect("call-port scope");

        let closure = Value::Closure(runmat_value::Closure::from_data(
            runmat_value::ClosureData {
                lexical_environment: None,
                function_name: "sin".into(),
                binding: None,
                source_role: runmat_types::FunctionSourceRole::Unknown,
                source_file: None,
                captures: Vec::new().into(),
            },
        ));

        let result =
            block_on(feval_builtin(closure, vec![Value::Num(0.0)])).expect("sin fallback works");
        assert_eq!(result, Value::Num(0.0));
    }

    #[test]
    fn feval_external_function_handle_errors_when_unresolved() {
        let _resolver_guard = crate::user_functions::install_semantic_function_resolver(None)
            .expect("call-port scope");
        let err = block_on(feval_builtin(
            Value::ExternalFunctionHandle("missing.external".into()),
            vec![Value::Num(1.0)],
        ))
        .expect_err("external function handle should error when unresolved");
        assert_eq!(err.identifier(), Some("RunMat:UndefinedFunction"));
        assert!(
            err.message().contains("missing.external"),
            "unexpected error: {err:?}"
        );
    }

    #[test]
    fn feval_single_segment_external_function_handle_uses_runtime_name_resolution() {
        let _resolver_guard =
            crate::user_functions::install_semantic_function_resolver(Some(Arc::new(|name| {
                (name == "resolved_target").then_some(4501)
            })))
            .expect("call-port scope");
        let _invoker_guard = crate::user_functions::install_semantic_function_invoker(Some(
            Arc::new(|function, args, requested_outputs| {
                assert_eq!(function, 4501);
                assert_eq!(requested_outputs, 1);
                assert_eq!(args, &[Value::Num(4.0)]);
                Box::pin(async {
                    crate::user_functions::test_support::rooted_single_value_sequence(Value::Num(
                        12.0,
                    ))
                })
            }),
        ))
        .expect("call-port scope");

        let result = block_on(feval_builtin(
            Value::ExternalFunctionHandle("resolved_target".into()),
            vec![Value::Num(4.0)],
        ))
        .expect("single-segment external function handle should use runtime-name resolution");
        assert_eq!(result, Value::Num(12.0));
    }

    #[test]
    fn feval_accepts_documented_plain_string_function_name() {
        let value = block_on(feval_builtin(
            Value::String("sin".to_string().into()),
            vec![Value::Num(0.0)],
        ))
        .expect("plain string function name should resolve");
        assert_eq!(value, Value::Num(0.0));
    }

    #[test]
    fn feval_accepts_documented_plain_char_function_name() {
        let value = block_on(feval_builtin(
            Value::CharArray(runmat_value::CharArray::new_row("sin")),
            vec![Value::Num(0.0)],
        ))
        .expect("plain character-vector function name should resolve");
        assert_eq!(value, Value::Num(0.0));
    }

    #[test]
    fn feval_forwards_all_integer_classes_without_conversion() {
        let _resolver_guard =
            crate::user_functions::install_semantic_function_resolver(Some(Arc::new(|name| {
                (name == "integer_identity").then_some(654_321)
            })))
            .expect("call-port scope");
        let _invoker_guard = crate::user_functions::install_semantic_function_invoker(Some(
            Arc::new(|function, args, requested_outputs| {
                assert_eq!(function, 654_321);
                assert_eq!(requested_outputs, 1);
                assert_eq!(args.len(), 1);
                let output = args[0].clone();
                Box::pin(async move {
                    crate::user_functions::test_support::rooted_single_value_sequence(output)
                })
            }),
        ))
        .expect("call-port scope");

        let storages = [
            IntegerStorage::I8(vec![i8::MIN, i8::MAX]),
            IntegerStorage::I16(vec![i16::MIN, i16::MAX]),
            IntegerStorage::I32(vec![i32::MIN, i32::MAX]),
            IntegerStorage::I64(vec![i64::MIN, i64::MAX]),
            IntegerStorage::U8(vec![0, u8::MAX]),
            IntegerStorage::U16(vec![0, u16::MAX]),
            IntegerStorage::U32(vec![0, u32::MAX]),
            IntegerStorage::U64(vec![0, u64::MAX]),
        ];

        for storage in storages {
            let input = Value::Tensor(
                Tensor::new_integer(storage, vec![1, 2]).expect("integer forwarding input"),
            );
            let output = block_on(feval_builtin(
                Value::CharArray(runmat_value::CharArray::new_row("integer_identity")),
                vec![input.clone()],
            ))
            .expect("integer argument should be forwarded unchanged");
            assert_eq!(output, input);
        }
    }

    #[test]
    fn feval_rejects_non_row_char_handle_with_identifier() {
        let chars = runmat_value::CharArray::from_unicode_scalars(vec!['@', 's'], 2, 1)
            .expect("char array construction should succeed");
        let err = block_on(feval_builtin(
            Value::CharArray(chars),
            vec![Value::Num(0.0)],
        ))
        .expect_err("feval non-row char handle should fail");
        assert_eq!(err.identifier(), Some("RunMat:FevalHandleShapeInvalid"));
    }

    #[test]
    fn feval_rejects_empty_at_string_handle_with_identifier() {
        let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
        let err = block_on(feval_builtin(
            Value::String("@".to_string().into()),
            vec![Value::Num(0.0)],
        ))
        .expect_err("feval empty @string handle should fail");
        assert_eq!(err.identifier(), Some("RunMat:FevalHandleNameInvalid"));
    }

    #[test]
    fn feval_strict_mode_rejects_at_prefixed_text_before_dispatch() {
        let _compat = crate::compatibility::push_runmat_extensions_enabled(false);
        let err = block_on(feval_builtin(
            Value::String("@sin".to_string().into()),
            vec![Value::Num(0.0)],
        ))
        .expect_err("@-prefixed text target should require RunMat mode");
        assert_eq!(
            err.identifier(),
            Some("RunMat:compatibility:FevalAtPrefixedTextTargetExtension")
        );
    }

    #[test]
    fn feval_rejects_empty_function_handle_value_with_identifier() {
        let err = block_on(feval_builtin(
            Value::FunctionHandle(String::new().into()),
            vec![Value::Num(0.0)],
        ))
        .expect_err("feval empty function-handle value should fail");
        assert_eq!(err.identifier(), Some("RunMat:FevalHandleNameInvalid"));
    }

    #[test]
    fn feval_trims_text_handle_name_for_resolution() {
        let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
        let _resolver_guard =
            crate::user_functions::install_semantic_function_resolver(Some(Arc::new(|name| {
                (name == "resolved_target").then_some(9876)
            })))
            .expect("call-port scope");
        let _invoker_guard = crate::user_functions::install_semantic_function_invoker(Some(
            Arc::new(|function, args, requested_outputs| {
                assert_eq!(function, 9876);
                assert_eq!(requested_outputs, 1);
                assert_eq!(args, &[Value::Num(4.0)]);
                Box::pin(async {
                    crate::user_functions::test_support::rooted_single_value_sequence(Value::Num(
                        12.0,
                    ))
                })
            }),
        ))
        .expect("call-port scope");

        let value = block_on(feval_builtin(
            Value::String("@ resolved_target ".to_string().into()),
            vec![Value::Num(4.0)],
        ))
        .expect("trimmed text handle should resolve");
        assert_eq!(value, Value::Num(12.0));
    }

    #[test]
    fn str2func_returns_semantic_handle_when_resolver_can_resolve() {
        let _resolver_guard =
            crate::user_functions::install_semantic_function_resolver(Some(Arc::new(|name| {
                (name == "resolved_target").then_some(145)
            })))
            .expect("call-port scope");
        let value = crate::builtins::introspection::function_handle_text::dispatch_str2func(
            Value::String("resolved_target".to_string().into()),
        )
        .expect("str2func should succeed");
        assert_eq!(
            value,
            Value::BoundFunctionHandle(runmat_value::BoundFunctionHandle::from_data(
                runmat_value::BoundFunctionHandleData {
                    name: "resolved_target".into(),
                    function: 145,
                    authority: runmat_value::BoundFunctionAuthority::unavailable(145),
                    source_role: runmat_types::FunctionSourceRole::Unknown,
                    source_file: None,
                }
            ))
        );
    }

    #[test]
    fn str2func_returns_dynamic_handle_when_resolver_cannot_resolve() {
        let _resolver_guard = crate::user_functions::install_semantic_function_resolver(None)
            .expect("call-port scope");
        let value = crate::builtins::introspection::function_handle_text::dispatch_str2func(
            Value::String("@missing_target".to_string().into()),
        )
        .expect("str2func should succeed");
        assert_eq!(value, Value::FunctionHandle("missing_target".into()));
    }

    #[test]
    fn str2func_returns_external_handle_for_qualified_name() {
        let _resolver_guard = crate::user_functions::install_semantic_function_resolver(None)
            .expect("call-port scope");
        let value = crate::builtins::introspection::function_handle_text::dispatch_str2func(
            Value::String("Point.origin".to_string().into()),
        )
        .expect("str2func should succeed");
        assert_eq!(value, Value::ExternalFunctionHandle("Point.origin".into()));
    }

    #[test]
    fn str2func_malformed_qualified_name_returns_dynamic_handle() {
        let _resolver_guard = crate::user_functions::install_semantic_function_resolver(None)
            .expect("call-port scope");
        let value = crate::builtins::introspection::function_handle_text::dispatch_str2func(
            Value::String("Point..origin".to_string().into()),
        )
        .expect("str2func should succeed");
        assert_eq!(value, Value::FunctionHandle("Point..origin".into()));
    }

    #[test]
    fn func2str_rejects_non_handle_with_identifier() {
        let err = crate::builtins::introspection::function_handle_text::dispatch_func2str(
            Value::Num(1.0),
        )
        .expect_err("func2str non-handle input should fail");
        assert_eq!(err.identifier(), Some("RunMat:Func2StrHandleTypeInvalid"));
    }

    #[test]
    fn str2func_rejects_empty_name_with_identifier() {
        let err = crate::builtins::introspection::function_handle_text::dispatch_str2func(
            Value::String("   ".to_string().into()),
        )
        .expect_err("empty function name should fail");
        assert_eq!(err.identifier(), Some("RunMat:Str2FuncNameInvalid"));
    }

    #[test]
    fn str2func_rejects_non_row_char_name_with_identifier() {
        let chars = runmat_value::CharArray::from_unicode_scalars(vec!['a', 'b'], 2, 1)
            .expect("char array construction should succeed");
        let err = crate::builtins::introspection::function_handle_text::dispatch_str2func(
            Value::CharArray(chars),
        )
        .expect_err("non-row char-array function name should fail");
        assert_eq!(err.identifier(), Some("RunMat:Str2FuncNameShapeInvalid"));
    }

    #[test]
    fn str2func_rejects_non_text_name_with_identifier() {
        let err = crate::builtins::introspection::function_handle_text::dispatch_str2func(
            Value::Num(1.0),
        )
        .expect_err("non-text function name should fail");
        assert_eq!(err.identifier(), Some("RunMat:Str2FuncNameTypeInvalid"));
    }

    #[test]
    fn str2func_accepts_scalar_string_array_name() {
        let _resolver_guard = crate::user_functions::install_semantic_function_resolver(None)
            .expect("call-port scope");
        let value = crate::builtins::introspection::function_handle_text::dispatch_str2func(
            Value::StringArray(
                runmat_value::StringArray::new(vec!["@missing_target".to_string()], vec![1, 1])
                    .expect("string array construction should succeed"),
            ),
        )
        .expect("scalar string-array function name should succeed");
        assert_eq!(value, Value::FunctionHandle("missing_target".into()));
    }

    #[test]
    fn str2func_rejects_nonscalar_string_array_name_with_identifier() {
        let _resolver_guard = crate::user_functions::install_semantic_function_resolver(None)
            .expect("call-port scope");
        let value = Value::StringArray(
            runmat_value::StringArray::new(vec!["@a".to_string(), "@b".to_string()], vec![1, 2])
                .expect("string array construction should succeed"),
        );
        let err = crate::builtins::introspection::function_handle_text::dispatch_str2func(value)
            .expect_err("nonscalar string-array function name must fail");
        assert_eq!(err.identifier(), Some("RunMat:Str2FuncNameShapeInvalid"));
    }

    #[test]
    fn str2func_scalar_string_array_prefers_semantic_handle_when_resolved() {
        let _resolver_guard =
            crate::user_functions::install_semantic_function_resolver(Some(Arc::new(|name| {
                (name == "resolved_target").then_some(445)
            })))
            .expect("call-port scope");
        let value = crate::builtins::introspection::function_handle_text::dispatch_str2func(
            Value::StringArray(
                runmat_value::StringArray::new(vec!["@resolved_target".to_string()], vec![1, 1])
                    .expect("string array construction should succeed"),
            ),
        )
        .expect("scalar string-array function name should resolve semantically");
        assert_eq!(
            value,
            Value::BoundFunctionHandle(runmat_value::BoundFunctionHandle::from_data(
                runmat_value::BoundFunctionHandleData {
                    name: "resolved_target".into(),
                    function: 445,
                    authority: runmat_value::BoundFunctionAuthority::unavailable(445),
                    source_role: runmat_types::FunctionSourceRole::Unknown,
                    source_file: None,
                }
            ))
        );
    }

    #[test]
    fn str2func_scalar_string_array_returns_external_handle_for_qualified_name() {
        let _resolver_guard = crate::user_functions::install_semantic_function_resolver(None)
            .expect("call-port scope");
        let value = crate::builtins::introspection::function_handle_text::dispatch_str2func(
            Value::StringArray(
                runmat_value::StringArray::new(vec!["Point.origin".to_string()], vec![1, 1])
                    .expect("string array construction should succeed"),
            ),
        )
        .expect("scalar string-array qualified name should succeed");
        assert_eq!(value, Value::ExternalFunctionHandle("Point.origin".into()));
    }

    #[test]
    fn str2func_scalar_string_array_malformed_qualified_name_returns_dynamic_handle() {
        let _resolver_guard = crate::user_functions::install_semantic_function_resolver(None)
            .expect("call-port scope");
        let value = crate::builtins::introspection::function_handle_text::dispatch_str2func(
            Value::StringArray(
                runmat_value::StringArray::new(vec!["Point..origin".to_string()], vec![1, 1])
                    .expect("string array construction should succeed"),
            ),
        )
        .expect("scalar string-array malformed qualified name should succeed");
        assert_eq!(value, Value::FunctionHandle("Point..origin".into()));
    }

    #[test]
    fn str2func_scalar_string_array_rejects_empty_name_with_identifier() {
        let _resolver_guard = crate::user_functions::install_semantic_function_resolver(None)
            .expect("call-port scope");
        let err = crate::builtins::introspection::function_handle_text::dispatch_str2func(
            Value::StringArray(
                runmat_value::StringArray::new(vec!["   ".to_string()], vec![1, 1])
                    .expect("string array construction should succeed"),
            ),
        )
        .expect_err("scalar string-array empty function name should fail");
        assert_eq!(err.identifier(), Some("RunMat:Str2FuncNameInvalid"));
    }

    #[test]
    fn str2func_scalar_string_array_qualified_name_prefers_semantic_handle_when_resolved() {
        let _resolver_guard =
            crate::user_functions::install_semantic_function_resolver(Some(Arc::new(|name| {
                (name == "pkg.resolved_target").then_some(446)
            })))
            .expect("call-port scope");
        let value = crate::builtins::introspection::function_handle_text::dispatch_str2func(
            Value::StringArray(
                runmat_value::StringArray::new(
                    vec!["@pkg.resolved_target".to_string()],
                    vec![1, 1],
                )
                .expect("string array construction should succeed"),
            ),
        )
        .expect("scalar string-array qualified function name should resolve semantically");
        assert_eq!(
            value,
            Value::BoundFunctionHandle(runmat_value::BoundFunctionHandle::from_data(
                runmat_value::BoundFunctionHandleData {
                    name: "pkg.resolved_target".into(),
                    function: 446,
                    authority: runmat_value::BoundFunctionAuthority::unavailable(446),
                    source_role: runmat_types::FunctionSourceRole::Unknown,
                    source_file: None,
                }
            ))
        );
    }

    #[test]
    fn getmethod_classref_returns_typed_external_function_handle() {
        let _extensions = crate::compatibility::push_runmat_extensions_enabled(true);
        let _resolver_guard = crate::user_functions::install_semantic_function_resolver(None)
            .expect("call-port scope");
        let value = crate::builtins::introspection::getmethod::dispatch_getmethod(
            Value::ClassRef("Point".into()),
            "origin".to_string(),
        )
        .expect("getmethod should resolve classref method handle");
        assert_eq!(value, Value::ExternalFunctionHandle("Point.origin".into()));
    }

    #[test]
    fn getmethod_rejects_empty_method_name() {
        let _extensions = crate::compatibility::push_runmat_extensions_enabled(true);
        let err = crate::builtins::introspection::getmethod::dispatch_getmethod(
            Value::ClassRef("Point".into()),
            "   ".to_string(),
        )
        .expect_err("empty method name should be rejected");
        assert_eq!(err.identifier(), Some("RunMat:GetMethodNameInvalid"));
    }

    #[test]
    fn getmethod_rejects_unsupported_receiver_with_identifier() {
        let _extensions = crate::compatibility::push_runmat_extensions_enabled(true);
        let err = crate::builtins::introspection::getmethod::dispatch_getmethod(
            Value::Num(1.0),
            "origin".to_string(),
        )
        .expect_err("unsupported receiver should be rejected");
        assert_eq!(
            err.identifier(),
            Some("RunMat:GetMethodReceiverUnsupported")
        );
    }

    #[test]
    fn create_class_object_handles_class_parent_cycles() {
        let class_a = unique_class_name("runtime_ctor_cycle_a");
        let class_b = unique_class_name("runtime_ctor_cycle_b");

        let mut props_a = HashMap::new();
        props_a.insert(
            "fromA".into(),
            crate::class_registry::RuntimeProperty {
                name: "fromA".into(),
                is_static: false,
                is_constant: false,
                is_dependent: false,
                get_access: MemberAccess::Public,
                set_access: MemberAccess::Public,
                default_value: Some(Value::Num(1.0)),
            },
        );
        let mut props_b = HashMap::new();
        props_b.insert(
            "fromB".into(),
            crate::class_registry::RuntimeProperty {
                name: "fromB".into(),
                is_static: false,
                is_constant: false,
                is_dependent: false,
                get_access: MemberAccess::Public,
                set_access: MemberAccess::Public,
                default_value: Some(Value::Num(2.0)),
            },
        );

        crate::class_registry::register_class(crate::class_registry::RuntimeClass {
            events: Default::default(),
            name: class_a.clone().into(),
            parent: Some(class_b.clone().into()),
            properties: props_a,
            methods: HashMap::new(),
        });
        crate::class_registry::register_class(crate::class_registry::RuntimeClass {
            events: Default::default(),
            name: class_b.into(),
            parent: Some(class_a.clone().into()),
            properties: props_b,
            methods: HashMap::new(),
        });

        let value = block_on(create_class_object(class_a.clone()))
            .expect("constructor should terminate under parent-cycle metadata");
        let Value::Object(obj) = value else {
            panic!("expected object result");
        };
        assert_eq!(obj.class_name().display_name(), class_a);
        assert_eq!(obj.property("fromA"), Some(&Value::Num(1.0)));
        assert_eq!(obj.property("fromB"), Some(&Value::Num(2.0)));
    }

    #[test]
    fn create_class_object_abstract_class_reports_stable_identifier() {
        let class_name = unique_class_name("runtime_ctor_abstract");
        crate::class_registry::register_class_with_modifiers(
            crate::class_registry::RuntimeClass {
                events: Default::default(),
                name: class_name.clone().into(),
                parent: None,
                properties: HashMap::new(),
                methods: HashMap::new(),
            },
            false,
            true,
        );

        let err = block_on(create_class_object(class_name))
            .expect_err("abstract class instantiation should fail");
        assert_eq!(err.identifier(), Some("RunMat:AbstractMethodMissing"));
        assert!(err.message().contains("Cannot instantiate abstract class"));
    }

    #[test]
    fn callable_identity_for_malformed_handle_name_stays_dynamic() {
        let (identity, fallback_policy) = callable_identity_for_handle_name("pkg..remote_inc");
        assert!(matches!(
            identity,
            runmat_types::CallableIdentity::DynamicName(runmat_types::SymbolName(name))
                if name == "pkg..remote_inc"
        ));
        assert_eq!(
            fallback_policy,
            runmat_types::CallableFallbackPolicy::RuntimeNameResolution
        );
    }

    #[test]
    fn unresolved_callable_without_display_name_reports_typed_identity() {
        let err = block_on(dispatch_callable_with_policy(
            runmat_types::CallableIdentity::AnonymousFunction(runmat_types::FunctionId(77)),
            runmat_types::CallableFallbackPolicy::RuntimeNameResolution,
            vec![],
            1,
        ))
        .expect_err("anonymous callable identity should fail unresolved");
        assert_eq!(err.identifier(), Some("RunMat:UndefinedFunction"));
        assert!(
            err.message().contains("AnonymousFunction(FunctionId(77))"),
            "unexpected error: {err:?}"
        );
    }

    #[test]
    fn unresolved_malformed_external_callable_reports_typed_identity() {
        let err = block_on(dispatch_callable_with_policy(
            runmat_types::CallableIdentity::ExternalName(runmat_types::QualifiedName(vec![
                runmat_types::SymbolName("pkg".to_string()),
                runmat_types::SymbolName("".to_string()),
                runmat_types::SymbolName("remote".to_string()),
            ])),
            runmat_types::CallableFallbackPolicy::ExternalBoundary,
            vec![],
            1,
        ))
        .expect_err("malformed external callable identity should fail unresolved");
        assert_eq!(err.identifier(), Some("RunMat:UndefinedFunction"));
        assert!(
            err.message()
                .contains("ExternalName(QualifiedName([SymbolName(\"pkg\"), SymbolName(\"\"), SymbolName(\"remote\")]))"),
            "unexpected error: {err:?}"
        );
    }

    #[test]
    fn unresolved_method_callable_reports_typed_identity() {
        let err = block_on(dispatch_callable_with_policy(
            runmat_types::CallableIdentity::Method(runmat_types::MethodId(
                "missing_method".to_string(),
            )),
            runmat_types::CallableFallbackPolicy::RuntimeNameResolution,
            vec![],
            1,
        ))
        .expect_err("method callable identity should fail unresolved");
        assert_eq!(err.identifier(), Some("RunMat:UndefinedFunction"));
        assert!(
            err.message()
                .contains("Method(MethodId(\"missing_method\"))"),
            "unexpected error: {err:?}"
        );
        assert!(
            err.message()
                .contains("Undefined function 'missing_method'"),
            "method diagnostic should retain its readable name alongside its typed identity: {err:?}"
        );
    }

    #[test]
    fn feval_qualified_at_handle_errors_as_unresolved_external() {
        let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
        let _resolver_guard = crate::user_functions::install_semantic_function_resolver(None)
            .expect("call-port scope");
        let err = block_on(feval_builtin(
            Value::String("@missing.external".to_string().into()),
            vec![Value::Num(1.0)],
        ))
        .expect_err("qualified @handle should error when unresolved");
        assert_eq!(err.identifier(), Some("RunMat:UndefinedFunction"));
        assert!(
            err.message().contains("missing.external"),
            "unexpected error: {err:?}"
        );
    }

    #[test]
    fn func2str_extracts_name_from_function_handles() {
        assert_eq!(
            crate::builtins::introspection::function_handle_text::dispatch_func2str(
                Value::FunctionHandle("sin".into())
            )
            .expect("func2str"),
            Value::String("sin".to_string().into())
        );
        assert_eq!(
            crate::builtins::introspection::function_handle_text::dispatch_func2str(
                Value::ExternalFunctionHandle("Point.origin".into())
            )
            .expect("func2str"),
            Value::String("Point.origin".to_string().into())
        );
        assert_eq!(
            crate::builtins::introspection::function_handle_text::dispatch_func2str(
                Value::BoundFunctionHandle(runmat_value::BoundFunctionHandle::from_data(
                    runmat_value::BoundFunctionHandleData {
                        name: "local_fn".into(),
                        function: 44,
                        authority: runmat_value::BoundFunctionAuthority::unavailable(44),
                        source_role: runmat_types::FunctionSourceRole::Unknown,
                        source_file: None,
                    }
                ))
            )
            .expect("func2str"),
            Value::String("local_fn".to_string().into())
        );
        assert_eq!(
            crate::builtins::introspection::function_handle_text::dispatch_func2str(
                Value::Closure(runmat_value::Closure::from_data(
                    runmat_value::ClosureData {
                        lexical_environment: None,
                        function_name: "captured_fn".into(),
                        binding: None,
                        source_role: runmat_types::FunctionSourceRole::Unknown,
                        source_file: None,
                        captures: Vec::new().into(),
                    }
                ))
            )
            .expect("func2str"),
            Value::String("captured_fn".to_string().into())
        );
    }

    #[test]
    fn none_policy_does_not_use_semantic_resolver() {
        let _resolver_guard =
            crate::user_functions::install_semantic_function_resolver(Some(Arc::new(|name| {
                (name == "resolved_target").then_some(45)
            })))
            .expect("call-port scope");
        let _invoker_guard = crate::user_functions::install_semantic_function_invoker(Some(
            Arc::new(|function, args, requested_outputs| {
                assert_eq!(function, 45);
                assert_eq!(requested_outputs, 1);
                assert_eq!(args, &[Value::Num(4.0)]);
                Box::pin(async {
                    crate::user_functions::test_support::rooted_single_value_sequence(Value::Num(
                        11.0,
                    ))
                })
            }),
        ))
        .expect("call-port scope");

        let request = crate::user_functions::CallableRequest::resolved(
            runmat_types::CallableIdentity::DynamicName(runmat_types::SymbolName(
                "resolved_target".to_string(),
            )),
            runmat_types::CallableFallbackPolicy::None,
            vec![Value::Num(4.0)],
            1,
        );

        let result = block_on(crate::user_functions::try_call_semantic_descriptor(request));
        assert!(result.is_none());
    }

    #[test]
    fn runtime_name_resolution_policy_uses_semantic_resolver() {
        let _resolver_guard =
            crate::user_functions::install_semantic_function_resolver(Some(Arc::new(|name| {
                (name == "resolved_target").then_some(45)
            })))
            .expect("call-port scope");
        let _invoker_guard = crate::user_functions::install_semantic_function_invoker(Some(
            Arc::new(|function, args, requested_outputs| {
                assert_eq!(function, 45);
                assert_eq!(requested_outputs, 1);
                assert_eq!(args, &[Value::Num(4.0)]);
                Box::pin(async {
                    crate::user_functions::test_support::rooted_single_value_sequence(Value::Num(
                        11.0,
                    ))
                })
            }),
        ))
        .expect("call-port scope");

        let request = crate::user_functions::CallableRequest::resolved(
            runmat_types::CallableIdentity::DynamicName(runmat_types::SymbolName(
                "resolved_target".to_string(),
            )),
            runmat_types::CallableFallbackPolicy::RuntimeNameResolution,
            vec![Value::Num(4.0)],
            1,
        );

        let result = block_on(crate::user_functions::try_call_semantic_descriptor(request))
            .expect("runtime resolution should attempt semantic resolver")
            .expect("semantic invoker should succeed");
        assert_eq!(result.as_slice(), &[Value::Num(11.0)]);
    }

    #[test]
    fn object_dispatch_policy_does_not_use_semantic_resolver() {
        let _resolver_guard =
            crate::user_functions::install_semantic_function_resolver(Some(Arc::new(|name| {
                (name == "resolved_target").then_some(45)
            })))
            .expect("call-port scope");
        let _invoker_guard = crate::user_functions::install_semantic_function_invoker(Some(
            Arc::new(|function, args, requested_outputs| {
                assert_eq!(function, 45);
                assert_eq!(requested_outputs, 1);
                assert_eq!(args, &[Value::Num(4.0)]);
                Box::pin(async {
                    crate::user_functions::test_support::rooted_single_value_sequence(Value::Num(
                        11.0,
                    ))
                })
            }),
        ))
        .expect("call-port scope");

        let request = crate::user_functions::CallableRequest::resolved(
            runmat_types::CallableIdentity::DynamicName(runmat_types::SymbolName(
                "resolved_target".to_string(),
            )),
            runmat_types::CallableFallbackPolicy::ObjectDispatch,
            vec![Value::Num(4.0)],
            1,
        );

        let result = block_on(crate::user_functions::try_call_semantic_descriptor(request));
        assert!(result.is_none());
    }

    #[test]
    fn external_name_runtime_name_resolution_policy_does_not_use_semantic_resolver() {
        let _resolver_guard =
            crate::user_functions::install_semantic_function_resolver(Some(Arc::new(|name| {
                (name == "resolved_target").then_some(45)
            })))
            .expect("call-port scope");
        let _invoker_guard = crate::user_functions::install_semantic_function_invoker(Some(
            Arc::new(|function, args, requested_outputs| {
                assert_eq!(function, 45);
                assert_eq!(requested_outputs, 1);
                assert_eq!(args, &[Value::Num(4.0)]);
                Box::pin(async {
                    crate::user_functions::test_support::rooted_single_value_sequence(Value::Num(
                        11.0,
                    ))
                })
            }),
        ))
        .expect("call-port scope");

        let request = crate::user_functions::CallableRequest::resolved(
            runmat_types::CallableIdentity::ExternalName(runmat_types::QualifiedName(vec![
                runmat_types::SymbolName("resolved_target".to_string()),
            ])),
            runmat_types::CallableFallbackPolicy::RuntimeNameResolution,
            vec![Value::Num(4.0)],
            1,
        );

        let result = block_on(crate::user_functions::try_call_semantic_descriptor(request));
        assert!(result.is_none());
    }

    #[test]
    fn external_boundary_policy_uses_semantic_resolver() {
        let _resolver_guard =
            crate::user_functions::install_semantic_function_resolver(Some(Arc::new(|name| {
                (name == "pkg.resolved_target").then_some(45)
            })))
            .expect("call-port scope");
        let _invoker_guard = crate::user_functions::install_semantic_function_invoker(Some(
            Arc::new(|function, args, requested_outputs| {
                assert_eq!(function, 45);
                assert_eq!(requested_outputs, 1);
                assert_eq!(args, &[Value::Num(4.0)]);
                Box::pin(async {
                    crate::user_functions::test_support::rooted_single_value_sequence(Value::Num(
                        11.0,
                    ))
                })
            }),
        ))
        .expect("call-port scope");

        let request = crate::user_functions::CallableRequest::resolved(
            runmat_types::CallableIdentity::ExternalName(runmat_types::QualifiedName(vec![
                runmat_types::SymbolName("pkg".to_string()),
                runmat_types::SymbolName("resolved_target".to_string()),
            ])),
            runmat_types::CallableFallbackPolicy::ExternalBoundary,
            vec![Value::Num(4.0)],
            1,
        );

        let result = block_on(crate::user_functions::try_call_semantic_descriptor(request))
            .expect("external boundary policy should attempt semantic resolver")
            .expect("semantic invoker should succeed");
        assert_eq!(result.as_slice(), &[Value::Num(11.0)]);
    }

    #[test]
    fn external_boundary_policy_malformed_external_identity_does_not_use_semantic_resolver() {
        let _resolver_guard =
            crate::user_functions::install_semantic_function_resolver(Some(Arc::new(|name| {
                (name == "pkg..resolved_target").then_some(45)
            })))
            .expect("call-port scope");
        let _invoker_guard = crate::user_functions::install_semantic_function_invoker(Some(
            Arc::new(|function, args, requested_outputs| {
                assert_eq!(function, 45);
                assert_eq!(requested_outputs, 1);
                assert_eq!(args, &[Value::Num(4.0)]);
                Box::pin(async {
                    crate::user_functions::test_support::rooted_single_value_sequence(Value::Num(
                        11.0,
                    ))
                })
            }),
        ))
        .expect("call-port scope");

        let request = crate::user_functions::CallableRequest::resolved(
            runmat_types::CallableIdentity::ExternalName(runmat_types::QualifiedName(vec![
                runmat_types::SymbolName("pkg..resolved_target".to_string()),
            ])),
            runmat_types::CallableFallbackPolicy::ExternalBoundary,
            vec![Value::Num(4.0)],
            1,
        );

        let result = block_on(crate::user_functions::try_call_semantic_descriptor(request));
        assert!(result.is_none());
    }

    #[test]
    fn runtime_name_resolution_policy_uses_semantic_resolver_after_object_probe() {
        let _resolver_guard =
            crate::user_functions::install_semantic_function_resolver(Some(Arc::new(|name| {
                (name == "resolved_target").then_some(45)
            })))
            .expect("call-port scope");
        let _invoker_guard = crate::user_functions::install_semantic_function_invoker(Some(
            Arc::new(|function, args, requested_outputs| {
                assert_eq!(function, 45);
                assert_eq!(requested_outputs, 1);
                assert_eq!(args, &[Value::Num(4.0)]);
                Box::pin(async {
                    crate::user_functions::test_support::rooted_single_value_sequence(Value::Num(
                        11.0,
                    ))
                })
            }),
        ))
        .expect("call-port scope");

        let request = crate::user_functions::CallableRequest::resolved(
            runmat_types::CallableIdentity::DynamicName(runmat_types::SymbolName(
                "resolved_target".to_string(),
            )),
            runmat_types::CallableFallbackPolicy::RuntimeNameResolution,
            vec![Value::Num(4.0)],
            1,
        );

        let result = block_on(crate::user_functions::try_call_semantic_descriptor(request))
            .expect("post-object-probe runtime-name policy should attempt semantic resolver")
            .expect("semantic invoker should succeed");
        assert_eq!(result.as_slice(), &[Value::Num(11.0)]);
    }

    #[test]
    fn method_identity_runtime_name_resolution_policy_uses_semantic_resolver() {
        let _resolver_guard =
            crate::user_functions::install_semantic_function_resolver(Some(Arc::new(|name| {
                (name == "resolved_target").then_some(45)
            })))
            .expect("call-port scope");
        let _invoker_guard = crate::user_functions::install_semantic_function_invoker(Some(
            Arc::new(|function, args, requested_outputs| {
                assert_eq!(function, 45);
                assert_eq!(requested_outputs, 1);
                assert_eq!(args, &[Value::Num(4.0)]);
                Box::pin(async {
                    crate::user_functions::test_support::rooted_single_value_sequence(Value::Num(
                        11.0,
                    ))
                })
            }),
        ))
        .expect("call-port scope");

        let request = crate::user_functions::CallableRequest::resolved(
            runmat_types::CallableIdentity::Method(runmat_types::MethodId(
                "resolved_target".to_string(),
            )),
            runmat_types::CallableFallbackPolicy::RuntimeNameResolution,
            vec![Value::Num(4.0)],
            1,
        );

        let result = block_on(crate::user_functions::try_call_semantic_descriptor(request))
            .expect("method runtime-name policy should attempt semantic resolver")
            .expect("semantic invoker should succeed");
        assert_eq!(result.as_slice(), &[Value::Num(11.0)]);
    }

    #[test]
    fn imported_identity_runtime_name_resolution_policy_uses_semantic_resolver() {
        let _resolver_guard =
            crate::user_functions::install_semantic_function_resolver(Some(Arc::new(|name| {
                (name == "Point.origin").then_some(45)
            })))
            .expect("call-port scope");
        let _invoker_guard = crate::user_functions::install_semantic_function_invoker(Some(
            Arc::new(|function, args, requested_outputs| {
                assert_eq!(function, 45);
                assert_eq!(requested_outputs, 1);
                assert_eq!(args, &[Value::Num(4.0)]);
                Box::pin(async {
                    crate::user_functions::test_support::rooted_single_value_sequence(Value::Num(
                        11.0,
                    ))
                })
            }),
        ))
        .expect("call-port scope");

        let request = crate::user_functions::CallableRequest::resolved(
            runmat_types::CallableIdentity::Imported(runmat_types::DefPath {
                package: runmat_types::PackageName("Point".to_string()),
                module: runmat_types::QualifiedName(vec![
                    runmat_types::SymbolName("Point".to_string()),
                    runmat_types::SymbolName("origin".to_string()),
                ]),
                item: vec![runmat_types::DefPathSegment::Function(
                    runmat_types::SymbolName("origin".to_string()),
                )],
            }),
            runmat_types::CallableFallbackPolicy::RuntimeNameResolution,
            vec![Value::Num(4.0)],
            1,
        );

        let result = block_on(crate::user_functions::try_call_semantic_descriptor(request))
            .expect("imported runtime-name policy should attempt semantic resolver")
            .expect("semantic invoker should succeed");
        assert_eq!(result.as_slice(), &[Value::Num(11.0)]);
    }

    #[test]
    fn imported_identity_runtime_name_resolution_policy_rejects_malformed_path_without_semantic_probe(
    ) {
        let resolver_calls = Arc::new(AtomicUsize::new(0));
        let resolver_calls_for_closure = Arc::clone(&resolver_calls);
        let _resolver_guard =
            crate::user_functions::install_semantic_function_resolver(Some(Arc::new(move |_| {
                resolver_calls_for_closure.fetch_add(1, Ordering::Relaxed);
                Some(45)
            })))
            .expect("call-port scope");
        let _invoker_guard = crate::user_functions::install_semantic_function_invoker(Some(
            Arc::new(|function, args, requested_outputs| {
                assert_eq!(function, 45);
                assert_eq!(requested_outputs, 1);
                assert_eq!(args, &[Value::Num(4.0)]);
                Box::pin(async {
                    crate::user_functions::test_support::rooted_single_value_sequence(Value::Num(
                        11.0,
                    ))
                })
            }),
        ))
        .expect("call-port scope");

        let request = crate::user_functions::CallableRequest::resolved(
            runmat_types::CallableIdentity::Imported(runmat_types::DefPath {
                package: runmat_types::PackageName("Point".to_string()),
                module: runmat_types::QualifiedName(vec![
                    runmat_types::SymbolName("Point".to_string()),
                    runmat_types::SymbolName("origin".to_string()),
                ]),
                item: vec![runmat_types::DefPathSegment::Function(
                    runmat_types::SymbolName("other".to_string()),
                )],
            }),
            runmat_types::CallableFallbackPolicy::RuntimeNameResolution,
            vec![Value::Num(4.0)],
            1,
        );

        let result = block_on(crate::user_functions::try_call_semantic_descriptor(request));
        assert!(
            result.is_none(),
            "mismatched imported identity should not attempt semantic resolver"
        );
        assert_eq!(
            resolver_calls.load(Ordering::Relaxed),
            0,
            "malformed imported identity should be rejected before resolver probe"
        );
    }

    #[test]
    fn call_method_fallback_preserves_requested_outputs() {
        let _output_guard =
            crate::output_count::push_output_count(Some(3)).expect("output-count scope");
        let base = Value::Object(runmat_value::ObjectInstance::new(
            "NoSuchMethodClass".to_string(),
        ));
        let result = block_on(
            crate::builtins::introspection::call_method::dispatch_call_method(
                None,
                base.clone(),
                "deal".to_string(),
                crate::execution::RootedValues::new(vec![Value::Num(9.0), Value::Num(10.0)])
                    .expect("ordinary argument owner")
                    .into(),
            ),
        )
        .expect("call_method fallback should succeed");
        match result.value() {
            Value::OutputList(values) => {
                assert_eq!(values.len(), 3);
                assert_eq!(values[0], base);
                assert_eq!(values[1], Value::Num(9.0));
                assert_eq!(values[2], Value::Num(10.0));
            }
            other => {
                panic!("expected output list from multi-output call_method fallback, got {other:?}")
            }
        }
    }

    #[test]
    fn call_method_trims_method_name_for_resolution() {
        let _output_guard =
            crate::output_count::push_output_count(Some(3)).expect("output-count scope");
        let base = Value::Object(runmat_value::ObjectInstance::new(
            "NoSuchMethodClass".to_string(),
        ));
        let result = block_on(
            crate::builtins::introspection::call_method::dispatch_call_method(
                None,
                base.clone(),
                "  deal  ".to_string(),
                crate::execution::RootedValues::new(vec![Value::Num(9.0), Value::Num(10.0)])
                    .expect("ordinary argument owner")
                    .into(),
            ),
        )
        .expect("call_method fallback should succeed after method-name trimming");
        match result.value() {
            Value::OutputList(values) => {
                assert_eq!(values.len(), 3);
                assert_eq!(values[0], base);
                assert_eq!(values[1], Value::Num(9.0));
                assert_eq!(values[2], Value::Num(10.0));
            }
            other => {
                panic!("expected output list from trimmed-name call_method fallback, got {other:?}")
            }
        }
    }

    #[test]
    fn feval_call_method_closure_fast_path_preserves_requested_outputs() {
        let _output_guard =
            crate::output_count::push_output_count(Some(3)).expect("output-count scope");
        let base = Value::Object(runmat_value::ObjectInstance::new(
            "NoSuchMethodClass".to_string(),
        ));
        let closure = Value::Closure(runmat_value::Closure::from_data(
            runmat_value::ClosureData {
                lexical_environment: None,
                function_name: CALL_METHOD_BUILTIN_NAME.into(),
                binding: None,
                source_role: runmat_types::FunctionSourceRole::Unknown,
                source_file: None,
                captures: vec![
                    base.clone(),
                    Value::String("deal".to_string().into()),
                    Value::Num(9.0),
                ]
                .into(),
            },
        ));
        let result = block_on(feval_builtin(closure, vec![Value::Num(10.0)]))
            .expect("feval call_method closure should succeed");
        match result.value() {
            Value::OutputList(values) => {
                assert_eq!(values.len(), 3);
                assert_eq!(values[0], base);
                assert_eq!(values[1], Value::Num(9.0));
                assert_eq!(values[2], Value::Num(10.0));
            }
            other => {
                panic!(
                    "expected output list from feval call_method closure fast path, got {other:?}"
                )
            }
        }
    }

    #[test]
    fn feval_call_method_closure_fast_path_trims_method_name_for_resolution() {
        let _output_guard =
            crate::output_count::push_output_count(Some(3)).expect("output-count scope");
        let base = Value::Object(runmat_value::ObjectInstance::new(
            "NoSuchMethodClass".to_string(),
        ));
        for method in [
            Value::String("  deal  ".into()),
            Value::StringArray(
                runmat_value::StringArray::new(vec!["  deal  ".into()], vec![1, 1]).unwrap(),
            ),
            Value::CharArray(runmat_value::CharArray::new_row("  deal  ")),
            Value::CharArray(
                runmat_value::CharArray::from_unicode_scalars_with_shape(
                    "  deal  ".chars().collect(),
                    vec![1, 8, 1],
                )
                .unwrap(),
            ),
        ] {
            let closure = Value::Closure(runmat_value::Closure::from_data(
                runmat_value::ClosureData {
                    lexical_environment: None,
                    function_name: CALL_METHOD_BUILTIN_NAME.into(),
                    binding: None,
                    source_role: runmat_types::FunctionSourceRole::Unknown,
                    source_file: None,
                    captures: vec![base.clone(), method, Value::Num(9.0)].into(),
                },
            ));
            let result = block_on(feval_builtin(closure, vec![Value::Num(10.0)]))
                .expect("feval call_method closure should succeed after method-name trimming");
            match result.value() {
                Value::OutputList(values) => {
                    assert_eq!(values.len(), 3);
                    assert_eq!(values[0], base);
                    assert_eq!(values[1], Value::Num(9.0));
                    assert_eq!(values[2], Value::Num(10.0));
                }
                other => {
                    panic!(
                    "expected output list from trimmed call_method closure fast path, got {other:?}"
                )
                }
            }
        }
    }

    #[test]
    fn feval_call_method_closure_rejects_nontext_method_capture_with_identifier() {
        for method in [
            Value::Num(1.0),
            Value::Int(runmat_value::IntValue::U8(1)),
            Value::Bool(true),
            Value::StringArray(
                runmat_value::StringArray::new(vec!["a".into(), "b".into()], vec![1, 2]).unwrap(),
            ),
            Value::CharArray(
                runmat_value::CharArray::from_unicode_scalars("ab".chars().collect(), 2, 1)
                    .unwrap(),
            ),
        ] {
            let closure = Value::Closure(runmat_value::Closure::from_data(
                runmat_value::ClosureData {
                    lexical_environment: None,
                    function_name: CALL_METHOD_BUILTIN_NAME.into(),
                    binding: None,
                    source_role: runmat_types::FunctionSourceRole::Unknown,
                    source_file: None,
                    captures: vec![
                        Value::Object(runmat_value::ObjectInstance::new("Point".to_string())),
                        method,
                    ]
                    .into(),
                },
            ));
            let err = block_on(feval_builtin(closure, Vec::new()))
                .expect_err("feval call_method closure should reject nontext method capture");
            assert_eq!(err.identifier(), Some("RunMat:CallMethodNameInvalid"));
        }
    }

    #[test]
    fn call_method_rejects_non_object_receiver_with_identifier() {
        let err = block_on(
            crate::builtins::introspection::call_method::dispatch_call_method(
                None,
                Value::Num(1.0),
                "origin".to_string(),
                crate::execution::RootedValues::new(Vec::new())
                    .expect("ordinary argument owner")
                    .into(),
            ),
        )
        .expect_err("non-object receiver should fail");
        assert_eq!(err.identifier(), Some("RunMat:InvalidObjectDispatch"));
    }

    #[test]
    fn call_method_rejects_empty_method_name_with_identifier() {
        let err = block_on(
            crate::builtins::introspection::call_method::dispatch_call_method(
                None,
                Value::Object(runmat_value::ObjectInstance::new("Point".to_string())),
                "  ".to_string(),
                crate::execution::RootedValues::new(Vec::new())
                    .expect("ordinary argument owner")
                    .into(),
            ),
        )
        .expect_err("empty method name should fail");
        assert_eq!(err.identifier(), Some("RunMat:CallMethodNameInvalid"));
    }

    #[test]
    fn subsref_uses_default_indexing_for_non_object_receiver() {
        let err = block_on(
            crate::builtins::introspection::object_indexing::subsref_builtin(
                Value::Num(1.0),
                standard_subscript(
                    OBJECT_INDEX_PAREN,
                    Value::Cell(runmat_value::CellArray::new(vec![Value::Num(2.0)], 1, 1).unwrap()),
                ),
            ),
        )
        .expect_err("scalar numeric index two should be out of bounds");
        assert_eq!(err.identifier(), Some("RunMat:IndexOutOfBounds"));
    }

    #[test]
    fn subsasgn_uses_default_numeric_growth_for_non_object_receiver() {
        let result = block_on(
            crate::builtins::introspection::object_indexing::dispatch_subsasgn(
                Value::Num(1.0),
                standard_subscript(
                    OBJECT_INDEX_PAREN,
                    Value::Cell(runmat_value::CellArray::new(vec![Value::Num(2.0)], 1, 1).unwrap()),
                ),
                Value::Num(3.0),
            ),
        )
        .expect("numeric subsasgn should use canonical default growth");
        let Value::Tensor(result) = result.value() else {
            panic!("numeric scalar growth must materialize a tensor")
        };
        assert_eq!(result.shape(), &[1, 2]);
        assert_eq!(result.materialize_f64(), vec![1.0, 3.0]);
    }

    #[test]
    fn subsref_missing_protocol_uses_default_object_indexing() {
        let result = block_on(
            crate::builtins::introspection::object_indexing::subsref_builtin(
                Value::Object(runmat_value::ObjectInstance::new(
                    "NoSubsrefProtocolClass".to_string(),
                )),
                standard_subscript(
                    OBJECT_INDEX_PAREN,
                    Value::Cell(runmat_value::CellArray::new(vec![Value::Num(1.0)], 1, 1).unwrap()),
                ),
            ),
        )
        .expect("missing subsref protocol should use default object indexing");
        assert!(matches!(
            result.value(),
            Value::Object(object)
                if object.class_name().display_name() == "NoSubsrefProtocolClass"
        ));
    }

    #[test]
    fn subsasgn_missing_protocol_uses_default_object_array_validation() {
        let err = block_on(
            crate::builtins::introspection::object_indexing::dispatch_subsasgn(
                Value::Object(runmat_value::ObjectInstance::new(
                    "NoSubsasgnProtocolClass".to_string(),
                )),
                standard_subscript(
                    OBJECT_INDEX_PAREN,
                    Value::Cell(runmat_value::CellArray::new(vec![Value::Num(1.0)], 1, 1).unwrap()),
                ),
                Value::Num(3.0),
            ),
        )
        .expect_err("default object assignment must preserve the receiver class");
        assert_eq!(err.identifier(), Some("RunMat:ObjectArrayClassMismatch"));
    }

    #[test]
    fn get_p_rejects_non_object_receiver_with_identifier() {
        let err = block_on(builtins::introspection::dependent_property::get_p_builtin(
            Value::Num(1.0),
        ))
        .expect_err("get.p should reject non-object receiver");
        assert_eq!(err.identifier(), Some("RunMat:GetPReceiverInvalid"));
    }

    #[test]
    fn set_p_rejects_non_object_receiver_with_identifier() {
        let err = block_on(builtins::introspection::dependent_property::set_p_builtin(
            Value::Num(1.0),
            Value::Num(2.0),
        ))
        .expect_err("set.p should reject non-object receiver");
        assert_eq!(err.identifier(), Some("RunMat:SetPReceiverInvalid"));
    }

    #[test]
    fn point_move_rejects_non_object_receiver_with_identifier() {
        let err = block_on(point_move_method(Value::Num(1.0), 2.0, 3.0))
            .expect_err("Point.move should reject non-object receiver");
        assert_eq!(err.identifier(), Some("RunMat:PointMoveReceiverInvalid"));
    }

    #[test]
    fn circle_area_rejects_non_object_receiver_with_identifier() {
        let err = block_on(circle_area_method(Value::Num(1.0)))
            .expect_err("Circle.area should reject non-object receiver");
        assert_eq!(err.identifier(), Some("RunMat:CircleAreaReceiverInvalid"));
    }

    #[test]
    fn overidx_plus_rejects_non_object_receiver_with_identifier() {
        let err = block_on(overidx_plus(Value::Num(1.0), Value::Num(2.0)))
            .expect_err("OverIdx.plus should reject non-object receiver");
        assert_eq!(err.identifier(), Some("RunMat:OverIdxReceiverInvalid"));
    }

    #[test]
    fn overidx_subsref_accepts_standard_substruct_payload() {
        let result = block_on(overidx_subsref(
            Value::Object(runmat_value::ObjectInstance::new("OverIdx".to_string())),
            standard_subscript(
                OBJECT_INDEX_PAREN,
                Value::Cell(runmat_value::CellArray::new(vec![Value::Num(1.0)], 1, 1).unwrap()),
            ),
        ))
        .expect("OverIdx.subsref should accept a standard substruct payload");
        assert_eq!(result, Value::Num(99.0));
    }

    #[test]
    fn overidx_subsasgn_accepts_standard_substruct_payload() {
        let result = block_on(overidx_subsasgn(
            Value::Object(runmat_value::ObjectInstance::new("OverIdx".to_string())),
            standard_subscript(
                OBJECT_INDEX_PAREN,
                Value::Cell(runmat_value::CellArray::new(vec![Value::Num(1.0)], 1, 1).unwrap()),
            ),
            Value::Num(2.0),
        ))
        .expect("OverIdx.subsasgn should accept a standard substruct payload");
        let Value::Object(object) = result.value() else {
            panic!("expected updated OverIdx object")
        };
        assert_eq!(object.property("last"), Some(&Value::Num(2.0)));
    }

    #[test]
    fn feval_object_receiver_uses_default_indexing_when_subsref_is_absent() {
        let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
        let result = block_on(feval_builtin(
            Value::Object(runmat_value::ObjectInstance::new(
                "NoSubsrefProtocolClass".to_string(),
            )),
            vec![Value::Num(1.0)],
        ))
        .expect("an object without subsref should use default scalar indexing");
        assert!(matches!(
            result.value(),
            Value::Object(object)
                if object.class_name().display_name() == "NoSubsrefProtocolClass"
        ));
    }

    #[test]
    fn feval_default_object_indexing_preserves_standard_colon_selector() {
        let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
        let receiver = Value::Object(runmat_value::ObjectInstance::new(
            "NoSubsrefProtocolClass".to_string(),
        ));
        let result = block_on(crate::call_feval_async_with_outputs(
            receiver.clone(),
            &[Value::String(":".into())],
            1,
        ))
        .expect("standard colon retains default scalar-object indexing");
        assert_eq!(result, receiver);
    }

    #[test]
    fn feval_strict_mode_rejects_object_receiver_before_subsref_dispatch() {
        let _compat = crate::compatibility::push_runmat_extensions_enabled(false);
        let err = block_on(feval_builtin(
            Value::Object(runmat_value::ObjectInstance::new(
                "NoSubsrefProtocolClass".to_string(),
            )),
            vec![Value::Num(1.0)],
        ))
        .expect_err("object receiver should require RunMat mode");
        assert_eq!(
            err.identifier(),
            Some("RunMat:compatibility:FevalObjectReceiverExtension")
        );
    }

    #[test]
    fn feval_unsupported_callable_value_errors_with_identifier() {
        let err = block_on(feval_builtin(Value::Num(1.0), vec![Value::Num(2.0)]))
            .expect_err("numeric callable value should fail");
        assert_eq!(
            err.identifier(),
            Some("RunMat:FevalFunctionValueUnsupported")
        );
    }

    #[test]
    fn shape_checked_cell_builder_maps_shape_identifier() {
        let err = super::build_shape_checked_cell(vec![Value::Num(1.0)], 2, 2, "test")
            .expect_err("expected shape mismatch");
        assert_eq!(err.identifier(), Some("RunMat:ShapeMismatch"));
    }

    #[test]
    fn feval_accepts_scalar_string_array_function_name() {
        let handle =
            runmat_value::StringArray::new(vec!["sin".to_string()], vec![1, 1]).expect("sa");
        let result = block_on(feval_builtin(
            Value::StringArray(handle),
            vec![Value::Num(0.0)],
        ))
        .expect("scalar string-array function name should succeed");
        assert_eq!(result, Value::Num(0.0));
    }

    #[test]
    fn feval_rejects_nonscalar_string_array_handle_with_identifier() {
        let handle = runmat_value::StringArray::new(
            vec!["@sin".to_string(), "@cos".to_string()],
            vec![1, 2],
        )
        .expect("sa");
        let err = block_on(feval_builtin(
            Value::StringArray(handle),
            vec![Value::Num(0.0)],
        ))
        .expect_err("nonscalar string-array handle should fail");
        assert_eq!(err.identifier(), Some("RunMat:FevalHandleShapeInvalid"));
    }

    #[test]
    fn call_feval_async_with_outputs_preserves_unresolved_identifier() {
        let err = block_on(super::call_feval_async_with_outputs(
            Value::ExternalFunctionHandle("missing.external".into()),
            &[Value::Num(3.0)],
            1,
        ))
        .expect_err("unresolved external handle should fail");
        assert_eq!(err.identifier(), Some("RunMat:UndefinedFunction"));
    }

    #[test]
    fn addlistener_rejects_non_object_target_with_identifier() {
        for target in [Value::Num(1.0), Value::Int(runmat_value::IntValue::U64(1))] {
            let err = block_on(addlistener_builtin(
                target,
                "Changed".to_string(),
                Value::FunctionHandle("sin".into()),
            ))
            .expect_err("addlistener should reject non-object target");
            assert_eq!(err.identifier(), Some("RunMat:AddListenerTargetInvalid"));
        }
    }

    #[test]
    fn addlistener_rejects_invalid_handle_target_with_identifier() {
        listener_gc_test(|| {
            let target = runmat_value::HandleRef {
                class_name: "EventTarget".into(),
                target: runmat_gc::gc_allocate(Value::Object(runmat_value::ObjectInstance::new(
                    "EventTarget".to_string(),
                )))
                .expect("allocate target"),
                valid: true,
            };
            assert!(set_handle_valid(&target, false));

            let err = block_on(addlistener_builtin(
                Value::HandleObject(target),
                "Changed".to_string(),
                Value::FunctionHandle("sin".into()),
            ))
            .expect_err("addlistener should reject invalid handle target");
            assert_eq!(err.identifier(), Some("RunMat:AddListenerTargetInvalid"));
        });
    }

    #[test]
    fn addlistener_rejects_invalid_callback_before_registration() {
        listener_gc_test(|| {
            let target =
                block_on(new_handle_object_builtin("EventTarget".into())).expect("event target");
            let error = block_on(addlistener_builtin(
                target.clone(),
                "Changed".into(),
                Value::Num(7.0),
            ))
            .unwrap_err();
            assert_eq!(
                error.identifier(),
                runmat_builtins::ADDLISTENER_ERROR_CALLBACK.identifier
            );
            let listener = block_on(addlistener_builtin(
                target,
                "Changed".into(),
                Value::FunctionHandle("disp".into()),
            ))
            .expect("valid listener");
            assert!(matches!(listener, Value::Listener(value) if value.id == 1));
        });
    }

    #[test]
    fn addlistener_preserves_handle_target_when_callback_allocation_collects() {
        listener_gc_test(|| {
            reset_event_registry_for_test();
            struct ConfigGuard(runmat_gc::GcConfig);

            impl Drop for ConfigGuard {
                fn drop(&mut self) {
                    runmat_gc::gc_configure(self.0.clone())
                        .expect("restore GC configuration after listener test");
                }
            }

            let _config_guard = ConfigGuard(runmat_gc::gc_get_config());
            let config = runmat_gc::GcConfig {
                young_generation_size: 64 * 1024 * 1024,
                minor_gc_threshold: 0.35,
                major_gc_threshold: 0.9,
                ..runmat_gc::GcConfig::default()
            };
            runmat_gc::gc_configure(config).expect("configure aggressive periodic minor GC");

            for i in 0..30 {
                let _ = runmat_gc::gc_allocate(Value::Num(i as f64)).expect("seed allocation");
            }

            let target =
                block_on(new_handle_object_builtin("EventTarget".to_string())).expect("target");
            let listener = block_on(addlistener_builtin(
                target,
                "ChangedSoundness".to_string(),
                Value::FunctionHandle("sin".into()),
            ))
            .expect("listener registered");

            let Value::Listener(listener) = listener else {
                panic!("expected listener value");
            };
            let target = runmat_gc::gc_clone_value(&listener.target)
                .expect("listener target should survive construction");
            assert!(matches!(
                target,
                Value::Object(ref object) if object.class_name().is(runmat_types::StaticClassIdentity::new("EventTarget"))
            ));
            assert_eq!(
                runmat_gc::gc_clone_value(&listener.callback)
                    .expect("listener callback should survive construction"),
                Value::FunctionHandle("sin".into())
            );
        });
    }

    #[test]
    fn notify_rejects_non_object_target_with_identifier() {
        let err = block_on(notify_builtin(
            None,
            Value::Num(1.0),
            "Changed".to_string(),
            crate::builtins::common::test_support::builtin_arguments(Vec::new()),
        ))
        .expect_err("notify should reject non-object target");
        assert_eq!(err.identifier(), Some("RunMat:NotifyTargetInvalid"));
    }

    #[test]
    fn addlistener_function_handle_prefers_semantic_identity_when_resolved() {
        listener_gc_test(|| {
            let _resolver_guard =
                crate::user_functions::install_semantic_function_resolver(Some(Arc::new(|name| {
                    (name == "event_callback").then_some(61)
                })))
                .expect("call-port scope");
            let target = block_on(new_handle_object_builtin("EventTarget".to_string()))
                .expect("handle target");
            let listener = block_on(addlistener_builtin(
                target,
                "Changed".to_string(),
                Value::FunctionHandle("event_callback".into()),
            ))
            .expect("listener registered");
            let Value::Listener(listener) = listener else {
                panic!("expected listener value");
            };
            let callback = runmat_gc::gc_clone_value(&listener.callback).expect("callback value");
            assert!(matches!(
                &callback,
                Value::BoundFunctionHandle(handle)
                    if handle.name == "event_callback" && handle.function == 61
            ));
        });
    }

    #[test]
    fn addlistener_external_function_handle_prefers_semantic_identity_when_resolved() {
        listener_gc_test(|| {
            let _resolver_guard =
                crate::user_functions::install_semantic_function_resolver(Some(Arc::new(|name| {
                    (name == "pkg.event_callback").then_some(62)
                })))
                .expect("call-port scope");
            let target = block_on(new_handle_object_builtin("EventTarget".to_string()))
                .expect("handle target");
            let listener = block_on(addlistener_builtin(
                target,
                "Changed".to_string(),
                Value::ExternalFunctionHandle("pkg.event_callback".into()),
            ))
            .expect("listener registered");
            let Value::Listener(listener) = listener else {
                panic!("expected listener value");
            };
            let callback = runmat_gc::gc_clone_value(&listener.callback).expect("callback value");
            assert!(matches!(
                &callback,
                Value::BoundFunctionHandle(handle)
                    if handle.name == "pkg.event_callback" && handle.function == 62
            ));
        });
    }

    #[test]
    fn addlistener_string_handle_prefers_semantic_identity_when_resolved() {
        listener_gc_test(|| {
            let _resolver_guard =
                crate::user_functions::install_semantic_function_resolver(Some(Arc::new(|name| {
                    (name == "event_callback").then_some(63)
                })))
                .expect("call-port scope");
            let target = block_on(new_handle_object_builtin("EventTarget".to_string()))
                .expect("handle target");
            let listener = block_on(addlistener_builtin(
                target,
                "Changed".to_string(),
                Value::String("@event_callback".to_string().into()),
            ))
            .expect("listener registered");
            let Value::Listener(listener) = listener else {
                panic!("expected listener value");
            };
            let callback = runmat_gc::gc_clone_value(&listener.callback).expect("callback value");
            assert!(matches!(
                &callback,
                Value::BoundFunctionHandle(handle)
                    if handle.name == "event_callback" && handle.function == 63
            ));
        });
    }

    #[test]
    fn addlistener_char_handle_prefers_semantic_identity_when_resolved() {
        listener_gc_test(|| {
            let _resolver_guard =
                crate::user_functions::install_semantic_function_resolver(Some(Arc::new(|name| {
                    (name == "event_callback").then_some(64)
                })))
                .expect("call-port scope");
            let target = block_on(new_handle_object_builtin("EventTarget".to_string()))
                .expect("handle target");
            let listener = block_on(addlistener_builtin(
                target,
                "Changed".to_string(),
                Value::CharArray(runmat_value::CharArray::new_row("@event_callback")),
            ))
            .expect("listener registered");
            let Value::Listener(listener) = listener else {
                panic!("expected listener value");
            };
            let callback = runmat_gc::gc_clone_value(&listener.callback).expect("callback value");
            assert!(matches!(
                &callback,
                Value::BoundFunctionHandle(handle)
                    if handle.name == "event_callback" && handle.function == 64
            ));
        });
    }

    #[test]
    fn addlistener_string_array_handle_prefers_semantic_identity_when_resolved() {
        listener_gc_test(|| {
            let _resolver_guard =
                crate::user_functions::install_semantic_function_resolver(Some(Arc::new(|name| {
                    (name == "event_callback").then_some(66)
                })))
                .expect("call-port scope");
            let target = block_on(new_handle_object_builtin("EventTarget".to_string()))
                .expect("handle target");
            let callback =
                runmat_value::StringArray::new(vec!["@event_callback".to_string()], vec![1, 1])
                    .expect("string array");
            let listener = block_on(addlistener_builtin(
                target,
                "Changed".to_string(),
                Value::StringArray(callback),
            ))
            .expect("listener registered");
            let Value::Listener(listener) = listener else {
                panic!("expected listener value");
            };
            let callback = runmat_gc::gc_clone_value(&listener.callback).expect("callback value");
            assert!(matches!(
                &callback,
                Value::BoundFunctionHandle(handle)
                    if handle.name == "event_callback" && handle.function == 66
            ));
        });
    }

    #[test]
    fn addlistener_closure_prefers_embedded_semantic_identity_when_resolved() {
        listener_gc_test(|| {
            let _resolver_guard =
                crate::user_functions::install_semantic_function_resolver(Some(Arc::new(|name| {
                    (name == "event_callback").then_some(65)
                })))
                .expect("call-port scope");
            let target = block_on(new_handle_object_builtin("EventTarget".to_string()))
                .expect("handle target");
            let callback = Value::Closure(runmat_value::Closure::from_data(
                runmat_value::ClosureData {
                    lexical_environment: None,
                    function_name: "event_callback".into(),
                    binding: None,
                    source_role: runmat_types::FunctionSourceRole::Unknown,
                    source_file: None,
                    captures: vec![Value::Num(9.0)].into(),
                },
            ));
            let listener = block_on(addlistener_builtin(target, "Changed".to_string(), callback))
                .expect("listener registered");
            let Value::Listener(listener) = listener else {
                panic!("expected listener value");
            };
            let callback = runmat_gc::gc_clone_value(&listener.callback).expect("callback value");
            assert!(matches!(
                &callback,
                Value::Closure(closure)
                    if closure.lexical_environment.is_none()
                        && closure.function_name == "event_callback"
                        && closure.binding.as_ref()
                            == Some(&runmat_value::BoundFunctionBinding::unavailable(65))
                        && closure.captures.as_slice() == [Value::Num(9.0)]
            ));
        });
    }

    #[test]
    fn notify_semantic_function_handle_uses_semantic_identity() {
        listener_gc_test(|| {
            let runtime = crate::user_functions::test_support::exact_fixture_runtime();
            let _entry = runtime.enter();
            register_event_target_test_class();
            let binding = crate::user_functions::test_support::exact_local_binding(&runtime, 44);
            let _bound =
                crate::user_functions::test_support::install_bound_forwarder(binding.clone());
            let calls = Arc::new(AtomicUsize::new(0));
            let seen_calls = Arc::clone(&calls);
            let _guard = crate::user_functions::install_semantic_function_invoker(Some(Arc::new(
                move |function, args, requested_outputs| {
                    assert_eq!(function, 44);
                    assert_eq!(requested_outputs, 0);
                    assert_eq!(args.len(), 2);
                    assert!(matches!(args[0], Value::HandleObject(_)));
                    assert!(crate::class_registry::is_event_data_value(&args[1]));
                    seen_calls.fetch_add(1, Ordering::SeqCst);
                    Box::pin(async {
                        crate::user_functions::test_support::rooted_single_value_sequence(
                            Value::Num(0.0),
                        )
                    })
                },
            )))
            .expect("call-port scope");
            let target = block_on(new_handle_object_builtin("EventTarget".to_string()))
                .expect("handle target");
            let callback =
                crate::user_functions::test_support::exact_local_value("event_callback", binding);

            block_on(addlistener_builtin(
                target.clone(),
                "Changed".to_string(),
                callback,
            ))
            .expect("listener registered");
            block_on(notify_builtin(
                None,
                target,
                "Changed".to_string(),
                crate::builtins::common::test_support::builtin_arguments(Vec::new()),
            ))
            .expect("notify succeeds");
            assert_eq!(calls.load(Ordering::SeqCst), 1);
        });
    }

    #[test]
    fn notify_after_one_callback_failure_keeps_original_error_without_replay() {
        listener_gc_test(|| {
            let runtime = crate::user_functions::test_support::exact_fixture_runtime();
            let _entry = runtime.enter();
            register_event_target_test_class();
            let binding = crate::user_functions::test_support::exact_local_binding(&runtime, 45);
            let _bound =
                crate::user_functions::test_support::install_bound_forwarder(binding.clone());
            let calls = Arc::new(AtomicUsize::new(0));
            let observed = Arc::clone(&calls);
            let _invoker = crate::user_functions::install_semantic_function_invoker(Some(
                Arc::new(move |function, _args, requested_outputs| {
                    assert_eq!(function, 45);
                    assert_eq!(requested_outputs, 0);
                    let ordinal = observed.fetch_add(1, Ordering::SeqCst);
                    Box::pin(async move {
                        if ordinal == 0 {
                            crate::user_functions::test_support::rooted_single_value_sequence(
                                Value::Num(0.0),
                            )
                        } else {
                            Err(
                                crate::build_runtime_error("callback failed after first delivery")
                                    .with_identifier("RunMat:EventCallbackFailure")
                                    .with_gpu_gather_retry(crate::GpuGatherRetry::Requested)
                                    .build(),
                            )
                        }
                    })
                }),
            ))
            .expect("call-port scope");
            let target = block_on(new_handle_object_builtin("EventTarget".into())).expect("target");
            for _ in 0..2 {
                block_on(addlistener_builtin(
                    target.clone(),
                    "Changed".into(),
                    crate::user_functions::test_support::exact_local_value(
                        "callback",
                        binding.clone(),
                    ),
                ))
                .expect("listener");
            }
            let error = block_on(notify_builtin(
                Some(&runtime),
                target,
                "Changed".into(),
                crate::call::BuiltinArguments::from_borrowed(&[], Some(&runtime)).unwrap(),
            ))
            .unwrap_err();
            assert_eq!(calls.load(Ordering::SeqCst), 2);
            assert_eq!(error.identifier(), Some("RunMat:EventCallbackFailure"));
            assert_eq!(error.gpu_gather_retry, crate::GpuGatherRetry::Never);
        });
    }

    #[test]
    fn dropping_pending_notify_does_not_replay_or_invalidate_listener() {
        listener_gc_test(|| {
            use std::future::Future;
            use std::task::{Context, Poll};

            let runtime = crate::user_functions::test_support::exact_fixture_runtime();
            let _entry = runtime.enter();
            let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
            register_event_target_test_class();
            let binding = crate::user_functions::test_support::exact_local_binding(&runtime, 47);
            let _bound =
                crate::user_functions::test_support::install_bound_forwarder(binding.clone());
            let calls = Arc::new(AtomicUsize::new(0));
            let observed = Arc::clone(&calls);
            let _invoker = crate::user_functions::install_semantic_function_invoker(Some(
                Arc::new(move |function, args, requested_outputs| {
                    assert_eq!(function, 47);
                    assert_eq!(requested_outputs, 0);
                    assert_eq!(args.len(), 3);
                    assert!(matches!(args[1], Value::Cell(_)));
                    assert_eq!(args[2], Value::Num(7.0));
                    observed.fetch_add(1, Ordering::SeqCst);
                    Box::pin(async {
                        std::future::pending::<()>().await;
                        crate::user_functions::test_support::rooted_single_value_sequence(
                            Value::Num(0.0),
                        )
                    })
                }),
            ))
            .expect("call-port scope");
            let target = block_on(new_handle_object_builtin("EventTarget".into())).expect("target");
            let listener = block_on(addlistener_builtin(
                target.clone(),
                "Changed".into(),
                crate::user_functions::test_support::exact_local_value("callback", binding.clone()),
            ))
            .expect("listener");
            block_on(addlistener_builtin(
                target.clone(),
                "Changed".into(),
                crate::user_functions::test_support::exact_local_value("callback", binding),
            ))
            .expect("later listener");
            let data_root = runmat_gc::gc_allocate_rooted(Value::Object(
                runmat_value::ObjectInstance::new("EventPayload".to_string()),
            ))
            .expect("payload root");
            let payload_handle = data_root.handle();
            let payload = Value::Cell(
                runmat_value::CellArray::new(
                    vec![Value::HandleObject(runmat_value::HandleRef {
                        class_name: "EventPayload".into(),
                        target: payload_handle,
                        valid: true,
                    })],
                    1,
                    1,
                )
                .expect("nested payload"),
            );
            let mut notification = Box::pin(notify_builtin(
                Some(&runtime),
                target,
                "Changed".into(),
                crate::call::BuiltinArguments::from_borrowed(
                    &[payload, Value::Num(7.0)],
                    Some(&runtime),
                )
                .unwrap(),
            ));
            let waker = futures::task::noop_waker();
            let mut context = Context::from_waker(&waker);
            assert!(matches!(
                Future::poll(notification.as_mut(), &mut context),
                Poll::Pending
            ));
            assert_eq!(calls.load(Ordering::SeqCst), 1);
            drop(data_root);
            runmat_gc::gc_collect_major().expect("collect while callback is pending");
            assert!(runmat_gc::gc_clone_value(&payload_handle).is_ok());
            drop(notification);
            assert_eq!(calls.load(Ordering::SeqCst), 1);
            let Value::Listener(listener) = listener else {
                panic!("registered listener")
            };
            assert!(listener.valid);
            assert!(runmat_gc::gc_clone_value(&listener.callback).is_ok());
            assert!(runmat_gc::gc_clone_value(&listener.target).is_ok());
        });
    }

    #[test]
    fn notify_char_handle_callback_surfaces_unresolved_identifier() {
        listener_gc_test(|| {
            let _resolver_guard = crate::user_functions::install_semantic_function_resolver(None)
                .expect("call-port scope");
            let target = block_on(new_handle_object_builtin("EventTarget".to_string()))
                .expect("handle target");
            block_on(addlistener_builtin(
                target.clone(),
                "Changed".to_string(),
                Value::CharArray(runmat_value::CharArray::new_row(
                    "@definitely_missing_callback",
                )),
            ))
            .expect("listener registered");
            let err = block_on(notify_builtin(
                None,
                target,
                "Changed".to_string(),
                crate::builtins::common::test_support::builtin_arguments(Vec::new()),
            ))
            .expect_err("unresolved char callback should fail");
            assert_eq!(err.identifier(), Some("RunMat:UndefinedFunction"));
        });
    }

    #[test]
    fn notify_string_array_handle_callback_surfaces_unresolved_identifier() {
        listener_gc_test(|| {
            let _resolver_guard = crate::user_functions::install_semantic_function_resolver(None)
                .expect("call-port scope");
            let target = block_on(new_handle_object_builtin("EventTarget".to_string()))
                .expect("handle target");
            let callback = runmat_value::StringArray::new(
                vec!["@definitely_missing_callback".to_string()],
                vec![1, 1],
            )
            .expect("string array");
            block_on(addlistener_builtin(
                target.clone(),
                "Changed".to_string(),
                Value::StringArray(callback),
            ))
            .expect("listener registered");
            let err = block_on(notify_builtin(
                None,
                target,
                "Changed".to_string(),
                crate::builtins::common::test_support::builtin_arguments(Vec::new()),
            ))
            .expect_err("unresolved string-array callback should fail");
            assert_eq!(err.identifier(), Some("RunMat:UndefinedFunction"));
        });
    }

    #[test]
    fn feval_semantic_handle_honors_zero_requested_outputs() {
        let runtime = crate::user_functions::test_support::exact_fixture_runtime();
        let _entry = runtime.enter();
        let binding = crate::user_functions::test_support::exact_local_binding(&runtime, 46);
        let _bound = crate::user_functions::test_support::install_bound_forwarder(binding.clone());
        let _guard = crate::user_functions::install_semantic_function_invoker(Some(Arc::new(
            |function, args, requested_outputs| {
                assert_eq!(function, 46);
                assert_eq!(requested_outputs, 0);
                assert_eq!(args, &[Value::Num(5.0)]);
                Box::pin(async {
                    crate::execution::RootedValueSequence::new(
                        crate::sequence::ValueSequence::empty(),
                    )
                    .map_err(|error| error.into_runtime_error())
                })
            },
        )))
        .expect("call-port scope");
        let _output_guard =
            crate::output_count::push_output_count(Some(0)).expect("output-count scope");
        let handle =
            crate::user_functions::test_support::exact_local_value("function_target", binding);

        let result = block_on(feval_builtin(handle, vec![Value::Num(5.0)]))
            .expect("semantic function handle feval succeeds");
        assert_eq!(result, Value::output_list(Vec::new()));
    }

    #[test]
    fn feval_semantic_handle_honors_multi_requested_outputs() {
        let runtime = crate::user_functions::test_support::exact_fixture_runtime();
        let _entry = runtime.enter();
        let binding = crate::user_functions::test_support::exact_local_binding(&runtime, 47);
        let _bound = crate::user_functions::test_support::install_bound_forwarder(binding.clone());
        let _guard = crate::user_functions::install_semantic_function_invoker(Some(Arc::new(
            |function, args, requested_outputs| {
                assert_eq!(function, 47);
                assert_eq!(requested_outputs, 2);
                assert_eq!(args, &[Value::Num(6.0)]);
                Box::pin(async {
                    crate::sequence::ValueSequence::comma_separated(vec![
                        Value::Num(1.0),
                        Value::Num(2.0),
                    ])
                    .map_err(crate::sequence::sequence_error_to_runtime)
                    .and_then(|sequence| {
                        crate::execution::RootedValueSequence::new(sequence)
                            .map_err(|error| error.into_runtime_error())
                    })
                })
            },
        )))
        .expect("call-port scope");
        let _output_guard =
            crate::output_count::push_output_count(Some(2)).expect("output-count scope");
        let handle =
            crate::user_functions::test_support::exact_local_value("function_target", binding);

        let result = block_on(feval_builtin(handle, vec![Value::Num(6.0)]))
            .expect("semantic function handle feval succeeds");
        assert_eq!(
            result,
            Value::output_list(vec![Value::Num(1.0), Value::Num(2.0)])
        );
    }

    #[test]
    fn feval_semantic_closure_errors_when_semantic_invoker_unavailable() {
        let _guard =
            crate::user_functions::clear_semantic_function_invoker().expect("call-port scope");
        let closure = Value::Closure(runmat_value::Closure::from_data(
            runmat_value::ClosureData {
                lexical_environment: None,
                function_name: "function_target".into(),
                binding: Some(runmat_value::BoundFunctionBinding::unavailable(9044)),
                source_role: runmat_types::FunctionSourceRole::Unknown,
                source_file: None,
                captures: vec![Value::Num(1.0)].into(),
            },
        ));

        let err = block_on(feval_builtin(closure, vec![Value::Num(2.0)]))
            .expect_err("semantic closure should not fall back to name-based dispatch");
        assert_eq!(err.identifier(), Some("RunMat:UndefinedSemanticFunction"));
        assert!(
            err.message()
                .contains("'function_target' could not invoke semantic function 9044"),
            "unexpected error: {err:?}"
        );
    }
}

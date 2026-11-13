use crate::sequence::ResolveValueSequence;
use crate::{build_runtime_error, make_cell_with_shape, RuntimeError};
use runmat_accelerate_api::GpuTensorHandle;
use runmat_builtins::builtin_functions;

use runmat_types::ClassIdentity;
use runmat_value::Value;
use std::cell::RefCell;

#[path = "dispatcher/class_access_identity.rs"]
mod class_access_identity;
pub use class_access_identity::ClassAccessIdentity;

#[path = "dispatcher/constructor.rs"]
mod constructor;
pub(crate) mod direct_call_observation;
pub use constructor::{
    call_bound_constructor_sequence_in_context,
    call_portable_prepared_constructor_sequence_in_context,
    call_portable_prepared_constructor_sequence_with_argument_count_in_context,
    call_prepared_constructor_sequence_in_context,
    call_prepared_constructor_sequence_with_argument_count_in_context,
    call_rooted_bound_constructor_sequence_in_context, PreparedConstructorInvoke,
};
use constructor::{ConstructorArguments, ConstructorInvocationTarget};
#[path = "dispatcher/cell_alias.rs"]
mod cell_alias;
#[path = "dispatcher/managed_gather.rs"]
mod managed_gather;
#[path = "dispatcher/owned_gather.rs"]
mod owned_gather;
use crate::execution::structure::replacement as structure_alias;
mod constructor_normalization;
#[path = "dispatcher/owned_gpu.rs"]
mod owned_gpu;
mod provenance;
#[path = "dispatcher/workspace_effect.rs"]
mod workspace_effect;

pub use owned_gather::gather_if_needed_owned_async;

#[cfg(test)]
#[path = "dispatcher/output_projection_tests.rs"]
mod output_projection_tests;
#[cfg(test)]
mod provenance_tests;

thread_local! {
    static CLASS_ACCESS_CONTEXT: RefCell<Option<ClassIdentity>> = const { RefCell::new(None) };
}

pub(crate) fn resolve_single_sequence(
    sequence: crate::execution::RootedValueSequence,
) -> Result<crate::execution::RootedValue, RuntimeError> {
    sequence.try_map_owned(
        |sequence| {
            let values = sequence.resolve(
                runmat_types::SequenceUse::RequireSingle,
                crate::sequence::SequenceResolutionContext::default(),
            )?;
            Ok(crate::sequence::into_first_output(values)?
                .expect("one required output was resolved"))
        },
        |error| error.into_runtime_error(),
    )
}

#[cfg(target_arch = "wasm32")]
fn ensure_wasm_builtins_registered() {
    crate::builtins::wasm_registry::register_all();
}

#[cfg(not(target_arch = "wasm32"))]
fn ensure_wasm_builtins_registered() {}

pub struct ClassAccessContextGuard {
    previous: Option<ClassIdentity>,
    projection: Option<crate::context::call_projection::CallProjectionGuard>,
    context: Option<std::rc::Rc<crate::context::RuntimeContextState>>,
}

impl Drop for ClassAccessContextGuard {
    fn drop(&mut self) {
        if self.projection.take().is_some() {
            return;
        }
        let previous = self.previous.take();
        if let Some(context) = &self.context {
            context.call.borrow_mut().class_access = previous;
        } else {
            CLASS_ACCESS_CONTEXT.with(|slot| {
                *slot.borrow_mut() = previous;
            });
        }
    }
}

pub fn push_class_access_context(
    class_name: Option<ClassIdentity>,
) -> Result<ClassAccessContextGuard, RuntimeError> {
    if let Some(runtime) = crate::context::legacy::active() {
        return Ok(ClassAccessContextGuard {
            previous: None,
            projection: Some(crate::context::call_projection::push_class_access(
                runtime.call_projection(),
                class_name,
            )?),
            context: None,
        });
    }
    let context =
        crate::context::legacy::active().map(|context| std::rc::Rc::clone(context.state()));
    let previous = if let Some(context) = &context {
        std::mem::replace(&mut context.call.borrow_mut().class_access, class_name)
    } else {
        CLASS_ACCESS_CONTEXT.with(|slot| std::mem::replace(&mut *slot.borrow_mut(), class_name))
    };
    Ok(ClassAccessContextGuard {
        previous,
        projection: None,
        context,
    })
}

fn current_class_access_context() -> Option<ClassAccessIdentity> {
    if let Some(context) = crate::context::legacy::active() {
        return context
            .call_projection()
            .borrow()
            .class_access()
            .unwrap_or_else(|| {
                context
                    .state()
                    .call
                    .borrow()
                    .class_access
                    .clone()
                    .map(ClassAccessIdentity::from)
            });
    }
    CLASS_ACCESS_CONTEXT.with(|slot| slot.borrow().clone().map(ClassAccessIdentity::from))
}

pub fn class_access_context() -> Option<ClassAccessIdentity> {
    current_class_access_context()
}

/// Return `true` when the passed value is a GPU-resident tensor handle.
pub fn is_gpu_value(value: &Value) -> bool {
    matches!(value, Value::GpuTensor(_))
}

/// Returns true when the value (or nested elements) contains any GPU-resident tensors.
pub fn value_contains_gpu(value: &Value) -> bool {
    let mut found = false;
    let inspected = try_visit_gpu_handles(value, &mut |_| found = true);
    found || inspected.is_err()
}

/// Convert GPU-resident values to host tensors when an acceleration provider exists.
/// Non-GPU inputs are passed through unchanged.
pub async fn gather_if_needed_async(value: &Value) -> Result<Value, RuntimeError> {
    let runtime = crate::context::legacy::active();
    gather_if_needed_in_context_async(value, runtime.as_ref()).await
}

pub(crate) async fn gather_if_needed_in_context_async(
    value: &Value,
    runtime: Option<&crate::context::RuntimeContext>,
) -> Result<Value, RuntimeError> {
    if managed_gather::is_managed(runtime)? {
        return managed_gather::borrowed(value, runtime.expect("managed gather context")).await;
    }
    try_visit_gathered_gpu_handles(value, &mut |_| {})?;
    let future = gather_if_needed_async_impl(value);
    if let Some(runtime) = runtime {
        poll_gather_in_context(runtime, future).await
    } else {
        future.await
    }
}

/// Stack-pinned poll scope: no second erased future/metadata allocation.
async fn poll_gather_in_context<F: std::future::Future>(
    runtime: &crate::context::RuntimeContext,
    future: F,
) -> F::Output {
    let mut future = std::pin::pin!(future);
    std::future::poll_fn(|cx| {
        let _scope = runtime.enter();
        future.as_mut().poll(cx)
    })
    .await
}

/// Preserve the admitted argument owner when no transfer is necessary. A
/// resident batch gets one admitted final slot collection, with the original
/// roots retained across every download and successor publication.
pub(crate) async fn gather_builtin_arguments_async(
    values: crate::call::BuiltinArguments,
    runtime: Option<&crate::context::RuntimeContext>,
) -> Result<crate::call::BuiltinArguments, RuntimeError> {
    let mut resident = false;
    for value in &values {
        try_visit_gathered_gpu_handles(value, &mut |_| resident = true)?;
    }
    if !resident {
        return Ok(values);
    }
    values
        .try_map_preserving_roots_async(values.len(), runtime, async |index| {
            gather_if_needed_in_context_async(&values[index], runtime).await
        })
        .await
}

fn gather_if_needed_async_impl<'a>(
    value: &'a Value,
) -> std::pin::Pin<Box<dyn std::future::Future<Output = Result<Value, RuntimeError>> + 'a>> {
    Box::pin(async move {
        match value {
            Value::GpuTensor(handle) => {
                let readback = runmat_accelerate_api::numeric_readback_for_handle(handle).map_err(
                    |source| {
                        crate::runtime_error::with_provider_execution_source(
                            build_runtime_error(
                                "gather: exact source provider is unavailable or not authorized",
                            )
                            .with_identifier("RunMat:gather:ProviderUnavailable")
                            .with_gpu_gather_retry(crate::GpuGatherRetry::Never)
                            .build(),
                            source.into(),
                        )
                    },
                )?;
                let host = crate::builtins::common::gpu_helpers::download_numeric_readback_preserving_residency_async(
                    &readback, None,
                    crate::builtins::common::gpu_helpers::NumericDownloadDiagnostic::Gather,
                ).await?;
                crate::builtins::common::gpu_helpers::value_from_numeric_download(
                    host,
                    runmat_accelerate_api::handle_is_logical(handle),
                )
                .map_err(|error| {
                    build_runtime_error(format!("gather: {error}"))
                        .with_identifier("RunMat:gpu:ProviderPayloadMismatch")
                        .with_gpu_gather_retry(crate::GpuGatherRetry::Never)
                        .with_source(ProviderDownloadPayloadMismatch)
                        .build()
                })
            }
            Value::Cell(ca) => {
                let mut gathered = Vec::with_capacity(ca.data().len());
                for ptr in ca.data() {
                    gathered.push(gather_if_needed_async_impl(ptr).await?);
                }
                make_cell_with_shape(gathered, ca.shape().to_vec()).map_err(|err| {
                    build_runtime_error(format!("gather: {err}"))
                        .with_identifier("RunMat:gather:CellShapeError")
                        .build()
                })
            }
            Value::Struct(sv) => {
                if !sv.fields.values().any(owned_gpu::has_transformable_gpu) {
                    return Ok(Value::Struct(sv.clone()));
                }
                let context = crate::context::legacy::active();
                let prepared = structure_alias::record(sv, context.as_ref())?;
                let _roots = prepared.1;
                let mut gathered = prepared.0;
                for value in gathered
                    .fields
                    .try_values_mut_unique()
                    .expect("new admitted map is unique")
                {
                    *value =
                        owned_gather::gather_if_needed_owned_async(value.clone(), None).await?;
                }
                Ok(Value::Struct(gathered))
            }
            Value::StructArray(array) => {
                if !array
                    .column_owners()
                    .any(|(_, values)| values.iter().any(owned_gpu::has_transformable_gpu))
                {
                    return Ok(Value::StructArray(array.clone()));
                }
                let context = crate::context::legacy::active();
                let prepared = structure_alias::array(array, context.as_ref())?;
                let _roots = prepared.1;
                let mut gathered = prepared.0;
                for name in array.field_names() {
                    for value in gathered
                        .try_column_mut_unique(name)
                        .expect("new admitted column is unique")
                    {
                        *value =
                            owned_gather::gather_if_needed_owned_async(value.clone(), None).await?;
                    }
                }
                Ok(Value::StructArray(gathered))
            }
            Value::Object(obj) => {
                if !obj
                    .properties()
                    .values()
                    .any(owned_gpu::has_transformable_gpu)
                {
                    return Ok(Value::Object(obj.clone()));
                }
                let context = crate::context::legacy::active();
                let prepared =
                    crate::execution::object::replacement::copy_generation(obj, context.as_ref())?;
                let _roots = prepared.1;
                let mut cloned = prepared.0;
                for (name, value) in obj.properties() {
                    let gathered = gather_if_needed_async_impl(value).await?;
                    cloned
                        .try_replace_property_unique(name, gathered)
                        .map_err(|_| {
                            crate::runtime_error::semantic_error(
                                "RunMat:gather:Object",
                                "private object destination lost its unique property slot",
                            )
                        })?;
                }
                Ok(Value::Object(cloned))
            }
            Value::ObjectArray(array) => {
                if !array.data().iter().any(owned_gpu::has_transformable_gpu) {
                    return Ok(Value::ObjectArray(array.clone()));
                }
                let context = crate::context::legacy::active();
                let prepared =
                    crate::execution::object_array::copy_source(array, context.as_ref())?
                        .into_value_and_lease();
                let _roots = prepared.1;
                let Value::ObjectArray(mut gathered) = prepared.0 else {
                    unreachable!("canonical object array output")
                };
                for index in 0..gathered.len() {
                    let child = owned_gather::gather_if_needed_owned_async(
                        gathered.data()[index].clone(),
                        None,
                    )
                    .await?;
                    gathered
                        .try_replace_linear_unique(index, child)
                        .map_err(|_| {
                            crate::runtime_error::semantic_error(
                                "ObjectArrayConstruction",
                                "gather changed object-array class or storage kind",
                            )
                        })?;
                }
                Ok(Value::ObjectArray(gathered))
            }
            Value::Closure(closure) => {
                runmat_value::validate_no_transient_sequence(value).map_err(|_| {
                    crate::runtime_error::semantic_error(
                        "TransientSequenceBelowDispatch",
                        "legacy output-list carrier reached scalar gather dispatch",
                    )
                })?;
                if !closure
                    .captures
                    .iter()
                    .any(owned_gpu::has_transformable_gpu)
                {
                    return Ok(Value::Closure(closure.clone()));
                }
                let mut captures = Vec::with_capacity(closure.captures.len());
                for value in &closure.captures {
                    captures.push(gather_if_needed_async_impl(value).await?);
                }
                let mut cloned = closure.clone().into_data();
                cloned.captures = runmat_value::OutputListStorage::shared_unbounded(captures);
                Ok(Value::Closure(runmat_value::Closure::from_data(cloned)))
            }
            Value::OutputList(_) => Err(crate::runtime_error::semantic_error(
                "TransientSequenceBelowDispatch",
                "legacy output-list carrier reached scalar gather dispatch",
            )),
            other => Ok(other.clone()),
        }
    })
}

/// A provider-owned download returned bytes that violate the admitted handle.
/// The public diagnostic remains a RuntimeError; placement can inspect this
/// source without treating unrelated host allocation failures as GPU failures.
#[derive(Debug)]
pub(crate) struct ProviderDownloadPayloadMismatch;

impl std::fmt::Display for ProviderDownloadPayloadMismatch {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str("provider download payload violates its admitted descriptor")
    }
}

impl std::error::Error for ProviderDownloadPayloadMismatch {}

#[cfg(not(target_arch = "wasm32"))]
pub fn gather_if_needed(value: &Value) -> Result<Value, RuntimeError> {
    futures::executor::block_on(gather_if_needed_async(value))
}

#[cfg(target_arch = "wasm32")]
pub fn gather_if_needed(_value: &Value) -> Result<Value, RuntimeError> {
    Err(
        build_runtime_error("gather: synchronous gather is unavailable on wasm")
            .with_identifier("RunMat:gather:UnavailableOnWasm")
            .build(),
    )
}

/// Call a registered language builtin by name.
/// Selects one implementation by the current binding authority. Supported
/// argument forms are admitted by that implementation, not by replaying errors.
/// Returns an error if no builtin with that name and compatible arguments is found.
pub fn call_builtin(
    name: &str,
    args: &[Value],
) -> Result<crate::execution::RootedValue, RuntimeError> {
    futures::executor::block_on(call_builtin_async(name, args))
}

#[derive(Clone, Copy)]
struct ResolvedBuiltinName<'a> {
    requested: &'a str,
    canonical: &'a str,
}

mod instance_execution;
mod preparation;
mod retained_execution;
mod selected_execution;
mod selection;
mod sequence_api;
use preparation::SelectedBindingOrigin;
use selected_execution::SelectedImplementation;
pub use selection::{
    observe_builtin_dispatch_target, ObservedBuiltinDispatchIdentity, ObservedBuiltinDispatchTarget,
};
pub use sequence_api::{
    call_builtin_sequence_async, call_builtin_sequence_async_in_context,
    call_builtin_sequence_async_with_outputs, call_builtin_sequence_async_with_outputs_in_context,
};

impl<'a> ResolvedBuiltinName<'a> {
    fn resolve(requested: &'a str) -> Self {
        Self {
            requested,
            canonical: runmat_builtins::canonical_builtin_name(requested).unwrap_or(requested),
        }
    }
}

#[async_recursion::async_recursion(?Send)]
async fn call_builtin_sequence_async_impl(
    name: &str,
    args: &[Value],
    output_count: Option<usize>,
    context: Option<crate::context::RuntimeContext>,
    observation: Option<std::rc::Rc<direct_call_observation::CallObservation>>,
) -> Result<crate::execution::RootedValueSequence, RuntimeError> {
    try_call_builtin_sequence_async_impl(name, args, output_count, context, observation)
        .await?
        .ok_or_else(|| undefined_builtin_error(name))
}

/// The language diagnostic for a name that had no selected builtin target.
pub fn undefined_builtin_error(name: &str) -> RuntimeError {
    let resolved_name = ResolvedBuiltinName::resolve(name);
    build_runtime_error(format!("Undefined function: {}", resolved_name.requested))
        .with_identifier("RunMat:UndefinedFunction")
        .with_failure_origin(crate::runtime_error::RuntimeFailureOrigin::LanguageEvaluation)
        .build()
}

#[async_recursion::async_recursion(?Send)]
async fn try_call_builtin_sequence_async_impl(
    name: &str,
    args: &[Value],
    output_count: Option<usize>,
    context: Option<crate::context::RuntimeContext>,
    observation: Option<std::rc::Rc<direct_call_observation::CallObservation>>,
) -> Result<Option<crate::execution::RootedValueSequence>, RuntimeError> {
    let args = crate::call::BuiltinArguments::from_borrowed(args, context.as_ref())?;
    try_call_retained_builtin_sequence_async(name, args, output_count, context, observation).await
}

/// Dispatch the original argument owner through the same selection policies as
/// embedding calls. Captures and routing prefixes remain logical segments.
#[derive(Clone, Copy, PartialEq, Eq)]
enum RetainedDispatchPolicy {
    Normal,
    RegisteredOnly,
}

fn try_call_retained_builtin_sequence_async<'a>(
    name: &'a str,
    args: crate::call::BuiltinArguments,
    output_count: Option<usize>,
    context: Option<crate::context::RuntimeContext>,
    observation: Option<std::rc::Rc<direct_call_observation::CallObservation>>,
) -> impl std::future::Future<
    Output = Result<Option<crate::execution::RootedValueSequence>, RuntimeError>,
> + 'a {
    try_call_retained_builtin_sequence_with_policy(
        name,
        args,
        output_count,
        context,
        observation,
        RetainedDispatchPolicy::Normal,
    )
}

fn try_call_retained_builtin_sequence_with_policy<'a>(
    name: &'a str,
    args: crate::call::BuiltinArguments,
    output_count: Option<usize>,
    context: Option<crate::context::RuntimeContext>,
    observation: Option<std::rc::Rc<direct_call_observation::CallObservation>>,
    policy: RetainedDispatchPolicy,
) -> impl std::future::Future<
    Output = Result<Option<crate::execution::RootedValueSequence>, RuntimeError>,
> + 'a {
    use futures::TryFutureExt;
    // Selection stays on first poll. Its data-only frame returns before the
    // selected route is polled, using the existing single dispatch allocation.
    futures::future::lazy(move |_| {
        retained_execution::select(name, args, output_count, context, observation, policy)
    })
    .try_flatten()
}

pub(crate) async fn try_call_builtin_sequence_async_with_outputs(
    name: &str,
    args: &[Value],
    output_count: usize,
    context: Option<&crate::context::RuntimeContext>,
) -> Result<Option<crate::execution::RootedValueSequence>, RuntimeError> {
    let args = crate::call::BuiltinArguments::from_borrowed(args, context)?;
    try_call_retained_builtin_sequence_with_outputs(name, args, output_count, context).await
}

pub(crate) async fn try_call_retained_builtin_sequence_with_outputs(
    name: &str,
    args: crate::call::BuiltinArguments,
    output_count: usize,
    context: Option<&crate::context::RuntimeContext>,
) -> Result<Option<crate::execution::RootedValueSequence>, RuntimeError> {
    try_call_retained_builtin_sequence_with_outputs_policy(
        name,
        args,
        output_count,
        context,
        RetainedDispatchPolicy::Normal,
    )
    .await
}

/// Explicit registered-builtin bypass: no semantic, method, distributed or
/// constructor selection. Preparation, execution, compatibility and effect
/// observation remain the same canonical pipeline as normal dispatch.
pub(crate) async fn call_retained_registered_builtin_sequence_with_outputs(
    name: &str,
    args: crate::call::BuiltinArguments,
    output_count: usize,
    context: Option<&crate::context::RuntimeContext>,
) -> Result<crate::execution::RootedValueSequence, RuntimeError> {
    try_call_retained_builtin_sequence_with_outputs_policy(
        name,
        args,
        output_count,
        context,
        RetainedDispatchPolicy::RegisteredOnly,
    )
    .await?
    .ok_or_else(|| undefined_builtin_error(name))
}

async fn try_call_retained_builtin_sequence_with_outputs_policy(
    name: &str,
    args: crate::call::BuiltinArguments,
    output_count: usize,
    context: Option<&crate::context::RuntimeContext>,
    policy: RetainedDispatchPolicy,
) -> Result<Option<crate::execution::RootedValueSequence>, RuntimeError> {
    let fence = workspace_effect::PostCallWorkspaceEffectFence::capture(context);
    let observation =
        direct_call_observation::CallObservation::capture(context).map(std::rc::Rc::new);
    let context = context.cloned();
    let result = match &context {
        Some(context) => {
            context
                .scope(try_call_retained_builtin_sequence_with_policy(
                    name,
                    args,
                    Some(output_count),
                    Some(context.clone()),
                    observation.clone(),
                    policy,
                ))
                .await
        }
        None => {
            try_call_retained_builtin_sequence_with_policy(
                name,
                args,
                Some(output_count),
                None,
                None,
                policy,
            )
            .await
        }
    };
    let (result, outcome) = fence.finish_optional_owned_sequence_observed(result);
    if let Some(observation) = observation {
        observation.finish_completed(outcome);
    }
    result
}

/// Execute the selected builtin once, distinguishing no selected target from
/// an implementation error, including an implementation's UndefinedFunction.
pub async fn try_call_builtin_sequence_async_with_outputs_in_context(
    context: &crate::context::RuntimeContext,
    name: &str,
    args: &[Value],
    output_count: usize,
) -> Result<Option<crate::execution::RootedValueSequence>, RuntimeError> {
    try_call_builtin_sequence_async_with_outputs(name, args, output_count, Some(context)).await
}

async fn try_distributed_builtin(
    name: &str,
    args: &crate::call::BuiltinArguments,
    output_count: Option<usize>,
    runtime: Option<&crate::context::RuntimeContext>,
) -> Result<Option<crate::execution::RootedValueSequence>, RuntimeError> {
    if !args
        .iter()
        .any(|value| matches!(value, Value::Distributed(_)))
    {
        return Ok(None);
    }
    let entry = runmat_builtins::builtin_catalog_entry_by_name(name).ok_or_else(|| {
        build_runtime_error(format!(
            "{name}: distributed arguments require a canonical builtin placement contract"
        ))
        .with_identifier("RunMat:parallel:DistributedBuiltinUnsupported")
        .build()
    })?;
    match entry.placement.distributed {
        runmat_builtins::BuiltinDistributedPolicy::Unsupported => Err(build_runtime_error(
            format!("{name}: distributed inputs are not supported by this builtin"),
        )
        .with_identifier("RunMat:parallel:DistributedBuiltinUnsupported")
        .build()),
        runmat_builtins::BuiltinDistributedPolicy::InspectHandles => Ok(None),
        runmat_builtins::BuiltinDistributedPolicy::MaterializeArguments => {
            let (context, service) = distributed_service(name, runtime)?;
            let materialized = args
                .try_map_mixed_async(args.len(), Some(&context), |index| {
                    let argument = &args[index];
                    let context = &context;
                    let service = &service;
                    async move {
                        if let Value::Distributed(handle) = argument {
                            crate::parallel::lease::validate_distributed(context, handle)?;
                            service
                                .materialize(handle.clone())
                                .await
                                .map(crate::execution::ArgumentProjection::Rooted)
                        } else {
                            Ok(crate::execution::ArgumentProjection::Source(
                                argument.clone(),
                            ))
                        }
                    }
                })
                .await?;
            try_call_retained_builtin_sequence_async(
                name,
                materialized,
                output_count,
                Some(context),
                None,
            )
            .await?
            .ok_or_else(|| undefined_builtin_error(name))
            .map(Some)
        }
        runmat_builtins::BuiltinDistributedPolicy::MapUnary
        | runmat_builtins::BuiltinDistributedPolicy::MapUnaryConstrained(_)
        | runmat_builtins::BuiltinDistributedPolicy::MapUnaryOrScalarLikePrototype
        | runmat_builtins::BuiltinDistributedPolicy::ScalarLikePrototype => {
            let (context, service) = distributed_service(name, runtime)?;
            for argument in args {
                if let Value::Distributed(handle) = argument {
                    crate::parallel::lease::validate_distributed(&context, handle)?;
                }
            }
            let requested_outputs = output_count.unwrap_or(1);
            let literals = distributed_literal_context(args.view());
            let inference = runmat_builtins::infer_partition_local_call(
                entry,
                &runmat_types::CallRequest {
                    arguments: args.iter().map(crate::value_fact::value_fact).collect(),
                    literals: literals.clone(),
                    outputs: runmat_types::OutputSelection::new(match requested_outputs {
                        0 => runmat_types::RequestedOutputCount::Zero,
                        1 => runmat_types::RequestedOutputCount::One,
                        count => runmat_types::RequestedOutputCount::Exactly(count),
                    }),
                },
            );
            if !inference.diagnostics.is_empty() {
                return Err(build_runtime_error(format!(
                    "{name}: distributed execution does not satisfy the builtin input contract"
                ))
                .with_identifier("RunMat:parallel:DistributedInputContract")
                .build());
            }
            if inference.outputs.len() != requested_outputs {
                return Err(build_runtime_error(format!(
                    "{name}: distributed execution described {} outputs for {requested_outputs} requested outputs",
                    inference.outputs.len()
                ))
                .with_identifier("RunMat:parallel:DistributedOutputContract")
                .build());
            }
            let result = service
                .invoke(crate::context::RuntimeDistributedCallRequest {
                    builtin: runmat_types::BuiltinId(name.into()),
                    arguments: args.fork(),
                    literals,
                    requested_outputs,
                    outputs: inference.outputs,
                    invocation: context.service_ports().collective().map_or(
                        crate::context::RuntimeDistributedInvocation::Client,
                        |collective| {
                            crate::context::RuntimeDistributedInvocation::Worker(
                                collective.context().clone(),
                            )
                        },
                    ),
                })
                .await?;
            crate::execution::RootedValueSequence::new(result)
                .map(Some)
                .map_err(|error| error.into_runtime_error())
        }
    }
}

fn distributed_literal_context(
    args: crate::call::FunctionArgumentView<'_>,
) -> runmat_types::LiteralContext {
    runmat_types::LiteralContext::new(
        args.iter()
            .map(|value| match value {
                Value::String(value) => value
                    .as_str()
                    .map(|text| runmat_types::LiteralValue::String(text.to_owned()))
                    .unwrap_or(runmat_types::LiteralValue::Unknown),
                Value::StringArray(value) if value.data().len() == 1 => value.data()[0]
                    .as_str()
                    .map(|text| runmat_types::LiteralValue::String(text.to_owned()))
                    .unwrap_or(runmat_types::LiteralValue::Unknown),
                Value::CharArray(value) if value.rows == 1 => {
                    // Inference cannot publish an argument error; the selected
                    // builtin decodes and validates the actual text.
                    match value.try_row_string() {
                        Ok(Some(text)) => runmat_types::LiteralValue::Character(text),
                        Ok(None) | Err(_) => runmat_types::LiteralValue::Unknown,
                    }
                }
                _ => runmat_types::LiteralValue::Unknown,
            })
            .collect(),
    )
}

fn distributed_service(
    name: &str,
    runtime: Option<&crate::context::RuntimeContext>,
) -> Result<
    (
        crate::context::RuntimeContext,
        std::rc::Rc<dyn crate::context::RuntimeDistributedService>,
    ),
    RuntimeError,
> {
    let context = runtime.cloned().ok_or_else(|| {
        build_runtime_error(format!(
            "{name}: distributed arguments require an active RunMat execution context"
        ))
        .with_identifier("RunMat:parallel:DistributedRuntimeUnavailable")
        .build()
    })?;
    let service = context
        .service_ports()
        .require_distributed("distributed builtin")
        .map_err(|error| {
            build_runtime_error(error.to_string())
                .with_identifier("RunMat:parallel:DistributedRuntimeUnavailable")
                .build()
        })?
        .clone();
    Ok((context, service))
}

pub(crate) fn compatibility_checked_builtin_result(
    name: &str,
    args: &[Value],
    result: crate::execution::RootedValue,
) -> Result<crate::execution::RootedValue, RuntimeError> {
    result.try_map_owned(
        |mut value| {
            crate::compatibility::ensure_value_compatible(&value, name)?;
            propagate_gpu_provenance(args, &mut value)?;
            Ok(value)
        },
        |error| error.into_runtime_error(),
    )
}

fn compatibility_checked_builtin_sequence(
    name: &str,
    args: crate::call::FunctionArgumentView<'_>,
    result: crate::execution::RootedValueSequence,
) -> Result<crate::execution::RootedValueSequence, RuntimeError> {
    let explicit = explicit_gpu_provenance(args)?;
    result.try_map_owned(
        |sequence| {
            sequence.try_map(
                |mut value| {
                    crate::compatibility::ensure_value_compatible(&value, name)?;
                    provenance::project(&mut value, explicit)?;
                    Ok(value)
                },
                crate::sequence::sequence_error_to_runtime,
            )
        },
        |error| error.into_runtime_error(),
    )
}

fn propagate_gpu_provenance(args: &[Value], result: &mut Value) -> Result<(), RuntimeError> {
    provenance::project(
        result,
        explicit_gpu_provenance(crate::call::FunctionArgumentView::from_slice(args))?,
    )
}

fn explicit_gpu_provenance(
    args: crate::call::FunctionArgumentView<'_>,
) -> Result<bool, RuntimeError> {
    let mut explicit = false;
    for arg in args {
        try_visit_gpu_handles(arg, &mut |handle| {
            explicit |= runmat_accelerate_api::handle_is_explicit(handle);
        })?;
    }
    Ok(explicit)
}

/// Visit resident handles, including nested object/callable payloads, without
/// revisiting cycles in GC-owned object graphs.
pub fn visit_gpu_handles(value: &Value, visitor: &mut impl FnMut(&GpuTensorHandle)) {
    let _ = try_visit_gpu_handles(value, visitor);
}

/// Prove the resident handles reachable from a value. Inaccessible GC payloads
/// are errors, not evidence that a borrowed or returned value contains no alias.
pub fn try_visit_gpu_handles(
    value: &Value,
    visitor: &mut impl FnMut(&GpuTensorHandle),
) -> Result<(), RuntimeError> {
    visit_gpu_handles_inner(value, visitor, &mut Vec::new(), GpuTraversal::Ownership)
}

/// Inspect precisely the resident payloads that scalar gather will download.
/// Identity-bearing passthrough values are deliberately not followed.
pub(crate) fn try_visit_gathered_gpu_handles(
    value: &Value,
    visitor: &mut impl FnMut(&GpuTensorHandle),
) -> Result<(), RuntimeError> {
    visit_gpu_handles_inner(value, visitor, &mut Vec::new(), GpuTraversal::Gather)
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum GpuTraversal {
    Ownership,
    Gather,
}

fn visit_gpu_handles_inner(
    value: &Value,
    visitor: &mut impl FnMut(&GpuTensorHandle),
    seen: &mut Vec<runmat_gc_api::GcHandle>,
    traversal: GpuTraversal,
) -> Result<(), RuntimeError> {
    match value {
        Value::GpuTensor(handle) => visitor(handle),
        Value::Cell(cell) => cell
            .data()
            .iter()
            .try_for_each(|value| visit_gpu_handles_inner(value, visitor, seen, traversal))?,
        Value::Struct(value) => value
            .fields
            .values()
            .try_for_each(|value| visit_gpu_handles_inner(value, visitor, seen, traversal))?,
        Value::StructArray(array) => {
            let mut result = Ok(());
            array.for_each_value(|value| {
                if result.is_ok() {
                    result = visit_gpu_handles_inner(value, visitor, seen, traversal);
                }
            });
            result?;
        }
        Value::Object(value) => value
            .properties()
            .values()
            .try_for_each(|value| visit_gpu_handles_inner(value, visitor, seen, traversal))?,
        Value::ObjectArray(array) => array
            .data()
            .iter()
            .try_for_each(|value| visit_gpu_handles_inner(value, visitor, seen, traversal))?,
        Value::HandleObject(handle) if traversal == GpuTraversal::Ownership => {
            visit_gpu_target(&handle.target, visitor, seen)?;
        }
        Value::Listener(listener) if traversal == GpuTraversal::Ownership => {
            visit_gpu_target(&listener.target, visitor, seen)?;
            visit_gpu_target(&listener.callback, visitor, seen)?;
        }
        Value::Closure(value) => value
            .captures
            .iter()
            .try_for_each(|value| visit_gpu_handles_inner(value, visitor, seen, traversal))?,
        Value::OutputList(_) if traversal == GpuTraversal::Gather => {
            return Err(crate::runtime_error::semantic_error(
                "TransientSequenceBelowDispatch",
                "legacy output-list carrier reached scalar gather dispatch",
            ));
        }
        Value::OutputList(values) => values
            .iter()
            .try_for_each(|value| visit_gpu_handles_inner(value, visitor, seen, traversal))?,
        _ => {}
    }
    Ok(())
}

fn visit_gpu_target(
    target: &runmat_gc_api::GcHandle,
    visitor: &mut impl FnMut(&GpuTensorHandle),
    seen: &mut Vec<runmat_gc_api::GcHandle>,
) -> Result<(), RuntimeError> {
    if seen.contains(target) {
        return Ok(());
    }
    seen.push(*target);
    // Release the GC access scope before following another handle edge.
    let value = runmat_gc::gc_with_value(target, Clone::clone).map_err(|error| {
        crate::build_runtime_error(format!("cannot inspect resident aliases: {error}"))
            .with_identifier("RunMat:Gpu:AliasInspectionFailed")
            .build()
    })?;
    visit_gpu_handles_inner(&value, visitor, seen, GpuTraversal::Ownership)
}

pub(crate) async fn try_call_registered_instance_method(
    method_name: &str,
    args: &[Value],
    output_count: Option<usize>,
    context: Option<&crate::context::RuntimeContext>,
) -> Result<Option<crate::execution::RootedValue>, RuntimeError> {
    try_call_registered_instance_method_sequence(method_name, args, output_count, context)
        .await
        .and_then(|sequence| {
            sequence
                .map(crate::call::arguments::project_rooted_legacy_builtin_value_abi)
                .transpose()
        })
}

async fn try_call_registered_instance_method_sequence(
    method_name: &str,
    args: &[Value],
    output_count: Option<usize>,
    context: Option<&crate::context::RuntimeContext>,
) -> Result<Option<crate::execution::RootedValueSequence>, RuntimeError> {
    let args = crate::call::BuiltinArguments::from_borrowed(args, context)?;
    try_call_retained_instance_method_sequence(method_name, &args, output_count, context).await
}

async fn try_call_retained_instance_method_sequence(
    method_name: &str,
    args: &crate::call::BuiltinArguments,
    output_count: Option<usize>,
    context: Option<&crate::context::RuntimeContext>,
) -> Result<Option<crate::execution::RootedValueSequence>, RuntimeError> {
    let Some(execution) = instance_execution::select(method_name, args, output_count, context)?
    else {
        return Ok(None);
    };
    execution.await
}

fn finalize_instance_method_result(
    context: Option<&crate::context::RuntimeContext>,
    method_name: &runmat_types::MethodName,
    receiver: &Value,
    result: Result<crate::execution::RootedValueSequence, RuntimeError>,
) -> Result<crate::execution::RootedValueSequence, RuntimeError> {
    let result = result?;
    const DELETE_METHOD: runmat_types::StaticMethodName =
        runmat_types::StaticMethodName::new("delete");
    if DELETE_METHOD.is(method_name) {
        if let Value::HandleObject(handle) = receiver {
            if !crate::object::handle_validity::set(context, handle, false)? {
                return Err(build_runtime_error(format!(
                    "delete: failed to invalidate handle object '{}' after its destructor completed",
                    handle.class_name
                ))
                .with_identifier("RunMat:delete:InvalidHandle")
                .build());
            }
        }
    }
    Ok(result)
}

#[derive(Debug, Clone)]
pub enum RegisteredStaticMethodTarget {
    Bound(runmat_value::BoundFunctionBinding),
    Semantic(usize),
    Builtin(runmat_types::BuiltinId),
}

/// Observe the actual selected runtime builtin inventory without causing
/// browser registration or dispatch. Catalog-only declarations do not win a
/// dynamic fallback over a registered class constructor.
pub fn observed_dispatch_builtin_name(name: &str) -> Option<runmat_types::BuiltinId> {
    let resolved = ResolvedBuiltinName::resolve(name);
    let name = resolved.canonical;
    let scoped = crate::context::legacy::active()
        .and_then(|context| context.service_ports().builtin().cloned());
    let selected_name = scoped.map_or_else(
        || {
            crate::builtin::registration::registered_runtime_builtin_bindings_by_name(name)
                .first()
                .map(|row| row.binding().identity.builtin.name)
                .or_else(|| {
                    builtin_functions()
                        .into_iter()
                        .find(|builtin| builtin.name == name)
                        .map(|builtin| builtin.name)
                })
        },
        |service| {
            service
                .bindings_by_name(name)
                .first()
                .map(|binding| binding.identity.builtin.name)
        },
    );
    selected_name.map(|selected_name| {
        runmat_builtins::builtin_id_by_name(selected_name)
            .unwrap_or_else(|| runmat_types::BuiltinId(selected_name.to_owned()))
    })
}

#[derive(Debug, Clone)]
pub struct RegisteredStaticMethodResolution {
    pub registration: crate::class_registry::RegisteredStaticMethodName,
    pub target: RegisteredStaticMethodTarget,
}

/// Read-only static-method body selection used by both dispatcher and name
/// observation. Registration alone is not an executable method body.
pub fn observe_registered_static_method(
    qualified_name: &str,
) -> Option<RegisteredStaticMethodResolution> {
    let registration = crate::class_registry::registered_public_static_method(qualified_name)?;
    observe_registered_static_member(qualified_name, registration)
}

pub(crate) fn observe_registered_static_member(
    qualified_name: &str,
    registration: crate::class_registry::RegisteredStaticMethodName,
) -> Option<RegisteredStaticMethodResolution> {
    if let Some(binding) =
        crate::class_registry::lookup_bound_method(&registration.class, &registration.method_name)
            .and_then(|method| method.bound_function)
    {
        return Some(RegisteredStaticMethodResolution {
            registration,
            target: RegisteredStaticMethodTarget::Bound(binding),
        });
    }
    let executable_semantic = crate::user_functions::current_semantic_function_invoker().is_some();
    let select = |name: &str| {
        if executable_semantic {
            if let Some(function) = crate::user_functions::resolve_semantic_function_by_name(name) {
                return Some(RegisteredStaticMethodTarget::Semantic(function));
            }
        }
        (name != qualified_name)
            .then(|| runmat_builtins::builtin_id_by_name(name))
            .flatten()
            .map(RegisteredStaticMethodTarget::Builtin)
    };
    let target = select(&registration.declaration.function_name).or_else(|| {
        let inherited = format!(
            "{}.{}",
            registration.declaring_class, registration.method_name
        );
        (inherited != registration.declaration.function_name)
            .then(|| select(&inherited))
            .flatten()
    })?;
    Some(RegisteredStaticMethodResolution {
        registration,
        target,
    })
}

async fn try_call_registered_static_method(
    qualified_name: &str,
    args: &crate::call::BuiltinArguments,
    output_count: Option<usize>,
    context: Option<&crate::context::RuntimeContext>,
) -> Result<Option<crate::execution::RootedValueSequence>, RuntimeError> {
    let Some(resolved) = observe_registered_static_method(qualified_name) else {
        return Ok(None);
    };
    invoke_registered_static_method(qualified_name, resolved, args, output_count, context)
        .await
        .map(Some)
}

/// Execute the exact registry-selected body once. A selected body that fails
/// or becomes unavailable is terminal, never a name-fallback invitation.
pub(crate) async fn invoke_registered_static_method(
    qualified_name: &str,
    resolved: RegisteredStaticMethodResolution,
    args: &crate::call::BuiltinArguments,
    output_count: Option<usize>,
    context: Option<&crate::context::RuntimeContext>,
) -> Result<crate::execution::RootedValueSequence, RuntimeError> {
    let original_arguments = args.fork();
    let method = resolved.registration.declaration;
    let args = if let Some(class) = method.implicit_class_argument.as_ref() {
        let text = crate::call::argument_text::copy_str(class.display_name(), context)?;
        let inserted = crate::execution::RootedValue::in_optional_context(
            Value::String(text.into()),
            context,
        )?;
        match context {
            Some(context) => args
                .fork()
                .try_insert_in_context(args.len(), inserted, context)?,
            // Context-free embedding has no managed allocation authority.
            None => crate::execution::RootedValues::new(
                args.iter()
                    .cloned()
                    .chain(std::iter::once(inserted.value().clone()))
                    .collect(),
            )
            .map(crate::call::BuiltinArguments::from)
            .map_err(|error| error.into_runtime_error())?,
        }
    } else {
        args.fork()
    };
    let result = match resolved.target {
        RegisteredStaticMethodTarget::Bound(binding) => {
            let requested = output_count.unwrap_or(1);
            let invoked = match context {
                Some(runtime) => {
                    crate::user_functions::try_call_rooted_bound_function_with_environment_in_context(
                        runtime, binding, args.into_supplied_function_arguments(context)?, requested, None,
                    )
                    .await
                }
                None => {
                    crate::user_functions::try_call_rooted_bound_function(binding, args.into_supplied_function_arguments(context)?, requested).await
                }
            };
            invoked.ok_or_else(|| {
                crate::runtime_error::semantic_error(
                    "UnavailableBoundFunction",
                    "registered static method has no bound executor",
                )
            })?
        }
        RegisteredStaticMethodTarget::Semantic(function) => {
            crate::user_functions::try_call_rooted_semantic_function_in_context(
                context,
                function,
                args.into_supplied_function_arguments(context)?,
                output_count.unwrap_or(1),
            )
            .await
            .ok_or_else(|| {
                crate::runtime_error::semantic_error(
                    "UnavailableFunction",
                    "registered static method has no semantic executor",
                )
            })?
        }
        RegisteredStaticMethodTarget::Builtin(name) => try_call_retained_builtin_sequence_async(
            &name.0,
            args,
            output_count,
            context.cloned(),
            None,
        )
        .await
        .and_then(|result| {
            result.ok_or_else(|| {
                crate::runtime_error::semantic_error(
                    "UnavailableBuiltin",
                    "registered static method has no builtin executor",
                )
            })
        }),
    };
    result
        .and_then(|result| {
            compatibility_checked_builtin_sequence(
                qualified_name,
                original_arguments.view(),
                result,
            )
        })
        .map_err(|mut error| {
            // The selected body was attempted. Preserve its original cause and
            // control fields, but an outer feval gather retry must not replay it.
            error.gpu_gather_retry = crate::GpuGatherRetry::Never;
            error
        })
}

#[cfg(test)]
async fn call_registered_class_constructor(
    class_name: &str,
    args: &[Value],
    output_count: Option<usize>,
    context: Option<&crate::context::RuntimeContext>,
) -> Result<crate::execution::RootedValue, RuntimeError> {
    let args = crate::call::BuiltinArguments::from_borrowed(args, context)?;
    call_retained_class_constructor(class_name, args, output_count, context).await
}

async fn call_retained_class_constructor(
    class_name: &str,
    args: crate::call::BuiltinArguments,
    output_count: Option<usize>,
    context: Option<&crate::context::RuntimeContext>,
) -> Result<crate::execution::RootedValue, RuntimeError> {
    let class_identity = ClassIdentity::new(class_name).map_err(|error| {
        build_runtime_error(format!("Invalid class name '{class_name}': {error}"))
            .with_identifier("RunMat:InvalidClassName")
            .build()
    })?;
    call_class_constructor(
        class_identity,
        ConstructorArguments::Retained(args),
        output_count,
        context,
        ConstructorInvocationTarget::Registered,
    )
    .await
}

async fn call_class_constructor(
    class_identity: ClassIdentity,
    args: ConstructorArguments<'_>,
    output_count: Option<usize>,
    context: Option<&crate::context::RuntimeContext>,
    target: ConstructorInvocationTarget<'_>,
) -> Result<crate::execution::RootedValue, RuntimeError> {
    let requested_outputs = output_count.unwrap_or(1);
    if class_identity.is(runmat_types::standard::EVENT_DATA) && !args.is_empty() {
        return Err(build_runtime_error(
            "event.EventData constructor does not accept input arguments",
        )
        .with_identifier("RunMat:ConstructorArgumentsInvalid")
        .build());
    }
    let qualified_class = class_identity.qualified_name();
    let constructor_method_name = qualified_class
        .0
        .last()
        .expect("validated class identity has a final segment")
        .0
        .as_str();
    let constructor_identity = runmat_types::MethodName::from(constructor_method_name);
    let qualified_constructor_identity =
        runmat_types::MethodName::from(class_identity.display_name());
    // Exact bound ownership is checked before any default-object effects.
    let exact_method = match &target {
        ConstructorInvocationTarget::Registered => None,
        ConstructorInvocationTarget::ExactBound { owner, .. }
        | ConstructorInvocationTarget::ExactPrepared { owner, .. } => {
            if owner.role != runmat_types::ClassMethodRole::Constructor
                || owner.is_static
                || owner.declaring_class != class_identity
            {
                return Err(build_runtime_error(
                    "bound constructor owner does not match its class",
                )
                .with_identifier("RunMat:ConstructorOwnerInvalid")
                .build());
            }
            let Some((method, declaring_class)) =
                crate::class_registry::lookup_method(&class_identity, &owner.method)
            else {
                return Err(
                    build_runtime_error("bound constructor lacks its registered method")
                        .with_identifier("RunMat:ConstructorOwnerInvalid")
                        .build(),
                );
            };
            if declaring_class != owner.declaring_class || method.is_static {
                return Err(build_runtime_error(
                    "bound constructor disagrees with registered method",
                )
                .with_identifier("RunMat:ConstructorOwnerInvalid")
                .build());
            }
            Some((method, declaring_class))
        }
    };
    let lookup = exact_method.or_else(|| {
        crate::class_registry::lookup_method(&class_identity, &constructor_identity).or_else(|| {
            crate::class_registry::lookup_method(&class_identity, &qualified_constructor_identity)
        })
    });
    let Some((ctor, owner)) = lookup else {
        if matches!(&target, ConstructorInvocationTarget::Registered) {
            return crate::execution::RootedValue::new(
                crate::create_class_object_by_identity(class_identity).await?,
            )
            .map_err(|error| error.into_runtime_error());
        }
        return Err(
            build_runtime_error("bound constructor lacks its registered method")
                .with_identifier("RunMat:ConstructorOwnerInvalid")
                .build(),
        );
    };
    let owner_qualified = format!("{owner}.{constructor_method_name}");
    let caller_class = current_class_access_context();
    let ctor_access_allowed = match ctor.access {
        runmat_types::MemberAccess::Public => true,
        runmat_types::MemberAccess::Private => caller_class.as_deref() == Some(&owner),
        runmat_types::MemberAccess::Protected => caller_class
            .as_deref()
            .is_some_and(|caller| crate::class_registry::is_class_or_subclass(caller, &owner)),
    };
    if !ctor_access_allowed {
        return Err(build_runtime_error(format!(
            "Constructor '{}' is not accessible from current context.",
            class_identity.display_name()
        ))
        .with_identifier("RunMat:MethodPrivate")
        .build());
    }
    let default_object = crate::execution::RootedValue::new(
        crate::create_class_object_by_identity(class_identity.clone()).await?,
    )
    .map_err(|error| error.into_runtime_error())?;
    let constructor_result =
        crate::with_constructor_receiver(default_object.value().clone(), async {
            match target {
                ConstructorInvocationTarget::ExactBound { binding, .. } => {
                    let selected = context.ok_or_else(|| {
                        build_runtime_error("bound constructor lacks a selected runtime")
                            .with_identifier("RunMat:ConstructorOwnerInvalid")
                            .build()
                    })?;
                    let result = crate::user_functions::try_call_rooted_bound_function_with_environment_in_context(
                        selected,
                        binding,
                        args.retained(context)?.into_supplied_function_arguments(context)?,
                        requested_outputs,
                        None,
                    )
                    .await
                    .ok_or_else(|| {
                        build_runtime_error("bound constructor has no exact invoker")
                            .with_identifier("RunMat:BoundFunction:UnavailableInvoker")
                            .build()
                    })??;
                    return Ok(Some(resolve_single_sequence(result)?));
                }
                ConstructorInvocationTarget::ExactPrepared {
                    owner,
                    invoke,
                    guard,
                } => {
                    let selected = context.ok_or_else(|| {
                        build_runtime_error("prepared constructor lacks a selected runtime")
                            .with_identifier("RunMat:ConstructorOwnerInvalid")
                            .build()
                    })?;
                    // Capture and root the exact constructor instance while this
                    // receiver poll guard is active. The child may not be polled
                    // until after this parent yields to the local-task driver.
                    let selected = selected
                        .clone()
                        .with_constructor_receiver_for_method(Some(&owner))?;
                    let future = invoke.invoke(selected.clone(), guard)?;
                    return Ok(Some(resolve_single_sequence(
                        crate::context::run_owned_request_continuation(&selected, future).await?,
                    )?));
                }
                ConstructorInvocationTarget::Registered => {
                    // A linked class declaration already carries the exact
                    // method body. Public textual name resolution is not an
                    // authority for a constructor Method-role function.
                    if let Some(bound) =
                        crate::class_registry::lookup_bound_method(&class_identity, &ctor.name)
                            .filter(|bound| bound.declaring_class == owner)
                            .and_then(|bound| bound.bound_function)
                    {
                        let selected = context.ok_or_else(|| {
                            build_runtime_error(
                                "registered bound constructor lacks a selected runtime",
                            )
                            .with_identifier("RunMat:ConstructorOwnerInvalid")
                            .build()
                        })?;
                        let result = crate::user_functions::try_call_rooted_bound_function_with_environment_in_context(
                            selected,
                            bound,
                            args.retained(context)?.into_supplied_function_arguments(context)?,
                            requested_outputs,
                            None,
                        )
                        .await
                        .ok_or_else(|| {
                            build_runtime_error("registered constructor has no exact invoker")
                                .with_identifier("RunMat:BoundFunction:UnavailableInvoker")
                                .build()
                        })??;
                        return Ok(Some(resolve_single_sequence(result)?));
                    }
                }
            }
            if let std::ops::ControlFlow::Break(result) =
                crate::user_functions::try_call_rooted_semantic_name_in_context(
                    context,
                    &ctor.function_name,
                    args.retained(context)?.into_supplied_function_arguments(context)?,
                    requested_outputs,
                )
                .await
            {
                return Ok::<Option<crate::execution::RootedValue>, RuntimeError>(Some(
                    resolve_single_sequence(result?)?,
                ));
            }
            if runmat_builtins::builtin_name_is_known(&ctor.function_name)
                && ctor.function_name != class_identity.display_name()
            {
                let result = try_call_retained_builtin_sequence_async(
                    &ctor.function_name,
                    args.retained(context)?,
                    output_count,
                    context.cloned(),
                    None,
                )
                .await?.ok_or_else(|| undefined_builtin_error(&ctor.function_name))?;
                let result = resolve_single_sequence(result)?;
                return Ok::<Option<crate::execution::RootedValue>, RuntimeError>(Some(result));
            }
            if let std::ops::ControlFlow::Break(result) =
                crate::user_functions::try_call_rooted_semantic_name_in_context(
                    context,
                    &owner_qualified,
                    args.retained(context)?.into_supplied_function_arguments(context)?,
                    requested_outputs,
                )
                .await
            {
                return Ok::<Option<crate::execution::RootedValue>, RuntimeError>(Some(
                    resolve_single_sequence(result?)?,
                ));
            }
            if runmat_builtins::builtin_name_is_known(&owner_qualified)
                && owner_qualified != class_identity.display_name()
            {
                let result = try_call_retained_builtin_sequence_async(
                    &owner_qualified,
                    args.retained(context)?,
                    output_count,
                    context.cloned(),
                    None,
                )
                .await?.ok_or_else(|| undefined_builtin_error(&owner_qualified))?;
                let result = resolve_single_sequence(result)?;
                return Ok::<Option<crate::execution::RootedValue>, RuntimeError>(Some(result));
            }
            Ok::<Option<crate::execution::RootedValue>, RuntimeError>(None)
        })
        .await?;
    let Some(result) = constructor_result else {
        return Ok(default_object);
    };
    result.with_owned(|result| {
        constructor_normalization::normalize(
            context,
            default_object.value().clone(),
            result,
            requested_outputs,
        )
    })
}

/// Direct embedding call with no session binding authority. Entering a
/// runtime's poll-local scope does not change this selection contract; session
/// callers must use `call_builtin_async_in_context`.
pub async fn call_builtin_async(
    name: &str,
    args: &[Value],
) -> Result<crate::execution::RootedValue, RuntimeError> {
    crate::context::check_active_host()?;
    let result = call_builtin_sequence_async(name, args)
        .await
        .and_then(crate::call::arguments::project_rooted_legacy_builtin_value_abi);
    crate::context::check_active_host()?;
    result
}

/// Internal forwarding keeps the original logical argument owner and selected
/// session authority instead of passing through borrowed embedding ingress.
pub(crate) async fn call_retained_builtin_async(
    name: &str,
    args: crate::call::BuiltinArguments,
    context: Option<&crate::context::RuntimeContext>,
) -> Result<crate::execution::RootedValue, RuntimeError> {
    crate::context::check_active_host()?;
    let result = try_call_retained_builtin_sequence_async(name, args, None, context.cloned(), None)
        .await
        .and_then(|result| result.ok_or_else(|| undefined_builtin_error(name)))
        .and_then(crate::call::arguments::project_rooted_legacy_builtin_value_abi);
    crate::context::check_active_host()?;
    result
}

/// Session-owned entrypoint. Binding selection, preparation and execution all
/// receive this same authority; poll-local scope serves existing subordinate
/// APIs without becoming the selection mechanism.
pub async fn call_builtin_async_in_context(
    context: &crate::context::RuntimeContext,
    name: &str,
    args: &[Value],
) -> Result<crate::execution::RootedValue, RuntimeError> {
    crate::context::check_active_host()?;
    let result = call_builtin_sequence_async_in_context(context, name, args)
        .await
        .and_then(crate::call::arguments::project_rooted_legacy_builtin_value_abi);
    crate::context::check_active_host()?;
    result
}

/// Direct embedding call with no session binding authority. Use the
/// in-context variant when the call belongs to a VM, native, or foreign session.
pub async fn call_builtin_async_with_outputs(
    name: &str,
    args: &[Value],
    output_count: usize,
) -> Result<crate::execution::RootedValue, RuntimeError> {
    crate::context::check_active_host()?;
    let result = call_builtin_sequence_async_with_outputs(name, args, output_count)
        .await
        .and_then(crate::call::arguments::project_rooted_legacy_builtin_value_abi);
    crate::context::check_active_host()?;
    result
}

pub async fn call_builtin_async_with_outputs_in_context(
    context: &crate::context::RuntimeContext,
    name: &str,
    args: &[Value],
    output_count: usize,
) -> Result<crate::execution::RootedValue, RuntimeError> {
    crate::context::check_active_host()?;
    let result =
        call_builtin_sequence_async_with_outputs_in_context(context, name, args, output_count)
            .await
            .and_then(crate::call::arguments::project_rooted_legacy_builtin_value_abi);
    crate::context::check_active_host()?;
    result
}

/// Invoke the implementation already selected for this invocation. Re-resolving
/// a scoped service here would invalidate operand-domain batching admission.
#[cfg(test)]
pub(crate) async fn call_selected_builtin_async_with_outputs(
    binding: crate::builtin::RuntimeBuiltinBinding,
    args: &[Value],
    output_count: usize,
    context: Option<crate::context::RuntimeContext>,
) -> Result<crate::execution::RootedValueSequence, RuntimeError> {
    call_selected_builtin_with_arguments(
        binding,
        |runtime| crate::call::BuiltinArguments::from_borrowed(args, runtime),
        output_count,
        context,
    )
    .await
}

pub(crate) async fn call_selected_retained_builtin_async_with_outputs(
    binding: crate::builtin::RuntimeBuiltinBinding,
    args: crate::call::BuiltinArguments,
    output_count: usize,
    context: Option<crate::context::RuntimeContext>,
) -> Result<crate::execution::RootedValueSequence, RuntimeError> {
    call_selected_builtin_with_arguments(binding, move |_| Ok(args), output_count, context).await
}

pub(crate) async fn call_selected_builtin_with_arguments(
    binding: crate::builtin::RuntimeBuiltinBinding,
    ingress: impl FnOnce(
        Option<&crate::context::RuntimeContext>,
    ) -> Result<crate::call::BuiltinArguments, RuntimeError>,
    output_count: usize,
    context: Option<crate::context::RuntimeContext>,
) -> Result<crate::execution::RootedValueSequence, RuntimeError> {
    let fence_context = context.clone();
    let fence = workspace_effect::PostCallWorkspaceEffectFence::capture(fence_context.as_ref());
    let observation =
        direct_call_observation::CallObservation::capture(context.as_ref()).map(std::rc::Rc::new);
    let inner_observation = observation.clone();
    let scoped_context = context.clone();
    let result = async move {
        let args = ingress(context.as_ref())?;
        let name = binding.identity.builtin.name;
        let _output_guard = crate::output_count::push_output_count(Some(output_count))?;
        if let Some(result) =
            try_distributed_builtin(name, &args, Some(output_count), context.as_ref()).await?
        {
            return Ok(result);
        }
        if !SelectedImplementation::Canonical(binding).is_handle_validity_query(args.first()) {
            if let Some(result) = try_call_retained_instance_method_sequence(
                name,
                &args,
                Some(output_count),
                context.as_ref(),
            )
            .await?
            {
                return compatibility_checked_builtin_sequence(name, args.view(), result);
            }
        }
        let prepared = preparation::prepare_retained_selected(
            SelectedBindingOrigin::CanonicalOrScoped,
            name,
            args.fork(),
            context.as_ref(),
        )
        .await?;
        let authority = context.as_ref().and_then(|context| {
            if let Some(service) = context.service_ports().builtin() {
                service
                    .binding_registration_revision(binding)
                    .map(
                        |generation| direct_call_observation::RegistrationAuthority::Scoped {
                            service: service.clone(),
                            generation,
                        },
                    )
            } else {
                crate::builtin::registration::registered_runtime_builtin_bindings_by_name(name)
                    .into_iter()
                    .find(|row| row.registration().certifies(binding))
                    .map(|row| {
                        direct_call_observation::RegistrationAuthority::Global(row.registration())
                    })
            }
        });
        let result = selected_execution::execute_retained(
            SelectedImplementation::Canonical(binding),
            prepared,
            context,
            Some(output_count),
            inner_observation.as_deref(),
            authority,
        )
        .await?;
        compatibility_checked_builtin_sequence(name, args.view(), result)
    };
    let result = match scoped_context {
        Some(context) => context.scope(result).await,
        None => result.await,
    };
    let (result, outcome) = fence.finish_owned_sequence_observed(result);
    if let Some(observation) = observation {
        observation.finish_completed(outcome);
    }
    result
}

#[cfg(test)]
fn should_retry_with_gpu_gather(err: &RuntimeError, args: &[Value]) -> bool {
    should_retry_with_gpu_gather_values(err, args.iter())
}

fn should_retry_with_gpu_gather_values<'a>(
    err: &RuntimeError,
    args: impl Iterator<Item = &'a Value> + Clone,
) -> bool {
    if crate::context::current_acceleration_policy()
        .is_some_and(|policy| !policy.allow_host_fallback)
    {
        return false;
    }
    if !args.clone().any(value_contains_gpu) {
        return false;
    }
    if error_chain_has_gpu_gather_retry(err, crate::GpuGatherRetry::Never) {
        return false;
    }
    if args.clone().any(value_contains_explicit_gpu) {
        return false;
    }
    // Compatibility errors are policy decisions. Retain this source-chain
    // defense for wrappers that have not yet propagated an explicit policy.
    if error_chain_has_identifier_prefix(err, "RunMat:compatibility:") {
        return false;
    }
    error_chain_has_gpu_gather_retry(err, crate::GpuGatherRetry::Requested)
}

/// Returns whether a value, including any recursively captured value, carries
/// explicit device-residency intent. Execution boundaries may gather automatic
/// residency, but must not silently erase an explicit `gpuArray` request.
pub fn value_contains_explicit_gpu(value: &Value) -> bool {
    match value {
        Value::GpuTensor(handle) => runmat_accelerate_api::handle_is_explicit(handle),
        Value::Cell(cell) => cell.data().iter().any(value_contains_explicit_gpu),
        Value::Struct(value) => value.fields.values().any(value_contains_explicit_gpu),
        Value::StructArray(array) => {
            let mut found = false;
            array.for_each_value(|value| found |= value_contains_explicit_gpu(value));
            found
        }
        Value::Object(value) => value.properties().values().any(value_contains_explicit_gpu),
        Value::Closure(value) => value.captures.iter().any(value_contains_explicit_gpu),
        Value::OutputList(values) => values.iter().any(value_contains_explicit_gpu),
        _ => false,
    }
}

fn error_chain_has_gpu_gather_retry(err: &RuntimeError, policy: crate::GpuGatherRetry) -> bool {
    let mut current: Option<&(dyn std::error::Error + 'static)> = Some(err);
    while let Some(error) = current {
        if error
            .downcast_ref::<RuntimeError>()
            .is_some_and(|error| error.gpu_gather_retry() == policy)
        {
            return true;
        }
        current = error.source();
    }
    false
}

fn error_chain_has_identifier_prefix(err: &RuntimeError, prefix: &str) -> bool {
    let mut current: Option<&(dyn std::error::Error + 'static)> = Some(err);
    while let Some(error) = current {
        if error
            .downcast_ref::<RuntimeError>()
            .and_then(RuntimeError::identifier)
            .is_some_and(|identifier| identifier.starts_with(prefix))
        {
            return true;
        }
        current = error.source();
    }
    false
}

#[cfg(test)]
mod gpu_visit_tests;
#[cfg(test)]
mod preparation_tests;
#[cfg(test)]
mod tests {
    use super::{
        call_builtin, gather_if_needed_async, should_retry_with_gpu_gather, value_contains_gpu,
    };
    use futures::executor::block_on;
    use runmat_accelerate_api::{GpuTensorHandle, ThreadProviderGuard};
    use runmat_types::MemberAccess;
    use runmat_value::{Closure, Value};
    use runmat_value::{IntegerStorage, Tensor};
    use std::collections::HashMap;
    use std::sync::atomic::{AtomicU64, Ordering};

    fn gpu_fixture(shape: Vec<usize>) -> GpuTensorHandle {
        runmat_accelerate_api::test_support::TestAllocationOwner::new()
            .allocate_zeroed(
                shape,
                runmat_accelerate_api::NumericElementType::F64,
                runmat_accelerate_api::GpuTensorStorage::Real,
            )
            .unwrap()
            .handle
    }

    static TEST_CLASS_COUNTER: AtomicU64 = AtomicU64::new(0);

    #[test]
    fn event_data_constructor_admits_only_zero_inputs() {
        let rejected = block_on(super::call_registered_class_constructor(
            "event.EventData",
            &[Value::Num(1.0)],
            Some(1),
            None,
        ))
        .unwrap_err();
        assert_eq!(
            rejected.identifier(),
            Some("RunMat:ConstructorArgumentsInvalid")
        );
        let instance = block_on(super::call_registered_class_constructor(
            "event.EventData",
            &[],
            Some(1),
            None,
        ))
        .expect("zero-input EventData");
        assert!(crate::class_registry::is_event_data_value(&instance));
    }

    #[test]
    fn instance_method_sequence_preserves_output_kind_and_cardinality() {
        use runmat_value::ValueSequence;
        let method = runmat_types::MethodName::from("display");
        let receiver = Value::Num(0.0);
        for values in [
            vec![],
            vec![Value::Num(1.0)],
            vec![Value::Num(1.0), Value::Num(2.0)],
        ] {
            let output = super::finalize_instance_method_result(
                None,
                &method,
                &receiver,
                crate::execution::RootedValueSequence::new(
                    ValueSequence::comma_separated(values.clone()).unwrap(),
                )
                .map_err(|error| error.into_runtime_error()),
            )
            .unwrap();
            assert_eq!(
                output.kind(),
                runmat_value::ValueSequenceKind::CommaSeparated
            );
            assert_eq!(output.as_slice(), values);
        }
        assert_eq!(
            super::finalize_instance_method_result(
                None,
                &method,
                &receiver,
                crate::user_functions::test_support::rooted_single_value_sequence(Value::Num(3.0)),
            )
            .unwrap(),
            ValueSequence::single(Value::Num(3.0)).unwrap(),
        );
    }

    struct EmptyBuiltinService {
        lookups: std::rc::Rc<std::cell::Cell<usize>>,
    }

    impl crate::context::RuntimeBuiltinService for EmptyBuiltinService {
        fn bindings_by_name(&self, _name: &str) -> Vec<crate::builtin::RuntimeBuiltinBinding> {
            self.lookups.set(self.lookups.get() + 1);
            Vec::new()
        }
    }

    #[test]
    fn catalog_backed_builtin_dispatches_without_legacy_authority() {
        assert!(runmat_builtins::builtin_function_by_name("full").is_none());
        let input = Value::Num(7.0);
        assert_eq!(
            call_builtin("full", std::slice::from_ref(&input)).unwrap(),
            input
        );
    }

    #[test]
    fn scoped_builtin_authority_does_not_fall_back_to_global_discovery() {
        let lookups = std::rc::Rc::new(std::cell::Cell::new(0));
        let ports = crate::context::RuntimeServicePorts::default().with_builtin(std::rc::Rc::new(
            EmptyBuiltinService {
                lookups: std::rc::Rc::clone(&lookups),
            },
        ));
        let runtime = crate::context::RuntimeContext::new(std::rc::Rc::new(
            crate::execution::RuntimeExecutionService::new(),
        ))
        .with_service_ports(ports);
        let _scope = runtime.enter();
        let initial_output = crate::output_count::current_output_scope();
        let unpolled = super::try_call_retained_builtin_sequence_with_policy(
            "full",
            crate::call::BuiltinArguments::from_borrowed(&[], Some(&runtime)).unwrap(),
            Some(2),
            Some(runtime.clone()),
            None,
            super::RetainedDispatchPolicy::RegisteredOnly,
        );
        assert_eq!(lookups.get(), 0);
        drop(unpolled);
        assert_eq!(lookups.get(), 0);
        assert_eq!(crate::output_count::current_output_scope(), initial_output);
        let error = block_on(super::call_builtin_async_in_context(
            &runtime,
            "full",
            &[Value::Num(7.0)],
        ))
        .expect_err("exact registry miss");
        assert!(lookups.get() > 0);
        assert_eq!(error.identifier(), Some("RunMat:UndefinedFunction"));
        // The embedding entrypoint deliberately has no session authority.
        assert_eq!(
            call_builtin("full", &[Value::Num(7.0)]).unwrap(),
            Value::Num(7.0)
        );
    }

    #[test]
    fn operation_floating_projection_uses_native_download_and_rejects_integers() {
        crate::builtins::common::test_support::with_test_provider(|provider| {
            let single = Tensor::from_f32(vec![1.25, -2.5], vec![1, 2]).unwrap();
            let single_handle =
                crate::builtins::common::gpu_helpers::upload_tensor(provider, &single).unwrap();
            let projected = block_on(
                crate::builtins::common::gpu_helpers::download_floating_projection_async(
                    provider,
                    &single_handle,
                ),
            )
            .unwrap();
            assert_eq!(projected.data, vec![1.25, -2.5]);
            assert_eq!(projected.shape, vec![1, 2]);

            let integer =
                Tensor::new_integer(IntegerStorage::U64(vec![1_u64 << 63, u64::MAX]), vec![1, 2])
                    .unwrap();
            let integer_handle =
                crate::builtins::common::gpu_helpers::upload_tensor(provider, &integer).unwrap();
            let error = block_on(
                crate::builtins::common::gpu_helpers::download_floating_projection_async(
                    provider,
                    &integer_handle,
                ),
            )
            .expect_err("floating projection must reject native integer storage");
            assert_eq!(
                error.identifier(),
                Some("RunMat:gpu:IntegerFloatingProjection")
            );

            drop((single_handle, integer_handle));
        });
    }

    #[test]
    fn compatibility_errors_never_trigger_automatic_gpu_gather_retry() {
        let gpu = Value::GpuTensor(gpu_fixture(vec![1, 1]));
        let compatibility_error =
            crate::build_runtime_error("example gpuArray call form is a RunMat extension")
                .with_identifier("RunMat:compatibility:ExampleExtension")
                .build();
        assert!(!should_retry_with_gpu_gather(
            &compatibility_error,
            std::slice::from_ref(&gpu)
        ));

        let wrapped_compatibility_error =
            crate::build_runtime_error("GPU implementation failed while checking the call")
                .with_identifier("RunMat:example:GpuFailure")
                .with_source(compatibility_error)
                .build();
        assert!(!should_retry_with_gpu_gather(
            &wrapped_compatibility_error,
            std::slice::from_ref(&gpu)
        ));
        let requested_compatibility_error =
            crate::build_runtime_error("host implementation requested")
                .with_gpu_gather_retry(crate::GpuGatherRetry::Requested)
                .with_source(
                    crate::build_runtime_error("RunMat-only GPU form")
                        .with_identifier("RunMat:compatibility:ExampleExtension")
                        .build(),
                )
                .build();
        assert!(!should_retry_with_gpu_gather(
            &requested_compatibility_error,
            std::slice::from_ref(&gpu)
        ));

        let automatic_gpu = gpu_fixture(vec![1, 1]);
        let automatic_gpu = automatic_gpu
            .with_provenance(runmat_accelerate_api::GpuHandleProvenance::Automatic)
            .unwrap();
        let ordinary_gpu_error = crate::build_runtime_error("GPU input requires host fallback")
            .with_identifier("RunMat:example:UnsupportedGpuPath")
            .build();
        assert!(!should_retry_with_gpu_gather(
            &ordinary_gpu_error,
            &[Value::GpuTensor(automatic_gpu.clone())]
        ));
        let automatic_gpu = automatic_gpu
            .with_provenance(runmat_accelerate_api::GpuHandleProvenance::Explicit)
            .unwrap();
        assert!(!should_retry_with_gpu_gather(
            &ordinary_gpu_error,
            &[Value::GpuTensor(automatic_gpu.clone())]
        ));

        let terminal_gpu_error = crate::build_runtime_error("GPU input is semantically invalid")
            .with_gpu_gather_retry(crate::GpuGatherRetry::Never)
            .build();
        assert!(!should_retry_with_gpu_gather(
            &terminal_gpu_error,
            &[Value::GpuTensor(gpu_fixture(vec![1, 1]))]
        ));

        let nested_terminal = crate::build_runtime_error("terminal provider decision")
            .with_gpu_gather_retry(crate::GpuGatherRetry::Never)
            .build();
        let wrapped_terminal = crate::build_runtime_error("GPU implementation failed")
            .with_source(nested_terminal)
            .build();
        assert!(!should_retry_with_gpu_gather(
            &wrapped_terminal,
            &[Value::GpuTensor(gpu_fixture(vec![1, 1]))]
        ));

        let nested_request = crate::build_runtime_error("host implementation is required")
            .with_gpu_gather_retry(crate::GpuGatherRetry::Requested)
            .build();
        let wrapped_request = crate::build_runtime_error("provider path unavailable")
            .with_source(nested_request)
            .build();
        let requested_handle = gpu_fixture(vec![1, 1]);
        assert!(should_retry_with_gpu_gather(
            &wrapped_request,
            &[Value::GpuTensor(requested_handle.clone())]
        ));
        // The same wrapped provider cause remains classified as requested,
        // but invocation policy separately decides whether fallback is legal.
        for fallback in [false, true] {
            let runtime = crate::context::RuntimeContext::new(std::rc::Rc::new(
                crate::execution::RuntimeExecutionService::new(),
            ))
            .with_acceleration_policy(crate::context::RuntimeAccelerationPolicy {
                enabled: true,
                provider: crate::context::RuntimeProviderPreference::Auto,
                automatic_offload: false,
                allow_host_fallback: fallback,
                calibrate: false,
            });
            let _scope = runtime.enter();
            assert_eq!(
                should_retry_with_gpu_gather(
                    &wrapped_request,
                    &[Value::GpuTensor(requested_handle.clone())]
                ),
                fallback
            );
            let unsupported = runmat_accelerate_api::unsupported_provider_operation("matrix");
            assert!(
                crate::builtins::common::gpu_helpers::provider_hook_is_unsupported(&unsupported)
            );
            assert_eq!(
                crate::builtins::common::gpu_helpers::provider_hook_allows_host_fallback(
                    &unsupported
                ),
                fallback
            );
        }
        let requested_handle = requested_handle
            .with_provenance(runmat_accelerate_api::GpuHandleProvenance::Explicit)
            .unwrap();
        assert!(!should_retry_with_gpu_gather(
            &wrapped_request,
            &[Value::GpuTensor(requested_handle.clone())]
        ));
    }

    #[test]
    fn builtin_result_provenance_follows_gpu_input_intent() {
        let explicit = gpu_fixture(vec![1, 1]);
        let automatic = gpu_fixture(vec![1, 1]);
        let explicit_result = gpu_fixture(vec![1, 1]);
        let automatic_result = gpu_fixture(vec![1, 1]);
        let explicit = explicit
            .with_provenance(runmat_accelerate_api::GpuHandleProvenance::Explicit)
            .unwrap();
        let automatic = automatic
            .with_provenance(runmat_accelerate_api::GpuHandleProvenance::Automatic)
            .unwrap();

        let mut explicit_value = Value::GpuTensor(explicit_result.clone());
        super::propagate_gpu_provenance(&[Value::GpuTensor(explicit.clone())], &mut explicit_value)
            .unwrap();
        let mut automatic_value = Value::GpuTensor(automatic_result.clone());
        super::propagate_gpu_provenance(
            &[Value::GpuTensor(automatic.clone())],
            &mut automatic_value,
        )
        .unwrap();

        assert_eq!(
            runmat_accelerate_api::handle_provenance(&explicit_result),
            None
        );
        let Value::GpuTensor(explicit_value) = explicit_value else {
            unreachable!()
        };
        assert_eq!(
            explicit_value.view().provenance(),
            Some(runmat_accelerate_api::GpuHandleProvenance::Explicit)
        );
        assert_eq!(
            runmat_accelerate_api::handle_provenance(&automatic_result),
            None
        );
        let Value::GpuTensor(automatic_value) = automatic_value else {
            unreachable!()
        };
        assert_eq!(
            automatic_value.view().provenance(),
            Some(runmat_accelerate_api::GpuHandleProvenance::Automatic)
        );
    }

    #[test]
    fn builtin_result_provenance_preserves_producer_intent_without_gpu_inputs() {
        let closure = Value::Closure(Closure::new(
            "callback".into(),
            None,
            runmat_types::FunctionSourceRole::Anonymous,
            None,
            vec![Value::Num(7.)],
            None,
        ));
        let instance = closure.callable_instance_key();
        let mut cloned = closure.clone();
        super::propagate_gpu_provenance(&[], &mut cloned).unwrap();
        assert_eq!(cloned.callable_instance_key(), instance);
        let gathered = futures::executor::block_on(super::gather_if_needed_async(&cloned))
            .expect("host-only callback gather is a no-op");
        assert_eq!(gathered.callable_instance_key(), instance);

        let explicitly_created = gpu_fixture(vec![1, 1])
            .with_provenance(runmat_accelerate_api::GpuHandleProvenance::Explicit)
            .unwrap();
        let automatically_created = gpu_fixture(vec![1, 1]);

        let mut explicit_value = Value::GpuTensor(explicitly_created.clone());
        super::propagate_gpu_provenance(&[], &mut explicit_value).unwrap();
        let mut automatic_value = Value::GpuTensor(automatically_created.clone());
        super::propagate_gpu_provenance(&[], &mut automatic_value).unwrap();

        let Value::GpuTensor(explicit_value) = explicit_value else {
            unreachable!()
        };
        assert_eq!(
            explicit_value.view().provenance(),
            Some(runmat_accelerate_api::GpuHandleProvenance::Explicit)
        );
        let Value::GpuTensor(automatic_value) = automatic_value else {
            unreachable!()
        };
        assert_eq!(
            automatic_value.view().provenance(),
            Some(runmat_accelerate_api::GpuHandleProvenance::Automatic)
        );
    }

    #[test]
    fn builtin_result_provenance_does_not_downgrade_explicit_producer_intent() {
        let automatic_input = gpu_fixture(vec![1, 1])
            .with_provenance(runmat_accelerate_api::GpuHandleProvenance::Automatic)
            .unwrap();
        let explicit_result = gpu_fixture(vec![1, 1])
            .with_provenance(runmat_accelerate_api::GpuHandleProvenance::Explicit)
            .unwrap();

        let mut result = Value::GpuTensor(explicit_result.clone());
        super::propagate_gpu_provenance(&[Value::GpuTensor(automatic_input.clone())], &mut result)
            .unwrap();

        let Value::GpuTensor(result) = &result else {
            unreachable!()
        };
        assert_eq!(
            result.view().provenance(),
            Some(runmat_accelerate_api::GpuHandleProvenance::Explicit)
        );
    }

    #[test]
    fn array_constructor_forms_preserve_explicit_gpu_intent_without_dispatcher_name_policy() {
        let _random_guard = crate::builtins::common::random::test_guard();
        crate::builtins::common::test_support::with_test_provider(|provider| {
            let prototype_tensor = Tensor::new(vec![0.0; 4], vec![2, 2]).expect("prototype");
            let mut prototype =
                crate::builtins::common::gpu_helpers::upload_tensor(provider, &prototype_tensor)
                    .expect("upload prototype");
            prototype = prototype
                .with_provenance(runmat_accelerate_api::GpuHandleProvenance::Explicit)
                .unwrap();

            let dimension_args = || vec![Value::Num(2.0), Value::Num(2.0)];
            let class_selected = |name: &'static str| {
                let mut arguments = dimension_args();
                arguments.push(Value::from("gpuArray"));
                (name, arguments)
            };
            let like_selected = |name: &'static str, mut arguments: Vec<Value>| {
                arguments.push(Value::from("like"));
                arguments.push(Value::GpuTensor(prototype.clone()));
                (name, arguments)
            };
            let cases = [
                class_selected("zeros"),
                class_selected("ones"),
                class_selected("inf"),
                class_selected("nan"),
                class_selected("rand"),
                class_selected("randn"),
                like_selected("randi", vec![Value::Num(9.0), Value::Num(2.0)]),
                like_selected("eye", vec![Value::Num(2.0)]),
                like_selected("true", dimension_args()),
                like_selected("false", dimension_args()),
            ];

            for (name, arguments) in cases {
                let output = call_builtin(name, &arguments)
                    .unwrap_or_else(|error| panic!("{name} explicit constructor failed: {error}"));
                let Value::GpuTensor(handle) = output.value() else {
                    panic!("{name} explicit constructor returned a host value")
                };
                assert!(
                    runmat_accelerate_api::handle_is_explicit(handle),
                    "{name} lost explicit gpuArray intent"
                );
                drop(output);
            }

            drop(prototype);
        });
    }

    fn unique_class_name(prefix: &str) -> String {
        let id = TEST_CLASS_COUNTER.fetch_add(1, Ordering::Relaxed);
        format!("{}_{}", prefix, id)
    }

    #[test]
    fn value_contains_gpu_detects_nested_closure_captures() {
        let value = Value::Closure(Closure::from_data(runmat_value::ClosureData {
            lexical_environment: None,
            function_name: "worker".into(),
            binding: None,
            source_role: runmat_types::FunctionSourceRole::Unknown,
            source_file: None,
            captures: vec![Value::GpuTensor(gpu_fixture(vec![1]))].into(),
        }));
        assert!(value_contains_gpu(&value));
    }

    #[test]
    fn value_contains_gpu_detects_output_list_entries() {
        let value = Value::output_list(vec![
            Value::Num(1.0),
            Value::GpuTensor(gpu_fixture(vec![1])),
        ]);
        assert!(value_contains_gpu(&value));
    }

    #[test]
    fn gather_if_needed_rejects_nested_output_list_before_provider_access() {
        runmat_accelerate_api::clear_provider();
        let _provider_guard = ThreadProviderGuard::set(None);
        let value = Value::output_list(vec![Value::GpuTensor(gpu_fixture(vec![1]))]);
        let err = futures::executor::block_on(gather_if_needed_async(&value))
            .expect_err("transient sequence must fail before provider access");
        assert_eq!(
            err.identifier(),
            Some("RunMat:TransientSequenceBelowDispatch")
        );
    }

    #[test]
    fn gather_if_needed_reports_provider_unavailable_for_closure_capture_gpu() {
        runmat_accelerate_api::clear_provider();
        let _provider_guard = ThreadProviderGuard::set(None);
        let value = Value::Closure(Closure::from_data(runmat_value::ClosureData {
            lexical_environment: None,
            function_name: "worker".into(),
            binding: None,
            source_role: runmat_types::FunctionSourceRole::Unknown,
            source_file: None,
            captures: vec![Value::GpuTensor(gpu_fixture(vec![1]))].into(),
        }));
        let err = futures::executor::block_on(gather_if_needed_async(&value))
            .expect_err("missing provider should fail closure-captured gather");
        assert_eq!(err.identifier(), Some("RunMat:gather:ProviderUnavailable"));
    }

    #[test]
    fn malformed_download_carries_provider_payload_provenance() {
        let handle = gpu_fixture(vec![1, 1]);
        let error = crate::builtins::common::gpu_helpers::provider_payload_mismatch(
            &handle,
            &[2, 1],
            "download shape differs from handle".into(),
        );
        assert_eq!(
            error.identifier(),
            Some("RunMat:gpu:ProviderPayloadMismatch")
        );
        assert!(error
            .source
            .as_deref()
            .is_some_and(|source| source.is::<super::ProviderDownloadPayloadMismatch>()));
        let unavailable = crate::build_runtime_error("provider not registered")
            .with_identifier("RunMat:gather:ProviderUnavailable")
            .build();
        assert!(unavailable.source.is_none());
    }

    #[test]
    fn constructor_fallback_uses_inherited_constructor_metadata_with_semantic_invoker() {
        let parent_name = unique_class_name("runtime_ctor_parent");
        let child_name = unique_class_name("runtime_ctor_child");
        let ctor_fn_name = unique_class_name("runtime_ctor_fn");
        let ctor_fn_name_for_resolver = ctor_fn_name.clone();
        let ctor_fn_name_for_invoker = ctor_fn_name.clone();
        let _resolver_guard = crate::user_functions::install_semantic_function_resolver(Some(
            std::sync::Arc::new(move |name| (name == ctor_fn_name_for_resolver).then_some(10101)),
        ))
        .expect("call-port scope");
        let _invoker_guard = crate::user_functions::install_semantic_function_invoker(Some(
            std::sync::Arc::new(move |function, _args, requested_outputs| {
                assert_eq!(function, 10101);
                assert_eq!(requested_outputs, 1);
                let mut sv = runmat_value::UnboundedStructValueBuilder::new();
                sv.fields.insert("x".to_string(), Value::Num(12.0));
                Box::pin(async move {
                    crate::user_functions::test_support::rooted_single_value_sequence(
                        Value::Struct(sv.finish()),
                    )
                })
            }),
        ))
        .expect("call-port scope");

        let mut parent_methods = HashMap::new();
        parent_methods.insert(
            child_name.clone().into(),
            crate::class_registry::RuntimeMethod {
                name: child_name.clone().into(),
                is_static: true,
                is_abstract: false,
                is_sealed: false,
                access: MemberAccess::Public,
                function_name: ctor_fn_name_for_invoker,
                implicit_class_argument: None,
                role: runmat_types::ClassMethodRole::Constructor,
            },
        );
        crate::class_registry::register_class(crate::class_registry::RuntimeClass {
            events: Default::default(),
            name: parent_name.clone().into(),
            parent: None,
            properties: HashMap::new(),
            methods: parent_methods,
        });
        crate::class_registry::register_class(crate::class_registry::RuntimeClass {
            events: Default::default(),
            name: child_name.clone().into(),
            parent: Some(parent_name.into()),
            properties: HashMap::new(),
            methods: HashMap::new(),
        });

        let out =
            call_builtin(&child_name, &[]).expect("inherited static constructor should dispatch");
        let Value::Object(obj) = out.value() else {
            panic!("expected object from constructor dispatch");
        };
        assert_eq!(obj.class_name().display_name(), child_name);
        assert_eq!(obj.properties().get("x"), Some(&Value::Num(12.0)));
    }

    #[test]
    fn constructor_fallback_defaults_when_constructor_is_private_or_unavailable() {
        let private_class_name = unique_class_name("runtime_ctor_private");
        let mut private_methods = HashMap::new();
        private_methods.insert(
            private_class_name.clone().into(),
            crate::class_registry::RuntimeMethod {
                name: private_class_name.clone().into(),
                is_static: true,
                is_abstract: false,
                is_sealed: false,
                access: MemberAccess::Private,
                function_name: "Point.origin".into(),
                implicit_class_argument: None,
                role: runmat_types::ClassMethodRole::Ordinary,
            },
        );
        crate::class_registry::register_class(crate::class_registry::RuntimeClass {
            events: Default::default(),
            name: private_class_name.clone().into(),
            parent: None,
            properties: HashMap::new(),
            methods: private_methods,
        });
        let err = call_builtin(&private_class_name, &[])
            .expect_err("private constructor should enforce access before default fallback");
        assert_eq!(err.identifier(), Some("RunMat:MethodPrivate"));

        let public_class_name = unique_class_name("runtime_ctor_public_no_semantic");
        let mut public_methods = HashMap::new();
        public_methods.insert(
            public_class_name.clone().into(),
            crate::class_registry::RuntimeMethod {
                name: public_class_name.clone().into(),
                is_static: true,
                is_abstract: false,
                is_sealed: false,
                access: MemberAccess::Public,
                function_name: unique_class_name("runtime_ctor_missing_body"),
                implicit_class_argument: None,
                role: runmat_types::ClassMethodRole::Constructor,
            },
        );
        crate::class_registry::register_class(crate::class_registry::RuntimeClass {
            events: Default::default(),
            name: public_class_name.clone().into(),
            parent: None,
            properties: HashMap::new(),
            methods: public_methods,
        });

        let out = call_builtin(&public_class_name, &[])
            .expect("public ctor metadata without semantic body should default-construct");
        let Value::Object(obj) = out.value() else {
            panic!("expected object result");
        };
        assert_eq!(obj.class_name().display_name(), public_class_name);
    }

    #[test]
    fn dotted_static_method_name_dispatches_to_registered_class_method() {
        let class_name = unique_class_name("runtime_static_dispatch");
        let fn_name = unique_class_name("runtime_static_fn");
        crate::class_registry::register_class(crate::class_registry::RuntimeClass {
            events: Default::default(),
            name: class_name.clone().into(),
            parent: None,
            properties: HashMap::new(),
            methods: {
                let mut methods = HashMap::new();
                methods.insert(
                    "zero".into(),
                    crate::class_registry::RuntimeMethod {
                        name: "zero".into(),
                        is_static: true,
                        is_abstract: false,
                        is_sealed: false,
                        access: MemberAccess::Public,
                        function_name: fn_name.clone(),
                        implicit_class_argument: None,
                        role: runmat_types::ClassMethodRole::Ordinary,
                    },
                );
                let mut private = methods[&runmat_types::MethodName::from("zero")].clone();
                private.name = "privateZero".into();
                private.access = MemberAccess::Private;
                methods.insert(private.name.clone(), private);
                let mut instance = methods[&runmat_types::MethodName::from("zero")].clone();
                instance.name = "instanceZero".into();
                instance.is_static = false;
                methods.insert(instance.name.clone(), instance);
                let mut unavailable = methods[&runmat_types::MethodName::from("zero")].clone();
                unavailable.name = "unavailableZero".into();
                unavailable.function_name = unique_class_name("unavailable_static_body");
                methods.insert(unavailable.name.clone(), unavailable);
                methods
            },
        });

        let fn_name_for_resolver = fn_name.clone();
        let _resolver_guard = crate::user_functions::install_semantic_function_resolver(Some(
            std::sync::Arc::new(move |name| (name == fn_name_for_resolver).then_some(20202)),
        ))
        .expect("call-port scope");
        let calls = std::sync::Arc::new(AtomicU64::new(0));
        let invoker_calls = calls.clone();
        let _invoker_guard = crate::user_functions::install_semantic_function_invoker(Some(
            std::sync::Arc::new(move |function, _args, requested_outputs| {
                assert_eq!(function, 20202);
                invoker_calls.fetch_add(1, Ordering::SeqCst);
                Box::pin(async move {
                    if requested_outputs == 3 {
                        Err(crate::build_runtime_error("selected static body failed")
                            .with_identifier("RunMat:StaticFixture:Failure")
                            .with_source(std::io::Error::other("original static body source"))
                            .with_gpu_gather_retry(crate::GpuGatherRetry::Requested)
                            .build())
                    } else if requested_outputs == 1 {
                        crate::user_functions::test_support::rooted_single_value_sequence(
                            Value::Num(77.0),
                        )
                    } else {
                        runmat_value::ValueSequence::comma_separated(
                            (0..requested_outputs)
                                .map(|index| Value::Num(index as f64))
                                .collect(),
                        )
                        .map_err(crate::sequence::sequence_error_to_runtime)
                        .and_then(|sequence| {
                            crate::execution::RootedValueSequence::new(sequence)
                                .map_err(|error| error.into_runtime_error())
                        })
                    }
                })
            }),
        ))
        .expect("call-port scope");

        let out = call_builtin(&format!("{class_name}.zero"), &[])
            .expect("dotted static class method call should dispatch");
        assert_eq!(out, Value::Num(77.0));
        let child = unique_class_name("runtime_static_child");
        crate::class_registry::register_class(crate::class_registry::RuntimeClass {
            events: Default::default(),
            name: child.clone().into(),
            parent: Some(class_name.clone().into()),
            properties: HashMap::new(),
            methods: HashMap::new(),
        });
        for owner in [&class_name, &child] {
            for requested in [0, 1, 2] {
                let before = calls.load(Ordering::SeqCst);
                let result = block_on(crate::call_feval_sequence_async_with_outputs(
                    Value::from(format!("{owner}.zero")),
                    &[],
                    requested,
                ))
                .expect("qualified feval selects the registered inherited static body");
                let expected = if requested == 1 {
                    vec![Value::Num(77.0)]
                } else {
                    (0..requested).map(|i| Value::Num(i as f64)).collect()
                };
                assert_eq!(
                    result.as_slice(),
                    expected.as_slice(),
                    "qualified static call must preserve requested cardinality and values"
                );
                assert_eq!(calls.load(Ordering::SeqCst), before + 1);
            }
        }
        for member in ["privateZero", "instanceZero", "unavailableZero", "abs"] {
            let before = calls.load(Ordering::SeqCst);
            let error = block_on(crate::call_feval_sequence_async_with_outputs(
                Value::from(format!("{class_name}.{member}")),
                &[],
                1,
            ))
            .expect_err("qualified spelling is not executable public static authority");
            assert_eq!(error.identifier(), Some("RunMat:UndefinedFunction"));
            assert_eq!(calls.load(Ordering::SeqCst), before);
        }
        let before = calls.load(Ordering::SeqCst);
        let error = block_on(crate::call_feval_sequence_async_with_outputs(
            Value::from(format!("{class_name}.zero")),
            &[],
            3,
        ))
        .expect_err("a selected static error cannot become fallback or gather retry");
        assert_eq!(calls.load(Ordering::SeqCst), before + 1);
        assert_eq!(error.identifier(), Some("RunMat:StaticFixture:Failure"));
        assert_eq!(error.message(), "selected static body failed");
        assert_eq!(error.gpu_gather_retry, crate::GpuGatherRetry::Never);
        assert!(error.source.as_deref().is_some_and(|source| source
            .downcast_ref::<std::io::Error>()
            .is_some_and(|source| source.to_string() == "original static body source")));
        for requested_outputs in [0, 2] {
            let out = block_on(super::try_call_registered_static_method(
                &format!("{class_name}.zero"),
                &crate::builtins::common::test_support::builtin_arguments(Vec::new()),
                Some(requested_outputs),
                None,
            ))
            .unwrap()
            .unwrap();
            assert_eq!(
                out,
                runmat_value::ValueSequence::comma_separated(
                    (0..requested_outputs)
                        .map(|index| Value::Num(index as f64))
                        .collect(),
                )
                .unwrap()
            );
        }
    }

    #[test]
    fn registered_static_observation_and_handles_keep_exact_method_binding() {
        let runtime = crate::context::RuntimeContext::new(std::rc::Rc::new(
            crate::execution::RuntimeExecutionService::new(),
        ));
        let _scope = runtime.enter();
        let class = runmat_types::ClassIdentity::from(unique_class_name("bound_static"));
        let method = runmat_types::MethodName::from("build");
        let origin = runmat_execution::FunctionBindingOrigin {
            program: runmat_execution::Digest::sha256("bound static method fixture"),
            function: runmat_types::ProgramFunctionId(42),
        };
        let binding = runmat_value::BoundFunctionBinding {
            function: 142,
            authority: runmat_value::BoundFunctionAuthority::Portable { origin },
        };
        let instance_method = runmat_types::MethodName::from("advance");
        let instance_binding = runmat_value::BoundFunctionBinding {
            function: 143,
            authority: runmat_value::BoundFunctionAuthority::Portable {
                origin: runmat_execution::FunctionBindingOrigin {
                    program: origin.program,
                    function: runmat_types::ProgramFunctionId(43),
                },
            },
        };
        let declaration = crate::class_registry::RuntimeMethod {
            name: method.clone(),
            is_static: true,
            is_abstract: false,
            is_sealed: false,
            access: MemberAccess::Public,
            function_name: "unpublished_method_body".into(),
            implicit_class_argument: None,
            role: runmat_types::ClassMethodRole::Ordinary,
        };
        let catalog = crate::user_functions::install_source_function_catalog(Some(
            std::sync::Arc::new(vec![
                crate::user_functions::SourceFunctionInfo {
                    source_id: runmat_types::SourceId(1),
                    name: "unpublished_method_body".into(),
                    function: binding.function,
                    role: runmat_types::FunctionSourceRole::Method,
                    binding_origin: Some(origin),
                    source_file: None,
                    definition_owners: None,
                },
                crate::user_functions::SourceFunctionInfo {
                    source_id: runmat_types::SourceId(1),
                    name: "unpublished_instance_body".into(),
                    function: instance_binding.function,
                    role: runmat_types::FunctionSourceRole::Method,
                    binding_origin: instance_binding.authority.portable_origin(),
                    source_file: None,
                    definition_owners: None,
                },
            ]),
        ))
        .expect("install exact method source catalog");
        crate::class_registry::register_checked_classes(vec![
            crate::class_registry::CheckedClassRegistration {
                class: crate::class_registry::RuntimeClass {
                    name: class.clone(),
                    parent: None,
                    properties: HashMap::new(),
                    methods: HashMap::from([
                        (method.clone(), declaration),
                        (
                            instance_method.clone(),
                            crate::class_registry::RuntimeMethod {
                                name: instance_method.clone(),
                                is_static: false,
                                is_abstract: false,
                                is_sealed: false,
                                access: MemberAccess::Public,
                                function_name: "unpublished_instance_body".into(),
                                implicit_class_argument: None,
                                role: runmat_types::ClassMethodRole::Ordinary,
                            },
                        ),
                    ]),
                    events: HashMap::new(),
                },
                method_bindings: HashMap::from([
                    (method.clone(), binding.clone()),
                    (instance_method, instance_binding.clone()),
                ]),
                is_sealed: false,
                is_abstract: false,
                enumerations: Vec::new(),
            },
        ])
        .expect("register exact static body");
        let qualified = format!("{class}.build");
        let resolved = super::observe_registered_static_method(&qualified)
            .expect("registered static method is executable by exact binding");
        assert!(matches!(
            resolved.target,
            super::RegisteredStaticMethodTarget::Bound(ref selected) if selected == &binding
        ));
        let Value::Closure(loaded) =
            crate::object::resolve::load_static_member(&class, "build", None)
                .expect("static member closure")
        else {
            panic!("static method must load a closure");
        };
        assert_eq!(loaded.binding.as_ref(), Some(&binding));
        let _extensions = crate::compatibility::push_runmat_extensions_enabled(true);
        let Value::Closure(handle) = crate::builtins::introspection::getmethod::dispatch_getmethod(
            Value::ClassRef(class.clone().into()),
            "build".into(),
        )
        .expect("getmethod closure") else {
            panic!("getmethod must return a closure");
        };
        assert_eq!(handle.binding.as_ref(), Some(&binding));
        let selected = super::selection::select_instance_method(
            "advance",
            crate::call::FunctionArgumentView::from_slice(&[Value::Object(
                runmat_value::ObjectInstance::new(class.clone()),
            )]),
        )
        .expect("registered instance selection")
        .expect("instance method declaration");
        assert!(matches!(
            selected.routes.as_slice(),
            [super::selection::InstanceMethodRoute::Bound(selected)]
                if selected == &instance_binding
        ));
        let declaration = crate::class_registry::get_class(&class).expect("registered declaration");
        let registration = |binding| crate::class_registry::CheckedClassRegistration {
            class: declaration.clone(),
            method_bindings: HashMap::from([(method.clone(), binding)]),
            is_sealed: false,
            is_abstract: false,
            enumerations: Vec::new(),
        };
        let generation = crate::class_registry::class_registry_generation();
        for forged in [
            runmat_value::BoundFunctionBinding::portable(42, origin),
            runmat_value::BoundFunctionBinding::portable(
                binding.function,
                runmat_execution::FunctionBindingOrigin {
                    program: runmat_execution::Digest::sha256("foreign method program"),
                    function: origin.function,
                },
            ),
        ] {
            let error = crate::class_registry::register_checked_classes(vec![registration(forged)])
                .expect_err("matching name or portable ordinal is not method authority");
            assert_eq!(error.identifier(), Some("RunMat:ClassMethodBinding"));
            assert_eq!(
                crate::class_registry::class_registry_generation(),
                generation
            );
        }
        drop(catalog);
        let error = crate::class_registry::register_checked_classes(vec![registration(binding)])
            .expect_err("a portable method binding requires its original source issuance");
        assert_eq!(error.identifier(), Some("RunMat:ClassMethodBinding"));
        assert_eq!(
            crate::class_registry::class_registry_generation(),
            generation
        );
    }
}

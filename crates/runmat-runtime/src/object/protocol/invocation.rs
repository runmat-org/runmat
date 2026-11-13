use super::ResolvedObjectMethodTarget;
use super::{ObjectProtocolCallingConvention, ProtocolResolution, ResolvedObjectMethod};
use crate::call::descriptor::{CallableCallKind, CallableDescriptor};
use crate::context::RuntimeContext;
use crate::execution::{RootedValue, RootedValueSequence};
use crate::object::indexing::ObjectSubscriptPath;
use crate::runtime_error::semantic_error;
use crate::RuntimeError;
use runmat_value::Value;
use std::future::Future;
use std::rc::Rc;
mod deferred_native;

pub fn invoke_resolved_object_protocol<'resolution>(
    resolution: &'resolution ProtocolResolution,
    base: Value,
    path: ObjectSubscriptPath,
    requested_outputs: usize,
) -> impl Future<Output = Result<RootedValueSequence, RuntimeError>> + use<'resolution> {
    invoke_resolved_object_protocol_with_runtime(
        crate::context::legacy::active(),
        resolution,
        base,
        path,
        requested_outputs,
    )
}

pub fn invoke_resolved_object_protocol_in_context<'runtime, 'resolution>(
    runtime: &'runtime RuntimeContext,
    resolution: &'resolution ProtocolResolution,
    base: Value,
    path: ObjectSubscriptPath,
    requested_outputs: usize,
) -> impl Future<Output = Result<RootedValueSequence, RuntimeError>> + use<'runtime, 'resolution> {
    invoke_resolved_object_protocol_with_context(
        Some(runtime),
        resolution,
        base,
        path,
        requested_outputs,
    )
}

pub(crate) fn invoke_resolved_object_protocol_with_context<'runtime, 'resolution>(
    runtime: Option<&'runtime RuntimeContext>,
    resolution: &'resolution ProtocolResolution,
    base: Value,
    path: ObjectSubscriptPath,
    requested_outputs: usize,
) -> impl Future<Output = Result<RootedValueSequence, RuntimeError>> + use<'runtime, 'resolution> {
    invoke_resolved_object_protocol_with_runtime(
        runtime.cloned(),
        resolution,
        base,
        path,
        requested_outputs,
    )
}
fn invoke_resolved_object_protocol_with_runtime<'resolution>(
    runtime: Option<RuntimeContext>,
    resolution: &'resolution ProtocolResolution,
    base: Value,
    path: ObjectSubscriptPath,
    requested_outputs: usize,
) -> impl Future<Output = Result<RootedValueSequence, RuntimeError>> + use<'resolution> {
    let inputs = crate::context::with_original_scope(runtime.as_ref(), || {
        retain_protocol_inputs(runtime.as_ref(), &base, &path)
    });
    let selected = runtime.clone();
    let work = async move {
        let runtime = runtime.as_ref();
        let _inputs = inputs?;
        invoke_resolved_object_protocol_in_scope(runtime, resolution, base, path, requested_outputs)
            .await
    };
    super::context::scope(selected.as_ref(), work)
}

async fn invoke_resolved_object_protocol_in_scope(
    runtime: Option<&RuntimeContext>,
    resolution: &ProtocolResolution,
    base: Value,
    path: ObjectSubscriptPath,
    requested_outputs: usize,
) -> Result<RootedValueSequence, RuntimeError> {
    let Some(method) = resolution else {
        return Err(semantic_error(
            "MissingObjectProtocol",
            "object indexing does not define the requested protocol",
        ));
    };
    let arguments = invocation_arguments(runtime, method, base, path)?;
    invoke_resolved_object_method_retained(runtime, method, arguments, requested_outputs).await
}

/// Invoke a prepared assignment protocol with the exact comma-separated RHS.
/// The sequence remains an argument list and can never be stored as one
/// language value.
pub fn invoke_resolved_object_assignment<'method>(
    method: &'method ResolvedObjectMethod,
    base: Value,
    path: ObjectSubscriptPath,
    values: Vec<Value>,
) -> impl Future<Output = Result<RootedValue, RuntimeError>> + use<'method> {
    let values = runmat_value::ValueSequence::comma_separated(values)
        .map_err(crate::sequence::sequence_error_to_runtime);
    invoke_resolved_object_assignment_result(
        crate::context::legacy::active(),
        method,
        base,
        path,
        values,
    )
}

pub fn invoke_resolved_object_assignment_in_context<'runtime, 'method>(
    runtime: &'runtime RuntimeContext,
    method: &'method ResolvedObjectMethod,
    base: Value,
    path: ObjectSubscriptPath,
    values: Vec<Value>,
) -> impl Future<Output = Result<RootedValue, RuntimeError>> + use<'runtime, 'method> {
    let values = crate::context::with_original_scope(Some(runtime), || {
        runmat_value::ValueSequence::comma_separated(values)
            .map_err(crate::sequence::sequence_error_to_runtime)
    });
    invoke_resolved_object_assignment_result(Some(runtime.clone()), method, base, path, values)
}

pub(crate) fn invoke_resolved_object_assignment_sequence<'method>(
    method: &'method ResolvedObjectMethod,
    base: Value,
    path: ObjectSubscriptPath,
    values: runmat_value::ValueSequence,
) -> impl Future<Output = Result<RootedValue, RuntimeError>> + use<'method> {
    invoke_resolved_object_assignment_with_runtime(
        crate::context::legacy::active(),
        method,
        base,
        path,
        values,
    )
}

pub fn invoke_resolved_object_assignment_sequence_in_context<'runtime, 'method>(
    runtime: &'runtime RuntimeContext,
    method: &'method ResolvedObjectMethod,
    base: Value,
    path: ObjectSubscriptPath,
    values: runmat_value::ValueSequence,
) -> impl Future<Output = Result<RootedValue, RuntimeError>> + use<'runtime, 'method> {
    invoke_resolved_object_assignment_with_context(Some(runtime), method, base, path, values)
}

pub(crate) fn invoke_resolved_object_assignment_with_context<'runtime, 'method>(
    runtime: Option<&'runtime RuntimeContext>,
    method: &'method ResolvedObjectMethod,
    base: Value,
    path: ObjectSubscriptPath,
    values: runmat_value::ValueSequence,
) -> impl Future<Output = Result<RootedValue, RuntimeError>> + use<'runtime, 'method> {
    invoke_resolved_object_assignment_with_runtime(runtime.cloned(), method, base, path, values)
}
fn invoke_resolved_object_assignment_with_runtime<'method>(
    runtime: Option<RuntimeContext>,
    method: &'method ResolvedObjectMethod,
    base: Value,
    path: ObjectSubscriptPath,
    values: runmat_value::ValueSequence,
) -> impl Future<Output = Result<RootedValue, RuntimeError>> + use<'method> {
    invoke_resolved_object_assignment_result(runtime, method, base, path, Ok(values))
}
fn invoke_resolved_object_assignment_result<'method>(
    runtime: Option<RuntimeContext>,
    method: &'method ResolvedObjectMethod,
    base: Value,
    path: ObjectSubscriptPath,
    values: Result<runmat_value::ValueSequence, RuntimeError>,
) -> impl Future<Output = Result<RootedValue, RuntimeError>> + use<'method> {
    let (inputs, values) = crate::context::with_original_scope(runtime.as_ref(), || {
        let inputs = retain_protocol_inputs(runtime.as_ref(), &base, &path);
        let values = values.and_then(|values| match runtime.as_ref() {
            Some(runtime) => RootedValueSequence::in_context(values, runtime),
            None => RootedValueSequence::new(values).map_err(|e| e.into_runtime_error()),
        });
        (inputs, values)
    });
    let selected = runtime.clone();
    let work = async move {
        let runtime = runtime.as_ref();
        let _inputs = inputs?;
        invoke_resolved_object_assignment_in_scope(runtime, method, base, path, values?).await
    };
    super::context::scope(selected.as_ref(), work)
}

async fn invoke_resolved_object_assignment_in_scope(
    runtime: Option<&RuntimeContext>,
    method: &ResolvedObjectMethod,
    base: Value,
    path: ObjectSubscriptPath,
    values: RootedValueSequence,
) -> Result<RootedValue, RuntimeError> {
    if path.has_deferred_end() {
        return deferred_native::assign(runtime, method, base, path, values).await;
    }
    let arguments = invocation_arguments(runtime, method, base, path)?;
    let count = arguments
        .len()
        .checked_add(values.len())
        .ok_or_else(|| semantic_error("CallableArguments", "argument count overflows"))?;
    let combined = initialize_arguments(runtime, count, |index| {
        Ok(if index < arguments.len() {
            arguments.get(index).expect("prefix argument").clone()
        } else {
            values.as_slice()[index - arguments.len()].clone()
        })
    })?;
    invoke_resolved_object_method_retained(runtime, method, combined, 1)
        .await?
        .try_into_single()
}

pub fn invoke_resolved_object_method<'method>(
    method: &'method ResolvedObjectMethod,
    arguments: Vec<Value>,
    requested_outputs: usize,
) -> impl Future<Output = Result<RootedValueSequence, RuntimeError>> + use<'method> {
    invoke_resolved_object_method_with_runtime(
        crate::context::legacy::active(),
        method,
        arguments,
        requested_outputs,
    )
}

pub fn invoke_resolved_object_method_in_context<'runtime, 'method>(
    runtime: &'runtime RuntimeContext,
    method: &'method ResolvedObjectMethod,
    arguments: Vec<Value>,
    requested_outputs: usize,
) -> impl Future<Output = Result<RootedValueSequence, RuntimeError>> + use<'runtime, 'method> {
    invoke_resolved_object_method_with_context(Some(runtime), method, arguments, requested_outputs)
}

fn invoke_resolved_object_method_with_context<'runtime, 'method>(
    runtime: Option<&'runtime RuntimeContext>,
    method: &'method ResolvedObjectMethod,
    arguments: Vec<Value>,
    requested_outputs: usize,
) -> impl Future<Output = Result<RootedValueSequence, RuntimeError>> + use<'runtime, 'method> {
    invoke_resolved_object_method_with_runtime(
        runtime.cloned(),
        method,
        arguments,
        requested_outputs,
    )
}
fn invoke_resolved_object_method_with_runtime<'method>(
    runtime: Option<RuntimeContext>,
    method: &'method ResolvedObjectMethod,
    arguments: Vec<Value>,
    requested_outputs: usize,
) -> impl Future<Output = Result<RootedValueSequence, RuntimeError>> + use<'method> {
    let arguments = crate::context::with_original_scope(runtime.as_ref(), || {
        retained_arguments(runtime.as_ref(), arguments)
    });
    let selected = runtime.clone();
    let work = async move {
        let runtime = runtime.as_ref();
        invoke_resolved_object_method_in_scope(runtime, method, arguments?, requested_outputs).await
    };
    super::context::scope(selected.as_ref(), work)
}

pub(crate) fn invoke_resolved_object_method_retained<'a>(
    runtime: Option<&'a RuntimeContext>,
    method: &'a ResolvedObjectMethod,
    arguments: crate::call::BuiltinArguments,
    requested_outputs: usize,
) -> impl Future<Output = Result<RootedValueSequence, RuntimeError>> + 'a {
    super::context::scope(
        runtime,
        invoke_resolved_object_method_in_scope(runtime, method, arguments, requested_outputs),
    )
}
fn retained_arguments(
    runtime: Option<&RuntimeContext>,
    arguments: Vec<Value>,
) -> Result<crate::call::BuiltinArguments, RuntimeError> {
    match runtime {
        Some(runtime) => crate::execution::RootedValues::in_context(arguments, runtime),
        None => crate::execution::RootedValues::new(arguments).map_err(|e| e.into_runtime_error()),
    }
    .map(Into::into)
}
fn initialize_arguments(
    runtime: Option<&RuntimeContext>,
    count: usize,
    mut initialize: impl FnMut(usize) -> Result<Value, RuntimeError>,
) -> Result<crate::call::BuiltinArguments, RuntimeError> {
    match runtime {
        Some(runtime) => {
            crate::execution::RootedValues::try_initialize_in_context(count, runtime, initialize)
        }
        None => {
            let mut values = runmat_value::exact_capacity_vec(count)
                .map_err(|e| crate::object::indexing::allocation_error(None, e.into()))?;
            for index in 0..count {
                values.push(initialize(index)?);
            }
            crate::execution::RootedValues::new(values).map_err(|e| e.into_runtime_error())
        }
    }
    .map(Into::into)
}
async fn invoke_resolved_object_method_in_scope(
    runtime: Option<&RuntimeContext>,
    method: &ResolvedObjectMethod,
    arguments: crate::call::BuiltinArguments,
    requested_outputs: usize,
) -> Result<RootedValueSequence, RuntimeError> {
    if !super::resolved_method_is_current(method) {
        return Err(semantic_error(
            "StaleObjectProtocolResolution",
            "object protocol binding changed after the operation was prepared",
        ));
    }
    match &method.target {
        ResolvedObjectMethodTarget::Selected(facts, owner, binding) => {
            let selected = runtime
                .cloned()
                .or_else(crate::context::legacy::active)
                .ok_or_else(|| {
                    semantic_error(
                        "SelectedObjectMethodContextMissing",
                        "selected object method has no issuing runtime context",
                    )
                })?;
            let selected = selected
                .with_selected_object_class(Rc::clone(facts))
                .with_selected_method_body(Rc::clone(facts), owner.clone(), binding.clone());
            let descriptor = CallableDescriptor::bound(
                binding.clone(),
                Vec::new(),
                requested_outputs,
                CallableCallKind::Direct,
            )
            .with_retained_arguments(arguments)?;
            crate::call::descriptor::execute_retained_callable_descriptor(
                Some(&selected),
                descriptor,
            )
            .await
        }
        ResolvedObjectMethodTarget::RegisteredBound(binding) => {
            let descriptor = CallableDescriptor::bound(
                binding.clone(),
                Vec::new(),
                requested_outputs,
                CallableCallKind::Direct,
            )
            .with_retained_arguments(arguments)?;
            match runtime {
                Some(runtime) => {
                    crate::call::descriptor::execute_retained_callable_descriptor(
                        Some(runtime),
                        descriptor,
                    )
                    .await
                }
                None => {
                    crate::call::descriptor::execute_retained_callable_descriptor(None, descriptor)
                        .await
                }
            }
        }
        ResolvedObjectMethodTarget::Named(callable) => {
            let descriptor = CallableDescriptor::resolved(
                callable.clone(),
                Vec::new(),
                requested_outputs,
                method.fallback,
                CallableCallKind::Direct,
            )
            .with_retained_arguments(arguments)?;
            match runtime {
                Some(runtime) => {
                    crate::call::descriptor::execute_retained_callable_descriptor(
                        Some(runtime),
                        descriptor,
                    )
                    .await
                }
                None => {
                    crate::call::descriptor::execute_retained_callable_descriptor(None, descriptor)
                        .await
                }
            }
        }
    }
}

/// Invoke an object-defined `end(obj, k, n)` selector exactly once. `k` and
/// `n` use MATLAB's one-based component convention.
pub fn invoke_object_end<'access>(
    receiver: Value,
    component_index: usize,
    component_count: usize,
    access: &'access super::ObjectAccessContext,
) -> impl Future<Output = Result<Option<RootedValue>, RuntimeError>> + use<'access> {
    let runtime = crate::context::legacy::active();
    async move {
        invoke_object_end_with_context(
            runtime.as_ref(),
            receiver,
            component_index,
            component_count,
            access,
        )
        .await
    }
}

pub fn invoke_object_end_in_context<'runtime, 'access>(
    runtime: &'runtime RuntimeContext,
    receiver: Value,
    component_index: usize,
    component_count: usize,
    access: &'access super::ObjectAccessContext,
) -> impl Future<Output = Result<Option<RootedValue>, RuntimeError>> + use<'runtime, 'access> {
    invoke_object_end_with_context(
        Some(runtime),
        receiver,
        component_index,
        component_count,
        access,
    )
}

fn invoke_object_end_with_context<'runtime, 'access>(
    runtime: Option<&'runtime RuntimeContext>,
    receiver: Value,
    component_index: usize,
    component_count: usize,
    access: &'access super::ObjectAccessContext,
) -> impl Future<Output = Result<Option<RootedValue>, RuntimeError>> + use<'runtime, 'access> {
    let work =
        invoke_object_end_in_scope(runtime, receiver, component_index, component_count, access);
    super::context::scope(runtime, work)
}

async fn invoke_object_end_in_scope(
    runtime: Option<&RuntimeContext>,
    receiver: Value,
    component_index: usize,
    component_count: usize,
    access: &super::ObjectAccessContext,
) -> Result<Option<RootedValue>, RuntimeError> {
    let resolution = super::resolve_object_protocol(&receiver, super::ObjectProtocol::End, access)?;
    let Some(method) = resolution else {
        return Ok(None);
    };
    invoke_prepared_object_end_with_context(
        runtime,
        &method,
        receiver,
        component_index,
        component_count,
    )
    .await
    .map(Some)
}

pub fn invoke_prepared_object_end<'method>(
    method: &'method ResolvedObjectMethod,
    receiver: Value,
    component_index: usize,
    component_count: usize,
) -> impl Future<Output = Result<RootedValue, RuntimeError>> + use<'method> {
    let runtime = crate::context::legacy::active();
    async move {
        super::context::scope(
            runtime.as_ref(),
            invoke_prepared_object_end_with_context(
                runtime.as_ref(),
                method,
                receiver,
                component_index,
                component_count,
            ),
        )
        .await
    }
}

pub(super) async fn invoke_prepared_object_end_with_context(
    runtime: Option<&RuntimeContext>,
    method: &ResolvedObjectMethod,
    receiver: Value,
    component_index: usize,
    component_count: usize,
) -> Result<RootedValue, RuntimeError> {
    let k = component_index.checked_add(1).ok_or_else(|| {
        semantic_error(
            "ObjectEndIndexOverflow",
            "object end component index overflowed",
        )
    })?;
    const MAX_EXACT_DOUBLE_INTEGER: u64 = 1u64 << 53;
    let k = u64::try_from(k).map_err(|_| {
        semantic_error(
            "ObjectEndIndexOverflow",
            "object end component index is too large",
        )
    })?;
    let component_count = u64::try_from(component_count).map_err(|_| {
        semantic_error(
            "ObjectEndIndexOverflow",
            "object end component count is too large",
        )
    })?;
    if k > MAX_EXACT_DOUBLE_INTEGER || component_count > MAX_EXACT_DOUBLE_INTEGER {
        return Err(semantic_error(
            "ObjectEndIndexOverflow",
            "object end component metadata exceeds exact double range",
        ));
    }
    let arguments = crate::call::BuiltinArguments::from_borrowed(
        &[
            receiver,
            Value::Num(k as f64),
            Value::Num(component_count as f64),
        ],
        runtime,
    )?;
    invoke_resolved_object_method_retained(runtime, method, arguments, 1)
        .await?
        .try_into_single()
}

pub(crate) fn retain_protocol_inputs(
    runtime: Option<&RuntimeContext>,
    base: &Value,
    path: &ObjectSubscriptPath,
) -> Result<crate::execution::RootedValueLease, RuntimeError> {
    let values = std::iter::once(base).chain(path.steps().iter().flat_map(|step| {
        let components: &[crate::object::indexing::ObjectIndexComponent] = match step.selector() {
            crate::object::indexing::ObjectIndexSelector::IndexValues { components } => components,
            _ => &[],
        };
        components.iter().filter_map(|component| match component {
            crate::object::indexing::ObjectIndexComponent::Value(value) => Some(value),
            crate::object::indexing::ObjectIndexComponent::Colon
            | crate::object::indexing::ObjectIndexComponent::End => None,
        })
    }));
    match runtime {
        Some(runtime) => {
            crate::execution::RootedValueLease::for_values_iterator_in_context(values, runtime)
        }
        None => crate::execution::RootedValueLease::for_values(values)
            .map_err(|e| e.into_runtime_error()),
    }
}

fn invocation_arguments(
    runtime: Option<&RuntimeContext>,
    method: &ResolvedObjectMethod,
    base: Value,
    path: ObjectSubscriptPath,
) -> Result<crate::call::BuiltinArguments, RuntimeError> {
    match method.convention {
        ObjectProtocolCallingConvention::StandardSubstruct => {
            if let Some(runtime) = runtime {
                let descriptor = path.to_standard_substruct_value_in_context(runtime)?;
                crate::call::BuiltinArguments::from_borrowed(
                    &[base, descriptor.value().clone()],
                    Some(runtime),
                )
            } else {
                let descriptor = path.to_standard_substruct_value()?;
                crate::call::BuiltinArguments::from_borrowed(&[base, descriptor], None)
            }
        }
        ObjectProtocolCallingConvention::DirectArguments => Err(semantic_error(
            "InvalidObjectProtocolInvocation",
            "this object protocol uses direct typed arguments, not a substruct descriptor",
        )),
    }
}

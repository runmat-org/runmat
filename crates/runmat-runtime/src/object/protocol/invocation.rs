use super::{ObjectProtocolCallingConvention, ProtocolResolution, ResolvedObjectMethod};
use crate::call::descriptor::{
    execute_callable_descriptor, execute_callable_descriptor_in_context, CallableCallKind,
    CallableDescriptor,
};
use crate::context::RuntimeContext;
use crate::object::indexing::ObjectSubscriptPath;
use crate::runtime_error::semantic_error;
use crate::sequence::{ResolveValueSequence, SequenceResolutionContext};
use crate::RuntimeError;
use runmat_value::{Value, ValueSequence};

pub async fn invoke_resolved_object_protocol(
    resolution: &ProtocolResolution,
    base: Value,
    path: ObjectSubscriptPath,
    requested_outputs: usize,
) -> Result<ValueSequence, RuntimeError> {
    invoke_resolved_object_protocol_with_context(None, resolution, base, path, requested_outputs)
        .await
}

pub async fn invoke_resolved_object_protocol_in_context(
    runtime: &RuntimeContext,
    resolution: &ProtocolResolution,
    base: Value,
    path: ObjectSubscriptPath,
    requested_outputs: usize,
) -> Result<ValueSequence, RuntimeError> {
    invoke_resolved_object_protocol_with_context(
        Some(runtime),
        resolution,
        base,
        path,
        requested_outputs,
    )
    .await
}

async fn invoke_resolved_object_protocol_with_context(
    runtime: Option<&RuntimeContext>,
    resolution: &ProtocolResolution,
    base: Value,
    path: ObjectSubscriptPath,
    requested_outputs: usize,
) -> Result<ValueSequence, RuntimeError> {
    let ProtocolResolution::Method(method) = resolution else {
        return Err(semantic_error(
            "MissingObjectProtocol",
            "object indexing does not define the requested protocol",
        ));
    };
    let arguments = invocation_arguments(method, base, path)?;
    invoke_resolved_object_method_with_context(runtime, method, arguments, requested_outputs).await
}

/// Invoke a prepared assignment protocol with the exact comma-separated RHS.
/// The sequence remains an argument list and can never be stored as one
/// language value.
pub async fn invoke_resolved_object_assignment(
    method: &ResolvedObjectMethod,
    base: Value,
    path: ObjectSubscriptPath,
    values: Vec<Value>,
) -> Result<Value, RuntimeError> {
    invoke_resolved_object_assignment_with_context(None, method, base, path, values).await
}

pub async fn invoke_resolved_object_assignment_in_context(
    runtime: &RuntimeContext,
    method: &ResolvedObjectMethod,
    base: Value,
    path: ObjectSubscriptPath,
    values: Vec<Value>,
) -> Result<Value, RuntimeError> {
    invoke_resolved_object_assignment_with_context(Some(runtime), method, base, path, values).await
}

async fn invoke_resolved_object_assignment_with_context(
    runtime: Option<&RuntimeContext>,
    method: &ResolvedObjectMethod,
    base: Value,
    path: ObjectSubscriptPath,
    values: Vec<Value>,
) -> Result<Value, RuntimeError> {
    let mut arguments = invocation_arguments(method, base, path)?;
    arguments.extend(values);
    let mut values = invoke_resolved_object_method_with_context(runtime, method, arguments, 1)
        .await?
        .resolve(
            runmat_types::SequenceUse::RequireSingle,
            SequenceResolutionContext::default(),
        )?;
    values.pop().ok_or_else(|| {
        semantic_error(
            "ObjectAssignmentOutput",
            "object assignment protocol did not return its updated receiver",
        )
    })
}

pub async fn invoke_resolved_object_method(
    method: &ResolvedObjectMethod,
    arguments: Vec<Value>,
    requested_outputs: usize,
) -> Result<ValueSequence, RuntimeError> {
    invoke_resolved_object_method_with_context(None, method, arguments, requested_outputs).await
}

pub async fn invoke_resolved_object_method_in_context(
    runtime: &RuntimeContext,
    method: &ResolvedObjectMethod,
    arguments: Vec<Value>,
    requested_outputs: usize,
) -> Result<ValueSequence, RuntimeError> {
    invoke_resolved_object_method_with_context(Some(runtime), method, arguments, requested_outputs)
        .await
}

async fn invoke_resolved_object_method_with_context(
    runtime: Option<&RuntimeContext>,
    method: &ResolvedObjectMethod,
    arguments: Vec<Value>,
    requested_outputs: usize,
) -> Result<ValueSequence, RuntimeError> {
    if !super::resolved_method_is_current(method) {
        return Err(semantic_error(
            "StaleObjectProtocolResolution",
            "object protocol binding changed after the operation was prepared",
        ));
    }
    let descriptor = CallableDescriptor::resolved(
        method.callable.clone(),
        arguments,
        requested_outputs,
        method.fallback,
        CallableCallKind::Direct,
    );
    match runtime {
        Some(runtime) => execute_callable_descriptor_in_context(runtime, descriptor).await,
        None => execute_callable_descriptor(descriptor).await,
    }
}

/// Invoke an object-defined `end(obj, k, n)` selector exactly once. `k` and
/// `n` use MATLAB's one-based component convention.
pub async fn invoke_object_end(
    receiver: Value,
    component_index: usize,
    component_count: usize,
    access: &super::ObjectAccessContext,
) -> Result<Option<Value>, RuntimeError> {
    invoke_object_end_with_context(None, receiver, component_index, component_count, access).await
}

pub async fn invoke_object_end_in_context(
    runtime: &RuntimeContext,
    receiver: Value,
    component_index: usize,
    component_count: usize,
    access: &super::ObjectAccessContext,
) -> Result<Option<Value>, RuntimeError> {
    invoke_object_end_with_context(
        Some(runtime),
        receiver,
        component_index,
        component_count,
        access,
    )
    .await
}

async fn invoke_object_end_with_context(
    runtime: Option<&RuntimeContext>,
    receiver: Value,
    component_index: usize,
    component_count: usize,
    access: &super::ObjectAccessContext,
) -> Result<Option<Value>, RuntimeError> {
    let resolution = super::resolve_object_protocol(&receiver, super::ObjectProtocol::End, access)?;
    let ProtocolResolution::Method(method) = resolution else {
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

pub async fn invoke_prepared_object_end(
    method: &ResolvedObjectMethod,
    receiver: Value,
    component_index: usize,
    component_count: usize,
) -> Result<Value, RuntimeError> {
    invoke_prepared_object_end_with_context(
        None,
        method,
        receiver,
        component_index,
        component_count,
    )
    .await
}

pub(super) async fn invoke_prepared_object_end_with_context(
    runtime: Option<&RuntimeContext>,
    method: &ResolvedObjectMethod,
    receiver: Value,
    component_index: usize,
    component_count: usize,
) -> Result<Value, RuntimeError> {
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
    let mut values = invoke_resolved_object_method_with_context(
        runtime,
        method,
        vec![
            receiver,
            Value::Num(k as f64),
            Value::Num(component_count as f64),
        ],
        1,
    )
    .await?
    .resolve(
        runmat_types::SequenceUse::RequireSingle,
        SequenceResolutionContext::default(),
    )?;
    values.pop().ok_or_else(|| {
        semantic_error(
            "ObjectEndOutput",
            "object end protocol did not return one scalar value",
        )
    })
}

fn invocation_arguments(
    method: &ResolvedObjectMethod,
    base: Value,
    path: ObjectSubscriptPath,
) -> Result<Vec<Value>, RuntimeError> {
    match method.convention {
        ObjectProtocolCallingConvention::StandardSubstruct => {
            let arguments = vec![base, path.to_standard_substruct_value()?];
            Ok(arguments)
        }
        ObjectProtocolCallingConvention::DirectArguments => Err(semantic_error(
            "InvalidObjectProtocolInvocation",
            "this object protocol uses direct typed arguments, not a substruct descriptor",
        )),
    }
}

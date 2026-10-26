//! Functional field access to canonical native protocol owners. Ordinary stored,
//! dependent and selected-class properties retain their existing routes.
use crate::object::protocol::{
    self, ObjectProtocol, ObjectProtocolCallingConvention, ResolvedObjectMethod,
    ResolvedObjectMethodTarget,
};
use crate::{
    context::RuntimeContext, execution::CpuCooperationBudget, BuiltinResult, RuntimeError,
};
use runmat_types::{CallableFallbackPolicy, CallableIdentity};
use runmat_value::{ObjectInstance, Value};
mod invocation;
mod selectors;

pub(in crate::builtins::structs::core) struct Prepared {
    runtime: Option<RuntimeContext>,
    budget: Option<CpuCooperationBudget>,
    issuance: crate::class_registry::BuiltinClassIssuance,
    method: ResolvedObjectMethod,
    deferred: Option<crate::builtins::math::optim::optimmodel::native_fields::Target>,
}
pub(in crate::builtins::structs::core) fn invalid(message: &'static str) -> RuntimeError {
    crate::build_runtime_error(message)
        .with_identifier("RunMat:Object:NativeFieldProtocol")
        .with_gpu_gather_retry(crate::GpuGatherRetry::Never)
        .build()
}
pub(in crate::builtins::structs::core) fn prepare(
    runtime: Option<&RuntimeContext>,
    object: &ObjectInstance,
    protocol: ObjectProtocol,
    property: &str,
) -> BuiltinResult<Option<Prepared>> {
    if !object.has_native_payload() || object.selected_class().is_some() {
        return Ok(None);
    }
    // The current canonical declaration factory allocates metadata. Do not
    // introduce an unclaimed factory call on existing managed field routes.
    // Native model production itself still refuses finite workspace upstream.
    let policy = crate::context::host_allocation::HostAllocationReceipt::policy(runtime).map_err(
        |source| {
            crate::context::host_allocation::host_allocation_runtime_error(
                runtime,
                "RunMat:Object:FieldPathAllocation",
                "native field metadata selection",
                source,
            )
        },
    )?;
    if policy != crate::context::host_allocation::HostAllocationPolicy::Unbounded {
        return Ok(None);
    }
    let Some(declaration) = runmat_builtins::standard_class_declaration(object.class_name()) else {
        return Ok(None);
    };
    // Stored-property classes keep their existing functional-field behavior.
    // Only an actual declared dependent property delegates to its native protocol.
    let Some(property) = declaration
        .properties
        .iter()
        .find(|known| known.name.0 == property)
    else {
        return Ok(None);
    };
    let access = match protocol {
        ObjectProtocol::Subsref => property.attributes.get_access,
        ObjectProtocol::Subsasgn => property.attributes.set_access,
        _ => return Ok(None),
    };
    if !property.attributes.is_dependent
        || property.attributes.is_static
        || access != runmat_types::MemberAccess::Public
        || (protocol == ObjectProtocol::Subsasgn && property.attributes.is_constant)
    {
        // Contextual protected/private/static access remains with the original
        // field owner; a public native protocol is not an access bypass.
        return Ok(None);
    }
    let Some(canonical) = declaration
        .methods
        .iter()
        .find(|method| method.name == protocol.method())
    else {
        return Ok(None);
    };
    if !matches!(canonical.callable, CallableIdentity::Builtin(_)) {
        return Ok(None);
    }
    let select = || {
        let issuance = match runtime {
            Some(runtime) => {
                crate::class_registry::issue_builtin_class_in_context(runtime, object.class_name())
            }
            None => crate::class_registry::issue_builtin_class_ordinary(object.class_name()),
        }
        .ok_or_else(|| {
            invalid("native field protocol requires its current builtin class issuance")
        })?;
        let method = protocol::resolve_object_protocol(
            &Value::Object(object.clone()),
            protocol,
            &protocol::ObjectAccessContext::default(),
        )?
        .ok_or_else(|| invalid("native class protocol is unavailable"))?;
        if method.class != *object.class_name()
            || method.declaring_class != *object.class_name()
            || method.convention != ObjectProtocolCallingConvention::StandardSubstruct
            || method.fallback != CallableFallbackPolicy::None
            || !matches!(&method.target,ResolvedObjectMethodTarget::Named(callable) if callable==&canonical.callable)
        {
            return Err(invalid(
                "native field protocol differs from its canonical declaration",
            ));
        }
        Ok((issuance, method))
    };
    let (issuance, method) = match runtime {
        Some(runtime) => {
            let _scope = runtime.enter();
            select()?
        }
        None => {
            let scope = crate::context::PreparedUnselectedScope::prepare(None)?;
            let _scope = scope.enter();
            select()?
        }
    };
    let (budget, runtime) = runtime
        .map(|runtime| {
            let budget = runtime.cpu_cooperation();
            let selected = runtime
                .clone()
                .with_cancellation_token(budget.cancellation_token());
            (budget, selected)
        })
        .unzip();
    let deferred =
        crate::builtins::math::optim::optimmodel::native_fields::Target::select(object, &method);
    Ok(Some(Prepared {
        runtime,
        budget,
        issuance,
        method,
        deferred,
    }))
}

//! Effect-free selection of one user-call route before the recursive await.
//!
//! Method lookup and access checks run on the caller's native frame. The
//! interpreter then awaits only the selected callable descriptor, so unrelated
//! static-method and fallback futures do not accumulate in each Process poll.

use super::{imported_static_method_owner, CallableCallKind, CallableDescriptor, RuntimeError};
use runmat_hir::{CallableFallbackPolicy, CallableIdentity};
use runmat_types::{ClassIdentity, MemberAccess, MethodName};
use runmat_value::{BoundFunctionBinding, Value};

pub(super) struct PreparedUserCall {
    pub(super) descriptor: CallableDescriptor,
    /// Existing static-method dispatch bypasses the caller's local catch route
    /// and does not install a class access guard around the selected call.
    pub(super) route: UserCallRoute,
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub(super) enum UserCallRoute {
    StaticMethod,
    CallerScope,
}

#[inline(never)]
pub(super) fn prepare(
    identity: CallableIdentity,
    bound: Option<BoundFunctionBinding>,
    fallback_policy: CallableFallbackPolicy,
    out_count: usize,
    args: Vec<Value>,
    caller_class: Option<&ClassIdentity>,
    imports: &[(Vec<String>, bool)],
) -> Result<PreparedUserCall, RuntimeError> {
    if let Some(method) = selected_static_method(&identity, caller_class, imports)? {
        let method_identity = if method.function_name.contains('.') {
            CallableIdentity::ExternalName(runmat_hir::QualifiedName(
                method
                    .function_name
                    .split('.')
                    .map(|segment| runmat_hir::SymbolName(segment.trim().to_string()))
                    .collect(),
            ))
        } else {
            CallableIdentity::DynamicName(runmat_hir::SymbolName(method.function_name.clone()))
        };
        return Ok(PreparedUserCall {
            descriptor: CallableDescriptor::resolved(
                method_identity,
                args,
                out_count,
                CallableFallbackPolicy::ExternalBoundary,
                CallableCallKind::Direct,
            ),
            route: UserCallRoute::StaticMethod,
        });
    }

    let descriptor = if let Some(binding) = bound {
        CallableDescriptor::bound(binding, args, out_count, CallableCallKind::Direct)
    } else {
        CallableDescriptor::resolved(
            identity,
            args,
            out_count,
            fallback_policy,
            CallableCallKind::Direct,
        )
    };
    Ok(PreparedUserCall {
        descriptor,
        route: UserCallRoute::CallerScope,
    })
}

fn selected_static_method(
    identity: &CallableIdentity,
    caller_class: Option<&ClassIdentity>,
    imports: &[(Vec<String>, bool)],
) -> Result<Option<runmat_runtime::class_registry::RuntimeMethod>, RuntimeError> {
    let explicit = match identity {
        CallableIdentity::ExternalName(runmat_hir::QualifiedName(segments))
            if segments.len() >= 2 =>
        {
            let class_name = segments[..segments.len() - 1]
                .iter()
                .map(|segment| segment.0.as_str())
                .collect::<Vec<_>>()
                .join(".");
            Some((
                ClassIdentity::new(class_name).map_err(|error| {
                    crate::interpreter::errors::mex("InvalidClassIdentity", &error.to_string())
                })?,
                MethodName::from(segments[segments.len() - 1].0.as_str()),
            ))
        }
        CallableIdentity::DynamicName(runmat_hir::SymbolName(name)) if name.contains('.') => name
            .rsplit_once('.')
            .map(|(class_name, method_name)| {
                ClassIdentity::new(class_name)
                    .map(|class_name| (class_name, MethodName::from(method_name)))
            })
            .transpose()
            .map_err(|error| {
                crate::interpreter::errors::mex("InvalidClassIdentity", &error.to_string())
            })?,
        _ => None,
    };
    if caller_class.is_some() {
        if let Some((class_name, method_name)) = explicit {
            if runmat_runtime::class_registry::get_class(&class_name).is_some() {
                if let Some(method) = checked_method(&class_name, &method_name, caller_class)? {
                    return Ok(Some(method));
                }
            }
        }
    }

    let local_method = match identity {
        CallableIdentity::DynamicName(runmat_hir::SymbolName(name))
            if !name.contains('.') && !name.trim().is_empty() =>
        {
            Some(MethodName::from(name.trim()))
        }
        CallableIdentity::ExternalName(runmat_hir::QualifiedName(segments))
            if segments.len() == 1 && !segments[0].0.trim().is_empty() =>
        {
            Some(MethodName::from(segments[0].0.trim()))
        }
        _ => None,
    };
    if let (Some(class_name), Some(method_name)) = (caller_class, local_method.as_ref()) {
        if let Some(method) = checked_method(class_name, method_name, caller_class)? {
            return Ok(Some(method));
        }
    }
    if let Some(method_name) = local_method.as_ref() {
        if let Some(owner_class) = imported_static_method_owner(imports, method_name)? {
            if let Some(method) = checked_method(&owner_class, method_name, caller_class)? {
                return Ok(Some(method));
            }
        }
    }
    Ok(None)
}

fn checked_method(
    class_name: &ClassIdentity,
    method_name: &MethodName,
    caller_class: Option<&ClassIdentity>,
) -> Result<Option<runmat_runtime::class_registry::RuntimeMethod>, RuntimeError> {
    let Some((method, owner)) =
        runmat_runtime::class_registry::lookup_method(class_name, method_name)
    else {
        return Ok(None);
    };
    if !method.is_static {
        return Ok(None);
    }
    let allowed = match method.access {
        MemberAccess::Public => true,
        MemberAccess::Private => caller_class == Some(&owner),
        MemberAccess::Protected => caller_class.is_some_and(|caller_class| {
            runmat_runtime::class_registry::is_class_or_subclass(caller_class, &owner)
        }),
    };
    if !allowed {
        return Err(crate::interpreter::errors::mex(
            "MethodPrivate",
            &format!("Method '{}' is private", method_name),
        ));
    }
    Ok(Some(method))
}

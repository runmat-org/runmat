//! Read-only evidence for the dispatcher's executable name authority.
//! Import resolution observes candidates here before invoking any body.

use super::*;
use std::rc::Rc;

#[cfg(test)]
mod validity_tests;

pub(super) struct SelectedRegisteredBuiltin {
    pub(super) origin: SelectedBindingOrigin,
    pub(super) implementation: SelectedImplementation,
    pub(super) authority: Option<direct_call_observation::RegistrationAuthority>,
}

/// The one registration selected by the dispatcher's scoped/global authority.
/// This is shared by execution and import observation, including duplicate
/// authority rejection and the legacy no-category priority.
pub(super) fn select_registered_builtin(
    name: &str,
    context: Option<&crate::context::RuntimeContext>,
) -> Result<Option<SelectedRegisteredBuiltin>, RuntimeError> {
    let scoped: Option<Rc<dyn crate::context::RuntimeBuiltinService>> =
        context.and_then(|context| context.service_ports().builtin().cloned());
    let global = scoped
        .is_none()
        .then(|| crate::builtin::registration::registered_runtime_builtin_bindings_by_name(name));
    let selected_binding = scoped.as_ref().map_or_else(
        || {
            global
                .as_ref()
                .and_then(|rows| rows.last())
                .map(|row| row.binding())
        },
        |service| service.bindings_by_name(name).last().copied(),
    );
    let selected_legacy = if scoped.is_some() {
        None
    } else {
        runmat_builtins::with_builtin_functions_by_exact_name(name, |entries| {
            entries
                .iter()
                .rev()
                .find(|entry| entry.category.is_empty())
                .or_else(|| entries.last())
                .copied()
        })
    };
    if selected_binding.is_some() && selected_legacy.is_some() {
        return Err(build_runtime_error(format!(
            "builtin `{name}` has both canonical and legacy runtime bindings"
        ))
        .with_identifier("RunMat:Catalog:DuplicateBindingAuthority")
        .build());
    }
    if let Some(binding) = selected_binding {
        let authority = if let Some(service) = scoped {
            service
                .binding_registration_revision(binding)
                .map(
                    |generation| direct_call_observation::RegistrationAuthority::Scoped {
                        service,
                        generation,
                    },
                )
        } else {
            global.as_ref().and_then(|rows| rows.last()).map(|row| {
                direct_call_observation::RegistrationAuthority::Global(row.registration())
            })
        };
        return Ok(Some(SelectedRegisteredBuiltin {
            origin: SelectedBindingOrigin::CanonicalOrScoped,
            implementation: SelectedImplementation::Canonical(binding),
            authority,
        }));
    }
    Ok(selected_legacy.map(|builtin| SelectedRegisteredBuiltin {
        origin: SelectedBindingOrigin::LegacyInventory,
        implementation: SelectedImplementation::Legacy(builtin.implementation),
        authority: None,
    }))
}

#[derive(Debug, Clone)]
pub enum ObservedBuiltinDispatchTarget {
    Builtin(runmat_types::BuiltinId),
    InstanceMethod {
        owner: ClassIdentity,
        method: runmat_types::MethodName,
    },
    StaticMethod {
        owner: ClassIdentity,
        method: runmat_types::MethodName,
    },
    Constructor(ClassIdentity),
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum ObservedBuiltinDispatchIdentity {
    Builtin(runmat_types::BuiltinId),
    InstanceMethod(ClassIdentity, runmat_types::MethodName),
    StaticMethod(ClassIdentity, runmat_types::MethodName),
    Constructor(ClassIdentity),
}

impl ObservedBuiltinDispatchTarget {
    pub fn identity(&self) -> ObservedBuiltinDispatchIdentity {
        match self {
            Self::Builtin(identity) => ObservedBuiltinDispatchIdentity::Builtin(identity.clone()),
            Self::InstanceMethod { owner, method } => {
                ObservedBuiltinDispatchIdentity::InstanceMethod(owner.clone(), method.clone())
            }
            Self::StaticMethod { owner, method } => {
                ObservedBuiltinDispatchIdentity::StaticMethod(owner.clone(), method.clone())
            }
            Self::Constructor(class) => ObservedBuiltinDispatchIdentity::Constructor(class.clone()),
        }
    }
}

/// Observe whether this exact spelling has an executable builtin-dispatch
/// target under the invocation's binding and class authority. No implementation
/// is called and no source/native path is loaded.
pub fn observe_builtin_dispatch_target(
    context: &crate::context::RuntimeContext,
    requested: &str,
    args: &[Value],
) -> Result<Option<ObservedBuiltinDispatchTarget>, RuntimeError> {
    let resolved = ResolvedBuiltinName::resolve(requested);
    let name = resolved.canonical;
    let registered = select_registered_builtin(name, Some(context))?;
    if !registered.as_ref().is_some_and(|selected| {
        selected
            .implementation
            .is_handle_validity_query(args.first())
    }) {
        if let Some(selected) =
            select_instance_method(name, crate::call::FunctionArgumentView::from_slice(args))?
                .filter(|selected| selected.has_executable_route())
        {
            return Ok(Some(ObservedBuiltinDispatchTarget::InstanceMethod {
                owner: selected.owner,
                method: selected.method,
            }));
        }
    }
    if let Some(registered) = registered {
        let identity = match registered.implementation {
            SelectedImplementation::Canonical(binding) => binding.identity.builtin.name,
            SelectedImplementation::Legacy(_) => name,
        };
        return Ok(Some(ObservedBuiltinDispatchTarget::Builtin(
            runmat_types::BuiltinId(identity.into()),
        )));
    }
    if let Some(method) = observe_registered_static_method(name) {
        return Ok(Some(ObservedBuiltinDispatchTarget::StaticMethod {
            owner: method.registration.class,
            method: method.registration.method_name,
        }));
    }
    Ok(crate::class_registry::registered_constructor_identity(name)
        .map(ObservedBuiltinDispatchTarget::Constructor))
}

pub(super) enum InstanceMethodRoute {
    Selected(crate::object::protocol::ResolvedObjectMethod),
    Bound(runmat_value::BoundFunctionBinding),
    SemanticName(String),
    BuiltinName(String),
}

pub(super) struct SelectedInstanceMethod {
    pub(super) owner: ClassIdentity,
    pub(super) method: runmat_types::MethodName,
    pub(super) routes: Vec<InstanceMethodRoute>,
}

impl SelectedInstanceMethod {
    fn has_executable_route(&self) -> bool {
        self.routes.iter().any(|route| match route {
            InstanceMethodRoute::Selected(_) => true,
            InstanceMethodRoute::Bound(_) => true,
            InstanceMethodRoute::SemanticName(name) => semantic_name_is_executable(name),
            InstanceMethodRoute::BuiltinName(_) => true,
        })
    }
}

/// Resolve one instance-method declaration, access decision, and the ordered
/// semantic/builtin routes consumed by both execution and import observation.
pub(super) fn select_instance_method(
    method_name: &str,
    args: crate::call::FunctionArgumentView<'_>,
) -> Result<Option<SelectedInstanceMethod>, RuntimeError> {
    let Some(receiver) = args.first() else {
        return Ok(None);
    };
    let class_name = match receiver {
        Value::Object(object) => object.class_name(),
        Value::HandleObject(handle) => &*handle.class_name,
        _ => return Ok(None),
    };
    let method_identity = runmat_types::MethodName::from(method_name);
    if crate::object::protocol::selected_receiver_has_authority(receiver)? {
        let access = crate::object::protocol::ObjectAccessContext {
            caller_class: current_class_access_context().as_deref().cloned(),
            ..Default::default()
        };
        return crate::object::protocol::resolve_declared_object_method(
            receiver,
            &method_identity,
            &access,
        )
        .map(|method| {
            method.map(|method| SelectedInstanceMethod {
                owner: method.declaring_class.clone(),
                method: method_identity,
                routes: vec![InstanceMethodRoute::Selected(method)],
            })
        });
    }
    let Some((method, owner)) = crate::class_registry::lookup_method(class_name, &method_identity)
    else {
        return Ok(None);
    };
    if method.is_static {
        return Ok(None);
    }
    let caller_class = current_class_access_context();
    let access_allowed = match method.access {
        runmat_types::MemberAccess::Public => true,
        runmat_types::MemberAccess::Private => caller_class.as_deref() == Some(&owner),
        runmat_types::MemberAccess::Protected => caller_class
            .as_deref()
            .is_some_and(|caller| crate::class_registry::is_class_or_subclass(caller, &owner)),
    };
    if !access_allowed {
        return Err(build_runtime_error(format!(
            "Method '{}' is not accessible from current context.",
            method_name
        ))
        .with_identifier("RunMat:MethodPrivate")
        .build());
    }
    if let Some(binding) = crate::class_registry::lookup_bound_method(class_name, &method_identity)
        .and_then(|method| method.bound_function)
    {
        return Ok(Some(SelectedInstanceMethod {
            owner,
            method: method_identity,
            routes: vec![InstanceMethodRoute::Bound(binding)],
        }));
    }
    let mut routes = Vec::with_capacity(4);
    let mut append_routes = |name: &str| {
        routes.push(InstanceMethodRoute::SemanticName(name.to_owned()));
        if name != method_name && runmat_builtins::builtin_name_is_known(name) {
            routes.push(InstanceMethodRoute::BuiltinName(name.to_owned()));
        }
    };
    append_routes(&method.function_name);
    let owner_qualified = format!("{owner}.{method_name}");
    if owner_qualified != method.function_name {
        append_routes(&owner_qualified);
    }
    Ok(Some(SelectedInstanceMethod {
        owner,
        method: method_identity,
        routes,
    }))
}

fn semantic_name_is_executable(name: &str) -> bool {
    let Some(function) = crate::user_functions::resolve_semantic_function_by_name(name) else {
        return false;
    };
    let selected_scoped = crate::user_functions::current_scoped_name_target_resolver()
        .and_then(|resolver| resolver(name))
        .filter(|target| target.function().0 == function);
    match selected_scoped {
        Some(crate::user_functions::ScopedSemanticNameTarget::Retained(_)) => {
            crate::context::legacy::active().is_some()
                && crate::user_functions::current_bound_function_invoker().is_some()
        }
        Some(crate::user_functions::ScopedSemanticNameTarget::Published(_)) | None => {
            crate::user_functions::current_semantic_function_invoker().is_some()
        }
    }
}

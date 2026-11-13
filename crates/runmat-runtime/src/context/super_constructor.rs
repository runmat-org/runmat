//! Invocation-local, exact retained superclass constructor selections.

use runmat_execution::FunctionBindingOrigin;
use runmat_types::{ClassIdentity, ClassMethodOwner, ClassMethodRole, MemberAccess, MethodName};
use runmat_value::BoundFunctionBinding;

/// Invocation-only projection of the selected product's immutable class
/// declaration. VM RegisterClass and Native IR are the two admitted issuers;
/// this is not a mutable registry entry or another source of class identity.
#[derive(Clone, Debug)]
pub struct RetainedClassDeclaration {
    pub class: ClassIdentity,
    pub parent: Option<ClassIdentity>,
    pub constructor: Option<RetainedConstructorDeclaration>,
}

#[derive(Clone, Debug)]
pub struct RetainedConstructorDeclaration {
    pub name: MethodName,
    pub is_static: bool,
    pub is_abstract: bool,
    pub access: MemberAccess,
}

#[derive(Clone, Debug)]
pub struct RetainedSuperConstructorBinding {
    pub(crate) requesting_class: ClassIdentity,
    pub(crate) caller_origin: FunctionBindingOrigin,
    pub(crate) target: RetainedSuperConstructorTarget,
}

#[derive(Clone, Debug)]
pub(crate) enum RetainedSuperConstructorTarget {
    Bound {
        origin: FunctionBindingOrigin,
        owner: ClassMethodOwner,
        binding: BoundFunctionBinding,
        access: MemberAccess,
    },
    Implicit {
        parent: ClassIdentity,
    },
}

impl RetainedSuperConstructorTarget {
    pub(super) fn is_admitted_in(&self, runtime: &super::RuntimeContext) -> bool {
        let (origin, owner, binding) = match self {
            Self::Bound {
                origin,
                owner,
                binding,
                ..
            } => (origin, owner, binding),
            Self::Implicit { .. } => return true,
        };
        if owner.role != ClassMethodRole::Constructor || owner.is_static {
            return false;
        }
        match &binding.authority {
            runmat_value::BoundFunctionAuthority::Local { .. } => {
                binding.authority.admits_local(&runtime.session_lease())
                    && binding.authority.local_lease(binding.function).is_some()
                    && binding.authority.export_origin(&runtime.session_lease()) == Some(*origin)
            }
            runmat_value::BoundFunctionAuthority::Portable { origin: issued } => {
                issued == origin
                    && crate::user_functions::resolve_binding_origin_binding_from_runtime(
                        runtime, *origin,
                    )
                    .is_some_and(|issued| issued == *binding)
            }
            runmat_value::BoundFunctionAuthority::SelectedNative { .. } => {
                crate::user_functions::selected_native_binding_admitted(runtime, binding, *origin)
                    && crate::user_functions::resolve_binding_origin_binding_from_runtime(
                        runtime, *origin,
                    )
                    .is_some_and(|issued| issued == *binding)
            }
        }
    }

    pub(crate) fn parent(&self) -> &ClassIdentity {
        match self {
            Self::Bound { owner, .. } => &owner.declaring_class,
            Self::Implicit { parent } => parent,
        }
    }
}

impl RetainedSuperConstructorBinding {
    /// Check the immutable parent/method facts against the exact owner and
    /// leased target. Runtime rechecks session lease and origin at install.
    pub fn from_admitted_definition(
        caller: &RetainedClassDeclaration,
        target: &RetainedClassDeclaration,
        caller_origin: FunctionBindingOrigin,
        target_origin: FunctionBindingOrigin,
        owner: ClassMethodOwner,
        binding: BoundFunctionBinding,
    ) -> Result<Self, crate::RuntimeError> {
        let Some(method) = &target.constructor else {
            return Err(invalid_declaration());
        };
        if caller.parent.as_ref() != Some(&target.class)
            || owner.declaring_class != target.class
            || owner.method != method.name
            || owner.role != ClassMethodRole::Constructor
            || owner.is_static
            || method.is_static
            || method.is_abstract
        {
            return Err(invalid_declaration());
        }
        Ok(Self {
            requesting_class: caller.class.clone(),
            caller_origin,
            target: RetainedSuperConstructorTarget::Bound {
                origin: target_origin,
                owner,
                binding,
                access: method.access,
            },
        })
    }

    /// A linked class declaration with no constructor body, or an intrinsic
    /// standard parent, has an exact no-op superclass target. It must not
    /// consult a mutable class registry or later same-named source body.
    pub fn from_admitted_linked_implicit(
        caller: &RetainedClassDeclaration,
        parent: &RetainedClassDeclaration,
        caller_origin: FunctionBindingOrigin,
    ) -> Result<Self, crate::RuntimeError> {
        if caller.parent.as_ref() != Some(&parent.class) || parent.constructor.is_some() {
            return Err(invalid_declaration());
        }
        Ok(Self {
            requesting_class: caller.class.clone(),
            caller_origin,
            target: RetainedSuperConstructorTarget::Implicit {
                parent: parent.class.clone(),
            },
        })
    }

    pub fn from_admitted_intrinsic(
        caller: &RetainedClassDeclaration,
        caller_origin: FunctionBindingOrigin,
    ) -> Result<Self, crate::RuntimeError> {
        let Some(parent) = caller.parent.as_ref() else {
            return Err(invalid_declaration());
        };
        if !crate::class_registry::has_intrinsic_implicit_constructor(parent) {
            return Err(invalid_declaration());
        }
        Ok(Self {
            requesting_class: caller.class.clone(),
            caller_origin,
            target: RetainedSuperConstructorTarget::Implicit {
                parent: parent.clone(),
            },
        })
    }

    pub(super) fn matches(
        &self,
        caller_origin: FunctionBindingOrigin,
        requesting_class: &ClassIdentity,
        super_class: &ClassIdentity,
    ) -> bool {
        self.caller_origin == caller_origin
            && &self.requesting_class == requesting_class
            && self.target.parent() == super_class
    }
}

fn invalid_declaration() -> crate::RuntimeError {
    crate::build_runtime_error("retained superclass declaration disagrees with its owner")
        .with_identifier("RunMat:ConstructorOwnerInvalid")
        .build()
}

#[cfg(test)]
mod tests {
    use std::{rc::Rc, sync::Arc};

    use runmat_execution::Digest;
    use runmat_types::{MethodName, ProgramFunctionId};
    use runmat_value::{FunctionDefinitionOwner, LocalExport};

    use super::*;
    use crate::execution::RuntimeExecutionService;

    fn origin(function: u32) -> FunctionBindingOrigin {
        FunctionBindingOrigin {
            program: Digest::sha256("retained-super-product"),
            function: ProgramFunctionId(function),
        }
    }

    fn binding(
        runtime: &super::super::RuntimeContext,
        target_origin: FunctionBindingOrigin,
    ) -> RetainedSuperConstructorBinding {
        let definition = Arc::new(FunctionDefinitionOwner::default());
        let base = ClassIdentity::new("Base").expect("base");
        RetainedSuperConstructorBinding::from_admitted_definition(
            &RetainedClassDeclaration {
                class: ClassIdentity::new("Derived").expect("derived"),
                parent: Some(base.clone()),
                constructor: None,
            },
            &RetainedClassDeclaration {
                class: base.clone(),
                parent: None,
                constructor: Some(RetainedConstructorDeclaration {
                    name: MethodName::from("Base"),
                    is_static: false,
                    is_abstract: false,
                    access: MemberAccess::Protected,
                }),
            },
            origin(1),
            target_origin,
            ClassMethodOwner {
                declaring_class: base,
                method: MethodName::from("Base"),
                is_static: false,
                role: ClassMethodRole::Constructor,
            },
            BoundFunctionBinding::local(
                2,
                runtime.session_lease(),
                FunctionDefinitionOwner::issue(&definition, 2),
                Vec::new(),
                LocalExport::Exact(origin(2)),
            ),
        )
        .expect("paired immutable declaration")
    }

    #[test]
    fn exact_local_origin_and_caller_are_required() {
        let runtime = super::super::RuntimeContext::new(Rc::new(RuntimeExecutionService::new()));
        assert!(runtime
            .clone()
            .with_retained_super_constructors(vec![binding(&runtime, origin(3))], origin(1))
            .is_err());
        let target = binding(&runtime, origin(2));
        let selected = runtime
            .with_retained_super_constructors(vec![target], origin(1))
            .expect("exact local definition");
        let base = ClassIdentity::new("Base").expect("base");
        let derived = ClassIdentity::new("Derived").expect("derived");
        assert!(matches!(
            selected
                .retained_super_constructor(&derived, &base)
                .expect("selected caller")
                .expect("retained target")
                .target,
            RetainedSuperConstructorTarget::Bound {
                access: MemberAccess::Protected,
                ..
            }
        ));
        assert!(selected
            .with_retained_super_caller_origin(origin(9))
            .retained_super_constructor(&derived, &base)
            .is_err());
    }

    #[test]
    fn separately_admitted_parent_uses_its_exact_local_definition() {
        let runtime = super::super::RuntimeContext::new(Rc::new(RuntimeExecutionService::new()));
        let base = ClassIdentity::new("Base").expect("base");
        let derived = ClassIdentity::new("Derived").expect("derived");
        let target_origin = FunctionBindingOrigin {
            program: Digest::sha256("separately-admitted-base"),
            function: ProgramFunctionId(2),
        };
        let definition = Arc::new(FunctionDefinitionOwner::default());
        let binding = RetainedSuperConstructorBinding::from_admitted_definition(
            &RetainedClassDeclaration {
                class: derived.clone(),
                parent: Some(base.clone()),
                constructor: None,
            },
            &RetainedClassDeclaration {
                class: base.clone(),
                parent: None,
                constructor: Some(RetainedConstructorDeclaration {
                    name: MethodName::from("Base"),
                    is_static: false,
                    is_abstract: false,
                    access: MemberAccess::Protected,
                }),
            },
            origin(1),
            target_origin,
            ClassMethodOwner {
                declaring_class: base.clone(),
                method: MethodName::from("Base"),
                is_static: false,
                role: ClassMethodRole::Constructor,
            },
            BoundFunctionBinding::local(
                2,
                runtime.session_lease(),
                FunctionDefinitionOwner::issue(&definition, 2),
                Vec::new(),
                LocalExport::Exact(target_origin),
            ),
        )
        .expect("paired declarations and exact external origin");
        let selected = runtime
            .with_retained_super_constructors(vec![binding], origin(1))
            .expect("exact selected dependency may cross product origins");
        let target = selected
            .retained_super_constructor(&derived, &base)
            .expect("caller selection")
            .expect("retained parent");
        assert!(matches!(
            &target.target,
            RetainedSuperConstructorTarget::Bound { origin, .. } if *origin == target_origin
        ));
    }

    #[test]
    fn forged_parent_or_method_fails_before_binding_installation() {
        let base = ClassIdentity::new("Base").expect("base");
        let unrelated = ClassIdentity::new("Other").expect("other");
        let caller = RetainedClassDeclaration {
            class: ClassIdentity::new("Derived").expect("derived"),
            parent: Some(unrelated),
            constructor: None,
        };
        let target = RetainedClassDeclaration {
            class: base.clone(),
            parent: None,
            constructor: Some(RetainedConstructorDeclaration {
                name: MethodName::from("Base"),
                is_static: false,
                is_abstract: false,
                access: MemberAccess::Protected,
            }),
        };
        let owner = ClassMethodOwner {
            declaring_class: base.clone(),
            method: MethodName::from("Base"),
            is_static: false,
            role: ClassMethodRole::Constructor,
        };
        let binding = BoundFunctionBinding::portable(2, origin(2));
        assert!(RetainedSuperConstructorBinding::from_admitted_definition(
            &caller,
            &target,
            origin(1),
            origin(2),
            owner.clone(),
            binding.clone(),
        )
        .is_err());
        let caller = RetainedClassDeclaration {
            parent: Some(base),
            ..caller
        };
        let wrong_owner = ClassMethodOwner {
            method: MethodName::from("other"),
            ..owner
        };
        assert!(RetainedSuperConstructorBinding::from_admitted_definition(
            &caller,
            &target,
            origin(1),
            origin(2),
            wrong_owner,
            binding,
        )
        .is_err());
    }

    #[test]
    fn linked_implicit_parent_keeps_original_receiver_without_name_lookup() {
        let base = ClassIdentity::new("ImplicitBase").expect("base");
        let derived = ClassIdentity::new("ImplicitDerived").expect("derived");
        let caller = RetainedClassDeclaration {
            class: derived.clone(),
            parent: Some(base.clone()),
            constructor: None,
        };
        let parent = RetainedClassDeclaration {
            class: base.clone(),
            parent: None,
            constructor: None,
        };
        let runtime = super::super::RuntimeContext::new(Rc::new(RuntimeExecutionService::new()))
            .with_retained_super_constructors(
                vec![
                    RetainedSuperConstructorBinding::from_admitted_linked_implicit(
                        &caller,
                        &parent,
                        origin(1),
                    )
                    .expect("implicit declaration"),
                ],
                origin(1),
            )
            .expect("selected context");
        let original =
            runmat_value::Value::Object(runmat_value::ObjectInstance::new(derived.clone()));
        let _scope = runtime.enter();
        let result = futures::executor::block_on(crate::with_constructor_receiver(
            original.clone(),
            crate::call_super_constructor(&runtime, derived, base, Vec::new()),
        ))
        .expect("implicit super call");
        assert_eq!(result.value(), &original);
    }

    #[test]
    fn superclass_body_intent_is_exact_and_consumed_before_child_dispatch() {
        let runtime = super::super::RuntimeContext::new(Rc::new(RuntimeExecutionService::new()));
        let owner = ClassMethodOwner {
            declaring_class: ClassIdentity::new("Base").expect("class"),
            method: MethodName::from("Base"),
            is_static: false,
            role: ClassMethodRole::Constructor,
        };
        let binding = crate::user_functions::test_support::exact_local_binding(&runtime, 2);
        let selected = runtime.with_retained_super_body(owner.clone(), binding.clone());
        assert!(selected
            .clone()
            .take_retained_super_body_for(&runmat_value::BoundFunctionBinding::unavailable(2))
            .is_err());
        let (child, actual_owner) = selected
            .take_retained_super_body_for(&binding)
            .expect("exact selected definition");
        assert_eq!(actual_owner, Some(owner));
        assert_eq!(
            child
                .take_retained_super_body_for(&binding)
                .expect("intent consumed")
                .1,
            None
        );
    }
}

//! Canonical builtin class issuance in the actual original session. A source
//! replacement is never overwritten merely because a native payload exists.
use super::model_policy::Policy;
use crate::{context::RuntimeContext, BuiltinResult};
use runmat_types::StaticClassIdentity;

pub(in crate::builtins::math::optim) fn ensure(
    class: StaticClassIdentity,
    parent_classes: &[StaticClassIdentity],
    policy: Policy,
    runtime: Option<&RuntimeContext>,
) -> BuiltinResult<()> {
    match runtime {
        Some(runtime) => {
            let _scope = runtime.enter();
            ensure_selected(class, parent_classes, policy)
        }
        None => {
            let scope = crate::context::PreparedUnselectedScope::prepare(None)?;
            let _guard = scope.enter();
            ensure_selected(class, parent_classes, policy)
        }
    }
}
fn ensure_selected(
    class: StaticClassIdentity,
    parent_classes: &[StaticClassIdentity],
    policy: Policy,
) -> BuiltinResult<()> {
    let identity = class.owned();
    if crate::class_registry::issue_builtin_class_ordinary(&identity).is_some() {
        return Ok(());
    }
    if crate::class_registry::get_class(&identity).is_some() {
        return Err(policy.invalid("the canonical builtin class has been replaced"));
    }
    let declaration = runmat_builtins::standard_class_declaration(&identity)
        .ok_or_else(|| policy.invalid("canonical builtin class declaration is unavailable"))?;
    if let Some(parent) = declaration.parent.as_ref() {
        let parent = parent_classes
            .iter()
            .copied()
            .find(|class| *parent == class.owned().qualified_name())
            .ok_or_else(|| policy.invalid("unsupported canonical builtin class ancestry"))?;
        ensure_selected(parent, parent_classes, policy)?;
    }
    let parent = declaration
        .parent
        .as_ref()
        .map(runmat_types::ClassIdentity::from_qualified_name)
        .transpose()
        .map_err(|source| policy.invalid(source))?;
    let properties = declaration
        .properties
        .into_iter()
        .map(|property| {
            let name = property.name;
            (
                name.clone(),
                crate::class_registry::RuntimeProperty {
                    name,
                    is_static: property.attributes.is_static,
                    is_constant: property.attributes.is_constant,
                    is_dependent: property.attributes.is_dependent,
                    get_access: property.attributes.get_access,
                    set_access: property.attributes.set_access,
                    default_value: None,
                },
            )
        })
        .collect();
    crate::class_registry::register_builtin_class(
        class,
        crate::class_registry::RuntimeClass {
            name: identity,
            parent,
            properties,
            methods: crate::class_registry::standard_runtime_methods(class),
            events: Default::default(),
        },
    );
    Ok(())
}

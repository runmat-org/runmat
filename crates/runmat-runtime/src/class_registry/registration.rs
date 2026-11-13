//! Atomic checked class publication shared by VM and standalone native hosts.

use super::{ClassRegistryRevision, RuntimeClass, RuntimeClassState};
use runmat_types::ClassIdentity;
use std::collections::BTreeMap;
use std::sync::atomic::{AtomicU64, Ordering};

#[path = "registration/preflight.rs"]
mod preflight;

pub struct CheckedClassRegistration {
    pub class: RuntimeClass,
    /// Exact executable-method bindings. Empty for ordinary source/builtin
    /// registration, whose existing method resolution remains unchanged.
    pub method_bindings:
        std::collections::HashMap<runmat_types::MethodName, runmat_value::BoundFunctionBinding>,
    pub is_sealed: bool,
    pub is_abstract: bool,
    pub enumerations: Vec<String>,
}

pub fn register_checked_class(
    class: RuntimeClass,
    is_sealed: bool,
    is_abstract: bool,
    enumerations: Vec<String>,
) -> Result<(), crate::RuntimeError> {
    register_checked_classes(vec![CheckedClassRegistration {
        class,
        method_bindings: std::collections::HashMap::new(),
        is_sealed,
        is_abstract,
        enumerations,
    }])
}

/// Stage the entire proposed inheritance graph against one registry snapshot.
/// No declaration becomes observable if any member of the batch is invalid.
pub fn register_checked_classes(
    classes: Vec<CheckedClassRegistration>,
) -> Result<(), crate::RuntimeError> {
    register_classes(classes, preflight::ParentAdmission::Closed)
}

/// Publish source declarations without claiming that every external parent is
/// executable now. Constructor admission uses the closed variant above.
pub fn register_open_class_declarations(
    classes: Vec<CheckedClassRegistration>,
) -> Result<(), crate::RuntimeError> {
    register_classes(classes, preflight::ParentAdmission::Open)
}

/// Validate a complete selected product without publishing it into the
/// session's mutable current-class registry. Older retained constructors can
/// then carry these exact defaults/access facts across sibling suspension.
pub fn prepare_selected_class_projection(
    classes: Vec<CheckedClassRegistration>,
) -> Result<std::rc::Rc<super::SelectedClassProjection>, crate::RuntimeError> {
    static NEXT_SELECTED_GENERATION: AtomicU64 = AtomicU64::new(1);
    let _input_roots =
        crate::execution::RootedValueLease::for_values(classes.iter().flat_map(|registration| {
            registration
                .class
                .properties
                .values()
                .filter_map(|property| property.default_value.as_ref())
        }))
        .map_err(|error| error.into_runtime_error())?;
    let mut proposed = BTreeMap::new();
    for registration in classes {
        validate_method_bindings(&registration)?;
        if proposed
            .insert(registration.class.name.clone(), registration)
            .is_some()
        {
            return Err(error(
                "RunMat:ClassDuplicate",
                "class batch repeats a declaration",
            ));
        }
    }
    // Builtin classes are the only external ancestry admitted here. A
    // selected custom parent must be in the same exact product projection.
    let baseline = RuntimeClassState::default();
    let order = preflight::validate(&proposed, &baseline, preflight::ParentAdmission::Closed)?;
    let generation =
        ClassRegistryRevision::Selected(NEXT_SELECTED_GENERATION.fetch_add(1, Ordering::Relaxed));
    let mut classes = std::collections::HashMap::new();
    let mut method_bindings = std::collections::HashMap::new();
    let mut sealed = std::collections::HashSet::new();
    let mut abstract_classes = std::collections::HashSet::new();
    let mut enumerations = std::collections::HashMap::new();
    for name in order {
        let registration = proposed
            .remove(&name)
            .expect("selected preflight returned only proposed classes");
        let class = registration.class;
        method_bindings.extend(super::bind_methods_for_class_with(
            &class,
            generation,
            Some(&registration.method_bindings),
        ));
        if registration.is_sealed {
            sealed.insert(name.clone());
        }
        if registration.is_abstract {
            abstract_classes.insert(name.clone());
        }
        enumerations.insert(
            name.clone(),
            registration.enumerations.into_iter().collect(),
        );
        classes.insert(name, class);
    }
    // Closed preflight proved each external parent against the immutable
    // standard baseline. Retain those exact builtin declarations in this
    // product too: receiver metadata must not consult a later ambient class
    // registry when walking, for example, Base -> dynamicprops -> handle.
    let mut builtin_ancestors = std::collections::HashMap::new();
    for class in classes.values() {
        let mut parent = class.parent.as_ref();
        while let Some(name) = parent {
            if classes.contains_key(name) {
                break;
            }
            let builtin = baseline.classes.get(name).ok_or_else(|| {
                error(
                    "RunMat:ClassParentMissing",
                    "selected class ancestry is outside its checked product",
                )
            })?;
            builtin_ancestors
                .entry(name.clone())
                .or_insert_with(|| builtin.clone());
            parent = builtin.parent.as_ref();
        }
    }
    classes.extend(builtin_ancestors);
    Ok(std::rc::Rc::new(super::SelectedClassProjection::new(
        classes,
        method_bindings,
        sealed,
        abstract_classes,
        enumerations,
        generation,
    )?))
}

fn register_classes(
    classes: Vec<CheckedClassRegistration>,
    parent_admission: preflight::ParentAdmission,
) -> Result<(), crate::RuntimeError> {
    if classes.is_empty() {
        return Ok(());
    }
    let _default_roots =
        crate::execution::RootedValueLease::for_values(classes.iter().flat_map(|registration| {
            registration
                .class
                .properties
                .values()
                .filter_map(|property| property.default_value.as_ref())
        }))
        .map_err(|error| error.into_runtime_error())?;
    let mut proposed = BTreeMap::new();
    for registration in classes {
        validate_method_bindings(&registration)?;
        if proposed
            .insert(registration.class.name.clone(), registration)
            .is_some()
        {
            return Err(error(
                "RunMat:ClassDuplicate",
                "class batch repeats a declaration",
            ));
        }
    }
    // Register the existing retained-value visitor before transferring any
    // default into the live state. The temporary lease spans the entire move.
    for registration in proposed.values() {
        super::retained::note_default_roots(&registration.class);
    }
    let retired = super::with_state_mut(|state| {
        let order = preflight::validate(&proposed, state, parent_admission)?;
        let mut retired = Vec::with_capacity(order.len());
        for name in order {
            state.builtin_issuances.remove(&name);
            let registration = proposed
                .remove(&name)
                .expect("preflight returned only proposed classes");
            let class = registration.class;
            state.generation = state.generation.wrapping_add(1).max(1);
            let bindings = super::bind_methods_for_class_with(
                &class,
                ClassRegistryRevision::Current(state.generation),
                Some(&registration.method_bindings),
            )
            .collect::<Vec<_>>();
            state.method_bindings.retain(|(owner, _), _| owner != &name);
            state.method_bindings.extend(bindings);
            if let Some(previous) = state.classes.insert(name.clone(), class) {
                retired.push(previous);
            }
            super::set_membership(&mut state.sealed, &name, registration.is_sealed);
            super::set_membership(&mut state.abstract_classes, &name, registration.is_abstract);
            state
                .enumerations
                .insert(name, registration.enumerations.into_iter().collect());
        }
        Ok::<_, crate::RuntimeError>(retired)
    })?;
    drop(retired);
    Ok(())
}
fn validate_method_bindings(
    registration: &CheckedClassRegistration,
) -> Result<(), crate::RuntimeError> {
    if registration
        .method_bindings
        .iter()
        .any(|(method, binding)| {
            if !registration.class.methods.contains_key(method) {
                return true;
            }
            let issued = match &binding.authority {
                runmat_value::BoundFunctionAuthority::Portable { origin } => Some(*origin),
                runmat_value::BoundFunctionAuthority::SelectedNative { origin, lease }
                    if lease.is_live()
                        && lease.origin() == *origin
                        && lease.compiled_function().0 == binding.function =>
                {
                    Some(*origin)
                }
                _ => None,
            };
            issued.is_none_or(|origin| {
                // The portable ordinal is not the local compiler coordinate.
                // Resolve through the invocation's already-issued source facts,
                // not a class/name resolver (the class batch is not published).
                crate::user_functions::resolve_binding_origin_binding(origin)
                    .is_none_or(|issued| issued != *binding)
            })
        })
    {
        return Err(error(
            "RunMat:ClassMethodBinding",
            "exact method binding has no declared method or live issued authority",
        ));
    }
    Ok(())
}
fn error(identifier: &str, message: &str) -> crate::RuntimeError {
    crate::runtime_error::semantic_error(identifier, message)
}

#[cfg(test)]
mod tests {
    use std::{collections::HashMap, rc::Rc};

    use super::*;

    fn class(name: &str, parent: Option<ClassIdentity>) -> RuntimeClass {
        RuntimeClass {
            name: ClassIdentity::new(name).expect("class identity"),
            parent,
            properties: HashMap::new(),
            methods: HashMap::new(),
            events: HashMap::new(),
        }
    }

    fn checked(class: RuntimeClass, is_sealed: bool) -> CheckedClassRegistration {
        CheckedClassRegistration {
            class,
            method_bindings: std::collections::HashMap::new(),
            is_sealed,
            is_abstract: false,
            enumerations: Vec::new(),
        }
    }

    #[test]
    fn empty_batch_keeps_the_current_registry_generation() {
        let runtime = crate::context::RuntimeContext::new(Rc::new(
            crate::execution::RuntimeExecutionService::new(),
        ));
        let _scope = runtime.enter();
        let generation = super::super::class_registry_generation();
        register_checked_classes(Vec::new()).expect("empty selected metadata");
        assert_eq!(super::super::class_registry_generation(), generation);
    }

    #[test]
    fn later_sealed_parent_rejection_publishes_no_class() {
        let runtime = crate::context::RuntimeContext::new(Rc::new(
            crate::execution::RuntimeExecutionService::new(),
        ));
        let _scope = runtime.enter();
        let base = ClassIdentity::new("BatchBase").expect("base");
        let child = ClassIdentity::new("BatchChild").expect("child");
        let generation = super::super::class_registry_generation();
        assert!(register_checked_classes(vec![
            checked(class("BatchChild", Some(base.clone())), false),
            checked(class("BatchBase", None), true),
        ])
        .is_err());
        assert_eq!(super::super::class_registry_generation(), generation);
        assert!(super::super::get_class(&base).is_none());
        assert!(super::super::get_class(&child).is_none());
    }

    #[test]
    fn later_cycle_rejection_publishes_no_class() {
        let runtime = crate::context::RuntimeContext::new(Rc::new(
            crate::execution::RuntimeExecutionService::new(),
        ));
        let _scope = runtime.enter();
        let base = ClassIdentity::new("CycleBase").expect("base");
        let child = ClassIdentity::new("CycleChild").expect("child");
        let generation = super::super::class_registry_generation();
        assert!(register_checked_classes(vec![
            checked(class("CycleBase", Some(child.clone())), false),
            checked(class("CycleChild", Some(base.clone())), false),
        ])
        .is_err());
        assert_eq!(super::super::class_registry_generation(), generation);
        assert!(super::super::get_class(&base).is_none());
        assert!(super::super::get_class(&child).is_none());
    }

    #[test]
    fn child_before_parent_commits_one_checked_batch() {
        let runtime = crate::context::RuntimeContext::new(Rc::new(
            crate::execution::RuntimeExecutionService::new(),
        ));
        let _scope = runtime.enter();
        let base = ClassIdentity::new("OrderedBase").expect("base");
        let child = ClassIdentity::new("OrderedChild").expect("child");
        register_checked_classes(vec![
            checked(class("OrderedChild", Some(base.clone())), false),
            checked(class("OrderedBase", None), false),
        ])
        .expect("complete graph");
        assert_eq!(
            super::super::get_class(&child).and_then(|class| class.parent),
            Some(base.clone())
        );
        assert!(super::super::get_class(&base).is_some());
    }

    #[test]
    fn replacing_existing_parent_cannot_close_a_mixed_cycle() {
        let runtime = crate::context::RuntimeContext::new(Rc::new(
            crate::execution::RuntimeExecutionService::new(),
        ));
        let _scope = runtime.enter();
        let base = ClassIdentity::new("MixedCycleBase").expect("base");
        let child = ClassIdentity::new("MixedCycleChild").expect("child");
        register_checked_classes(vec![
            checked(class("MixedCycleBase", None), false),
            checked(class("MixedCycleChild", Some(base.clone())), false),
        ])
        .expect("initial graph");
        let generation = super::super::class_registry_generation();
        assert!(
            register_checked_class(class("MixedCycleBase", Some(child)), false, false, vec![])
                .is_err()
        );
        assert_eq!(super::super::class_registry_generation(), generation);
        assert_eq!(
            super::super::get_class(&base).and_then(|class| class.parent),
            None
        );
    }
}

#[cfg(test)]
#[path = "registration/open_tests.rs"]
mod open_tests;

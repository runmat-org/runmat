use super::*;
use std::collections::{HashMap, HashSet};

#[derive(Clone, Copy)]
pub(super) enum ParentAdmission {
    Open,
    Closed,
}

fn class<'a>(
    name: &ClassIdentity,
    proposed: &'a BTreeMap<ClassIdentity, CheckedClassRegistration>,
    existing: &'a RuntimeClassState,
) -> Option<&'a RuntimeClass> {
    proposed
        .get(name)
        .map(|registration| &registration.class)
        .or_else(|| existing.classes.get(name))
}

fn sealed(
    name: &ClassIdentity,
    proposed: &BTreeMap<ClassIdentity, CheckedClassRegistration>,
    existing: &RuntimeClassState,
) -> bool {
    proposed
        .get(name)
        .map(|registration| registration.is_sealed)
        .unwrap_or_else(|| existing.sealed.contains(name))
}

/// Validate known proposed+existing edges without copying class Values. Open
/// source declarations may retain unresolved external parents; closed
/// executable admission requires its proposed ancestry to be complete. The
/// returned order visits every proposed parent before its child.
pub(super) fn validate(
    proposed: &BTreeMap<ClassIdentity, CheckedClassRegistration>,
    existing: &RuntimeClassState,
    parent_admission: ParentAdmission,
) -> Result<Vec<ClassIdentity>, crate::RuntimeError> {
    let mut completeness = HashMap::<ClassIdentity, bool>::new();
    for start in proposed.keys().chain(existing.classes.keys()) {
        if completeness.contains_key(start) {
            continue;
        }
        let mut visiting = HashSet::new();
        let mut path = Vec::new();
        let mut cursor = Some(start);
        let complete = loop {
            let Some(name) = cursor else {
                break true;
            };
            if let Some(&known) = completeness.get(name) {
                break known;
            }
            if !visiting.insert(name) {
                return Err(error(
                    "RunMat:ClassInheritanceCycle",
                    "class registration would create an inheritance cycle",
                ));
            }
            let Some(current) = class(name, proposed, existing) else {
                break super::super::has_intrinsic_implicit_constructor(name);
            };
            path.push(name);
            cursor = current.parent.as_ref();
        };
        if !complete
            && matches!(parent_admission, ParentAdmission::Closed)
            && path.iter().any(|visited| proposed.contains_key(*visited))
        {
            return Err(error(
                "RunMat:ClassParentMissing",
                "class inheritance references an unregistered parent",
            ));
        }
        for name in path {
            completeness.insert(name.clone(), complete);
        }
    }
    let mut checked = HashSet::new();
    for name in proposed.keys().chain(existing.classes.keys()) {
        if !checked.insert(name) {
            continue;
        }
        let current = class(name, proposed, existing)
            .expect("validation only visits registered or proposed classes");
        let lineage = lineage(current.parent.as_ref(), proposed, existing);
        if !proposed.contains_key(name)
            && !lineage
                .iter()
                .any(|ancestor| proposed.contains_key(&ancestor.name))
        {
            continue;
        }
        if let Some(parent) = current.parent.as_ref() {
            if sealed(parent, proposed, existing) {
                return Err(error(
                    "RunMat:ClassSealed",
                    &format!("Cannot subclass sealed class '{}'.", parent),
                ));
            }
        }
        let mut inherited_sealed = HashSet::new();
        let mut required_abstract = HashSet::new();
        for ancestor in lineage.into_iter().rev() {
            for (name, method) in &ancestor.methods {
                if method.is_sealed {
                    inherited_sealed.insert(name.clone());
                }
                if method.is_abstract {
                    required_abstract.insert(name.clone());
                } else {
                    required_abstract.remove(name);
                }
            }
        }
        if let Some(conflict) = current
            .methods
            .keys()
            .find(|method| inherited_sealed.contains(*method))
        {
            return Err(error(
                "RunMat:MethodSealed",
                &format!(
                    "Class '{}' cannot override sealed method '{}'.",
                    current.name, conflict
                ),
            ));
        }
        let is_abstract = proposed
            .get(name)
            .map(|registration| registration.is_abstract)
            .unwrap_or_else(|| existing.abstract_classes.contains(name));
        if !is_abstract {
            for (name, method) in &current.methods {
                if method.is_abstract {
                    required_abstract.insert(name.clone());
                } else {
                    required_abstract.remove(name);
                }
            }
            if let Some(missing) = required_abstract.into_iter().next() {
                return Err(error(
                    "RunMat:AbstractMethodMissing",
                    &format!(
                        "Class '{}' must implement abstract method '{}'.",
                        current.name, missing
                    ),
                ));
            }
        }
    }
    let mut order = Vec::with_capacity(proposed.len());
    let mut emitted = HashSet::new();
    for start in proposed.keys() {
        let mut path = Vec::new();
        let mut cursor = Some(start);
        while let Some(name) = cursor {
            if emitted.contains(name) || !proposed.contains_key(name) {
                break;
            }
            path.push(name.clone());
            cursor = proposed[name].class.parent.as_ref();
        }
        for name in path.into_iter().rev() {
            if emitted.insert(name.clone()) {
                order.push(name);
            }
        }
    }
    Ok(order)
}

fn lineage<'a>(
    parent: Option<&ClassIdentity>,
    proposed: &'a BTreeMap<ClassIdentity, CheckedClassRegistration>,
    existing: &'a RuntimeClassState,
) -> Vec<&'a RuntimeClass> {
    let mut ancestors = Vec::new();
    let mut cursor = parent;
    while let Some(name) = cursor {
        let Some(ancestor) = class(name, proposed, existing) else {
            break;
        };
        ancestors.push(ancestor);
        cursor = ancestor.parent.as_ref();
    }
    ancestors
}

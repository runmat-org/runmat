//! Build an execution view from exact prior definitions and newly issued
//! bodies. A one-pass remap may leave speculative copies in the compiled
//! product, but they are never published over the prior definition owner.

use std::collections::{HashMap, HashSet};
use std::rc::Rc;
use std::sync::Arc;

use runmat_hir::FunctionId;
use runmat_types::FunctionSourceRole;

use super::{
    semantic_fingerprint::{
        semantic_definition_fingerprint, semantic_definition_fingerprint_with_source,
    },
    BytecodeSourceRef, FunctionBytecode, FunctionRegistry,
};

fn same_selected_issuance(left: &FunctionBytecode, right: &FunctionBytecode) -> bool {
    let same_source = match (&left.selected_source_owner, &right.selected_source_owner) {
        (None, None) => true,
        (Some(left), Some(right)) => left.same_issuance(right),
        _ => false,
    };
    let same_product = match (
        &left.selected_executable_product,
        &right.selected_executable_product,
    ) {
        (None, None) => true,
        (Some(left), Some(right)) => Rc::ptr_eq(left, right),
        _ => false,
    };
    let same_capture = match (
        &left.selected_static_capture,
        &right.selected_static_capture,
    ) {
        (None, None) => true,
        (Some(left), Some(right)) => Rc::ptr_eq(left, right),
        _ => false,
    };
    same_source && same_product && same_capture
}

impl FunctionRegistry {
    /// Select the runtime registry after the compiler has proved its closed
    /// remap. `included_current` is the fresh callable set admitted by the
    /// new product; `partitioned_reused` is its disjoint set of proved prior
    /// definitions. Neither a matching numeric ID nor a matching origin can
    /// replace the exact prior body/source/lease for the latter set.
    /// Other prior definitions omitted from the projection are retired; they
    /// cannot reappear simply because selection starts from the prior owner.
    pub fn selected_execution_registry_from_prior(
        prior: &Self,
        projected: &Self,
        included_current: &HashSet<FunctionId>,
        partitioned_reused: &HashSet<FunctionId>,
        projected_to_prior_source: impl Fn(BytecodeSourceRef) -> Option<BytecodeSourceRef>,
    ) -> Result<Self, String> {
        if !included_current.is_disjoint(partitioned_reused) {
            return Err("current and reused definition partitions overlap".into());
        }
        for id in partitioned_reused {
            let old = prior
                .get_current(*id)
                .ok_or("reused definition is absent from exact prior registry")?;
            if old.binding_origin.is_none() {
                return Err("reused definition has no exact selected origin".into());
            }
            // An active compiled view contains the speculative remap body;
            // a publication-only view may omit it entirely. In either case
            // the selected execution body is the exact prior definition.
            if let Some(speculative) = projected.get_current(*id) {
                if old.binding_origin != speculative.binding_origin
                    || old.display_name != speculative.display_name
                    || old.source_role != speculative.source_role
                    || !Arc::ptr_eq(&old.definition_owner, &speculative.definition_owner)
                {
                    return Err("reused definition differs from its exact prior owner".into());
                }
            }
        }
        // The projected registry also carries inherited bodies. They must be
        // the same selected definitions, never a newly compiled body silently
        // omitted from the current set.
        for (id, body) in &projected.functions {
            if included_current.contains(id) || partitioned_reused.contains(id) {
                continue;
            }
            let old = prior
                .get_current(*id)
                .ok_or("projected body has no selected current partition")?;
            let source_agrees = if old.source_id == body.source_id {
                true
            } else {
                match (body.source_id, old.source_id) {
                    (Some(projected), Some(prior_source)) => {
                        projected_to_prior_source(projected) == Some(prior_source)
                    }
                    _ => false,
                }
            };
            if old.binding_origin != body.binding_origin
                || !Arc::ptr_eq(&old.definition_owner, &body.definition_owner)
                || !same_selected_issuance(old, body)
                || old.private_source_owner != body.private_source_owner
                || !source_agrees
                || semantic_definition_fingerprint(old).map_err(|error| error.to_string())?
                    != semantic_definition_fingerprint_with_source(body, old.source_id)
                        .map_err(|error| error.to_string())?
            {
                return Err("projected inherited body differs from exact prior issuance".into());
            }
        }
        let prior_owners = prior
            .functions
            .values()
            .chain(prior.retired.values().flatten())
            .map(|body| Arc::as_ptr(&body.definition_owner) as usize)
            .collect::<HashSet<_>>();
        let mut display_names = HashMap::new();
        let mut fresh_owners = HashSet::new();
        let mut fresh = Vec::with_capacity(included_current.len());
        for id in included_current {
            let body = projected
                .get_current(*id)
                .ok_or("fresh definition is absent from projected registry")?;
            let owner = Arc::as_ptr(&body.definition_owner) as usize;
            if body.binding_origin.is_none()
                || prior_owners.contains(&owner)
                || !fresh_owners.insert(owner)
            {
                return Err("fresh definition owner is not unique to its selected body".into());
            }
            if prior
                .resolve_name(&body.display_name)
                .is_some_and(|old_id| partitioned_reused.contains(&old_id))
            {
                return Err("fresh definition would replace a reused name".into());
            }
            if body.source_role != FunctionSourceRole::Nested
                && projected.resolve_name(&body.display_name) != Some(*id)
            {
                return Err("fresh display name selects a different projected definition".into());
            }
            if body.source_role != FunctionSourceRole::Nested
                && display_names
                    .insert(body.display_name.as_str(), *id)
                    .is_some()
            {
                return Err("fresh definitions repeat a published display name".into());
            }
            fresh.push((*id, body.clone()));
        }
        for (name, id) in &projected.names {
            if included_current.contains(id)
                && prior
                    .resolve_name(name)
                    .is_some_and(|old_id| partitioned_reused.contains(&old_id))
            {
                return Err("fresh alias would replace a reused name".into());
            }
            if included_current.contains(id)
                && projected
                    .get_current(*id)
                    .is_none_or(|body| body.source_role == FunctionSourceRole::Nested)
            {
                return Err(format!("fresh alias {name:?} has no named definition"));
            }
        }
        fresh.sort_by_key(|(id, _)| id.0);
        let mut selected = prior.clone();
        let retired = prior
            .functions
            .keys()
            .filter(|id| !projected.functions.contains_key(id) && !partitioned_reused.contains(id))
            .copied()
            .collect::<Vec<_>>();
        for id in retired {
            selected.remove(id);
        }
        for (id, body) in fresh {
            // Removing the genuinely old owner first prevents the ordinary
            // same-origin insertion policy from grafting it onto a new body.
            // Old escaped handles stay in retired under their own leases.
            if selected.get_current(id).is_some() {
                selected.remove(id);
            }
            let expected_owner = Arc::clone(&body.definition_owner);
            selected.insert_replacing_name(body);
            if selected
                .get_current(id)
                .is_none_or(|selected| !Arc::ptr_eq(&selected.definition_owner, &expected_owner))
            {
                return Err("selected insertion changed the fresh definition lease".into());
            }
        }
        for (name, id) in &projected.names {
            if included_current.contains(id) {
                selected.names.insert(name.clone(), *id);
            }
        }
        Ok(selected)
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use runmat_execution::{Digest, FunctionBindingOrigin};
    use runmat_types::{ProgramFunctionId, ProgramSourceId};
    use runmat_value::FunctionDefinitionOwner;

    use super::*;

    #[test]
    fn omitted_definition_loses_its_name_but_keeps_escaped_handles_alive() {
        let id = FunctionId(7);
        let body = FunctionBytecode {
            function: id,
            display_name: "removed".into(),
            ..FunctionBytecode::default()
        };
        let lease = FunctionDefinitionOwner::issue(&body.definition_owner, id.0);
        let mut prior = FunctionRegistry::new(HashMap::from([(id, body.clone())]));
        prior.names.insert(body.display_name.clone(), id);
        let selected = FunctionRegistry::selected_execution_registry_from_prior(
            &prior,
            &FunctionRegistry::default(),
            &HashSet::new(),
            &HashSet::new(),
            |_| None,
        )
        .expect("retire omitted definition");
        assert!(selected.resolve_name("removed").is_none());
        assert!(selected.get_current(id).is_none());
        assert!(Arc::ptr_eq(
            &selected.get_owned(id, &lease).unwrap().definition_owner,
            &body.definition_owner,
        ));
        assert_eq!(prior.resolve_name("removed"), Some(id));
    }

    #[test]
    fn reused_owner_keeps_its_prior_source_while_new_body_is_published() {
        let reused = FunctionId(7);
        let fresh = FunctionId(8);
        let prior_body = FunctionBytecode {
            function: reused,
            display_name: "old".into(),
            source_id: Some(BytecodeSourceRef::Program(ProgramSourceId(1))),
            binding_origin: Some(FunctionBindingOrigin {
                program: Digest::sha256(b"prior selected program"),
                function: ProgramFunctionId(7),
            }),
            ..FunctionBytecode::default()
        };
        let lease = FunctionDefinitionOwner::issue(&prior_body.definition_owner, reused.0);
        let prior = FunctionRegistry::new(HashMap::from([(reused, prior_body.clone())]));
        let mut speculative = prior_body.clone();
        speculative.source_id = Some(BytecodeSourceRef::Program(ProgramSourceId(2)));
        let fresh_body = FunctionBytecode {
            function: fresh,
            display_name: "new".into(),
            source_id: Some(BytecodeSourceRef::Program(ProgramSourceId(2))),
            binding_origin: Some(FunctionBindingOrigin {
                program: Digest::sha256(b"new selected program"),
                function: ProgramFunctionId(8),
            }),
            ..FunctionBytecode::default()
        };
        let projected =
            FunctionRegistry::new(HashMap::from([(reused, speculative), (fresh, fresh_body)]));
        let selected = FunctionRegistry::selected_execution_registry_from_prior(
            &prior,
            &projected,
            &HashSet::from([fresh]),
            &HashSet::from([reused]),
            |_| None,
        )
        .expect("exact partition");
        assert_eq!(
            selected
                .get_owned(reused, &lease)
                .and_then(|body| body.source_id),
            prior_body.source_id
        );
        assert!(Arc::ptr_eq(
            &selected.get_current(reused).unwrap().definition_owner,
            &prior_body.definition_owner,
        ));
        assert_eq!(selected.get_current(fresh).unwrap().display_name, "new");
        assert!(!selected.retired.contains_key(&reused));
        let publication_only = FunctionRegistry::new(HashMap::from([(
            fresh,
            projected.get_current(fresh).unwrap().clone(),
        )]));
        let published = FunctionRegistry::selected_execution_registry_from_prior(
            &prior,
            &publication_only,
            &HashSet::from([fresh]),
            &HashSet::from([reused]),
            |_| None,
        )
        .expect("publication selects the exact omitted prior body");
        assert!(Arc::ptr_eq(
            &published.get_current(reused).unwrap().definition_owner,
            &prior_body.definition_owner,
        ));
        assert!(!published.retired.contains_key(&reused));
    }

    #[test]
    fn fresh_name_cannot_retire_a_partitioned_prior_definition() {
        let reused = FunctionId(7);
        let fresh = FunctionId(8);
        let prior_body = FunctionBytecode {
            function: reused,
            display_name: "old".into(),
            binding_origin: Some(FunctionBindingOrigin {
                program: Digest::sha256(b"prior selected program"),
                function: ProgramFunctionId(7),
            }),
            ..FunctionBytecode::default()
        };
        let prior = FunctionRegistry::new(HashMap::from([(reused, prior_body.clone())]));
        let fresh_body = FunctionBytecode {
            function: fresh,
            display_name: "old".into(),
            binding_origin: Some(FunctionBindingOrigin {
                program: Digest::sha256(b"new selected program"),
                function: ProgramFunctionId(8),
            }),
            ..FunctionBytecode::default()
        };
        let projected = FunctionRegistry::new(HashMap::from([
            (reused, prior_body.clone()),
            (fresh, fresh_body),
        ]));
        assert!(FunctionRegistry::selected_execution_registry_from_prior(
            &prior,
            &projected,
            &HashSet::from([fresh]),
            &HashSet::from([reused]),
            |_| None,
        )
        .is_err());
        assert!(Arc::ptr_eq(
            &prior.get_current(reused).unwrap().definition_owner,
            &prior_body.definition_owner,
        ));
    }

    #[test]
    fn two_fresh_definitions_cannot_share_one_owner() {
        let first = FunctionId(8);
        let second = FunctionId(9);
        let first_body = FunctionBytecode {
            function: first,
            display_name: "first".into(),
            binding_origin: Some(FunctionBindingOrigin {
                program: Digest::sha256(b"new selected program"),
                function: ProgramFunctionId(8),
            }),
            ..FunctionBytecode::default()
        };
        let second_body = FunctionBytecode {
            function: second,
            display_name: "second".into(),
            binding_origin: Some(FunctionBindingOrigin {
                program: Digest::sha256(b"new selected program"),
                function: ProgramFunctionId(9),
            }),
            definition_owner: Arc::clone(&first_body.definition_owner),
            ..FunctionBytecode::default()
        };
        let projected =
            FunctionRegistry::new(HashMap::from([(first, first_body), (second, second_body)]));
        assert!(FunctionRegistry::selected_execution_registry_from_prior(
            &FunctionRegistry::default(),
            &projected,
            &HashSet::from([first, second]),
            &HashSet::new(),
            |_| None,
        )
        .is_err());
    }

    #[test]
    fn inherited_projection_requires_an_exact_source_alias() {
        let id = FunctionId(3);
        let prior_body = FunctionBytecode {
            function: id,
            display_name: "inherited".into(),
            source_id: Some(BytecodeSourceRef::Program(ProgramSourceId(1))),
            binding_origin: Some(FunctionBindingOrigin {
                program: Digest::sha256(b"inherited product"),
                function: ProgramFunctionId(3),
            }),
            ..FunctionBytecode::default()
        };
        let prior = FunctionRegistry::new(HashMap::from([(id, prior_body.clone())]));
        let mut projected_body = prior_body;
        projected_body.source_id = Some(BytecodeSourceRef::Program(ProgramSourceId(2)));
        let projected = FunctionRegistry::new(HashMap::from([(id, projected_body)]));
        let empty = HashSet::new();
        assert!(FunctionRegistry::selected_execution_registry_from_prior(
            &prior,
            &projected,
            &empty,
            &empty,
            |_| None,
        )
        .is_err());
        let selected = FunctionRegistry::selected_execution_registry_from_prior(
            &prior,
            &projected,
            &empty,
            &empty,
            |source| {
                (source == BytecodeSourceRef::Program(ProgramSourceId(2)))
                    .then_some(BytecodeSourceRef::Program(ProgramSourceId(1)))
            },
        )
        .expect("authenticated source alias selects exact prior body");
        assert_eq!(
            selected.get_current(id).unwrap().source_id,
            prior.get_current(id).unwrap().source_id,
        );
        let mut without_prior_source = prior.get_current(id).unwrap().clone();
        without_prior_source.source_id = None;
        let missing_source_prior =
            FunctionRegistry::new(HashMap::from([(id, without_prior_source)]));
        assert!(FunctionRegistry::selected_execution_registry_from_prior(
            &missing_source_prior,
            &projected,
            &empty,
            &empty,
            |_| None,
        )
        .is_err());
    }
}

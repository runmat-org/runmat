use std::collections::{BTreeMap, BTreeSet};

use super::{
    exact_topology_validation_support::{
        claim_unique, invalid, require_ordered_refs, require_reference, validate_transform,
    },
    ExactBRepTopologyV2, GeometryContractError, PersistentEntityId, PersistentEntityKind,
};

pub(super) fn validate_assembly_occurrences(
    topology: &ExactBRepTopologyV2,
    assemblies: &BTreeSet<PersistentEntityId>,
    instances: &BTreeSet<PersistentEntityId>,
    bodies: &BTreeSet<PersistentEntityId>,
) -> Result<(), GeometryContractError> {
    require_reference(
        "root assembly",
        &topology.root_assembly_id,
        PersistentEntityKind::Assembly,
        assemblies,
    )?;

    let mut claimed_instances = BTreeSet::new();
    let mut claimed_instance_parents = BTreeMap::new();
    let mut claimed_bodies = BTreeSet::new();
    let mut occurrence_paths = BTreeSet::new();
    for assembly in &topology.assemblies {
        if assembly.definition_digest == [0; 32] {
            return Err(invalid(
                "assembly definition",
                "definition identity must be nonzero",
            ));
        }
        if !occurrence_paths.insert(assembly.id.assembly_path.clone()) {
            return Err(invalid(
                "assembly occurrence path",
                "each expanded assembly occurrence must have one stable path",
            ));
        }
        require_ordered_refs(
            "assembly bodies",
            &assembly.body_ids,
            PersistentEntityKind::Body,
            bodies,
            false,
        )?;
        if assembly
            .body_ids
            .iter()
            .any(|body| body.assembly_path != assembly.id.assembly_path)
        {
            return Err(invalid(
                "assembly body identity",
                "body semantic identities must be scoped to their assembly occurrence",
            ));
        }
        claim_unique("body ownership", &assembly.body_ids, &mut claimed_bodies)?;
        require_ordered_refs(
            "assembly child instances",
            &assembly.child_instance_ids,
            PersistentEntityKind::Instance,
            instances,
            false,
        )?;
        claim_unique(
            "instance ownership",
            &assembly.child_instance_ids,
            &mut claimed_instances,
        )?;
        for instance_id in &assembly.child_instance_ids {
            claimed_instance_parents.insert(instance_id.clone(), assembly.id.clone());
        }
    }
    if claimed_instances != *instances {
        return Err(invalid(
            "instance ownership",
            "every instance must have one parent assembly owner",
        ));
    }
    if claimed_bodies != *bodies {
        return Err(invalid(
            "body ownership",
            "every body must have one assembly occurrence owner",
        ));
    }

    let assembly_by_id = topology
        .assemblies
        .iter()
        .map(|assembly| (&assembly.id, assembly))
        .collect::<BTreeMap<_, _>>();
    let instance_by_id = topology
        .instances
        .iter()
        .map(|instance| (&instance.id, instance))
        .collect::<BTreeMap<_, _>>();
    let mut instantiated_assemblies = BTreeSet::new();
    for instance in &topology.instances {
        require_reference(
            "instance parent assembly",
            &instance.parent_assembly_id,
            PersistentEntityKind::Assembly,
            assemblies,
        )?;
        require_reference(
            "instantiated assembly",
            &instance.instantiated_assembly_id,
            PersistentEntityKind::Assembly,
            assemblies,
        )?;
        let parent = assembly_by_id
            .get(&instance.parent_assembly_id)
            .ok_or_else(|| invalid("instance parent assembly", "reference index is incomplete"))?;
        let child = assembly_by_id
            .get(&instance.instantiated_assembly_id)
            .ok_or_else(|| invalid("instantiated assembly", "reference index is incomplete"))?;
        if instance.id.assembly_path != child.id.assembly_path
            || child.id.assembly_path.len() <= parent.id.assembly_path.len()
            || !child.id.assembly_path.starts_with(&parent.id.assembly_path)
        {
            return Err(invalid(
                "instance semantic identity",
                "instance and child assembly must share a strict descendant occurrence path",
            ));
        }
        if claimed_instance_parents.get(&instance.id) != Some(&instance.parent_assembly_id) {
            return Err(invalid(
                "assembly instance incidence",
                "every instance must occur in its parent assembly",
            ));
        }
        if instance.instantiated_assembly_id == topology.root_assembly_id
            || !instantiated_assemblies.insert(instance.instantiated_assembly_id.clone())
        {
            return Err(invalid(
                "assembly occurrence ownership",
                "the root has no parent and every other assembly has one instance owner",
            ));
        }
        validate_transform(&instance.transform.0)?;
    }
    let expected_children = assemblies
        .iter()
        .filter(|id| *id != &topology.root_assembly_id)
        .cloned()
        .collect::<BTreeSet<_>>();
    if instantiated_assemblies != expected_children {
        return Err(invalid(
            "assembly occurrence ownership",
            "every non-root assembly must be instantiated exactly once",
        ));
    }

    let mut reachable = BTreeSet::from([topology.root_assembly_id.clone()]);
    let mut frontier = vec![topology.root_assembly_id.clone()];
    while let Some(assembly_id) = frontier.pop() {
        let assembly = assembly_by_id
            .get(&assembly_id)
            .ok_or_else(|| invalid("assembly occurrence graph", "assembly index is incomplete"))?;
        for instance_id in &assembly.child_instance_ids {
            let child = &instance_by_id
                .get(instance_id)
                .ok_or_else(|| {
                    invalid("assembly occurrence graph", "instance index is incomplete")
                })?
                .instantiated_assembly_id;
            if reachable.insert(child.clone()) {
                frontier.push(child.clone());
            }
        }
    }
    if reachable != *assemblies {
        return Err(invalid(
            "assembly occurrence graph",
            "all assemblies must be reachable from the root without cycles",
        ));
    }
    Ok(())
}

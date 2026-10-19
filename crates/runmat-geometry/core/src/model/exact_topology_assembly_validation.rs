use super::{
    exact_topology_validation_support::{
        claim_unique, invalid, require_ordered_refs, require_reference, validate_transform,
    },
    validation_storage::{lookup, Claims, IdIndex, ValidationWork},
    ExactBRepTopology, GeometryContractError, PersistentEntityKind,
};

pub(super) fn validate_assembly_occurrences(
    topology: &ExactBRepTopology,
    assemblies: &IdIndex<'_>,
    instances: &IdIndex<'_>,
    bodies: &IdIndex<'_>,
    work: ValidationWork<'_>,
) -> Result<(), GeometryContractError> {
    require_reference(
        "root assembly",
        &topology.root_assembly_id,
        PersistentEntityKind::Assembly,
        assemblies,
    )?;
    let mut claimed_instances = Claims::new(instances, work)?;
    let mut claimed_bodies = Claims::new(bodies, work)?;
    let mut paths = work.buffer(topology.assemblies.len())?;
    for assembly in &topology.assemblies {
        paths.push(&assembly.id.assembly_path);
    }
    paths.sort_unstable();
    if paths.windows(2).any(|pair| pair[0] == pair[1]) {
        return Err(invalid(
            "assembly occurrence path",
            "each expanded assembly occurrence must have one stable path",
        ));
    }
    drop(paths);
    for assembly in &topology.assemblies {
        work.checkpoint()?;
        if assembly.definition_digest == [0; 32] {
            return Err(invalid(
                "assembly definition",
                "definition identity must be nonzero",
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
        for id in &assembly.child_instance_ids {
            let instance =
                lookup(&topology.instances, id, |instance| &instance.id).ok_or_else(|| {
                    invalid(
                        "assembly instance incidence",
                        "instance index is incomplete",
                    )
                })?;
            if instance.parent_assembly_id != assembly.id {
                return Err(invalid(
                    "assembly instance incidence",
                    "every instance must occur in its parent assembly",
                ));
            }
        }
    }
    if !claimed_instances.complete() {
        return Err(invalid(
            "instance ownership",
            "every instance must have one parent assembly owner",
        ));
    }
    if !claimed_bodies.complete() {
        return Err(invalid(
            "body ownership",
            "every body must have one assembly occurrence owner",
        ));
    }
    let mut instantiated = Claims::new(assemblies, work)?;
    for instance in &topology.instances {
        work.checkpoint()?;
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
        let parent = lookup(
            &topology.assemblies,
            &instance.parent_assembly_id,
            |assembly| &assembly.id,
        )
        .ok_or_else(|| invalid("instance parent assembly", "reference index is incomplete"))?;
        let child = lookup(
            &topology.assemblies,
            &instance.instantiated_assembly_id,
            |assembly| &assembly.id,
        )
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
        if instance.instantiated_assembly_id == topology.root_assembly_id
            || !instantiated.insert(&instance.instantiated_assembly_id)
        {
            return Err(invalid(
                "assembly occurrence ownership",
                "the root has no parent and every other assembly has one instance owner",
            ));
        }
        validate_transform(&instance.transform.0)?;
    }
    if instantiated.count() + 1 != assemblies.len() {
        return Err(invalid(
            "assembly occurrence ownership",
            "every non-root assembly must be instantiated exactly once",
        ));
    }
    let mut reachable = Claims::new(assemblies, work)?;
    let mut frontier = work.buffer(topology.assemblies.len())?;
    reachable.insert(&topology.root_assembly_id);
    frontier.push(&topology.root_assembly_id);
    while let Some(id) = frontier.pop() {
        work.checkpoint()?;
        let assembly = lookup(&topology.assemblies, id, |assembly| &assembly.id)
            .ok_or_else(|| invalid("assembly occurrence graph", "assembly index is incomplete"))?;
        for instance_id in &assembly.child_instance_ids {
            let child = &lookup(&topology.instances, instance_id, |instance| &instance.id)
                .ok_or_else(|| {
                    invalid("assembly occurrence graph", "instance index is incomplete")
                })?
                .instantiated_assembly_id;
            if reachable.insert(child) {
                frontier.push(child);
            }
        }
    }
    if !reachable.complete() {
        return Err(invalid(
            "assembly occurrence graph",
            "all assemblies must be reachable from the root without cycles",
        ));
    }
    Ok(())
}

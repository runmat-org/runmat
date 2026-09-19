use std::collections::{BTreeMap, BTreeSet};

use runmat_geometry_core::{
    ExactBRepTopology, GeometryRegionRole, PersistentEntityId, PersistentEntityKind,
};
use runmat_geometry_ops::{GeometryEntityLineage, GeometryTopologyHistory};

use crate::exact::ImportedExactCad;

use super::{feature::OcctExactFeatureError, ffi};

pub(super) fn topology_history(
    dependencies: &[(&str, &ImportedExactCad)],
    output: &ImportedExactCad,
    payload: &ffi::bridge::OcctFeatureExecutionPayload,
) -> Result<GeometryTopologyHistory, OcctExactFeatureError> {
    let result_entities = role_inventory(&output.topology);
    let result = result_entities
        .values()
        .flatten()
        .cloned()
        .collect::<BTreeSet<_>>();
    let mut source_lookup = BTreeMap::new();
    let mut expected_sources = BTreeSet::new();
    for (operand_index, (_, geometry)) in dependencies.iter().enumerate() {
        for entity in kernel_topology_ids(&geometry.topology) {
            let key = (
                operand_index as u64,
                entity.kind,
                entity.source_topology_id.clone(),
            );
            if source_lookup.insert(key.clone(), entity.clone()).is_some() {
                return Err(OcctExactFeatureError::Contract(
                    "feature dependencies contain colliding persistent entity identities".into(),
                ));
            }
            expected_sources.insert(key);
        }
    }
    let mut reported_sources = BTreeSet::new();
    let mut deleted = BTreeMap::new();
    let mut source_to_result = BTreeMap::<PersistentEntityId, Vec<PersistentEntityId>>::new();
    for source in &payload.sources {
        let kind = persistent_kind(source.kind)?;
        let key = (
            source.operand_index,
            kind,
            digest_name(&source.identity_digest)?,
        );
        let entity = source_lookup.get(&key).ok_or_else(|| {
            let expected = source_lookup
                .keys()
                .filter(|candidate| candidate.0 == key.0 && candidate.1 == key.1)
                .map(|candidate| candidate.2.as_str())
                .collect::<Vec<_>>();
            OcctExactFeatureError::Contract(format!(
                "kernel history source operand {} {:?} `{}` does not match admitted identities {expected:?}",
                key.0, key.1, key.2,
            ))
        })?;
        if !reported_sources.insert(key) {
            return Err(OcctExactFeatureError::Contract(
                "kernel history repeats a dependency entity".into(),
            ));
        }
        deleted.insert(entity.clone(), source.deleted);
        let mut descendants = Vec::new();
        if result.contains(entity) {
            descendants.push(entity.clone());
        }
        source_to_result.insert(entity.clone(), descendants);
    }
    if reported_sources != expected_sources {
        return Err(OcctExactFeatureError::Contract(
            "kernel history source inventory is incomplete".into(),
        ));
    }

    let mut introduced = BTreeSet::new();
    for relation in &payload.relations {
        let source_kind = persistent_kind(relation.source_kind)?;
        let source_key = (
            relation.operand_index,
            source_kind,
            digest_name(&relation.source_identity_digest)?,
        );
        let source = source_lookup.get(&source_key).ok_or_else(|| {
            OcctExactFeatureError::Contract(
                "kernel relation source does not match an admitted dependency entity".into(),
            )
        })?;
        let result_kind = persistent_kind(relation.result_kind)?;
        let target_key = (1, result_kind, relation.result_shape_key);
        let target = output.kernel_entity_ids.get(&target_key).ok_or_else(|| {
            OcctExactFeatureError::Contract(format!(
                "kernel relation target {:?} shape key {} does not match imported result topology",
                result_kind, relation.result_shape_key
            ))
        })?;
        source_to_result
            .get_mut(source)
            .ok_or_else(|| {
                OcctExactFeatureError::Contract(
                    "kernel relation source is absent from its complete inventory".into(),
                )
            })?
            .push(target.clone());
        if relation.relation == ffi::bridge::OcctFeatureRelationKind::Generated {
            introduced.insert(target.clone());
        }
    }

    append_region_lineage(
        dependencies,
        &output.topology,
        &mut source_to_result,
        &mut deleted,
        &mut introduced,
    )?;
    for descendants in source_to_result.values_mut() {
        descendants.sort();
        descendants.dedup();
    }
    for (source, descendants) in &source_to_result {
        if descendants.is_empty() && !deleted.get(source).copied().unwrap_or(false) {
            return Err(OcctExactFeatureError::Contract(format!(
                "complete kernel history does not account for source {:?} entity `{}`",
                source.kind, source.source_topology_id
            )));
        }
    }
    let reached = source_to_result
        .values()
        .flatten()
        .cloned()
        .collect::<BTreeSet<_>>();
    introduced.extend(result.difference(&reached).cloned());
    let history = GeometryTopologyHistory {
        inventory_complete: true,
        result_entities,
        introduced: introduced.into_iter().collect(),
        source_lineage: source_to_result
            .into_iter()
            .map(|(source, descendants)| GeometryEntityLineage {
                source,
                descendants,
            })
            .collect(),
    };
    history
        .validate()
        .map_err(|error| OcctExactFeatureError::Contract(error.to_string()))?;
    Ok(history)
}

fn kernel_topology_ids(topology: &ExactBRepTopology) -> impl Iterator<Item = &PersistentEntityId> {
    topology
        .solids
        .iter()
        .map(|entity| &entity.id)
        .chain(topology.faces.iter().map(|entity| &entity.id))
        .chain(topology.edges.iter().map(|entity| &entity.id))
        .chain(topology.vertices.iter().map(|entity| &entity.id))
}

pub(super) fn role_inventory(
    topology: &ExactBRepTopology,
) -> BTreeMap<GeometryRegionRole, Vec<PersistentEntityId>> {
    let mut inventory = BTreeMap::from([
        (
            GeometryRegionRole::Vertex,
            topology
                .vertices
                .iter()
                .map(|entity| entity.id.clone())
                .collect::<Vec<_>>(),
        ),
        (
            GeometryRegionRole::Edge,
            topology
                .edges
                .iter()
                .map(|entity| entity.id.clone())
                .collect::<Vec<_>>(),
        ),
        (
            GeometryRegionRole::Face,
            topology
                .faces
                .iter()
                .map(|entity| entity.id.clone())
                .collect::<Vec<_>>(),
        ),
        (
            GeometryRegionRole::Solid,
            topology
                .solids
                .iter()
                .map(|entity| entity.id.clone())
                .collect::<Vec<_>>(),
        ),
        (
            GeometryRegionRole::Cell,
            topology
                .regions
                .iter()
                .map(|entity| entity.id.clone())
                .collect::<Vec<_>>(),
        ),
    ]);
    for entities in inventory.values_mut() {
        entities.sort();
        entities.dedup();
    }
    inventory
}

fn persistent_kind(
    kind: ffi::bridge::OcctFeatureEntityKind,
) -> Result<PersistentEntityKind, OcctExactFeatureError> {
    use ffi::bridge::OcctFeatureEntityKind as KernelKind;
    if kind == KernelKind::Vertex {
        Ok(PersistentEntityKind::Vertex)
    } else if kind == KernelKind::Edge {
        Ok(PersistentEntityKind::Edge)
    } else if kind == KernelKind::Face {
        Ok(PersistentEntityKind::Face)
    } else if kind == KernelKind::Solid {
        Ok(PersistentEntityKind::Solid)
    } else {
        Err(OcctExactFeatureError::Contract(
            "kernel history contains an unknown entity kind".into(),
        ))
    }
}

fn digest_name(digest: &[u8]) -> Result<String, OcctExactFeatureError> {
    use std::fmt::Write as _;
    if digest.len() != 32 {
        return Err(OcctExactFeatureError::Contract(
            "kernel history identity digest has an invalid length".into(),
        ));
    }
    let mut name = String::with_capacity(69);
    name.push_str("occt:");
    for byte in digest {
        write!(&mut name, "{byte:02x}").map_err(|error| {
            OcctExactFeatureError::Contract(format!(
                "kernel history identity formatting failed: {error}"
            ))
        })?;
    }
    Ok(name)
}

fn append_region_lineage(
    dependencies: &[(&str, &ImportedExactCad)],
    output: &ExactBRepTopology,
    source_to_result: &mut BTreeMap<PersistentEntityId, Vec<PersistentEntityId>>,
    deleted: &mut BTreeMap<PersistentEntityId, bool>,
    introduced: &mut BTreeSet<PersistentEntityId>,
) -> Result<(), OcctExactFeatureError> {
    let source_regions = dependencies
        .iter()
        .flat_map(|(_, geometry)| geometry.topology.regions.iter())
        .map(|region| (region.solid_id.clone(), region.id.clone()))
        .collect::<BTreeMap<_, _>>();
    let result_regions = output
        .regions
        .iter()
        .map(|region| (region.solid_id.clone(), region.id.clone()))
        .collect::<BTreeMap<_, _>>();
    let source_solids = source_to_result
        .keys()
        .filter(|entity| entity.kind == PersistentEntityKind::Solid)
        .cloned()
        .collect::<Vec<_>>();
    for solid in source_solids {
        let region = source_regions.get(&solid).ok_or_else(|| {
            OcctExactFeatureError::Contract(
                "exact dependency solid has no projected analysis region".into(),
            )
        })?;
        let target_regions = source_to_result[&solid]
            .iter()
            .filter(|target| target.kind == PersistentEntityKind::Solid)
            .map(|target| {
                result_regions.get(target).cloned().ok_or_else(|| {
                    OcctExactFeatureError::Contract(
                        "exact result solid has no projected analysis region".into(),
                    )
                })
            })
            .collect::<Result<Vec<_>, _>>()?;
        deleted.insert(region.clone(), target_regions.is_empty());
        if source_to_result
            .insert(region.clone(), target_regions)
            .is_some()
        {
            return Err(OcctExactFeatureError::Contract(
                "exact dependency region identity collides with another source entity".into(),
            ));
        }
    }
    let introduced_solids = introduced
        .iter()
        .filter(|entity| entity.kind == PersistentEntityKind::Solid)
        .cloned()
        .collect::<Vec<_>>();
    for solid in introduced_solids {
        let region = result_regions.get(&solid).ok_or_else(|| {
            OcctExactFeatureError::Contract(
                "introduced exact result solid has no projected analysis region".into(),
            )
        })?;
        introduced.insert(region.clone());
    }
    Ok(())
}

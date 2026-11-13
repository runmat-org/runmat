mod resources;
use super::*;
pub(super) use resources::*;

pub(super) fn require_selected(count: usize, maximum: usize) -> Result<(), ObservationError> {
    if count > maximum {
        return Err(ObservationError::Limit("selected topology entities"));
    }
    Ok(())
}

pub(super) fn reserved<T>(count: usize) -> Result<Vec<T>, ObservationError> {
    crate::topology::storage::reserve(count).map_err(storage_error)
}

pub(super) fn selected_indices(
    indices: impl Iterator<Item = usize> + Clone,
    maximum: usize,
) -> Result<Vec<usize>, ObservationError> {
    let count = indices.clone().count();
    require_selected(count, maximum)?;
    let mut values = reserved(count)?;
    values.extend(indices);
    Ok(values)
}

/// Parent-face/element counts are not a bound on the deduplicated nodal selection. Materialize
/// only previously unseen indices, and reject before inserting the first over-limit index.
pub(super) fn unique_nodes(
    indices: impl Iterator<Item = usize>,
    domain: usize,
    limits: ObservationLimits,
) -> Result<Vec<usize>, ObservationError> {
    use crate::topology::storage::{DenseIndexUnion, DenseUnionError};
    require_workspace(
        node_union_resident_bytes(domain, limits.maximum_selected_entities)?,
        limits,
    )?;
    DenseIndexUnion::new(domain, limits.maximum_selected_entities)
        .map_err(storage_error)?
        .collect(indices)
        .map_err(|error| match error {
            DenseUnionError::Storage(error) => storage_error(error),
            DenseUnionError::Domain => {
                ObservationError::Identity("resolved node target exceeds structural topology")
            }
            DenseUnionError::Limit => ObservationError::Limit("selected topology entities"),
        })
}

pub(super) fn nodes_target(
    indices: Vec<usize>,
    topology: &ObservationTopology,
    maximum: usize,
) -> Result<ResolvedObservationTarget, ObservationError> {
    require_selected(indices.len(), maximum)?;
    let mut identities = reserved(indices.len())?;
    for &index in &indices {
        identities.push(topology.nodes.get(index).map(|node| node.identity).ok_or(
            ObservationError::Identity("resolved node target exceeds structural topology"),
        )?);
    }
    Ok(ResolvedObservationTarget::Nodes {
        indices,
        identities,
    })
}

pub(super) fn elements_target(
    indices: Vec<usize>,
    domain: RecoveredFieldEntityDomain,
    topology: &ObservationTopology,
    maximum: usize,
) -> Result<ResolvedObservationTarget, ObservationError> {
    require_selected(indices.len(), maximum)?;
    let elements = topology.elements(domain)?;
    let mut identities = reserved(indices.len())?;
    for &index in &indices {
        identities.push(elements.get(index).map(|element| element.identity).ok_or(
            ObservationError::Identity("resolved element target exceeds structural topology"),
        )?);
    }
    Ok(ResolvedObservationTarget::Elements {
        domain,
        indices,
        identities,
    })
}

pub(super) fn matching_elements<'a>(
    target: &'a runmat_analysis_core::RegionTarget,
    domain: RecoveredFieldEntityDomain,
    topology: &'a ObservationTopology,
    limits: ObservationLimits,
    target_bytes: u64,
) -> Result<impl Iterator<Item = usize> + Clone + 'a, ObservationError> {
    let elements = topology.elements(domain)?;
    let projection = if domain == RecoveredFieldEntityDomain::VolumeElements {
        topology.solid_projection.as_deref()
    } else {
        None
    };
    require_workspace(
        element_selection_resident_bytes(target, topology, projection, target_bytes)?,
        limits,
    )?;
    if let Some(projection) = projection {
        if topology.finite_volume.is_some() || !target.resolved_entities().is_empty() {
            validate_exact_coverage(
                target,
                projection.elements.iter().flat_map(|element| {
                    std::iter::once(&element.region_id).chain(element.provenance.iter())
                }),
            )?;
        }
    }
    let source_indices =
        if let (Some(layout), Some(projection)) = (&topology.finite_volume, projection) {
            let by_identity = crate::topology::storage::SortedValues::derive(
                projection
                    .elements
                    .iter()
                    .enumerate()
                    .map(|(i, e)| (e.stable_identity, i)),
                projection.elements.len(),
            )
            .map_err(storage_error)?;
            let mut indices = reserved(layout.cell_identities().len())?;
            for identity in layout.cell_identities() {
                indices.push(*by_identity.get(identity).ok_or(ObservationError::Identity(
                    "native cell identity absent from source projection",
                ))?);
            }
            Some(std::sync::Arc::new(indices))
        } else {
            None
        };
    let count = elements.len();
    Ok((0..count).filter(move |&index| match projection {
        Some(projection) => {
            let source_index = source_indices
                .as_ref()
                .map_or(index, |indices| indices[index]);
            let element = &projection.elements[source_index];
            target.selects_volume_element(&element.region_id, &element.provenance)
        }
        None => elements[index].region_id == target.region_id,
    }))
}

/// Exact native selections may not silently degrade to a surviving subset.
pub(super) fn validate_exact_coverage<'a>(
    target: &runmat_analysis_core::RegionTarget,
    original: impl Iterator<Item = &'a runmat_meshing_core::PersistentEntityId>,
) -> Result<(), ObservationError> {
    target
        .validate_resolution()
        .map_err(|e| ObservationError::Invalid(e.to_string()))?;
    if target.resolved_entities().is_empty() {
        return Err(ObservationError::Invalid(
            "native observation requires exact resolved entities".into(),
        ));
    }
    let mut covered = reserved(target.resolved_entities().len())?;
    covered.resize(target.resolved_entities().len(), false);
    for entity in original {
        if let Ok(i) = target.resolved_entities().binary_search(entity) {
            covered[i] = true;
        }
    }
    if covered.into_iter().any(|seen| !seen) {
        return Err(ObservationError::Identity(
            "native observation target contains an absent original entity",
        ));
    }
    Ok(())
}

pub(super) fn matching_faces<'a>(
    target: &'a runmat_analysis_core::RegionTarget,
    projection: &'a AnalysisProjection,
) -> impl Iterator<Item = usize> + Clone + 'a {
    projection
        .boundary_faces
        .iter()
        .enumerate()
        .filter_map(move |(index, face)| {
            face.provenance
                .iter()
                .any(|entity| target.matches_persistent_entity(entity))
                .then_some(index)
        })
}

pub(super) fn solid_projection<'a>(
    topology: &'a ObservationTopology,
    spec: &ObservationSpec,
) -> Result<&'a AnalysisProjection, ObservationError> {
    topology.solid_projection.as_deref().ok_or_else(|| {
        target_error(
            spec,
            "volume and boundary observations require admitted solid topology",
        )
    })
}

pub(super) fn parse_node_selector(
    selector: &str,
    spec: &ObservationSpec,
) -> Result<u64, ObservationError> {
    selector
        .strip_prefix("node:")
        .and_then(|value| value.parse().ok())
        .ok_or_else(|| target_error(spec, "mesh-node targets require a `node:<id>` selector"))
}

pub(super) fn enforce_cardinality(
    cardinality: TargetCardinality,
    count: usize,
    spec: &ObservationSpec,
) -> Result<(), ObservationError> {
    if count == 0 {
        Err(target_error(
            spec,
            "region resolves to no admitted topology",
        ))
    } else if cardinality == TargetCardinality::One && count != 1 {
        Err(target_error(
            spec,
            "region cardinality requires exactly one matched entity",
        ))
    } else {
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use runmat_analysis_core::{EntityDimension, ExpectedTopology, RegionTarget};
    use runmat_meshing_core::{
        fixtures::canonical_tetrahedron_solver_mesh, ElementOrder, PersistentEntityId,
        PersistentEntityKind, StableDigest,
    };

    #[test]
    fn solid_observation_uses_original_cad_provenance_and_rejects_partial_coverage() {
        let solid = PersistentEntityId::new(
            PersistentEntityKind::Solid,
            "cad-body".into(),
            vec!["part".into()],
        );
        let mut mesh = canonical_tetrahedron_solver_mesh(ElementOrder::Tet4);
        let region = mesh.topology.volume_elements[0].region_id.clone();
        assert_ne!(region, solid);
        for element in &mut mesh.topology.volume_elements {
            element.provenance.push(solid.clone());
            element.provenance.sort();
        }
        mesh.canonical_digest = StableDigest::ZERO;
        mesh.seal_canonical_digest().unwrap();
        let projection = AnalysisProjection::derive(&mesh).unwrap();
        assert_eq!(projection.elements[0].region_id, region);
        assert!(projection.elements[0].provenance.contains(&solid));
        let topology = ObservationTopology::from_projection(&projection).unwrap();
        let mut target = RegionTarget::new(
            "body",
            EntityDimension::Volume,
            ExpectedTopology::MeshElement,
            TargetCardinality::OneOrMore,
        );
        target.bind_resolved_entities(vec![solid.clone()]).unwrap();
        assert_eq!(
            matching_elements(
                &target,
                RecoveredFieldEntityDomain::VolumeElements,
                &topology,
                ObservationLimits::default(),
                0,
            )
            .unwrap()
            .collect::<Vec<_>>(),
            (0..projection.elements.len()).collect::<Vec<_>>()
        );
        let missing = {
            let __base = &(solid.clone());
            PersistentEntityId::new(
                __base.kind,
                "absent-body".into(),
                __base.assembly_path.clone(),
            )
        };
        target.bind_resolved_entities(vec![solid, missing]).unwrap();
        assert!(matches!(
            matching_elements(
                &target,
                RecoveredFieldEntityDomain::VolumeElements,
                &topology,
                ObservationLimits::default(),
                0,
            ),
            Err(ObservationError::Identity(
                "native observation target contains an absent original entity"
            ))
        ));
    }
}

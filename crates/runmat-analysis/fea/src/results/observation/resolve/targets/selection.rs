use super::*;

pub(super) fn require_selected(count: usize, maximum: usize) -> Result<(), ObservationError> {
    if count > maximum {
        return Err(ObservationError::Limit("selected topology entities"));
    }
    Ok(())
}

pub(super) fn reserved<T>(count: usize) -> Result<Vec<T>, ObservationError> {
    let mut values = Vec::new();
    values
        .try_reserve_exact(count)
        .map_err(|_| ObservationError::Limit("observation target allocation"))?;
    Ok(values)
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
    maximum: usize,
) -> Result<Vec<usize>, ObservationError> {
    let mut unique = std::collections::HashSet::new();
    for index in indices {
        if !unique.contains(&index) {
            if unique.len() >= maximum {
                return Err(ObservationError::Limit("selected topology entities"));
            }
            unique
                .try_reserve(1)
                .map_err(|_| ObservationError::Limit("observation target allocation"))?;
            unique.insert(index);
        }
    }
    let mut values = reserved(unique.len())?;
    values.extend(unique);
    values.sort_unstable();
    Ok(values)
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
) -> Result<impl Iterator<Item = usize> + Clone + 'a, ObservationError> {
    let elements = topology.elements(domain)?;
    let projection = if domain == RecoveredFieldEntityDomain::VolumeElements {
        topology.solid_projection.as_ref()
    } else {
        None
    };
    let source_indices =
        if let (Some(layout), Some(projection)) = (&topology.finite_volume, projection) {
            validate_exact_coverage(target, projection.elements.iter().map(|e| &e.region_id))?;
            let by_identity = projection
                .elements
                .iter()
                .enumerate()
                .map(|(i, e)| (e.stable_identity, i))
                .collect::<std::collections::BTreeMap<_, _>>();
            let indices = layout
                .cell_identities()
                .iter()
                .map(|identity| {
                    by_identity
                        .get(identity)
                        .copied()
                        .ok_or(ObservationError::Identity(
                            "native cell identity absent from source projection",
                        ))
                })
                .collect::<Result<Vec<_>, _>>()?;
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
            target.matches_persistent_entity(&projection.elements[source_index].region_id)
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
    let mut covered = vec![false; target.resolved_entities().len()];
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
    topology.solid_projection.as_ref().ok_or_else(|| {
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

//! Exact local inverse of admitted adaptive CDT insertions.

use std::collections::{BTreeMap, BTreeSet};

use runmat_meshing_core::{MeshingCancellationSignal, StableDigest};

use super::{
    adaptive_lineage::{tetrahedron_record_key, tetrahedron_records},
    adaptive_refinement::{decision_mark, validate_marked_delaunay_volume_refinement},
    evaluate_delaunay_volume_quality,
    insertion::validate_constrained_delaunay_volume_topology,
    topology::build_delaunay_volume_topology_with_regions,
    validate_delaunay_volume_provenance, DelaunayAdaptiveInsertionLineage,
    DelaunayAdaptiveRefinementDecision, DelaunayAdaptiveRefinementError,
    DelaunayAdaptiveRefinementErrorKind, DelaunayAdaptiveRefinementOptions,
    DelaunayAdaptiveRefinementResult, DelaunayInsertionError, DelaunayInsertionErrorKind,
    DelaunayTopologyError, DelaunayTopologyErrorKind, DelaunayVolumeQuality,
    DelaunayVolumeQualityError, DelaunayVolumeQualityErrorKind, DelaunayVolumeRefinementInput,
    DelaunayVolumeTopology,
};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DelaunayAdaptiveCoarseningOptions {
    pub refinement: DelaunayAdaptiveRefinementOptions,
    pub maximum_removals: u64,
    pub cancellation_check_interval: u64,
}

impl Default for DelaunayAdaptiveCoarseningOptions {
    fn default() -> Self {
        Self {
            refinement: DelaunayAdaptiveRefinementOptions::default(),
            maximum_removals: 10_000_000,
            cancellation_check_interval: 1_024,
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct DelaunayAdaptiveCoarseningResult {
    pub topology: DelaunayVolumeTopology,
    pub quality: DelaunayVolumeQuality,
    /// Removed nodes in reverse insertion order, which is the canonical inverse order.
    pub removed_node_identities: Vec<StableDigest>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DelaunayAdaptiveCoarseningErrorKind {
    InvalidOptions,
    InvalidInput,
    InvalidRemovals,
    DependencyConflict,
    InvalidResult,
    ResourceLimit,
    Cancelled,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DelaunayAdaptiveCoarseningError {
    pub kind: DelaunayAdaptiveCoarseningErrorKind,
    pub reason: String,
}

impl std::fmt::Display for DelaunayAdaptiveCoarseningError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            formatter,
            "marked Delaunay coarsening {:?}: {}",
            self.kind, self.reason
        )
    }
}

impl std::error::Error for DelaunayAdaptiveCoarseningError {}

pub fn coarsen_marked_delaunay_volume(
    original: DelaunayVolumeRefinementInput<'_>,
    refinement: &DelaunayAdaptiveRefinementResult,
    removal_node_identities: &[StableDigest],
    options: DelaunayAdaptiveCoarseningOptions,
    cancellation: &dyn MeshingCancellationSignal,
) -> Result<DelaunayAdaptiveCoarseningResult, DelaunayAdaptiveCoarseningError> {
    let removals = validate_input(
        original,
        refinement,
        removal_node_identities,
        options,
        cancellation,
    )?;
    let result = apply_removals(original, refinement, &removals, options, cancellation)?;
    validate_marked_delaunay_volume_coarsening(
        original,
        refinement,
        removal_node_identities,
        &result,
        options,
        cancellation,
    )?;
    Ok(result)
}

pub fn validate_marked_delaunay_volume_coarsening(
    original: DelaunayVolumeRefinementInput<'_>,
    refinement: &DelaunayAdaptiveRefinementResult,
    removal_node_identities: &[StableDigest],
    result: &DelaunayAdaptiveCoarseningResult,
    options: DelaunayAdaptiveCoarseningOptions,
    cancellation: &dyn MeshingCancellationSignal,
) -> Result<(), DelaunayAdaptiveCoarseningError> {
    let removals = validate_input(
        original,
        refinement,
        removal_node_identities,
        options,
        cancellation,
    )?;
    let replay = apply_removals(original, refinement, &removals, options, cancellation)?;
    if replay != *result {
        return Err(error(
            DelaunayAdaptiveCoarseningErrorKind::InvalidResult,
            "coarsened topology, quality, or reverse mutation lineage does not match replay",
        ));
    }
    Ok(())
}

fn validate_input<'a>(
    original: DelaunayVolumeRefinementInput<'_>,
    refinement: &'a DelaunayAdaptiveRefinementResult,
    removal_node_identities: &[StableDigest],
    options: DelaunayAdaptiveCoarseningOptions,
    cancellation: &dyn MeshingCancellationSignal,
) -> Result<Vec<&'a DelaunayAdaptiveInsertionLineage>, DelaunayAdaptiveCoarseningError> {
    if options.maximum_removals == 0 || options.cancellation_check_interval == 0 {
        return Err(error(
            DelaunayAdaptiveCoarseningErrorKind::InvalidOptions,
            "coarsening removal limit and cancellation interval must be nonzero",
        ));
    }
    if removal_node_identities.len() as u64 > options.maximum_removals {
        return Err(resource(format!(
            "adaptive removal inventory {} exceeds its hard limit {}",
            removal_node_identities.len(),
            options.maximum_removals
        )));
    }
    let marks = refinement
        .decisions
        .iter()
        .map(decision_mark)
        .collect::<Vec<_>>();
    validate_marked_delaunay_volume_refinement(
        original,
        &marks,
        refinement,
        options.refinement,
        cancellation,
    )
    .map_err(refinement_error)?;
    let requested = removal_node_identities
        .iter()
        .copied()
        .collect::<BTreeSet<_>>();
    if requested.len() != removal_node_identities.len() || requested.contains(&StableDigest::ZERO) {
        return Err(invalid_removals());
    }
    let mut admitted = BTreeSet::new();
    let mut removals = Vec::with_capacity(requested.len());
    for decision in refinement.decisions.iter().rev() {
        let DelaunayAdaptiveRefinementDecision::Inserted { lineage, .. } = decision else {
            continue;
        };
        if requested.contains(&lineage.node.identity) {
            admitted.insert(lineage.node.identity);
            removals.push(lineage);
        }
    }
    if admitted != requested {
        return Err(invalid_removals());
    }
    Ok(removals)
}

fn apply_removals(
    original: DelaunayVolumeRefinementInput<'_>,
    refinement: &DelaunayAdaptiveRefinementResult,
    removals: &[&DelaunayAdaptiveInsertionLineage],
    options: DelaunayAdaptiveCoarseningOptions,
    cancellation: &dyn MeshingCancellationSignal,
) -> Result<DelaunayAdaptiveCoarseningResult, DelaunayAdaptiveCoarseningError> {
    let mut records = tetrahedron_records(&refinement.topology);
    let mut node_use_counts = BTreeMap::<StableDigest, u64>::new();
    for record in records.values() {
        for identity in record.node_identities {
            *node_use_counts.entry(identity).or_default() += 1;
        }
    }
    let mut removed_node_identities = Vec::with_capacity(removals.len());
    for (index, lineage) in removals.iter().enumerate() {
        checkpoint(index as u64, options, cancellation)?;
        for created in &lineage.created_tetrahedra {
            let key = tetrahedron_record_key(created.node_identities);
            if records.get(&key) != Some(created) {
                return Err(dependency(
                    "a requested insertion is not a removable leaf of the retained mutation DAG",
                ));
            }
        }
        for restored in &lineage.removed_tetrahedra {
            if records.contains_key(&tetrahedron_record_key(restored.node_identities)) {
                return Err(dependency(
                    "inverse insertion would duplicate an already retained tetrahedron",
                ));
            }
        }
        for created in &lineage.created_tetrahedra {
            records.remove(&tetrahedron_record_key(created.node_identities));
            update_node_uses(&mut node_use_counts, created.node_identities, false)?;
        }
        if node_use_counts
            .get(&lineage.node.identity)
            .copied()
            .unwrap_or(0)
            != 0
        {
            return Err(dependency(
                "a retained descendant tetrahedron still references the requested node",
            ));
        }
        for restored in &lineage.removed_tetrahedra {
            records.insert(
                tetrahedron_record_key(restored.node_identities),
                restored.clone(),
            );
            update_node_uses(&mut node_use_counts, restored.node_identities, true)?;
        }
        removed_node_identities.push(lineage.node.identity);
    }

    let removed = removed_node_identities
        .iter()
        .copied()
        .collect::<BTreeSet<_>>();
    let nodes = refinement
        .topology
        .nodes
        .iter()
        .copied()
        .filter(|node| !removed.contains(&node.identity))
        .collect::<Vec<_>>();
    let node_indices = nodes
        .iter()
        .enumerate()
        .map(|(index, node)| (node.identity, index as u32))
        .collect::<BTreeMap<_, _>>();
    let tetrahedra = records
        .values()
        .map(|record| {
            let index = |identity| {
                node_indices
                    .get(&identity)
                    .copied()
                    .ok_or_else(|| dependency("inverse insertion references a removed node"))
            };
            let [a, b, c, d] = record.node_identities;
            Ok((
                [index(a)?, index(b)?, index(c)?, index(d)?],
                record.region_id.clone(),
            ))
        })
        .collect::<Result<Vec<_>, DelaunayAdaptiveCoarseningError>>()?;
    let topology = build_delaunay_volume_topology_with_regions(
        nodes,
        tetrahedra,
        options.refinement.insertion.topology,
        cancellation,
    )
    .map_err(topology_error)?;
    let protected_faces = original
        .provenance
        .facets
        .iter()
        .map(|facet| facet.node_identities)
        .collect::<Vec<_>>();
    validate_constrained_delaunay_volume_topology(
        &topology,
        &protected_faces,
        options.refinement.insertion,
        cancellation,
    )
    .map_err(insertion_error)?;
    validate_delaunay_volume_provenance(
        &topology,
        original.provenance,
        original.quality_options.provenance,
        cancellation,
    )
    .map_err(|failure| {
        error(
            DelaunayAdaptiveCoarseningErrorKind::InvalidResult,
            failure.to_string(),
        )
    })?;
    let quality = evaluate_delaunay_volume_quality(
        &topology,
        original.metric_request,
        original.provenance,
        original.quality_options,
        cancellation,
    )
    .map_err(quality_error)?;
    Ok(DelaunayAdaptiveCoarseningResult {
        topology,
        quality,
        removed_node_identities,
    })
}

fn update_node_uses(
    counts: &mut BTreeMap<StableDigest, u64>,
    identities: [StableDigest; 4],
    add: bool,
) -> Result<(), DelaunayAdaptiveCoarseningError> {
    for identity in identities {
        let count = counts.entry(identity).or_default();
        if add {
            *count = count.checked_add(1).ok_or_else(|| {
                resource("adaptive node-incidence accounting exceeded its hard integer bound")
            })?;
        } else {
            *count = count.checked_sub(1).ok_or_else(|| {
                error(
                    DelaunayAdaptiveCoarseningErrorKind::InvalidInput,
                    "adaptive insertion lineage has inconsistent node incidence",
                )
            })?;
        }
    }
    Ok(())
}

fn checkpoint(
    work: u64,
    options: DelaunayAdaptiveCoarseningOptions,
    cancellation: &dyn MeshingCancellationSignal,
) -> Result<(), DelaunayAdaptiveCoarseningError> {
    if work.is_multiple_of(options.cancellation_check_interval) && cancellation.is_cancelled() {
        return Err(error(
            DelaunayAdaptiveCoarseningErrorKind::Cancelled,
            "cancelled",
        ));
    }
    Ok(())
}

fn topology_error(failure: DelaunayTopologyError) -> DelaunayAdaptiveCoarseningError {
    let kind = match failure.kind {
        DelaunayTopologyErrorKind::InvalidOptions => {
            DelaunayAdaptiveCoarseningErrorKind::InvalidOptions
        }
        DelaunayTopologyErrorKind::ResourceLimit => {
            DelaunayAdaptiveCoarseningErrorKind::ResourceLimit
        }
        DelaunayTopologyErrorKind::Cancelled => DelaunayAdaptiveCoarseningErrorKind::Cancelled,
        _ => DelaunayAdaptiveCoarseningErrorKind::InvalidResult,
    };
    error(kind, failure.to_string())
}

fn refinement_error(failure: DelaunayAdaptiveRefinementError) -> DelaunayAdaptiveCoarseningError {
    let kind = match failure.kind {
        DelaunayAdaptiveRefinementErrorKind::InvalidOptions => {
            DelaunayAdaptiveCoarseningErrorKind::InvalidOptions
        }
        DelaunayAdaptiveRefinementErrorKind::ResourceLimit => {
            DelaunayAdaptiveCoarseningErrorKind::ResourceLimit
        }
        DelaunayAdaptiveRefinementErrorKind::Cancelled => {
            DelaunayAdaptiveCoarseningErrorKind::Cancelled
        }
        _ => DelaunayAdaptiveCoarseningErrorKind::InvalidInput,
    };
    error(kind, failure.to_string())
}

fn insertion_error(failure: DelaunayInsertionError) -> DelaunayAdaptiveCoarseningError {
    let kind = match failure.kind {
        DelaunayInsertionErrorKind::InvalidOptions => {
            DelaunayAdaptiveCoarseningErrorKind::InvalidOptions
        }
        DelaunayInsertionErrorKind::ResourceLimit => {
            DelaunayAdaptiveCoarseningErrorKind::ResourceLimit
        }
        DelaunayInsertionErrorKind::Cancelled => DelaunayAdaptiveCoarseningErrorKind::Cancelled,
        _ => DelaunayAdaptiveCoarseningErrorKind::InvalidResult,
    };
    error(kind, failure.to_string())
}

fn quality_error(failure: DelaunayVolumeQualityError) -> DelaunayAdaptiveCoarseningError {
    let kind = match failure.kind {
        DelaunayVolumeQualityErrorKind::InvalidOptions => {
            DelaunayAdaptiveCoarseningErrorKind::InvalidOptions
        }
        DelaunayVolumeQualityErrorKind::ResourceLimit => {
            DelaunayAdaptiveCoarseningErrorKind::ResourceLimit
        }
        DelaunayVolumeQualityErrorKind::Cancelled => DelaunayAdaptiveCoarseningErrorKind::Cancelled,
        _ => DelaunayAdaptiveCoarseningErrorKind::InvalidResult,
    };
    error(kind, failure.to_string())
}

fn invalid_removals() -> DelaunayAdaptiveCoarseningError {
    error(
        DelaunayAdaptiveCoarseningErrorKind::InvalidRemovals,
        "removals must be unique nonzero nodes from admitted insertion lineage",
    )
}

fn dependency(reason: impl Into<String>) -> DelaunayAdaptiveCoarseningError {
    error(
        DelaunayAdaptiveCoarseningErrorKind::DependencyConflict,
        reason,
    )
}

fn resource(reason: impl Into<String>) -> DelaunayAdaptiveCoarseningError {
    error(DelaunayAdaptiveCoarseningErrorKind::ResourceLimit, reason)
}

fn error(
    kind: DelaunayAdaptiveCoarseningErrorKind,
    reason: impl Into<String>,
) -> DelaunayAdaptiveCoarseningError {
    DelaunayAdaptiveCoarseningError {
        kind,
        reason: reason.into(),
    }
}

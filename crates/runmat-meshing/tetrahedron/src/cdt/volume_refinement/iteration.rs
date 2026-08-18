use std::collections::BTreeSet;

use runmat_meshing_core::{MeshingCancellationSignal, StableDigest};

use super::{
    insert_delaunay_volume_refinement_candidate,
    insertion::{error, insertion_error, quality_error},
    select_delaunay_volume_refinement_candidate, DelaunayVolumeRefinement,
    DelaunayVolumeRefinementInput, DelaunayVolumeRefinementOptions,
    DelaunayVolumeRefinementStepError, DelaunayVolumeRefinementStepErrorKind,
};
use crate::cdt::{
    insertion::validate_constrained_delaunay_volume_topology, validate_delaunay_volume_quality,
};

pub fn refine_delaunay_volume(
    input: DelaunayVolumeRefinementInput<'_>,
    options: DelaunayVolumeRefinementOptions,
    cancellation: &dyn MeshingCancellationSignal,
) -> Result<DelaunayVolumeRefinement, DelaunayVolumeRefinementStepError> {
    validate_options(options)?;
    let mut topology = input.topology.clone();
    let mut quality = input.quality.clone();
    let mut inserted_node_identities = Vec::new();
    for _ in 0..options.maximum_insertions {
        let current = DelaunayVolumeRefinementInput {
            topology: &topology,
            metric_request: input.metric_request,
            provenance: input.provenance,
            quality: &quality,
            quality_options: input.quality_options,
        };
        let Some(candidate) = select_delaunay_volume_refinement_candidate(
            current,
            options.step.candidate,
            cancellation,
        )
        .map_err(super::insertion::candidate_error)?
        else {
            inserted_node_identities.sort_unstable();
            let refinement = DelaunayVolumeRefinement {
                topology,
                quality,
                inserted_node_identities,
            };
            validate_delaunay_volume_refinement(input, &refinement, options, cancellation)?;
            return Ok(refinement);
        };
        let step = insert_delaunay_volume_refinement_candidate(
            current,
            &candidate,
            options.step,
            cancellation,
        )?;
        inserted_node_identities.push(candidate.node.identity);
        topology = step.topology;
        quality = step.quality;
    }
    if quality.worst_refinement_tetrahedron.is_none() {
        inserted_node_identities.sort_unstable();
        let refinement = DelaunayVolumeRefinement {
            topology,
            quality,
            inserted_node_identities,
        };
        validate_delaunay_volume_refinement(input, &refinement, options, cancellation)?;
        return Ok(refinement);
    }
    let last_violation = quality
        .tetrahedra
        .iter()
        .filter(|tetrahedron| tetrahedron.requires_refinement())
        .map(|tetrahedron| tetrahedron.refinement_violation_ratio)
        .max_by(f64::total_cmp)
        .unwrap_or(1.0);
    Err(error(
        DelaunayVolumeRefinementStepErrorKind::ResourceLimit,
        format!(
            "volume refinement exhausted {} insertions with worst violation ratio {last_violation}",
            options.maximum_insertions
        ),
    ))
}

pub fn validate_delaunay_volume_refinement(
    input: DelaunayVolumeRefinementInput<'_>,
    refinement: &DelaunayVolumeRefinement,
    options: DelaunayVolumeRefinementOptions,
    cancellation: &dyn MeshingCancellationSignal,
) -> Result<(), DelaunayVolumeRefinementStepError> {
    validate_options(options)?;
    if refinement.inserted_node_identities.len() as u64 > options.maximum_insertions
        || refinement
            .inserted_node_identities
            .contains(&StableDigest::ZERO)
        || refinement
            .inserted_node_identities
            .windows(2)
            .any(|pair| pair[0] >= pair[1])
    {
        return Err(error(
            DelaunayVolumeRefinementStepErrorKind::InvalidTopology,
            "inserted node identities must be bounded, nonzero, unique, and canonical",
        ));
    }
    if refinement.quality.worst_refinement_tetrahedron.is_some() {
        return Err(error(
            DelaunayVolumeRefinementStepErrorKind::InvalidQuality,
            "completed volume refinement must satisfy every configured quality bound",
        ));
    }
    if input.quality.worst_refinement_tetrahedron.is_none()
        && (refinement.topology != *input.topology
            || refinement.quality != *input.quality
            || !refinement.inserted_node_identities.is_empty())
    {
        return Err(error(
            DelaunayVolumeRefinementStepErrorKind::InvalidInput,
            "already converged input must remain unchanged",
        ));
    }
    let inserted = refinement
        .inserted_node_identities
        .iter()
        .copied()
        .collect::<BTreeSet<_>>();
    if refinement.topology.nodes.len()
        != input.topology.nodes.len() + refinement.inserted_node_identities.len()
        || input.topology.nodes.iter().any(|old| {
            refinement
                .topology
                .nodes
                .binary_search_by_key(&old.identity, |node| node.identity)
                .ok()
                .and_then(|index| refinement.topology.nodes.get(index))
                != Some(old)
        })
        || refinement.topology.nodes.iter().any(|node| {
            input
                .topology
                .nodes
                .binary_search_by_key(&node.identity, |old| old.identity)
                .is_err()
                && !inserted.contains(&node.identity)
        })
    {
        return Err(error(
            DelaunayVolumeRefinementStepErrorKind::InvalidTopology,
            "refinement topology does not exactly match old nodes plus its lineage",
        ));
    }
    let protected_faces = input
        .provenance
        .facets
        .iter()
        .map(|facet| facet.node_identities)
        .collect::<Vec<_>>();
    validate_constrained_delaunay_volume_topology(
        &refinement.topology,
        &protected_faces,
        options.step.insertion,
        cancellation,
    )
    .map_err(insertion_error)?;
    validate_delaunay_volume_quality(
        &refinement.topology,
        input.metric_request,
        input.provenance,
        &refinement.quality,
        input.quality_options,
        cancellation,
    )
    .map_err(quality_error)?;
    Ok(())
}

fn validate_options(
    options: DelaunayVolumeRefinementOptions,
) -> Result<(), DelaunayVolumeRefinementStepError> {
    if options.maximum_insertions == 0 {
        return Err(error(
            DelaunayVolumeRefinementStepErrorKind::InvalidOptions,
            "maximum volume-refinement insertions must be nonzero",
        ));
    }
    Ok(())
}

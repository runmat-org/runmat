use std::collections::BTreeMap;

use runmat_meshing_core::{MeshingCancellationSignal, StableDigest};

use super::{
    insert_delaunay_volume_refinement_candidate,
    insertion::{error, insertion_error, quality_error},
    select_delaunay_volume_refinement_candidate, DelaunayVolumeRefinement,
    DelaunayVolumeRefinementInput, DelaunayVolumeRefinementMutation,
    DelaunayVolumeRefinementOptions, DelaunayVolumeRefinementStepError,
    DelaunayVolumeRefinementStepErrorKind,
};
use crate::cdt::{
    insertion::validate_constrained_delaunay_volume_topology, treat_delaunay_volume_slivers,
    validate_delaunay_volume_quality, DelaunayVolumeSliverError, DelaunayVolumeSliverErrorKind,
};

pub fn refine_delaunay_volume(
    input: DelaunayVolumeRefinementInput<'_>,
    options: DelaunayVolumeRefinementOptions,
    cancellation: &dyn MeshingCancellationSignal,
) -> Result<DelaunayVolumeRefinement, DelaunayVolumeRefinementStepError> {
    validate_options(options)?;
    let mut topology = input.topology.clone();
    let mut quality = input.quality.clone();
    let mut mutations = Vec::new();
    let mut insertion_count = 0u64;
    loop {
        let current = DelaunayVolumeRefinementInput {
            topology: &topology,
            metric_request: input.metric_request,
            provenance: input.provenance,
            quality: &quality,
            quality_options: input.quality_options,
        };
        if quality.tetrahedra.iter().any(|tetrahedron| {
            tetrahedron.metric_scaled_jacobian
                < input.quality_options.minimum_metric_scaled_jacobian
        }) {
            match treat_delaunay_volume_slivers(current, options.sliver, cancellation) {
                Ok(treatment) => {
                    mutations.extend(
                        treatment
                            .relocations
                            .into_iter()
                            .map(DelaunayVolumeRefinementMutation::Relocated),
                    );
                    topology = treatment.topology;
                    quality = treatment.quality;
                    continue;
                }
                Err(failure)
                    if failure.kind == DelaunayVolumeSliverErrorKind::NoAdmissibleRelocation => {}
                Err(failure) => return Err(sliver_error(failure)),
            }
        }
        let Some(candidate) = select_delaunay_volume_refinement_candidate(
            current,
            options.step.candidate,
            cancellation,
        )
        .map_err(super::insertion::candidate_error)?
        else {
            let refinement = DelaunayVolumeRefinement {
                topology,
                quality,
                mutations,
            };
            validate_delaunay_volume_refinement(input, &refinement, options, cancellation)?;
            return Ok(refinement);
        };
        if insertion_count >= options.maximum_insertions {
            break;
        }
        let step = insert_delaunay_volume_refinement_candidate(
            current,
            &candidate,
            options.step,
            cancellation,
        )?;
        insertion_count += 1;
        mutations.push(DelaunayVolumeRefinementMutation::Inserted(candidate.node));
        topology = step.topology;
        quality = step.quality;
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
    if refinement.quality.worst_refinement_tetrahedron.is_some() {
        return Err(error(
            DelaunayVolumeRefinementStepErrorKind::InvalidQuality,
            "completed volume refinement must satisfy every configured quality bound",
        ));
    }
    if input.quality.worst_refinement_tetrahedron.is_none()
        && (refinement.topology != *input.topology
            || refinement.quality != *input.quality
            || !refinement.mutations.is_empty())
    {
        return Err(error(
            DelaunayVolumeRefinementStepErrorKind::InvalidInput,
            "already converged input must remain unchanged",
        ));
    }
    validate_mutation_lineage(input, refinement, options)?;
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
    super::super::insertion::validate_options(options.step.insertion).map_err(insertion_error)?;
    super::super::volume_sliver::validate_options(options.sliver).map_err(sliver_error)?;
    Ok(())
}

fn validate_mutation_lineage(
    input: DelaunayVolumeRefinementInput<'_>,
    refinement: &DelaunayVolumeRefinement,
    options: DelaunayVolumeRefinementOptions,
) -> Result<(), DelaunayVolumeRefinementStepError> {
    let insertion_count = refinement
        .mutations
        .iter()
        .filter(|mutation| matches!(mutation, DelaunayVolumeRefinementMutation::Inserted(_)))
        .count() as u64;
    let relocation_count = refinement.mutations.len() as u64 - insertion_count;
    let maximum_relocations = insertion_count
        .saturating_add(1)
        .saturating_mul(options.sliver.maximum_passes);
    if insertion_count > options.maximum_insertions || relocation_count > maximum_relocations {
        return Err(error(
            DelaunayVolumeRefinementStepErrorKind::InvalidTopology,
            "refinement mutation evidence exceeds its configured hard limits",
        ));
    }
    let mut nodes = input
        .topology
        .nodes
        .iter()
        .map(|node| (node.identity, *node))
        .collect::<BTreeMap<_, _>>();
    for mutation in &refinement.mutations {
        match mutation {
            DelaunayVolumeRefinementMutation::Inserted(node) => {
                if node.identity == StableDigest::ZERO
                    || node.coordinates_m.iter().any(|value| !value.is_finite())
                    || nodes.insert(node.identity, *node).is_some()
                {
                    return Err(invalid_lineage());
                }
            }
            DelaunayVolumeRefinementMutation::Relocated(relocation) => {
                if !relocation
                    .source_tetrahedron_node_identities
                    .contains(&relocation.source_node_identity)
                    || relocation
                        .source_tetrahedron_node_identities
                        .iter()
                        .any(|identity| !nodes.contains_key(identity))
                    || !super::super::volume_sliver::relocation_identity_is_valid(relocation)
                    || nodes.remove(&relocation.source_node_identity).is_none()
                    || nodes
                        .insert(
                            relocation.replacement_node.identity,
                            relocation.replacement_node,
                        )
                        .is_some()
                {
                    return Err(invalid_lineage());
                }
            }
        }
    }
    if nodes.values().copied().collect::<Vec<_>>() != refinement.topology.nodes {
        return Err(invalid_lineage());
    }
    Ok(())
}

fn invalid_lineage() -> DelaunayVolumeRefinementStepError {
    error(
        DelaunayVolumeRefinementStepErrorKind::InvalidTopology,
        "refinement topology does not exactly match its ordered insertion and relocation lineage",
    )
}

fn sliver_error(failure: DelaunayVolumeSliverError) -> DelaunayVolumeRefinementStepError {
    let kind = match failure.kind {
        DelaunayVolumeSliverErrorKind::InvalidOptions => {
            DelaunayVolumeRefinementStepErrorKind::InvalidOptions
        }
        DelaunayVolumeSliverErrorKind::InvalidInput => {
            DelaunayVolumeRefinementStepErrorKind::InvalidInput
        }
        DelaunayVolumeSliverErrorKind::InvalidTopology => {
            DelaunayVolumeRefinementStepErrorKind::InvalidTopology
        }
        DelaunayVolumeSliverErrorKind::InvalidProvenance => {
            DelaunayVolumeRefinementStepErrorKind::InvalidProvenance
        }
        DelaunayVolumeSliverErrorKind::InvalidQuality => {
            DelaunayVolumeRefinementStepErrorKind::InvalidQuality
        }
        DelaunayVolumeSliverErrorKind::NoAdmissibleRelocation => {
            DelaunayVolumeRefinementStepErrorKind::InvalidQuality
        }
        DelaunayVolumeSliverErrorKind::ResourceLimit => {
            DelaunayVolumeRefinementStepErrorKind::ResourceLimit
        }
        DelaunayVolumeSliverErrorKind::Cancelled => {
            DelaunayVolumeRefinementStepErrorKind::Cancelled
        }
    };
    error(kind, failure.to_string())
}

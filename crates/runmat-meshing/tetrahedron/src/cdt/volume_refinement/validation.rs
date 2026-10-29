use runmat_meshing_core::MeshingCancellationSignal;

use super::metric_circumcenter::metric_circumcenter;

use super::{
    candidate::{
        canonical_interior_points, metric_edge_midpoint, metric_offcenter_points,
        strictly_inside_region_from,
    },
    candidate_identity, error, quality_error, validate_options, CandidateWork,
    DelaunayRefinementCandidateKind, DelaunayVolumeRefinementCandidate,
    DelaunayVolumeRefinementCandidateError, DelaunayVolumeRefinementCandidateErrorKind,
    DelaunayVolumeRefinementCandidateOptions, DelaunayVolumeRefinementInput,
};

pub fn validate_delaunay_volume_refinement_candidate(
    input: DelaunayVolumeRefinementInput<'_>,
    candidate: &Option<DelaunayVolumeRefinementCandidate>,
    options: DelaunayVolumeRefinementCandidateOptions,
    cancellation: &dyn MeshingCancellationSignal,
) -> Result<(), DelaunayVolumeRefinementCandidateError> {
    validate_options(options)?;
    super::validate_delaunay_volume_quality(
        input.topology,
        input.metric_request,
        input.provenance,
        input.quality,
        input.quality_options,
        cancellation,
    )
    .map_err(quality_error)?;
    validate_delaunay_volume_refinement_candidate_from_admitted_input(
        input,
        candidate,
        options,
        cancellation,
    )
}

pub(super) fn validate_delaunay_volume_refinement_candidate_from_admitted_input(
    input: DelaunayVolumeRefinementInput<'_>,
    candidate: &Option<DelaunayVolumeRefinementCandidate>,
    options: DelaunayVolumeRefinementCandidateOptions,
    cancellation: &dyn MeshingCancellationSignal,
) -> Result<(), DelaunayVolumeRefinementCandidateError> {
    if input.quality.worst_refinement_tetrahedron.is_none() {
        if candidate.is_some() {
            return Err(invalid("converged quality must not carry a candidate"));
        }
        return Ok(());
    }
    let observed = candidate
        .as_ref()
        .ok_or_else(|| invalid("violating quality requires one refinement candidate"))?;
    let source_identity = observed.source_node_identities;
    let source_index = input
        .quality
        .tetrahedra
        .iter()
        .position(|tetrahedron| tetrahedron.node_identities == source_identity)
        .ok_or_else(|| invalid("candidate source is absent from quality evidence"))?;
    let source_quality = &input.quality.tetrahedra[source_index];
    if !source_quality.requires_refinement() {
        return Err(invalid("candidate source does not require refinement"));
    }
    let topology_index =
        crate::cdt::topology::queries::find_oriented_tetrahedron(input.topology, source_identity)
            .map(|(index, _)| index)
            .ok_or_else(|| invalid("candidate source is absent from topology"))?;
    let mut points = input.topology.tetrahedra[topology_index]
        .vertex_indices
        .map(|vertex| {
            input.topology.nodes[vertex as usize]
                .coordinates_m
                .canonical_value()
        });
    points.sort_by(compare_points);
    let mut work = CandidateWork::new(options, cancellation);
    work.evaluate()?;
    let (construction_index, construction) =
        crate::cdt::topology::queries::find_oriented_tetrahedron(
            input.topology,
            observed.construction_node_identities,
        )
        .ok_or_else(|| invalid("candidate construction tetrahedron is absent from topology"))?;
    if construction.region_id.as_ref() != Some(&source_quality.region_id) {
        return Err(invalid(
            "candidate construction tetrahedron is outside the source region",
        ));
    }
    let mut construction_points = construction.vertex_indices.map(|vertex| {
        input.topology.nodes[vertex as usize]
            .coordinates_m
            .canonical_value()
    });
    construction_points.sort_by(compare_points);
    work.evaluate()?;
    let expected_coordinates = match observed.kind {
        DelaunayRefinementCandidateKind::MetricCircumcenter => {
            if observed.construction_node_identities != source_identity
                || observed.construction_point_index != 0
            {
                return Err(invalid(
                    "metric circumcenter must be constructed from the quality source",
                ));
            }
            let center = metric_circumcenter(points, source_quality.resolved_metric)
                .ok_or_else(|| invalid("metric circumcenter is not representable"))?;
            if !strictly_inside_region_from(
                input.topology,
                construction_index,
                &source_quality.region_id,
                center,
                cancellation,
            )? {
                return Err(invalid(
                    "metric circumcenter is outside its admitted domain",
                ));
            }
            center
        }
        DelaunayRefinementCandidateKind::MetricOffcenter => {
            if observed.construction_node_identities != source_identity {
                return Err(invalid(
                    "metric off-center must be constructed from the quality source",
                ));
            }
            let center = metric_circumcenter(points, source_quality.resolved_metric)
                .ok_or_else(|| invalid("metric circumcenter is not representable"))?;
            metric_offcenter_points(
                input.topology,
                topology_index,
                &source_quality.region_id,
                points,
                center,
                cancellation,
            )?
            .get(observed.construction_point_index as usize)
            .copied()
            .ok_or_else(|| {
                invalid("metric off-center point index exceeds its construction family")
            })?
        }
        DelaunayRefinementCandidateKind::MetricEdgeMidpoint => {
            if observed.construction_node_identities != source_identity
                || observed.construction_point_index != 0
            {
                return Err(invalid(
                    "metric edge midpoint must be constructed from the quality source",
                ));
            }
            let point = metric_edge_midpoint(
                input.topology,
                topology_index,
                source_quality,
                input.quality_options,
            )
            .ok_or_else(|| {
                invalid("source has no size-violating edge with an interior endpoint")
            })?;
            if !strictly_inside_region_from(
                input.topology,
                topology_index,
                &source_quality.region_id,
                point,
                cancellation,
            )? {
                return Err(invalid(
                    "metric edge midpoint is outside its admitted domain",
                ));
            }
            point
        }
        DelaunayRefinementCandidateKind::InteriorPoint => {
            canonical_interior_points(construction_points)?
                .get(observed.construction_point_index as usize)
                .copied()
                .ok_or_else(|| invalid("candidate point index exceeds its construction family"))?
        }
    };
    let coordinates_consistent = coordinates_match(
        observed.node.coordinates_m.canonical_value(),
        expected_coordinates,
        construction_points,
    );
    if observed.source_node_identities != source_identity
        || observed.region_id != source_quality.region_id
        || observed.incident_metric_entity_ids != source_quality.incident_metric_entity_ids
        || observed.resolved_metric != source_quality.resolved_metric
        || observed.source_violation_ratio != source_quality.refinement_violation_ratio
        || !coordinates_consistent
        || observed.node.identity
            != candidate_identity(
                source_identity,
                observed.kind,
                observed.construction_point_index,
            )
        || input.topology.nodes.iter().any(|node| {
            node.identity == observed.node.identity
                || node.coordinates_m.canonical_value()
                    == observed.node.coordinates_m.canonical_value()
        })
    {
        return Err(invalid(format!(
            "candidate does not match independent construction: observed kind {:?}, observed coordinates {:?}, expected coordinates {expected_coordinates:?}, coordinate match {coordinates_consistent}",
            observed.kind, observed.node.coordinates_m.canonical_value()
        )));
    }
    Ok(())
}

fn compare_points(left: &[f64; 3], right: &[f64; 3]) -> std::cmp::Ordering {
    left.iter()
        .zip(right.iter())
        .find_map(|(left, right)| (left != right).then(|| left.total_cmp(right)))
        .unwrap_or(std::cmp::Ordering::Equal)
}

fn coordinates_match(left: [f64; 3], right: [f64; 3], source: [[f64; 3]; 4]) -> bool {
    left.into_iter()
        .zip(right)
        .enumerate()
        .all(|(axis, (left, right))| {
            let source_scale = source
                .iter()
                .map(|point| point[axis].abs())
                .fold(0.0_f64, f64::max);
            left.is_finite()
                && right.is_finite()
                && (left - right).abs()
                    <= 64.0
                        * f64::EPSILON
                        * left
                            .abs()
                            .max(right.abs())
                            .max(source_scale)
                            .max(f64::MIN_POSITIVE)
        })
}

fn invalid(reason: impl Into<String>) -> DelaunayVolumeRefinementCandidateError {
    error(
        DelaunayVolumeRefinementCandidateErrorKind::InvalidCandidate,
        reason,
    )
}

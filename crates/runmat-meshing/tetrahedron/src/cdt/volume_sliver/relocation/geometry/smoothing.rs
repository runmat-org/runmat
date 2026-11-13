use super::score::{PreparedStar, SmoothingObjective, StarScore};
use super::*;
pub(in crate::cdt::volume_sliver) fn smooth_vertex_star_points_with_metrics(
    topology: &(impl crate::cdt::topology::VolumeTopologyRead + ?Sized),
    source_index: usize,
    metrics: &dyn Fn(u32) -> runmat_meshing_size::metric::MetricTensor3,
    objective: SmoothingObjective,
    cancellation: &dyn MeshingCancellationSignal,
) -> Result<Vec<[f64; 3]>, DelaunayVolumeSliverError> {
    let star = &topology.ordered_vertex_star((source_index) as u32);
    // Prepare immutable coordinates, per-cell metrics and boundary incidence once.
    // Trial admission still recomputes the full grading dependency closure.
    let prepared = objective.prepare(topology, star, source_index, metrics);
    let mut link_faces = star
        .iter()
        .map(|tetrahedron| {
            let mut face = topology
                .cell(*tetrahedron)
                .vertex_indices
                .into_iter()
                .filter(|vertex| *vertex as usize != source_index)
                .collect::<Vec<_>>();
            face.sort_unstable_by_key(|vertex| topology.node(*vertex).identity);
            <[u32; 3]>::try_from(face).expect("vertex-star tetrahedron has one source vertex")
        })
        .collect::<Vec<_>>();
    link_faces.sort_unstable_by_key(|face| face.map(|vertex| topology.node(vertex).identity));
    link_faces.dedup();
    let start = topology
        .node((source_index) as u32)
        .coordinates_m
        .canonical_value();
    let initial_score = prepared.score(start)?;
    let mut results = Vec::new();
    for step in CAVITY_SMOOTHING_STEPS {
        let mut point = start;
        let mut score = initial_score.clone();
        let mut improved = false;
        for iteration in 0..MAXIMUM_CAVITY_SMOOTHING_ITERATIONS {
            if iteration.is_multiple_of(16) && cancellation.is_cancelled() {
                return Err(error(
                    DelaunayVolumeSliverErrorKind::Cancelled,
                    "vertex-star smoothing was cancelled",
                ));
            }
            let mut best = None;
            for face in link_faces.iter().take(MAXIMUM_CAVITY_SEARCH_DIRECTIONS) {
                let centroid: [f64; 3] = std::array::from_fn(|axis| {
                    face.iter()
                        .map(|vertex| topology.node(*vertex).coordinates_m.canonical_value()[axis])
                        .sum::<f64>()
                        / 3.0
                });
                let candidate =
                    std::array::from_fn(|axis| point[axis] + step * (centroid[axis] - point[axis]));
                if !point_is_in_vertex_star_kernel(topology, star, source_index, candidate)? {
                    continue;
                }
                let candidate_score = prepared.score(candidate)?;
                if candidate_score >= score {
                    continue;
                }
                if best
                    .as_ref()
                    .is_none_or(|(best_score, best_point): &(StarScore, [f64; 3])| {
                        candidate_score < *best_score
                            || candidate_score == *best_score
                                && candidate.map(f64::to_bits) < best_point.map(f64::to_bits)
                    })
                {
                    best = Some((candidate_score, candidate));
                }
            }
            let Some((best_score, best_point)) = best else {
                break;
            };
            score = best_score;
            point = best_point;
            improved = true;
        }
        if improved && !results.contains(&point) {
            results.push(point);
        }
    }
    let mut starts = Vec::with_capacity(results.len() + 1);
    starts.push(start);
    starts.extend(results.iter().copied());
    for start in starts {
        if let Some(point) = pattern_search_vertex_star_with_metrics(
            topology,
            star,
            source_index,
            start,
            &prepared,
            cancellation,
        )? {
            if !results.contains(&point) {
                results.push(point);
            }
        }
    }
    Ok(results)
}

/// Deterministic derivative-free maximin smoothing over one fixed vertex star.
///
/// The scaled-Jacobian objective is nonsmooth where the identity of the worst incident
/// tetrahedron changes, so a finite-difference gradient is unreliable. A shrinking 26-direction
/// stencil instead searches the whole local kernel while evaluating only incident tetrahedra.
/// Candidate topology construction and canonical global admission remain the caller's concern.
pub(in crate::cdt::volume_sliver) fn pattern_search_vertex_star_with_metrics(
    topology: &(impl crate::cdt::topology::VolumeTopologyRead + ?Sized),
    star: &[u32],
    source_index: usize,
    start: [f64; 3],
    prepared: &PreparedStar,
    cancellation: &dyn MeshingCancellationSignal,
) -> Result<Option<[f64; 3]>, DelaunayVolumeSliverError> {
    let source = topology
        .node((source_index) as u32)
        .coordinates_m
        .canonical_value();
    let mut neighbors = star
        .iter()
        .flat_map(|tetrahedron| topology.cell(*tetrahedron).vertex_indices)
        .filter(|vertex| *vertex as usize != source_index)
        .collect::<Vec<_>>();
    neighbors.sort_unstable();
    neighbors.dedup();
    let local_scale = neighbors
        .into_iter()
        .map(|vertex| {
            topology
                .node(vertex)
                .coordinates_m
                .canonical_value()
                .into_iter()
                .zip(source)
                .map(|(left, right)| (left - right) * (left - right))
                .sum::<f64>()
                .sqrt()
        })
        .filter(|distance| distance.is_finite() && *distance > 0.0)
        .min_by(f64::total_cmp)
        .unwrap_or(0.0);
    if local_scale == 0.0 {
        return Ok(None);
    }

    let mut point = start;
    let initial_score = prepared.score(point)?;
    let mut score = initial_score.clone();
    let mut radius = local_scale * 0.25;
    let minimum_radius = local_scale * 2_f64.powi(-24);
    for iteration in 0..MAXIMUM_CAVITY_SMOOTHING_ITERATIONS {
        if iteration.is_multiple_of(8) && cancellation.is_cancelled() {
            return Err(error(
                DelaunayVolumeSliverErrorKind::Cancelled,
                "vertex-star pattern search was cancelled",
            ));
        }
        let mut best = None;
        for x in -1_i8..=1 {
            for y in -1_i8..=1 {
                for z in -1_i8..=1 {
                    if x == 0 && y == 0 && z == 0 {
                        continue;
                    }
                    let norm = f64::from(x * x + y * y + z * z).sqrt();
                    let direction = [
                        f64::from(x) / norm,
                        f64::from(y) / norm,
                        f64::from(z) / norm,
                    ];
                    let candidate =
                        std::array::from_fn(|axis| point[axis] + radius * direction[axis]);
                    if !point_is_in_vertex_star_kernel(topology, star, source_index, candidate)? {
                        continue;
                    }
                    let candidate_score = prepared.score(candidate)?;
                    if candidate_score >= score {
                        continue;
                    }
                    if best.as_ref().is_none_or(
                        |(best_score, best_point): &(StarScore, [f64; 3])| {
                            candidate_score < *best_score
                                || candidate_score == *best_score
                                    && candidate.map(f64::to_bits) < best_point.map(f64::to_bits)
                        },
                    ) {
                        best = Some((candidate_score, candidate));
                    }
                }
            }
        }
        if let Some((best_score, best_point)) = best {
            score = best_score;
            point = best_point;
        } else {
            radius *= 0.5;
            if radius < minimum_radius {
                break;
            }
        }
    }
    Ok((score < initial_score).then_some(point))
}

#[cfg(test)]
pub(in crate::cdt::volume_sliver) fn pattern_search_vertex_star(
    topology: &(impl crate::cdt::topology::VolumeTopologyRead + ?Sized),
    star: &[u32],
    source_index: usize,
    start: [f64; 3],
    metric: runmat_meshing_size::metric::MetricTensor3,
    cancellation: &dyn MeshingCancellationSignal,
) -> Result<Option<[f64; 3]>, DelaunayVolumeSliverError> {
    pattern_search_vertex_star_with_metrics(
        topology,
        star,
        source_index,
        start,
        &SmoothingObjective::Shape.prepare(topology, star, source_index, &|_| metric),
        cancellation,
    )
}

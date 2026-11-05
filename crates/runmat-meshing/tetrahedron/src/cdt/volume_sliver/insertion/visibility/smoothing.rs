//! Prepared boundary metrics and exact visibility constrain cavity smoothing.
use super::kernel::VisibilityKernel;
use super::*;
pub(in crate::cdt::volume_sliver) fn optimize_cavity_point(
    topology: &(impl VolumeTopologyRead + ?Sized),
    removed: &BTreeSet<u32>,
    start: [f64; 3],
    metrics: InsertionMetric<'_>,
    cancellation: &dyn MeshingCancellationSignal,
) -> Result<Option<[f64; 3]>, DelaunayVolumeSliverError> {
    let Some(scale) = local_scale(topology, removed) else {
        return Ok(None);
    };
    let boundary = cavity_boundary_faces(topology, removed);
    let kernel = VisibilityKernel::prepare(topology, &boundary, metrics)?;
    optimize(&kernel, scale, start, cancellation)
}
fn local_scale(
    topology: &(impl VolumeTopologyRead + ?Sized),
    removed: &BTreeSet<u32>,
) -> Option<f64> {
    let vertices = removed
        .iter()
        .flat_map(|index| topology.cell(*index).vertex_indices)
        .collect::<BTreeSet<_>>();
    let mut minimum = [f64::INFINITY; 3];
    let mut maximum = [f64::NEG_INFINITY; 3];
    for vertex in vertices {
        let point = topology.node(vertex).coordinates_m.canonical_value();
        for axis in 0..3 {
            minimum[axis] = minimum[axis].min(point[axis]);
            maximum[axis] = maximum[axis].max(point[axis]);
        }
    }
    let scale = (0..3)
        .map(|axis| maximum[axis] - minimum[axis])
        .fold(0., f64::hypot);
    (scale.is_finite() && scale > 0.).then_some(scale)
}
fn optimize(
    kernel: &VisibilityKernel,
    local_scale: f64,
    start: [f64; 3],
    cancellation: &dyn MeshingCancellationSignal,
) -> Result<Option<[f64; 3]>, DelaunayVolumeSliverError> {
    let mut point = start;
    let initial_score = kernel.score(point)?;
    let mut best_score = initial_score;
    let mut radius = local_scale * 0.25;
    let minimum_radius = local_scale * 2_f64.powi(-24);
    for iteration in 0..MAXIMUM_CAVITY_SMOOTHING_ITERATIONS {
        if iteration.is_multiple_of(8) && cancellation.is_cancelled() {
            return Err(error(
                DelaunayVolumeSliverErrorKind::Cancelled,
                "sliver cavity smoothing was cancelled",
            ));
        }
        let best = kernel.best_trial(point, radius, best_score)?;
        if let Some(candidate) = best {
            best_score = candidate.score;
            point = candidate.point;
        } else {
            radius *= 0.5;
            if radius < minimum_radius {
                break;
            }
        }
    }
    Ok((best_score > initial_score).then_some(point))
}

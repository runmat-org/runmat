//! Metric-aware opposite-face proposals may leave the fixed star. A later
//! closed-cavity transaction supplies complete geometry and quality authority.
use super::*;

pub(in crate::cdt::volume_sliver) fn target(
    topology: &(impl VolumeTopologyRead + ?Sized),
    source: &DelaunayTetrahedronQuality,
    identity: StableDigest,
    cancellation: &dyn MeshingCancellationSignal,
) -> Result<Option<[f64; 3]>, DelaunayVolumeSliverError> {
    let points = source.node_identities.map(|id| {
        topology
            .node(topology.node_index(id).expect("source corner"))
            .coordinates_m
            .canonical_value()
    });
    let slot = source
        .node_identities
        .iter()
        .position(|id| *id == identity)
        .expect("source corner");
    let face = (0..4)
        .filter(|i| *i != slot)
        .map(|i| points[i])
        .collect::<Vec<_>>();
    let cross = |a: [f64; 3], b: [f64; 3]| {
        [
            a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0],
        ]
    };
    let subtract = |a: [f64; 3], b: [f64; 3]| std::array::from_fn::<_, 3, _>(|i| a[i] - b[i]);
    let normal = cross(subtract(face[1], face[0]), subtract(face[2], face[0]));
    let side = |point: [f64; 3]| {
        normal
            .iter()
            .zip(subtract(point, face[0]))
            .map(|(n, d)| n * d)
            .sum::<f64>()
    };
    let expected = side(points[slot]);
    let metric = crate::cdt::volume_quality::PreparedJacobian::new(source.resolved_metric)
        .map_err(|cause| {
            error(
                DelaunayVolumeSliverErrorKind::Accounting(cause),
                "relocation metric numerical accounting failed",
            )
        })?
        .expect("admitted metric");
    let score = |point| {
        let mut trial = points;
        trial[slot] = point;
        Ok::<f64, DelaunayVolumeSliverError>(
            metric
                .score(trial)
                .map_err(|cause| {
                    error(
                        DelaunayVolumeSliverErrorKind::Accounting(cause),
                        "relocation score numerical accounting failed",
                    )
                })?
                .unwrap_or(0.),
        )
    };
    let mut point = points[slot];
    let initial = score(point)?;
    let mut best = initial;
    let scale = face
        .iter()
        .flat_map(|a| {
            face.iter()
                .map(move |b| subtract(*a, *b).iter().map(|v| v * v).sum::<f64>().sqrt())
        })
        .fold(0., f64::max);
    let mut radius = scale * 0.25;
    for iteration in 0..MAXIMUM_CAVITY_SMOOTHING_ITERATIONS {
        if iteration.is_multiple_of(8) && cancellation.is_cancelled() {
            return Err(error(
                DelaunayVolumeSliverErrorKind::Cancelled,
                "opposite-face metric search cancelled",
            ));
        }
        let mut next = None;
        for x in -1_i8..=1 {
            for y in -1_i8..=1 {
                for z in -1_i8..=1 {
                    if [x, y, z] == [0, 0, 0] {
                        continue;
                    }
                    let norm = f64::from(x * x + y * y + z * z).sqrt();
                    let trial = std::array::from_fn(|axis| {
                        point[axis] + radius * f64::from([x, y, z][axis]) / norm
                    });
                    if side(trial) * expected <= 0. {
                        continue;
                    }
                    let value = score(trial)?;
                    if value > best
                        && next
                            .as_ref()
                            .is_none_or(|(v, _): &(f64, [f64; 3])| value > *v)
                    {
                        next = Some((value, trial));
                    }
                }
            }
        }
        if let Some((value, trial)) = next {
            best = value;
            point = trial;
        } else {
            radius *= 0.5;
            if radius < scale * 2_f64.powi(-24) {
                break;
            }
        }
    }
    Ok((best > initial).then_some(point))
}

//! Bounded metric-normal proposals shared by point insertion and relocation.
use super::*;
use crate::cdt::volume_quality::PreparedJacobian;
pub(super) fn points_for_face(
    face: [[f64; 3]; 3],
    interior: [f64; 3],
    metric: runmat_meshing_size::metric::MetricTensor3,
    target: f64,
    cancellation: &dyn MeshingCancellationSignal,
) -> Result<Vec<([f64; 3], f64)>, DelaunayVolumeSliverError> {
    let edge = |a: usize| std::array::from_fn::<_, 3, _>(|i| face[a][i] - face[0][i]);
    let a = edge(1);
    let b = edge(2);
    let mut normal = [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ];
    let norm = normal.into_iter().fold(0., f64::hypot);
    if !norm.is_finite() || norm <= 0. {
        return Ok(Vec::new());
    }
    normal = normal.map(|v| v / norm);
    let mut height = (0..3)
        .map(|i| normal[i] * (interior[i] - face[0][i]))
        .sum::<f64>();
    if height < 0. {
        normal = normal.map(|v| -v);
        height = -height;
    }
    let accounting = |cause| {
        error(
            DelaunayVolumeSliverErrorKind::Accounting(cause),
            "metric offcenter numerical accounting failed",
        )
    };
    let Some(prepared) = PreparedJacobian::new(metric).map_err(accounting)? else {
        return Ok(Vec::new());
    };
    let Some(direction) = prepared.normal_step(normal).map_err(accounting)? else {
        return Ok(Vec::new());
    };
    let centroid: [f64; 3] = std::array::from_fn(|i| (face[0][i] + face[1][i] + face[2][i]) / 3.);
    let target = target * 1.01;
    let maximum_height = height * 0.9375;
    let mut result = Vec::new();
    for corner in face {
        for blend in [0., 0.0625, 0.125, 0.25] {
            if cancellation.is_cancelled() {
                return Err(error(
                    DelaunayVolumeSliverErrorKind::Cancelled,
                    "metric boundary offcenter cancelled",
                ));
            }
            let foot: [f64; 3] =
                std::array::from_fn(|i| corner[i] + blend * (centroid[i] - corner[i]));
            let point = |height: f64| std::array::from_fn(|i| foot[i] + height * direction[i]);
            let score = |height| {
                Ok::<f64, DelaunayVolumeSliverError>(
                    prepared
                        .score([face[0], face[1], face[2], point(height)])
                        .map_err(accounting)?
                        .unwrap_or(0.),
                )
            };
            let mut lower = 0.;
            let mut bracket = None;
            for sample in 1..=8 {
                let upper = maximum_height * f64::from(sample) / 8.;
                if score(upper)? >= target {
                    bracket = Some(upper);
                    break;
                }
                lower = upper;
            }
            let Some(mut upper) = bracket else {
                continue;
            };
            for _ in 0..32 {
                let middle = 0.5 * (lower + upper);
                if score(middle)? >= target {
                    upper = middle;
                } else {
                    lower = middle;
                }
            }
            let candidate = point(upper);
            let quality = score(upper)?;
            if candidate.iter().all(|v| v.is_finite()) && quality >= target {
                result.push((candidate, quality));
                let outer = point(maximum_height);
                let outer_quality = score(maximum_height)?;
                if outer_quality > quality {
                    result.push((outer, outer_quality));
                }
            }
        }
    }
    result.sort_by(|a, b| {
        let h = |point: [f64; 3]| {
            (0..3)
                .map(|i| normal[i] * (point[i] - face[0][i]))
                .sum::<f64>()
        };
        h(a.0)
            .total_cmp(&h(b.0))
            .then_with(|| a.0.map(f64::to_bits).cmp(&b.0.map(f64::to_bits)))
    });
    Ok(result)
}

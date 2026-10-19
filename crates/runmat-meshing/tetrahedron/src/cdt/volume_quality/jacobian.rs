//! Shared metric geometry preparation. Search asks for shape only; publication
//! additionally constructs radius evidence. Operand order and admission are identical.
mod gradient;
use runmat_meshing_core::predicate::tetrahedron_scaled_jacobian;
use runmat_meshing_size::metric::MetricTensor3;

#[derive(Clone, Copy)]
pub(in crate::cdt) struct PreparedJacobian {
    coefficients: [f64; 6],
}
pub(super) struct Geometry {
    pub(super) points: [[f64; 3]; 4],
    pub(super) minimum_edge: f64,
    pub(super) maximum_edge: f64,
    pub(super) jacobian: f64,
}
impl PreparedJacobian {
    pub(in crate::cdt) fn new(metric: MetricTensor3) -> Option<Self> {
        let l00 = metric.xx.sqrt();
        let l10 = metric.xy / l00;
        let l20 = metric.xz / l00;
        let l11 = (metric.yy - l10 * l10).sqrt();
        let l21 = (metric.yz - l20 * l10) / l11;
        let l22 = (metric.zz - l20 * l20 - l21 * l21).sqrt();
        let coefficients = [l00, l10, l20, l11, l21, l22];
        coefficients
            .into_iter()
            .all(f64::is_finite)
            .then_some(Self { coefficients })
    }
    pub(super) fn transform(&self, points: [[f64; 3]; 4]) -> [[f64; 3]; 4] {
        let [l00, l10, l20, l11, l21, l22] = self.coefficients;
        points.map(|point| {
            let relative = [
                point[0] - points[0][0],
                point[1] - points[0][1],
                point[2] - points[0][2],
            ];
            [
                l00 * relative[0] + l10 * relative[1] + l20 * relative[2],
                l11 * relative[1] + l21 * relative[2],
                l22 * relative[2],
            ]
        })
    }
    pub(super) fn geometry(&self, mut points: [[f64; 3]; 4]) -> Option<Geometry> {
        points.sort_by(|left, right| {
            left.iter()
                .zip(right.iter())
                .find_map(|(left, right)| (left != right).then(|| left.total_cmp(right)))
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        let points = self.transform(points);
        let (minimum_edge, maximum_edge) = edge_extrema(points)?;
        let jacobian = tetrahedron_scaled_jacobian(points);
        (jacobian.is_finite() && jacobian >= 0.).then_some(Geometry {
            points,
            minimum_edge,
            maximum_edge,
            jacobian,
        })
    }
    /// A metric-normal step with one unit of signed physical normal height.
    /// This is proposal geometry; exact topology and current quality admit it.
    pub(in crate::cdt) fn normal_step(&self, normal: [f64; 3]) -> Option<[f64; 3]> {
        let [l00, l10, l20, l11, l21, l22] = self.coefficients;
        let y0 = normal[0] / l00;
        let y1 = (normal[1] - l10 * y0) / l11;
        let y2 = (normal[2] - l20 * y0 - l21 * y1) / l22;
        let z = y2 / l22;
        let y = (y1 - l21 * z) / l11;
        let x = (y0 - l10 * y - l20 * z) / l00;
        let step = [x, y, z];
        let height = (0..3).map(|i| normal[i] * step[i]).sum::<f64>();
        (height.is_finite() && height > 0.)
            .then(|| step.map(|v| v / height))
            .filter(|step| step.iter().all(|v| v.is_finite()))
    }
    pub(in crate::cdt) fn score(&self, points: [[f64; 3]; 4]) -> Option<f64> {
        self.geometry(points).map(|geometry| geometry.jacobian)
    }
}

pub(in crate::cdt) fn evaluate_tetrahedron_jacobian(
    points: [[f64; 3]; 4],
    metric: MetricTensor3,
) -> Option<f64> {
    PreparedJacobian::new(metric)?.score(points)
}

fn edge_extrema(points: [[f64; 3]; 4]) -> Option<(f64, f64)> {
    let mut minimum = f64::INFINITY;
    let mut maximum = 0.0_f64;
    for left in 0..4 {
        for right in (left + 1)..4 {
            let squared = (0..3)
                .map(|axis| (points[right][axis] - points[left][axis]).powi(2))
                .sum::<f64>();
            let mut length = squared.sqrt();
            if !length.is_finite() {
                return None;
            }
            if length == 0.0 && points[right] != points[left] {
                length = f64::MIN_POSITIVE;
            }
            if length == 0.0 {
                return None;
            }
            minimum = minimum.min(length);
            maximum = maximum.max(length);
        }
    }
    Some((minimum, maximum))
}

#[cfg(test)]
mod tests;

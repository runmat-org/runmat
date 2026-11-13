//! Shared metric geometry preparation. Search asks for shape only; publication
//! additionally constructs radius evidence. Operand order and admission are identical.
mod gradient;
use runmat_meshing_core::predicate::{
    tetrahedron_scaled_jacobian, tetrahedron_scaled_jacobian_with_owner,
};
use runmat_meshing_size::metric::MetricTensor3;
use runmat_simulation_usage::{LedgerError, OperationCount, UsageCategory, UsageContext};

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
    pub(in crate::cdt) fn new(metric: MetricTensor3) -> Result<Option<Self>, LedgerError> {
        // Cholesky schedule: three square roots, three divisions, four
        // products, and four subtractions = 14 reference operations.
        let usage = UsageContext::capture();
        let mut batch = usage
            .as_ref()
            .map(|usage| {
                usage
                    .ledger()
                    .reserve(UsageCategory::Meshing, OperationCount::from_operations(14))
            })
            .transpose()?;
        let l00 = metric.xx.sqrt();
        let l10 = metric.xy / l00;
        let l20 = metric.xz / l00;
        let l11 = (metric.yy - l10 * l10).sqrt();
        let l21 = (metric.yz - l20 * l10) / l11;
        let l22 = (metric.zz - l20 * l20 - l21 * l21).sqrt();
        let coefficients = [l00, l10, l20, l11, l21, l22];
        if let Some(batch) = &mut batch {
            batch.complete(OperationCount::from_operations(14))?;
        }
        Ok(coefficients
            .into_iter()
            .all(f64::is_finite)
            .then_some(Self { coefficients }))
    }
    pub(super) fn transform(&self, points: [[f64; 3]; 4]) -> Result<[[f64; 3]; 4], LedgerError> {
        // Four points, each with three differences and triangular transform
        // rows of 5, 3, and 1 operations: 4 * 12 = 48.
        let usage = UsageContext::capture();
        let mut batch = usage
            .as_ref()
            .map(|usage| {
                usage
                    .ledger()
                    .reserve(UsageCategory::Meshing, OperationCount::from_operations(48))
            })
            .transpose()?;
        let transformed = self.transform_reference(points);
        if let Some(batch) = &mut batch {
            batch.complete(OperationCount::from_operations(48))?;
        }
        Ok(transformed)
    }
    pub(super) fn geometry(
        &self,
        mut points: [[f64; 3]; 4],
    ) -> Result<Option<Geometry>, LedgerError> {
        sort_points(&mut points);
        let points = self.transform(points)?;
        let Some((minimum_edge, maximum_edge)) = edge_extrema(points)? else {
            return Ok(None);
        };
        let jacobian = tetrahedron_scaled_jacobian(points)?;
        Ok(
            (jacobian.is_finite() && jacobian >= 0.).then_some(Geometry {
                points,
                minimum_edge,
                maximum_edge,
                jacobian,
            }),
        )
    }
    /// A metric-normal step with one unit of signed physical normal height.
    /// This is proposal geometry; exact topology and current quality admit it.
    pub(in crate::cdt) fn normal_step(
        &self,
        normal: [f64; 3],
    ) -> Result<Option<[f64; 3]>, LedgerError> {
        let usage = UsageContext::capture();
        let mut batch = usage
            .as_ref()
            .map(|usage| {
                usage
                    .ledger()
                    .reserve(UsageCategory::Meshing, OperationCount::from_operations(26))
            })
            .transpose()?;
        let [l00, l10, l20, l11, l21, l22] = self.coefficients;
        let y0 = normal[0] / l00;
        let y1 = (normal[1] - l10 * y0) / l11;
        let y2 = (normal[2] - l20 * y0 - l21 * y1) / l22;
        let z = y2 / l22;
        let y = (y1 - l21 * z) / l11;
        let x = (y0 - l10 * y - l20 * z) / l00;
        let step = [x, y, z];
        let height = (0..3).map(|i| normal[i] * step[i]).sum::<f64>();
        if !height.is_finite() || height <= 0. {
            if let Some(batch) = &mut batch {
                batch.complete(OperationCount::from_operations(23))?;
            }
            return Ok(None);
        }
        let step = step.map(|v| v / height);
        if let Some(batch) = &mut batch {
            batch.complete(OperationCount::from_operations(26))?;
        }
        Ok(step.iter().all(|v| v.is_finite()).then_some(step))
    }
    pub(in crate::cdt) fn score(&self, points: [[f64; 3]; 4]) -> Result<Option<f64>, LedgerError> {
        let Some(usage) = UsageContext::capture() else {
            return Ok(self.geometry(points)?.map(|geometry| geometry.jacobian));
        };
        // One bounded score kernel: transform 48 + six edge lengths 54 + four
        // scaled-Jacobian corners up to 280 = 382 reference operations. The
        // local prefix is settled once, including early invalid-edge exits.
        let mut batch = usage
            .ledger()
            .reserve(UsageCategory::Meshing, OperationCount::from_operations(382))?;
        let mut points = points;
        sort_points(&mut points);
        let transformed = self.transform_reference(points);
        let (edges, edge_work) = edge_extrema_reference(transformed);
        let mut completed = 48 + edge_work;
        let Some(_) = edges else {
            batch.complete(OperationCount::from_operations(completed))?;
            return Ok(None);
        };
        let jacobian =
            match tetrahedron_scaled_jacobian_with_owner(transformed, &mut batch, &mut completed) {
                Ok(value) => value,
                Err(cause) => {
                    batch.complete(OperationCount::from_operations(completed))?;
                    return Err(cause);
                }
            };
        batch.complete(OperationCount::from_operations(completed))?;
        Ok((jacobian.is_finite() && jacobian >= 0.0).then_some(jacobian))
    }

    fn transform_reference(&self, points: [[f64; 3]; 4]) -> [[f64; 3]; 4] {
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
}

fn sort_points(points: &mut [[f64; 3]; 4]) {
    points.sort_by(|left, right| {
        left.iter()
            .zip(right.iter())
            .find_map(|(left, right)| (left != right).then(|| left.total_cmp(right)))
            .unwrap_or(std::cmp::Ordering::Equal)
    });
}

pub(in crate::cdt) fn evaluate_tetrahedron_jacobian(
    points: [[f64; 3]; 4],
    metric: MetricTensor3,
) -> Result<Option<f64>, LedgerError> {
    let Some(prepared) = PreparedJacobian::new(metric)? else {
        return Ok(None);
    };
    prepared.score(points)
}

fn edge_extrema(points: [[f64; 3]; 4]) -> Result<Option<(f64, f64)>, LedgerError> {
    let usage = UsageContext::capture();
    let mut batch = usage
        .as_ref()
        .map(|usage| {
            usage
                .ledger()
                .reserve(UsageCategory::Meshing, OperationCount::from_operations(54))
        })
        .transpose()?;
    let (edges, completed) = edge_extrema_reference(points);
    if let Some(batch) = &mut batch {
        batch.complete(OperationCount::from_operations(completed))?;
    }
    Ok(edges)
}

fn edge_extrema_reference(points: [[f64; 3]; 4]) -> (Option<(f64, f64)>, u64) {
    let mut minimum = f64::INFINITY;
    let mut maximum = 0.0_f64;
    let mut completed = 0;
    for left in 0..4 {
        for right in (left + 1)..4 {
            let squared = (0..3)
                .map(|axis| (points[right][axis] - points[left][axis]).powi(2))
                .sum::<f64>();
            let mut length = squared.sqrt();
            completed += 9;
            if !length.is_finite() {
                return (None, completed);
            }
            if length == 0.0 && points[right] != points[left] {
                length = f64::MIN_POSITIVE;
            }
            if length == 0.0 {
                return (None, completed);
            }
            minimum = minimum.min(length);
            maximum = maximum.max(length);
        }
    }
    (Some((minimum, maximum)), completed)
}

#[cfg(test)]
mod tests;

//! Smooth corner derivatives are proposal data, never geometric admission.
use super::PreparedJacobian;
use runmat_simulation_usage::{LedgerError, OperationCount, UsageCategory, UsageContext};

pub(in crate::cdt) struct CornerGradient {
    pub(in crate::cdt) value: f64,
    pub(in crate::cdt) gradients: [[f64; 3]; 4],
}
fn dot(a: [f64; 3], b: [f64; 3]) -> f64 {
    (0..3).map(|i| a[i] * b[i]).sum()
}
fn cross(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}
impl PreparedJacobian {
    pub(in crate::cdt) fn corner_gradients(
        &self,
        points: [[f64; 3]; 4],
    ) -> Result<Option<[CornerGradient; 4]>, LedgerError> {
        // Norm3D is the declared reference sqrt(x²+y²+z²): three products,
        // two additions, one square root = six operations. The three `hypot`
        // calls lower that stable norm and do not add reference work.
        // A valid corner: edges 9 + three Norm3D 18 + unit divisions 9 +
        // determinant cross/dot 14 + three derivative edges (42 each) 126 +
        // final sqrt(2) product 2 = 178. Four corners plus transform 48 = 760.
        let usage = UsageContext::capture();
        let mut batch = usage
            .as_ref()
            .map(|usage| {
                usage
                    .ledger()
                    .reserve(UsageCategory::Meshing, OperationCount::from_operations(760))
            })
            .transpose()?;
        let transformed = self.transform_reference(points);
        let mut completed = 48;
        let [l00, l10, l20, l11, l21, l22] = self.coefficients;
        let mut result = Vec::with_capacity(4);
        for corner in 0..4 {
            let neighbors = (0..4).filter(|i| *i != corner).collect::<Vec<_>>();
            let edges: [[f64; 3]; 3] = std::array::from_fn(|i| {
                std::array::from_fn(|axis| {
                    transformed[neighbors[i]][axis] - transformed[corner][axis]
                })
            });
            let lengths = edges.map(|edge| edge.into_iter().fold(0., f64::hypot));
            completed += 27;
            if lengths.iter().any(|v| !v.is_finite() || *v <= 0.) {
                if let Some(batch) = &mut batch {
                    batch.complete(OperationCount::from_operations(completed))?;
                }
                return Ok(None);
            }
            let unit: [[f64; 3]; 3] = std::array::from_fn(|i| edges[i].map(|v| v / lengths[i]));
            let determinant = dot(cross(unit[0], unit[1]), unit[2]);
            completed += 23;
            if !determinant.is_finite() || determinant == 0. {
                if let Some(batch) = &mut batch {
                    batch.complete(OperationCount::from_operations(completed))?;
                }
                return Ok(None);
            }
            let mut gradients = [[0.; 3]; 4];
            for edge in 0..3 {
                let cof = cross(unit[(edge + 1) % 3], unit[(edge + 2) % 3]);
                let g: [f64; 3] = std::array::from_fn(|axis| {
                    2_f64.sqrt() / lengths[edge]
                        * (determinant.signum() * cof[axis] - determinant.abs() * unit[edge][axis])
                });
                let world = [
                    l00 * g[0],
                    l10 * g[0] + l11 * g[1],
                    l20 * g[0] + l21 * g[1] + l22 * g[2],
                ];
                for (axis, value) in world.into_iter().enumerate() {
                    gradients[neighbors[edge]][axis] += value;
                    gradients[corner][axis] -= value;
                }
                completed += 42;
            }
            if gradients.iter().flatten().any(|v| !v.is_finite()) {
                if let Some(batch) = &mut batch {
                    batch.complete(OperationCount::from_operations(completed))?;
                }
                return Ok(None);
            }
            completed += 2;
            result.push(CornerGradient {
                value: 2_f64.sqrt() * determinant.abs(),
                gradients,
            });
        }
        if let Some(batch) = &mut batch {
            batch.complete(OperationCount::from_operations(completed))?;
        }
        Ok(result.try_into().ok())
    }
}
#[cfg(test)]
mod tests;

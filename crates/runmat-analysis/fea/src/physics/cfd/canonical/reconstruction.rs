//! Geometry-only least-squares authority. No field or boundary-value state is retained.
use super::{geometry_ops::*, CanonicalFluidError as Error};

pub(super) struct GradientMetric {
    matrix: [[f64; 3]; 3],
}
impl GradientMetric {
    pub(super) fn derive(displacements: &[V]) -> Result<Self, Error> {
        let mut matrix = [[0.0; 3]; 3];
        for &d in displacements {
            let dsq = dot(d, d);
            if dsq <= 0.0 || !dsq.is_finite() {
                return Err(Error::Invalid("gradient stencil metric"));
            }
            for i in 0..3 {
                for j in 0..3 {
                    matrix[i][j] += d[i] * d[j] / dsq;
                }
            }
        }
        Ok(Self { matrix })
    }
    /// Whether the sample directions span three dimensions with margin: `det M ≥ 1e-6 (tr M / 3)³`.
    /// `M` is symmetric positive semidefinite, so `det M / (tr M / 3)³ = Π λ_i / λ̄³ ∈ [0, 1]`: zero
    /// exactly when the samples are coplanar or collinear, and at least `λ_min / λ_max · λ_mid / λ̄`
    /// otherwise. On the compliant-tube mesh every other stencil sits above 0.048; coplanar rings sit at 0.
    pub(super) fn spans_three_dimensions(&self) -> bool {
        let m = &self.matrix;
        let det = m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
            - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
            + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]);
        let mean = (m[0][0] + m[1][1] + m[2][2]) / 3.0;
        mean > 0.0 && det >= 1e-6 * mean * mean * mean
    }
    pub(super) fn weight(&self, displacement: V) -> Result<V, Error> {
        solve_three(
            self.matrix,
            scale(displacement, 1.0 / dot(displacement, displacement)),
        )
    }
}

/// Names the cell whose stencil failed, so a geometry defect can be traced to its element.
pub(super) fn at_cell(
    error: Error,
    cell: &crate::topology::FiniteVolumeCell,
    samples: usize,
) -> Error {
    match error {
        Error::Invalid(reason) => Error::InvalidCell {
            reason,
            element_id: cell.element_id,
            centroid_m: cell.centroid_m,
            samples,
        },
        other => other,
    }
}

fn solve_three(mut a: [[f64; 3]; 3], mut b: V) -> Result<V, Error> {
    let scale = a.iter().flatten().fold(0.0f64, |m, x| m.max(x.abs()));
    for k in 0..3 {
        let pivot = (k..3)
            .max_by(|i, j| a[*i][k].abs().total_cmp(&a[*j][k].abs()))
            .unwrap();
        if a[pivot][k].abs() <= scale * f64::EPSILON * 128.0 {
            return Err(Error::Invalid("rank-deficient physical gradient stencil"));
        }
        a.swap(k, pivot);
        b.swap(k, pivot);
        for i in k + 1..3 {
            let q = a[i][k] / a[k][k];
            for j in k..3 {
                a[i][j] -= q * a[k][j];
            }
            b[i] -= q * b[k];
        }
    }
    let mut x = [0.0; 3];
    for i in (0..3).rev() {
        x[i] = (b[i] - (i + 1..3).map(|j| a[i][j] * x[j]).sum::<f64>()) / a[i][i];
    }
    if x.iter().any(|v| !v.is_finite()) {
        return Err(Error::Invalid("nonfinite physical gradient"));
    }
    Ok(x)
}

#[cfg(test)]
mod tests;

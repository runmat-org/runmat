//! One owned sample family and genuine interpolated interval/rectangle peaks.
//! Column-major callback storage and physical sampling axes remain explicit.
use super::bernstein::{self, invalid, Surface};
use crate::builtins::math::optim::common::ScalarCooperation;
use crate::BuiltinResult;
use std::sync::Arc;
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub(super) struct FamilyId(pub usize);
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub(super) struct MeshRevision(pub u64);
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub(super) struct PeakId {
    pub family: FamilyId,
    pub revision: MeshRevision,
    pub row: usize,
    pub column: usize,
}
#[derive(Clone, Copy, Debug)]
pub(super) struct Spacing {
    pub column: f64,
    pub row: Option<f64>,
}
impl Spacing {
    pub fn validate(self) -> BuiltinResult<Self> {
        if !self.column.is_finite()
            || self.column <= 0.0
            || self.row.is_some_and(|x| !x.is_finite() || x <= 0.0)
        {
            return Err(invalid("returned sampling intervals must be finite and positive; a vector's second interval is zero"));
        }
        Ok(self)
    }
    pub fn refine(self) -> BuiltinResult<Self> {
        let next = Self {
            column: self.column * 0.5,
            row: self.row.map(|x| x * 0.5),
        };
        if next.column == 0.0 || next.column == self.column || next.row == Some(0.0) {
            return Err(invalid(
                "sampling interval cannot be refined in finite arithmetic",
            ));
        }
        Ok(next)
    }
}
pub(super) struct Samples {
    pub values: Vec<f64>,
    pub rows: usize,
    pub columns: usize,
    pub spacing: Spacing,
}
impl Samples {
    pub fn new(
        values: Vec<f64>,
        rows: usize,
        columns: usize,
        spacing: Spacing,
    ) -> BuiltinResult<Self> {
        if rows == 0
            || columns == 0
            || rows.checked_mul(columns) != Some(values.len())
            || values.iter().any(|v| !v.is_finite())
        {
            return Err(invalid(
                "semi-infinite samples must be a nonempty finite real vector or matrix",
            ));
        }
        if rows == 1 || columns == 1 {
            // The documented vector form consumes only S(i,1); S(i,2)
            // is dormant, not an additional two-dimensional obligation.
            let spacing = Spacing {
                column: spacing.column,
                row: None,
            }
            .validate()?;
            return Ok(Self {
                columns: values.len(),
                rows: 1,
                values,
                spacing,
            });
        }
        let spacing = spacing.validate()?;
        if spacing.row.is_none() {
            return Err(invalid(
                "matrix sample families require two positive sampling intervals",
            ));
        }
        Ok(Self {
            values,
            rows,
            columns,
            spacing,
        })
    }
    pub fn same_domain(&self, other: &Self) -> bool {
        if self.spacing.row.is_some() != other.spacing.row.is_some() {
            return false;
        }
        [
            (
                self.columns,
                self.spacing.column,
                other.columns,
                other.spacing.column,
            ),
            (
                self.rows,
                self.spacing.row.unwrap_or(0.0),
                other.rows,
                other.spacing.row.unwrap_or(0.0),
            ),
        ]
        .into_iter()
        .all(|(n, h, m, k)| {
            let a = (n - 1) as f64 * h;
            let b = (m - 1) as f64 * k;
            a.is_finite()
                && b.is_finite()
                && (a - b).abs() <= 32.0 * f64::EPSILON * a.abs().max(b.abs())
        })
    }
    pub fn interpolated_at(&self, location: [f64; 2]) -> BuiltinResult<f64> {
        let coordinate = |value: f64, h: f64, n: usize| -> BuiltinResult<(usize, f64)> {
            if n == 1 {
                if value != 0.0 {
                    return Err(invalid("frozen peak is outside the returned mesh"));
                }
                return Ok((0, 0.0));
            }
            let parameter = value / h;
            let upper = (n - 1) as f64;
            if !parameter.is_finite()
                || parameter < 0.0
                || parameter > upper + 32.0 * f64::EPSILON * upper
            {
                return Err(invalid("frozen physical peak is outside the returned mesh"));
            }
            let parameter = parameter.min(upper);
            let cell = (parameter.floor() as usize).min(n - 2);
            Ok((cell, parameter - cell as f64))
        };
        let (column, u) = coordinate(location[0], self.spacing.column, self.columns)?;
        let (row, v) = coordinate(location[1], self.spacing.row.unwrap_or(1.0), self.rows)?;
        Window::new(row, column, self, 3)
            .stencil(self, v, u)
            .evaluate(&self.values)
    }
    fn value(&self, row: usize, column: usize) -> f64 {
        self.values[row + self.rows * column]
    }
}
#[derive(Clone, Debug)]
pub(super) struct Stencil {
    indices: [usize; 16],
    weights: [f64; 16],
    count: usize,
}
impl Stencil {
    pub fn evaluate(&self, values: &[f64]) -> BuiltinResult<f64> {
        if self.indices[..self.count]
            .iter()
            .any(|&index| index >= values.len())
        {
            return Err(invalid("frozen peak stencil has a different sample extent"));
        }
        let mut scale = 0.0f64;
        for &index in &self.indices[..self.count] {
            scale = scale.max(values[index].abs());
        }
        if scale == 0.0 {
            return Ok(0.0);
        }
        let mut value = 0.0;
        let mut correction = 0.0;
        for (&index, &weight) in self.indices[..self.count]
            .iter()
            .zip(&self.weights[..self.count])
        {
            let term = weight * (values[index] / scale) - correction;
            let next = value + term;
            correction = (next - value) - term;
            value = next;
        }
        let value = value * scale;
        if !value.is_finite() {
            return Err(invalid("frozen peak interpolation overflow"));
        }
        Ok(value)
    }
}
#[derive(Clone, Debug)]
pub(super) struct Peak {
    pub id: PeakId,
    /// Parameter coordinates relative to the callback's fixed interval origin.
    pub location: [f64; 2],
    pub value: f64,
    pub upper: f64,
    pub stencil: Stencil,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Recommendation {
    Keep,
    Refine,
    Coarsen,
}
pub(super) struct Analysis {
    pub samples: Arc<Samples>,
    pub peaks: Vec<Peak>,
    pub recommended: Spacing,
    pub recommendation: Recommendation,
}
struct Window {
    row: usize,
    column: usize,
    row_degree: usize,
    column_degree: usize,
    row_start: usize,
    column_start: usize,
}
fn axis(cell: usize, length: usize, maximum_degree: usize) -> (usize, usize) {
    let degree = maximum_degree.min(length - 1);
    let start = cell.saturating_sub(1).min(length - degree - 1);
    (degree, start)
}
impl Window {
    fn new(row: usize, column: usize, samples: &Samples, degree: usize) -> Self {
        let (row_degree, row_start) = axis(row, samples.rows, degree);
        let (column_degree, column_start) = axis(column, samples.columns, degree);
        Self {
            row,
            column,
            row_degree,
            column_degree,
            row_start,
            column_start,
        }
    }
    fn stencil(&self, samples: &Samples, row: f64, column: f64) -> Stencil {
        let rw = bernstein::weights(
            self.row_degree,
            self.row as f64 - self.row_start as f64 + row,
        );
        let cw = bernstein::weights(
            self.column_degree,
            self.column as f64 - self.column_start as f64 + column,
        );
        let mut out = Stencil {
            indices: [0; 16],
            weights: [0.0; 16],
            count: 0,
        };
        for (r, &a) in rw.iter().enumerate().take(self.row_degree + 1) {
            for (c, &b) in cw.iter().enumerate().take(self.column_degree + 1) {
                out.indices[out.count] =
                    self.row_start + r + samples.rows * (self.column_start + c);
                out.weights[out.count] = a * b;
                out.count += 1;
            }
        }
        out
    }
    fn extrapolation_defect(&self, samples: &Samples) -> BuiltinResult<f64> {
        let mut defect = 0.0f64;
        if self.column_degree == 3 {
            for row in self.row_start..=self.row_start + self.row_degree {
                for (start, target, t) in [
                    (self.column_start, self.column_start + 3, 3.0),
                    (self.column_start + 1, self.column_start, -1.0),
                ] {
                    let weights = bernstein::weights(2, t);
                    let mut stencil = Stencil {
                        indices: [0; 16],
                        weights: [0.0; 16],
                        count: 3,
                    };
                    for (i, &weight) in weights.iter().enumerate().take(3) {
                        stencil.indices[i] = row + samples.rows * (start + i);
                        stencil.weights[i] = weight;
                    }
                    let error =
                        (stencil.evaluate(&samples.values)? - samples.value(row, target)).abs();
                    if !error.is_finite() {
                        return Err(invalid("interpolation extrapolation diagnostic overflow"));
                    }
                    defect = defect.max(error);
                }
            }
        }
        if self.row_degree == 3 {
            for col in self.column_start..=self.column_start + self.column_degree {
                for (start, target, t) in [
                    (self.row_start, self.row_start + 3, 3.0),
                    (self.row_start + 1, self.row_start, -1.0),
                ] {
                    let weights = bernstein::weights(2, t);
                    let mut stencil = Stencil {
                        indices: [0; 16],
                        weights: [0.0; 16],
                        count: 3,
                    };
                    for (i, &weight) in weights.iter().enumerate().take(3) {
                        stencil.indices[i] = start + i + samples.rows * col;
                        stencil.weights[i] = weight;
                    }
                    let error =
                        (stencil.evaluate(&samples.values)? - samples.value(target, col)).abs();
                    if !error.is_finite() {
                        return Err(invalid("interpolation extrapolation diagnostic overflow"));
                    }
                    defect = defect.max(error);
                }
            }
        }
        Ok(defect)
    }
    fn surface(&self, samples: &Samples) -> BuiltinResult<Surface> {
        let rb = bernstein::basis(self.row_degree, self.row - self.row_start);
        let cb = bernstein::basis(self.column_degree, self.column - self.column_start);
        let mut scale = 0.0f64;
        for r in 0..=self.row_degree {
            for c in 0..=self.column_degree {
                scale = scale.max(
                    samples
                        .value(self.row_start + r, self.column_start + c)
                        .abs(),
                );
            }
        }
        if scale == 0.0 {
            scale = 1.0;
        }
        let mut coefficients = [0.0; 16];
        for r in 0..=self.row_degree {
            for c in 0..=self.column_degree {
                for a in 0..=self.row_degree {
                    for b in 0..=self.column_degree {
                        coefficients[r * 4 + c] += rb[r][a]
                            * cb[c][b]
                            * (samples.value(self.row_start + a, self.column_start + b) / scale);
                    }
                }
                if !coefficients[r * 4 + c].is_finite() {
                    return Err(invalid("interpolation coefficient overflow"));
                }
            }
        }
        Ok(Surface {
            coefficients,
            row_degree: self.row_degree,
            column_degree: self.column_degree,
            scale,
        })
    }
}
pub(super) async fn analyze(
    samples: Arc<Samples>,
    family: FamilyId,
    revision: MeshRevision,
    tolerance: f64,
    cooperation: &mut ScalarCooperation,
) -> BuiltinResult<Analysis> {
    if !tolerance.is_finite() || tolerance < 0.0 {
        return Err(invalid("invalid mesh tolerance"));
    }
    let row_cells = samples.rows.saturating_sub(1).max(1);
    let column_cells = samples.columns.saturating_sub(1).max(1);
    let cells = row_cells
        .checked_mul(column_cells)
        .ok_or_else(|| invalid("interpolation patch count overflow"))?;
    let mut peaks = runmat_value::exact_capacity_vec(cells).map_err(|source| {
        crate::build_runtime_error("fseminf: peak allocation failed")
            .with_source(source)
            .build()
    })?;
    let mut refine = samples.columns < 4 || (samples.rows > 1 && samples.rows < 4);
    let mut max_defect = 0.0f64;
    let mut max_gap = 0.0f64;
    let mut max_upper = f64::NEG_INFINITY;
    for row in 0..row_cells {
        for column in 0..column_cells {
            cooperation.completed(1).await?;
            let window = Window::new(row, column, &samples, 3);
            let maximum =
                bernstein::maximize(window.surface(&samples)?, tolerance, cooperation).await?;
            let stencil = window.stencil(&samples, maximum.row, maximum.column);
            let value = stencil.evaluate(&samples.values)?;
            let quadratic = Window::new(row, column, &samples, 2)
                .stencil(&samples, maximum.row, maximum.column)
                .evaluate(&samples.values)?;
            let mut corner = f64::NEG_INFINITY;
            for r in [row, (row + 1).min(samples.rows - 1)] {
                for c in [column, (column + 1).min(samples.columns - 1)] {
                    corner = corner.max(samples.value(r, c));
                }
            }
            let defect = (value - quadratic)
                .abs()
                .max(window.extrapolation_defect(&samples)?);
            // Peak/sample separation and quadratic/cubic disagreement are actual
            // interpolation diagnostics, not a claim about unsampled arbitrary
            // continuous functions. Accepted-generation refresh must compare the
            // successor grid before publishing any nonlinear KKT result.
            max_defect = max_defect.max(defect);
            max_gap = max_gap.max((maximum.upper - corner).max(0.0));
            max_upper = max_upper.max(maximum.upper);
            // A resolved off-grid maximum is not an interpolation error.
            // Refinement measures actual quadratic/cubic/held-out disagreement,
            // rather than forcing sampled nodes to approach the same maximum.
            refine |= defect > tolerance.max((-maximum.upper).max(0.0) * 0.1);
            let location = [
                (column as f64 + maximum.column) * samples.spacing.column,
                (row as f64 + maximum.row) * samples.spacing.row.unwrap_or(0.0),
            ];
            if location.iter().any(|x| !x.is_finite()) {
                return Err(invalid("sampling parameter extent overflow"));
            }
            peaks.push(Peak {
                id: PeakId {
                    family,
                    revision,
                    row,
                    column,
                },
                location,
                value,
                upper: maximum.upper,
                stencil,
            });
        }
    }
    let coarsen = !refine
        && (samples.columns > 7 || samples.rows > 7)
        && max_upper < -100.0 * tolerance
        && max_defect <= tolerance * 0.01
        && max_gap < (-max_upper) * 0.1;
    let recommendation = if refine {
        Recommendation::Refine
    } else if coarsen {
        Recommendation::Coarsen
    } else {
        Recommendation::Keep
    };
    let recommended = match recommendation {
        Recommendation::Refine => samples.spacing.refine()?,
        Recommendation::Coarsen => Spacing {
            column: if samples.columns > 7 {
                samples.spacing.column * 2.0
            } else {
                samples.spacing.column
            },
            row: samples
                .spacing
                .row
                .map(|value| if samples.rows > 7 { value * 2.0 } else { value }),
        }
        .validate()?,
        Recommendation::Keep => samples.spacing,
    };
    Ok(Analysis {
        samples,
        peaks,
        recommended,
        recommendation,
    })
}
#[cfg(test)]
#[path = "mesh/tests.rs"]
mod tests;

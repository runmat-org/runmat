//! Tensor Bernstein bounds for genuine maxima of local interpolation surfaces.
//! No finite set of sampled nodes is substituted for an interval maximum.
use crate::builtins::math::optim::common::ScalarCooperation;
use crate::{BuiltinResult, RuntimeError};
pub(super) fn invalid(detail: impl AsRef<str>) -> RuntimeError {
    crate::build_runtime_error(format!("fseminf: {}", detail.as_ref()))
        .with_builtin("fseminf")
        .with_identifier(
            runmat_builtins::FSEMINF_ERROR_INVALID_ARGUMENT
                .identifier
                .expect("canonical fseminf identifier"),
        )
        .with_gpu_gather_retry(crate::GpuGatherRetry::Never)
        .build()
}
#[derive(Clone)]
pub(super) struct Surface {
    pub coefficients: [f64; 16],
    pub row_degree: usize,
    pub column_degree: usize,
    pub scale: f64,
}
#[derive(Clone, Copy, Debug)]
pub(super) struct Maximum {
    pub row: f64,
    pub column: f64,
    pub value: f64,
    pub upper: f64,
}
impl Surface {
    pub fn evaluate(&self, row: f64, column: f64) -> f64 {
        let mut columns = [0.0; 4];
        for (col, out) in columns.iter_mut().enumerate().take(self.column_degree + 1) {
            let mut values = [0.0; 4];
            for (r, value) in values.iter_mut().enumerate().take(self.row_degree + 1) {
                *value = self.coefficients[r * 4 + col];
            }
            *out = casteljau(values, self.row_degree, row);
        }
        casteljau(columns, self.column_degree, column)
    }
    fn upper(&self) -> f64 {
        let mut upper = f64::NEG_INFINITY;
        for row in 0..=self.row_degree {
            for col in 0..=self.column_degree {
                upper = upper.max(self.coefficients[row * 4 + col]);
            }
        }
        upper
    }
    fn split(&self, row: bool) -> (Self, Self) {
        let mut left = self.clone();
        let mut right = self.clone();
        if row {
            for col in 0..=self.column_degree {
                let mut values = [0.0; 4];
                for (r, value) in values.iter_mut().enumerate().take(self.row_degree + 1) {
                    *value = self.coefficients[r * 4 + col];
                }
                let (a, b) = split_curve(values, self.row_degree);
                for r in 0..=self.row_degree {
                    left.coefficients[r * 4 + col] = a[r];
                    right.coefficients[r * 4 + col] = b[r];
                }
            }
        } else {
            for r in 0..=self.row_degree {
                let mut values = [0.0; 4];
                for (col, value) in values.iter_mut().enumerate().take(self.column_degree + 1) {
                    *value = self.coefficients[r * 4 + col];
                }
                let (a, b) = split_curve(values, self.column_degree);
                for col in 0..=self.column_degree {
                    left.coefficients[r * 4 + col] = a[col];
                    right.coefficients[r * 4 + col] = b[col];
                }
            }
        }
        (left, right)
    }
}
fn casteljau(mut values: [f64; 4], degree: usize, t: f64) -> f64 {
    for level in (1..=degree).rev() {
        for i in 0..level {
            values[i] = (1.0 - t) * values[i] + t * values[i + 1];
        }
    }
    values[0]
}
fn split_curve(mut values: [f64; 4], degree: usize) -> ([f64; 4], [f64; 4]) {
    let mut left = [0.0; 4];
    let mut right = [0.0; 4];
    left[0] = values[0];
    right[degree] = values[degree];
    for level in 1..=degree {
        for i in 0..=degree - level {
            values[i] = 0.5 * values[i] + 0.5 * values[i + 1];
        }
        left[level] = values[0];
        right[degree - level] = values[degree - level];
    }
    (left, right)
}
struct Region {
    surface: Surface,
    row: [f64; 2],
    column: [f64; 2],
}
pub(super) async fn maximize(
    surface: Surface,
    tolerance: f64,
    cooperation: &mut ScalarCooperation,
) -> BuiltinResult<Maximum> {
    if !surface.scale.is_finite()
        || surface.scale <= 0.0
        || !tolerance.is_finite()
        || tolerance < 0.0
        || surface.row_degree > 3
        || surface.column_degree > 3
    {
        return Err(invalid("invalid interpolation bound geometry"));
    }
    let scale = surface.scale;
    let mut best = Maximum {
        row: 0.0,
        column: 0.0,
        value: surface.evaluate(0.0, 0.0),
        upper: f64::NEG_INFINITY,
    };
    let mut pending = runmat_value::exact_capacity_vec(1).map_err(|source| {
        crate::build_runtime_error("fseminf: interpolation queue allocation failed")
            .with_source(source)
            .build()
    })?;
    pending.push(Region {
        surface,
        row: [0.0, 1.0],
        column: [0.0, 1.0],
    });
    let mut terminal_upper = best.value;
    while let Some(region) = pending.pop() {
        cooperation.completed(1).await?;
        let upper = region.surface.upper();
        if !upper.is_finite() {
            return Err(invalid("interpolation coefficients overflow"));
        }
        for row in [0.0, 0.5, 1.0] {
            for col in [0.0, 0.5, 1.0] {
                let value = region.surface.evaluate(row, col);
                if !value.is_finite() {
                    return Err(invalid("interpolation evaluation overflow"));
                }
                if value > best.value {
                    best.row = region.row[0] + row * (region.row[1] - region.row[0]);
                    best.column = region.column[0] + col * (region.column[1] - region.column[0]);
                    best.value = value;
                }
            }
        }
        let normalized_tolerance = tolerance / region.surface.scale;
        if upper - best.value <= normalized_tolerance {
            terminal_upper = terminal_upper.max(upper);
            continue;
        }
        let row_width = region.row[1] - region.row[0];
        let column_width = region.column[1] - region.column[0];
        let split_row = region.surface.row_degree > 0
            && (region.surface.column_degree == 0 || row_width >= column_width);
        let axis = if split_row { region.row } else { region.column };
        let middle = axis[0] + 0.5 * (axis[1] - axis[0]);
        if middle == axis[0] || middle == axis[1] {
            return Err(invalid(
                "interpolation maximum cannot be resolved at the requested tolerance",
            ));
        }
        let (a, b) = region.surface.split(split_row);
        pending.try_reserve(2).map_err(|source| {
            crate::build_runtime_error("fseminf: interpolation queue allocation failed")
                .with_source(source)
                .build()
        })?;
        let mut first = Region {
            surface: a,
            row: region.row,
            column: region.column,
        };
        let mut second = Region {
            surface: b,
            row: region.row,
            column: region.column,
        };
        if split_row {
            first.row[1] = middle;
            second.row[0] = middle;
        } else {
            first.column[1] = middle;
            second.column[0] = middle;
        }
        pending.push(second);
        pending.push(first);
    }
    best.upper = terminal_upper.max(best.value);
    best.value *= scale;
    best.upper *= scale;
    if !best.value.is_finite() || !best.upper.is_finite() {
        return Err(invalid("interpolated maximum exceeds finite arithmetic"));
    }
    Ok(best)
}
pub(super) fn binomial(n: usize, k: usize) -> f64 {
    const TABLE: [[f64; 4]; 4] = [
        [1.0, 0.0, 0.0, 0.0],
        [1.0, 1.0, 0.0, 0.0],
        [1.0, 2.0, 1.0, 0.0],
        [1.0, 3.0, 3.0, 1.0],
    ];
    TABLE[n][k]
}
/// Lagrange basis on a consecutive equispaced stencil, expressed in
/// Bernstein coefficients over one unit sampling cell.
pub(super) fn basis(degree: usize, offset: usize) -> [[f64; 4]; 4] {
    let mut result = [[0.0; 4]; 4];
    for node in 0..=degree {
        let mut power = [0.0; 4];
        power[0] = 1.0;
        let mut order = 0;
        let mut denominator = 1.0;
        for other in 0..=degree {
            if other == node {
                continue;
            }
            let shift = offset as f64 - other as f64;
            let mut next = [0.0; 4];
            for j in 0..=order {
                next[j] += shift * power[j];
                next[j + 1] += power[j];
            }
            power = next;
            order += 1;
            denominator *= node as f64 - other as f64;
        }
        for k in 0..=degree {
            for (j, &coefficient) in power.iter().enumerate().take(k + 1) {
                result[k][node] += coefficient / denominator * binomial(k, j) / binomial(degree, j);
            }
        }
    }
    result
}
pub(super) fn weights(degree: usize, t: f64) -> [f64; 4] {
    let mut result = [0.0; 4];
    for (node, out) in result.iter_mut().enumerate().take(degree + 1) {
        *out = 1.0;
        for other in 0..=degree {
            if other != node {
                *out *= (t - other as f64) / (node as f64 - other as f64);
            }
        }
    }
    result
}

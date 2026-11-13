use super::*;

#[derive(Clone)]
pub(super) struct Observation {
    pub(super) point: Point,
    pub(super) objective: f64,
    pub(super) constraint_violations: Vec<f64>,
    pub(super) objective_time: f64,
}

impl Observation {
    pub(super) fn feasible(&self) -> bool {
        self.constraint_violations.iter().all(|value| *value <= 0.0)
    }

    fn total_violation(&self) -> f64 {
        self.constraint_violations
            .iter()
            .map(|value| value.max(0.0))
            .sum()
    }

    pub(super) fn penalized_objective(&self) -> f64 {
        self.objective + self.total_violation() * 1.0e6
    }
}

#[derive(Default)]
pub(super) struct SurrogateSummary {
    pub(super) length_scale: f64,
    pub(super) noise: f64,
}

pub(super) struct GaussianProcess {
    x: Vec<Vec<f64>>,
    y_mean: f64,
    y_scale: f64,
    alpha: Vec<f64>,
    pub(super) length_scale: f64,
}

impl GaussianProcess {
    pub(super) fn fit(
        runtime: Option<&crate::context::RuntimeContext>,
        observations: &[Observation],
    ) -> BuiltinResult<Self> {
        check_cancellation(runtime)?;
        if observations.is_empty() {
            return Ok(Self {
                x: Vec::new(),
                y_mean: 0.0,
                y_scale: 1.0,
                alpha: Vec::new(),
                length_scale: 0.35,
            });
        }
        let x = observations
            .iter()
            .map(|obs| obs.point.encoded.clone())
            .collect::<Vec<_>>();
        let y_raw = observations
            .iter()
            .map(Observation::penalized_objective)
            .collect::<Vec<_>>();
        let y_mean = y_raw.iter().sum::<f64>() / y_raw.len() as f64;
        let variance = y_raw
            .iter()
            .map(|value| (value - y_mean).powi(2))
            .sum::<f64>()
            / y_raw.len().max(1) as f64;
        let y_scale = variance.sqrt().max(1.0);
        let y = y_raw
            .iter()
            .map(|value| (value - y_mean) / y_scale)
            .collect::<Vec<_>>();
        let dims = x.first().map(Vec::len).unwrap_or(1).max(1);
        let length_scale = (0.35 / (dims as f64).sqrt()).max(0.08);
        let mut kernel = vec![vec![0.0; x.len()]; x.len()];
        for row in 0..x.len() {
            check_cancellation(runtime)?;
            for col in 0..x.len() {
                kernel[row][col] = rbf(&x[row], &x[col], length_scale);
            }
            kernel[row][row] += GP_NOISE;
        }
        let alpha = solve_spd(runtime, kernel, y)?.unwrap_or_else(|| vec![0.0; x.len()]);
        Ok(Self {
            x,
            y_mean,
            y_scale,
            alpha,
            length_scale,
        })
    }

    pub(super) fn predict(&self, point: &[f64]) -> (f64, f64) {
        if self.x.is_empty() {
            return (0.0, 1.0);
        }
        let k = self
            .x
            .iter()
            .map(|x| rbf(x, point, self.length_scale))
            .collect::<Vec<_>>();
        let normalized_mean = k
            .iter()
            .zip(&self.alpha)
            .map(|(ki, alpha)| ki * alpha)
            .sum::<f64>();
        let nearest = self
            .x
            .iter()
            .map(|x| squared_distance(x, point))
            .fold(f64::INFINITY, f64::min);
        let variance = (1.0 - (-nearest / (2.0 * self.length_scale * self.length_scale)).exp())
            .clamp(1.0e-9, 1.0);
        (
            self.y_mean + normalized_mean * self.y_scale,
            variance * self.y_scale * self.y_scale,
        )
    }
}

fn rbf(a: &[f64], b: &[f64], length_scale: f64) -> f64 {
    (-squared_distance(a, b) / (2.0 * length_scale * length_scale)).exp()
}

pub(super) fn squared_distance(a: &[f64], b: &[f64]) -> f64 {
    a.iter()
        .zip(b)
        .map(|(ai, bi)| (ai - bi).powi(2))
        .sum::<f64>()
}

fn solve_spd(
    runtime: Option<&crate::context::RuntimeContext>,
    mut matrix: Vec<Vec<f64>>,
    mut rhs: Vec<f64>,
) -> BuiltinResult<Option<Vec<f64>>> {
    let n = rhs.len();
    if n == 0 {
        return Ok(Some(Vec::new()));
    }
    for i in 0..n {
        check_cancellation(runtime)?;
        let pivot = matrix[i][i];
        if !pivot.is_finite() || pivot.abs() < 1.0e-14 {
            return Ok(None);
        }
        for j in (i + 1)..n {
            let factor = matrix[j][i] / pivot;
            for k in i..n {
                matrix[j][k] -= factor * matrix[i][k];
            }
            rhs[j] -= factor * rhs[i];
        }
    }
    let mut x = vec![0.0; n];
    for i in (0..n).rev() {
        let mut acc = rhs[i];
        for j in (i + 1)..n {
            acc -= matrix[i][j] * x[j];
        }
        x[i] = acc / matrix[i][i];
    }
    Ok(Some(x))
}

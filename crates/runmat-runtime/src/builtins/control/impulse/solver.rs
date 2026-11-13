//! Real SISO impulse-response realization and simulation.
use super::*;
use nalgebra::DMatrix;
use runmat_value::Tensor;

#[derive(Clone, Debug)]
pub(super) struct ImpulseResponse {
    t: Vec<f64>,
    y: Vec<f64>,
    #[cfg(feature = "plot-core")]
    pub(super) discrete: bool,
}

impl ImpulseResponse {
    pub(super) fn y_value(&self) -> BuiltinResult<Value> {
        let tensor = Tensor::new(self.y.clone(), vec![self.y.len(), 1]).map_err(|err| {
            impulse_error_with_detail(
                &IMPULSE_ERROR_INTERNAL,
                format!("failed to build response tensor: {err}"),
            )
        })?;
        Ok(Value::Tensor(tensor))
    }

    pub(super) fn t_value(&self) -> BuiltinResult<Value> {
        let tensor = Tensor::new(self.t.clone(), vec![self.t.len(), 1]).map_err(|err| {
            impulse_error_with_detail(
                &IMPULSE_ERROR_INTERNAL,
                format!("failed to build time tensor: {err}"),
            )
        })?;
        Ok(Value::Tensor(tensor))
    }
}

pub(super) fn evaluate_impulse(
    system: &TfSystem,
    time: TimeSpec,
) -> BuiltinResult<ImpulseResponse> {
    let TimeSpec::Values(t) = time;
    let realization = Realization::from_tf(system)?;
    let y = if system.is_discrete() {
        discrete_response(system, &realization, &t)?
    } else {
        continuous_response(&realization, &t)
    };
    Ok(ImpulseResponse {
        t,
        y,
        #[cfg(feature = "plot-core")]
        discrete: system.is_discrete(),
    })
}

#[derive(Clone, Debug)]
struct Realization {
    a: DMatrix<f64>,
    c: Vec<f64>,
}

impl Realization {
    fn from_tf(system: &TfSystem) -> BuiltinResult<Self> {
        if system.numerator.is_empty() {
            let order = system.denominator.len().saturating_sub(1).max(1);
            return Ok(Self {
                a: DMatrix::zeros(order, order),
                c: vec![0.0; order],
            });
        }
        let (mut numerator, denominator) = system.normalized()?;
        let order = denominator.len() - 1;
        while numerator.len() < order {
            numerator.insert(0, 0.0);
        }

        let mut a = DMatrix::<f64>::zeros(order, order);
        for row in 0..order.saturating_sub(1) {
            a[(row, row + 1)] = 1.0;
        }
        for col in 0..order {
            a[(order - 1, col)] = -denominator[order - col];
        }
        let c = numerator.into_iter().rev().collect();
        Ok(Self { a, c })
    }
}

fn continuous_response(realization: &Realization, t: &[f64]) -> Vec<f64> {
    t.iter()
        .map(|&time| {
            let exp_at = matrix_exp(&(realization.a.clone() * time));
            dot_c_with_last_column(&realization.c, &exp_at)
        })
        .collect()
}

fn discrete_response(
    system: &TfSystem,
    realization: &Realization,
    t: &[f64],
) -> BuiltinResult<Vec<f64>> {
    if t.len() > MAX_DISCRETE_SAMPLES {
        return Err(impulse_error_with_detail(
            &IMPULSE_ERROR_DISCRETE_LIMIT,
            format!("discrete response would require more than {MAX_DISCRETE_SAMPLES} samples"),
        ));
    }
    let sample_indices: Vec<usize> = t
        .iter()
        .map(|value| checked_discrete_sample_index(system, *value))
        .collect::<BuiltinResult<_>>()?;
    let max_index = sample_indices.iter().copied().max().unwrap_or(0);
    let order = realization.c.len();
    let value_count = max_index.checked_add(1).ok_or_else(|| {
        impulse_error_with_detail(
            &IMPULSE_ERROR_DISCRETE_LIMIT,
            "discrete sample index exceeds platform limits",
        )
    })?;
    let mut values = vec![0.0; value_count];
    if order == 0 {
        return Ok(sample_indices.into_iter().map(|idx| values[idx]).collect());
    }

    let mut state = vec![0.0; order];
    state[order - 1] = 1.0;
    let impulse_scale = 1.0 / system.sample_time;
    for k in 1..=max_index {
        values[k] = dot(&realization.c, &state) * impulse_scale;
        state = mat_vec_mul(&realization.a, &state);
    }
    Ok(sample_indices.into_iter().map(|idx| values[idx]).collect())
}

fn dot_c_with_last_column(c: &[f64], matrix: &DMatrix<f64>) -> f64 {
    if c.is_empty() {
        return 0.0;
    }
    let last_col = matrix.ncols() - 1;
    c.iter()
        .enumerate()
        .map(|(row, coeff)| coeff * matrix[(row, last_col)])
        .sum()
}

fn dot(lhs: &[f64], rhs: &[f64]) -> f64 {
    lhs.iter().zip(rhs).map(|(a, b)| a * b).sum()
}

fn mat_vec_mul(matrix: &DMatrix<f64>, vector: &[f64]) -> Vec<f64> {
    let mut out = vec![0.0; matrix.nrows()];
    for row in 0..matrix.nrows() {
        let mut acc = 0.0;
        for col in 0..matrix.ncols() {
            acc += matrix[(row, col)] * vector[col];
        }
        out[row] = acc;
    }
    out
}

fn matrix_exp(matrix: &DMatrix<f64>) -> DMatrix<f64> {
    let norm = matrix_one_norm(matrix);
    let scale_power = if norm <= 0.5 {
        0usize
    } else {
        norm.log2().ceil().max(0.0) as usize + 1
    };
    let scale = 2.0_f64.powi(scale_power as i32);
    let scaled = matrix / scale;
    let n = matrix.nrows();
    let mut result = DMatrix::<f64>::identity(n, n);
    let mut term = DMatrix::<f64>::identity(n, n);
    for k in 1..=48 {
        term = (&term * &scaled) / (k as f64);
        result += &term;
        if matrix_one_norm(&term) <= 1.0e-14 {
            break;
        }
    }
    for _ in 0..scale_power {
        result = &result * &result;
    }
    result
}

fn matrix_one_norm(matrix: &DMatrix<f64>) -> f64 {
    let mut best = 0.0;
    for col in 0..matrix.ncols() {
        let mut sum = 0.0;
        for row in 0..matrix.nrows() {
            sum += matrix[(row, col)].abs();
        }
        if sum > best {
            best = sum;
        }
    }
    best
}

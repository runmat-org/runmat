//! Steihaug's preconditioned trust-ball iteration. No inverse or dense
//! Hessian is constructed, including for an opaque Hessian product callback.
use super::*;

pub(crate) struct Preconditioner {
    order: usize,
    bandwidth: usize,
    lower: Vec<f64>,
}

impl Preconditioner {
    pub(crate) async fn new(
        model: &mut impl Operator,
        n: usize,
        bandwidth: usize,
        cooperation: &mut ScalarCooperation,
    ) -> BuiltinResult<Option<Self>> {
        let policy = model.policy();
        let bandwidth = bandwidth.min(n.saturating_sub(1));
        let stride = bandwidth
            .checked_add(1)
            .ok_or_else(|| policy.invalid("preconditioner extent overflow"))?;
        let length = n
            .checked_mul(stride)
            .ok_or_else(|| policy.invalid("preconditioner extent overflow"))?;
        let mut lower = policy.zeros(length, cooperation).await?;
        let copied = match model
            .copy_band(n, bandwidth, &mut lower, cooperation)
            .await?
        {
            Materialization::Complete => true,
            Materialization::Unavailable => false,
            Materialization::Undefined => return Ok(None),
        };
        if !copied {
            let mut unit = policy.zeros(n, cooperation).await?;
            let mut product = policy.zeros(n, cooperation).await?;
            // Only the explicitly requested lower band is retained. Temporary
            // product and basis vectors remain O(n), even for Hinfo callbacks.
            for column in 0..n {
                unit[column] = 1.0;
                if model
                    .apply(&unit, &mut product, cooperation)
                    .await?
                    .is_none()
                {
                    return Ok(None);
                }
                unit[column] = 0.0;
                for row in column..n.min(column.saturating_add(stride)) {
                    lower[row * stride + row - column] = product[row];
                }
                cooperation.completed(stride as u64).await?;
            }
        }
        // Gershgorin diagonal boosts make the requested symmetric band positive
        // definite before its band Cholesky factorization. This is solely a
        // preconditioner: the trust model still uses the original operator.
        for row in 0..n {
            let mut off = 0.0;
            for column in row.saturating_sub(bandwidth)..row {
                off += lower[row * stride + row - column].abs();
            }
            for column in row + 1..n.min(row.saturating_add(stride)) {
                off += lower[column * stride + column - row].abs();
            }
            if !off.is_finite() {
                return Ok(None);
            }
            let diagonal = lower[row * stride];
            if diagonal <= off {
                // Off-diagonal scale matters. Do not perturb already positive
                // diagonally dominant rows using another row's huge scale.
                let scale = diagonal.abs().max(off);
                let margin = if scale == 0.0 {
                    1.0
                } else {
                    let ulp = scale - f64::from_bits(scale.to_bits() - 1);
                    (f64::EPSILON.sqrt() * scale).max(ulp)
                };
                let boosted = off + margin;
                if !(boosted > off && boosted.is_finite()) {
                    return Ok(None);
                }
                lower[row * stride] = boosted;
            }
            cooperation.completed(stride as u64).await?;
        }
        for row in 0..n {
            for column in row.saturating_sub(bandwidth)..=row {
                let mut value = lower[row * stride + row - column];
                let first = row
                    .saturating_sub(bandwidth)
                    .max(column.saturating_sub(bandwidth));
                for k in first..column {
                    value -= lower[row * stride + row - k] * lower[column * stride + column - k];
                }
                if column == row {
                    if !(value > 0.0 && value.is_finite()) {
                        return Ok(None);
                    }
                    lower[row * stride] = value.sqrt();
                } else {
                    value /= lower[column * stride];
                    if !value.is_finite() {
                        return Ok(None);
                    }
                    lower[row * stride + row - column] = value;
                }
                cooperation.completed((column - first + 1) as u64).await?;
            }
        }
        Ok(Some(Self {
            order: n,
            bandwidth,
            lower,
        }))
    }

    async fn apply(
        &self,
        residual: &[f64],
        output: &mut [f64],
        cooperation: &mut ScalarCooperation,
    ) -> BuiltinResult<bool> {
        let stride = self.bandwidth + 1;
        for row in 0..self.order {
            let mut value = residual[row];
            for column in row.saturating_sub(self.bandwidth)..row {
                value -= self.lower[row * stride + row - column] * output[column];
            }
            output[row] = value / self.lower[row * stride];
            cooperation.completed(stride as u64).await?;
        }
        for row in (0..self.order).rev() {
            let mut value = output[row];
            for column in row + 1..self.order.min(row.saturating_add(stride)) {
                value -= self.lower[column * stride + column - row] * output[column];
            }
            output[row] = value / self.lower[row * stride];
            if !output[row].is_finite() {
                return Ok(false);
            }
            cooperation.completed(stride as u64).await?;
        }
        Ok(true)
    }
}

pub(crate) struct Step {
    pub displacement: Vec<f64>,
    pub boundary: bool,
    pub iterations: usize,
}
pub(crate) struct IterationControl<'a> {
    pub maximum: usize,
    pub tolerance: f64,
    pub completed: Option<&'a mut usize>,
}

pub(crate) fn step<'a>(
    model: &'a mut impl Operator,
    preconditioner: &'a Preconditioner,
    gradient: &'a [f64],
    radius: f64,
    maximum_cg_iterations: usize,
    cg_tolerance: f64,
    cooperation: &'a mut ScalarCooperation,
) -> impl std::future::Future<Output = BuiltinResult<Option<Step>>> + 'a {
    step_with_control(
        model,
        preconditioner,
        gradient,
        radius,
        IterationControl {
            maximum: maximum_cg_iterations,
            tolerance: cg_tolerance,
            completed: None,
        },
        cooperation,
    )
}

/// Reports attempted iterations even when the numerical step is unavailable.
pub(crate) async fn step_with_control(
    model: &mut impl Operator,
    preconditioner: &Preconditioner,
    gradient: &[f64],
    radius: f64,
    control: IterationControl<'_>,
    cooperation: &mut ScalarCooperation,
) -> BuiltinResult<Option<Step>> {
    let IterationControl {
        maximum: maximum_cg_iterations,
        tolerance: cg_tolerance,
        completed: mut iterations,
    } = control;
    if let Some(count) = iterations.as_deref_mut() {
        *count = 0;
    }
    let policy = model.policy();
    let n = gradient.len();
    let mut displacement = policy.zeros(n, cooperation).await?;
    let mut residual = policy.buffer(n)?;
    residual.extend_from_slice(gradient);
    let mut z = policy.zeros(n, cooperation).await?;
    if !preconditioner.apply(&residual, &mut z, cooperation).await? {
        return Ok(None);
    }
    if model.project(&mut z, cooperation).await?.is_none() {
        return Ok(None);
    }
    let mut direction = policy.buffer(n)?;
    direction.extend(z.iter().map(|value| -*value));
    let mut rz = dot(&residual, &z);
    let initial_norm = norm(gradient);
    let mut product = policy.zeros(n, cooperation).await?;
    if !rz.is_finite() || rz <= 0.0 {
        return Ok(None);
    }
    for iteration in 1..=maximum_cg_iterations {
        if let Some(count) = iterations.as_deref_mut() {
            *count = iteration;
        }
        if model
            .apply(&direction, &mut product, cooperation)
            .await?
            .is_none()
        {
            return Ok(None);
        }
        let curvature = dot(&direction, &product);
        if !curvature.is_finite() {
            return Ok(None);
        }
        if curvature <= 0.0 {
            if !to_boundary(&mut displacement, &direction, radius) {
                return Ok(None);
            }
            return Ok(Some(Step {
                displacement,
                boundary: true,
                iterations: iteration,
            }));
        }
        let alpha = rz / curvature;
        if !(alpha > 0.0 && alpha.is_finite()) {
            return Ok(None);
        }
        let candidate_norm = displacement
            .iter()
            .zip(&direction)
            .fold(0.0f64, |length, (&p, &d)| length.hypot(p + alpha * d));
        if candidate_norm >= radius {
            if !to_boundary(&mut displacement, &direction, radius) {
                return Ok(None);
            }
            return Ok(Some(Step {
                displacement,
                boundary: true,
                iterations: iteration,
            }));
        }
        for index in 0..n {
            displacement[index] += alpha * direction[index];
            residual[index] += alpha * product[index];
        }
        cooperation.completed(n as u64).await?;
        if norm(&residual) <= cg_tolerance * initial_norm {
            return Ok(Some(Step {
                displacement,
                boundary: false,
                iterations: iteration,
            }));
        }
        if !preconditioner.apply(&residual, &mut z, cooperation).await? {
            return Ok(None);
        }
        if model.project(&mut z, cooperation).await?.is_none() {
            return Ok(None);
        }
        let next_rz = dot(&residual, &z);
        if !(next_rz > 0.0 && next_rz.is_finite()) {
            return Ok(None);
        }
        let beta = next_rz / rz;
        for index in 0..n {
            direction[index] = -z[index] + beta * direction[index];
        }
        rz = next_rz;
        cooperation.completed(n as u64).await?;
    }
    Ok(Some(Step {
        displacement,
        boundary: false,
        iterations: maximum_cg_iterations,
    }))
}

// Scale the equation by the radius and direction norm before solving its
// positive root, avoiding squared physical coordinates in the intersection.
fn to_boundary(displacement: &mut [f64], direction: &[f64], radius: f64) -> bool {
    let direction_norm = norm(direction);
    if !(direction_norm > 0.0 && direction_norm.is_finite() && radius > 0.0) {
        return false;
    }
    let b = displacement
        .iter()
        .zip(direction)
        .map(|(&p, &d)| (p / radius) * (d / direction_norm))
        .sum::<f64>();
    let c = displacement
        .iter()
        .map(|&p| (p / radius).powi(2))
        .sum::<f64>()
        - 1.0;
    let root = b.hypot((-c).max(0.0).sqrt());
    let tau = if b > 0.0 { -c / (b + root) } else { root - b };
    for (p, &d) in displacement.iter_mut().zip(direction) {
        *p += radius * tau * (d / direction_norm);
    }
    displacement.iter().all(|p| p.is_finite())
}

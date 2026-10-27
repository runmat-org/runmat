//! The explicit direct subproblem uses the existing self-adjoint EVD kernel
//! and a secular solve, including the indefinite hard case. Explicit direct
//! mode may assemble an opaque operator's basis products; CG never does.
use super::{cg::Step, *};
use crate::builtins::math::linalg::ops::dense_evd::kernel::{self, Kind};
use num_complex::Complex64;

pub(crate) struct Factorization {
    eigenvalues: Vec<Complex64>,
    vectors: Vec<Complex64>,
}

fn add_compensated(total: &mut f64, correction: &mut f64, value: f64) -> Option<()> {
    if !value.is_finite() {
        return None;
    }
    let next = *total + value;
    if !next.is_finite() {
        return None;
    }
    *correction += if total.abs() >= value.abs() {
        (*total - next) + value
    } else {
        (value - next) + *total
    };
    if !correction.is_finite() {
        return None;
    }
    *total = next;
    Some(())
}

/// Recover curvature against the unchanged direct input. A normwise stable
/// EVD can lose a tiny eigenvalue to subtraction (including diagonal 2x2
/// inputs); that value may still decide the trust-ball forcing/root branch.
fn refine_values(
    matrix: &[Complex64],
    values: &mut [Complex64],
    vectors: &[Complex64],
) -> Option<()> {
    let n = values.len();
    if n == 0 {
        return Some(());
    }
    for (value, vector) in values.iter_mut().zip(vectors.chunks_exact(n)) {
        let (mut squared_norm, mut norm_correction) = (0.0, 0.0);
        for q in vector {
            add_compensated(&mut squared_norm, &mut norm_correction, q.norm_sqr())?;
        }
        let squared_norm = squared_norm + norm_correction;
        if !(squared_norm > 0.0 && squared_norm.is_finite()) {
            return None;
        }
        let (mut numerator, mut correction) = (0.0, 0.0);
        // Input symmetry and real class were proved before this job. Pairing
        // conjugate off-diagonal terms gives 2*Re(conj(q_i)*q_j): arbitrary
        // eigenvector phases are preserved, not discarded as imaginary noise.
        for column in 0..n {
            add_compensated(
                &mut numerator,
                &mut correction,
                matrix[column * n + column].re * vector[column].norm_sqr(),
            )?;
            for row in 0..column {
                let weight = 2.0 * (vector[row].conj() * vector[column]).re;
                add_compensated(
                    &mut numerator,
                    &mut correction,
                    matrix[column * n + row].re * weight,
                )?;
            }
        }
        let quotient = (numerator + correction) / squared_norm;
        if !quotient.is_finite() {
            return None;
        }
        *value = Complex64::new(quotient, 0.0);
    }
    Some(())
}

#[test]
fn original_rayleigh_curvature_preserves_complex_phase_normalization_and_pair_order() {
    let matrix = [
        Complex64::new(1e100, 0.0),
        Complex64::default(),
        Complex64::default(),
        Complex64::new(1e-100, 0.0),
    ];
    let vectors = [
        Complex64::new(0.0, 2.0),
        Complex64::default(),
        Complex64::default(),
        Complex64::new(0.0, -3.0),
    ];
    let mut values = [Complex64::default(); 2];
    assert!(refine_values(&matrix, &mut values, &vectors).is_some());
    assert_eq!(values[0], matrix[0]);
    assert!((values[1].re / 1e-100 - 1.0).abs() <= 4.0 * f64::EPSILON);
    assert_eq!(values[1].im, 0.0);
    assert!(values[1].re<values[0].re,"associated eigenvectors stay in place; consumers find the actual corrected minimum rather than assuming old sorted order");
}

impl Factorization {
    pub(crate) async fn new(
        model: &mut impl Operator,
        n: usize,
        cooperation: &mut ScalarCooperation,
    ) -> BuiltinResult<Option<Self>> {
        let policy = model.policy();
        let length = n
            .checked_mul(n)
            .ok_or_else(|| policy.invalid("direct trust-region matrix extent overflow"))?;
        let mut matrix = policy.buffer(length)?;
        let copied = match model.copy_matrix(n, &mut matrix, cooperation).await? {
            Materialization::Complete => true,
            Materialization::Unavailable => false,
            Materialization::Undefined => return Ok(None),
        };
        if !copied {
            let mut unit = policy.zeros(n, cooperation).await?;
            let mut product = policy.zeros(n, cooperation).await?;
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
                for chunk in product.chunks(1024) {
                    matrix.extend(chunk.iter().map(|&value| Complex64::new(value, 0.0)));
                    cooperation.completed(chunk.len() as u64).await?;
                }
            }
        }
        for column in 0..n {
            for row in column + 1..n {
                if matrix[column * n + row] != matrix[row * n + column] {
                    return Err(policy
                        .invalid("direct Hessian products must describe a symmetric operator"));
                }
                cooperation.completed(1).await?;
            }
        }
        let mut eigenvalues = policy.buffer(n)?;
        eigenvalues.resize(n, Complex64::default());
        let mut vectors = policy.buffer(length)?;
        while vectors.len() < length {
            let end = vectors.len().saturating_add(1024).min(length);
            let count = end - vectors.len();
            vectors.resize(end, Complex64::default());
            cooperation.completed(count as u64).await?;
        }
        let requirement = kernel::requirement(n, Kind::SelfAdjoint, false);
        let mut scratch = faer::dyn_stack::MemBuffer::try_new(requirement).map_err(|_| {
            (policy.allocation)(runmat_value::ExactHostNumericAllocationError::AllocationFailed)
        })?;
        let work = move || {
            let result = kernel::compute(
                n,
                Kind::SelfAdjoint,
                &matrix,
                &mut eigenvalues,
                &mut vectors,
                None,
                &mut scratch,
            );
            // The already-owned input is moved, never cloned or retained
            // after this job. This refinement adds no callback evaluations.
            let result = result
                .ok()
                .and_then(|()| refine_values(&matrix, &mut eigenvalues, &vectors));
            (eigenvalues, vectors, result)
        };
        // Follow the existing dense-EVD owned CPU port. Workspace is moved
        // once and stays alive if the awaiting continuation is cancelled.
        let (runtime, budget) = cooperation.runtime_budget();
        let (eigenvalues, mut vectors, result) = match (runtime, budget) {
            (Some(runtime), Some(budget))
                if runtime.service_ports().blocking_cpu().is_some()
                    || runtime.execution().cpu_cooperation_policy()
                        != crate::execution::CpuCooperationPolicy::NeverYield =>
            {
                budget
                    .run_owned_blocking_in_context(
                        runtime,
                        runtime
                            .service_ports()
                            .blocking_cpu()
                            .map(|service| service.as_ref()),
                        (),
                        work,
                    )
                    .await
                    .map_err(|source| source.into_runtime_error_in_context(Some(runtime)))?
            }
            _ => work(),
        };
        cooperation.completed(0).await?;
        if result.is_none()
            || eigenvalues.iter().any(|v| !v.re.is_finite() || v.im != 0.0)
            || vectors
                .iter()
                .any(|v| !v.re.is_finite() || !v.im.is_finite())
        {
            return Ok(None);
        }
        // The shared kernel is complex-valued. A real symmetric matrix can
        // acquire a harmless eigenvector phase. Remove that phase, not an
        // arbitrary imaginary part, before the real trust subproblem.
        for vector in vectors.chunks_exact_mut(n) {
            let Some(pivot) = vector
                .iter()
                .max_by(|a, b| a.norm().total_cmp(&b.norm()))
                .copied()
            else {
                return Ok(None);
            };
            let length = pivot.norm();
            if !(length > 0.0 && length.is_finite()) {
                return Ok(None);
            }
            let phase = pivot.conj() / length;
            for value in vector.iter_mut() {
                *value *= phase;
                if value.im.abs() > 128.0 * f64::EPSILON {
                    return Ok(None);
                }
                value.im = 0.0;
            }
            cooperation.completed(n as u64).await?;
        }
        Ok(Some(Self {
            eigenvalues,
            vectors,
        }))
    }

    pub(crate) async fn step(
        &self,
        model: &mut impl Operator,
        gradient: &[f64],
        radius: f64,
        cooperation: &mut ScalarCooperation,
    ) -> BuiltinResult<Option<Step>> {
        let policy = model.policy();
        let n = gradient.len();
        let mut projected = policy.buffer(n)?;
        for vector in self.vectors.chunks_exact(n) {
            projected.push(
                vector
                    .iter()
                    .zip(gradient)
                    .map(|(q, &g)| q.re * g)
                    .sum::<f64>(),
            );
            cooperation.completed(n as u64).await?;
        }
        let Some((minimum_index, minimum)) = self
            .eigenvalues
            .iter()
            .enumerate()
            .min_by(|(_, a), (_, b)| a.re.total_cmp(&b.re))
        else {
            return Ok(None);
        };
        let lower = (-minimum.re).max(0.0);
        let gradient_norm = norm(gradient);
        let mut coefficients = policy.zeros(n, cooperation).await?;
        let mut singular_forcing = false;
        for i in 0..n {
            let denominator = self.eigenvalues[i].re + lower;
            if denominator == 0.0 {
                // Another coordinate's large gradient cannot erase forcing
                // in this null mode. The original residual/boundary checks
                // below still decide whether a numerical candidate is valid.
                singular_forcing |= projected[i] != 0.0;
            } else {
                coefficients[i] = -projected[i] / denominator;
            }
        }
        let base_norm = norm(&coefficients);
        let boundary;
        let multiplier;
        let residual_tolerance = 128.0 * f64::EPSILON * n.max(1) as f64;
        if !singular_forcing && base_norm <= radius {
            boundary = lower > 0.0;
            multiplier = lower;
            if boundary {
                // Hard case: g is orthogonal to a minimum-eigenvalue
                // eigenspace, so add its missing boundary component.
                let relative = base_norm / radius;
                let missing = radius * ((1.0 - relative) * (1.0 + relative)).max(0.0).sqrt();
                coefficients[minimum_index] = if projected[minimum_index] > 0.0 {
                    -missing
                } else {
                    missing
                };
            }
        } else {
            boundary = true;
            let mut lo = lower;
            let mut hi = lower + gradient_norm / radius;
            if !(hi > lo && hi.is_finite()) {
                return Ok(None);
            }
            // This bracket follows ||(H+lambda I)^-1 g|| <= ||g|| /
            // (lambda + lambda_min), not an arbitrary expansion cap.
            // Arithmetic bisection strictly reduces a finite representable
            // interval. It may require hundreds of steps for a wide spectrum;
            // an unvalidated fixed iteration count is not a boundary solution.
            loop {
                let lambda = lo * 0.5 + hi * 0.5;
                if lambda == lo || lambda == hi {
                    break;
                }
                let length = projected
                    .iter()
                    .zip(&self.eigenvalues)
                    .fold(0.0f64, |length, (&g, d)| length.hypot(g / (d.re + lambda)));
                if length.is_nan() {
                    return Ok(None);
                }
                if length > radius {
                    lo = lambda;
                } else {
                    hi = lambda;
                }
                cooperation.completed(n as u64).await?;
                if length <= radius && (1.0 - length / radius).abs() <= residual_tolerance {
                    break;
                }
            }
            multiplier = hi;
            for i in 0..n {
                coefficients[i] = -projected[i] / (self.eigenvalues[i].re + hi);
            }
        }
        let mut displacement = policy.zeros(n, cooperation).await?;
        for (column, &coefficient) in self.vectors.chunks_exact(n).zip(&coefficients) {
            for (to, value) in displacement.iter_mut().zip(column) {
                *to += coefficient * value.re;
            }
            cooperation.completed(n as u64).await?;
        }
        if displacement.iter().any(|value| !value.is_finite()) {
            return Ok(None);
        }
        let length = norm(&displacement);
        if boundary && (length / radius - 1.0).abs() > residual_tolerance
            || length > radius * (1.0 + residual_tolerance)
        {
            return Ok(None);
        }
        // Certify stationarity in original model units, not just in the
        // computed spectral coordinates. Callback/ownership errors propagate;
        // a numerical residual failure does not manufacture convergence.
        let mut product = policy.zeros(n, cooperation).await?;
        if model
            .apply(&displacement, &mut product, cooperation)
            .await?
            .is_none()
        {
            return Ok(None);
        }
        for i in 0..n {
            let shifted = multiplier * displacement[i];
            if !shifted.is_finite() {
                return Ok(None);
            }
            let scale = gradient[i].abs().max(product[i].abs()).max(shifted.abs());
            if scale > 0.0
                && (gradient[i] / scale + product[i] / scale + shifted / scale).abs()
                    > residual_tolerance
            {
                return Ok(None);
            }
            cooperation.completed(1).await?;
        }
        Ok(Some(Step {
            displacement,
            boundary,
            iterations: 0,
        }))
    }
}

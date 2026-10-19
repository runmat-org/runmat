//! Publication gate shared by standard continuous and discrete Riccati calls.
//! A finite algebraic root is not enough: the selected feedback must also be
//! stabilizing and satisfy the original, untransformed equation.

use super::{closed_loop_poles, max_abs_matrix, LqrDomain, LqrEval, RiccatiApi, RiccatiIssue};
use crate::builtins::math::linalg::factor::qz::riccati::RiccatiFactorization;
use crate::BuiltinResult;
use num_complex::Complex64;

pub(super) const RESIDUAL_TOLERANCE: f64 = 1.0e-7;

pub(super) fn validate_solution(
    factor: RiccatiFactorization,
    domain: LqrDomain,
    api: RiccatiApi,
) -> BuiltinResult<LqrEval> {
    let RiccatiFactorization {
        a,
        b,
        q,
        r,
        cross: n,
        solution: s,
    } = factor;
    if !s.iter().all(|value| value.is_finite()) {
        return Err(failed(api, "Riccati solution is nonfinite"));
    }
    let (mut denominator, mut k) = match domain {
        LqrDomain::Continuous => (r, b.transpose() * &s + n.transpose()),
        LqrDomain::Discrete => {
            let denominator = &r + b.transpose() * &s * &b;
            (denominator, b.transpose() * &s * &a + n.transpose())
        }
    };
    let m = k.nrows();
    let states = k.ncols();
    super::gain::solve_in_place(
        denominator.as_mut_slice(),
        k.as_mut_slice(),
        m,
        states,
        api,
        || Ok(()),
    )?;
    if !k.iter().all(|value| value.is_finite()) {
        return Err(failed(api, "Riccati feedback gain is nonfinite"));
    }
    let (residual, scale) = match domain {
        LqrDomain::Continuous => {
            let left = a.transpose() * &s;
            let right = &s * &a;
            let control = (&s * &b + &n) * &k;
            let scale = [
                max_abs_matrix(&left),
                max_abs_matrix(&right),
                max_abs_matrix(&control),
                max_abs_matrix(&q),
            ]
            .into_iter()
            .fold(0.0_f64, f64::max);
            (left + right - control + &q, scale)
        }
        LqrDomain::Discrete => {
            let evolution = a.transpose() * &s * &a;
            let control = (a.transpose() * &s * &b + &n) * &k;
            let scale = [
                max_abs_matrix(&evolution),
                max_abs_matrix(&s),
                max_abs_matrix(&control),
                max_abs_matrix(&q),
            ]
            .into_iter()
            .fold(0.0_f64, f64::max);
            (evolution - &s - control + &q, scale)
        }
    };
    let residual_norm = max_abs_matrix(&residual);
    if !scale.is_finite()
        || !residual_norm.is_finite()
        || residual_norm > RESIDUAL_TOLERANCE * scale
    {
        return Err(failed(api, "Riccati residual is nonfinite or too large"));
    }
    let closed_loop = &a - &b * &k;
    if !closed_loop.iter().all(|value| value.is_finite()) {
        return Err(failed(api, "closed-loop matrix is nonfinite"));
    }
    let poles = closed_loop_poles(&closed_loop, api)?;
    if poles.len() != a.nrows() || !poles.iter().all(|pole| stable(*pole, domain)) {
        return Err(failed(
            api,
            "Riccati solution does not stabilize the closed-loop system",
        ));
    }
    Ok(LqrEval {
        k,
        s,
        poles: Some(poles),
        api,
    })
}

pub(super) fn stable(pole: Complex64, domain: LqrDomain) -> bool {
    if !pole.re.is_finite() || !pole.im.is_finite() {
        return false;
    }
    let margin = 64.0 * f64::EPSILON;
    match domain {
        LqrDomain::Continuous => pole.re < -margin * pole.norm().max(1.0),
        LqrDomain::Discrete => pole.norm() < 1.0 - margin,
    }
}

fn failed(api: RiccatiApi, message: &'static str) -> crate::RuntimeError {
    api.error(RiccatiIssue::SolverFailed, message)
}

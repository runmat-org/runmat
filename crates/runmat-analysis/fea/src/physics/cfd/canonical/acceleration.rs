//! Shared Anderson proposal; original physical equations decide whether to retain it.
//! Only pressure residuals determine the scalar coefficient (no mixed-unit norm).
//! The same affine combination acts on U/p/q; no source/reference solution is used.
use super::{geometry_ops::V, recovery, CanonicalFluidError as Error, CanonicalFluidProblem};
use crate::solve::fixed_point::{propose, FixedPointError};
use runmat_simulation_usage::{LedgerError, OperationCount, UsageCategory};

pub(super) struct Fields<'a> {
    pub velocity: &'a mut Vec<V>,
    pub pressure: &'a mut Vec<f64>,
    pub flux: &'a mut Vec<f64>,
}

pub(super) struct History {
    velocity: Vec<V>,
    pressure: Vec<f64>,
    flux: Vec<f64>,
    pressure_defect: Vec<f64>,
}

pub(super) fn consider(
    p: &CanonicalFluidProblem<'_>,
    fields: Fields<'_>,
    pressure_defect: Vec<f64>,
    history: &mut Option<History>,
    original: recovery::CanonicalFluidResidual,
    time: Option<super::transient::TimeDerivative<'_>>,
) -> Result<recovery::CanonicalFluidResidual, Error> {
    let Some(policy) = p.policy.fixed_point_acceleration else {
        return Ok(original);
    };
    if original.accepted(p.policy.physical_relative_tolerance) {
        return Ok(original);
    }
    let current = History {
        velocity: fields.velocity.clone(),
        pressure: fields.pressure.clone(),
        flux: fields.flux.clone(),
        pressure_defect,
    };
    let previous = history.replace(current);
    let Some(previous) = previous else {
        return Ok(original);
    };
    let current = history.as_ref().unwrap();
    let candidate = match propose(
        &previous.pressure,
        &previous.pressure_defect,
        &current.pressure,
        &current.pressure_defect,
        policy,
        crate::is_fea_cancelled(),
    ) {
        Ok(candidate) => candidate,
        // Overflow in speculative coefficient arithmetic must not invalidate
        // the already-evaluated ordinary image. Discard this unusable history.
        Err(FixedPointError::NonfiniteArithmetic) => {
            *history = None;
            return Ok(original);
        }
        Err(FixedPointError::Accounting(cause)) => return Err(Error::Accounting(cause)),
        Err(FixedPointError::Cancelled) => return Err(Error::Cancelled),
        Err(error) => return Err(Error::Operator(error.to_string())),
    };
    let Some(mut candidate) = candidate else {
        return Ok(original);
    };
    match p.pressure_reference_plan.normalize(p, &mut candidate.image) {
        Ok(()) => {}
        Err(error @ Error::Accounting(_)) => return Err(error),
        Err(_) => return Ok(original),
    }
    let gamma = candidate.coefficient;
    // U has three scalar components. Each component of U and q uses one
    // difference, one product and one subtraction. The existing zip semantics
    // select the exact immutable number of paired entries in this batch.
    let velocity_work = OperationCount::from_operations(
        u64::try_from(current.velocity.len().min(previous.velocity.len()))
            .map_err(|_| LedgerError::CountOverflow)?,
    )
    .checked_mul(9)
    .map_err(LedgerError::from)?;
    let flux_work = OperationCount::from_operations(
        u64::try_from(current.flux.len().min(previous.flux.len()))
            .map_err(|_| LedgerError::CountOverflow)?,
    )
    .checked_mul(3)
    .map_err(LedgerError::from)?;
    let affine_work = velocity_work
        .checked_add(flux_work)
        .map_err(LedgerError::from)?;
    let (velocity, flux) = crate::algebra::capacity::measured_sync(
        UsageCategory::Solve,
        affine_work,
        || -> Result<_, Error> {
            let velocity = current
                .velocity
                .iter()
                .zip(&previous.velocity)
                .map(|(a, b)| std::array::from_fn(|k| a[k] - gamma * (a[k] - b[k])))
                .collect::<Vec<V>>();
            let flux = current
                .flux
                .iter()
                .zip(&previous.flux)
                .map(|(a, b)| a - gamma * (a - b))
                .collect::<Vec<_>>();
            Ok((velocity, flux))
        },
    )?;
    if velocity
        .iter()
        .flatten()
        .chain(&flux)
        .any(|v| !v.is_finite())
    {
        return Ok(original);
    }
    // Backflow, nonfinite geometry actions or any failed original recovery reject
    // this speculative proposal. The already-validated ordinary SIMPLE image remains.
    let trial = match recovery::evaluate_with_time(p, &velocity, &candidate.image, &flux, time) {
        Ok(trial) => trial,
        Err(error @ Error::Accounting(_)) => return Err(error),
        Err(_) => return Ok(original),
    };
    if merit(&trial) >= merit(&original) {
        return Ok(original);
    }
    *fields.velocity = velocity;
    *fields.pressure = candidate.image;
    *fields.flux = flux;
    Ok(trial)
}

fn merit(r: &recovery::CanonicalFluidResidual) -> f64 {
    r.cell_momentum_relative.iter().copied().fold(
        r.momentum_relative
            .max(r.continuity_relative)
            .max(r.flux_constitutive_relative)
            .max(r.global_momentum_relative),
        f64::max,
    )
}

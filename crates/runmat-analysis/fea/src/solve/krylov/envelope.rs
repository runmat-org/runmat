//! Count-only authority for the shared restarted GMRES recurrence. Requested
//! heap excludes borrowed inputs, operator/reduction internals, worker placement
//! and allocator/RSS overhead. No SPD factorization proof is implied.
use super::GmresPolicy;
use std::mem::size_of;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum GmresEnvelopeError {
    InvalidPolicy,
    IterationCounterOverflow,
    ArithmeticOverflow,
    HostAllocationOverflow,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct GmresWorkEnvelope {
    pub dimension: u64,
    pub restart: u64,
    pub maximum_iterations: u32,
    pub basis_vector_count: u64,
    pub hessenberg_scalar_count: u64,
    pub recurrence_scalar_count_upper_bound: u64,
    pub resident_bytes_upper_bound: u64,
    pub operator_applications_upper_bound: u64,
    pub dot_calls_upper_bound: u64,
    pub axpy_calls_upper_bound: u64,
    /// Vector recurrence and projected-system arithmetic; operator/dot callbacks
    /// are charged separately by the owning component.
    pub recurrence_scalar_operations_upper_bound: u64,
}

impl GmresWorkEnvelope {
    pub(crate) fn derive(
        dimension: usize,
        policy: GmresPolicy,
    ) -> Result<Self, GmresEnvelopeError> {
        if dimension == 0
            || policy.restart == 0
            || policy.maximum_iterations == 0
            || !policy.relative_tolerance.is_finite()
            || policy.relative_tolerance <= 0.0
        {
            return Err(GmresEnvelopeError::InvalidPolicy);
        }
        let iterations = u32::try_from(policy.maximum_iterations)
            .map_err(|_| GmresEnvelopeError::IterationCounterOverflow)?;
        let n = count(dimension)?;
        let r = count(policy.restart.min(dimension).min(policy.maximum_iterations))?;
        let basis = add(r, 1)?;
        let hessenberg = mul(basis, r)?;
        // r+1 basis vectors, solution+residual and three concurrent temporary
        // full vectors (operator output, replacement residual, normalization).
        // Hessenberg, rotations, projected RHS and coefficients coexist.
        let scalars = add(
            mul(add(basis, 5)?, n)?,
            add(hessenberg, add(mul(r, 4)?, 1)?)?,
        )?;
        // Growing Vec-of-Vec basis capacity can double. Reserve double payload
        // slots too, covering old/new allocation during growth and scoped vector
        // construction, plus growing headers and fixed tiny allocations.
        let scalar_bytes = mul(mul(scalars, 2)?, size_of::<f64>() as u64)?;
        // Three basis-header capacities cover old+new growth buffers; one
        // additional header per Hessenberg row coexists with those buffers.
        let headers = mul(add(mul(basis, 4)?, 8)?, size_of::<Vec<f64>>() as u64)?;
        let bytes = add(scalar_bytes, headers)?;
        let host_limit = isize::MAX as u64;
        if mul(n, size_of::<f64>() as u64)? > host_limit
            || mul(r, size_of::<f64>() as u64)? > host_limit
            || mul(mul(basis, 2)?, size_of::<Vec<f64>>() as u64)? > host_limit
        {
            return Err(GmresEnvelopeError::HostAllocationOverflow);
        }
        let i = u64::from(iterations);
        // A cycle may stop early, so cycles<=i+1, NOT ceil(i/r). Each cycle
        // can have one unsuccessful Arnoldi column before its counter increment;
        // a preceding successful column permits another outer cycle afterwards.
        // Thus attempted columns<=2*i+1, not merely i+1.
        let attempted = add(mul(i, 2)?, 1)?;
        Ok(Self {
            dimension: n,
            restart: r,
            maximum_iterations: iterations,
            basis_vector_count: basis,
            hessenberg_scalar_count: hessenberg,
            recurrence_scalar_count_upper_bound: scalars,
            resident_bytes_upper_bound: bytes,
            operator_applications_upper_bound: add(mul(i, 3)?, 2)?,
            dot_calls_upper_bound: add(mul(attempted, add(r, 2)?)?, add(mul(i, 2)?, 4)?)?,
            axpy_calls_upper_bound: add(mul(attempted, r)?, i)?,
            // Normalization/residual/solution vector passes plus rotations and
            // triangular projected solves. Includes unsuccessful Arnoldi columns.
            recurrence_scalar_operations_upper_bound: add(
                mul(add(mul(attempted, add(r, 4)?)?, 8)?, mul(n, 8)?)?,
                mul(add(i, 1)?, mul(mul(basis, basis)?, 32)?)?,
            )?,
        })
    }
}
fn count(value: usize) -> Result<u64, GmresEnvelopeError> {
    u64::try_from(value).map_err(|_| GmresEnvelopeError::ArithmeticOverflow)
}
fn add(left: u64, right: u64) -> Result<u64, GmresEnvelopeError> {
    left.checked_add(right)
        .ok_or(GmresEnvelopeError::ArithmeticOverflow)
}
fn mul(left: u64, right: u64) -> Result<u64, GmresEnvelopeError> {
    left.checked_mul(right)
        .ok_or(GmresEnvelopeError::ArithmeticOverflow)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::Cell;
    fn policy(restart: usize, iterations: usize) -> GmresPolicy {
        GmresPolicy {
            relative_tolerance: 1e-12,
            restart,
            maximum_iterations: iterations,
        }
    }
    #[test]
    fn small_envelope_matches_manual_inventory_and_effective_restart() {
        let envelope = GmresWorkEnvelope::derive(3, policy(2, 7)).unwrap();
        assert_eq!(envelope.basis_vector_count, 3);
        assert_eq!(envelope.hessenberg_scalar_count, 6);
        assert_eq!(
            envelope.recurrence_scalar_count_upper_bound,
            8 * 3 + 6 + 8 + 1
        );
        assert_eq!(
            envelope.resident_bytes_upper_bound,
            2 * 39 * 8 + 20 * size_of::<Vec<f64>>() as u64
        );
        assert_eq!(envelope.operator_applications_upper_bound, 23);
        assert_eq!(envelope.dot_calls_upper_bound, 78);
        assert_eq!(envelope.axpy_calls_upper_bound, 37);
        assert_eq!(
            GmresWorkEnvelope::derive(3, policy(100, 1))
                .unwrap()
                .restart,
            1
        );
    }
    #[test]
    fn invalid_counter_arithmetic_and_host_allocations_fail_closed() {
        for (dimension, value) in [(0, policy(1, 1)), (1, policy(0, 1)), (1, policy(1, 0))] {
            assert_eq!(
                GmresWorkEnvelope::derive(dimension, value),
                Err(GmresEnvelopeError::InvalidPolicy)
            );
        }
        for tolerance in [0.0, -1.0, f64::NAN, f64::INFINITY] {
            let mut value = policy(1, 1);
            value.relative_tolerance = tolerance;
            assert_eq!(
                GmresWorkEnvelope::derive(1, value),
                Err(GmresEnvelopeError::InvalidPolicy)
            );
        }
        if usize::BITS > 32 {
            assert_eq!(
                GmresWorkEnvelope::derive(1, policy(1, u32::MAX as usize + 1)),
                Err(GmresEnvelopeError::IterationCounterOverflow)
            );
        }
        assert!(GmresWorkEnvelope::derive(usize::MAX, policy(1, 1)).is_err());
        assert!(GmresWorkEnvelope::derive(usize::MAX / 16, policy(1, 1)).is_err());
        assert_eq!(
            add(u64::MAX, 1),
            Err(GmresEnvelopeError::ArithmeticOverflow)
        );
        assert_eq!(
            mul(u64::MAX, 2),
            Err(GmresEnvelopeError::ArithmeticOverflow)
        );
    }
    // witness: FEA-SHARED-OPERATORS-001/compatible_solver_and_preconditioner_selection @
    //   shared_infrastructure/canonical_mixed_topology/numerical_kernel; coverage: partial;
    //   boundary: An invalid policy is rejected before any operator or reduction.
    #[test]
    fn invalid_policy_is_rejected_by_solver_before_any_operator_or_reduction() {
        let calls = Cell::new(0);
        for invalid in [policy(0, 1), policy(1, 0)] {
            let result = super::super::restarted_gmres(
                &[1.0],
                &[0.0],
                invalid,
                |value| {
                    calls.set(calls.get() + 1);
                    Ok::<_, ()>(value.to_vec())
                },
                || false,
            )
            .unwrap();
            assert!(result.is_none());
        }
        if usize::BITS > 32 {
            assert!(super::super::restarted_gmres(
                &[1.0],
                &[0.0],
                policy(1, u32::MAX as usize + 1),
                |value| {
                    calls.set(calls.get() + 1);
                    Ok::<_, ()>(value.to_vec())
                },
                || false
            )
            .unwrap()
            .is_none());
        }
        assert_eq!(calls.get(), 0);
    }

    #[test]
    fn observed_restarted_recurrence_callbacks_fit_checked_work_envelope() {
        for restart in [1, 2, 3] {
            let policy = policy(restart, 7);
            let envelope = GmresWorkEnvelope::derive(3, policy).unwrap();
            let apply_calls = Cell::new(0_u64);
            let result = super::super::restarted_gmres(
                &[1.0, 2.0, 3.0],
                &[0.0; 3],
                policy,
                |x| {
                    apply_calls.set(apply_calls.get() + 1);
                    Ok::<_, ()>(vec![
                        4.0 * x[0] + x[1],
                        -x[0] + 3.0 * x[1] + x[2],
                        2.0 * x[2] - x[1],
                    ])
                },
                || false,
            )
            .unwrap()
            .unwrap();
            assert!(result.iterations <= envelope.maximum_iterations);
            assert!(apply_calls.get() <= envelope.operator_applications_upper_bound);
        }
    }
}

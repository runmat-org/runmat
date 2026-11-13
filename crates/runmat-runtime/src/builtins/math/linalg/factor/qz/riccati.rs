//! Stable invariant-subspace extraction for standard real CARE/DARE problems.
//!
//! This is an internal consumer of the same QZ and ordered-Schur kernels as
//! the public factorization. The extended pencil is deflated by its input
//! columns before QZ; factoring the singular full pencil directly does not
//! provide the required two-block Riccati subspace.

use faer::c64;
use faer::dyn_stack::MemBuffer;
use nalgebra::DMatrix;
use runmat_value::exact_capacity_vec_with;

use super::work::QzWork;
use crate::context::host_allocation::{HostAllocationReceipt, HostTypedWorkspace};
use crate::context::RuntimeContext;
use crate::RuntimeError;

mod balance;
mod kernel;
mod managed_workspace;

use managed_workspace::RiccatiScratch;
pub(crate) use managed_workspace::{ManagedRiccatiWorkspace, RiccatiWorkspacePlan};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum RiccatiTimeDomain {
    Continuous,
    Discrete,
}

pub(crate) struct RiccatiFactorization {
    pub(crate) a: DMatrix<f64>,
    pub(crate) b: DMatrix<f64>,
    pub(crate) q: DMatrix<f64>,
    pub(crate) r: DMatrix<f64>,
    pub(crate) cross: DMatrix<f64>,
    pub(crate) solution: DMatrix<f64>,
}

/// All matrices are already finite, shape-checked and host-owned by the
/// caller. QZ work moves into the installed blocking worker; dropping the
/// awaiting caller never drops matrices while that worker is still running.
pub(crate) async fn stabilizing_subspace(
    a: DMatrix<f64>,
    b: DMatrix<f64>,
    q: DMatrix<f64>,
    r: DMatrix<f64>,
    cross: DMatrix<f64>,
    domain: RiccatiTimeDomain,
    runtime: Option<&RuntimeContext>,
    name: &'static str,
    numerical_error: impl FnOnce(&'static str) -> RuntimeError,
) -> Result<RiccatiFactorization, RuntimeError> {
    let mut work = QzWork::new(runtime, None, name);
    work.checkpoint(0).await?;
    let result = work
        .blocking(move || solve_owned(a, b, q, r, cross, domain))
        .await?;
    result.map_err(numerical_error)
}

fn solve_owned(
    a: DMatrix<f64>,
    b: DMatrix<f64>,
    q: DMatrix<f64>,
    r: DMatrix<f64>,
    cross: DMatrix<f64>,
    domain: RiccatiTimeDomain,
) -> Result<RiccatiFactorization, &'static str> {
    let n = a.nrows();
    let m = b.ncols();
    if n == 0
        || m == 0
        || a.ncols() != n
        || b.nrows() != n
        || q.shape() != (n, n)
        || r.shape() != (m, m)
        || cross.shape() != (n, m)
    {
        return Err("Riccati pencil dimensions are invalid");
    }
    let plan = RiccatiWorkspacePlan::new(n, m).map_err(|_| "Riccati pencil extent overflow")?;
    let mut h = complex_buffer(plan.pencil_len)?;
    let mut j = complex_buffer(plan.pencil_len)?;
    let mut columns = complex_buffer(plan.column_len)?;
    let mut householder = complex_buffer(plan.householder_len)?;
    let mut aa = complex_buffer(plan.reduced_len)?;
    let mut bb = complex_buffer(plan.reduced_len)?;
    let mut alpha = complex_buffer(plan.doubled)?;
    let mut beta = complex_buffer(plan.doubled)?;
    let mut lu = complex_buffer(plan.subspace_len)?;
    let mut rhs = complex_buffer(plan.subspace_len)?;
    let mut labels = exact_capacity_vec_with(plan.doubled, |_| 0_u64)
        .map_err(|_| "Riccati cluster metadata allocation failed")?;
    let mut solution = exact_capacity_vec_with(plan.subspace_len, |_| 0.0_f64)
        .map_err(|_| "Riccati solution allocation failed")?;
    let mut state_scales = exact_capacity_vec_with(n, |_| 1.0_f64)
        .map_err(|_| "Riccati balancing workspace allocation failed")?;
    let mut qr_memory = MemBuffer::try_new(plan.qr_scratch)
        .map_err(|_| "Riccati QR workspace allocation failed")?;
    let mut qz_memory = MemBuffer::try_new(plan.qz_scratch)
        .map_err(|_| "Riccati QZ workspace allocation failed")?;
    kernel::solve(
        a.as_slice(),
        b.as_slice(),
        q.as_slice(),
        r.as_slice(),
        cross.as_slice(),
        &plan,
        domain,
        RiccatiScratch {
            h: &mut h,
            j: &mut j,
            columns: &mut columns,
            householder: &mut householder,
            aa: &mut aa,
            bb: &mut bb,
            alpha: &mut alpha,
            beta: &mut beta,
            lu: &mut lu,
            rhs: &mut rhs,
            labels: &mut labels,
            solution: &mut solution,
            state_scales: &mut state_scales,
            qr_scratch: &mut qr_memory,
            qz_scratch: &mut qz_memory,
        },
    )?;
    Ok(RiccatiFactorization {
        a,
        b,
        q,
        r,
        cross,
        solution: DMatrix::from_vec(n, n, solution),
    })
}

/// Exactly claimed source matrices moved into the same selected QZ worker as
/// its scratch. The worker returns these owners for the independent original-
/// equation residual and closed-loop publication checks.
pub(crate) struct ManagedRiccatiInputs {
    pub(crate) a: HostTypedWorkspace<f64>,
    pub(crate) b: HostTypedWorkspace<f64>,
    pub(crate) q: HostTypedWorkspace<f64>,
    pub(crate) r: HostTypedWorkspace<f64>,
    pub(crate) cross: HostTypedWorkspace<f64>,
}

pub(crate) struct ManagedRiccatiFactorization {
    pub(crate) inputs: ManagedRiccatiInputs,
    pub(crate) plan: RiccatiWorkspacePlan,
    pub(crate) workspace: ManagedRiccatiWorkspace,
}

pub(crate) async fn stabilizing_subspace_managed(
    inputs: ManagedRiccatiInputs,
    plan: RiccatiWorkspacePlan,
    mut workspace: ManagedRiccatiWorkspace,
    domain: RiccatiTimeDomain,
    runtime: &RuntimeContext,
    receipt: &HostAllocationReceipt,
    name: &'static str,
    numerical_error: impl FnOnce(&'static str) -> RuntimeError,
) -> Result<ManagedRiccatiFactorization, RuntimeError> {
    let mut work = QzWork::new(Some(runtime), Some(receipt), name);
    work.checkpoint(0).await?;
    let result = work
        .blocking(move || {
            kernel::solve(
                inputs.a.as_slice(),
                inputs.b.as_slice(),
                inputs.q.as_slice(),
                inputs.r.as_slice(),
                inputs.cross.as_slice(),
                &plan,
                domain,
                workspace.numerical_scratch(),
            )?;
            Ok(ManagedRiccatiFactorization {
                inputs,
                plan,
                workspace,
            })
        })
        .await?;
    result.map_err(numerical_error)
}

/// The independent closed-loop QZ also runs under the installed blocking
/// worker. A cancelled awaiting caller cannot retire either the selected
/// factorization or its claimed A-BK input while the kernel still reads it.
pub(crate) async fn closed_loop_poles_managed(
    mut factor: ManagedRiccatiFactorization,
    closed_loop: HostTypedWorkspace<f64>,
    runtime: &RuntimeContext,
    receipt: &HostAllocationReceipt,
    name: &'static str,
    numerical_error: impl FnOnce(&'static str) -> RuntimeError,
) -> Result<(ManagedRiccatiFactorization, HostTypedWorkspace<f64>), RuntimeError> {
    let mut work = QzWork::new(Some(runtime), Some(receipt), name);
    work.checkpoint(0).await?;
    let result = work
        .blocking(move || {
            factor
                .workspace
                .closed_loop_poles_into(&factor.plan, closed_loop.as_slice())?;
            Ok((factor, closed_loop))
        })
        .await?;
    result.map_err(numerical_error)
}

#[cfg(test)]
fn real_symmetric_subspace_solution(
    right: faer::mat::MatRef<'_, c64>,
    n: usize,
) -> Result<DMatrix<f64>, &'static str> {
    let extent = n.checked_mul(n).ok_or("Riccati subspace extent overflow")?;
    let mut lu = complex_buffer(extent)?;
    let mut rhs = complex_buffer(extent)?;
    let mut solution = exact_capacity_vec_with(extent, |_| 0.0_f64)
        .map_err(|_| "Riccati solution allocation failed")?;
    let state_scales = exact_capacity_vec_with(n, |_| 1.0_f64)
        .map_err(|_| "Riccati balancing workspace allocation failed")?;
    kernel::subspace_into(right, n, &mut lu, &mut rhs, &mut solution, &state_scales)?;
    Ok(DMatrix::<f64>::from_vec(n, n, solution))
}

fn complex_buffer(len: usize) -> Result<Vec<c64>, &'static str> {
    exact_capacity_vec_with(len, |_| c64::new(0.0, 0.0))
        .map_err(|_| "Riccati workspace allocation failed")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scalar(value: f64) -> DMatrix<f64> {
        DMatrix::from_element(1, 1, value)
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[cfg_attr(not(target_arch = "wasm32"), test)]
    fn ordered_subspace_selects_stabilizing_care_and_dare_branches() {
        let care = solve_owned(
            scalar(-1.0),
            scalar(1.0),
            scalar(1.0),
            scalar(1.0),
            scalar(0.0),
            RiccatiTimeDomain::Continuous,
        )
        .expect("stable CARE subspace");
        assert!((care.solution[(0, 0)] - (2.0_f64.sqrt() - 1.0)).abs() < 1.0e-8);

        let dare = solve_owned(
            scalar(1.0),
            scalar(1.0),
            scalar(1.0),
            scalar(1.0),
            scalar(0.0),
            RiccatiTimeDomain::Discrete,
        )
        .expect("stable DARE subspace");
        assert!((dare.solution[(0, 0)] - (1.0 + 5.0_f64.sqrt()) * 0.5).abs() < 1.0e-8);

        // Q=0 makes the old fixed-point iteration stop at the unstable zero
        // root. An ordered pencil must either select the stabilizing root or
        // explicitly decline; it must never publish zero as stabilizing.
        let weakly_detectable = solve_owned(
            scalar(2.0),
            scalar(1.0),
            scalar(0.0),
            scalar(1.0),
            scalar(0.0),
            RiccatiTimeDomain::Discrete,
        );
        assert!(
            weakly_detectable
                .as_ref()
                .is_ok_and(|solution| (solution.solution[(0, 0)] - 3.0).abs() < 1.0e-8)
                || weakly_detectable.is_err()
        );

        // U0 = [0 1; 1 1] requires an LU row pivot. With U1 = X U0 for
        // X = [2 .5; .5 1], both right-hand columns must survive that pivot.
        let right = [
            c64::new(0.0, 0.0),
            c64::new(1.0, 0.0),
            c64::new(0.5, 0.0),
            c64::new(1.0, 0.0),
            c64::new(1.0, 0.0),
            c64::new(1.0, 0.0),
            c64::new(2.5, 0.0),
            c64::new(1.5, 0.0),
            c64::new(0.0, 0.0),
            c64::new(0.0, 0.0),
            c64::new(0.0, 0.0),
            c64::new(0.0, 0.0),
            c64::new(0.0, 0.0),
            c64::new(0.0, 0.0),
            c64::new(0.0, 0.0),
            c64::new(0.0, 0.0),
        ];
        let solution = real_symmetric_subspace_solution(
            faer::mat::MatRef::from_column_major_slice(&right, 4, 4),
            2,
        )
        .expect("pivoted two-column subspace solve");
        for (row, col, expected) in [(0, 0, 2.0), (0, 1, 0.5), (1, 0, 0.5), (1, 1, 1.0)] {
            assert!((solution[(row, col)] - expected).abs() < 1.0e-12);
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[cfg_attr(not(target_arch = "wasm32"), test)]
    fn balanced_pendulum_care_retains_real_symmetric_stabilizing_solution() {
        // Maintained control-systems/pendulum-lqr example. Its semidefinite
        // Q and full-rank controllable (A,B) are a supported CARE, despite the
        // much larger state cost than the dynamics and input coefficients.
        let p = 0.0132;
        let a = DMatrix::from_column_slice(
            4,
            4,
            &[
                0.0,
                0.0,
                0.0,
                0.0,
                1.0,
                -0.0024 / p,
                0.0,
                -0.006 / p,
                0.0,
                0.035316 / p,
                0.0,
                0.41202 / p,
                0.0,
                0.0,
                1.0,
                0.0,
            ],
        );
        let b = DMatrix::from_column_slice(4, 1, &[0.0, 0.024 / p, 0.0, 0.06 / p]);
        let q = DMatrix::from_diagonal(&nalgebra::DVector::from_vec(vec![5000.0, 0.0, 100.0, 0.0]));
        let factor = solve_owned(
            a,
            b,
            q,
            DMatrix::from_element(1, 1, 1.0),
            DMatrix::zeros(4, 1),
            RiccatiTimeDomain::Continuous,
        )
        .expect("balanced stable CARE subspace");
        let x = &factor.solution;
        for col in 0..4 {
            for row in 0..4 {
                assert!(x[(row, col)].is_finite());
                assert!((x[(row, col)] - x[(col, row)]).abs() < 1.0e-9);
            }
        }
    }
}

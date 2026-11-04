//! The one bridge between FEA operators and the shared sparse solver layer
//! (`runmat-sparse-solvers`). Every family's linear and eigen solve goes through here, so solver
//! policy, threading, cancellation and provenance are decided in one place.

use crate::algebra::capacity::PhaseCapacityOwner;
use std::borrow::Cow;
use std::future::Future;

use runmat_simulation_usage::{LedgerError, OperationCount, UsageCategory, UsageContext};
use runmat_sparse_solvers::{control::SolveControl, CsrRef};

use crate::{algebra::RealSparseMatrix, planning::resource::LocalExecutionPlan};

/// Index arrays of a FEA matrix as `usize` slices (zero-copy on 64-bit hosts).
fn as_usize(values: &[u64]) -> Cow<'_, [usize]> {
    #[cfg(target_pointer_width = "64")]
    {
        // SAFETY: u64 and usize have identical size and alignment on 64-bit targets and every
        // bit pattern is valid for both.
        Cow::Borrowed(unsafe {
            std::slice::from_raw_parts(values.as_ptr().cast::<usize>(), values.len())
        })
    }
    #[cfg(not(target_pointer_width = "64"))]
    {
        Cow::Owned(values.iter().map(|&value| value as usize).collect())
    }
}

/// Borrowed CSR arrays of a canonical FEA matrix.
pub(crate) struct CsrArrays<'a> {
    rows: usize,
    columns: usize,
    row_ptr: Cow<'a, [usize]>,
    col_idx: Cow<'a, [usize]>,
    values: &'a [f64],
}

impl<'a> CsrArrays<'a> {
    pub(crate) fn of(matrix: &'a RealSparseMatrix) -> Self {
        Self {
            rows: matrix.rows(),
            columns: matrix.columns(),
            row_ptr: as_usize(matrix.pattern().row_offsets()),
            col_idx: as_usize(matrix.pattern().column_indices()),
            values: matrix.values(),
        }
    }
    /// Canonical patterns are validated at construction (sorted, in-bounds), so the view needs no
    /// further checks.
    pub(crate) fn view(&self) -> CsrRef<'_, f64> {
        CsrRef::new_unchecked(
            self.rows,
            self.columns,
            &self.row_ptr,
            &self.col_idx,
            self.values,
        )
    }
}

/// Runs shared-solver work on the admitted CPU plan (inline when serial or already inside an
/// admitted pool).
pub(crate) fn with_plan<R: Send>(plan: LocalExecutionPlan, work: impl FnOnce() -> R + Send) -> R {
    let threads = usize::from(effective_plan(plan).thread_count().get());
    if threads <= 1 || runmat_cpu_execution::current_threads() > 1 {
        return work();
    }
    // Cancellation and progress are thread-bound; carry them onto the pool worker.
    let context = crate::progress::capture_fea_progress_context();
    let usage = runmat_simulation_usage::UsageContext::capture();
    match runmat_cpu_execution::with_cpu_pool(threads, move || {
        let _guard = crate::progress::replace_fea_progress_context(context);
        let _usage = usage
            .as_ref()
            .map(runmat_simulation_usage::UsageContext::enter);
        work()
    }) {
        Ok(result) => result,
        Err(error) => panic!("admitted CPU pool unavailable for the sparse solve: {error}"),
    }
}

/// Cancellation hook bound to the FEA progress context.
pub(crate) fn fea_cancelled() -> bool {
    crate::is_fea_cancelled()
}

pub(crate) fn control() -> SolveControl<'static> {
    SolveControl {
        cancelled: Some(&fea_cancelled),
        progress: None,
        progress_stride: 0,
    }
}

/// Concrete SPD strategy after policy resolution.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum SpdStrategy {
    /// Supernodal Cholesky with nested dissection (exact solve plus refinement).
    Cholesky,
    /// Conjugate gradients with the named preconditioner.
    Pcg(runmat_sparse_solvers::policy::PreconditionerChoice),
}

/// Outcome of a shared SPD solve in the prepared equation space.
pub(crate) struct SpdSolveOutcome {
    pub solution: Vec<f64>,
    pub iterations: u32,
    pub residual_norm: f64,
    pub reference_norm: f64,
    pub converged: bool,
    pub cancelled: bool,
    pub failure: Option<String>,
    /// The multigrid preconditioner ran on the accelerator (f32 inner solve, f64 outer FCG).
    pub device_preconditioner: bool,
    /// PCG only: the Lanczos λ_min estimate and the forward error it implies.
    pub spectral_estimate: Option<crate::contracts::SpectralErrorEstimate>,
}

/// The accelerator a solve may use (native only; browser builds always solve on the host).
#[cfg(not(target_arch = "wasm32"))]
pub(crate) type SolverDevice<'a> = &'a dyn runmat_accelerate_api::AccelProvider;
#[cfg(target_arch = "wasm32")]
pub(crate) type SolverDevice<'a> = std::marker::PhantomData<&'a ()>;

/// Device inner-CG steps per outer application (variable preconditioner; outer flexible CG).
pub(crate) const DEVICE_INNER_CG_STEPS: u32 = 10;

/// Inputs of a shared SPD solve.
pub(crate) struct SpdSolveRequest<'a> {
    pub matrix: &'a RealSparseMatrix,
    pub rhs: &'a [f64],
    pub initial_guess: Option<&'a [f64]>,
    pub relative_tolerance: f64,
    pub plan: LocalExecutionPlan,
    pub strategy: SpdStrategy,
    /// Degrees of freedom per node for block smoothing in multigrid (1 when unknown).
    pub block_size: usize,
    /// A factorization of exactly `matrix` (from the coercivity proof) to reuse.
    pub factor: Option<&'a runmat_sparse_solvers::direct::DirectSolver<f64>>,
    /// Accelerator for a multigrid strategy: the hierarchy is built on the host and applied on
    /// the device as an inner single-precision CG (the outer f64 loop and its true-residual
    /// certificate stay on the host). `None`, or a provider without device multigrid, keeps the
    /// host hierarchy.
    pub device: Option<SolverDevice<'a>>,
}

fn measured_solve<R>(work: OperationCount, run: impl FnOnce() -> R) -> Result<R, LedgerError> {
    let Some(context) = UsageContext::capture() else {
        return Ok(run());
    };
    let mut reservation = context.ledger().reserve(UsageCategory::Solve, work)?;
    let result = run();
    reservation.complete(work)?;
    Ok(result)
}

fn real_norm_work(n: usize) -> Result<OperationCount, LedgerError> {
    OperationCount::from_operations(u64::try_from(n).map_err(|_| LedgerError::CountOverflow)?)
        .checked_mul(2)
        .map_err(LedgerError::from)
}

fn real_residual_work(a: &CsrRef<'_, f64>) -> Result<OperationCount, LedgerError> {
    a.row_ptr()
        .windows(2)
        .try_fold(OperationCount::ZERO, |total, row| {
            // A nonempty row computes a length-k dot, then subtracts it from the RHS:
            // (2k - 1) + 1 = 2k. An empty row copies the RHS.
            let entries = u64::try_from(row[1] - row[0]).map_err(|_| LedgerError::CountOverflow)?;
            let row_work = OperationCount::from_operations(entries)
                .checked_mul(2)
                .map_err(LedgerError::from)?;
            total.checked_add(row_work).map_err(LedgerError::from)
        })
}

fn reference_norm(rhs: &[f64], residual: f64) -> Result<f64, LedgerError> {
    let b = measured_solve(real_norm_work(rhs.len())?, || {
        runmat_sparse_solvers::par::norm2(rhs)
    })?;
    Ok(if b > 0.0 { b } else { residual })
}

pub(crate) async fn solve_cholesky_with_owner<O, F, A>(
    a: CsrRef<'_, f64>,
    rhs: &[f64],
    initial_guess: Option<&[f64]>,
    relative_tolerance: f64,
    factor: Option<&runmat_sparse_solvers::direct::DirectSolver<f64>>,
    mut before_capacity: O,
) -> Result<SpdSolveOutcome, crate::algebra::coercivity::NumericalCapacityError<LedgerError, A>>
where
    O: FnMut(crate::algebra::NumericalCapacityPhase, OperationCount) -> F,
    F: std::future::Future<Output = Result<(), A>>,
{
    use crate::algebra::{coercivity::NumericalCapacityError, NumericalCapacityPhase};
    use runmat_sparse_solvers::direct::{
        DirectOptions, DirectOwnedError, DirectSolver, FactorKind,
    };
    let n = a.nrows();
    let mut x = match initial_guess {
        Some(guess) if guess.len() == n => guess.to_vec(),
        _ => vec![0.0; n],
    };
    let mut residual = vec![0.0; n];
    let residual_work = real_residual_work(&a)?;
    let norm_work = real_norm_work(n)?;
    before_capacity(NumericalCapacityPhase::SolvePrimitive, residual_work)
        .await
        .map_err(NumericalCapacityError::Admission)?;
    measured_solve(residual_work, || a.residual(&x, rhs, &mut residual))?;
    before_capacity(NumericalCapacityPhase::SolvePrimitive, norm_work)
        .await
        .map_err(NumericalCapacityError::Admission)?;
    let initial_norm = measured_solve(norm_work, || runmat_sparse_solvers::par::norm2(&residual))?;
    before_capacity(NumericalCapacityPhase::SolvePrimitive, norm_work)
        .await
        .map_err(NumericalCapacityError::Admission)?;
    let reference = reference_norm(rhs, initial_norm)?;
    if initial_norm <= relative_tolerance * reference {
        return Ok(SpdSolveOutcome {
            converged: true,
            solution: x,
            iterations: 0,
            residual_norm: initial_norm,
            reference_norm: reference,
            cancelled: false,
            failure: None,
            device_preconditioner: false,
            spectral_estimate: None,
        });
    }
    let owned;
    let solver = match factor {
        Some(solver) => solver,
        None => {
            owned = match DirectSolver::<f64>::new(
                a.into(),
                DirectOptions {
                    refinement_steps: 2,
                    ..DirectOptions::new(FactorKind::Cholesky)
                },
                control(),
            ) {
                Ok(solver) => solver,
                Err(runmat_sparse_solvers::SolverError::Usage(error)) => return Err(error.into()),
                Err(error) => {
                    return Ok(cholesky_failure(
                        x,
                        reference,
                        format!("sparse Cholesky failed: {error}"),
                    ))
                }
            };
            &owned
        }
    };
    let mut correction = vec![0.0; n];
    let refined = solver
        .solve_refined_with_owner(a.into(), &residual, &mut correction, 1, |phase, count| {
            before_capacity(NumericalCapacityPhase::DirectSolve(phase), count)
        })
        .await;
    match refined {
        Ok(_) => {}
        Err(DirectOwnedError::Numerical(runmat_sparse_solvers::SolverError::Usage(error))) => {
            return Err(error.into())
        }
        Err(DirectOwnedError::Numerical(error)) => {
            return Ok(cholesky_failure(
                x,
                reference,
                format!("sparse Cholesky solve failed: {error}"),
            ))
        }
        Err(DirectOwnedError::Admission(error)) => {
            return Err(NumericalCapacityError::Admission(error))
        }
    }
    let update_work =
        OperationCount::from_operations(u64::try_from(n).map_err(|_| LedgerError::CountOverflow)?);
    before_capacity(NumericalCapacityPhase::SolvePrimitive, update_work)
        .await
        .map_err(NumericalCapacityError::Admission)?;
    measured_solve(update_work, || {
        for (value, delta) in x.iter_mut().zip(&correction) {
            *value += delta;
        }
    })?;
    before_capacity(NumericalCapacityPhase::SolvePrimitive, residual_work)
        .await
        .map_err(NumericalCapacityError::Admission)?;
    measured_solve(residual_work, || a.residual(&x, rhs, &mut residual))?;
    before_capacity(NumericalCapacityPhase::SolvePrimitive, norm_work)
        .await
        .map_err(NumericalCapacityError::Admission)?;
    let residual_norm = measured_solve(norm_work, || runmat_sparse_solvers::par::norm2(&residual))?;
    Ok(SpdSolveOutcome {
        converged: residual_norm <= relative_tolerance.max(1e-14) * reference
            || residual_norm == 0.0,
        solution: x,
        iterations: 1,
        residual_norm,
        reference_norm: reference,
        cancelled: false,
        failure: None,
        device_preconditioner: false,
        spectral_estimate: None,
    })
}

pub(crate) async fn solve_pcg_with_owner<O, F, A>(
    a: CsrRef<'_, f64>,
    rhs: &[f64],
    initial_guess: Option<&[f64]>,
    relative_tolerance: f64,
    choice: runmat_sparse_solvers::policy::PreconditionerChoice,
    mut before_capacity: O,
) -> Result<
    SpdSolveOutcome,
    crate::algebra::coercivity::NumericalCapacityError<runmat_sparse_solvers::SolverError, A>,
>
where
    O: FnMut(crate::algebra::NumericalCapacityPhase, OperationCount) -> F,
    F: Future<Output = Result<(), A>>,
{
    use crate::algebra::{coercivity::NumericalCapacityError, NumericalCapacityPhase};
    use runmat_sparse_solvers::{
        krylov::{pcg_solve_with_capacity, KrylovOptions, PcgCapacityError},
        operator::MeteredCsrOperator,
        policy::PreconditionerChoice,
        precond::{Amg, AmgOptions, Ilu0, Jacobi},
        Preconditioner,
    };
    let n = a.nrows();
    let preconditioner: Box<dyn Preconditioner<f64>> = match choice {
        PreconditionerChoice::Jacobi => {
            let setup_work = OperationCount::from_operations(
                u64::try_from(n).map_err(|_| runmat_sparse_solvers::SolverError::IndexOverflow)?,
            );
            before_capacity(NumericalCapacityPhase::PcgPreconditionerSetup, setup_work)
                .await
                .map_err(NumericalCapacityError::Admission)?;
            Box::new(Jacobi::new(a)?)
        }
        PreconditionerChoice::Ilu0 => {
            let owner = PhaseCapacityOwner::new(
                &mut before_capacity,
                NumericalCapacityPhase::PcgPreconditionerSetup,
            );
            let ilu =
                Ilu0::new_with_capacity(a, true, &owner)
                    .await
                    .map_err(|error| match error {
                        runmat_sparse_solvers::precond::Ilu0CapacityError::Numerical(error) => {
                            NumericalCapacityError::Numerical(error)
                        }
                        runmat_sparse_solvers::precond::Ilu0CapacityError::Admission(error) => {
                            NumericalCapacityError::Admission(error)
                        }
                    })?;
            Box::new(ilu)
        }
        PreconditionerChoice::Amg => {
            let owner = PhaseCapacityOwner::new(
                &mut before_capacity,
                NumericalCapacityPhase::PcgPreconditionerSetup,
            );
            let amg = Amg::new_with_capacity(a, AmgOptions::default(), control(), &owner)
                .await
                .map_err(|error| match error {
                    runmat_sparse_solvers::precond::AmgCapacityError::Numerical(error) => {
                        NumericalCapacityError::Numerical(error)
                    }
                    runmat_sparse_solvers::precond::AmgCapacityError::Admission(error) => {
                        NumericalCapacityError::Admission(error)
                    }
                })?;
            Box::new(amg)
        }
        _ => {
            return Err(NumericalCapacityError::Numerical(
                runmat_sparse_solvers::SolverError::Unsupported(
                    "awaited PCG preconditioner has no staged reference-work owner".into(),
                ),
            ))
        }
    };
    let context = UsageContext::capture().ok_or_else(|| {
        runmat_sparse_solvers::SolverError::Unsupported(
            "awaited PCG requires active usage context".into(),
        )
    })?;
    let operator = MeteredCsrOperator::new(a, context.ledger(), UsageCategory::Solve)?;
    let mut x = match initial_guess {
        Some(guess) if guess.len() == n => guess.to_vec(),
        _ => vec![0.0; n],
    };
    let options = KrylovOptions {
        rtol: relative_tolerance,
        max_iterations: n.saturating_mul(2).clamp(256, 100_000),
        ..Default::default()
    };
    let owner = PhaseCapacityOwner::new(&mut before_capacity, NumericalCapacityPhase::PcgBatch);
    let report = pcg_solve_with_capacity(
        &operator,
        Some(preconditioner.as_ref()),
        rhs,
        &mut x,
        &options,
        control(),
        &owner,
    )
    .await
    .map_err(|error| match error {
        PcgCapacityError::Numerical(error) => NumericalCapacityError::Numerical(error),
        PcgCapacityError::Admission(error) => NumericalCapacityError::Admission(error),
    })?;
    let norm_work = real_norm_work(rhs.len()).map_err(runmat_sparse_solvers::SolverError::Usage)?;
    before_capacity(NumericalCapacityPhase::SolvePrimitive, norm_work)
        .await
        .map_err(NumericalCapacityError::Admission)?;
    let reference = reference_norm(rhs, report.residual_norm)
        .map_err(runmat_sparse_solvers::SolverError::Usage)?;
    let spectral_estimate = match report.spectrum {
        Some(spectrum) => spectral_error_estimate_with_owner(
            &a,
            preconditioner.as_ref(),
            rhs,
            &x,
            spectrum,
            &mut before_capacity,
        )
        .await
        .map_err(|error| error.map_numerical(runmat_sparse_solvers::SolverError::Usage))?,
        None => None,
    };
    Ok(SpdSolveOutcome {
        spectral_estimate,
        converged: report.converged(),
        solution: x,
        iterations: u32::try_from(report.iterations).unwrap_or(u32::MAX),
        residual_norm: report.residual_norm,
        reference_norm: reference,
        cancelled: false,
        failure: (!report.converged()).then(|| {
            format!(
                "pcg stopped with flag {} after {} iterations at relative residual {:.3e}",
                report.flag.code(),
                report.iterations,
                report.relative_residual,
            )
        }),
        device_preconditioner: false,
    })
}

fn cholesky_failure(solution: Vec<f64>, reference_norm: f64, message: String) -> SpdSolveOutcome {
    SpdSolveOutcome {
        solution,
        iterations: 0,
        residual_norm: f64::INFINITY,
        reference_norm,
        converged: false,
        cancelled: crate::is_fea_cancelled(),
        failure: Some(message),
        device_preconditioner: false,
        spectral_estimate: None,
    }
}

/// Solves a prepared SPD system with the requested strategy on the admitted plan.
pub(crate) fn solve_spd(
    request: SpdSolveRequest<'_>,
) -> Result<SpdSolveOutcome, runmat_simulation_usage::LedgerError> {
    use runmat_sparse_solvers::{
        krylov::{pcg, KrylovOptions},
        policy::PreconditionerChoice,
        precond::{Amg, AmgOptions, BlockJacobi, Ilu0, Jacobi},
        Preconditioner,
    };
    let SpdSolveRequest {
        matrix,
        rhs,
        initial_guess,
        relative_tolerance,
        plan,
        strategy,
        block_size,
        factor,
        device,
    } = request;
    let mut device_preconditioner = false;
    let n = matrix.rows();
    let arrays = CsrArrays::of(matrix);
    let failure = |message: String| -> Result<SpdSolveOutcome, LedgerError> {
        let reference_norm = measured_solve(real_norm_work(rhs.len())?, || {
            runmat_sparse_solvers::par::norm2(rhs)
        })?;
        Ok(SpdSolveOutcome {
            solution: vec![0.0; n],
            iterations: 0,
            residual_norm: f64::INFINITY,
            reference_norm,
            converged: false,
            cancelled: crate::is_fea_cancelled(),
            failure: Some(message),
            device_preconditioner: false,
            spectral_estimate: None,
        })
    };
    let solver_failure =
        |error: runmat_sparse_solvers::SolverError, context: &'static str| match error {
            runmat_sparse_solvers::SolverError::Usage(error) => Err(error),
            other => failure(format!("{context}: {other}")),
        };
    with_plan(plan, || {
        let a = arrays.view();
        match strategy {
            SpdStrategy::Cholesky => {
                let future = solve_cholesky_with_owner(
                    a,
                    rhs,
                    initial_guess,
                    relative_tolerance,
                    factor,
                    |_, _| std::future::ready(Ok::<(), std::convert::Infallible>(())),
                );
                let mut future = std::pin::pin!(future);
                let mut context = std::task::Context::from_waker(std::task::Waker::noop());
                match std::future::Future::poll(future.as_mut(), &mut context) {
                    std::task::Poll::Ready(result) => result.map_err(|error| match error {
                        crate::algebra::coercivity::NumericalCapacityError::Numerical(error) => {
                            error
                        }
                        crate::algebra::coercivity::NumericalCapacityError::Admission(never) => {
                            match never {}
                        }
                    }),
                    std::task::Poll::Pending => {
                        unreachable!("immediate Cholesky owner cannot suspend")
                    }
                }
            }
            SpdStrategy::Pcg(choice) => {
                let preconditioner: Box<dyn Preconditioner<f64>> = match choice {
                    PreconditionerChoice::Amg => match Amg::new(
                        a,
                        AmgOptions {
                            block_size: block_size.max(1),
                            strength_threshold: if block_size > 1 { 0.02 } else { 0.0 },
                            ..Default::default()
                        },
                        control(),
                    ) {
                        Ok(amg) => match device_multigrid(device, &amg) {
                            Some(on_device) => {
                                device_preconditioner = true;
                                on_device
                            }
                            None => Box::new(amg),
                        },
                        Err(error) => return solver_failure(error, "multigrid setup failed"),
                    },
                    PreconditionerChoice::Ilu0 => match Ilu0::new(a, true) {
                        Ok(ilu) => Box::new(ilu),
                        Err(error) => return solver_failure(error, "ILU(0) setup failed"),
                    },
                    PreconditionerChoice::BlockJacobi if block_size > 1 => {
                        match BlockJacobi::new(a, block_size) {
                            Ok(p) => Box::new(p),
                            Err(error) => {
                                return solver_failure(error, "block Jacobi setup failed")
                            }
                        }
                    }
                    _ => match Jacobi::new(a) {
                        Ok(p) => Box::new(p),
                        Err(error) => return solver_failure(error, "Jacobi setup failed"),
                    },
                };
                let mut x = match initial_guess {
                    Some(guess) if guess.len() == n => guess.to_vec(),
                    _ => vec![0.0; n],
                };
                let options = KrylovOptions {
                    rtol: relative_tolerance,
                    max_iterations: n.saturating_mul(2).clamp(256, 100_000),
                    ..Default::default()
                };
                let usage = runmat_simulation_usage::UsageContext::capture();
                let metered = match usage
                    .as_ref()
                    .map(|context| {
                        runmat_sparse_solvers::operator::MeteredCsrOperator::new(
                            a,
                            context.ledger(),
                            runmat_simulation_usage::UsageCategory::Solve,
                        )
                    })
                    .transpose()
                {
                    Ok(operator) => operator,
                    Err(error) => return solver_failure(error, "metered CSR operator failed"),
                };
                let operator: &dyn runmat_sparse_solvers::LinearOperator<f64> =
                    match metered.as_ref() {
                        Some(operator) => operator,
                        None => &a,
                    };
                match pcg(
                    operator,
                    Some(preconditioner.as_ref()),
                    rhs,
                    &mut x,
                    &options,
                    control(),
                ) {
                    Ok(report) => {
                        let reference = reference_norm(rhs, report.residual_norm)?;
                        let spectral_estimate = match report.spectrum {
                            Some(spectrum) => spectral_error_estimate(
                                &a,
                                preconditioner.as_ref(),
                                rhs,
                                &x,
                                spectrum,
                            )?,
                            None => None,
                        };
                        Ok(SpdSolveOutcome {
                            spectral_estimate,
                            converged: report.converged(),
                            solution: x,
                            iterations: u32::try_from(report.iterations).unwrap_or(u32::MAX),
                            residual_norm: report.residual_norm,
                            reference_norm: reference,
                            cancelled: false,
                            failure: (!report.converged()).then(|| {
                                format!(
                                    "pcg stopped with flag {} after {} iterations at relative residual {:.3e}",
                                    report.flag.code(),
                                    report.iterations,
                                    report.relative_residual
                                )
                            }),
                            device_preconditioner,
                        })
                    }
                    Err(runmat_sparse_solvers::SolverError::Cancelled) => {
                        failure("cancelled".into()).map(|mut outcome| {
                            outcome.cancelled = true;
                            outcome
                        })
                    }
                    Err(error) => solver_failure(error, "pcg failed"),
                }
            }
        }
    })
}

/// The forward error the PCG Lanczos estimate implies for the returned `x`: with the true
/// residual `r = b − A x` and `z = M⁻¹ r`, `‖e‖_A ≈ √(rᵀz / θ_min)`. One extra operator and
/// preconditioner application; `None` when either fails or the quantities are not finite.
fn spectral_error_estimate(
    a: &CsrRef<'_, f64>,
    preconditioner: &dyn runmat_sparse_solvers::Preconditioner<f64>,
    rhs: &[f64],
    x: &[f64],
    spectrum: runmat_sparse_solvers::krylov::lanczos::PcgSpectrumEstimate,
) -> Result<Option<crate::contracts::SpectralErrorEstimate>, runmat_simulation_usage::LedgerError> {
    let future = spectral_error_estimate_with_owner(a, preconditioner, rhs, x, spectrum, |_, _| {
        std::future::ready(Ok::<(), std::convert::Infallible>(()))
    });
    let mut future = std::pin::pin!(future);
    let mut context = std::task::Context::from_waker(std::task::Waker::noop());
    match std::future::Future::poll(future.as_mut(), &mut context) {
        std::task::Poll::Ready(result) => result.map_err(|error| match error {
            crate::algebra::coercivity::NumericalCapacityError::Numerical(error) => error,
            crate::algebra::coercivity::NumericalCapacityError::Admission(never) => match never {},
        }),
        std::task::Poll::Pending => unreachable!("immediate spectral owner cannot suspend"),
    }
}

async fn spectral_error_estimate_with_owner<O, F, A>(
    a: &CsrRef<'_, f64>,
    preconditioner: &dyn runmat_sparse_solvers::Preconditioner<f64>,
    rhs: &[f64],
    x: &[f64],
    spectrum: runmat_sparse_solvers::krylov::lanczos::PcgSpectrumEstimate,
    mut before_capacity: O,
) -> Result<
    Option<crate::contracts::SpectralErrorEstimate>,
    crate::algebra::coercivity::NumericalCapacityError<LedgerError, A>,
>
where
    O: FnMut(crate::algebra::NumericalCapacityPhase, OperationCount) -> F,
    F: Future<Output = Result<(), A>>,
{
    use crate::algebra::{coercivity::NumericalCapacityError, NumericalCapacityPhase};
    use runmat_simulation_usage::{LedgerError, OperationCount, UsageCategory, UsageContext};
    let measured = |count: OperationCount, work: &mut dyn FnMut()| -> Result<(), LedgerError> {
        let Some(context) = UsageContext::capture() else {
            work();
            return Ok(());
        };
        let mut reservation = context.ledger().reserve(UsageCategory::Solve, count)?;
        work();
        reservation.complete(count)?;
        Ok(())
    };
    let n = rhs.len();
    let mut ax = vec![0.0; n];
    if let Some(context) = UsageContext::capture() {
        use runmat_sparse_solvers::LinearOperator;
        let operator = match runmat_sparse_solvers::operator::MeteredCsrOperator::new(
            *a,
            context.ledger(),
            UsageCategory::Solve,
        ) {
            Ok(operator) => operator,
            Err(runmat_sparse_solvers::SolverError::Usage(error)) => return Err(error.into()),
            Err(_) => return Ok(None),
        };
        before_capacity(
            NumericalCapacityPhase::SolvePrimitive,
            operator.operations_per_apply(),
        )
        .await
        .map_err(NumericalCapacityError::Admission)?;
        match operator.apply(x, &mut ax) {
            Ok(()) => {}
            Err(runmat_sparse_solvers::SolverError::Usage(error)) => {
                return Err(NumericalCapacityError::Numerical(error))
            }
            Err(_) => return Ok(None),
        }
    } else {
        a.spmv(x, &mut ax);
    }
    let mut residual = vec![0.0; n];
    let residual_work =
        OperationCount::from_operations(u64::try_from(n).map_err(|_| LedgerError::CountOverflow)?);
    before_capacity(NumericalCapacityPhase::SolvePrimitive, residual_work)
        .await
        .map_err(NumericalCapacityError::Admission)?;
    measured(residual_work, &mut || {
        for ((residual, b), v) in residual.iter_mut().zip(rhs).zip(&ax) {
            *residual = b - v;
        }
    })?;
    let mut z = vec![0.0; n];
    match preconditioner.reference_apply_work() {
        Ok(Some(work)) => before_capacity(NumericalCapacityPhase::SolvePrimitive, work)
            .await
            .map_err(NumericalCapacityError::Admission)?,
        Ok(None) => {}
        Err(runmat_sparse_solvers::SolverError::Usage(error)) => {
            return Err(NumericalCapacityError::Numerical(error))
        }
        Err(_) => return Ok(None),
    }
    match preconditioner.apply(&residual, &mut z) {
        Ok(()) => {}
        Err(runmat_sparse_solvers::SolverError::Usage(error)) => {
            return Err(NumericalCapacityError::Numerical(error))
        }
        Err(_) => return Ok(None),
    }
    let dot_work =
        OperationCount::direct_real_dot(u64::try_from(n).map_err(|_| LedgerError::CountOverflow)?)
            .map_err(LedgerError::from)?;
    let mut rz = 0.0;
    before_capacity(NumericalCapacityPhase::SolvePrimitive, dot_work)
        .await
        .map_err(NumericalCapacityError::Admission)?;
    measured(dot_work, &mut || {
        rz = residual.iter().zip(&z).map(|(r, z)| r * z).sum();
    })?;
    let mut energy_squared = 0.0;
    before_capacity(NumericalCapacityPhase::SolvePrimitive, dot_work)
        .await
        .map_err(NumericalCapacityError::Admission)?;
    measured(dot_work, &mut || {
        energy_squared = x.iter().zip(&ax).map(|(x, v)| x * v).sum();
    })?;
    let mut energy_error_estimate = 0.0;
    before_capacity(
        NumericalCapacityPhase::SolvePrimitive,
        OperationCount::from_operations(2),
    )
    .await
    .map_err(NumericalCapacityError::Admission)?;
    measured(OperationCount::from_operations(2), &mut || {
        energy_error_estimate = (rz.max(0.0) / spectrum.lambda_min).sqrt();
    })?;
    let estimate = crate::contracts::SpectralErrorEstimate {
        lambda_min_estimate: spectrum.lambda_min,
        lambda_max_estimate: spectrum.lambda_max,
        lanczos_steps: u32::try_from(spectrum.lanczos_steps).unwrap_or(u32::MAX),
        flexible: spectrum.flexible,
        energy_error_estimate,
        relative_energy_error_estimate: if energy_squared > 0.0 {
            let mut value = 0.0;
            before_capacity(
                NumericalCapacityPhase::SolvePrimitive,
                OperationCount::from_operations(2),
            )
            .await
            .map_err(NumericalCapacityError::Admission)?;
            measured(OperationCount::from_operations(2), &mut || {
                value = energy_error_estimate / energy_squared.sqrt();
            })?;
            Some(value)
        } else {
            None
        },
    };
    Ok((rz.is_finite()
        && energy_error_estimate.is_finite()
        && spectrum.lambda_min > 0.0
        && spectrum.lambda_max >= spectrum.lambda_min)
        .then_some(estimate))
}

/// The accelerator for a multigrid PCG solve on an accelerator backend: the host hierarchy then
/// runs as a device inner CG under a host f64 flexible CG; `None` keeps the host path.
pub(crate) fn multigrid_device(
    backend_kind: crate::solve::backend::kind::LinearAlgebraBackendKind,
    strategy: SpdStrategy,
) -> Option<SolverDevice<'static>> {
    #[cfg(not(target_arch = "wasm32"))]
    {
        (backend_kind == crate::solve::backend::kind::LinearAlgebraBackendKind::Accelerator
            && matches!(
                strategy,
                SpdStrategy::Pcg(runmat_sparse_solvers::policy::PreconditionerChoice::Amg)
            ))
        .then(runmat_accelerate_api::provider)
        .flatten()
    }
    #[cfg(target_arch = "wasm32")]
    {
        let _ = (backend_kind, strategy);
        None
    }
}

/// The host hierarchy uploaded to `device` as an inner single-precision CG, when the provider
/// supports device multigrid (browser builds keep the host path).
fn device_multigrid<'a>(
    device: Option<SolverDevice<'a>>,
    amg: &runmat_sparse_solvers::precond::Amg,
) -> Option<Box<dyn runmat_sparse_solvers::Preconditioner<f64> + 'a>> {
    #[cfg(not(target_arch = "wasm32"))]
    {
        let provider = device?;
        runmat_sparse_accelerate::DeviceAmg::new(
            provider,
            amg,
            runmat_sparse_accelerate::DeviceApply::InnerCg {
                iterations: DEVICE_INNER_CG_STEPS,
            },
        )
        .ok()
        .map(|device| Box::new(device) as Box<dyn runmat_sparse_solvers::Preconditioner<f64> + 'a>)
    }
    #[cfg(target_arch = "wasm32")]
    {
        let _ = (device, amg);
        None
    }
}

/// Bytes a single solve may plan for when the study gives no explicit budget: half the host.
pub(crate) fn default_memory_budget() -> Option<u64> {
    runmat_sparse_solvers::estimate::host_memory_bytes().map(|bytes| bytes / 2)
}

/// Resolves the requested SPD selection into a concrete strategy.
pub(crate) fn resolve_spd_strategy(
    kind: crate::solve::preconditioner::SpdPreconditionerKind,
    n: usize,
    nnz: usize,
    block_size: usize,
    factor_available: bool,
) -> SpdStrategy {
    use crate::solve::preconditioner::SpdPreconditionerKind;
    use runmat_sparse_solvers::policy::{
        choose, PreconditionerChoice, ProblemClass, ProblemDescriptor, SolverOverride,
    };
    match kind {
        SpdPreconditionerKind::Direct => SpdStrategy::Cholesky,
        SpdPreconditionerKind::Amg => SpdStrategy::Pcg(PreconditionerChoice::Amg),
        SpdPreconditionerKind::Jacobi => SpdStrategy::Pcg(PreconditionerChoice::Jacobi),
        SpdPreconditionerKind::Ilu0 => SpdStrategy::Pcg(PreconditionerChoice::Ilu0),
        SpdPreconditionerKind::Auto if factor_available => SpdStrategy::Cholesky,
        SpdPreconditionerKind::Auto => {
            let decision = choose(
                &ProblemDescriptor {
                    n,
                    nnz,
                    class: ProblemClass::SymmetricPositiveDefinite,
                    block_size: block_size.max(1),
                    complex: false,
                    spatial_dimension: 3,
                    near_nullspace_available: false,
                    solves_per_factorization: 1,
                    memory_budget_bytes: default_memory_budget(),
                },
                SolverOverride::Auto,
            );
            if decision.strategy.is_direct() {
                SpdStrategy::Cholesky
            } else {
                SpdStrategy::Pcg(PreconditionerChoice::Amg)
            }
        }
    }
}

impl SpdStrategy {
    /// Provenance labels: (method for generic SPD families, preconditioner).
    pub(crate) fn provenance(
        self,
    ) -> (
        crate::contracts::SolverMethod,
        crate::contracts::SolverPreconditioner,
    ) {
        use crate::contracts::{SolverMethod, SolverPreconditioner};
        use runmat_sparse_solvers::policy::PreconditionerChoice;
        match self {
            SpdStrategy::Cholesky => (
                SolverMethod::SparseDirectCholesky,
                SolverPreconditioner::None,
            ),
            SpdStrategy::Pcg(choice) => (
                SolverMethod::AssembledPcg,
                match choice {
                    PreconditionerChoice::Amg => SolverPreconditioner::Amg,
                    PreconditionerChoice::Ilu0 => SolverPreconditioner::Ilu0,
                    PreconditionerChoice::BlockJacobi => SolverPreconditioner::BlockJacobi,
                    PreconditionerChoice::None => SolverPreconditioner::None,
                    _ => SolverPreconditioner::Jacobi,
                },
            ),
        }
    }

    pub(crate) fn label(self) -> &'static str {
        use runmat_sparse_solvers::policy::PreconditionerChoice;
        match self {
            SpdStrategy::Cholesky => "solver=sparse_cholesky ordering=nested_dissection",
            SpdStrategy::Pcg(PreconditionerChoice::Amg) => "solver=pcg preconditioner=amg",
            SpdStrategy::Pcg(PreconditionerChoice::Ilu0) => "solver=pcg preconditioner=ilu0",
            SpdStrategy::Pcg(_) => "solver=pcg preconditioner=jacobi",
        }
    }
}

thread_local! {
    static AMBIENT_PLAN: std::cell::Cell<Option<LocalExecutionPlan>> = const { std::cell::Cell::new(None) };
}

/// Installs the admitted CPU plan for every shared solve issued by `work` on this thread (family
/// pipelines whose solver call chain carries no plan of its own: modal, buckling, harmonic,
/// electromagnetic, acoustic, fluid).
pub fn with_solver_plan<R>(plan: LocalExecutionPlan, work: impl FnOnce() -> R) -> R {
    let previous = AMBIENT_PLAN.with(|slot| slot.replace(Some(plan)));
    struct Restore(Option<LocalExecutionPlan>);
    impl Drop for Restore {
        fn drop(&mut self) {
            AMBIENT_PLAN.with(|slot| slot.set(self.0));
        }
    }
    let _restore = Restore(previous);
    work()
}

/// The plan installed by [`with_solver_plan`], serial otherwise.
pub(crate) fn ambient_plan() -> LocalExecutionPlan {
    AMBIENT_PLAN
        .with(|slot| slot.get())
        .unwrap_or_else(LocalExecutionPlan::serial)
}

/// `requested` when it is parallel, otherwise the ambient plan.
pub(crate) fn effective_plan(requested: LocalExecutionPlan) -> LocalExecutionPlan {
    if requested.is_serial() {
        ambient_plan()
    } else {
        requested
    }
}

/// `Σ c_k A_k` for real FEA matrices of equal shape and complex coefficients (dynamic stiffness
/// `K − ω²M + iωC`), on the union pattern.
pub(crate) fn complex_combination(
    terms: &[(runmat_sparse_solvers::c64, &RealSparseMatrix)],
) -> Result<
    runmat_sparse_solvers::CsrMatrix<runmat_sparse_solvers::c64>,
    runmat_sparse_solvers::SolverError,
> {
    use runmat_sparse_solvers::{c64, CsrMatrix};
    let Some((_, first)) = terms.first() else {
        return Err(runmat_sparse_solvers::SolverError::Dimension(
            "empty combination".into(),
        ));
    };
    let (rows, columns) = (first.rows(), first.columns());
    if terms
        .iter()
        .any(|(_, m)| m.rows() != rows || m.columns() != columns)
    {
        return Err(runmat_sparse_solvers::SolverError::Dimension(
            "combination shapes differ".into(),
        ));
    }
    let arrays: Vec<CsrArrays<'_>> = terms.iter().map(|(_, m)| CsrArrays::of(m)).collect();
    let mut row_ptr = Vec::with_capacity(rows + 1);
    row_ptr.push(0usize);
    let mut col_idx = Vec::new();
    let mut values: Vec<c64> = Vec::new();
    let mut merged: Vec<(usize, c64)> = Vec::new();
    for row in 0..rows {
        merged.clear();
        for ((coefficient, _), array) in terms.iter().zip(&arrays) {
            let view = array.view();
            let (cols, vals) = view.row(row);
            merged.extend(
                cols.iter()
                    .zip(vals)
                    .map(|(&c, &v)| (c, coefficient.scale(v))),
            );
        }
        merged.sort_by_key(|entry| entry.0);
        for &(c, v) in &merged {
            if col_idx.len() > row_ptr[row] && *col_idx.last().unwrap() == c {
                *values.last_mut().unwrap() += v;
            } else {
                col_idx.push(c);
                values.push(v);
            }
        }
        row_ptr.push(col_idx.len());
    }
    CsrMatrix::try_new(rows, columns, row_ptr, col_idx, values)
}

/// Direct solve of a general real system (LU) or a symmetric indefinite / saddle-point system
/// (Bunch–Kaufman, then signed-regularized LDLᵀ for saddle points whose leading `primal_count`
/// unknowns form a semidefinite block). Returns the solution and `‖b − A x‖ / ‖b‖`.
pub(crate) fn solve_direct_real(
    matrix: &RealSparseMatrix,
    rhs: &[f64],
    structure: DirectStructure,
) -> Result<(Vec<f64>, f64), runmat_sparse_solvers::SolverError> {
    use runmat_sparse_solvers::direct::{DirectOptions, DirectSolver, FactorKind};
    let arrays = CsrArrays::of(matrix);
    let n = matrix.rows();
    with_plan(LocalExecutionPlan::serial(), || {
        let a = arrays.view();
        let attempts: Vec<DirectOptions> = match structure {
            DirectStructure::SaddlePoint { primal_count } => {
                let scale = a
                    .diagonal()
                    .iter()
                    .fold(0.0f64, |m, v| m.max(v.abs()))
                    .max(1.0);
                let mut signs = vec![1i8; n];
                for sign in signs.iter_mut().skip(primal_count) {
                    *sign = -1;
                }
                vec![
                    DirectOptions::new(FactorKind::BunchKaufman),
                    DirectOptions {
                        pivot_signs: Some(signs),
                        regularization_delta: 1e-10 * scale,
                        regularization_epsilon: 1e-13 * scale,
                        refinement_steps: 10,
                        ..DirectOptions::new(FactorKind::Ldlt)
                    },
                ]
            }
        };
        let mut last_error =
            runmat_sparse_solvers::SolverError::Breakdown("no factorization attempted".into());
        let b_norm = runmat_sparse_solvers::par::norm2(rhs);
        for options in attempts {
            control().check()?;
            let solver = match DirectSolver::<f64>::new(a.into(), options, control()) {
                Ok(solver) => solver,
                Err(error @ runmat_sparse_solvers::SolverError::Cancelled) => return Err(error),
                Err(error) => {
                    last_error = error;
                    continue;
                }
            };
            let mut x = vec![0.0; n];
            match solver.solve_refined(a.into(), rhs, &mut x, 1) {
                Ok(report)
                    if report.relative_residual.is_finite() && report.relative_residual <= 1e-9 =>
                {
                    return Ok((
                        x,
                        if b_norm > 0.0 {
                            report.relative_residual
                        } else {
                            0.0
                        },
                    ));
                }
                Ok(report) => {
                    last_error = runmat_sparse_solvers::SolverError::Breakdown(format!(
                        "{} relative residual {:.3e}",
                        solver.kind().as_str(),
                        report.relative_residual
                    ))
                }
                Err(error) => last_error = error,
            }
        }
        Err(last_error)
    })
}

/// Algebraic structure that selects the direct factorization.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum DirectStructure {
    SaddlePoint { primal_count: usize },
}

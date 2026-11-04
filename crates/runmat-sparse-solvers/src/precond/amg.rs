//! Smoothed-aggregation algebraic multigrid (Vaněk–Mandel–Brezina) for SPD systems.
//!
//! Setup per level: node strength graph from block norms (`|A_ij| ≥ θ √(|A_ii| |A_jj|)`), three-pass
//! aggregation, tentative prolongator from a local QR of the near-nullspace on each aggregate,
//! Jacobi prolongator smoothing `P = (I − ω D⁻¹ A) T` with `ω = 4/(3ρ(D⁻¹A))`, Galerkin coarse
//! operator `Pᵀ A P`. Cycle: symmetric V-cycle with Chebyshev smoothing on `D⁻¹A` (block diagonal
//! for vector problems) and a sparse Cholesky on the coarsest level, so the preconditioner is a
//! fixed SPD operator suitable for PCG. Every kernel is threaded with fixed partitioning, so the
//! hierarchy and each application are identical for any thread count.

use crate::bsr::BsrMatrix;
use crate::control::{Progress, ProgressPhase, SolveControl};
use crate::csr::{CsrMatrix, CsrRef};
use crate::dense::{cholesky_in_place, cholesky_solve};
use crate::direct::{DirectOptions, DirectSolver, FactorKind, SymbolicFactorization};
use crate::error::{SolverError, SolverResult};
use crate::operator::Preconditioner;
use crate::par;
use crate::spgemm;
use runmat_simulation_usage::{
    CountOverflow, LedgerError, OperationCount, UsageCategory, UsageContext,
};

pub enum AmgCapacityError<E> {
    Numerical(SolverError),
    Admission(E),
}

impl<E> From<SolverError> for AmgCapacityError<E> {
    fn from(error: SolverError) -> Self {
        Self::Numerical(error)
    }
}

fn map_spgemm_error<E>(error: spgemm::SpGemmCapacityError<E>) -> AmgCapacityError<E> {
    match error {
        spgemm::SpGemmCapacityError::Numerical(error) => AmgCapacityError::Numerical(error),
        spgemm::SpGemmCapacityError::Admission(error) => AmgCapacityError::Admission(error),
    }
}

fn immediate<R>(
    future: impl std::future::Future<Output = Result<R, AmgCapacityError<std::convert::Infallible>>>,
) -> SolverResult<R> {
    let mut future = std::pin::pin!(future);
    let mut context = std::task::Context::from_waker(std::task::Waker::noop());
    match std::future::Future::poll(future.as_mut(), &mut context) {
        std::task::Poll::Ready(result) => result.map_err(|error| match error {
            AmgCapacityError::Numerical(error) => error,
            AmgCapacityError::Admission(never) => match never {},
        }),
        std::task::Poll::Pending => unreachable!("immediate AMG owner cannot suspend"),
    }
}

async fn measured_with_capacity<R, E>(
    owner: Option<&dyn runmat_simulation_usage::AsyncCapacityOwner<E>>,
    category: UsageCategory,
    count: OperationCount,
    work: impl FnOnce() -> SolverResult<R>,
) -> Result<R, AmgCapacityError<E>> {
    if let Some(owner) = owner {
        owner
            .ensure_capacity(count)
            .await
            .map_err(AmgCapacityError::Admission)?;
    }
    super::usage::measured(category, count, work).map_err(AmgCapacityError::Numerical)
}

fn dense_llt_work(n: usize) -> SolverResult<OperationCount> {
    let n = u64::try_from(n).map_err(|_| LedgerError::CountOverflow)?;
    (1..=n)
        .try_fold(OperationCount::ZERO, |total, width| {
            total.checked_add(OperationCount::from_operations(width).checked_mul(width)?)
        })
        .map_err(|error| LedgerError::from(error).into())
}

fn dense_llt_solve_work(n: usize, rhs: usize) -> SolverResult<OperationCount> {
    let n = u64::try_from(n).map_err(|_| LedgerError::CountOverflow)?;
    let rhs = u64::try_from(rhs).map_err(|_| LedgerError::CountOverflow)?;
    OperationCount::from_operations(2)
        .checked_mul(n)
        .and_then(|work| work.checked_mul(n))
        .and_then(|work| work.checked_mul(rhs))
        .map_err(|error| LedgerError::from(error).into())
}

fn real_spmv_work(a: &CsrMatrix<f64>) -> SolverResult<OperationCount> {
    a.row_ptr()
        .windows(2)
        .try_fold(OperationCount::ZERO, |total, row| {
            total.checked_add(OperationCount::direct_real_dot(
                u64::try_from(row[1] - row[0]).map_err(|_| CountOverflow)?,
            )?)
        })
        .map_err(|error| LedgerError::from(error).into())
}

fn measured_spmv(
    a: &CsrMatrix<f64>,
    x: &[f64],
    y: &mut [f64],
    reference: OperationCount,
    category: UsageCategory,
) -> SolverResult<()> {
    super::usage::measured(category, reference, || {
        a.as_ref().spmv(x, y);
        Ok(())
    })
}

/// Near-nullspace candidates: `count` column-major vectors of length `n` (e.g. rigid-body modes).
#[derive(Debug, Clone, PartialEq)]
pub struct NearNullspace {
    pub count: usize,
    pub vectors: Vec<f64>,
}

/// Rigid-body modes of a `dim`-dimensional elasticity problem with `dim` displacement dofs per
/// node, interleaved (`u_x, u_y[, u_z]` per node). `coordinates` has `dim` entries per node.
pub fn rigid_body_modes(coordinates: &[f64], dim: usize) -> NearNullspace {
    assert!(dim == 2 || dim == 3);
    let nodes = coordinates.len() / dim;
    let n = nodes * dim;
    let count = if dim == 3 { 6 } else { 3 };
    let mut vectors = vec![0.0; n * count];
    // Center for conditioning.
    let mut center = vec![0.0; dim];
    for node in 0..nodes {
        for d in 0..dim {
            center[d] += coordinates[node * dim + d] / nodes.max(1) as f64;
        }
    }
    for node in 0..nodes {
        let x = coordinates[node * dim] - center[0];
        let y = coordinates[node * dim + 1] - center[1];
        let z = if dim == 3 {
            coordinates[node * dim + 2] - center[2]
        } else {
            0.0
        };
        let base = node * dim;
        for d in 0..dim {
            vectors[d * n + base + d] = 1.0;
        }
        if dim == 2 {
            vectors[2 * n + base] = -y;
            vectors[2 * n + base + 1] = x;
        } else {
            // Rotations about x, y, z.
            vectors[3 * n + base + 1] = -z;
            vectors[3 * n + base + 2] = y;
            vectors[4 * n + base] = z;
            vectors[4 * n + base + 2] = -x;
            vectors[5 * n + base] = -y;
            vectors[5 * n + base + 1] = x;
        }
    }
    NearNullspace { count, vectors }
}

#[derive(Debug, Clone, PartialEq)]
pub struct AmgOptions {
    /// Degrees of freedom per node (1 for scalar problems, 2/3 for elasticity).
    pub block_size: usize,
    /// Near-nullspace; defaults to the per-component constants.
    pub near_nullspace: Option<NearNullspace>,
    pub strength_threshold: f64,
    pub max_levels: usize,
    /// Factor the coarsest level directly at or below this size.
    pub coarse_size: usize,
    pub smoother_degree: usize,
    /// Ratio λmax/λmin targeted by the Chebyshev smoother.
    pub chebyshev_ratio: f64,
    /// Prolongator damping numerator (`ω = damping / ρ(D⁻¹A)`).
    pub prolongator_damping: f64,
}

impl Default for AmgOptions {
    fn default() -> Self {
        Self {
            block_size: 1,
            near_nullspace: None,
            strength_threshold: 0.0,
            max_levels: 25,
            coarse_size: 3000,
            smoother_degree: 2,
            chebyshev_ratio: 30.0,
            prolongator_damping: 4.0 / 3.0,
        }
    }
}

/// Hierarchy summary for provenance.
#[derive(Debug, Clone, PartialEq)]
pub struct AmgStatistics {
    pub level_rows: Vec<usize>,
    pub level_nnz: Vec<usize>,
    /// Σ nnz(A_l) / nnz(A_0).
    pub operator_complexity: f64,
}

/// A level operator: CSR during setup; block levels switch to BSR for the cycle (fewer index
/// loads per entry, and the CSR copy is released).
enum LevelMatrix {
    Csr(CsrMatrix<f64>),
    Bsr(BsrMatrix<f64>),
}

impl LevelMatrix {
    fn nrows(&self) -> usize {
        match self {
            Self::Csr(a) => a.nrows(),
            Self::Bsr(a) => a.nrows(),
        }
    }
    fn spmv(&self, x: &[f64], y: &mut [f64], reference: OperationCount) -> SolverResult<()> {
        super::usage::measured(UsageCategory::Solve, reference, || {
            match self {
                Self::Csr(a) => a.as_ref().spmv(x, y),
                Self::Bsr(a) => a.spmv(x, y),
            }
            Ok(())
        })
    }
    fn csr(&self) -> &CsrMatrix<f64> {
        match self {
            Self::Csr(a) => a,
            Self::Bsr(_) => unreachable!("setup operators are CSR"),
        }
    }
}

struct Level {
    a: LevelMatrix,
    spmv_reference: OperationCount,
    restriction_reference: OperationCount,
    prolongation_reference: OperationCount,
    /// Block size of this level's unknowns.
    block: usize,
    /// Inverse diagonal blocks (column-major `block × block` per node).
    inverse_blocks: Vec<f64>,
    lambda_max: f64,
    p: Option<CsrMatrix<f64>>,
    r: Option<CsrMatrix<f64>>,
}

enum CoarseSolver {
    Sparse(DirectSolver<f64>),
    Dense { factor: Vec<f64>, n: usize },
    Smoothing,
}

/// Reusable vectors of one level's cycle and smoother (every entry is written before it is read).
struct LevelScratch {
    ax: Vec<f64>,
    residual: Vec<f64>,
    coarse_b: Vec<f64>,
    coarse_x: Vec<f64>,
    smoother: SmootherScratch,
}

struct SmootherScratch {
    r: Vec<f64>,
    ax: Vec<f64>,
    d: Vec<f64>,
    z: Vec<f64>,
}

impl SmootherScratch {
    fn new(n: usize) -> Self {
        Self {
            r: vec![0.0; n],
            ax: vec![0.0; n],
            d: vec![0.0; n],
            z: vec![0.0; n],
        }
    }
}

/// Smoothed-aggregation AMG preconditioner.
pub struct Amg {
    levels: Vec<Level>,
    coarse: CoarseSolver,
    options: AmgOptions,
    statistics: AmgStatistics,
    /// Cycle vectors, allocated once and reused by every application (a concurrent second
    /// application allocates its own).
    scratch: std::sync::Mutex<Vec<LevelScratch>>,
}

impl Amg {
    fn smoother_reference(level: &Level, degree: usize) -> SolverResult<OperationCount> {
        let n = level.a.nrows();
        let d = degree.max(1);
        let inverse_per_row = level
            .block
            .checked_mul(2)
            .and_then(|value| value.checked_sub(1))
            .ok_or(LedgerError::CountOverflow)?;
        let inverse = super::usage::count(
            n,
            u64::try_from(inverse_per_row).map_err(|_| LedgerError::CountOverflow)?,
        )?;
        let degree_operations = u64::try_from(d)
            .map_err(|_| LedgerError::CountOverflow)?
            .checked_mul(2)
            .ok_or(LedgerError::CountOverflow)?;
        let mut work = super::usage::count(1, 7)?;
        for part in [
            level.spmv_reference,
            super::usage::count(n, 1)?,
            inverse,
            super::usage::count(n, 1)?,
            super::usage::count(n, degree_operations)?,
        ] {
            work = work.checked_add(part).map_err(LedgerError::from)?;
        }
        for _ in 1..d {
            for part in [
                level.spmv_reference,
                super::usage::count(n, 2)?,
                inverse,
                super::usage::count(1, 6)?,
                super::usage::count(n, 3)?,
            ] {
                work = work.checked_add(part).map_err(LedgerError::from)?;
            }
        }
        Ok(work)
    }

    fn cycle_reference(&self, level: usize) -> SolverResult<OperationCount> {
        let l = &self.levels[level];
        if level + 1 == self.levels.len() {
            return match &self.coarse {
                CoarseSolver::Sparse(solver) => solver
                    .symbolic()
                    .reference_work()
                    .map_err(|error| SolverError::Unsupported(error.to_string()))?
                    .solve_operations(1)
                    .map_err(|error| SolverError::Usage(error.into())),
                CoarseSolver::Dense { n, .. } => dense_llt_solve_work(*n, 1),
                CoarseSolver::Smoothing => {
                    Self::smoother_reference(l, self.options.smoother_degree.max(3))?
                        .checked_mul(10)
                        .map_err(|error| SolverError::Usage(error.into()))
                }
            };
        }
        let mut work = self.cycle_reference(level + 1)?;
        let smoother = Self::smoother_reference(l, self.options.smoother_degree)?;
        let prolongation = l
            .prolongation_reference
            .checked_add(super::usage::count(l.a.nrows(), 1)?)
            .map_err(LedgerError::from)?;
        for part in [
            smoother,
            l.spmv_reference,
            super::usage::count(l.a.nrows(), 1)?,
            l.restriction_reference,
            prolongation,
            smoother,
        ] {
            work = work.checked_add(part).map_err(LedgerError::from)?;
        }
        Ok(work)
    }

    pub fn new(
        matrix: CsrRef<'_, f64>,
        options: AmgOptions,
        control: SolveControl<'_>,
    ) -> SolverResult<Self> {
        immediate(
            Self::new_with_optional_capacity::<std::convert::Infallible>(
                matrix, options, control, None,
            ),
        )
    }

    pub async fn new_with_capacity<E>(
        matrix: CsrRef<'_, f64>,
        options: AmgOptions,
        control: SolveControl<'_>,
        owner: &dyn runmat_simulation_usage::AsyncCapacityOwner<E>,
    ) -> Result<Self, AmgCapacityError<E>> {
        Self::new_with_optional_capacity(matrix, options, control, Some(owner)).await
    }

    async fn new_with_optional_capacity<E>(
        matrix: CsrRef<'_, f64>,
        options: AmgOptions,
        control: SolveControl<'_>,
        owner: Option<&dyn runmat_simulation_usage::AsyncCapacityOwner<E>>,
    ) -> Result<Self, AmgCapacityError<E>> {
        let n = matrix.nrows();
        let b = options.block_size.max(1);
        if !n.is_multiple_of(b) {
            return Err(
                SolverError::Dimension(format!("block size {b} does not divide {n}")).into(),
            );
        }
        let mut candidates = match &options.near_nullspace {
            Some(ns) => {
                if ns.vectors.len() != n * ns.count {
                    return Err(SolverError::Dimension("near-nullspace length".into()).into());
                }
                ns.clone()
            }
            None => {
                let mut vectors = vec![0.0; n * b];
                for i in 0..n {
                    vectors[(i % b) * n + i] = 1.0;
                }
                NearNullspace { count: b, vectors }
            }
        };
        let mut levels = Vec::new();
        let mut a = matrix.to_owned();
        let mut block = b;
        loop {
            control.check()?;
            control.report(Progress {
                phase: ProgressPhase::MultigridSetup,
                iteration: levels.len(),
                residual: a.nrows() as f64,
                converged: 0,
            });
            let inverse_blocks = match owner {
                Some(owner) => inverse_diagonal_blocks_with_capacity(&a, block, owner).await?,
                None => inverse_diagonal_blocks(&a, block)?,
            };
            let lambda_max =
                spectral_radius_with_capacity(&a, block, &inverse_blocks, owner).await?;
            let spmv_reference = real_spmv_work(&a)?;
            let rows = a.nrows();
            let stop = rows <= options.coarse_size || levels.len() + 1 >= options.max_levels;
            if stop {
                levels.push(Level {
                    a: LevelMatrix::Csr(a),
                    spmv_reference,
                    restriction_reference: OperationCount::ZERO,
                    prolongation_reference: OperationCount::ZERO,
                    block,
                    inverse_blocks,
                    lambda_max,
                    p: None,
                    r: None,
                });
                break;
            }
            let aggregates = match owner {
                Some(owner) => {
                    aggregate_with_capacity(&a, block, options.strength_threshold, owner).await?
                }
                None => aggregate(&a, block, options.strength_threshold)?,
            };
            if aggregates.count == 0 || aggregates.count * candidates.count >= rows {
                levels.push(Level {
                    a: LevelMatrix::Csr(a),
                    spmv_reference,
                    restriction_reference: OperationCount::ZERO,
                    prolongation_reference: OperationCount::ZERO,
                    block,
                    inverse_blocks,
                    lambda_max,
                    p: None,
                    r: None,
                });
                break;
            }
            let (tentative, coarse_candidates) = match owner {
                Some(owner) => {
                    tentative_prolongator_with_capacity(
                        &aggregates,
                        block,
                        &candidates,
                        rows,
                        owner,
                    )
                    .await?
                }
                None => tentative_prolongator(&aggregates, block, &candidates, rows)?,
            };
            let p = match owner {
                Some(owner) => {
                    smooth_prolongator_with_capacity(
                        &a,
                        block,
                        &inverse_blocks,
                        lambda_max,
                        &tentative,
                        options.prolongator_damping,
                        owner,
                    )
                    .await?
                }
                None => smooth_prolongator(
                    &a,
                    block,
                    &inverse_blocks,
                    lambda_max,
                    &tentative,
                    options.prolongator_damping,
                )?,
            };
            let r = p.as_ref().transpose(false);
            let restriction_reference = real_spmv_work(&r)?;
            let prolongation_reference = real_spmv_work(&p)?;
            let ap = match owner {
                Some(owner) => spgemm::multiply_with_capacity(a.as_ref(), p.as_ref(), owner)
                    .await
                    .map_err(map_spgemm_error)?,
                None => spgemm::multiply(a.as_ref(), p.as_ref())?,
            };
            let coarse = match owner {
                Some(owner) => spgemm::multiply_with_capacity(r.as_ref(), ap.as_ref(), owner)
                    .await
                    .map_err(map_spgemm_error)?,
                None => spgemm::multiply(r.as_ref(), ap.as_ref())?,
            };
            let coarse = match owner {
                Some(owner) => repair_empty_rows_with_capacity(coarse, Some(owner)).await?,
                None => repair_empty_rows(coarse)?,
            };
            levels.push(Level {
                a: LevelMatrix::Csr(a),
                spmv_reference,
                restriction_reference,
                prolongation_reference,
                block,
                inverse_blocks,
                lambda_max,
                p: Some(p),
                r: Some(r),
            });
            a = coarse;
            block = candidates.count;
            candidates = coarse_candidates;
        }
        let coarsest = levels.last().unwrap().a.csr();
        let coarse = build_coarse_solver_with_capacity(coarsest, owner).await?;
        let nnz0 = levels[0].a.csr().nnz().max(1) as f64;
        let operator_complexity = measured_with_capacity(
            owner,
            UsageCategory::Factorization,
            super::usage::count(levels.len().saturating_sub(1) + 1, 1)?,
            || Ok(levels.iter().map(|l| l.a.csr().nnz() as f64).sum::<f64>() / nnz0),
        )
        .await?;
        let statistics = AmgStatistics {
            level_rows: levels.iter().map(|l| l.a.nrows()).collect(),
            level_nnz: levels.iter().map(|l| l.a.csr().nnz()).collect(),
            operator_complexity,
        };
        // Block levels above the coarsest run their smoother and residual products in BSR.
        let last = levels.len() - 1;
        for level in &mut levels[..last] {
            if matches!(level.block, 2 | 3 | 6) {
                if let Ok(bsr) = BsrMatrix::from_csr(level.a.csr().as_ref(), level.block) {
                    level.a = LevelMatrix::Bsr(bsr);
                }
            }
        }
        let scratch = std::sync::Mutex::new(Self::scratch_for(&levels));
        Ok(Self {
            levels,
            coarse,
            options,
            statistics,
            scratch,
        })
    }

    pub fn statistics(&self) -> &AmgStatistics {
        &self.statistics
    }

    /// Bytes the hierarchy holds while it preconditions: every level operator (CSR or BSR),
    /// prolongator and restrictor, smoother inverse blocks, the cycle scratch of one
    /// application, and the coarse solver (its dense factor, or the sparse Cholesky
    /// reservation of the coarsest operator). Admission compares this with its budget.
    pub fn resident_bytes(&self) -> u64 {
        let word = std::mem::size_of::<usize>() as u64;
        let csr = |m: &CsrMatrix<f64>| m.nnz() as u64 * (8 + word) + (m.nrows() as u64 + 1) * word;
        let mut total = 0u64;
        for level in &self.levels {
            total += match &level.a {
                LevelMatrix::Csr(a) => csr(a),
                LevelMatrix::Bsr(a) => a.heap_bytes(),
            };
            total += level.p.as_ref().map_or(0, csr) + level.r.as_ref().map_or(0, csr);
            total += level.inverse_blocks.len() as u64 * 8;
            let n = level.a.nrows() as u64;
            let coarse = level.r.as_ref().map_or(0, |r| r.nrows()) as u64;
            // LevelScratch: ax, residual and four smoother vectors of n; coarse b and x.
            total += (6 * n + 2 * coarse) * 8;
        }
        total
            + match &self.coarse {
                CoarseSolver::Sparse(_) => crate::estimate::direct_reservation_bytes(
                    *self.statistics.level_rows.last().unwrap_or(&0),
                    *self.statistics.level_nnz.last().unwrap_or(&0),
                    false,
                ),
                CoarseSolver::Dense { factor, .. } => factor.len() as u64 * 8,
                CoarseSolver::Smoothing => 0,
            }
    }

    fn scratch_for(levels: &[Level]) -> Vec<LevelScratch> {
        levels
            .iter()
            .map(|level| {
                let n = level.a.nrows();
                let coarse = level.r.as_ref().map_or(0, |r| r.nrows());
                LevelScratch {
                    ax: vec![0.0; n],
                    residual: vec![0.0; n],
                    coarse_b: vec![0.0; coarse],
                    coarse_x: vec![0.0; coarse],
                    smoother: SmootherScratch::new(n),
                }
            })
            .collect()
    }

    fn cycle(
        &self,
        level: usize,
        b: &[f64],
        x: &mut [f64],
        scratch: &mut [LevelScratch],
    ) -> SolverResult<()> {
        let l = &self.levels[level];
        if level + 1 == self.levels.len() {
            match &self.coarse {
                CoarseSolver::Sparse(solver) => {
                    x.copy_from_slice(b);
                    solver.solve_in_place(x, 1)?;
                }
                CoarseSolver::Dense { factor, n } => {
                    x.copy_from_slice(b);
                    let width = *n;
                    let work = dense_llt_solve_work(width, 1)?;
                    super::usage::measured(UsageCategory::Solve, work, || {
                        cholesky_solve(factor, width, x, 1);
                        Ok(())
                    })?;
                }
                CoarseSolver::Smoothing => {
                    x.iter_mut().for_each(|v| *v = 0.0);
                    for _ in 0..10 {
                        chebyshev(
                            l,
                            b,
                            x,
                            self.options.smoother_degree.max(3),
                            self.options.chebyshev_ratio,
                            &mut scratch[0].smoother,
                        )?;
                    }
                }
            }
            return Ok(());
        }
        x.iter_mut().for_each(|v| *v = 0.0);
        // `scratch[0]` belongs to this level, the rest to the coarser ones.
        let (here, below) = scratch.split_first_mut().expect("scratch for every level");
        chebyshev(
            l,
            b,
            x,
            self.options.smoother_degree,
            self.options.chebyshev_ratio,
            &mut here.smoother,
        )?;
        l.a.spmv(x, &mut here.ax, l.spmv_reference)?;
        super::usage::measured(
            UsageCategory::Solve,
            super::usage::count(b.len(), 1)?,
            || {
                par::sub_into(b, &here.ax, &mut here.residual);
                Ok(())
            },
        )?;
        let r = l.r.as_ref().unwrap();
        let p = l.p.as_ref().unwrap();
        measured_spmv(
            r,
            &here.residual,
            &mut here.coarse_b,
            l.restriction_reference,
            UsageCategory::Solve,
        )?;
        // The coarse level zeroes (or overwrites) its iterate before use.
        self.cycle(level + 1, &here.coarse_b, &mut here.coarse_x, below)?;
        let prolongation = l
            .prolongation_reference
            .checked_add(super::usage::count(x.len(), 1)?)
            .map_err(LedgerError::from)?;
        super::usage::measured(UsageCategory::Solve, prolongation, || {
            p.as_ref().spmv_scaled(1.0, &here.coarse_x, 1.0, x);
            Ok(())
        })?;
        chebyshev(
            l,
            b,
            x,
            self.options.smoother_degree,
            self.options.chebyshev_ratio,
            &mut here.smoother,
        )?;
        Ok(())
    }
}

/// One level of an exported hierarchy (see [`Amg::export`]).
#[derive(Debug, Clone)]
pub struct AmgLevelExport {
    pub matrix: CsrMatrix<f64>,
    /// Unknowns per node of this level.
    pub block: usize,
    /// Inverse diagonal blocks, column-major `block × block` per node.
    pub inverse_blocks: Vec<f64>,
    /// Chebyshev upper bound of `ρ(D⁻¹A)`.
    pub lambda_max: f64,
    /// `P` (fine × coarse) and `R = Pᵀ`, absent on the coarsest level.
    pub prolongation: Option<CsrMatrix<f64>>,
    pub restriction: Option<CsrMatrix<f64>>,
}

/// A value-free description of the V-cycle for other executors (for example a reduced-precision
/// device cycle). The cycle is: zero iterate, Chebyshev pre-smoothing of `smoother_degree`,
/// restrict the residual, recurse, prolongate and add, Chebyshev post-smoothing; the coarsest
/// level applies `coarse_inverse` (dense, row-major) or, when it is absent, ten Chebyshev sweeps
/// of degree `max(smoother_degree, 3)`.
#[derive(Debug, Clone)]
pub struct AmgExport {
    pub levels: Vec<AmgLevelExport>,
    pub coarse_inverse: Option<Vec<f64>>,
    pub smoother_degree: usize,
    pub chebyshev_ratio: f64,
}

impl Amg {
    /// Exports the hierarchy (CSR level operators; the coarsest operator's dense inverse when
    /// the coarse solve is exact).
    pub fn export(&self) -> SolverResult<AmgExport> {
        let levels = self
            .levels
            .iter()
            .map(|level| AmgLevelExport {
                matrix: match &level.a {
                    LevelMatrix::Csr(a) => a.clone(),
                    LevelMatrix::Bsr(a) => a.to_csr(),
                },
                block: level.block,
                inverse_blocks: level.inverse_blocks.clone(),
                lambda_max: level.lambda_max,
                prolongation: level.p.clone(),
                restriction: level.r.clone(),
            })
            .collect::<Vec<_>>();
        let n = levels.last().map_or(0, |level| level.matrix.nrows());
        let coarse_inverse = match &self.coarse {
            CoarseSolver::Smoothing => None,
            CoarseSolver::Dense { factor, n } => {
                let mut identity = identity_columns(*n);
                cholesky_solve(factor, *n, &mut identity, *n);
                Some(transpose_square(&identity, *n))
            }
            CoarseSolver::Sparse(solver) => {
                let mut identity = identity_columns(n);
                solver.solve_in_place(&mut identity, n)?;
                Some(transpose_square(&identity, n))
            }
        };
        Ok(AmgExport {
            levels,
            coarse_inverse,
            smoother_degree: self.options.smoother_degree,
            chebyshev_ratio: self.options.chebyshev_ratio,
        })
    }
}

/// The `n × n` identity, column-major.
fn identity_columns(n: usize) -> Vec<f64> {
    let mut identity = vec![0.0; n * n];
    for i in 0..n {
        identity[i * n + i] = 1.0;
    }
    identity
}

/// Column-major `n × n` to row-major.
fn transpose_square(columns: &[f64], n: usize) -> Vec<f64> {
    let mut rows = vec![0.0; n * n];
    for c in 0..n {
        for r in 0..n {
            rows[r * n + c] = columns[c * n + r];
        }
    }
    rows
}

impl Preconditioner<f64> for Amg {
    fn reference_apply_work(&self) -> SolverResult<Option<OperationCount>> {
        self.cycle_reference(0).map(Some)
    }
    fn apply(&self, r: &[f64], z: &mut [f64]) -> SolverResult<()> {
        match self.scratch.try_lock() {
            Ok(mut scratch) => self.cycle(0, r, z, &mut scratch),
            Err(_) => self.cycle(0, r, z, &mut Self::scratch_for(&self.levels)),
        }
    }
    fn label(&self) -> &'static str {
        "amg"
    }
}

async fn build_coarse_solver_with_capacity<E>(
    a: &CsrMatrix<f64>,
    owner: Option<&dyn runmat_simulation_usage::AsyncCapacityOwner<E>>,
) -> Result<CoarseSolver, AmgCapacityError<E>> {
    let n = a.nrows();
    if n <= 400 {
        let mut dense = vec![0.0; n * n];
        for row in 0..n {
            let (cols, vals) = a.as_ref().row(row);
            for (&c, &v) in cols.iter().zip(vals) {
                dense[c * n + row] = v;
            }
        }
        match measured_with_capacity(
            owner,
            UsageCategory::Factorization,
            dense_llt_work(n)?,
            || cholesky_in_place(&mut dense, n),
        )
        .await
        {
            Ok(()) => return Ok(CoarseSolver::Dense { factor: dense, n }),
            Err(AmgCapacityError::Numerical(SolverError::NotPositiveDefinite { .. })) => {}
            Err(error) => return Err(error),
        }
        return Ok(CoarseSolver::Smoothing);
    }
    let options = DirectOptions::new(FactorKind::Cholesky);
    let symbolic =
        SymbolicFactorization::analyze::<f64>(a.as_ref().into(), &options, SolveControl::NONE)?;
    let work = symbolic
        .reference_work()
        .map_err(|error| SolverError::Unsupported(error.to_string()))?
        .factorization_operations()
        .map_err(|error| SolverError::Usage(error.into()))?;
    if let Some(owner) = owner {
        owner
            .ensure_capacity(work)
            .await
            .map_err(AmgCapacityError::Admission)?;
    }
    match DirectSolver::<f64>::factorize(
        symbolic,
        a.as_ref().into(),
        options,
        None,
        SolveControl::NONE,
    ) {
        Ok(solver) => Ok(CoarseSolver::Sparse(solver)),
        Err(SolverError::NotPositiveDefinite { .. }) => Ok(CoarseSolver::Smoothing),
        Err(error) => Err(error.into()),
    }
}

/// Rows with no entries (dead coarse dofs from rank-deficient aggregates) get a unit diagonal.
fn repair_empty_rows(a: CsrMatrix<f64>) -> SolverResult<CsrMatrix<f64>> {
    immediate(repair_empty_rows_with_capacity::<std::convert::Infallible>(
        a, None,
    ))
}

async fn repair_empty_rows_with_capacity<E>(
    a: CsrMatrix<f64>,
    owner: Option<&dyn runmat_simulation_usage::AsyncCapacityOwner<E>>,
) -> Result<CsrMatrix<f64>, AmgCapacityError<E>> {
    let n = a.nrows();
    let needs = (0..n).any(|row| {
        let (cols, vals) = a.as_ref().row(row);
        !cols.iter().zip(vals).any(|(&c, &v)| c == row && v != 0.0)
    });
    if !needs {
        return Ok(a);
    }
    let mut rows = Vec::with_capacity(n);
    for row in 0..n {
        let (cols, vals) = a.as_ref().row(row);
        let has_diagonal = cols.iter().zip(vals).any(|(&c, &v)| c == row && v != 0.0);
        if has_diagonal {
            rows.push((cols.to_vec(), vals.to_vec()));
        } else if vals.iter().all(|v| *v == 0.0) {
            rows.push((vec![row], vec![1.0]));
        } else {
            let mut entries: Vec<(usize, f64)> = cols
                .iter()
                .copied()
                .zip(vals.iter().copied())
                .filter(|e| e.0 != row)
                .collect();
            let scale = measured_with_capacity(
                owner,
                UsageCategory::Factorization,
                super::usage::count(entries.len().saturating_sub(1), 1)?,
                || Ok(entries.iter().map(|e| e.1.abs()).sum::<f64>().max(1.0)),
            )
            .await?;
            entries.push((row, scale));
            entries.sort_by_key(|e| e.0);
            rows.push(entries.into_iter().unzip());
        }
    }
    Ok(CsrMatrix::from_rows(n, a.ncols(), rows))
}

fn inverse_diagonal_blocks(a: &CsrMatrix<f64>, block: usize) -> SolverResult<Vec<f64>> {
    let nodes = a.nrows() / block;
    let usage = UsageContext::capture();
    let blocks: Vec<SolverResult<Vec<f64>>> = par::map_range(nodes, 512, |node| {
        let _usage_guard = usage.as_ref().map(UsageContext::enter);
        immediate(inverse_one_block::<std::convert::Infallible>(
            a, block, node, None,
        ))
    });
    Ok(blocks
        .into_iter()
        .collect::<SolverResult<Vec<_>>>()?
        .into_iter()
        .flatten()
        .collect())
}

async fn inverse_diagonal_blocks_with_capacity<E>(
    a: &CsrMatrix<f64>,
    block: usize,
    owner: &dyn runmat_simulation_usage::AsyncCapacityOwner<E>,
) -> Result<Vec<f64>, AmgCapacityError<E>> {
    let mut inverse = Vec::with_capacity(
        a.nrows()
            .checked_mul(block)
            .ok_or(SolverError::IndexOverflow)?,
    );
    for node in 0..a.nrows() / block {
        inverse.extend(inverse_one_block(a, block, node, Some(owner)).await?);
    }
    Ok(inverse)
}

async fn inverse_one_block<E>(
    a: &CsrMatrix<f64>,
    block: usize,
    node: usize,
    owner: Option<&dyn runmat_simulation_usage::AsyncCapacityOwner<E>>,
) -> Result<Vec<f64>, AmgCapacityError<E>> {
    let mut local = vec![0.0; block * block];
    for r in 0..block {
        let row = node * block + r;
        let (cols, vals) = a.as_ref().row(row);
        for (&c, &v) in cols.iter().zip(vals) {
            if c / block == node {
                local[(c % block) * block + r] = v;
            }
        }
    }
    let mut factor = local.clone();
    let mut inverse = vec![0.0; block * block];
    let factored = measured_with_capacity(
        owner,
        UsageCategory::Factorization,
        dense_llt_work(block)?,
        || cholesky_in_place(&mut factor, block),
    )
    .await;
    match factored {
        Ok(()) => {
            for c in 0..block {
                inverse[c * block + c] = 1.0;
            }
            measured_with_capacity(
                owner,
                UsageCategory::Factorization,
                dense_llt_solve_work(block, block)?,
                || {
                    cholesky_solve(&factor, block, &mut inverse, block);
                    Ok(())
                },
            )
            .await?;
        }
        Err(AmgCapacityError::Numerical(SolverError::NotPositiveDefinite { .. })) => {
            let scale = local.iter().fold(0.0f64, |m, v| m.max(v.abs()));
            let divisions = usize::from(scale > 0.0)
                + (0..block).filter(|&c| local[c * block + c] > 0.0).count();
            measured_with_capacity(
                owner,
                UsageCategory::Factorization,
                super::usage::count(divisions, 1)?,
                || {
                    let d = if scale > 0.0 { 1.0 / scale } else { 1.0 };
                    for c in 0..block {
                        let diagonal = local[c * block + c];
                        inverse[c * block + c] = if diagonal > 0.0 { 1.0 / diagonal } else { d };
                    }
                    Ok(())
                },
            )
            .await?;
        }
        Err(error) => return Err(error),
    }
    Ok(inverse)
}

fn apply_inverse_blocks(
    block: usize,
    inverse: &[f64],
    r: &[f64],
    z: &mut [f64],
    category: UsageCategory,
) -> SolverResult<()> {
    let per_row = block
        .checked_mul(2)
        .and_then(|n| n.checked_sub(1))
        .ok_or(LedgerError::CountOverflow)?;
    super::usage::measured(
        category,
        super::usage::count(z.len(), per_row as u64)?,
        || {
            match block {
                1 => par::mul_into(inverse, r, z),
                2 => apply_inverse_blocks_fixed::<2>(inverse, r, z),
                3 => apply_inverse_blocks_fixed::<3>(inverse, r, z),
                6 => apply_inverse_blocks_fixed::<6>(inverse, r, z),
                _ => apply_inverse_blocks_general(block, inverse, r, z),
            }
            Ok(())
        },
    )
}

/// Same arithmetic as the general kernel; the block size is a constant, so indexing divides by
/// a constant and the inner loop unrolls.
fn apply_inverse_blocks_fixed<const B: usize>(inverse: &[f64], r: &[f64], z: &mut [f64]) {
    par::for_each_chunk_mut(z, |start, chunk| {
        for (offset, value) in chunk.iter_mut().enumerate() {
            let global = start + offset;
            let (node, a) = (global / B, global % B);
            let base = node * B;
            let inv = &inverse[node * B * B..(node + 1) * B * B];
            let mut sum = 0.0;
            for c in 0..B {
                sum += inv[c * B + a] * r[base + c];
            }
            *value = sum;
        }
    });
}

fn apply_inverse_blocks_general(block: usize, inverse: &[f64], r: &[f64], z: &mut [f64]) {
    par::for_each_chunk_mut(z, |start, chunk| {
        for (offset, value) in chunk.iter_mut().enumerate() {
            let global = start + offset;
            let node = global / block;
            let a = global % block;
            let base = node * block;
            let inv = &inverse[node * block * block..(node + 1) * block * block];
            let mut sum = 0.0;
            for c in 0..block {
                sum += inv[c * block + a] * r[base + c];
            }
            *value = sum;
        }
    });
}

/// Power iteration estimate of ρ(D⁻¹A) (D⁻¹A is self-adjoint in the D-inner product).
async fn spectral_radius_with_capacity<E>(
    a: &CsrMatrix<f64>,
    block: usize,
    inverse: &[f64],
    owner: Option<&dyn runmat_simulation_usage::AsyncCapacityOwner<E>>,
) -> Result<f64, AmgCapacityError<E>> {
    let n = a.nrows();
    if n == 0 {
        return Ok(1.0);
    }
    let mut v = par::seeded_vector::<f64>(n, 0x5eed);
    let mut w = vec![0.0; n];
    let mut z = vec![0.0; n];
    let mut estimate = 1.0;
    let spmv_reference = real_spmv_work(a)?;
    for _ in 0..20 {
        let norm = measured_with_capacity(
            owner,
            UsageCategory::Factorization,
            super::usage::count(n, 2)?,
            || Ok(par::norm2(&v)),
        )
        .await?;
        if norm == 0.0 {
            break;
        }
        let scale = measured_with_capacity(
            owner,
            UsageCategory::Factorization,
            super::usage::count(1, 1)?,
            || Ok(1.0 / norm),
        )
        .await?;
        measured_with_capacity(
            owner,
            UsageCategory::Factorization,
            super::usage::count(n, 1)?,
            || {
                par::scale(scale, &mut v);
                Ok(())
            },
        )
        .await?;
        if let Some(owner) = owner {
            owner
                .ensure_capacity(spmv_reference)
                .await
                .map_err(AmgCapacityError::Admission)?;
        }
        measured_spmv(a, &v, &mut w, spmv_reference, UsageCategory::Factorization)?;
        if let Some(owner) = owner {
            let per_row = block
                .checked_mul(2)
                .and_then(|value| value.checked_sub(1))
                .ok_or(SolverError::Usage(LedgerError::CountOverflow))?;
            owner
                .ensure_capacity(super::usage::count(
                    n,
                    u64::try_from(per_row)
                        .map_err(|_| SolverError::Usage(LedgerError::CountOverflow))?,
                )?)
                .await
                .map_err(AmgCapacityError::Admission)?;
        }
        apply_inverse_blocks(block, inverse, &w, &mut z, UsageCategory::Factorization)?;
        estimate = measured_with_capacity(
            owner,
            UsageCategory::Factorization,
            super::usage::count(n, 2)?,
            || Ok(par::norm2(&z)),
        )
        .await?;
        std::mem::swap(&mut v, &mut z);
    }
    // Power iteration underestimates; pad for Chebyshev safety.
    measured_with_capacity(
        owner,
        UsageCategory::Factorization,
        super::usage::count(1, 1)?,
        || Ok(estimate * 1.1),
    )
    .await
}

fn chebyshev(
    level: &Level,
    b: &[f64],
    x: &mut [f64],
    degree: usize,
    ratio: f64,
    scratch: &mut SmootherScratch,
) -> SolverResult<()> {
    let n = level.a.nrows();
    let upper = level.lambda_max;
    let (theta, delta, sigma, mut rho) =
        super::usage::measured(UsageCategory::Solve, super::usage::count(1, 7)?, || {
            let lower = upper / ratio;
            let theta = 0.5 * (upper + lower);
            let delta = 0.5 * (upper - lower);
            let sigma = theta / delta;
            Ok((theta, delta, sigma, 1.0 / sigma))
        })?;
    let (r, ax, d, z) = (
        &mut scratch.r,
        &mut scratch.ax,
        &mut scratch.d,
        &mut scratch.z,
    );
    debug_assert_eq!(r.len(), n);
    level.a.spmv(x, ax, level.spmv_reference)?;
    super::usage::measured(UsageCategory::Solve, super::usage::count(n, 1)?, || {
        par::sub_into(b, ax, r);
        Ok(())
    })?;
    apply_inverse_blocks(
        level.block,
        &level.inverse_blocks,
        r,
        z,
        UsageCategory::Solve,
    )?;
    super::usage::measured(UsageCategory::Solve, super::usage::count(n, 1)?, || {
        par::for_each_chunk_mut(d, |start, chunk| {
            for (offset, value) in chunk.iter_mut().enumerate() {
                *value = z[start + offset] / theta;
            }
        });
        Ok(())
    })?;
    for k in 0..degree.max(1) {
        super::usage::measured(UsageCategory::Solve, super::usage::count(n, 2)?, || {
            par::axpy(1.0, d, x);
            Ok(())
        })?;
        if k + 1 == degree.max(1) {
            break;
        }
        level.a.spmv(d, ax, level.spmv_reference)?;
        super::usage::measured(UsageCategory::Solve, super::usage::count(n, 2)?, || {
            par::axpy(-1.0, ax, r);
            Ok(())
        })?;
        apply_inverse_blocks(
            level.block,
            &level.inverse_blocks,
            r,
            z,
            UsageCategory::Solve,
        )?;
        let (rho_next, c1, c2) =
            super::usage::measured(UsageCategory::Solve, super::usage::count(1, 6)?, || {
                let rho_next = 1.0 / (2.0 * sigma - rho);
                Ok((rho_next, rho_next * rho, 2.0 * rho_next / delta))
            })?;
        super::usage::measured(UsageCategory::Solve, super::usage::count(n, 3)?, || {
            par::axpby(c2, z, c1, d);
            Ok(())
        })?;
        rho = rho_next;
    }
    Ok(())
}

struct Aggregates {
    /// Aggregate of each node (`usize::MAX` = not aggregated, e.g. isolated Dirichlet nodes).
    of_node: Vec<usize>,
    count: usize,
}

type StrengthNode = (Vec<usize>, Vec<f64>, f64);
type ProlongationRow = (Vec<usize>, Vec<f64>);
type ProlongationBlock = Vec<ProlongationRow>;

fn aggregate_node_work(
    a: &CsrMatrix<f64>,
    block: usize,
    node: usize,
) -> SolverResult<OperationCount> {
    let mut neighbor_ids = Vec::new();
    let mut diagonal_entries = 0usize;
    for r in 0..block {
        let (cols, _) = a.as_ref().row(node * block + r);
        for &column in cols {
            if column / block == node {
                diagonal_entries += 1;
            } else {
                neighbor_ids.push(column / block);
            }
        }
    }
    let neighbor_entries = neighbor_ids.len();
    neighbor_ids.sort_unstable();
    neighbor_ids.dedup();
    let distinct = neighbor_ids.len();
    super::usage::count(diagonal_entries, 2)?
        .checked_add(super::usage::count(neighbor_entries, 1)?)
        .map_err(LedgerError::from)?
        .checked_add(super::usage::count(neighbor_entries - distinct, 1)?)
        .map_err(LedgerError::from)?
        .checked_add(super::usage::count(
            distinct.checked_add(1).ok_or(LedgerError::CountOverflow)?,
            1,
        )?)
        .map_err(LedgerError::from)
        .map_err(SolverError::from)
}

/// Three-pass aggregation on the node strength graph.
fn aggregate(a: &CsrMatrix<f64>, block: usize, theta: f64) -> SolverResult<Aggregates> {
    let nodes = a.nrows() / block;
    // Node-level block norms and strength.
    let usage = UsageContext::capture();
    let node_graph_batches: Vec<SolverResult<Vec<StrengthNode>>> =
        par::map_range(nodes.div_ceil(512), 1, |batch| {
            let _usage_guard = usage.as_ref().map(UsageContext::enter);
            let start = batch * 512;
            let end = (start + 512).min(nodes);
            let mut operations = OperationCount::ZERO;
            for node in start..end {
                operations = operations
                    .checked_add(aggregate_node_work(a, block, node)?)
                    .map_err(LedgerError::from)?;
            }
            super::usage::measured(UsageCategory::Factorization, operations, || {
                Ok((start..end)
                    .map(|node| aggregate_node(a, block, node))
                    .collect())
            })
        });
    let node_graph: Vec<StrengthNode> = node_graph_batches
        .into_iter()
        .collect::<SolverResult<Vec<_>>>()?
        .into_iter()
        .flatten()
        .collect();
    let strong = |node: usize| strong_neighbors(&node_graph, node, theta);
    let strong_batches: Vec<SolverResult<Vec<Vec<usize>>>> =
        par::map_range(nodes.div_ceil(512), 1, |batch| {
            let _usage_guard = usage.as_ref().map(UsageContext::enter);
            let start = batch * 512;
            let end = (start + 512).min(nodes);
            let positive = (start..end)
                .map(|node| {
                    node_graph[node]
                        .1
                        .iter()
                        .filter(|&&norm| norm > 0.0)
                        .count()
                })
                .sum::<usize>();
            super::usage::measured(
                UsageCategory::Factorization,
                super::usage::count(positive, 3)?,
                || Ok((start..end).map(strong).collect()),
            )
        });
    let strong_lists: Vec<Vec<usize>> = strong_batches
        .into_iter()
        .collect::<SolverResult<Vec<_>>>()?
        .into_iter()
        .flatten()
        .collect();
    Ok(assign_aggregates(&node_graph, &strong_lists))
}

fn strong_neighbors(graph: &[(Vec<usize>, Vec<f64>, f64)], node: usize, theta: f64) -> Vec<usize> {
    let (ids, norms, d) = &graph[node];
    ids.iter()
        .zip(norms)
        .filter(|(&j, &v)| v > 0.0 && v >= theta * (d * graph[j].2).sqrt())
        .map(|(&j, _)| j)
        .collect()
}

async fn aggregate_with_capacity<E>(
    a: &CsrMatrix<f64>,
    block: usize,
    theta: f64,
    owner: &dyn runmat_simulation_usage::AsyncCapacityOwner<E>,
) -> Result<Aggregates, AmgCapacityError<E>> {
    let nodes = a.nrows() / block;
    let mut graph = Vec::with_capacity(nodes);
    for node in 0..nodes {
        graph.push(
            measured_with_capacity(
                Some(owner),
                UsageCategory::Factorization,
                aggregate_node_work(a, block, node)?,
                || Ok(aggregate_node(a, block, node)),
            )
            .await?,
        );
    }
    let mut strong = Vec::with_capacity(nodes);
    for node in 0..nodes {
        let positive = graph[node].1.iter().filter(|&&norm| norm > 0.0).count();
        strong.push(
            measured_with_capacity(
                Some(owner),
                UsageCategory::Factorization,
                super::usage::count(positive, 3)?,
                || Ok(strong_neighbors(&graph, node, theta)),
            )
            .await?,
        );
    }
    Ok(assign_aggregates(&graph, &strong))
}

fn assign_aggregates(
    node_graph: &[(Vec<usize>, Vec<f64>, f64)],
    strong_lists: &[Vec<usize>],
) -> Aggregates {
    let nodes = node_graph.len();
    let mut of_node = vec![usize::MAX; nodes];
    let mut count = 0usize;
    // Pass 1: root nodes whose strong neighborhood is entirely free.
    for node in 0..nodes {
        if of_node[node] != usize::MAX || strong_lists[node].is_empty() {
            continue;
        }
        if strong_lists[node].iter().all(|&j| of_node[j] == usize::MAX) {
            of_node[node] = count;
            for &j in &strong_lists[node] {
                of_node[j] = count;
            }
            count += 1;
        }
    }
    // Pass 2: attach remaining nodes to the strongest adjacent aggregate (from pass 1).
    let snapshot = of_node.clone();
    for node in 0..nodes {
        if of_node[node] != usize::MAX || strong_lists[node].is_empty() {
            continue;
        }
        let (ids, norms, _) = &node_graph[node];
        let mut best = (usize::MAX, -1.0f64);
        for (&j, &v) in ids.iter().zip(norms) {
            if snapshot[j] != usize::MAX
                && strong_lists[node].binary_search(&j).is_ok()
                && v > best.1
            {
                best = (snapshot[j], v);
            }
        }
        if best.0 != usize::MAX {
            of_node[node] = best.0;
        }
    }
    // Pass 3: leftovers form their own aggregates with free strong neighbors.
    for node in 0..nodes {
        if of_node[node] != usize::MAX || strong_lists[node].is_empty() {
            continue;
        }
        of_node[node] = count;
        for &j in &strong_lists[node] {
            if of_node[j] == usize::MAX {
                of_node[j] = count;
            }
        }
        count += 1;
    }
    Aggregates { of_node, count }
}

fn aggregate_node(a: &CsrMatrix<f64>, block: usize, node: usize) -> (Vec<usize>, Vec<f64>, f64) {
    let mut neighbors: Vec<(usize, f64)> = Vec::new();
    let mut diagonal = 0.0;
    for r in 0..block {
        let row = node * block + r;
        let (cols, vals) = a.as_ref().row(row);
        for (&c, &v) in cols.iter().zip(vals) {
            let other = c / block;
            if other == node {
                diagonal += v * v;
            } else {
                neighbors.push((other, v * v));
            }
        }
    }
    neighbors.sort_by_key(|e| e.0);
    let mut ids = Vec::new();
    let mut norms: Vec<f64> = Vec::new();
    for (id, value) in neighbors {
        if ids.last() == Some(&id) {
            *norms.last_mut().unwrap() += value;
        } else {
            ids.push(id);
            norms.push(value);
        }
    }
    (
        ids,
        norms.into_iter().map(f64::sqrt).collect(),
        diagonal.sqrt(),
    )
}

/// Tentative prolongator from local QR of the candidates on each aggregate; returns `T` and the
/// coarse candidates (the `R` factors).
fn tentative_prolongator(
    aggregates: &Aggregates,
    block: usize,
    candidates: &NearNullspace,
    rows: usize,
) -> SolverResult<(CsrMatrix<f64>, NearNullspace)> {
    let k = candidates.count;
    let mut members: Vec<Vec<usize>> = vec![Vec::new(); aggregates.count];
    for (node, &agg) in aggregates.of_node.iter().enumerate() {
        if agg != usize::MAX {
            members[agg].push(node);
        }
    }
    // Per-aggregate QR (modified Gram–Schmidt with reorthogonalization).
    let usage = UsageContext::capture();
    let local: Vec<SolverResult<(Vec<f64>, Vec<f64>)>> =
        par::map_range(aggregates.count, 64, |agg| {
            let _usage_guard = usage.as_ref().map(UsageContext::enter);
            immediate(qr_aggregate::<std::convert::Infallible>(
                &members[agg],
                block,
                candidates,
                rows,
                None,
            ))
        });
    Ok(finish_tentative(
        &members,
        block,
        rows,
        k,
        local.into_iter().collect::<SolverResult<Vec<_>>>()?,
    ))
}

fn finish_tentative(
    members: &[Vec<usize>],
    block: usize,
    rows: usize,
    k: usize,
    local: Vec<(Vec<f64>, Vec<f64>)>,
) -> (CsrMatrix<f64>, NearNullspace) {
    let coarse_n = members.len() * k;
    let mut coarse_vectors = vec![0.0; coarse_n * k];
    let mut row_entries: Vec<(Vec<usize>, Vec<f64>)> = vec![(Vec::new(), Vec::new()); rows];
    for (agg, (q, r)) in local.into_iter().enumerate() {
        let dofs: Vec<usize> = members[agg]
            .iter()
            .flat_map(|&node| (0..block).map(move |rr| node * block + rr))
            .collect();
        let m = dofs.len();
        for (i, &dof) in dofs.iter().enumerate() {
            for c in 0..k {
                let value = q[c * m + i];
                if value != 0.0 {
                    row_entries[dof].0.push(agg * k + c);
                    row_entries[dof].1.push(value);
                }
            }
        }
        for c in 0..k {
            for p in 0..k {
                // Coarse candidate c at coarse dof (agg, p) = R[p, c].
                coarse_vectors[c * coarse_n + agg * k + p] = r[c * k + p];
            }
        }
    }
    let t = CsrMatrix::from_rows(rows, coarse_n, row_entries);
    (
        t,
        NearNullspace {
            count: k,
            vectors: coarse_vectors,
        },
    )
}

async fn tentative_prolongator_with_capacity<E>(
    aggregates: &Aggregates,
    block: usize,
    candidates: &NearNullspace,
    rows: usize,
    owner: &dyn runmat_simulation_usage::AsyncCapacityOwner<E>,
) -> Result<(CsrMatrix<f64>, NearNullspace), AmgCapacityError<E>> {
    let mut members: Vec<Vec<usize>> = vec![Vec::new(); aggregates.count];
    for (node, &agg) in aggregates.of_node.iter().enumerate() {
        if agg != usize::MAX {
            members[agg].push(node);
        }
    }
    let mut local = Vec::with_capacity(aggregates.count);
    for nodes in &members {
        local.push(qr_aggregate(nodes, block, candidates, rows, Some(owner)).await?);
    }
    Ok(finish_tentative(
        &members,
        block,
        rows,
        candidates.count,
        local,
    ))
}

async fn qr_aggregate<E>(
    nodes: &[usize],
    block: usize,
    candidates: &NearNullspace,
    rows: usize,
    owner: Option<&dyn runmat_simulation_usage::AsyncCapacityOwner<E>>,
) -> Result<(Vec<f64>, Vec<f64>), AmgCapacityError<E>> {
    let k = candidates.count;
    let dofs: Vec<usize> = nodes
        .iter()
        .flat_map(|&node| (0..block).map(move |r| node * block + r))
        .collect();
    let m = dofs.len();
    let mut q = vec![0.0; m * k];
    for c in 0..k {
        for (i, &dof) in dofs.iter().enumerate() {
            q[c * m + i] = candidates.vectors[c * rows + dof];
        }
    }
    let mut r = vec![0.0; k * k];
    for c in 0..k {
        let m_count =
            u64::try_from(m).map_err(|_| SolverError::Usage(LedgerError::CountOverflow))?;
        let c_count =
            u64::try_from(c).map_err(|_| SolverError::Usage(LedgerError::CountOverflow))?;
        let quote = OperationCount::from_operations(4)
            .checked_mul(m_count)
            .and_then(|work| {
                work.checked_add(
                    OperationCount::from_operations(8)
                        .checked_mul(m_count)?
                        .checked_mul(c_count)?,
                )
            })
            .and_then(|work| work.checked_add(OperationCount::from_operations(1)))
            .map_err(|error| SolverError::Usage(error.into()))?;
        let (norm, normalize) =
            measured_with_capacity(owner, UsageCategory::Factorization, quote, || {
                let original = (0..m).map(|i| q[c * m + i].powi(2)).sum::<f64>().sqrt();
                for _pass in 0..2 {
                    for p in 0..c {
                        let dot: f64 = (0..m).map(|i| q[p * m + i] * q[c * m + i]).sum();
                        r[c * k + p] += dot;
                        for i in 0..m {
                            q[c * m + i] -= dot * q[p * m + i];
                        }
                    }
                }
                let norm = (0..m).map(|i| q[c * m + i].powi(2)).sum::<f64>().sqrt();
                Ok((
                    norm,
                    norm > 1e-10 * original.max(f64::MIN_POSITIVE) && norm > 0.0,
                ))
            })
            .await?;
        if normalize {
            r[c * k + c] = norm;
            measured_with_capacity(
                owner,
                UsageCategory::Factorization,
                super::usage::count(m, 1)?,
                || {
                    for i in 0..m {
                        q[c * m + i] /= norm;
                    }
                    Ok(())
                },
            )
            .await?;
        } else {
            for i in 0..m {
                q[c * m + i] = 0.0;
            }
        }
    }
    Ok((q, r))
}

fn smooth_batch_columns_quote(
    at: &CsrMatrix<f64>,
    t: &CsrMatrix<f64>,
    block: usize,
    start: usize,
    end: usize,
) -> SolverResult<(Vec<Vec<usize>>, OperationCount)> {
    let mut node_columns = Vec::with_capacity(end - start);
    let mut quote = OperationCount::ZERO;
    for node in start..end {
        let mut cols = Vec::new();
        for r in 0..block {
            cols.extend_from_slice(at.as_ref().row(node * block + r).0);
        }
        cols.sort_unstable();
        cols.dedup();
        let mut overlaps = 0usize;
        for r in 0..block {
            let (tc, _) = t.as_ref().row(node * block + r);
            overlaps += tc
                .iter()
                .filter(|&&c| cols.binary_search(&c).is_ok())
                .count();
        }
        let row_columns = block
            .checked_mul(cols.len())
            .ok_or(LedgerError::CountOverflow)?;
        let per_column = block.checked_mul(2).ok_or(LedgerError::CountOverflow)?;
        quote = quote
            .checked_add(super::usage::count(
                row_columns,
                u64::try_from(per_column).map_err(|_| LedgerError::CountOverflow)?,
            )?)
            .map_err(LedgerError::from)?
            .checked_add(super::usage::count(overlaps, 1)?)
            .map_err(LedgerError::from)?;
        node_columns.push(cols);
    }
    Ok((node_columns, quote))
}

fn smooth_prolongator(
    a: &CsrMatrix<f64>,
    block: usize,
    inverse: &[f64],
    lambda_max: f64,
    t: &CsrMatrix<f64>,
    damping: f64,
) -> SolverResult<CsrMatrix<f64>> {
    let omega = super::usage::measured(
        UsageCategory::Factorization,
        super::usage::count(1, 1)?,
        || Ok(damping / lambda_max.max(f64::MIN_POSITIVE)),
    )?;
    let at = spgemm::multiply(a.as_ref(), t.as_ref())?;
    // P = T − ω D⁻¹ (A T), applying the block inverse row-block by row-block.
    let nodes = a.nrows() / block;
    let usage = UsageContext::capture();
    let rows: Vec<SolverResult<Vec<ProlongationBlock>>> =
        par::map_range(nodes.div_ceil(256), 1, |batch| {
            let _usage_guard = usage.as_ref().map(UsageContext::enter);
            let start = batch * 256;
            let end = (start + 256).min(nodes);
            let (node_columns, quote) = smooth_batch_columns_quote(&at, t, block, start, end)?;
            super::usage::measured(UsageCategory::Factorization, quote, || {
                Ok((start..end)
                    .zip(node_columns)
                    .map(|(node, cols)| smooth_node(node, block, inverse, &at, t, omega, cols))
                    .collect())
            })
        });
    Ok(CsrMatrix::from_rows(
        a.nrows(),
        t.ncols(),
        rows.into_iter()
            .collect::<SolverResult<Vec<_>>>()?
            .into_iter()
            .flatten()
            .flatten()
            .collect(),
    ))
}

async fn smooth_prolongator_with_capacity<E>(
    a: &CsrMatrix<f64>,
    block: usize,
    inverse: &[f64],
    lambda_max: f64,
    t: &CsrMatrix<f64>,
    damping: f64,
    owner: &dyn runmat_simulation_usage::AsyncCapacityOwner<E>,
) -> Result<CsrMatrix<f64>, AmgCapacityError<E>> {
    let omega = measured_with_capacity(
        Some(owner),
        UsageCategory::Factorization,
        super::usage::count(1, 1)?,
        || Ok(damping / lambda_max.max(f64::MIN_POSITIVE)),
    )
    .await?;
    let at = spgemm::multiply_with_capacity(a.as_ref(), t.as_ref(), owner)
        .await
        .map_err(map_spgemm_error)?;
    let mut rows = Vec::with_capacity(a.nrows());
    for node in 0..a.nrows() / block {
        let (mut columns, quote) = smooth_batch_columns_quote(&at, t, block, node, node + 1)?;
        let cols = columns.pop().expect("one node quote");
        let block_rows =
            measured_with_capacity(Some(owner), UsageCategory::Factorization, quote, || {
                Ok(smooth_node(node, block, inverse, &at, t, omega, cols))
            })
            .await?;
        rows.extend(block_rows);
    }
    Ok(CsrMatrix::from_rows(a.nrows(), t.ncols(), rows))
}

fn smooth_node(
    node: usize,
    block: usize,
    inverse: &[f64],
    at: &CsrMatrix<f64>,
    t: &CsrMatrix<f64>,
    omega: f64,
    cols: Vec<usize>,
) -> Vec<(Vec<usize>, Vec<f64>)> {
    let inv = &inverse[node * block * block..(node + 1) * block * block];
    let width = cols.len();
    let mut dense = vec![0.0; block * width];
    for r in 0..block {
        let (c, v) = at.as_ref().row(node * block + r);
        for (&cc, &vv) in c.iter().zip(v) {
            let position = cols.binary_search(&cc).unwrap();
            dense[position * block + r] = vv;
        }
    }
    let mut out = Vec::with_capacity(block);
    for r in 0..block {
        let row = node * block + r;
        let mut entries: Vec<(usize, f64)> = Vec::with_capacity(width + block);
        for (position, &col) in cols.iter().enumerate() {
            let mut sum = 0.0;
            for c in 0..block {
                sum += inv[c * block + r] * dense[position * block + c];
            }
            entries.push((col, -omega * sum));
        }
        let (tc, tv) = t.as_ref().row(row);
        for (&c, &v) in tc.iter().zip(tv) {
            match entries.binary_search_by_key(&c, |e| e.0) {
                Ok(position) => entries[position].1 += v,
                Err(position) => entries.insert(position, (c, v)),
            }
        }
        entries.retain(|e| e.1 != 0.0);
        out.push(entries.into_iter().unzip());
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::krylov::{pcg, KrylovOptions};
    use crate::testutil::poisson3d;

    fn ready<T>(future: impl std::future::Future<Output = T>) -> T {
        let mut future = std::pin::pin!(future);
        let mut context = std::task::Context::from_waker(std::task::Waker::noop());
        match std::future::Future::poll(future.as_mut(), &mut context) {
            std::task::Poll::Ready(value) => value,
            std::task::Poll::Pending => panic!("test capacity owner unexpectedly suspended"),
        }
    }

    #[test]
    fn awaited_preconditioner_ownership_matches_native_and_counts_one_cycle() {
        use crate::csr::TripletBuilder;
        use runmat_simulation_usage::{AsyncCapacityOwner, UsageLedger};
        use std::{cell::Cell, sync::Arc};

        struct Incremental(Arc<UsageLedger>);
        impl AsyncCapacityOwner<LedgerError> for Incremental {
            fn ensure_capacity(
                &self,
                additional: OperationCount,
            ) -> std::pin::Pin<Box<dyn std::future::Future<Output = Result<(), LedgerError>> + '_>>
            {
                Box::pin(async move {
                    let state = self.0.snapshot()?;
                    let required = state
                        .completed
                        .total()?
                        .checked_add(state.reserved)?
                        .checked_add(additional)?;
                    self.0.update_grant(required.max(state.granted))
                })
            }
        }

        let matrix = poisson3d::<f64>(4, 4, 4);
        let options = AmgOptions {
            coarse_size: 8,
            ..Default::default()
        };
        let native = Amg::new(matrix.as_ref(), options.clone(), SolveControl::NONE).unwrap();
        let budget = OperationCount::from_operations(10_000_000_000);
        let ledger = Arc::new(UsageLedger::new(budget, OperationCount::ZERO).unwrap());
        let owner = Incremental(Arc::clone(&ledger));
        let context = UsageContext::new(Arc::clone(&ledger));
        let awaited = context.run_sync(|| {
            match ready(Amg::new_with_capacity(
                matrix.as_ref(),
                options,
                SolveControl::NONE,
                &owner,
            )) {
                Ok(amg) => amg,
                Err(AmgCapacityError::Numerical(error)) => panic!("AMG setup: {error}"),
                Err(AmgCapacityError::Admission(error)) => panic!("AMG grant: {error}"),
            }
        });
        assert_eq!(native.statistics(), awaited.statistics());
        assert!(awaited.statistics().level_rows.len() > 1);
        let rhs = par::seeded_vector::<f64>(matrix.nrows(), 7);
        let mut expected = vec![0.0; rhs.len()];
        native.apply(&rhs, &mut expected).unwrap();
        let quote = awaited.reference_apply_work().unwrap().unwrap();
        let before = ledger.snapshot().unwrap().completed.solve.operations();
        let mut actual = vec![0.0; rhs.len()];
        context.run_sync(|| {
            ready(owner.ensure_capacity(quote)).unwrap();
            awaited.apply(&rhs, &mut actual).unwrap();
        });
        let after = ledger.snapshot().unwrap().completed.solve.operations();
        assert_eq!(after - before, quote.operations());
        assert_eq!(actual, expected);
        assert!(ledger.snapshot().unwrap().granted < budget);

        let mut builder = TripletBuilder::<f64>::new(2, 2);
        for (row, column, value) in [(0, 0, 1.0), (0, 1, 1.0), (1, 0, 1.0), (1, 1, 1.0 + 1e-13)] {
            builder.push(row, column, value);
        }
        let shifted = builder.build();
        let native_ilu = crate::precond::Ilu0::new(shifted.as_ref(), true).unwrap();
        assert!(native_ilu.shift() > 0.0);
        let awaited_ilu = ready(crate::precond::Ilu0::new_with_capacity(
            shifted.as_ref(),
            true,
            &owner,
        ));
        match awaited_ilu {
            Ok(ilu) => assert_eq!(ilu.shift(), native_ilu.shift()),
            Err(crate::precond::Ilu0CapacityError::Numerical(error)) => {
                panic!("ILU setup: {error}")
            }
            Err(crate::precond::Ilu0CapacityError::Admission(error)) => {
                panic!("ILU grant: {error}")
            }
        }

        struct DenyAfter(Cell<usize>);
        impl AsyncCapacityOwner<LedgerError> for DenyAfter {
            fn ensure_capacity(
                &self,
                _additional: OperationCount,
            ) -> std::pin::Pin<Box<dyn std::future::Future<Output = Result<(), LedgerError>> + '_>>
            {
                Box::pin(async move {
                    let remaining = self.0.get();
                    if remaining == 0 {
                        return Err(LedgerError::GrantUnavailable);
                    }
                    self.0.set(remaining - 1);
                    Ok(())
                })
            }
        }
        let denied = ready(crate::precond::Ilu0::new_with_capacity(
            shifted.as_ref(),
            true,
            &DenyAfter(Cell::new(4)),
        ));
        assert!(matches!(
            denied,
            Err(crate::precond::Ilu0CapacityError::Admission(
                LedgerError::GrantUnavailable
            ))
        ));
    }

    #[test]
    fn amg_pcg_converges_in_few_iterations_on_poisson() {
        let a = poisson3d::<f64>(30, 30, 30);
        let amg = Amg::new(
            a.as_ref(),
            AmgOptions {
                coarse_size: 500,
                ..Default::default()
            },
            SolveControl::NONE,
        )
        .unwrap();
        assert!(amg.statistics().level_rows.len() >= 2);
        let b = par::seeded_vector::<f64>(a.nrows(), 1);
        let mut x = vec![0.0; a.nrows()];
        let report = pcg(
            &a,
            Some(&amg),
            &b,
            &mut x,
            &KrylovOptions {
                rtol: 1e-8,
                max_iterations: 200,
                ..Default::default()
            },
            SolveControl::NONE,
        )
        .unwrap();
        assert!(report.converged(), "{report:?}");
        assert!(report.iterations < 40, "{report:?} {:?}", amg.statistics());
    }

    /// The exported cycle replayed from the export alone (the contract device executors follow).
    fn replay(export: &AmgExport, level: usize, b: &[f64]) -> Vec<f64> {
        let l = &export.levels[level];
        let n = l.matrix.nrows();
        let spmv = |a: &CsrMatrix<f64>, x: &[f64]| {
            let mut y = vec![0.0; a.nrows()];
            a.as_ref().spmv(x, &mut y);
            y
        };
        let smooth = |x: &mut Vec<f64>, degree: usize| {
            let upper = l.lambda_max;
            let lower = upper / export.chebyshev_ratio;
            let (theta, delta) = (0.5 * (upper + lower), 0.5 * (upper - lower));
            let sigma = theta / delta;
            let mut rho = 1.0 / sigma;
            let ax = spmv(&l.matrix, x);
            let mut r: Vec<f64> = b.iter().zip(&ax).map(|(b, a)| b - a).collect();
            let mut z = vec![0.0; n];
            apply_inverse_blocks(l.block, &l.inverse_blocks, &r, &mut z, UsageCategory::Solve)
                .unwrap();
            let mut d: Vec<f64> = z.iter().map(|v| v / theta).collect();
            for k in 0..degree.max(1) {
                x.iter_mut().zip(&d).for_each(|(x, d)| *x += d);
                if k + 1 == degree.max(1) {
                    break;
                }
                let ad = spmv(&l.matrix, &d);
                r.iter_mut().zip(&ad).for_each(|(r, a)| *r -= a);
                apply_inverse_blocks(l.block, &l.inverse_blocks, &r, &mut z, UsageCategory::Solve)
                    .unwrap();
                let rho_next = 1.0 / (2.0 * sigma - rho);
                let (c1, c2) = (rho_next * rho, 2.0 * rho_next / delta);
                d.iter_mut()
                    .zip(&z)
                    .for_each(|(d, z)| *d = c2 * z + c1 * *d);
                rho = rho_next;
            }
        };
        if level + 1 == export.levels.len() {
            return match &export.coarse_inverse {
                Some(inverse) => (0..n)
                    .map(|r| (0..n).map(|c| inverse[r * n + c] * b[c]).sum())
                    .collect(),
                None => {
                    let mut x = vec![0.0; n];
                    for _ in 0..10 {
                        smooth(&mut x, export.smoother_degree.max(3));
                    }
                    x
                }
            };
        }
        let mut x = vec![0.0; n];
        smooth(&mut x, export.smoother_degree);
        let ax = spmv(&l.matrix, &x);
        let residual: Vec<f64> = b.iter().zip(&ax).map(|(b, a)| b - a).collect();
        let coarse_b = spmv(l.restriction.as_ref().unwrap(), &residual);
        let coarse_x = replay(export, level + 1, &coarse_b);
        let correction = spmv(l.prolongation.as_ref().unwrap(), &coarse_x);
        x.iter_mut().zip(&correction).for_each(|(x, c)| *x += c);
        smooth(&mut x, export.smoother_degree);
        x
    }

    #[test]
    fn exported_hierarchy_replays_the_cycle() {
        let p = crate::testutil::elasticity::elasticity3d(8, 7, 6);
        let amg = Amg::new(
            p.matrix.as_ref(),
            AmgOptions {
                block_size: 3,
                near_nullspace: Some(rigid_body_modes(&p.coordinates, 3)),
                coarse_size: 200,
                ..Default::default()
            },
            SolveControl::NONE,
        )
        .unwrap();
        let export = amg.export().unwrap();
        assert!(export.levels.len() >= 2 && export.coarse_inverse.is_some());
        let r = par::seeded_vector::<f64>(p.matrix.nrows(), 8);
        let mut z = vec![0.0; r.len()];
        amg.apply(&r, &mut z).unwrap();
        let replayed = replay(&export, 0, &r);
        let scale = z.iter().fold(0.0f64, |m, v| m.max(v.abs()));
        let worst = z
            .iter()
            .zip(&replayed)
            .fold(0.0f64, |m, (a, b)| m.max((a - b).abs()));
        assert!(
            worst <= 1e-9 * scale,
            "replay differs by {worst} (scale {scale})"
        );
    }

    #[test]
    fn reused_cycle_scratch_gives_identical_applications() {
        let a = poisson3d::<f64>(20, 20, 20);
        let amg = Amg::new(
            a.as_ref(),
            AmgOptions {
                coarse_size: 300,
                ..Default::default()
            },
            SolveControl::NONE,
        )
        .unwrap();
        let r = par::seeded_vector::<f64>(a.nrows(), 2);
        let (mut first, mut second, mut fresh) = (
            vec![0.0; a.nrows()],
            vec![0.0; a.nrows()],
            vec![0.0; a.nrows()],
        );
        amg.apply(&r, &mut first).unwrap();
        amg.apply(&r, &mut second).unwrap();
        // A concurrent application (scratch held) uses freshly allocated vectors.
        let guard = amg.scratch.lock().unwrap();
        amg.apply(&r, &mut fresh).unwrap();
        drop(guard);
        assert!(first
            .iter()
            .zip(&second)
            .all(|(p, q)| p.to_bits() == q.to_bits()));
        assert!(first
            .iter()
            .zip(&fresh)
            .all(|(p, q)| p.to_bits() == q.to_bits()));
    }
}

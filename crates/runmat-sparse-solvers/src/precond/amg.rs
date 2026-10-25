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
use crate::direct::{DirectOptions, DirectSolver, FactorKind};
use crate::error::{SolverError, SolverResult};
use crate::operator::Preconditioner;
use crate::par;
use crate::spgemm;

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
        let z = if dim == 3 { coordinates[node * dim + 2] - center[2] } else { 0.0 };
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
    fn spmv(&self, x: &[f64], y: &mut [f64]) {
        match self {
            Self::Csr(a) => a.as_ref().spmv(x, y),
            Self::Bsr(a) => a.spmv(x, y),
        }
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
        Self { r: vec![0.0; n], ax: vec![0.0; n], d: vec![0.0; n], z: vec![0.0; n] }
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
    pub fn new(matrix: CsrRef<'_, f64>, options: AmgOptions, control: SolveControl<'_>) -> SolverResult<Self> {
        let n = matrix.nrows();
        let b = options.block_size.max(1);
        if n % b != 0 {
            return Err(SolverError::Dimension(format!("block size {b} does not divide {n}")));
        }
        let mut candidates = match &options.near_nullspace {
            Some(ns) => {
                if ns.vectors.len() != n * ns.count {
                    return Err(SolverError::Dimension("near-nullspace length".into()));
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
            let inverse_blocks = inverse_diagonal_blocks(&a, block)?;
            let lambda_max = spectral_radius(&a, block, &inverse_blocks);
            let rows = a.nrows();
            let stop = rows <= options.coarse_size || levels.len() + 1 >= options.max_levels;
            if stop {
                levels.push(Level {
                    a: LevelMatrix::Csr(a),
                    block,
                    inverse_blocks,
                    lambda_max,
                    p: None,
                    r: None,
                });
                break;
            }
            let aggregates = aggregate(&a, block, options.strength_threshold);
            if aggregates.count == 0 || aggregates.count * candidates.count >= rows {
                levels.push(Level {
                    a: LevelMatrix::Csr(a),
                    block,
                    inverse_blocks,
                    lambda_max,
                    p: None,
                    r: None,
                });
                break;
            }
            let (tentative, coarse_candidates) = tentative_prolongator(&aggregates, block, &candidates, rows)?;
            let p = smooth_prolongator(&a, block, &inverse_blocks, lambda_max, &tentative, options.prolongator_damping)?;
            let r = p.as_ref().transpose(false);
            let ap = spgemm::multiply(a.as_ref(), p.as_ref())?;
            let mut coarse = spgemm::multiply(r.as_ref(), ap.as_ref())?;
            coarse = repair_empty_rows(coarse);
            levels.push(Level {
                a: LevelMatrix::Csr(a),
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
        let coarse = build_coarse_solver(coarsest)?;
        let nnz0 = levels[0].a.csr().nnz().max(1) as f64;
        let statistics = AmgStatistics {
            level_rows: levels.iter().map(|l| l.a.nrows()).collect(),
            level_nnz: levels.iter().map(|l| l.a.csr().nnz()).collect(),
            operator_complexity: levels.iter().map(|l| l.a.csr().nnz() as f64).sum::<f64>() / nnz0,
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

    fn cycle(&self, level: usize, b: &[f64], x: &mut [f64], scratch: &mut [LevelScratch]) -> SolverResult<()> {
        let l = &self.levels[level];
        if level + 1 == self.levels.len() {
            match &self.coarse {
                CoarseSolver::Sparse(solver) => {
                    x.copy_from_slice(b);
                    solver.solve_in_place(x, 1)?;
                }
                CoarseSolver::Dense { factor, n } => {
                    x.copy_from_slice(b);
                    cholesky_solve(factor, *n, x, 1);
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
                        );
                    }
                }
            }
            return Ok(());
        }
        x.iter_mut().for_each(|v| *v = 0.0);
        // `scratch[0]` belongs to this level, the rest to the coarser ones.
        let (here, below) = scratch.split_first_mut().expect("scratch for every level");
        chebyshev(l, b, x, self.options.smoother_degree, self.options.chebyshev_ratio, &mut here.smoother);
        l.a.spmv(x, &mut here.ax);
        par::sub_into(b, &here.ax, &mut here.residual);
        let r = l.r.as_ref().unwrap();
        let p = l.p.as_ref().unwrap();
        r.as_ref().spmv(&here.residual, &mut here.coarse_b);
        // The coarse level zeroes (or overwrites) its iterate before use.
        self.cycle(level + 1, &here.coarse_b, &mut here.coarse_x, below)?;
        p.as_ref().spmv_scaled(1.0, &here.coarse_x, 1.0, x);
        chebyshev(l, b, x, self.options.smoother_degree, self.options.chebyshev_ratio, &mut here.smoother);
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

fn build_coarse_solver(a: &CsrMatrix<f64>) -> SolverResult<CoarseSolver> {
    let n = a.nrows();
    if n <= 400 {
        let mut dense = vec![0.0; n * n];
        for row in 0..n {
            let (cols, vals) = a.as_ref().row(row);
            for (&c, &v) in cols.iter().zip(vals) {
                dense[c * n + row] = v;
            }
        }
        if cholesky_in_place(&mut dense, n).is_ok() {
            return Ok(CoarseSolver::Dense { factor: dense, n });
        }
        return Ok(CoarseSolver::Smoothing);
    }
    match DirectSolver::<f64>::new(a.as_ref().into(), DirectOptions::new(FactorKind::Cholesky), SolveControl::NONE) {
        Ok(solver) => Ok(CoarseSolver::Sparse(solver)),
        Err(SolverError::NotPositiveDefinite { .. }) => Ok(CoarseSolver::Smoothing),
        Err(error) => Err(error),
    }
}

/// Rows with no entries (dead coarse dofs from rank-deficient aggregates) get a unit diagonal.
fn repair_empty_rows(a: CsrMatrix<f64>) -> CsrMatrix<f64> {
    let n = a.nrows();
    let needs = (0..n).any(|row| {
        let (cols, vals) = a.as_ref().row(row);
        !cols.iter().zip(vals).any(|(&c, &v)| c == row && v != 0.0)
    });
    if !needs {
        return a;
    }
    let rows: Vec<(Vec<usize>, Vec<f64>)> = (0..n)
        .map(|row| {
            let (cols, vals) = a.as_ref().row(row);
            let has_diagonal = cols.iter().zip(vals).any(|(&c, &v)| c == row && v != 0.0);
            if has_diagonal {
                (cols.to_vec(), vals.to_vec())
            } else if vals.iter().all(|v| *v == 0.0) {
                (vec![row], vec![1.0])
            } else {
                let mut entries: Vec<(usize, f64)> = cols.iter().copied().zip(vals.iter().copied()).filter(|e| e.0 != row).collect();
                let scale = entries.iter().map(|e| e.1.abs()).sum::<f64>().max(1.0);
                entries.push((row, scale));
                entries.sort_by_key(|e| e.0);
                entries.into_iter().unzip()
            }
        })
        .collect();
    CsrMatrix::from_rows(n, a.ncols(), rows)
}

fn inverse_diagonal_blocks(a: &CsrMatrix<f64>, block: usize) -> SolverResult<Vec<f64>> {
    let nodes = a.nrows() / block;
    let blocks = par::map_range(nodes, 512, |node| {
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
        // Symmetric positive definite blocks: invert through Cholesky; singular → identity
        // scaled by the largest magnitude (keeps the smoother defined on dead dofs).
        let mut factor = local.clone();
        let mut inverse = vec![0.0; block * block];
        if cholesky_in_place(&mut factor, block).is_ok() {
            for c in 0..block {
                inverse[c * block + c] = 1.0;
            }
            cholesky_solve(&factor, block, &mut inverse, block);
        } else {
            let scale = local.iter().fold(0.0f64, |m, v| m.max(v.abs()));
            let d = if scale > 0.0 { 1.0 / scale } else { 1.0 };
            for c in 0..block {
                let diagonal = local[c * block + c];
                inverse[c * block + c] = if diagonal > 0.0 { 1.0 / diagonal } else { d };
            }
        }
        inverse
    });
    Ok(blocks.into_iter().flatten().collect())
}

fn apply_inverse_blocks(block: usize, inverse: &[f64], r: &[f64], z: &mut [f64]) {
    match block {
        1 => par::mul_into(inverse, r, z),
        2 => apply_inverse_blocks_fixed::<2>(inverse, r, z),
        3 => apply_inverse_blocks_fixed::<3>(inverse, r, z),
        6 => apply_inverse_blocks_fixed::<6>(inverse, r, z),
        _ => apply_inverse_blocks_general(block, inverse, r, z),
    }
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
fn spectral_radius(a: &CsrMatrix<f64>, block: usize, inverse: &[f64]) -> f64 {
    let n = a.nrows();
    if n == 0 {
        return 1.0;
    }
    let mut v = par::seeded_vector::<f64>(n, 0x5eed);
    let mut w = vec![0.0; n];
    let mut z = vec![0.0; n];
    let mut estimate = 1.0;
    for _ in 0..20 {
        let norm = par::norm2(&v);
        if norm == 0.0 {
            break;
        }
        par::scale(1.0 / norm, &mut v);
        a.as_ref().spmv(&v, &mut w);
        apply_inverse_blocks(block, inverse, &w, &mut z);
        estimate = par::norm2(&z);
        std::mem::swap(&mut v, &mut z);
    }
    // Power iteration underestimates; pad for Chebyshev safety.
    estimate * 1.1
}

fn chebyshev(level: &Level, b: &[f64], x: &mut [f64], degree: usize, ratio: f64, scratch: &mut SmootherScratch) {
    let n = level.a.nrows();
    let upper = level.lambda_max;
    let lower = upper / ratio;
    let theta = 0.5 * (upper + lower);
    let delta = 0.5 * (upper - lower);
    let sigma = theta / delta;
    let mut rho = 1.0 / sigma;
    let (mut r, mut ax, mut d, mut z) = (&mut scratch.r, &mut scratch.ax, &mut scratch.d, &mut scratch.z);
    debug_assert_eq!(r.len(), n);
    level.a.spmv(x, &mut ax);
    par::sub_into(b, &ax, &mut r);
    apply_inverse_blocks(level.block, &level.inverse_blocks, &r, &mut z);
    par::for_each_chunk_mut(&mut d, |start, chunk| {
        for (offset, value) in chunk.iter_mut().enumerate() {
            *value = z[start + offset] / theta;
        }
    });
    for k in 0..degree.max(1) {
        par::axpy(1.0, &d, x);
        if k + 1 == degree.max(1) {
            break;
        }
        level.a.spmv(&d, &mut ax);
        par::axpy(-1.0, &ax, &mut r);
        apply_inverse_blocks(level.block, &level.inverse_blocks, &r, &mut z);
        let rho_next = 1.0 / (2.0 * sigma - rho);
        let (c1, c2) = (rho_next * rho, 2.0 * rho_next / delta);
        par::axpby(c2, &z, c1, &mut d);
        rho = rho_next;
    }
}

struct Aggregates {
    /// Aggregate of each node (`usize::MAX` = not aggregated, e.g. isolated Dirichlet nodes).
    of_node: Vec<usize>,
    count: usize,
}

/// Three-pass aggregation on the node strength graph.
fn aggregate(a: &CsrMatrix<f64>, block: usize, theta: f64) -> Aggregates {
    let nodes = a.nrows() / block;
    // Node-level block norms and strength.
    let node_graph: Vec<(Vec<usize>, Vec<f64>, f64)> = par::map_range(nodes, 512, |node| {
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
        (ids, norms.into_iter().map(f64::sqrt).collect(), diagonal.sqrt())
    });
    let strong = |node: usize| -> Vec<usize> {
        let (ids, norms, d) = &node_graph[node];
        ids.iter()
            .zip(norms)
            .filter(|(&j, &v)| v > 0.0 && v >= theta * (d * node_graph[j].2).sqrt())
            .map(|(&j, _)| j)
            .collect()
    };
    let strong_lists: Vec<Vec<usize>> = par::map_range(nodes, 512, strong);
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
            if snapshot[j] != usize::MAX && strong_lists[node].binary_search(&j).is_ok() && v > best.1 {
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
    let coarse_n = aggregates.count * k;
    let mut coarse_vectors = vec![0.0; coarse_n * k];
    // Per-aggregate QR (modified Gram–Schmidt with reorthogonalization).
    let local: Vec<(Vec<f64>, Vec<f64>)> = par::map_range(aggregates.count, 64, |agg| {
        let dofs: Vec<usize> = members[agg]
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
            if norm > 1e-10 * original.max(f64::MIN_POSITIVE) && norm > 0.0 {
                r[c * k + c] = norm;
                for i in 0..m {
                    q[c * m + i] /= norm;
                }
            } else {
                for i in 0..m {
                    q[c * m + i] = 0.0;
                }
            }
        }
        (q, r)
    });
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
    Ok((
        t,
        NearNullspace {
            count: k,
            vectors: coarse_vectors,
        },
    ))
}

fn smooth_prolongator(
    a: &CsrMatrix<f64>,
    block: usize,
    inverse: &[f64],
    lambda_max: f64,
    t: &CsrMatrix<f64>,
    damping: f64,
) -> SolverResult<CsrMatrix<f64>> {
    let omega = damping / lambda_max.max(f64::MIN_POSITIVE);
    let at = spgemm::multiply(a.as_ref(), t.as_ref())?;
    // P = T − ω D⁻¹ (A T), applying the block inverse row-block by row-block.
    let nodes = a.nrows() / block;
    let rows: Vec<Vec<(Vec<usize>, Vec<f64>)>> = par::map_range(nodes, 256, |node| {
        let inv = &inverse[node * block * block..(node + 1) * block * block];
        // Union pattern of the node's AT rows.
        let mut cols: Vec<usize> = Vec::new();
        for r in 0..block {
            cols.extend_from_slice(at.as_ref().row(node * block + r).0);
        }
        cols.sort_unstable();
        cols.dedup();
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
    });
    Ok(CsrMatrix::from_rows(a.nrows(), t.ncols(), rows.into_iter().flatten().collect()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::krylov::{pcg, KrylovOptions};
    use crate::testutil::poisson3d;

    #[test]
    fn amg_pcg_converges_in_few_iterations_on_poisson() {
        let a = poisson3d::<f64>(30, 30, 30);
        let amg = Amg::new(a.as_ref(), AmgOptions { coarse_size: 500, ..Default::default() }, SolveControl::NONE).unwrap();
        assert!(amg.statistics().level_rows.len() >= 2);
        let b = par::seeded_vector::<f64>(a.nrows(), 1);
        let mut x = vec![0.0; a.nrows()];
        let report = pcg(&a, Some(&amg), &b, &mut x, &KrylovOptions { rtol: 1e-8, max_iterations: 200, ..Default::default() }, SolveControl::NONE).unwrap();
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
            apply_inverse_blocks(l.block, &l.inverse_blocks, &r, &mut z);
            let mut d: Vec<f64> = z.iter().map(|v| v / theta).collect();
            for k in 0..degree.max(1) {
                x.iter_mut().zip(&d).for_each(|(x, d)| *x += d);
                if k + 1 == degree.max(1) {
                    break;
                }
                let ad = spmv(&l.matrix, &d);
                r.iter_mut().zip(&ad).for_each(|(r, a)| *r -= a);
                apply_inverse_blocks(l.block, &l.inverse_blocks, &r, &mut z);
                let rho_next = 1.0 / (2.0 * sigma - rho);
                let (c1, c2) = (rho_next * rho, 2.0 * rho_next / delta);
                d.iter_mut().zip(&z).for_each(|(d, z)| *d = c2 * z + c1 * *d);
                rho = rho_next;
            }
        };
        if level + 1 == export.levels.len() {
            return match &export.coarse_inverse {
                Some(inverse) => (0..n).map(|r| (0..n).map(|c| inverse[r * n + c] * b[c]).sum()).collect(),
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
            AmgOptions { block_size: 3, near_nullspace: Some(rigid_body_modes(&p.coordinates, 3)), coarse_size: 200, ..Default::default() },
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
        let worst = z.iter().zip(&replayed).fold(0.0f64, |m, (a, b)| m.max((a - b).abs()));
        assert!(worst <= 1e-9 * scale, "replay differs by {worst} (scale {scale})");
    }

    #[test]
    fn reused_cycle_scratch_gives_identical_applications() {
        let a = poisson3d::<f64>(20, 20, 20);
        let amg = Amg::new(a.as_ref(), AmgOptions { coarse_size: 300, ..Default::default() }, SolveControl::NONE).unwrap();
        let r = par::seeded_vector::<f64>(a.nrows(), 2);
        let (mut first, mut second, mut fresh) = (vec![0.0; a.nrows()], vec![0.0; a.nrows()], vec![0.0; a.nrows()]);
        amg.apply(&r, &mut first).unwrap();
        amg.apply(&r, &mut second).unwrap();
        // A concurrent application (scratch held) uses freshly allocated vectors.
        let guard = amg.scratch.lock().unwrap();
        amg.apply(&r, &mut fresh).unwrap();
        drop(guard);
        assert!(first.iter().zip(&second).all(|(p, q)| p.to_bits() == q.to_bits()));
        assert!(first.iter().zip(&fresh).all(|(p, q)| p.to_bits() == q.to_bits()));
    }
}

//! Sparse direct factorizations on faer with reusable symbolic analysis.
//!
//! Workflow:
//! 1. [`SymbolicFactorization::analyze`] computes the fill-reducing ordering and the symbolic
//!    factor once per sparsity pattern, and quotes the exact factor size ([`FactorEstimate`])
//!    before any numeric memory is committed.
//! 2. [`DirectSolver::factorize`] performs the numeric factorization (optionally against a byte
//!    budget); [`DirectSolver::refactorize`] reuses the symbolic analysis for new values on the
//!    same pattern (Newton tangents, time steps, frequency sweeps, eigen shifts).
//! 3. [`DirectSolver::solve_in_place`] / [`DirectSolver::solve_refined`] solve one or many
//!    right-hand sides; refinement uses the original matrix to remove regularization or pivoting
//!    error.
//!
//! Kinds: supernodal/simplicial Cholesky `LLᴴ` (HPD), `LDLᴴ` with optional dynamic
//! regularization by expected pivot signs (quasi-definite saddle points), Bunch–Kaufman `LBLᴴ`
//! with intranode pivoting (general Hermitian indefinite, e.g. shift-inverted pencils), and LU
//! with partial pivoting (general and complex-symmetric). Inertia is available for the
//! symmetric kinds (Sturm sequence checks in eigensolvers).
//!
//! Threading: the numeric factorization and solves use the caller's admitted pool through
//! faer's Rayon parallelism. Results do not depend on the thread count for the symmetric kinds
//! (supernode ordering is fixed by the symbolic analysis).
//!
//! Cancellation: checked between phases. faer's numeric kernels are not interruptible, so the
//! longest uninterruptible span is one numeric factorization; its cost is quoted beforehand.

mod cholesky;
mod lu;
mod qr;

pub use qr::LeastSquaresQr;

use std::sync::Arc;

use crate::control::{Progress, ProgressPhase, SolveControl};
use crate::csr::{CscRef, CsrRef};
use crate::error::{SolverError, SolverResult};
use crate::ordering::{Ordering, Permutation};
use crate::par;
use crate::scalar::Scalar;

/// Scalars accepted by the faer-backed factorizations.
pub trait FactorScalar: Scalar + faer::traits::ComplexField {}
impl FactorScalar for f64 {}
impl FactorScalar for f32 {}
impl FactorScalar for crate::scalar::c64 {}
impl FactorScalar for crate::scalar::c32 {}

/// Factorization kind.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FactorKind {
    /// `A = L Lᴴ` for Hermitian positive definite `A`.
    Cholesky,
    /// `A = L D Lᴴ` without pivoting; pair with expected pivot signs for quasi-definite systems.
    Ldlt,
    /// `A = L B Lᴴ` with Bunch–Kaufman pivoting inside supernodes (Hermitian indefinite).
    BunchKaufman,
    /// `P A Q = L U` with partial pivoting (general, including complex symmetric).
    Lu,
}

impl FactorKind {
    pub fn is_symmetric(self) -> bool {
        !matches!(self, FactorKind::Lu)
    }
    pub fn as_str(self) -> &'static str {
        match self {
            FactorKind::Cholesky => "cholesky",
            FactorKind::Ldlt => "ldlt",
            FactorKind::BunchKaufman => "bunch_kaufman",
            FactorKind::Lu => "lu",
        }
    }
}

/// Ownership contract selected before symbolic analysis and retained for numeric work.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DirectWorkspacePolicy {
    /// Performance-selected kernels may retain backend-managed packing storage.
    BackendManaged,
    /// Numeric work uses only the factor storage and the quoted explicit scratch.
    /// Supported for simplicial Cholesky and LDLT; pivoting routes require their
    /// own complete storage contract before they can accept this policy.
    CallerOwned,
}

/// Options for symbolic analysis and numeric factorization.
#[derive(Debug, Clone, PartialEq)]
pub struct DirectOptions {
    pub kind: FactorKind,
    pub workspace: DirectWorkspacePolicy,
    pub ordering: Ordering,
    /// Expected pivot signs (original numbering) for [`FactorKind::Ldlt`]: pivots with the wrong
    /// sign or magnitude below `regularization_epsilon` are replaced by `sign *
    /// regularization_delta`; refinement then removes the perturbation.
    pub pivot_signs: Option<Vec<i8>>,
    pub regularization_delta: f64,
    pub regularization_epsilon: f64,
    /// Maximum iterative refinement steps in [`DirectSolver::solve_refined`].
    pub refinement_steps: usize,
    /// Refinement stops once `‖b − A x‖ ≤ refinement_tolerance · ‖b‖`.
    pub refinement_tolerance: f64,
}

impl DirectOptions {
    pub fn new(kind: FactorKind) -> Self {
        Self {
            kind,
            workspace: DirectWorkspacePolicy::BackendManaged,
            ordering: Ordering::Auto,
            pivot_signs: None,
            regularization_delta: 0.0,
            regularization_epsilon: 0.0,
            refinement_steps: if kind == FactorKind::Cholesky { 0 } else { 3 },
            refinement_tolerance: 1.0e-14,
        }
    }
}

/// Factor payload and explicit numeric scratch, known after symbolic analysis.
/// Backend-managed kernels may retain additional packing storage; only a
/// CallerOwned, non-nominal analysis is eligible for finite numeric admission.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FactorEstimate {
    pub dimension: usize,
    /// Stored scalars of the factor(s).
    pub factor_entries: u64,
    /// Factor storage plus integer structure (bytes).
    pub factor_bytes: u64,
    /// Peak numeric factorization workspace (bytes).
    pub workspace_bytes: u64,
    /// Floating point operations of the numeric factorization (real flops; complex counts 4×).
    pub flops: f64,
    /// True when the factor size depends on pivoting (LU) and the figures are a nominal estimate
    /// from the symmetrized pattern rather than an exact count.
    pub nominal: bool,
}

impl FactorEstimate {
    /// Factor payload plus explicit scratch, excluding borrowed input arrays.
    /// This is a numeric admission quote only under CallerOwned policy.
    pub fn peak_bytes(&self) -> u64 {
        self.factor_bytes.saturating_add(self.workspace_bytes)
    }
}

/// Which storage the input arrays describe.
#[derive(Debug, Clone, Copy)]
pub enum SparseInput<'a, T> {
    Csr(CsrRef<'a, T>),
    Csc(CscRef<'a, T>),
}

impl<'a, T: Scalar> SparseInput<'a, T> {
    pub fn nrows(&self) -> usize {
        match self {
            SparseInput::Csr(m) => m.nrows(),
            SparseInput::Csc(m) => m.nrows(),
        }
    }
    pub fn ncols(&self) -> usize {
        match self {
            SparseInput::Csr(m) => m.ncols(),
            SparseInput::Csc(m) => m.ncols(),
        }
    }
    /// The arrays as a faer column matrix and whether that matrix is `Aᵀ` (CSR input).
    pub(crate) fn as_columns(&self) -> (CscRef<'a, T>, bool) {
        match self {
            SparseInput::Csr(m) => (m.as_transpose_csc(), true),
            SparseInput::Csc(m) => (*m, false),
        }
    }
    /// Evaluate the original operator's residual; CSC uses caller-owned row-error scratch.
    pub(crate) fn residual(&self, x: &[T], b: &[T], r: &mut [T], scratch: &mut [T]) {
        match self {
            SparseInput::Csr(m) => {
                m.residual(x, b, r);
            }
            SparseInput::Csc(m) => {
                m.residual(x, b, r, scratch);
            }
        }
    }
    pub(crate) fn pattern_fingerprint(&self) -> PatternFingerprint {
        let (columns, _) = self.as_columns();
        PatternFingerprint::of(columns.col_ptr(), columns.row_idx())
    }
}

impl<'a, T: Scalar> From<CsrRef<'a, T>> for SparseInput<'a, T> {
    fn from(value: CsrRef<'a, T>) -> Self {
        SparseInput::Csr(value)
    }
}
impl<'a, T: Scalar> From<CscRef<'a, T>> for SparseInput<'a, T> {
    fn from(value: CscRef<'a, T>) -> Self {
        SparseInput::Csc(value)
    }
}

/// Cheap identity for a sparsity pattern (dimension, nnz and a hash of the index arrays).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PatternFingerprint {
    pub dimension: usize,
    pub entries: usize,
    pub hash: u64,
}

impl PatternFingerprint {
    pub fn of(ptr: &[usize], idx: &[usize]) -> Self {
        let mut h: u64 = 0xcbf2_9ce4_8422_2325;
        for &value in ptr.iter().chain(idx.iter()) {
            h ^= value as u64;
            h = h.wrapping_mul(0x100_0000_01b3);
        }
        Self {
            dimension: ptr.len().saturating_sub(1),
            entries: idx.len(),
            hash: h,
        }
    }
}

pub(crate) enum SymbolicInner {
    Cholesky(faer::sparse::linalg::cholesky::SymbolicCholesky<usize>),
    Lu(faer::sparse::linalg::lu::SymbolicLu<usize>),
}

/// Ordering plus symbolic factor for one sparsity pattern; shareable across numeric
/// factorizations and threads.
pub struct SymbolicFactorization {
    pub(crate) kind: FactorKind,
    pub(crate) workspace: DirectWorkspacePolicy,
    pub(crate) inner: SymbolicInner,
    pub(crate) fingerprint: PatternFingerprint,
    pub(crate) input_transposed: bool,
    pub(crate) estimate: FactorEstimate,
    pub(crate) scalar_bytes: usize,
    pub(crate) complex: bool,
}

mod reference_work;
pub use reference_work::{DirectReferenceWork, ReferenceWorkError};

impl std::fmt::Debug for SymbolicFactorization {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("SymbolicFactorization")
            .field("kind", &self.kind)
            .field("fingerprint", &self.fingerprint)
            .field("estimate", &self.estimate)
            .finish()
    }
}

impl SymbolicFactorization {
    /// Orders and symbolically factors the pattern of `matrix` for scalar type `T`.
    pub fn analyze<T: FactorScalar>(
        matrix: SparseInput<'_, T>,
        options: &DirectOptions,
        control: SolveControl<'_>,
    ) -> SolverResult<Arc<Self>> {
        if options.workspace == DirectWorkspacePolicy::CallerOwned
            && !matches!(options.kind, FactorKind::Cholesky | FactorKind::Ldlt)
        {
            return Err(SolverError::Unsupported(
                "caller-owned numeric workspace requires nonpivoting Cholesky or LDLT".into(),
            ));
        }
        if matrix.nrows() != matrix.ncols() {
            return Err(SolverError::Dimension(
                "direct factorization requires a square matrix".into(),
            ));
        }
        control.check()?;
        control.report(Progress {
            phase: ProgressPhase::Ordering,
            iteration: 0,
            residual: 0.0,
            converged: 0,
        });
        let symbolic = match options.kind {
            FactorKind::Lu => lu::analyze::<T>(matrix, options)?,
            kind => {
                let permutation = symmetric_ordering(matrix, &options.ordering)?;
                control.check()?;
                control.report(Progress {
                    phase: ProgressPhase::SymbolicFactorization,
                    iteration: 0,
                    residual: 0.0,
                    converged: 0,
                });
                cholesky::analyze::<T>(matrix, kind, options.workspace, &permutation)?
            }
        };
        Ok(Arc::new(symbolic))
    }

    /// Symbolic analysis of a CSR pattern alone (resource planning before values exist).
    pub fn analyze_pattern<T: FactorScalar>(
        n: usize,
        row_ptr: &[usize],
        col_idx: &[usize],
        options: &DirectOptions,
        control: SolveControl<'_>,
    ) -> SolverResult<Arc<Self>> {
        let values = try_vec(col_idx.len(), T::zero())?;
        let matrix = CsrRef::try_new(n, n, row_ptr, col_idx, &values)?;
        Self::analyze::<T>(matrix.into(), options, control)
    }

    pub fn kind(&self) -> FactorKind {
        self.kind
    }
    pub fn workspace_policy(&self) -> DirectWorkspacePolicy {
        self.workspace
    }
    pub fn estimate(&self) -> FactorEstimate {
        self.estimate
    }
    pub fn fingerprint(&self) -> PatternFingerprint {
        self.fingerprint
    }
    pub fn dimension(&self) -> usize {
        self.estimate.dimension
    }
}

/// Fill-reducing ordering of `A + Aᵀ` for either input layout.
pub fn symmetric_ordering<T: Scalar>(
    matrix: SparseInput<'_, T>,
    ordering: &Ordering,
) -> SolverResult<Permutation> {
    match matrix {
        SparseInput::Csr(m) => crate::ordering::order(m, ordering),
        // The pattern of A + Aᵀ is the same whether the arrays describe A or Aᵀ.
        SparseInput::Csc(m) => crate::ordering::order(m.as_transpose_csr(), ordering),
    }
}

/// Signed eigenvalue counts of a symmetric factorization (Sylvester's law of inertia).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Inertia {
    pub positive: usize,
    pub negative: usize,
    pub zero: usize,
}

pub(crate) enum NumericInner<T> {
    Llt(Vec<T>),
    Ldlt(Vec<T>),
    Lblt {
        values: Vec<T>,
        subdiag: Vec<T>,
        perm: Vec<usize>,
        inverse: Vec<usize>,
    },
    Lu(Box<faer::sparse::linalg::lu::NumericLu<usize, T>>),
}

/// A numeric factorization bound to its (shared) symbolic analysis.
pub struct DirectSolver<T: FactorScalar> {
    symbolic: Arc<SymbolicFactorization>,
    numeric: NumericInner<T>,
    options: DirectOptions,
    /// Pivots replaced by dynamic regularization in the last factorization.
    regularized_pivots: usize,
}

impl<T: FactorScalar> std::fmt::Debug for DirectSolver<T> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("DirectSolver")
            .field("symbolic", &self.symbolic)
            .field("regularized_pivots", &self.regularized_pivots)
            .finish()
    }
}

/// Report of a refined solve.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DirectSolveReport {
    pub refinement_steps: usize,
    /// `‖b − A x‖ / ‖b‖` after the last step (max over right-hand sides).
    pub relative_residual: f64,
}

impl<T: FactorScalar> DirectSolver<T> {
    /// Symbolic analysis plus numeric factorization.
    pub fn new(
        matrix: SparseInput<'_, T>,
        options: DirectOptions,
        control: SolveControl<'_>,
    ) -> SolverResult<Self> {
        let symbolic = SymbolicFactorization::analyze::<T>(matrix, &options, control)?;
        Self::factorize(symbolic, matrix, options, None, control)
    }

    /// Numeric factorization on an existing analysis. A finite budget requires
    /// CallerOwned scratch and an exact quote; refusal precedes numeric allocation.
    pub fn factorize(
        symbolic: Arc<SymbolicFactorization>,
        matrix: SparseInput<'_, T>,
        options: DirectOptions,
        budget_bytes: Option<u64>,
        control: SolveControl<'_>,
    ) -> SolverResult<Self> {
        if options.kind != symbolic.kind || options.workspace != symbolic.workspace {
            return Err(SolverError::Unsupported(
                "numeric strategy differs from the quoted symbolic factorization".into(),
            ));
        }
        if let Some(budget) = budget_bytes {
            if symbolic.workspace != DirectWorkspacePolicy::CallerOwned || symbolic.estimate.nominal
            {
                return Err(SolverError::Unsupported(
                    "finite numeric admission requires an exact caller-owned workspace quote"
                        .into(),
                ));
            }
            let required = symbolic.estimate.peak_bytes();
            if required > budget {
                return Err(SolverError::OverBudget { required, budget });
            }
        }
        check_pattern(&symbolic, matrix)?;
        control.check()?;
        control.report(Progress {
            phase: ProgressPhase::NumericFactorization,
            iteration: 0,
            residual: 0.0,
            converged: 0,
        });
        let (numeric, regularized) = match symbolic.kind {
            FactorKind::Cholesky | FactorKind::Ldlt => reference_work::with_direct_usage(
                &symbolic,
                reference_work::DirectPhase::Factorization,
                || cholesky::factorize::<T>(&symbolic, matrix, &options, None),
            )?,
            FactorKind::Lu => (lu::factorize::<T>(&symbolic, matrix, None)?, 0),
            _ => cholesky::factorize::<T>(&symbolic, matrix, &options, None)?,
        };
        control.check()?;
        Ok(Self {
            symbolic,
            numeric,
            options,
            regularized_pivots: regularized,
        })
    }

    /// New values on the analyzed pattern; reuses the symbolic factor and the factor storage.
    pub fn refactorize(
        &mut self,
        matrix: SparseInput<'_, T>,
        control: SolveControl<'_>,
    ) -> SolverResult<()> {
        check_pattern(&self.symbolic, matrix)?;
        control.check()?;
        control.report(Progress {
            phase: ProgressPhase::NumericFactorization,
            iteration: 0,
            residual: 0.0,
            converged: 0,
        });
        if matches!(self.symbolic.kind, FactorKind::Cholesky | FactorKind::Ldlt) {
            let symbolic = Arc::clone(&self.symbolic);
            return reference_work::with_direct_usage(
                &symbolic,
                reference_work::DirectPhase::Factorization,
                || self.refactorize_numeric(matrix),
            );
        }
        self.refactorize_numeric(matrix)
    }

    fn refactorize_numeric(&mut self, matrix: SparseInput<'_, T>) -> SolverResult<()> {
        let previous = std::mem::replace(&mut self.numeric, NumericInner::Llt(Vec::new()));
        let (numeric, regularized) = match self.symbolic.kind {
            FactorKind::Lu => (
                lu::factorize::<T>(&self.symbolic, matrix, Some(previous))?,
                0,
            ),
            _ => cholesky::factorize::<T>(&self.symbolic, matrix, &self.options, Some(previous))?,
        };
        self.numeric = numeric;
        self.regularized_pivots = regularized;
        Ok(())
    }

    pub fn symbolic(&self) -> &Arc<SymbolicFactorization> {
        &self.symbolic
    }
    pub fn kind(&self) -> FactorKind {
        self.symbolic.kind
    }
    pub fn dimension(&self) -> usize {
        self.symbolic.dimension()
    }
    pub fn regularized_pivots(&self) -> usize {
        self.regularized_pivots
    }
    pub fn options(&self) -> &DirectOptions {
        &self.options
    }

    /// Solves `A X = B` in place for `k` column-major right-hand sides.
    pub fn solve_in_place(&self, rhs: &mut [T], k: usize) -> SolverResult<()> {
        let n = self.dimension();
        let expected = n.checked_mul(k).ok_or(SolverError::IndexOverflow)?;
        if rhs.len() != expected {
            return Err(SolverError::Dimension(format!(
                "rhs has {} entries, expected {}",
                rhs.len(),
                expected
            )));
        }
        if n == 0 || k == 0 {
            return Ok(());
        }
        match &self.numeric {
            numeric @ (NumericInner::Llt(_) | NumericInner::Ldlt(_)) => {
                reference_work::with_direct_usage(
                    &self.symbolic,
                    reference_work::DirectPhase::Solve { rhs_columns: k },
                    || {
                        cholesky::solve(&self.symbolic, numeric, rhs, k);
                        Ok(())
                    },
                )?
            }
            NumericInner::Lu(numeric) => lu::solve(&self.symbolic, numeric, rhs, k)?,
            numeric => cholesky::solve(&self.symbolic, numeric, rhs, k),
        }
        Ok(())
    }

    /// Solves `Aᵀ X = B` in place (for condition estimation and adjoint problems).
    pub fn solve_transpose_in_place(&self, rhs: &mut [T], k: usize) -> SolverResult<()> {
        let n = self.dimension();
        if rhs.len() != n.checked_mul(k).ok_or(SolverError::IndexOverflow)? {
            return Err(SolverError::Dimension("transpose rhs length".into()));
        }
        if n == 0 || k == 0 {
            return Ok(());
        }
        match &self.numeric {
            numeric @ (NumericInner::Llt(_) | NumericInner::Ldlt(_)) => {
                reference_work::with_direct_usage(
                    &self.symbolic,
                    reference_work::DirectPhase::Solve { rhs_columns: k },
                    || {
                        for value in rhs.iter_mut() {
                            *value = value.conj();
                        }
                        cholesky::solve(&self.symbolic, numeric, rhs, k);
                        for value in rhs.iter_mut() {
                            *value = value.conj();
                        }
                        Ok(())
                    },
                )
            }
            NumericInner::Lu(numeric) => lu::solve_transpose(&self.symbolic, numeric, rhs, k),
            _ => {
                // Hermitian A: Aᵀ = conj(A), so Aᵀ x = b  ⇔  A conj(x) = conj(b).
                for value in rhs.iter_mut() {
                    *value = value.conj();
                }
                self.solve_in_place(rhs, k)?;
                for value in rhs.iter_mut() {
                    *value = value.conj();
                }
                Ok(())
            }
        }
    }

    /// Hager–Higham estimate of `‖A⁻¹‖₁` (a few solves with `A` and `Aᵀ`; LAPACK `rcond` style).
    pub fn estimate_inverse_norm1(&self) -> SolverResult<f64> {
        let n = self.dimension();
        estimate_norm1(
            n,
            |x| {
                let mut y = x.to_vec();
                self.solve_in_place(&mut y, 1)?;
                Ok(y)
            },
            |x| {
                let mut y = x.to_vec();
                self.solve_transpose_in_place(&mut y, 1)?;
                Ok(y)
            },
        )
    }

    /// Estimate of `‖I − A F⁻¹‖₁` for the factorization `F ≈ A` (the right-inverse defect of
    /// the factorization). A value below one certifies that `A` is nonsingular whenever the
    /// estimate bounds the true norm; the caller applies its own safety factor.
    pub fn estimate_right_inverse_defect1(&self, matrix: SparseInput<'_, T>) -> SolverResult<f64> {
        let n = self.dimension();
        let apply = |x: &[T], transpose: bool| -> SolverResult<Vec<T>> {
            let mut y = x.to_vec();
            if transpose {
                // (I − A F⁻¹)ᵀ x = x − F⁻ᵀ Aᵀ x
                let mut at = vec![T::zero(); n];
                matrix_transpose_apply(matrix, x, &mut at);
                self.solve_transpose_in_place(&mut at, 1)?;
                for i in 0..n {
                    y[i] -= at[i];
                }
            } else {
                let mut z = x.to_vec();
                self.solve_in_place(&mut z, 1)?;
                let mut az = vec![T::zero(); n];
                matrix_apply(matrix, &z, &mut az);
                for i in 0..n {
                    y[i] -= az[i];
                }
            }
            Ok(y)
        };
        estimate_norm1(n, |x| apply(x, false), |x| apply(x, true))
    }

    /// Solves with iterative refinement against the original matrix (which must be the matrix
    /// that was factorized). `x` receives the solution.
    pub fn solve_refined(
        &self,
        matrix: SparseInput<'_, T>,
        b: &[T],
        x: &mut [T],
        k: usize,
    ) -> SolverResult<DirectSolveReport> {
        let n = self.dimension();
        if b.len() != n * k || x.len() != n * k {
            return Err(SolverError::Dimension("refined solve dimensions".into()));
        }
        x.copy_from_slice(b);
        self.solve_in_place(x, k)?;
        let mut residual = vec![T::zero(); n];
        let mut correction = vec![T::zero(); n];
        let mut worst = 0.0f64;
        let mut steps_taken = 0usize;
        for column in 0..k {
            let bc = &b[column * n..(column + 1) * n];
            let b_norm = par::norm2(bc);
            let xc = &mut x[column * n..(column + 1) * n];
            matrix.residual(xc, bc, &mut residual, &mut correction);
            let mut relative = if b_norm > 0.0 {
                par::norm2(&residual) / b_norm
            } else {
                0.0
            };
            let mut steps = 0;
            while steps < self.options.refinement_steps
                && relative > self.options.refinement_tolerance
            {
                correction.copy_from_slice(&residual);
                self.solve_in_place(&mut correction, 1)?;
                for i in 0..n {
                    xc[i] += correction[i];
                }
                matrix.residual(xc, bc, &mut residual, &mut correction);
                let next = if b_norm > 0.0 {
                    par::norm2(&residual) / b_norm
                } else {
                    0.0
                };
                steps += 1;
                if next.partial_cmp(&(0.5 * relative)) != Some(std::cmp::Ordering::Less) {
                    // xc already contains this correction. Report its measured residual,
                    // not the better value belonging to the preceding iterate.
                    relative = next;
                    break;
                }
                relative = next;
            }
            steps_taken = steps_taken.max(steps);
            worst = worst.max(relative);
        }
        Ok(DirectSolveReport {
            refinement_steps: steps_taken,
            relative_residual: worst,
        })
    }

    /// Inertia of a symmetric factorization (`None` for LU).
    pub fn inertia(&self) -> Option<Inertia> {
        cholesky::inertia(&self.symbolic, &self.numeric)
    }

    /// Pivots of a Cholesky or LDLᴴ factorization indexed by original row: `l_jj²` for Cholesky,
    /// `d_j` for LDLᴴ. `None` for Bunch–Kaufman (2×2 pivots) and LU.
    pub fn pivots(&self) -> Option<Vec<f64>> {
        cholesky::pivots(&self.symbolic, &self.numeric)
    }
}

fn matrix_apply<T: Scalar>(matrix: SparseInput<'_, T>, x: &[T], y: &mut [T]) {
    match matrix {
        SparseInput::Csr(m) => m.spmv(x, y),
        SparseInput::Csc(m) => m.as_transpose_csr().spmv_transpose(x, y, false),
    }
}

fn matrix_transpose_apply<T: Scalar>(matrix: SparseInput<'_, T>, x: &[T], y: &mut [T]) {
    match matrix {
        SparseInput::Csr(m) => m.spmv_transpose(x, y, false),
        SparseInput::Csc(m) => m.as_transpose_csr().spmv(x, y),
    }
}

/// Hager's 1-norm estimator (Higham's refinement, at most five sweeps) for an operator given by
/// its action and the action of its transpose.
pub(crate) fn estimate_norm1<T: Scalar>(
    n: usize,
    apply: impl Fn(&[T]) -> SolverResult<Vec<T>>,
    apply_transpose: impl Fn(&[T]) -> SolverResult<Vec<T>>,
) -> SolverResult<f64> {
    if n == 0 {
        return Ok(0.0);
    }
    let mut x = vec![T::from_f64(1.0 / n as f64); n];
    let mut estimate = 0.0f64;
    let mut previous_index = usize::MAX;
    for sweep in 0..5 {
        let y = apply(&x)?;
        let norm: f64 = y.iter().map(|v| v.abs()).sum();
        if sweep > 0 && norm <= estimate {
            break;
        }
        estimate = norm;
        let signs: Vec<T> = y
            .iter()
            .map(|v| {
                let a = v.abs();
                if a == 0.0 {
                    T::one()
                } else {
                    v.scale(1.0 / a)
                }
            })
            .collect();
        let z = apply_transpose(&signs)?;
        let (index, max) =
            z.iter()
                .enumerate()
                .map(|(i, v)| (i, v.abs()))
                .fold(
                    (0usize, 0.0f64),
                    |best, item| if item.1 > best.1 { item } else { best },
                );
        let ztx: f64 = z.iter().zip(&x).map(|(a, b)| (a.conj() * *b).re()).sum();
        if max <= ztx || index == previous_index {
            break;
        }
        previous_index = index;
        x.iter_mut().for_each(|v| *v = T::zero());
        x[index] = T::one();
    }
    // Higham's alternating-sign vector guards against estimator underestimation.
    let alternating: Vec<T> = (0..n)
        .map(|i| {
            let sign = if i % 2 == 0 { 1.0 } else { -1.0 };
            T::from_f64(sign * (1.0 + i as f64 / (n.max(2) - 1) as f64))
        })
        .collect();
    let y = apply(&alternating)?;
    let alternative = 2.0 * y.iter().map(|v| v.abs()).sum::<f64>() / (3.0 * n as f64);
    Ok(estimate.max(alternative))
}

fn check_pattern<T: Scalar>(
    symbolic: &SymbolicFactorization,
    matrix: SparseInput<'_, T>,
) -> SolverResult<()> {
    let (_, transposed) = matrix.as_columns();
    if transposed != symbolic.input_transposed
        || matrix.pattern_fingerprint() != symbolic.fingerprint
    {
        return Err(SolverError::PatternMismatch);
    }
    if symbolic.scalar_bytes != std::mem::size_of::<T>() || symbolic.complex != T::IS_COMPLEX {
        return Err(SolverError::Unsupported(
            "symbolic analysis was computed for another scalar type".into(),
        ));
    }
    Ok(())
}

/// Bytes of a `StackReq`.
pub(crate) fn stack_bytes(req: faer::dyn_stack::StackReq) -> u64 {
    req.size_bytes() as u64
}

pub(crate) fn memory(req: faer::dyn_stack::StackReq) -> SolverResult<faer::dyn_stack::MemBuffer> {
    faer::dyn_stack::MemBuffer::try_new(req).map_err(|_| SolverError::OutOfMemory {
        bytes: req.size_bytes() as u64,
    })
}

pub(crate) fn try_vec<T: Clone>(len: usize, value: T) -> SolverResult<Vec<T>> {
    let mut values = Vec::new();
    values
        .try_reserve_exact(len)
        .map_err(|_| SolverError::OutOfMemory {
            bytes: (len * std::mem::size_of::<T>()) as u64,
        })?;
    values.resize(len, value);
    Ok(values)
}

#[cfg(test)]
mod tests;

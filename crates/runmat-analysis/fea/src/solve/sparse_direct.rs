//! Narrow sparse direct factor/solve boundary for indefinite and complex operators.
//!
//! Electromagnetic curl-curl systems (and their terminal/gauge augmentations) are
//! indefinite or complex-symmetric with a near-null gradient space, where unpreconditioned
//! Krylov methods stall. This module is the single entry point those kernels use for an
//! exact sparse LU with partial pivoting. It owns no physics; callers assemble triplets in
//! SI units and receive the solution plus the measured relative residual of the ORIGINAL
//! assembled system. Every authored triplet is retained through deterministic duplicate
//! coalescing; that CSR is the actual operator factored, refined and certified. Measuring
//! raw triplets in another summation order would certify a different floating-point operator.
//! The factorization is the shared sparse layer's direct LU (`runmat-sparse-solvers`),
//! run on the ambient admitted CPU plan with FEA cancellation.
use runmat_sparse_solvers::c64;
use runmat_sparse_solvers::direct::{
    DirectOptions, DirectSolveReport, DirectSolver, FactorKind, FactorScalar,
};
use runmat_sparse_solvers::{CsrMatrix, TripletBuilder};

/// Factor and solve one square system with the shared direct LU.
fn shared_lu<T: FactorScalar + Send + Sync>(
    dimension: usize,
    matrix: CsrMatrix<T>,
    rhs: &[T],
) -> Result<(Vec<T>, DirectSolveReport, CsrMatrix<T>), SparseDirectError> {
    let plan = super::shared_sparse::ambient_plan();
    super::shared_sparse::with_plan(plan, || {
        let solver = DirectSolver::<T>::new(
            matrix.as_ref().into(),
            DirectOptions::new(FactorKind::Lu),
            super::shared_sparse::control(),
        )
        .map_err(|error| SparseDirectError::Factorization(error.to_string()))?;
        let mut x = vec![T::zero(); dimension];
        let report = solver
            .solve_refined(matrix.as_ref().into(), rhs, &mut x, 1)
            .map_err(|error| SparseDirectError::Factorization(error.to_string()))?;
        Ok((x, report, matrix))
    })
}

#[derive(Debug, Clone, PartialEq, thiserror::Error)]
pub enum SparseDirectError {
    #[error("sparse direct system is malformed: {0}")]
    Invalid(String),
    #[error("sparse direct factorization failed: {0}")]
    Factorization(String),
    #[error("sparse direct solve produced non-finite values (numerically singular system)")]
    Breakdown,
}

/// Complex triplet entry; duplicate (row, column) entries are summed.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ComplexEntry {
    pub row: usize,
    pub column: usize,
    pub real: f64,
    pub imaginary: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ComplexDirectSolution {
    /// Interleaved per unknown: (real, imaginary).
    pub solution: Vec<(f64, f64)>,
    /// Compensated ||A x - b|| / ||b|| on the factorized, duplicate-coalesced
    /// original operator (or ||A x - b|| when b = 0).
    pub relative_residual: f64,
    /// Compensated residual reported by the shared LU refinement, before the
    /// independent acceptance check on its retained canonical operator.
    pub refinement_relative_residual: f64,
    pub refinement_steps: usize,
}

#[derive(Debug, Clone, PartialEq)]
pub struct RealDirectSolution {
    pub solution: Vec<f64>,
    /// Compensated residual on the same duplicate-coalesced operator as LU.
    pub relative_residual: f64,
}

/// Solve a square complex sparse system with LU and partial pivoting.
pub fn solve_complex(
    dimension: usize,
    entries: &[ComplexEntry],
    rhs: &[(f64, f64)],
) -> Result<ComplexDirectSolution, SparseDirectError> {
    if rhs.len() != dimension {
        return Err(SparseDirectError::Invalid(
            "rhs length differs from dimension".into(),
        ));
    }
    if dimension == 0 {
        return Ok(ComplexDirectSolution {
            solution: Vec::new(),
            relative_residual: 0.0,
            refinement_relative_residual: 0.0,
            refinement_steps: 0,
        });
    }
    let matrix = complex_operator(dimension, entries)?;
    let rhs_values = complex_rhs(dimension, rhs)?;
    let (values, report, matrix) = shared_lu(dimension, matrix, &rhs_values)?;
    if values
        .iter()
        .any(|v| !v.re.is_finite() || !v.im.is_finite())
    {
        return Err(SparseDirectError::Breakdown);
    }
    let relative_residual = certified_residual(&matrix, &rhs_values, &values);
    Ok(ComplexDirectSolution {
        solution: values.into_iter().map(|z| (z.re, z.im)).collect(),
        relative_residual,
        refinement_relative_residual: report.relative_residual,
        refinement_steps: report.refinement_steps,
    })
}

fn complex_operator(
    dimension: usize,
    entries: &[ComplexEntry],
) -> Result<CsrMatrix<c64>, SparseDirectError> {
    let mut triplets = TripletBuilder::<c64>::with_capacity(dimension, dimension, entries.len());
    for entry in entries {
        if entry.row >= dimension
            || entry.column >= dimension
            || !entry.real.is_finite()
            || !entry.imaginary.is_finite()
        {
            return Err(SparseDirectError::Invalid(
                "entry index out of range or non-finite".into(),
            ));
        }
        triplets.push(
            entry.row,
            entry.column,
            c64::new(entry.real, entry.imaginary),
        );
    }
    Ok(triplets.build())
}

fn complex_rhs(dimension: usize, rhs: &[(f64, f64)]) -> Result<Vec<c64>, SparseDirectError> {
    if rhs.len() != dimension {
        return Err(SparseDirectError::Invalid(
            "rhs length differs from dimension".into(),
        ));
    }
    if rhs.iter().any(|(a, b)| !a.is_finite() || !b.is_finite()) {
        return Err(SparseDirectError::Invalid("non-finite rhs".into()));
    }
    Ok(rhs.iter().map(|(a, b)| c64::new(*a, *b)).collect())
}

fn certified_residual<T: FactorScalar>(matrix: &CsrMatrix<T>, rhs: &[T], values: &[T]) -> f64 {
    let mut residual = vec![T::zero(); rhs.len()];
    matrix.as_ref().residual(values, rhs, &mut residual);
    if rhs
        .iter()
        .chain(&residual)
        .any(|z| !z.re().is_finite() || !z.im().is_finite())
    {
        return f64::INFINITY;
    }
    let norm = |v: &[T], scale: f64| {
        v.iter().fold(0.0_f64, |n, z| {
            n.hypot(z.re() / scale).hypot(z.im() / scale)
        })
    };
    if rhs.iter().all(|z| z.re() == 0.0 && z.im() == 0.0) {
        return norm(&residual, 1.0);
    }
    // Form the ratio in one common scale. Even hypot can overflow when a
    // finite RHS has a norm above MAX; dividing a finite defect by that
    // infinity would falsely certify a zero relative residual.
    let scale = rhs
        .iter()
        .chain(&residual)
        .fold(0.0_f64, |s, z| s.max(z.re().abs()).max(z.im().abs()));
    norm(&residual, scale) / norm(rhs, scale)
}

/// Certify a retained solution against the same duplicate-coalesced operator and
/// compensated residual used by the shared LU refinement. Rebuilding the CSR
/// from original triplets preserves every contribution and its deterministic
/// push order, so cold validation cannot choose a different residual arithmetic.
pub(crate) fn complex_relative_residual(
    dimension: usize,
    entries: &[ComplexEntry],
    rhs: &[(f64, f64)],
    solution: &[(f64, f64)],
) -> Result<f64, SparseDirectError> {
    if solution.len() != dimension
        || solution
            .iter()
            .any(|(a, b)| !a.is_finite() || !b.is_finite())
    {
        return Err(SparseDirectError::Invalid(
            "solution length differs from dimension or is non-finite".into(),
        ));
    }
    let matrix = complex_operator(dimension, entries)?;
    let rhs = complex_rhs(dimension, rhs)?;
    let values: Vec<c64> = solution.iter().map(|(a, b)| c64::new(*a, *b)).collect();
    Ok(certified_residual(&matrix, &rhs, &values))
}

/// Solve a square real sparse system with LU and partial pivoting.
pub fn solve_real(
    dimension: usize,
    entries: &[(usize, usize, f64)],
    rhs: &[f64],
) -> Result<RealDirectSolution, SparseDirectError> {
    if rhs.len() != dimension {
        return Err(SparseDirectError::Invalid(
            "rhs length differs from dimension".into(),
        ));
    }
    if dimension == 0 {
        return Ok(RealDirectSolution {
            solution: Vec::new(),
            relative_residual: 0.0,
        });
    }
    let mut triplets = TripletBuilder::<f64>::with_capacity(dimension, dimension, entries.len());
    for &(row, column, value) in entries {
        if row >= dimension || column >= dimension || !value.is_finite() {
            return Err(SparseDirectError::Invalid(
                "entry index out of range or non-finite".into(),
            ));
        }
        triplets.push(row, column, value);
    }
    if rhs.iter().any(|v| !v.is_finite()) {
        return Err(SparseDirectError::Invalid("non-finite rhs".into()));
    }
    let (values, _, matrix) = shared_lu(dimension, triplets.build(), rhs)?;
    if values.iter().any(|v| !v.is_finite()) {
        return Err(SparseDirectError::Breakdown);
    }
    let relative_residual = certified_residual(&matrix, rhs, &values);
    Ok(RealDirectSolution {
        solution: values,
        relative_residual,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn complex_indefinite_system_is_solved_exactly() {
        // [[1, i], [i, -1+2i]] x = [1, 0]
        let entries = [
            ComplexEntry {
                row: 0,
                column: 0,
                real: 1.0,
                imaginary: 0.0,
            },
            ComplexEntry {
                row: 0,
                column: 1,
                real: 0.0,
                imaginary: 1.0,
            },
            ComplexEntry {
                row: 1,
                column: 0,
                real: 0.0,
                imaginary: 1.0,
            },
            ComplexEntry {
                row: 1,
                column: 1,
                real: -0.5,
                imaginary: 1.0,
            },
            ComplexEntry {
                row: 1,
                column: 1,
                real: -0.5,
                imaginary: 1.0,
            },
        ];
        let solved = solve_complex(2, &entries, &[(1.0, 0.0), (0.0, 0.0)]).unwrap();
        assert!(solved.relative_residual < 1e-14, "{solved:?}");
        assert_eq!(
            solved.relative_residual,
            complex_relative_residual(2, &entries, &[(1.0, 0.0), (0.0, 0.0)], &solved.solution)
                .unwrap()
        );
        assert!(solved.refinement_relative_residual.is_finite());
        // x1 = -i / (-1+2i - i*i) = -i/(2i) = -0.5 ; x0 = 1 - i*x1 = 1 + 0.5i
        let (a, b) = solved.solution[0];
        assert!((a - 1.0).abs() < 1e-14 && (b - 0.5).abs() < 1e-14);
        let (c, d) = solved.solution[1];
        assert!((c + 0.5).abs() < 1e-14 && d.abs() < 1e-14);
        let identity = (0..4)
            .map(|i| ComplexEntry {
                row: i,
                column: i,
                real: 1.0,
                imaginary: 0.0,
            })
            .collect::<Vec<_>>();
        let large_rhs = [(1e308, 0.0); 4];
        let wrong = [(1.1e308, 0.0); 4];
        let relative = complex_relative_residual(4, &identity, &large_rhs, &wrong).unwrap();
        assert!((relative - 0.1).abs() < 1e-14, "{relative}");
    }

    #[test]
    fn real_saddle_system_is_solved() {
        // [[2, 1], [1, 0]] x = [3, 1] -> x = [1, 1]
        let solved = solve_real(
            2,
            &[(0, 0, 1.0), (0, 0, 1.0), (0, 1, 1.0), (1, 0, 1.0)],
            &[3.0, 1.0],
        )
        .unwrap();
        assert!(solved.relative_residual < 1e-14);
        assert!((solved.solution[0] - 1.0).abs() < 1e-14);
        assert!((solved.solution[1] - 1.0).abs() < 1e-14);
    }

    #[test]
    fn singular_system_is_rejected() {
        let result = solve_real(2, &[(0, 0, 1.0), (1, 0, 1.0)], &[1.0, 1.0]);
        assert!(result.is_err() || result.unwrap().relative_residual > 1e-3);
    }
}

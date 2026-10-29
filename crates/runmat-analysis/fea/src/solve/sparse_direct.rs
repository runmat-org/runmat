//! Narrow sparse direct factor/solve boundary for indefinite and complex operators.
//!
//! Electromagnetic curl-curl systems (and their terminal/gauge augmentations) are
//! indefinite or complex-symmetric with a near-null gradient space, where unpreconditioned
//! Krylov methods stall. This module is the single entry point those kernels use for an
//! exact sparse LU with partial pivoting. It owns no physics; callers assemble triplets in
//! SI units and receive the solution plus the measured relative residual of the ORIGINAL
//! system. The factorization is the shared sparse layer's direct LU (`runmat-sparse-solvers`),
//! run on the ambient admitted CPU plan with FEA cancellation.
use runmat_sparse_solvers::c64;
use runmat_sparse_solvers::direct::{DirectOptions, DirectSolver, FactorKind, FactorScalar};
use runmat_sparse_solvers::TripletBuilder;

/// Factor and solve one square system with the shared direct LU.
fn shared_lu<T: FactorScalar + Send + Sync>(
    dimension: usize,
    triplets: TripletBuilder<T>,
    rhs: &[T],
) -> Result<Vec<T>, SparseDirectError> {
    let matrix = triplets.build();
    let plan = super::shared_sparse::ambient_plan();
    super::shared_sparse::with_plan(plan, || {
        let solver = DirectSolver::<T>::new(
            matrix.as_ref().into(),
            DirectOptions::new(FactorKind::Lu),
            super::shared_sparse::control(),
        )
        .map_err(|error| SparseDirectError::Factorization(error.to_string()))?;
        let mut x = vec![T::zero(); dimension];
        solver
            .solve_refined(matrix.as_ref().into(), rhs, &mut x, 1)
            .map_err(|error| SparseDirectError::Factorization(error.to_string()))?;
        Ok(x)
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
    /// ||A x - b|| / ||b|| on the original assembled entries (or ||A x - b|| when b = 0).
    pub relative_residual: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct RealDirectSolution {
    pub solution: Vec<f64>,
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
        });
    }
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
    if rhs.iter().any(|(a, b)| !a.is_finite() || !b.is_finite()) {
        return Err(SparseDirectError::Invalid("non-finite rhs".into()));
    }
    let values: Vec<c64> = rhs.iter().map(|(a, b)| c64::new(*a, *b)).collect();
    let values = shared_lu(dimension, triplets, &values)?;
    if values
        .iter()
        .any(|v| !v.re.is_finite() || !v.im.is_finite())
    {
        return Err(SparseDirectError::Breakdown);
    }
    let mut residual = rhs
        .iter()
        .map(|(a, b)| c64::new(-a, -b))
        .collect::<Vec<_>>();
    for entry in entries {
        let x = values[entry.column];
        let a = c64::new(entry.real, entry.imaginary);
        residual[entry.row] += a * x;
    }
    let norm = |v: &[c64]| v.iter().fold(0.0_f64, |n, z| n.hypot(z.re).hypot(z.im));
    let rhs_norm = rhs.iter().fold(0.0_f64, |n, (a, b)| n.hypot(*a).hypot(*b));
    let defect = norm(&residual);
    Ok(ComplexDirectSolution {
        solution: values.into_iter().map(|z| (z.re, z.im)).collect(),
        relative_residual: if rhs_norm > 0.0 {
            defect / rhs_norm
        } else {
            defect
        },
    })
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
    let values = shared_lu(dimension, triplets, rhs)?;
    if values.iter().any(|v| !v.is_finite()) {
        return Err(SparseDirectError::Breakdown);
    }
    let mut residual = rhs.iter().map(|v| -v).collect::<Vec<_>>();
    for &(row, column, value) in entries {
        residual[row] += value * values[column];
    }
    let norm = |v: &[f64]| v.iter().fold(0.0_f64, |n, z| n.hypot(*z));
    let rhs_norm = norm(rhs);
    let defect = norm(&residual);
    Ok(RealDirectSolution {
        solution: values,
        relative_residual: if rhs_norm > 0.0 {
            defect / rhs_norm
        } else {
            defect
        },
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
        // x1 = -i / (-1+2i - i*i) = -i/(2i) = -0.5 ; x0 = 1 - i*x1 = 1 + 0.5i
        let (a, b) = solved.solution[0];
        assert!((a - 1.0).abs() < 1e-14 && (b - 0.5).abs() < 1e-14);
        let (c, d) = solved.solution[1];
        assert!((c + 0.5).abs() < 1e-14 && d.abs() < 1e-14);
    }

    #[test]
    fn real_saddle_system_is_solved() {
        // [[2, 1], [1, 0]] x = [3, 1] -> x = [1, 1]
        let solved = solve_real(2, &[(0, 0, 2.0), (0, 1, 1.0), (1, 0, 1.0)], &[3.0, 1.0]).unwrap();
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

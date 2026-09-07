use nalgebra::{linalg::SVD, DMatrix};

use crate::BuiltinResult;

use super::super::{errors, SolveOrientation};

pub(super) fn solve<T>(
    orientation: SolveOrientation,
    lhs: &DMatrix<T>,
    rhs: &DMatrix<T>,
) -> BuiltinResult<DMatrix<T>>
where
    T: nalgebra::ComplexField<RealField = f64> + Copy,
{
    match orientation {
        SolveOrientation::Left => solve_left(orientation, lhs, rhs),
        SolveOrientation::Right => {
            let solved = solve_left(orientation, &rhs.transpose(), &lhs.transpose())?;
            Ok(solved.transpose())
        }
    }
}

fn solve_left<T>(
    orientation: SolveOrientation,
    coefficients: &DMatrix<T>,
    values: &DMatrix<T>,
) -> BuiltinResult<DMatrix<T>>
where
    T: nalgebra::ComplexField<RealField = f64> + Copy,
{
    let svd = SVD::new(coefficients.clone(), true, true);
    let tolerance = tolerance(
        svd.singular_values.as_slice(),
        coefficients.nrows(),
        coefficients.ncols(),
    );
    svd.solve(values, tolerance).map_err(|error| {
        errors::invalid_input(orientation, format!("{}: {error}", orientation.name()))
    })
}

fn tolerance(singular_values: &[f64], rows: usize, cols: usize) -> f64 {
    let largest = singular_values
        .iter()
        .copied()
        .fold(0.0_f64, |current, value| current.max(value.abs()));
    f64::EPSILON * rows.max(cols) as f64 * largest.max(1.0)
}

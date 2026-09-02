use runmat_meshing_core::{
    quality::predicate::{ExactDyadic, PredicateSign},
    MeshingCancellationSignal, TETRAHEDRON_MIDSIDE_EDGE_CORNERS,
};

use crate::cdt::solver_topology::{
    error, DelaunaySolverTopologyError, DelaunaySolverTopologyErrorKind,
};

use super::{invalid, resource, BARYCENTRIC_DERIVATIVES, REFERENCE_TETRAHEDRON};

pub(super) fn exact_bernstein_coefficients_are_positive(
    reference_jacobians: &[[[ExactDyadic; 3]; 3]; 4],
    cell: [[f64; 4]; 4],
    orientation: f64,
    maximum_search_work: u64,
    cancellation: &dyn MeshingCancellationSignal,
    work: &mut u64,
) -> Result<bool, DelaunaySolverTopologyError> {
    let mut jacobians = Vec::with_capacity(4);
    for barycentric in cell {
        jacobians.push(exact_jacobian_from_reference_basis(
            reference_jacobians,
            barycentric,
            maximum_search_work,
            cancellation,
            work,
        )?);
    }
    let jacobians: [[[ExactDyadic; 3]; 3]; 4] = jacobians
        .try_into()
        .map_err(|_| invalid("exact Tet10 cell requires four vertex Jacobians"))?;
    for first in 0..4 {
        for second in first..4 {
            for third in second..4 {
                consume_exact_work(maximum_search_work, cancellation, work)?;
                let sign = exact_mixed_determinant_sign(
                    [&jacobians[first], &jacobians[second], &jacobians[third]],
                    orientation,
                    maximum_search_work,
                    cancellation,
                    work,
                )?;
                if sign != PredicateSign::Positive {
                    return Ok(false);
                }
            }
        }
    }
    Ok(true)
}

pub(super) fn exact_jacobian_from_reference_basis(
    reference_jacobians: &[[[ExactDyadic; 3]; 3]; 4],
    barycentric: [f64; 4],
    maximum_search_work: u64,
    cancellation: &dyn MeshingCancellationSignal,
    work: &mut u64,
) -> Result<[[ExactDyadic; 3]; 3], DelaunaySolverTopologyError> {
    let mut result = std::array::from_fn(|_| std::array::from_fn(|_| ExactDyadic::zero()));
    for (row, result_row) in result.iter_mut().enumerate() {
        for (column, result_entry) in result_row.iter_mut().enumerate() {
            for vertex in 0..4 {
                consume_exact_work(maximum_search_work, cancellation, work)?;
                *result_entry = result_entry.clone().add_exact(
                    reference_jacobians[vertex][row][column]
                        .clone()
                        .multiply_exact(ExactDyadic::from_f64(barycentric[vertex])),
                );
            }
        }
    }
    Ok(result)
}

fn exact_mixed_determinant_sign(
    matrices: [&[[ExactDyadic; 3]; 3]; 3],
    orientation: f64,
    maximum_search_work: u64,
    cancellation: &dyn MeshingCancellationSignal,
    work: &mut u64,
) -> Result<PredicateSign, DelaunaySolverTopologyError> {
    const PERMUTATIONS: [[usize; 3]; 6] = [
        [0, 1, 2],
        [0, 2, 1],
        [1, 0, 2],
        [1, 2, 0],
        [2, 0, 1],
        [2, 1, 0],
    ];
    let mut sum = ExactDyadic::zero();
    for assignment in PERMUTATIONS {
        consume_exact_work(maximum_search_work, cancellation, work)?;
        sum = sum.add_exact(exact_determinant_columns([
            exact_column(matrices[assignment[0]], 0),
            exact_column(matrices[assignment[1]], 1),
            exact_column(matrices[assignment[2]], 2),
        ]));
    }
    Ok(match (sum.sign(), orientation > 0.0) {
        (PredicateSign::Positive, true) | (PredicateSign::Negative, false) => {
            PredicateSign::Positive
        }
        (PredicateSign::Negative, true) | (PredicateSign::Positive, false) => {
            PredicateSign::Negative
        }
        (PredicateSign::Zero, _) => PredicateSign::Zero,
    })
}

fn consume_exact_work(
    maximum_search_work: u64,
    cancellation: &dyn MeshingCancellationSignal,
    work: &mut u64,
) -> Result<(), DelaunaySolverTopologyError> {
    *work = work
        .checked_add(1)
        .ok_or_else(|| resource("Tet10 exact Jacobian work counter overflowed"))?;
    if *work > maximum_search_work {
        return Err(resource(format!(
            "Tet10 Jacobian certification exceeds its hard search-work limit of {maximum_search_work}"
        )));
    }
    if cancellation.is_cancelled() {
        return Err(error::failure(
            DelaunaySolverTopologyErrorKind::Cancelled,
            "cancelled",
        ));
    }
    Ok(())
}

fn exact_column(matrix: &[[ExactDyadic; 3]; 3], column: usize) -> [ExactDyadic; 3] {
    std::array::from_fn(|row| matrix[row][column].clone())
}

fn exact_determinant_columns(columns: [[ExactDyadic; 3]; 3]) -> ExactDyadic {
    columns[0][0]
        .clone()
        .multiply_exact(
            columns[1][1]
                .clone()
                .multiply_exact(columns[2][2].clone())
                .subtract_exact(columns[2][1].clone().multiply_exact(columns[1][2].clone())),
        )
        .subtract_exact(
            columns[1][0].clone().multiply_exact(
                columns[0][1]
                    .clone()
                    .multiply_exact(columns[2][2].clone())
                    .subtract_exact(columns[2][1].clone().multiply_exact(columns[0][2].clone())),
            ),
        )
        .add_exact(
            columns[2][0].clone().multiply_exact(
                columns[0][1]
                    .clone()
                    .multiply_exact(columns[1][2].clone())
                    .subtract_exact(columns[1][1].clone().multiply_exact(columns[0][2].clone())),
            ),
        )
}

pub(super) fn build_exact_reference_jacobians(
    coordinates: [[f64; 3]; 10],
    maximum_search_work: u64,
    cancellation: &dyn MeshingCancellationSignal,
    work: &mut u64,
) -> Result<[[[ExactDyadic; 3]; 3]; 4], DelaunaySolverTopologyError> {
    let mut result = Vec::with_capacity(4);
    for barycentric in REFERENCE_TETRAHEDRON {
        result.push(exact_jacobian_at(
            coordinates,
            barycentric,
            maximum_search_work,
            cancellation,
            work,
        )?);
    }
    result
        .try_into()
        .map_err(|_| invalid("exact Tet10 reference basis requires four Jacobians"))
}

pub(super) fn exact_jacobian_at(
    coordinates: [[f64; 3]; 10],
    barycentric: [f64; 4],
    maximum_search_work: u64,
    cancellation: &dyn MeshingCancellationSignal,
    work: &mut u64,
) -> Result<[[ExactDyadic; 3]; 3], DelaunaySolverTopologyError> {
    let barycentric = barycentric.map(ExactDyadic::from_f64);
    let mut result = std::array::from_fn(|_| std::array::from_fn(|_| ExactDyadic::zero()));
    for node in 0..4 {
        let factor = barycentric[node]
            .clone()
            .multiply_integer(4)
            .subtract_exact(ExactDyadic::from_f64(1.0));
        for reference_axis in 0..3 {
            consume_exact_work(maximum_search_work, cancellation, work)?;
            let derivative = factor.clone().multiply_exact(ExactDyadic::from_f64(
                BARYCENTRIC_DERIVATIVES[node][reference_axis],
            ));
            for (physical_axis, result_row) in result.iter_mut().enumerate() {
                result_row[reference_axis] = result_row[reference_axis].clone().add_exact(
                    derivative
                        .clone()
                        .multiply_exact(ExactDyadic::from_f64(coordinates[node][physical_axis])),
                );
            }
        }
    }
    for (local_edge, [left, right]) in TETRAHEDRON_MIDSIDE_EDGE_CORNERS.into_iter().enumerate() {
        for reference_axis in 0..3 {
            consume_exact_work(maximum_search_work, cancellation, work)?;
            let derivative = barycentric[right]
                .clone()
                .multiply_exact(ExactDyadic::from_f64(
                    BARYCENTRIC_DERIVATIVES[left][reference_axis],
                ))
                .add_exact(
                    barycentric[left]
                        .clone()
                        .multiply_exact(ExactDyadic::from_f64(
                            BARYCENTRIC_DERIVATIVES[right][reference_axis],
                        )),
                )
                .multiply_integer(4);
            for (physical_axis, result_row) in result.iter_mut().enumerate() {
                result_row[reference_axis] = result_row[reference_axis].clone().add_exact(
                    derivative.clone().multiply_exact(ExactDyadic::from_f64(
                        coordinates[4 + local_edge][physical_axis],
                    )),
                );
            }
        }
    }
    Ok(result)
}

use super::scaled_triangle::Stage;
use super::*;
use faer::linalg::triangular_solve as tri;

pub(super) fn run<T: DenseScalar + TriangularScalar>(
    data: &NumericFactorData,
    selected: SelectedSolve,
    mut rhs: MatMut<'_, T::Element>,
    coordinates: &mut [T::Element],
    real_lanes: &mut RealLanes,
    mut stack: &mut MemStack,
) -> Result<(), DenseFactorError> {
    let SelectedSolve {
        adjoint,
        columns,
        scale,
        ..
    } = selected;
    let (m, n) = data.dimensions();
    match data.parts() {
        NumericFactorCandidate::DenseQr(first) | NumericFactorCandidate::DenseCod { first, .. } => {
            let rank = first.rank;
            let completion =
                if let NumericFactorCandidate::DenseCod { completion, .. } = data.parts() {
                    completion.as_ref()
                } else {
                    None
                };
            let mut coordinates = MatMut::from_column_major_slice_mut(coordinates, rank, columns);
            if adjoint {
                let p =
                    PermRef::new_checked(&first.permutation.forward, &first.permutation.inverse, n);
                faer::perm::permute_rows_in_place(
                    rhs.rb_mut().subrows_mut(0, n),
                    p,
                    stack.rb_mut(),
                );
                if let Some(second) = completion {
                    real_lanes.apply::<T>(
                        second,
                        true,
                        rhs.rb_mut().subrows_mut(0, n),
                        stack.rb_mut(),
                    )?;
                }
                for c in 0..columns {
                    for j in 0..rank {
                        coordinates[(j, c)] = rhs[(j, c)];
                    }
                }
                if rank > 0 {
                    if let Some(second) = completion {
                        if scale == (1.0, 0.0) {
                            tri::solve_upper_triangular_in_place_with_conj(
                                packed::<T>(&second.packed).submatrix(0, 0, rank, rank),
                                Conj::No,
                                coordinates.rb_mut(),
                                Par::Seq,
                            );
                        } else {
                            scaled_triangle::solve::<T>(
                                Stage {
                                    packed: &second.packed,
                                    order: rank,
                                    upper: true,
                                    adjoint: false,
                                    unit_diagonal: false,
                                    scale,
                                    singular: false,
                                },
                                coordinates.rb_mut(),
                            )?;
                        }
                    } else if scale == (1.0, 0.0) {
                        tri::solve_lower_triangular_in_place_with_conj(
                            packed::<T>(&first.packed)
                                .submatrix(0, 0, rank, rank)
                                .transpose(),
                            Conj::Yes,
                            coordinates.rb_mut(),
                            Par::Seq,
                        );
                    } else {
                        scaled_triangle::solve::<T>(
                            Stage {
                                packed: &first.packed,
                                order: rank,
                                upper: true,
                                adjoint: true,
                                unit_diagonal: false,
                                scale,
                                singular: false,
                            },
                            coordinates.rb_mut(),
                        )?;
                    }
                }
                rhs.rb_mut().fill(T::Element::default());
                for c in 0..columns {
                    for j in 0..rank {
                        let row = completion.map_or(j, |q| q.permutation.forward[j]);
                        rhs[(row, c)] = coordinates[(j, c)];
                    }
                }
                real_lanes.apply::<T>(
                    first,
                    false,
                    rhs.rb_mut().subrows_mut(0, m),
                    stack.rb_mut(),
                )?;
            } else {
                real_lanes.apply::<T>(
                    first,
                    true,
                    rhs.rb_mut().subrows_mut(0, m),
                    stack.rb_mut(),
                )?;
                for c in 0..columns {
                    for j in 0..rank {
                        let row = completion.map_or(j, |q| q.permutation.forward[j]);
                        coordinates[(j, c)] = rhs[(row, c)];
                    }
                }
                if rank > 0 {
                    if let Some(second) = completion {
                        if scale == (1.0, 0.0) {
                            tri::solve_lower_triangular_in_place_with_conj(
                                packed::<T>(&second.packed)
                                    .submatrix(0, 0, rank, rank)
                                    .transpose(),
                                Conj::Yes,
                                coordinates.rb_mut(),
                                Par::Seq,
                            );
                        } else {
                            scaled_triangle::solve::<T>(
                                Stage {
                                    packed: &second.packed,
                                    order: rank,
                                    upper: true,
                                    adjoint: true,
                                    unit_diagonal: false,
                                    scale,
                                    singular: false,
                                },
                                coordinates.rb_mut(),
                            )?;
                        }
                    } else if scale == (1.0, 0.0) {
                        tri::solve_upper_triangular_in_place_with_conj(
                            packed::<T>(&first.packed).submatrix(0, 0, rank, rank),
                            Conj::No,
                            coordinates.rb_mut(),
                            Par::Seq,
                        );
                    } else {
                        scaled_triangle::solve::<T>(
                            Stage {
                                packed: &first.packed,
                                order: rank,
                                upper: true,
                                adjoint: false,
                                unit_diagonal: false,
                                scale,
                                singular: false,
                            },
                            coordinates.rb_mut(),
                        )?;
                    }
                }
                rhs.rb_mut().fill(T::Element::default());
                for c in 0..columns {
                    for j in 0..rank {
                        rhs[(j, c)] = coordinates[(j, c)];
                    }
                }
                if let Some(second) = completion {
                    real_lanes.apply::<T>(
                        second,
                        false,
                        rhs.rb_mut().subrows_mut(0, n),
                        stack.rb_mut(),
                    )?;
                }
                let p =
                    PermRef::new_checked(&first.permutation.inverse, &first.permutation.forward, n);
                faer::perm::permute_rows_in_place(
                    rhs.rb_mut().subrows_mut(0, n),
                    p,
                    stack.rb_mut(),
                );
            }
        }
        _ => unreachable!("validated dense factor"),
    }
    Ok(())
}

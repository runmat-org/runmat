use super::scaled_triangle::Stage;
use super::*;

pub(super) fn run<T: DenseScalar + TriangularScalar>(
    data: &NumericFactorData,
    selected: SelectedSolve,
    mut rhs: MatMut<'_, T::Element>,
    mut stack: &mut MemStack,
) -> Result<(), DenseFactorError> {
    let SelectedSolve {
        adjoint,
        scale,
        singular,
        ..
    } = selected;
    let (_, n) = data.dimensions();
    match data.parts() {
        NumericFactorCandidate::DenseLu {
            packed: lu_factor,
            rows,
        } => {
            let p = PermRef::new_checked(&rows.forward, &rows.inverse, n);
            if scale != (1.0, 0.0) && adjoint {
                scaled_triangle::solve::<T>(
                    Stage {
                        packed: lu_factor,
                        order: n,
                        upper: true,
                        adjoint: true,
                        unit_diagonal: false,
                        scale,
                        singular,
                    },
                    rhs.rb_mut(),
                )?;
                scaled_triangle::solve::<T>(
                    Stage {
                        packed: lu_factor,
                        order: n,
                        upper: false,
                        adjoint: true,
                        unit_diagonal: true,
                        scale: (1.0, 0.0),
                        singular,
                    },
                    rhs.rb_mut(),
                )?;
                let inverse = PermRef::new_checked(&rows.inverse, &rows.forward, n);
                faer::perm::permute_rows_in_place(rhs.rb_mut(), inverse, stack.rb_mut());
            } else if scale != (1.0, 0.0) {
                faer::perm::permute_rows_in_place(rhs.rb_mut(), p, stack.rb_mut());
                scaled_triangle::solve::<T>(
                    Stage {
                        packed: lu_factor,
                        order: n,
                        upper: false,
                        adjoint: false,
                        unit_diagonal: true,
                        scale: (1.0, 0.0),
                        singular,
                    },
                    rhs.rb_mut(),
                )?;
                scaled_triangle::solve::<T>(
                    Stage {
                        packed: lu_factor,
                        order: n,
                        upper: true,
                        adjoint: false,
                        unit_diagonal: false,
                        scale,
                        singular,
                    },
                    rhs.rb_mut(),
                )?;
            } else if adjoint {
                let lu = packed::<T>(lu_factor);
                lu_solve::solve_transpose_in_place_with_conj(
                    lu,
                    lu,
                    p,
                    Conj::Yes,
                    rhs,
                    Par::Seq,
                    stack.rb_mut(),
                );
            } else {
                let lu = packed::<T>(lu_factor);
                lu_solve::solve_in_place_with_conj(
                    lu,
                    lu,
                    p,
                    Conj::No,
                    rhs,
                    Par::Seq,
                    stack.rb_mut(),
                );
            }
        }
        _ => unreachable!("validated dense factor"),
    }
    Ok(())
}

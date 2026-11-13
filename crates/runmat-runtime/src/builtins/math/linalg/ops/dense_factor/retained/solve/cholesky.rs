use super::scaled_triangle::Stage;
use super::*;

pub(super) fn run<T: DenseScalar + TriangularScalar>(
    data: &NumericFactorData,
    selected: SelectedSolve,
    mut rhs: MatMut<'_, T::Element>,
    mut stack: &mut MemStack,
) -> Result<(), DenseFactorError> {
    let SelectedSolve {
        scale,
        cholesky_stages,
        ..
    } = selected;
    let (_, n) = data.dimensions();
    match data.parts() {
        NumericFactorCandidate::DenseCholesky { lower } => {
            if scale == (1.0, 0.0) {
                chol::solve_in_place_with_conj(
                    packed::<T>(lower),
                    Conj::No,
                    rhs,
                    Par::Seq,
                    stack.rb_mut(),
                )
            } else {
                let (forward_scale, reverse_scale) =
                    cholesky_stages.expect("pure Cholesky stage preflight");
                scaled_triangle::solve::<T>(
                    Stage {
                        packed: lower,
                        order: n,
                        upper: false,
                        adjoint: false,
                        unit_diagonal: false,
                        scale: forward_scale,
                        singular: false,
                    },
                    rhs.rb_mut(),
                )?;
                scaled_triangle::solve::<T>(
                    Stage {
                        packed: lower,
                        order: n,
                        upper: false,
                        adjoint: true,
                        unit_diagonal: false,
                        scale: reverse_scale,
                        singular: false,
                    },
                    rhs.rb_mut(),
                )?;
            }
        }
        _ => unreachable!("validated dense factor"),
    }
    Ok(())
}

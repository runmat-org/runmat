//! One accepted Jinfo generation and one selected original multiply callable.
use super::*;
use crate::call::{descriptor::RepeatableCallable, BuiltinArguments, RootedFunctionArguments};
mod invocation;
mod materialize;
pub(crate) struct Opaque {
    info: RootedValue,
    // Selected once for the invocation, shared by pending/accepted generations.
    function: runmat_execution::resource::ExactShared<RepeatableCallable>,
    rows: usize,
    columns: usize,
}
impl Opaque {
    pub(in crate::builtins::math::optim) fn new(
        info: RootedValue,
        function: runmat_execution::resource::ExactShared<RepeatableCallable>,
        rows: usize,
        columns: usize,
    ) -> Self {
        Self {
            info,
            function,
            rows,
            columns,
        }
    }
    pub(super) fn dimensions(&self) -> (usize, usize) {
        (self.rows, self.columns)
    }
    pub(super) async fn apply(
        &self,
        orientation: ProductOrientation,
        input: &[f64],
        output: &mut [f64],
        c: &mut ScalarCooperation,
    ) -> BuiltinResult<Option<()>> {
        let input = match orientation {
            ProductOrientation::Normal => invocation::Input::Forward(input),
            ProductOrientation::Adjoint => invocation::Input::Transpose(input),
        };
        let value = self.invoke(input, c).await?;
        let operator = product_view(value.value(), c).await?;
        match operator {
            LinearOperator::RealF32(view) => column_into(view, 0, output, c).await?,
            LinearOperator::RealF64(view) => column_into(view, 0, output, c).await?,
            _ => return Err(invalid("Jacobian multiply output changed numeric class")),
        }
        Ok(Some(()))
    }
    pub(super) async fn normal(
        &self,
        input: &[f64],
        output: &mut [f64],
        c: &mut ScalarCooperation,
    ) -> BuiltinResult<Option<()>> {
        // The common Krylov operator requests one direction. The documented
        // normal-product ABI is a pair; its second column is the exact zero
        // direction, not an objective/finite-difference probe.
        let value = self.invoke(invocation::Input::Normal(input), c).await?;
        let operator = product_view(value.value(), c).await?;
        match operator {
            LinearOperator::RealF32(view) => column_into(view, 0, output, c).await?,
            LinearOperator::RealF64(view) => column_into(view, 0, output, c).await?,
            _ => return Err(invalid("normal product changed numeric class")),
        }
        Ok(Some(()))
    }
}
async fn column_into<T: Real>(
    view: crate::builtins::math::linalg::ops::linear_operator::TypedLinearOperator<'_, T>,
    column: usize,
    output: &mut [f64],
    c: &mut ScalarCooperation,
) -> BuiltinResult<()> {
    use crate::builtins::math::linalg::ops::linear_operator::TypedLinearOperator as V;
    if output.len() != view.rows() || column >= view.cols() {
        return Err(invalid("Jacobian product column geometry mismatch"));
    }
    match view {
        V::Scalar(value) => {
            output[0] = value.real();
            c.completed(1).await?;
        }
        V::Dense(view) => {
            let source = &view.as_slice()[column * view.rows()..][..view.rows()];
            for (to, &value) in output.iter_mut().zip(source) {
                *to = value.real();
                c.completed(1).await?;
            }
        }
        V::Sparse(view) => {
            for chunk in output.chunks_mut(1024) {
                chunk.fill(0.);
                c.completed(chunk.len() as u64).await?;
            }
            let (rows, values) = view.column(column);
            for (&row, &value) in rows.iter().zip(values) {
                output[row] = value.real();
                c.completed(1).await?;
            }
        }
    }
    Ok(())
}

async fn product_view<'a>(
    value: &'a Value,
    c: &mut ScalarCooperation,
) -> BuiltinResult<LinearOperator<'a>> {
    let operator = borrow(value, c).await?;
    match operator {
        LinearOperator::RealF32(view) => finite(view, c).await?,
        LinearOperator::RealF64(view) => finite(view, c).await?,
        _ => return Err(invalid("Jacobian product changed numeric class")),
    }
    Ok(operator)
}

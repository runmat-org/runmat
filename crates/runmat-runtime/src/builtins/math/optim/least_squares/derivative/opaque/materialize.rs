//! Only selected direct/LM work or an actual matrix output forms columns.
use super::*;
impl Opaque {
    pub(in crate::builtins::math::optim::least_squares::derivative) async fn append_column(
        &self,
        axis: usize,
        scale: f64,
        offset: usize,
        indices: &mut Vec<usize>,
        values: &mut Vec<f64>,
        c: &mut ScalarCooperation,
    ) -> BuiltinResult<Option<()>> {
        let returned = self.invoke(invocation::Input::Column(axis), c).await?;
        match product_view(returned.value(), c).await? {
            LinearOperator::RealF32(view) => {
                append_typed(view, 0, scale, offset, indices, values, c).await
            }
            LinearOperator::RealF64(view) => {
                append_typed(view, 0, scale, offset, indices, values, c).await
            }
            _ => Err(invalid(
                "materialized Jacobian column changed numeric class",
            )),
        }
    }
    pub(in crate::builtins::math::optim::least_squares::derivative) async fn row_major(
        &self,
        c: &mut ScalarCooperation,
    ) -> BuiltinResult<Vec<f64>> {
        let mut output = self.storage(c).await?;
        for axis in 0..self.columns {
            let returned = self.invoke(invocation::Input::Column(axis), c).await?;
            match product_view(returned.value(), c).await? {
                LinearOperator::RealF32(view) => {
                    row_column(view, axis, self.columns, &mut output, c).await?
                }
                LinearOperator::RealF64(view) => {
                    row_column(view, axis, self.columns, &mut output, c).await?
                }
                _ => return Err(invalid("dense Jacobian work changed numeric class")),
            }
        }
        Ok(output)
    }
    pub(in crate::builtins::math::optim::least_squares::derivative) async fn published(
        &self,
        c: &mut ScalarCooperation,
    ) -> BuiltinResult<PublishedJacobian> {
        let mut output = self.storage(c).await?;
        for axis in 0..self.columns {
            let returned = self.invoke(invocation::Input::Column(axis), c).await?;
            let slice = &mut output[axis * self.rows..][..self.rows];
            match product_view(returned.value(), c).await? {
                LinearOperator::RealF32(view) => column_into(view, 0, slice, c).await?,
                LinearOperator::RealF64(view) => column_into(view, 0, slice, c).await?,
                _ => return Err(invalid("output Jacobian column changed numeric class")),
            }
        }
        let tensor = runmat_value::Tensor::new(output, vec![self.rows, self.columns])
            .map_err(|error| invalid(&error))?;
        crate::context::with_original_scope(c.runtime(), || {
            RootedValue::in_optional_context(Value::Tensor(tensor), c.runtime())
        })
        .map(PublishedJacobian::Supplied)
    }
    async fn storage(&self, c: &mut ScalarCooperation) -> BuiltinResult<Vec<f64>> {
        let count = checked_jacobian_work("lsqnonlin", self.rows, self.columns)?;
        let mut output = runmat_value::exact_capacity_vec(count).map_err(allocation)?;
        while output.len() < count {
            let end = count.min(output.len().saturating_add(1024));
            let work = end - output.len();
            output.resize(end, 0.);
            c.completed(work as u64).await?;
        }
        Ok(output)
    }
}
async fn row_column<T: Real>(
    view: crate::builtins::math::linalg::ops::linear_operator::TypedLinearOperator<'_, T>,
    axis: usize,
    columns: usize,
    output: &mut [f64],
    c: &mut ScalarCooperation,
) -> BuiltinResult<()> {
    use crate::builtins::math::linalg::ops::linear_operator::TypedLinearOperator as V;
    match view {
        V::Scalar(value) => {
            output[axis] = value.real();
            c.completed(1).await?;
        }
        V::Dense(view) => {
            for (row, &value) in view.as_slice().iter().enumerate() {
                output[row * columns + axis] = value.real();
                c.completed(1).await?;
            }
        }
        V::Sparse(view) => {
            let (rows, values) = view.column(0);
            for (&row, &value) in rows.iter().zip(values) {
                output[row * columns + axis] = value.real();
                c.completed(1).await?;
            }
        }
    }
    Ok(())
}

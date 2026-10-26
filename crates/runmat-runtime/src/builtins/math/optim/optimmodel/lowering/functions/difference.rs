//! Repeated adjacent differences reuse one numeric geometry, not a stencil
//! with rounded binomial coefficients or a global-column Jacobian mirror.
use super::*;
pub(super) async fn lower(
    mut input: Array,
    arguments: &[Value],
    shape: SharedShape,
    columns: usize,
    policy: Policy,
    c: &mut ScalarCooperation,
) -> Lowered<Array> {
    let runtime = c.runtime().cloned();
    let plan = crate::context::with_original_scope(runtime.as_ref(), || {
        crate::builtins::math::reduction::diff::difference_geometry(
            &input.shape,
            crate::call::FunctionArgumentView::from_slice(arguments),
        )
    })?;
    if !crate::builtins::common::shape::array_shapes_equal(&plan.output_shape(), &shape) {
        return Err(policy.invalid("difference geometry differs from original expression"));
    }
    for step in 0..plan.stages() {
        let output = SharedShape::new_unmanaged(plan.shape_after(step + 1));
        let count = evaluation::element_count(&output, policy)?;
        let mut left = runmat_value::exact_capacity_vec(count).map_err(|e| policy.storage(e))?;
        let mut right = runmat_value::exact_capacity_vec(count).map_err(|e| policy.storage(e))?;
        for row in 0..count {
            let (a, b) = plan
                .pair(step, row)
                .ok_or_else(|| policy.invalid("difference source exceeds stage"))?;
            left.push(a);
            right.push(b);
            c.completed(1).await?;
        }
        let high = match input
            .clone()
            .gather(&right, output.clone(), columns, policy, c)
            .await?
        {
            Ok(value) => value,
            Err(stop) => return Ok(Err(stop)),
        };
        let low = match input
            .gather(&left, output.clone(), columns, policy, c)
            .await?
        {
            Ok(value) => value,
            Err(stop) => return Ok(Err(stop)),
        };
        input = match high
            .add(
                low,
                runmat_types::OperatorKind::Subtract,
                output,
                columns,
                policy,
                c,
            )
            .await?
        {
            Ok(value) => value,
            Err(stop) => return Ok(Err(stop)),
        };
        if count == 0 {
            break;
        }
    }
    input.shape = shape;
    Ok(Ok(input))
}

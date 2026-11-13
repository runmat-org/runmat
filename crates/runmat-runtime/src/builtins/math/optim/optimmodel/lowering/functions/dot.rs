//! Real inner products retain original flat pairing, including opposite vector
//! orientations; no implicit n-by-n broadcast is substituted for dot.
use super::*;
pub(super) async fn lower(
    inputs: Vec<Array>,
    arguments: &[Value],
    shape: SharedShape,
    columns: usize,
    policy: Policy,
    c: &mut ScalarCooperation,
) -> Lowered<Array> {
    if inputs.len() != 2 {
        return Err(policy.invalid("dot requires two original operands"));
    }
    let mut inputs = inputs.into_iter();
    let left = inputs.next().expect("validated dot arity");
    let mut right = inputs.next().expect("validated dot arity");
    let runtime = c.runtime().cloned();
    let plan = crate::context::RuntimeScopeFactory::new(runtime, || {
        crate::builtins::math::linalg::ops::dot::dot_geometry(
            &left.shape,
            &right.shape,
            crate::call::FunctionArgumentView::from_slice(arguments),
        )
    })
    .construct()
    .await?;
    if !crate::builtins::common::shape::array_shapes_equal(plan.output_shape(), &shape) {
        return Err(policy.invalid("dot geometry differs from original expression"));
    }
    if plan.outputs() == 0 {
        return Ok(Ok(Array::zero(shape)));
    }
    if left.count(policy)? != right.count(policy)? {
        return Err(policy.invalid("dot operand extents differ"));
    }
    // Only compiler metadata is reoriented; flat source values/coefficients
    // remain in their original immutable backing and coordinate order.
    right.shape = left.shape.clone();
    let source_shape = left.shape.clone();
    let input = match left
        .product(right, source_shape, columns, policy, c)
        .await?
    {
        Ok(value) => value,
        Err(stop) => return Ok(Err(stop)),
    };
    let mut values =
        runmat_value::exact_capacity_vec(plan.outputs()).map_err(|e| policy.storage(e))?;
    for row in 0..plan.outputs() {
        let mut sum = 0.;
        for axis in 0..plan.factor() {
            let source = plan
                .source(row, axis)
                .ok_or_else(|| policy.invalid("dot source exceeds geometry"))?;
            if !input.assigned.at(source, policy)? {
                return Ok(Err(Stop::Unassigned));
            }
            sum += input.offset.at(source, policy)?;
            if !sum.is_finite() {
                return Ok(Err(Stop::NonFinite));
            }
            c.completed(1).await?;
        }
        values.push(sum);
        c.completed(1).await?;
    }
    let mut entries = Vec::new();
    for term in input.coefficients.entries() {
        let row = plan
            .output(term.row)
            .ok_or_else(|| policy.invalid("dot coefficient exceeds geometry"))?;
        entries.try_reserve(1).map_err(|e| policy.reserve(e))?;
        entries.push(Term { row, ..term });
        c.completed(1).await?;
    }
    let coefficients =
        match Coefficients::from_entries(entries, plan.outputs(), columns, policy, c).await? {
            Ok(value) => value,
            Err(stop) => return Ok(Err(stop)),
        };
    let mut dependent = std::collections::HashSet::new();
    for source in 0..input.count(policy)? {
        if input.degree.at(source) {
            let row = plan
                .output(source)
                .ok_or_else(|| policy.invalid("dot dependence exceeds geometry"))?;
            dependent.try_reserve(1).map_err(|e| policy.reserve(e))?;
            dependent.insert(row);
        }
        c.completed(1).await?;
    }
    let degree = Degree::collect(
        plan.outputs(),
        |row| Ok(dependent.contains(&row)),
        policy,
        c,
    )
    .await?;
    let offset = Offset::Dense(
        runmat_value::Tensor::from_numeric_storage(
            runmat_value::NumericStorage::F64(values),
            shape.clone(),
        )
        .map_err(|e| policy.invalid(e))?,
    );
    Ok(Ok(Array {
        shape,
        offset,
        coefficients,
        degree,
        assigned: Assigned::All,
    }))
}

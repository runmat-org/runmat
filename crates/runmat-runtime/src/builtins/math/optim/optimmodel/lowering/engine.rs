//! One joint source topology and compact affine proposal pass. Numerical
//! coefficient cancellation never changes the original variable/domain layout.
use super::*;
use array::Array;
use runmat_gc::GcHandle;
use runmat_types::OperatorKind as Op;
use std::collections::HashMap;

#[cfg(test)]
pub(super) async fn compile(
    source: evaluation::Plan,
    layout: layout::Layout,
    selected: Option<selection::Selection>,
    least_squares: Option<objective::LeastSquares>,
    roots: &[GcHandle],
    policy: Policy,
    cooperation: &mut ScalarCooperation,
) -> Lowered<Program> {
    let compiled = match arrays(
        &source,
        &layout,
        selected.as_ref(),
        roots,
        policy,
        cooperation,
    )
    .await?
    {
        Ok(value) => value,
        Err(stop) => return Ok(Err(stop)),
    };
    finish(
        source,
        layout,
        selected,
        least_squares,
        compiled,
        policy,
        cooperation,
    )
    .await
}
pub(super) async fn finish(
    source: evaluation::Plan,
    layout: layout::Layout,
    selected: Option<selection::Selection>,
    least_squares: Option<objective::LeastSquares>,
    compiled: Arrays,
    policy: Policy,
    cooperation: &mut ScalarCooperation,
) -> Lowered<Program> {
    let Arrays { arrays, facts } = compiled;
    let objective = match least_squares {
        None => objective::ModelObjective::Affine,
        Some(value) => {
            let mut dependent = false;
            for term in &value.terms {
                let fact = facts
                    .get(&term.residual)
                    .ok_or_else(|| policy.invalid("residual has no original dependence fact"))?;
                dependent |= fact.degree != array::Degree::Constant;
                cooperation.completed(1).await?;
            }
            if dependent {
                objective::ModelObjective::LeastSquares(value)
            } else {
                objective::ModelObjective::ConstantSquares(value)
            }
        }
    };
    Ok(Ok(Program {
        source,
        selected,
        objective,
        layout,
        arrays,
        facts,
    }))
}
// Family selection retains the exact original input owners on unsupported
// algebraic forms. Numerical/cancellation/allocation errors are never caught.
pub(super) async fn prepare(
    source: evaluation::Plan,
    layout: layout::Layout,
    selected: Option<selection::Selection>,
    least_squares: Option<objective::LeastSquares>,
    roots: &[GcHandle],
    formulation: &formulation::Formulation,
    policy: Policy,
    cooperation: &mut ScalarCooperation,
) -> Lowered<Prepared> {
    match arrays(
        &source,
        &layout,
        selected.as_ref(),
        roots,
        policy,
        cooperation,
    )
    .await?
    {
        Ok(arrays) => Ok(finish(
            source,
            layout,
            selected,
            least_squares,
            arrays,
            policy,
            cooperation,
        )
        .await?
        .map(Prepared::Algebraic)),
        Err(stop) => {
            nonlinear_or_stop(
                source,
                layout,
                formulation,
                least_squares,
                stop,
                policy,
                cooperation,
            )
            .await
        }
    }
}
pub(super) async fn nonlinear_or_stop(
    source: evaluation::Plan,
    layout: layout::Layout,
    formulation: &formulation::Formulation,
    squares: Option<objective::LeastSquares>,
    stop: Stop,
    policy: Policy,
    cooperation: &mut ScalarCooperation,
) -> Lowered<Prepared> {
    match stop {
        Stop::NonAffine | Stop::UnsupportedFunction => Ok(Ok(Prepared::Nonlinear(
            nonlinear::Model::prepare(source, layout, formulation, squares, policy, cooperation)
                .await?,
        ))),
        Stop::Unassigned | Stop::NonFinite => Ok(Err(stop)),
    }
}
pub(super) struct Arrays {
    pub arrays: HashMap<GcHandle, Array>,
    pub facts: HashMap<GcHandle, NodeFacts>,
}
// The retained input owners remain with the caller until an algebraic family
// is established. This is the same proposal loop, not another classification.
pub(super) async fn arrays(
    source: &evaluation::Plan,
    layout: &layout::Layout,
    selected: Option<&selection::Selection>,
    roots: &[GcHandle],
    policy: Policy,
    cooperation: &mut ScalarCooperation,
) -> Lowered<Arrays> {
    let roots = selected
        .as_ref()
        .map_or(roots, |selection| selection.roots.as_slice());
    let mut remaining = if let Some(selected) = &selected {
        selected.uses(source, policy, cooperation).await?
    } else {
        HashMap::new()
    };
    if selected.is_none() {
        remaining
            .try_reserve(source.uses.len())
            .map_err(|e| policy.reserve(e))?;
        for (&handle, &uses) in &source.uses {
            remaining.insert(handle, uses);
            cooperation.completed(1).await?;
        }
    }
    let mut arrays = HashMap::new();
    let mut facts = HashMap::new();
    for &handle in &source.order {
        if selected
            .as_ref()
            .is_some_and(|selection| !selection.nodes.contains(&handle))
        {
            cooperation.completed(1).await?;
            continue;
        }
        let node = source
            .nodes
            .get(&handle)
            .expect("planned original node")
            .node();
        let mut power = None;
        let result = match &node.kind {
            Kind::Unassigned => Ok(Array::unassigned(node.shape.clone())),
            Kind::EmptyInitializer | Kind::Zero => Ok(Array::zero(node.shape.clone())),
            Kind::Labels { source } => Ok(take(*source, &mut arrays, &mut remaining, policy)?),
            Kind::Variable(Value::Object(object)) => {
                Array::variable(object, layout, policy, cooperation).await?
            }
            Kind::Variable(_) => {
                return Err(policy.invalid("original variable node lost its native reference"))
            }
            Kind::Constant(value) => {
                let shape = construct::numeric_shape(value, policy, cooperation).await?;
                if !crate::builtins::common::shape::array_shapes_equal(&shape, &node.shape) {
                    return Err(policy
                        .invalid("constant backing differs from its original expression shape"));
                }
                if !construct::finite_numeric(value, policy, cooperation).await? {
                    return Ok(Err(Stop::NonFinite));
                }
                Array::constant(value, node.shape.clone(), policy)?
            }
            Kind::Unary { operator, operand } => {
                let input = take(*operand, &mut arrays, &mut remaining, policy)?;
                match operator {
                    Op::UnaryPlus => Ok(input),
                    Op::UnaryMinus => {
                        input
                            .scaled(-1., layout.columns, policy, cooperation)
                            .await?
                    }
                    Op::Transpose | Op::ConjugateTranspose => {
                        input
                            .transposed(node.shape.clone(), layout.columns, policy, cooperation)
                            .await?
                    }
                    _ => return Ok(Err(Stop::NonAffine)),
                }
            }
            Kind::Binary {
                operator,
                left,
                right,
            } => {
                let original_right = *right;
                let left = take(*left, &mut arrays, &mut remaining, policy)?;
                let right = take(*right, &mut arrays, &mut remaining, policy)?;
                if matches!(operator, Op::MatrixPower | Op::ElementwisePower) {
                    power = Some(std::rc::Rc::new(
                        match powers::evidence(
                            original_right,
                            &right,
                            source,
                            layout,
                            &facts,
                            policy,
                            cooperation,
                        )
                        .await?
                        {
                            Ok(value) => value,
                            Err(stop) => return Ok(Err(stop)),
                        },
                    ));
                }
                binary(
                    left,
                    right,
                    *operator,
                    power.as_deref(),
                    node.shape.clone(),
                    layout.columns,
                    policy,
                    cooperation,
                )
                .await?
            }
            Kind::Index { source, indices } => {
                take(*source, &mut arrays, &mut remaining, policy)?
                    .gather(
                        indices,
                        node.shape.clone(),
                        layout.columns,
                        policy,
                        cooperation,
                    )
                    .await?
            }
            Kind::Assign {
                source,
                replacement,
                indices,
                selection_shape,
            } => {
                let input = take(*source, &mut arrays, &mut remaining, policy)?;
                let replacement = take(*replacement, &mut arrays, &mut remaining, policy)?;
                input
                    .assign(
                        replacement,
                        indices,
                        selection_shape,
                        layout.columns,
                        policy,
                        cooperation,
                    )
                    .await?
            }
            Kind::Concatenate {
                dimension,
                operands,
            } => {
                let mut inputs = runmat_value::exact_capacity_vec(operands.len())
                    .map_err(|source| policy.storage(source))?;
                for &operand in operands {
                    inputs.push(take(operand, &mut arrays, &mut remaining, policy)?);
                    cooperation.completed(1).await?;
                }
                Array::concatenate(
                    inputs,
                    *dimension,
                    node.shape.clone(),
                    layout.columns,
                    policy,
                    cooperation,
                )
                .await?
            }
            Kind::Builtin {
                function,
                operands,
                arguments,
            } => {
                let Some(operation) = operations::Operation::selected(function.0.as_str()) else {
                    return Ok(Err(Stop::UnsupportedFunction));
                };
                let mut inputs = runmat_value::exact_capacity_vec(operands.len())
                    .map_err(|e| policy.storage(e))?;
                for &operand in operands {
                    inputs.push(take(operand, &mut arrays, &mut remaining, policy)?);
                    cooperation.completed(1).await?;
                }
                functions::lower(
                    operation,
                    inputs,
                    arguments,
                    node.shape.clone(),
                    layout.columns,
                    policy,
                    cooperation,
                )
                .await?
            }
        };
        let mut result = match result {
            Ok(result) => result,
            Err(stop) => return Ok(Err(stop)),
        };
        if !crate::builtins::common::shape::array_shapes_equal(&result.shape, &node.shape) {
            return Err(
                policy.invalid("compiled proposal differs from its original expression geometry")
            );
        }
        result.shape = node.shape.clone();
        arrays.try_reserve(1).map_err(|e| policy.reserve(e))?;
        facts
            .try_reserve(1)
            .map_err(|source| policy.reserve(source))?;
        facts.insert(
            handle,
            NodeFacts {
                degree: result.degree.clone(),
                complete: matches!(result.assigned, array::Assigned::All),
                power,
            },
        );
        arrays.insert(handle, result);
        cooperation.completed(1).await?;
    }
    for &root in roots {
        let array = arrays
            .get(&root)
            .ok_or_else(|| policy.invalid("compiled root is unavailable"))?;
        for row in 0..array.count(policy)? {
            if !array.assigned.at(row, policy)? {
                return Ok(Err(Stop::Unassigned));
            }
            cooperation.completed(1).await?;
        }
    }

    Ok(Ok(Arrays { arrays, facts }))
}

fn take(
    handle: GcHandle,
    arrays: &mut HashMap<GcHandle, Array>,
    remaining: &mut HashMap<GcHandle, usize>,
    policy: Policy,
) -> BuiltinResult<Array> {
    let count = remaining
        .get_mut(&handle)
        .ok_or_else(|| policy.invalid("compiled edge has no original use count"))?;
    *count = count
        .checked_sub(1)
        .ok_or_else(|| policy.invalid("compiled edge use count underflow"))?;
    if *count == 0 {
        arrays
            .remove(&handle)
            .ok_or_else(|| policy.invalid("compiled edge was retired before its last use"))
    } else {
        arrays
            .get(&handle)
            .cloned()
            .ok_or_else(|| policy.invalid("shared compiled edge is unavailable"))
    }
}
async fn binary(
    left: Array,
    right: Array,
    operator: Op,
    power: Option<&powers::Exponents>,
    shape: SharedShape,
    columns: usize,
    policy: Policy,
    cooperation: &mut ScalarCooperation,
) -> Lowered<Array> {
    match operator {
        Op::Add | Op::Subtract => {
            left.add(right, operator, shape, columns, policy, cooperation)
                .await
        }
        Op::MatrixMultiply | Op::ElementwiseMultiply => {
            if let Some(factor) = right.scalar_constant(policy)? {
                left.scaled(factor, columns, policy, cooperation).await
            } else if let Some(factor) = left.scalar_constant(policy)? {
                right.scaled(factor, columns, policy, cooperation).await
            } else if operator == Op::MatrixMultiply
                && left.count(policy)? != 1
                && right.count(policy)? != 1
            {
                left.matrix_product(right, shape, columns, policy, cooperation)
                    .await
            } else {
                left.product(right, shape, columns, policy, cooperation)
                    .await
            }
        }
        Op::Mrdivide | Op::ElementwiseDivide => {
            let Some(divisor) = right.scalar_constant(policy)? else {
                return Ok(Err(Stop::NonAffine));
            };
            left.scaled(1. / divisor, columns, policy, cooperation)
                .await
        }
        Op::Mldivide | Op::ElementwiseLeftDivide => {
            let Some(divisor) = left.scalar_constant(policy)? else {
                return Ok(Err(Stop::NonAffine));
            };
            right
                .scaled(1. / divisor, columns, policy, cooperation)
                .await
        }
        Op::MatrixPower | Op::ElementwisePower => {
            powers::proposal(
                left,
                right,
                operator,
                power.ok_or_else(|| policy.invalid("power lost original exponent evidence"))?,
                shape,
                columns,
                policy,
                cooperation,
            )
            .await
        }
        _ => Ok(Err(Stop::NonAffine)),
    }
}

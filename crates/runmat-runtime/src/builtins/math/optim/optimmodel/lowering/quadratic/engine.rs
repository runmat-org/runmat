//! Source topology is borrowed once. Last-use coefficient rows move out;
//! shared algebra rows retain immutable backing rather than duplicate a DAG.
use super::*;
use runmat_types::OperatorKind as Op;
use std::collections::HashMap;
pub(super) async fn compile(
    source: &evaluation::Plan,
    layout: &layout::Layout,
    selected: selection::Selection,
    squares: Option<objective::LeastSquares>,
    policy: Policy,
    c: &mut ScalarCooperation,
) -> Lowered<Compiled> {
    let atoms = match super::super::norm_square::Plan::prepare(source, &selected, policy, c).await?
    {
        Ok(plan) => plan,
        Err(stop) => return Ok(Err(stop)),
    };
    let mut remaining = selected.uses(source, policy, c).await?;
    atoms
        .redirect_uses(source, &mut remaining, policy, c)
        .await?;
    let mut arrays = HashMap::new();
    let mut facts = HashMap::new();
    for &handle in &source.order {
        if !selected.nodes.contains(&handle) {
            c.completed(1).await?;
            continue;
        }
        if atoms.is_norm(handle) {
            c.completed(1).await?;
            continue;
        }
        let node = source.nodes[&handle].node();
        let mut power = None;
        let result = match &node.kind {
            Kind::Unassigned => Ok(Array::missing(node.shape.clone())),
            Kind::Zero | Kind::EmptyInitializer => Ok(Array::zero(node.shape.clone())),
            Kind::Labels { source } => Ok(take(*source, &mut arrays, &mut remaining, policy)?
                .reshaped(node.shape.clone(), policy)?),
            Kind::Variable(value @ Value::Object(object)) => {
                let Some(
                    runmat_value::optimization_variable::OptimizationVariablePayload::Reference(
                        selection,
                    ),
                ) = object.optimization_variable_payload()
                else {
                    return Err(policy.invalid("quadratic variable lost native reference"));
                };
                let variable = layout.variable(selection.definition(), policy)?;
                Ok(Array::variable(
                    value.clone(),
                    variable.start,
                    node.shape.clone(),
                ))
            }
            Kind::Variable(_) => {
                return Err(policy.invalid("quadratic variable lost native reference"))
            }
            Kind::Constant(value) => {
                let shape = construct::numeric_shape(value, policy, c).await?;
                if !crate::builtins::common::shape::array_shapes_equal(&shape, &node.shape) {
                    return Err(
                        policy.invalid("quadratic constant backing differs from original shape")
                    );
                }
                if !construct::finite_numeric(value, policy, c).await? {
                    return Ok(Err(Stop::NonFinite));
                }
                match super::super::array::Offset::from_value(value, policy)? {
                    Ok(offset) => Ok(Array::constant(node.shape.clone(), offset)),
                    Err(stop) => Err(stop),
                }
            }
            Kind::Unary { operator, operand } => {
                let input = take(*operand, &mut arrays, &mut remaining, policy)?;
                match operator {
                    Op::UnaryPlus => Ok(input),
                    Op::UnaryMinus => Ok(input
                        .scaled(&Weight::constant(-1., policy)?, policy, c)
                        .await?),
                    Op::Transpose | Op::ConjugateTranspose => {
                        Ok(structural::transpose(input, node.shape.clone(), policy, c).await?)
                    }
                    _ => Err(Stop::NonAffine),
                }
            }
            Kind::Binary {
                operator,
                left,
                right,
            } => {
                let atomic = atoms.residual(handle);
                let a = take(atomic.unwrap_or(*left), &mut arrays, &mut remaining, policy)?;
                let b = take(*right, &mut arrays, &mut remaining, policy)?;
                if matches!(operator, Op::MatrixPower | Op::ElementwisePower) {
                    let candidate = match exponent(&b, policy, c).await? {
                        Ok(value) => value,
                        Err(stop) => return Ok(Err(stop)),
                    };
                    power = Some(std::rc::Rc::new(
                        match powers::evidence(
                            *right, &candidate, source, layout, &facts, policy, c,
                        )
                        .await?
                        {
                            Ok(value) => value,
                            Err(stop) => return Ok(Err(stop)),
                        },
                    ));
                }
                if atomic.is_some() {
                    norm_square(a, node.shape.clone(), policy, c).await?
                } else {
                    binary::binary(a, b, *operator, node.shape.clone(), policy, c).await?
                }
            }
            Kind::Index { source, indices } => {
                Ok(take(*source, &mut arrays, &mut remaining, policy)?
                    .gather(indices, node.shape.clone(), policy, c)
                    .await?)
            }
            Kind::Assign {
                source,
                replacement,
                indices,
                selection_shape,
            } => {
                let input = take(*source, &mut arrays, &mut remaining, policy)?;
                let replacement = take(*replacement, &mut arrays, &mut remaining, policy)?;
                Ok(
                    structural::assign(input, replacement, indices, selection_shape, policy, c)
                        .await?,
                )
            }
            Kind::Concatenate {
                dimension,
                operands,
            } => {
                let mut inputs = runmat_value::exact_capacity_vec(operands.len())
                    .map_err(|e| policy.storage(e))?;
                for &operand in operands {
                    inputs.push(take(operand, &mut arrays, &mut remaining, policy)?);
                    c.completed(1).await?;
                }
                Ok(
                    structural::concatenate(inputs, *dimension, node.shape.clone(), policy, c)
                        .await?,
                )
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
                    c.completed(1).await?;
                }
                functions::lower(inputs, operation, arguments, node.shape.clone(), policy, c)
                    .await?
            }
        };
        let result = match result {
            Ok(value) => value,
            Err(stop) => return Ok(Err(stop)),
        };
        if !crate::builtins::common::shape::array_shapes_equal(&result.shape, &node.shape) {
            return Err(policy.invalid("quadratic coefficient result differs from original shape"));
        }
        let mut complete = true;
        let degree = super::super::array::Degree::collect(
            result.count(policy)?,
            |row| {
                let value = result.at(row, policy)?;
                complete &= value.is_some();
                Ok(value.is_some_and(|value| value.degree() != 0))
            },
            policy,
            c,
        )
        .await?;
        facts.try_reserve(1).map_err(|e| policy.reserve(e))?;
        facts.insert(
            handle,
            NodeFacts {
                degree,
                complete,
                power,
            },
        );
        arrays.try_reserve(1).map_err(|e| policy.reserve(e))?;
        arrays.insert(handle, result);
        c.completed(1).await?;
    }
    let (cost, projection) = match squares {
        None => {
            let [root] = selected.roots.as_slice() else {
                return Err(
                    policy.invalid("quadratic objective selection requires one original root")
                );
            };
            let result = take(*root, &mut arrays, &mut remaining, policy)?;
            if result.count(policy)? != 1 {
                return Err(policy.invalid("quadratic solve requires a scalar objective"));
            }
            let Some(cost) = result.at(0, policy)? else {
                return Ok(Err(Stop::Unassigned));
            };
            (cost, Projection::Scalar)
        }
        Some(squares) => {
            let mut cost = Polynomial::zero();
            for term in &squares.terms {
                let residual = take(term.residual, &mut arrays, &mut remaining, policy)?;
                let weight = Weight {
                    exact: term.weight.exact.clone(),
                    proposal: term.weight.proposal,
                };
                for row in 0..residual.count(policy)? {
                    let Some(value) = residual.at(row, policy)? else {
                        return Ok(Err(Stop::Unassigned));
                    };
                    if value.degree() > 1 {
                        return Ok(Err(Stop::NonAffine));
                    }
                    let square = match value.product(&value, policy, c).await? {
                        Ok(value) => value,
                        Err(stop) => return Ok(Err(stop)),
                    };
                    cost = cost
                        .add(square.scaled(&weight, policy, c).await?, false, policy, c)
                        .await?;
                    c.completed(1).await?;
                }
            }
            (cost, Projection::Squares(squares))
        }
    };
    Ok(Ok(Compiled {
        cost,
        atoms,
        facts,
        selection: selected,
        projection,
    }))
}
fn take(
    handle: GcHandle,
    arrays: &mut HashMap<GcHandle, Array>,
    remaining: &mut HashMap<GcHandle, usize>,
    policy: Policy,
) -> BuiltinResult<Array> {
    let count = remaining
        .get_mut(&handle)
        .ok_or_else(|| policy.invalid("quadratic edge has no original use count"))?;
    *count = count
        .checked_sub(1)
        .ok_or_else(|| policy.invalid("quadratic edge use count underflow"))?;
    if *count == 0 {
        arrays
            .remove(&handle)
            .ok_or_else(|| policy.invalid("quadratic edge retired before last use"))
    } else {
        arrays
            .get(&handle)
            .cloned()
            .ok_or_else(|| policy.invalid("quadratic shared edge is unavailable"))
    }
}
async fn exponent(
    input: &Array,
    policy: Policy,
    c: &mut ScalarCooperation,
) -> Lowered<super::super::array::Array> {
    let mut values =
        runmat_value::exact_capacity_vec(input.count(policy)?).map_err(|e| policy.storage(e))?;
    for row in 0..input.count(policy)? {
        let Some(value) = input.at(row, policy)?.and_then(|value| value.constant()) else {
            return Ok(Err(Stop::NonAffine));
        };
        let Some(proposal) = Rational::from_float(value.proposal) else {
            return Ok(Err(Stop::NonFinite));
        };
        if value.exact.compare(&proposal, policy, c).await?.is_ne() {
            return Ok(Err(Stop::NonFinite));
        }
        values.push(value.proposal);
        c.completed(1).await?;
    }
    Ok(Ok(super::super::array::Array {
        shape: input.shape.clone(),
        offset: super::super::array::Offset::Dense(
            runmat_value::Tensor::from_numeric_storage(
                runmat_value::NumericStorage::F64(values),
                input.shape.clone(),
            )
            .map_err(|e| policy.invalid(e))?,
        ),
        coefficients: terms::Coefficients::Zero,
        assigned: super::super::array::Assigned::All,
        degree: super::super::array::Degree::Constant,
    }))
}

// The original pair issues this scalar algebra role; the norm itself is never
// represented by a fabricated rational or a floating square-root proposal.
async fn norm_square(
    input: Array,
    shape: SharedShape,
    policy: Policy,
    c: &mut ScalarCooperation,
) -> Lowered<Array> {
    let mut sum = Polynomial::zero();
    for row in 0..input.count(policy)? {
        let Some(value) = input.at(row, policy)? else {
            return Ok(Ok(Array::missing(shape)));
        };
        let square = match value.product(&value, policy, c).await? {
            Ok(value) => value,
            Err(stop) => return Ok(Err(stop)),
        };
        sum = sum.add(square, false, policy, c).await?;
        c.completed(1).await?;
    }
    let mut values = runmat_value::exact_capacity_vec(1).map_err(|e| policy.storage(e))?;
    values.push(Some(sum));
    Ok(Ok(Array::rows(shape, values, policy)?))
}

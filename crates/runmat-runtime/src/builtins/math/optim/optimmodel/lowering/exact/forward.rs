//! One cooperative original-DAG evaluation at a recovered rational point.
//! Intermediate slots retire at actual last use; shared subgraphs run once.
use super::*;
pub(in crate::builtins::math::optim::optimmodel::lowering) fn evaluate<'a>(
    program: &'a Program,
    point: Coordinates<'a>,
    retain: Retain,
    policy: Policy,
    cooperation: &'a mut ScalarCooperation,
) -> impl std::future::Future<Output = BuiltinResult<Evaluated>> + 'a {
    evaluate_selected(
        Source {
            source: &program.source,
            layout: &program.layout,
            facts: &program.facts,
        },
        program.selected.as_ref().map(|selected| Subset {
            nodes: &selected.nodes,
            roots: &selected.roots,
        }),
        point,
        retain,
        policy,
        cooperation,
    )
}
pub(in crate::builtins::math::optim::optimmodel::lowering) fn evaluate_selected<'a>(
    program: Source<'a>,
    selected: Option<Subset<'a>>,
    point: Coordinates<'a>,
    retain: Retain,
    policy: Policy,
    cooperation: &'a mut ScalarCooperation,
) -> impl std::future::Future<Output = BuiltinResult<Evaluated>> + 'a {
    evaluate_impl(program, selected, point, retain, None, policy, cooperation)
}
pub(in crate::builtins::math::optim::optimmodel::lowering) fn evaluate_atomic<'a>(
    program: Source<'a>,
    selected: Option<Subset<'a>>,
    point: Coordinates<'a>,
    retain: Retain,
    atoms: &'a super::super::norm_square::Plan,
    policy: Policy,
    cooperation: &'a mut ScalarCooperation,
) -> impl std::future::Future<Output = BuiltinResult<Evaluated>> + 'a {
    evaluate_impl(
        program,
        selected,
        point,
        retain,
        Some(atoms),
        policy,
        cooperation,
    )
}
async fn evaluate_impl(
    program: Source<'_>,
    selected: Option<Subset<'_>>,
    point: Coordinates<'_>,
    retain: Retain,
    atoms: Option<&super::super::norm_square::Plan>,
    policy: Policy,
    cooperation: &mut ScalarCooperation,
) -> BuiltinResult<Evaluated> {
    if matches!(&point,Coordinates::Point(values) if values.len()!=program.layout.columns) {
        return Err(policy.invalid("exact point differs from original variable layout"));
    }
    let mut remaining = HashMap::new();
    match selected {
        None => {
            remaining
                .try_reserve(program.source.uses.len())
                .map_err(|e| policy.reserve(e))?;
            for (&root, &uses) in &program.source.uses {
                remaining.insert(root, uses);
                cooperation.completed(1).await?;
            }
        }
        Some(subset) => {
            if subset.roots.iter().any(|root| !subset.nodes.contains(root)) {
                return Err(policy.invalid("exact subset lost original root"));
            }
            remaining
                .try_reserve(subset.nodes.len())
                .map_err(|e| policy.reserve(e))?;
            for &root in subset.nodes {
                let node = program
                    .source
                    .nodes
                    .get(&root)
                    .ok_or_else(|| policy.invalid("exact subset node is unavailable"))?;
                let mut edge = 0;
                while let Some(child) = node.dependency_at(edge) {
                    if !subset.nodes.contains(&child) {
                        return Err(policy.invalid("exact subset omits original dependency"));
                    }
                    let uses = remaining.entry(child).or_insert(0usize);
                    *uses = uses
                        .checked_add(1)
                        .ok_or_else(|| policy.invalid("exact subset use count overflow"))?;
                    edge += 1;
                    cooperation.completed(1).await?;
                }
                cooperation.completed(1).await?;
            }
            for &root in subset.roots {
                let uses = remaining.entry(root).or_insert(0usize);
                *uses = uses
                    .checked_add(1)
                    .ok_or_else(|| policy.invalid("exact subset root use count overflow"))?;
                cooperation.completed(1).await?;
            }
        }
    }
    if let Some(atoms) = atoms {
        atoms
            .redirect_uses(program.source, &mut remaining, policy, cooperation)
            .await?;
    }
    if matches!(retain, Retain::ConstantOperands | Retain::Operands) {
        for (&handle, node) in &program.source.nodes {
            if atoms.is_some_and(|atoms| atoms.is_norm(handle)) {
                cooperation.completed(1).await?;
                continue;
            }
            if selected.is_some_and(|subset| !subset.nodes.contains(&handle)) {
                cooperation.completed(1).await?;
                continue;
            }
            let pair;
            let operands = match &node.node().kind {
                Kind::Binary { left, right, .. } => {
                    pair = [*left, *right];
                    &pair[..]
                }
                Kind::Builtin { operands, .. } => &operands[..],
                _ => &[],
            };
            {
                for &operand in operands {
                    let handle = atoms.map_or(operand, |atoms| atoms.redirected(handle, operand));
                    if matches!(retain, Retain::Operands)
                        || program
                            .facts
                            .get(&handle)
                            .is_some_and(|fact| fact.degree != array::Degree::Affine)
                    {
                        let uses = remaining.get_mut(&handle).ok_or_else(|| {
                            policy.invalid("constant operand has no original edge")
                        })?;
                        *uses = uses
                            .checked_add(1)
                            .ok_or_else(|| policy.invalid("constant-operand use count overflow"))?;
                    }
                }
            }
            cooperation.completed(1).await?;
        }
    }
    let mut values = HashMap::new();
    for &handle in &program.source.order {
        if atoms.is_some_and(|atoms| atoms.is_norm(handle)) {
            cooperation.completed(1).await?;
            continue;
        }
        if selected.is_some_and(|subset| !subset.nodes.contains(&handle)) {
            cooperation.completed(1).await?;
            continue;
        }
        let node = program
            .source
            .nodes
            .get(&handle)
            .expect("retained exact source node")
            .node();
        let result = match &node.kind {
            Kind::Unassigned => Data::uniform(node.shape.clone(), None),
            Kind::EmptyInitializer | Kind::Zero => {
                Data::uniform(node.shape.clone(), Some(Rational::zero()))
            }
            Kind::Labels { source } => take(*source, &mut values, &mut remaining, policy)?
                .reshaped(node.shape.clone(), policy)?,
            Kind::Constant(value) => {
                let offset = match array::Offset::from_value(value, policy)? {
                    Ok(offset) => offset,
                    Err(_) => {
                        return Err(
                            policy.invalid("exact source constant is not finite real double")
                        )
                    }
                };
                Data::constant(node.shape.clone(), offset)
            }
            Kind::Variable(Value::Object(object)) => {
                let Some(
                    runmat_value::optimization_variable::OptimizationVariablePayload::Reference(
                        selection,
                    ),
                ) = object.optimization_variable_payload()
                else {
                    return Err(policy.invalid("exact variable lost its original reference"));
                };
                let variable = program.layout.variable(selection.definition(), policy)?;
                match &point {
                    Coordinates::Origin => {
                        Data::uniform(node.shape.clone(), Some(Rational::zero()))
                    }
                    Coordinates::Point(point) => {
                        let mut slots = data::buffer(selection.element_count(), policy)?;
                        for row in 0..selection.element_count() {
                            let input = selection.definition_index(row).ok_or_else(|| {
                                policy
                                    .invalid("exact variable reference exceeds original definition")
                            })?;
                            slots.push(Some(point[variable.start + input].clone()));
                            cooperation.completed(1).await?;
                        }
                        Data::explicit(node.shape.clone(), slots, policy)?
                    }
                }
            }
            Kind::Variable(_) => {
                return Err(policy.invalid("exact source variable lost its native reference"))
            }
            Kind::Unary { operator, operand } => {
                arithmetic::unary(
                    take(*operand, &mut values, &mut remaining, policy)?,
                    *operator,
                    node.shape.clone(),
                    policy,
                    cooperation,
                )
                .await?
            }
            Kind::Binary {
                operator,
                left,
                right,
            } => {
                if let Some(residual) = atoms.and_then(|atoms| atoms.residual(handle)) {
                    let input = take(residual, &mut values, &mut remaining, policy)?;
                    let exponent = take(*right, &mut values, &mut remaining, policy)?;
                    super::norm_square::evaluate(
                        input,
                        exponent,
                        node.shape.clone(),
                        policy,
                        cooperation,
                    )
                    .await?
                } else if matches!(
                    operator,
                    runmat_types::OperatorKind::MatrixPower
                        | runmat_types::OperatorKind::ElementwisePower
                ) {
                    let evidence = program
                        .facts
                        .get(&handle)
                        .and_then(|fact| fact.power.as_deref())
                        .ok_or_else(|| {
                            policy.invalid("exact power lost selected original exponent evidence")
                        })?;
                    super::power(
                        take(*left, &mut values, &mut remaining, policy)?,
                        take(*right, &mut values, &mut remaining, policy)?,
                        *operator,
                        evidence,
                        node.shape.clone(),
                        policy,
                        cooperation,
                    )
                    .await?
                } else {
                    arithmetic::binary(
                        take(*left, &mut values, &mut remaining, policy)?,
                        take(*right, &mut values, &mut remaining, policy)?,
                        *operator,
                        node.shape.clone(),
                        program.facts.get(left).is_some_and(|fact| fact.complete)
                            && program.facts.get(right).is_some_and(|fact| fact.complete),
                        policy,
                        cooperation,
                    )
                    .await?
                }
            }
            Kind::Index { source, indices } => {
                edits::gather(
                    take(*source, &mut values, &mut remaining, policy)?,
                    indices,
                    node.shape.clone(),
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
                edits::assign(
                    take(*source, &mut values, &mut remaining, policy)?,
                    take(*replacement, &mut values, &mut remaining, policy)?,
                    indices,
                    selection_shape,
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
                    .map_err(|e| policy.storage(e))?;
                for &operand in operands {
                    inputs.push(take(operand, &mut values, &mut remaining, policy)?);
                    cooperation.completed(1).await?;
                }
                edits::concatenate(inputs, *dimension, node.shape.clone(), policy, cooperation)
                    .await?
            }

            Kind::Builtin {
                function,
                operands,
                arguments,
            } => {
                let Some(operation) =
                    super::super::operations::Operation::selected(function.0.as_str())
                else {
                    return Err(policy.invalid("unadmitted exact original function"));
                };
                if operation == super::super::operations::Operation::Dot {
                    if operands.len() != 2 {
                        return Err(policy.invalid("exact dot has invalid original arity"));
                    }
                    functions::dot(
                        take(operands[0], &mut values, &mut remaining, policy)?,
                        take(operands[1], &mut values, &mut remaining, policy)?,
                        arguments,
                        node.shape.clone(),
                        policy,
                        cooperation,
                    )
                    .await?
                } else {
                    if operands.len() != 1 {
                        return Err(
                            policy.invalid("exact affine function has invalid original arity")
                        );
                    }
                    let input = take(operands[0], &mut values, &mut remaining, policy)?;
                    match operation {
                        super::super::operations::Operation::Difference => {
                            functions::difference(
                                input,
                                arguments,
                                node.shape.clone(),
                                policy,
                                cooperation,
                            )
                            .await?
                        }
                        super::super::operations::Operation::Product
                        | super::super::operations::Operation::Cumprod => {
                            functions::products(
                                input,
                                operation,
                                arguments,
                                node.shape.clone(),
                                policy,
                                cooperation,
                            )
                            .await?
                        }
                        super::super::operations::Operation::Sum => {
                            arithmetic::sum(
                                input,
                                arguments,
                                node.shape.clone(),
                                policy,
                                cooperation,
                            )
                            .await?
                        }
                        super::super::operations::Operation::Reshape => {
                            input.reshaped(node.shape.clone(), policy)?
                        }
                        _ => {
                            functions::linear(
                                input,
                                operation,
                                arguments,
                                node.shape.clone(),
                                policy,
                                cooperation,
                            )
                            .await?
                        }
                    }
                }
            }
        };
        values.try_reserve(1).map_err(|e| policy.reserve(e))?;
        values.insert(handle, result);
        cooperation.completed(1).await?;
    }
    Ok(Evaluated {
        values,
        retention: retain,
    })
}
fn take(
    handle: GcHandle,
    values: &mut HashMap<GcHandle, Data>,
    remaining: &mut HashMap<GcHandle, usize>,
    policy: Policy,
) -> BuiltinResult<Data> {
    let uses = remaining
        .get_mut(&handle)
        .ok_or_else(|| policy.invalid("exact edge has no original use count"))?;
    *uses = uses
        .checked_sub(1)
        .ok_or_else(|| policy.invalid("exact edge use count underflow"))?;
    if *uses == 0 {
        values.remove(&handle)
    } else {
        values.get(&handle).cloned()
    }
    .ok_or_else(|| policy.invalid("exact edge was retired before its original last use"))
}

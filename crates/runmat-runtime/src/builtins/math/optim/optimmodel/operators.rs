//! Symbolic operator construction uses canonical operator identities and shape
//! inference; evaluation still invokes the original numerical builtin owners.
use super::*;
use runmat_types::{
    CallRequest, InferenceSeverity, LiteralContext, LiteralValue, NumericClass, NumericDomain,
    NumericFact, OperatorKind, OutputSelection, RequestedOutputCount, ShapeFact, StorageFact,
    ValueFact, ValueKindFact,
};
mod coefficient;
pub(super) fn shape(
    operator: OperatorKind,
    nodes: &[&Node],
    policy: Policy,
) -> BuiltinResult<SharedShape> {
    let name = operator
        .overload_name()
        .ok_or_else(|| policy.invalid("operator is not a model arithmetic operation"))?;
    let entry = runmat_builtins::builtin_catalog_entry_by_name(name.display_name())
        .ok_or_else(|| policy.invalid("canonical numeric operator is unavailable"))?;
    let arguments = nodes
        .iter()
        .map(|node| {
            ValueFact::proven(
                ValueKindFact::Numeric(NumericFact {
                    class: NumericClass::Double,
                    domain: NumericDomain::Real,
                }),
                ShapeFact::from(node.shape.iter().map(|&n| Some(n)).collect::<Vec<_>>()),
                StorageFact::Dense,
            )
        })
        .collect();
    let literals = LiteralContext::new(
        nodes
            .iter()
            .map(|node| match &node.kind {
                Kind::Constant(Value::Num(value)) => LiteralValue::Number(*value),
                _ => LiteralValue::Unknown,
            })
            .collect(),
    );
    infer_shape(entry, arguments, literals, policy)
}
pub(super) fn infer_shape(
    entry: &runmat_builtins::BuiltinCatalogEntry,
    arguments: Vec<ValueFact>,
    literals: LiteralContext,
    policy: Policy,
) -> BuiltinResult<SharedShape> {
    let inference = runmat_builtins::infer_catalog_call(
        entry,
        &CallRequest {
            arguments,
            literals,
            outputs: OutputSelection::new(RequestedOutputCount::Exactly(1)),
        },
    );
    if let Some(error) = inference
        .diagnostics
        .into_iter()
        .find(|diagnostic| diagnostic.severity == InferenceSeverity::Error)
    {
        return Err(policy.invalid(error.message));
    }
    let shape = inference
        .outputs
        .first()
        .and_then(|value| value.shape.known_dims())
        .and_then(|dims| dims.into_iter().collect::<Option<Vec<_>>>())
        .ok_or_else(|| {
            policy.invalid("canonical operator did not establish a concrete model shape")
        })?;
    shape
        .iter()
        .try_fold(1usize, |n, d| n.checked_mul(*d))
        .ok_or_else(|| policy.invalid("operator result shape overflows usize"))?;
    Ok(SharedShape::new_unmanaged(shape))
}
pub(super) async fn unary(
    operator: OperatorKind,
    value: &Value,
    policy: Policy,
    cooperation: &mut ScalarCooperation,
) -> BuiltinResult<RootedValue> {
    if !matches!(
        operator,
        OperatorKind::UnaryPlus
            | OperatorKind::UnaryMinus
            | OperatorKind::Transpose
            | OperatorKind::ConjugateTranspose
    ) {
        return Err(policy.invalid("unsupported unary optimization operator"));
    }
    let input = owner::as_expression(value, policy, cooperation).await?;
    let operand = owner::reference(input.value(), policy)?;
    let retained = owner::OwnedNode::read(operand, policy)?;
    if matches!(retained.node().kind, Kind::EmptyInitializer) {
        return Ok(input);
    }
    let shape = shape(operator, &[retained.node()], policy)?;
    let index_names = construct::unnamed(shape.len(), policy)?;
    owner::publish(
        Node {
            shape,
            index_names,
            kind: Kind::Unary { operator, operand },
        },
        policy,
        cooperation,
    )
    .await
}
pub(super) async fn binary(
    operator: OperatorKind,
    left: &Value,
    right: &Value,
    policy: Policy,
    cooperation: &mut ScalarCooperation,
) -> BuiltinResult<RootedValue> {
    if let Some(relation) =
        runmat_value::optimization_model::OptimizationConstraintRelation::from_operator(operator)
    {
        return constraint(relation, left, right, policy, cooperation).await;
    }
    if !matches!(
        operator,
        OperatorKind::Add
            | OperatorKind::Subtract
            | OperatorKind::MatrixMultiply
            | OperatorKind::ElementwiseMultiply
            | OperatorKind::MatrixPower
            | OperatorKind::ElementwisePower
            | OperatorKind::Mldivide
            | OperatorKind::Mrdivide
            | OperatorKind::ElementwiseDivide
            | OperatorKind::ElementwiseLeftDivide
    ) {
        return Err(policy.invalid("strict/logical comparison is not an optimization constraint"));
    }
    let left_root = owner::as_expression(left, policy, cooperation).await?;
    let right_root = owner::as_expression(right, policy, cooperation).await?;
    let left = owner::reference(left_root.value(), policy)?;
    let right = owner::reference(right_root.value(), policy)?;
    let left_node = owner::OwnedNode::read(left, policy)?;
    let right_node = owner::OwnedNode::read(right, policy)?;
    use OperatorKind as Op;
    // Only the explicit nullary initializer has accumulator semantics.
    // An empty selection or an array of unassigned slots never becomes zero.
    if matches!(operator, Op::Add | Op::Subtract) {
        if matches!(left_node.node().kind, Kind::EmptyInitializer)
            && matches!(right_node.node().kind, Kind::EmptyInitializer)
        {
            return Ok(left_root);
        }

        if matches!(left_node.node().kind, Kind::EmptyInitializer) {
            return if operator == Op::Subtract {
                unary(Op::UnaryMinus, right_root.value(), policy, cooperation).await
            } else {
                Ok(right_root)
            };
        }
        if matches!(right_node.node().kind, Kind::EmptyInitializer) {
            return Ok(left_root);
        }
    }
    use coefficient::Shape;
    let coefficient = match operator {
        Op::MatrixPower | Op::ElementwisePower | Op::Mrdivide => {
            Some((right, right_node.node(), Shape::Scalar))
        }
        Op::Mldivide => Some((left, left_node.node(), Shape::Scalar)),
        Op::ElementwiseDivide => Some((right, right_node.node(), Shape::Array)),
        Op::ElementwiseLeftDivide => Some((left, left_node.node(), Shape::Array)),
        _ => None,
    };
    if let Some((root, node, shape)) = coefficient {
        if !coefficient::is_constant(root, node, shape, policy, cooperation).await? {
            return Err(policy.invalid("this operator requires the documented constant coefficient/exponent; use fcn2optimexpr for unsupported functions"));
        }
    }
    let shape = shape(operator, &[left_node.node(), right_node.node()], policy)?;
    let index_names = construct::unnamed(shape.len(), policy)?;
    owner::publish(
        Node {
            shape,
            index_names,
            kind: Kind::Binary {
                operator,
                left,
                right,
            },
        },
        policy,
        cooperation,
    )
    .await
}
async fn constraint(
    relation: runmat_value::optimization_model::OptimizationConstraintRelation,
    left: &Value,
    right: &Value,
    policy: Policy,
    cooperation: &mut ScalarCooperation,
) -> BuiltinResult<RootedValue> {
    let left_root = owner::as_expression(left, policy, cooperation).await?;
    let right_root = owner::as_expression(right, policy, cooperation).await?;
    let left = owner::reference(left_root.value(), policy)?;
    let right = owner::reference(right_root.value(), policy)?;
    let left_node = owner::OwnedNode::read(left, policy)?;
    let right_node = owner::OwnedNode::read(right, policy)?;
    if !crate::builtins::common::shape::array_shapes_equal(
        &left_node.node().shape,
        &right_node.node().shape,
    ) && left_node.node().shape.iter().product::<usize>() != 1
        && right_node.node().shape.iter().product::<usize>() != 1
    {
        return Err(
            policy.invalid("constraint expressions must have the same shape or one scalar side")
        );
    }
    let shape = shape(
        OperatorKind::Subtract,
        &[left_node.node(), right_node.node()],
        policy,
    )?;
    let class = match relation {
        runmat_value::optimization_model::OptimizationConstraintRelation::Equal => {
            standard::OPTIMIZATION_EQUALITY
        }
        runmat_value::optimization_model::OptimizationConstraintRelation::LessEqual
        | runmat_value::optimization_model::OptimizationConstraintRelation::GreaterEqual => {
            standard::OPTIMIZATION_INEQUALITY
        }
    };
    registration::ensure(class, policy, cooperation.runtime())?;
    let object =
        ObjectInstance::new(class.owned()).with_optimization_model_payload(Payload::Constraint(
            runmat_value::optimization_model::OptimizationConstraintRef {
                index_names: construct::unnamed(shape.len(), policy)?,
                left,
                right,
                relation,
                shape,
            },
        ));
    RootedValue::in_optional_context(Value::Object(object), cooperation.runtime())
}

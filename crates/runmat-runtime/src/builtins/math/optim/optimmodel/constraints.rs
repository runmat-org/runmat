//! Comparison values retain their actual relation and original expression
//! roots. Numerical constraint evaluation uses the canonical subtract/abs
//! owners, with the documented direction and equality residual semantics.
use super::*;
pub(super) mod collection;
use runmat_value::optimization_model::OptimizationConstraintRef;
use runmat_value::optimization_model::OptimizationConstraintRelation as Relation;

pub(super) fn reference(
    value: &Value,
    policy: Policy,
) -> BuiltinResult<&OptimizationConstraintRef> {
    let Value::Object(object) = value else {
        return Err(policy.invalid("expected an optimization constraint"));
    };
    if object.selected_class().is_some() {
        return Err(policy.invalid("expected an original native constraint"));
    }
    let Some(Payload::Constraint(reference)) = object.optimization_model_payload() else {
        return Err(policy.invalid("expected a native comparison constraint"));
    };
    let valid = match reference.relation {
        Relation::Equal => object.class_name().is(standard::OPTIMIZATION_EQUALITY),
        Relation::LessEqual | Relation::GreaterEqual => {
            object.class_name().is(standard::OPTIMIZATION_INEQUALITY)
        }
    };
    if !valid {
        return Err(policy.invalid("constraint class differs from its original relation"));
    }
    Ok(reference)
}

pub(super) async fn evaluate(
    value: &Value,
    point: &Value,
    policy: Policy,
    cooperation: &mut ScalarCooperation,
) -> BuiltinResult<RootedValue> {
    let _inputs = policy.roots([value, point].into_iter(), cooperation.runtime())?;
    let reference = reference(value, policy)?;
    let mut evaluated = evaluation::evaluate_roots(
        &[reference.left, reference.right],
        point,
        policy,
        cooperation,
    )
    .await?;
    let left = evaluated.take(reference.left, policy, cooperation).await?;
    let right = evaluated.take(reference.right, policy, cooperation).await?;
    let (first, second) = if reference.relation == Relation::GreaterEqual {
        (&right, &left)
    } else {
        (&left, &right)
    };
    let residual = evaluation::invoke(
        "minus",
        &[first.value().clone(), second.value().clone()],
        policy,
        cooperation,
    )
    .await?;
    let residual = if reference.relation == Relation::Equal {
        evaluation::invoke(
            "abs",
            std::slice::from_ref(residual.value()),
            policy,
            cooperation,
        )
        .await?
    } else {
        residual
    };
    let shape = construct::numeric_shape(residual.value(), policy, cooperation).await?;
    if !crate::builtins::common::shape::array_shapes_equal(&shape, &reference.shape) {
        return Err(policy.invalid("constraint evaluation changed its original shape"));
    }
    Ok(residual)
}

pub(super) async fn variables(
    value: &Value,
    policy: Policy,
    cooperation: &mut ScalarCooperation,
) -> BuiltinResult<RootedValue> {
    let reference = reference(value, policy)?;
    properties::variables_roots(&[reference.left, reference.right], policy, cooperation).await
}

pub(super) async fn read_property(
    value: &Value,
    property: runmat_builtins::OptimizationExpressionProperty,
    policy: Policy,
    cooperation: &mut ScalarCooperation,
) -> BuiltinResult<RootedValue> {
    match property {
        runmat_builtins::OptimizationExpressionProperty::Variables => {
            variables(value, policy, cooperation).await
        }
        runmat_builtins::OptimizationExpressionProperty::IndexNames => {
            properties::names_value(&reference(value, policy)?.index_names, policy, cooperation)
                .await
        }
    }
}
pub(super) async fn set_names(
    value: &Value,
    rhs: &Value,
    policy: Policy,
    cooperation: &mut ScalarCooperation,
) -> BuiltinResult<RootedValue> {
    let _roots = policy.roots([value, rhs].into_iter(), cooperation.runtime())?;
    let original = reference(value, policy)?;
    let names =
        super::super::common::model_dimensions::property(rhs, &original.shape, cooperation, policy)
            .await?;
    let class = match original.relation {
        Relation::Equal => standard::OPTIMIZATION_EQUALITY,
        Relation::LessEqual | Relation::GreaterEqual => standard::OPTIMIZATION_INEQUALITY,
    };
    registration::ensure(class, policy, cooperation.runtime())?;
    let object = ObjectInstance::new(class.owned()).with_optimization_model_payload(
        Payload::Constraint(OptimizationConstraintRef {
            index_names: names,
            left: original.left,
            right: original.right,
            relation: original.relation,
            shape: original.shape.clone(),
        }),
    );
    RootedValue::in_optional_context(Value::Object(object), cooperation.runtime())
}

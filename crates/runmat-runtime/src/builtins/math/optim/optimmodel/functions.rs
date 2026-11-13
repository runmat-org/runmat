//! Supported expression functions are retained canonical numerical calls.
//! Shape inference consumes real metadata and symbolic operand facts, never
//! fabricated numeric arrays or a user-visible evaluation callback.
use super::*;
use crate::call::FunctionArgumentView;
use runmat_types::{
    BuiltinId, LiteralContext, LiteralValue, NumericClass, NumericDomain, NumericFact, ShapeFact,
    StorageFact, ValueFact, ValueKindFact,
};
#[derive(Clone, Copy)]
pub(super) enum FunctionRole {
    Ordinary,
    TwoNorm,
}

pub(super) async fn builtin(
    function: runmat_builtins::BuiltinCatalogIdentity,
    operands: usize,
    role: FunctionRole,
    args: FunctionArgumentView<'_>,
    policy: Policy,
    cooperation: &mut ScalarCooperation,
) -> BuiltinResult<RootedValue> {
    if args.len() < operands {
        return Err(policy.invalid("not enough expression function operands"));
    }
    let _inputs = policy.roots(args.iter(), cooperation.runtime())?;
    let mut roots = runmat_value::exact_capacity_vec(operands).map_err(|e| policy.storage(e))?;
    let mut handles = runmat_value::exact_capacity_vec(operands).map_err(|e| policy.storage(e))?;
    let mut facts = runmat_value::exact_capacity_vec(args.len()).map_err(|e| policy.storage(e))?;
    let mut literals =
        runmat_value::exact_capacity_vec(args.len()).map_err(|e| policy.storage(e))?;
    for value in args.iter().take(operands) {
        let root = owner::as_expression(value, policy, cooperation).await?;
        let handle = owner::reference(root.value(), policy)?;
        let node = owner::OwnedNode::read(handle, policy)?;
        if matches!(role, FunctionRole::TwoNorm) {
            // Construction preserves the numerical operation. Only the
            // lowering classifier can certify a vector Euclidean norm square;
            // a matrix two-norm remains spectral, never an entry-square sum.
            crate::builtins::math::linalg::solve::norm::classify_tensor(&node.node().shape)?;
        }
        facts.push(ValueFact::proven(
            ValueKindFact::Numeric(NumericFact {
                class: NumericClass::Double,
                domain: NumericDomain::Real,
            }),
            ShapeFact::from(
                node.node()
                    .shape
                    .iter()
                    .map(|&n| Some(n))
                    .collect::<Vec<_>>(),
            ),
            StorageFact::Dense,
        ));
        literals.push(match &node.node().kind {
            Kind::Constant(Value::Num(value)) => LiteralValue::Number(*value),
            _ => LiteralValue::Unknown,
        });
        handles.push(handle);
        roots.push(root);
    }
    let mut arguments =
        runmat_value::exact_capacity_vec(args.len() - operands).map_err(|e| policy.storage(e))?;
    for value in args.iter().skip(operands) {
        literals.push(metadata(value, policy, cooperation).await?);
        facts.push(crate::value_fact::value_fact(value));
        arguments.push(value.clone());
    }
    if matches!(role, FunctionRole::TwoNorm)
        && args.len() > 1
        && (args.len() != 2
            || !literals.get(1).is_some_and(|literal| {
                matches!(literal, LiteralValue::Number(value) if *value == 2.0)
                    || literal.integer_value_i128() == Some(2)
            }))
    {
        return Err(policy.invalid("symbolic norm supports the default or numeric-two norm"));
    }
    let entry = runmat_builtins::builtin_catalog_entry_by_name(function.name)
        .ok_or_else(|| policy.invalid("canonical numerical function is unavailable"))?;
    let shape = operators::infer_shape(entry, facts, LiteralContext::new(literals), policy)?;
    let index_names = construct::unnamed(shape.len(), policy)?;
    owner::publish(
        Node {
            shape,
            index_names,
            kind: Kind::Builtin {
                function: BuiltinId(function.name.into()),
                operands: handles,
                arguments,
            },
        },
        policy,
        cooperation,
    )
    .await
}

pub(super) async fn concatenate(
    dimension: usize,
    args: FunctionArgumentView<'_>,
    policy: Policy,
    cooperation: &mut ScalarCooperation,
) -> BuiltinResult<RootedValue> {
    if args.is_empty() {
        return Err(policy.invalid("concatenation requires expression operands"));
    }
    let _inputs = policy.roots(args.iter(), cooperation.runtime())?;
    let mut roots = runmat_value::exact_capacity_vec(args.len()).map_err(|e| policy.storage(e))?;
    let mut handles =
        runmat_value::exact_capacity_vec(args.len()).map_err(|e| policy.storage(e))?;
    let count = dimension
        .checked_add(1)
        .ok_or_else(|| policy.invalid("concatenation dimension overflow"))?;
    let dimension_value = Value::Int(runmat_value::IntValue::U64(
        u64::try_from(count)
            .map_err(|_| policy.invalid("concatenation dimension exceeds uint64"))?,
    ));
    let mut facts = runmat_value::exact_capacity_vec(
        args.len()
            .checked_add(1)
            .ok_or_else(|| policy.invalid("concatenation arity overflow"))?,
    )
    .map_err(|e| policy.storage(e))?;
    let mut literals = runmat_value::exact_capacity_vec(
        args.len()
            .checked_add(1)
            .ok_or_else(|| policy.invalid("concatenation arity overflow"))?,
    )
    .map_err(|e| policy.storage(e))?;
    facts.push(crate::value_fact::value_fact(&dimension_value));
    literals.push(metadata(&dimension_value, policy, cooperation).await?);
    for value in args {
        let root = owner::as_expression(value, policy, cooperation).await?;
        let handle = owner::reference(root.value(), policy)?;
        let node = owner::OwnedNode::read(handle, policy)?;
        facts.push(ValueFact::proven(
            ValueKindFact::Numeric(NumericFact {
                class: NumericClass::Double,
                domain: NumericDomain::Real,
            }),
            ShapeFact::from(
                node.node()
                    .shape
                    .iter()
                    .map(|&n| Some(n))
                    .collect::<Vec<_>>(),
            ),
            StorageFact::Dense,
        ));
        literals.push(LiteralValue::Unknown);
        handles.push(handle);
        roots.push(root);
    }
    let entry = runmat_builtins::builtin_catalog_entry_by_name("cat")
        .ok_or_else(|| policy.invalid("canonical cat is unavailable"))?;
    let shape = operators::infer_shape(entry, facts, LiteralContext::new(literals), policy)?;
    let index_names = construct::unnamed(shape.len(), policy)?;
    owner::publish(
        Node {
            shape,
            index_names,
            kind: Kind::Concatenate {
                dimension,
                operands: handles,
            },
        },
        policy,
        cooperation,
    )
    .await
}

async fn metadata(
    value: &Value,
    policy: Policy,
    cooperation: &mut ScalarCooperation,
) -> BuiltinResult<LiteralValue> {
    if matches!(value, Value::Num(_) | Value::Int(_) | Value::Tensor(_)) {
        // Validate concrete shape before inspecting its logical numeric values.
        construct::input_shape(value, policy, cooperation).await?;
        construct::finite_constant(value, policy, cooperation).await?;
    }
    let result = match value {
        Value::Num(value) => LiteralValue::Number(*value),
        Value::Int(value) => integer(value.clone()),
        Value::Bool(value) => LiteralValue::Bool(*value),
        Value::String(_) | Value::StringArray(_) | Value::CharArray(_) => {
            LiteralValue::String(policy.text(value, cooperation.runtime())?.as_str().into())
        }
        Value::Tensor(value) => {
            value.validate_shape().map_err(|e| policy.invalid(e))?;
            let mut elements =
                runmat_value::exact_capacity_vec(value.len()).map_err(|e| policy.storage(e))?;
            for index in 0..value.len() {
                use runmat_value::NumericScalar as N;
                let scalar = value
                    .numeric_value_at(index)
                    .ok_or_else(|| policy.invalid("invalid numerical metadata storage"))?;
                elements.push(match scalar {
                    N::F64(v) => LiteralValue::Number(v),
                    N::F32(v) => LiteralValue::Number(f64::from(v)),
                    scalar => integer(scalar.into_int_value().expect("integer scalar")),
                });
                cooperation.completed(1).await?;
            }
            if elements.is_empty() {
                LiteralValue::Empty
            } else if elements.len() == 1 {
                elements.pop().expect("one literal")
            } else if value.shape().iter().filter(|&&n| n > 1).count() <= 1 {
                LiteralValue::Vector(elements)
            } else {
                return Err(policy.invalid("expression function metadata must be scalar or vector"));
            }
        }
        _ => {
            return Err(policy
                .invalid("expression function metadata must be concrete numeric, logical or text"))
        }
    };
    Ok(result)
}
fn integer(value: runmat_value::IntValue) -> LiteralValue {
    let ValueKindFact::Numeric(fact) =
        crate::value_fact::value_fact(&Value::Int(value.clone())).kind
    else {
        unreachable!()
    };
    LiteralValue::Integer {
        text: value.to_i128().to_string(),
        class: fact.class,
    }
}

//! Dimension/shape admission shared by the real expression producers.
use super::*;
use crate::builtins::common::shape::value_dimensions_borrowed;
pub(super) fn unnamed(
    rank: usize,
    policy: Policy,
) -> BuiltinResult<runmat_value::optimization_model::ModelAxisNames> {
    let mut names =
        runmat_value::exact_capacity_vec(rank).map_err(|source| policy.storage(source))?;
    for _ in 0..rank {
        names.push(None)
    }
    runmat_value::optimization_model::model_axis_names_unbounded(names)
        .map_err(|source| policy.storage(source))
}
pub(super) async fn input_shape(
    value: &Value,
    policy: Policy,
    cooperation: &mut ScalarCooperation,
) -> BuiltinResult<SharedShape> {
    match value {
        Value::Num(_) | Value::Int(_) => {}
        Value::Tensor(array) => {
            array
                .validate_shape()
                .map_err(|source| policy.invalid(source))?;
        }
        Value::SparseTensor(array) if array.numeric_dtype().is_some() && !array.is_complex() => {
            let mut validation = runmat_value::SparseCscValidation::new(
                array.rows,
                array.cols,
                array.col_ptrs.clone(),
                array.row_indices.clone(),
                array.nnz(),
            )
            .map_err(|source| policy.invalid(source))?;
            while !validation
                .advance(std::num::NonZeroUsize::new(1024).expect("positive chunk"))
                .map_err(|source| policy.invalid(source))?
            {
                cooperation.completed(1024).await?;
            }
        }
        _ => return Err(policy.invalid("model values require real numeric storage")),
    }
    let dimensions = value_dimensions_borrowed(value)?;
    dimensions
        .iter()
        .try_fold(1usize, |n, dimension| n.checked_mul(*dimension))
        .ok_or_else(|| policy.invalid("model constant shape overflows usize"))?;
    let mut shape = runmat_value::exact_capacity_vec(dimensions.len())
        .map_err(|source| policy.storage(source))?;
    shape.extend_from_slice(&dimensions);
    Ok(SharedShape::new_unmanaged(shape))
}
pub(super) async fn numeric_shape(
    value: &Value,
    policy: Policy,
    cooperation: &mut ScalarCooperation,
) -> BuiltinResult<SharedShape> {
    if !matches!(value, Value::Num(_))
        && !matches!(value, Value::Tensor(array) if array.numeric_dtype() == runmat_value::NumericDType::F64)
        && !matches!(value, Value::SparseTensor(array) if array.numeric_dtype() == Some(runmat_value::NumericDType::F64) && !array.is_complex())
    {
        return Err(policy.invalid("expression numerical evaluation requires real double storage"));
    }
    input_shape(value, policy, cooperation).await
}
pub(super) async fn normalize(
    value: &Value,
    policy: Policy,
    cooperation: &mut ScalarCooperation,
) -> BuiltinResult<RootedValue> {
    let _shape = input_shape(value, policy, cooperation).await?;
    if matches!(value, Value::Num(_))
        || matches!(value, Value::Tensor(array) if array.numeric_dtype() == runmat_value::NumericDType::F64)
        || matches!(value, Value::SparseTensor(array) if array.numeric_dtype() == Some(runmat_value::NumericDType::F64))
    {
        return RootedValue::in_optional_context(value.clone(), cooperation.runtime());
    }
    // Explicit conversion to evaluate's documented double result uses the
    // actual canonical cast owner, including its range/precision behavior.
    let output =
        evaluation::invoke("double", std::slice::from_ref(value), policy, cooperation).await?;
    numeric_shape(output.value(), policy, cooperation).await?;
    Ok(output)
}

pub(super) async fn empty(
    args: crate::call::FunctionArgumentView<'_>,
    policy: Policy,
    cooperation: &mut ScalarCooperation,
) -> BuiltinResult<RootedValue> {
    if args.is_empty() {
        return owner::publish(
            Node {
                shape: SharedShape::new_unmanaged(vec![0, 0]),
                index_names: unnamed(2, policy)?,
                kind: Kind::EmptyInitializer,
            },
            policy,
            cooperation,
        )
        .await;
    }
    let (shape, index_names) = super::super::common::model_dimensions::constructor(
        args,
        cooperation,
        super::super::common::model_dimensions::Policy {
            identity: policy.identity,
            invalid: policy.invalid,
        },
        super::super::common::model_dimensions::ConstructorSyntax::Expression,
    )
    .await?;
    owner::publish(
        Node {
            shape,
            index_names,
            kind: Kind::Unassigned,
        },
        policy,
        cooperation,
    )
    .await
}

pub(super) async fn zeros(
    args: crate::call::FunctionArgumentView<'_>,
    policy: Policy,
    cooperation: &mut ScalarCooperation,
) -> BuiltinResult<RootedValue> {
    let shape = crate::builtins::array::creation::zeros::parse_creation_dimensions(
        args,
        cooperation.runtime(),
    )
    .await?;
    evaluation::element_count(&shape, policy)?;
    let index_names = unnamed(shape.len(), policy)?;
    owner::publish(
        Node {
            shape: SharedShape::new_unmanaged(shape),
            index_names,
            kind: Kind::Zero,
        },
        policy,
        cooperation,
    )
    .await
}

/// Construction constants, unlike evaluation values, must be finite.
pub(super) async fn finite_constant(
    value: &Value,
    policy: Policy,
    cooperation: &mut ScalarCooperation,
) -> BuiltinResult<()> {
    if !finite_numeric(value, policy, cooperation).await? {
        return Err(policy.invalid("model construction constants must be finite and real"));
    }
    Ok(())
}
/// Shared storage inspection; the caller chooses construction refusal versus
/// a numerical undefined stage. Point conversion has no implicit finite gate.
pub(super) async fn finite_numeric(
    value: &Value,
    policy: Policy,
    cooperation: &mut ScalarCooperation,
) -> BuiltinResult<bool> {
    use runmat_value::NumericScalar as N;
    let (count, scalar) = match value {
        Value::Num(number) => (1, Some(N::F64(*number))),
        Value::Int(_) => return Ok(true),
        Value::Tensor(array) => (array.len(), None),
        Value::SparseTensor(array) => (array.nnz(), None),
        _ => return Err(policy.invalid("model constants require real numeric storage")),
    };
    for index in 0..count {
        let value = match value {
            Value::Tensor(array) => array.numeric_value_at(index),
            Value::SparseTensor(array) => array.numeric_value_at(index),
            _ => scalar,
        }
        .ok_or_else(|| policy.invalid("model constant storage differs from its shape"))?;
        if matches!(value,N::F64(value) if !value.is_finite())
            || matches!(value,N::F32(value) if !value.is_finite())
        {
            return Ok(false);
        }
        cooperation.completed(1).await?;
    }
    Ok(true)
}

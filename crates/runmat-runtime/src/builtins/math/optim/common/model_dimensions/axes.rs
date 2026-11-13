//! Numeric and named N-D dimensions; no index metadata is a second variable ID.
use super::super::model_storage;
use super::*;
use runmat_value::{
    optimization_model::{ModelAxisLabels, ModelAxisNames},
    SharedShape,
};
struct Axis {
    extent: usize,
    names: Option<ModelAxisLabels>,
    row: bool,
}
fn scalar_text(value: &Value) -> bool {
    matches!(value, Value::String(_) | Value::CharArray(_))
        || matches!(value,Value::StringArray(array) if array.data().len()==1)
}
fn positive_dimension(value: f64, policy: Policy) -> BuiltinResult<usize> {
    runmat_types::decode_shape_dimension(value)
        .ok()
        .filter(|value| *value > 0)
        .ok_or_else(|| policy.invalid("dimensions must be positive double integers"))
}
enum NumericDimensions<'a> {
    Scalar(f64),
    Tensor(&'a runmat_value::Tensor),
}
impl NumericDimensions<'_> {
    fn len(&self) -> usize {
        match self {
            Self::Scalar(_) => 1,
            Self::Tensor(value) => value.len(),
        }
    }
    fn at(&self, index: usize, policy: Policy) -> BuiltinResult<f64> {
        match self {
            Self::Scalar(value) if index == 0 => Ok(*value),
            Self::Tensor(value) => match value.numeric_value_at(index) {
                Some(runmat_value::NumericScalar::F64(value)) => Ok(value),
                _ => Err(policy.invalid("dimension storage is inconsistent")),
            },
            _ => Err(policy.invalid("dimension coordinate exceeds its carrier")),
        }
    }
}
fn numeric(value: &Value, policy: Policy) -> BuiltinResult<Option<NumericDimensions<'_>>> {
    Ok(match value {
        Value::Num(value) => Some(NumericDimensions::Scalar(*value)),
        Value::Tensor(array)
            if array.numeric_dtype() == runmat_value::NumericDType::F64
                && array.shape().len() == 2
                && (array.shape()[0] == 1 || array.shape()[1] == 1) =>
        {
            array
                .validate_shape()
                .map_err(|source| policy.invalid(source))?;
            Some(NumericDimensions::Tensor(array))
        }
        _ => None,
    })
}
async fn names(
    value: &Value,
    cooperation: &mut ScalarCooperation,
    policy: Policy,
    syntax: ConstructorSyntax,
) -> BuiltinResult<Axis> {
    let scalar_shape = [1, 1];
    let (shape, raw) = match value {
        Value::String(text @ runmat_value::StringElement::Text(_))
            if syntax == ConstructorSyntax::Expression =>
        {
            (&scalar_shape[..], Some(std::slice::from_ref(text)))
        }
        Value::Cell(cell) => {
            cell.validate_layout()
                .map_err(|_| policy.invalid("invalid named-axis cell layout"))?;
            (cell.shape(), None)
        }
        Value::StringArray(array)
            if array.data().len() > 1
                || (syntax == ConstructorSyntax::Expression && array.data().len() == 1) =>
        {
            array
                .validate_layout()
                .map_err(|source| policy.invalid(source))?;
            (&array.shape[..], Some(array.data()))
        }
        _ => {
            return Err(
                policy.invalid("named axes require a string vector or cell vector of scalar text")
            )
        }
    };
    if shape.len() != 2 || (shape[0] != 1 && shape[1] != 1) {
        return Err(policy.invalid("index names must form a nonempty vector"));
    }
    let len = match value {
        Value::Cell(cell) => cell.data().len(),
        _ => raw.expect("string carrier").len(),
    };
    if len == 0 {
        return Err(policy.invalid("named axes must be nonempty"));
    }
    let mut labels = model_storage::array(cooperation, len)?;
    for index in 0..len {
        cooperation.check_allocation_epoch()?;
        let name = match value {
            Value::Cell(cell) => policy.text(&cell.data()[index], cooperation.runtime())?,
            _ => crate::call::argument_text::convert(
                &Value::String(raw.expect("string carrier")[index].clone()),
                cooperation.runtime(),
            )?,
        };
        cooperation.check_allocation_epoch()?;
        if name.is_empty() {
            return Err(policy.invalid("index names must be nonempty"));
        }
        labels
            .try_push(name)
            .map_err(|_| policy.invalid("named-axis backing extent changed"))?;
        cooperation.completed(1).await?;
    }
    let labels = labels
        .try_finish()
        .map_err(|_| policy.invalid("named-axis backing is incomplete"))?;
    let storage = model_storage::Collections::new(cooperation)?;
    let mut unique = storage.set();
    storage.reserve_set(&mut unique, labels.len())?;
    for label in &labels {
        if !unique.insert(label.as_str()) {
            return Err(policy.invalid("index names on each axis must be unique"));
        }
        cooperation.completed(1).await?;
        storage.check()?;
    }
    drop(unique);
    Ok(Axis {
        extent: len,
        names: Some(labels),
        row: shape[0] == 1,
    })
}
async fn axis(
    value: &Value,
    cooperation: &mut ScalarCooperation,
    policy: Policy,
    syntax: ConstructorSyntax,
) -> BuiltinResult<Axis> {
    if let Some(values) = numeric(value, policy)? {
        if values.len() != 1 {
            return Err(policy.invalid("each separate numeric axis must be scalar"));
        }
        return Ok(Axis {
            extent: positive_dimension(values.at(0, policy)?, policy)?,
            names: None,
            row: false,
        });
    }
    names(value, cooperation, policy, syntax).await
}
pub(in crate::builtins::math::optim) async fn constructor(
    args: crate::call::FunctionArgumentView<'_>,
    cooperation: &mut ScalarCooperation,
    policy: Policy,
    syntax: ConstructorSyntax,
) -> BuiltinResult<(SharedShape, ModelAxisNames)> {
    if args.is_empty() {
        let shape = model_storage::shape(2, |_| Ok(1), cooperation).await?;
        let mut names = model_storage::array(cooperation, 2)?;
        for _ in 0..2 {
            names
                .try_push(None)
                .map_err(|_| policy.invalid("axis extent changed"))?;
        }
        return Ok((
            shape,
            names
                .try_finish()
                .map_err(|_| policy.invalid("axis backing is incomplete"))?,
        ));
    }
    let count = if args.len() == 1 {
        if let Some(values) = numeric(&args[0], policy)? {
            values.len()
        } else if let Value::Cell(cell) = &args[0] {
            if all_scalar_text(cell.data(), cooperation).await? {
                1
            } else {
                cell.data().len()
            }
        } else {
            1
        }
    } else {
        args.len()
    };
    let mut axes = model_storage::array(cooperation, count)?;
    if args.len() == 1 {
        if let Some(values) = numeric(&args[0], policy)? {
            for index in 0..values.len() {
                let value = values.at(index, policy)?;
                axes.try_push(Axis {
                    extent: positive_dimension(value, policy)?,
                    names: None,
                    row: false,
                })
                .map_err(|_| policy.invalid("axis extent changed"))?;
                cooperation.completed(1).await?;
            }
        } else if let Value::Cell(cell) = &args[0] {
            cell.validate_layout()
                .map_err(|_| policy.invalid("invalid dimensions cell layout"))?;
            if all_scalar_text(cell.data(), cooperation).await? {
                axes.try_push(names(&args[0], cooperation, policy, syntax).await?)
                    .map_err(|_| policy.invalid("axis extent changed"))?;
            } else {
                for value in cell.data() {
                    axes.try_push(axis(value, cooperation, policy, syntax).await?)
                        .map_err(|_| policy.invalid("axis extent changed"))?;
                }
            }
        } else {
            axes.try_push(axis(&args[0], cooperation, policy, syntax).await?)
                .map_err(|_| policy.invalid("axis extent changed"))?;
        }
    } else {
        for value in args {
            axes.try_push(axis(value, cooperation, policy, syntax).await?)
                .map_err(|_| policy.invalid("axis extent changed"))?;
        }
    }
    let axes = axes
        .try_finish()
        .map_err(|_| policy.invalid("axis backing is incomplete"))?;
    if axes.is_empty() {
        return Err(policy.invalid("dimension lists must be nonempty"));
    }
    let rank = axes.len().max(2);
    let shift = usize::from(axes.len() == 1 && axes[0].row && axes[0].names.is_some());
    let at = |index: usize| {
        if axes.len() == 1 {
            (index == shift).then(|| &axes[0])
        } else {
            axes.get(index)
        }
    };
    let shape = model_storage::shape(
        rank,
        |index| Ok(at(index).map_or(1, |axis| axis.extent)),
        cooperation,
    )
    .await?;
    let mut labels = model_storage::array(cooperation, rank)?;
    for index in 0..rank {
        labels
            .try_push(at(index).and_then(|axis| axis.names.clone()))
            .map_err(|_| policy.invalid("axis extent changed"))?;
        cooperation.completed(1).await?;
    }
    Ok((
        shape,
        labels
            .try_finish()
            .map_err(|_| policy.invalid("axis backing is incomplete"))?,
    ))
}

pub(in crate::builtins::math::optim) async fn property(
    value: &Value,
    shape: &[usize],
    cooperation: &mut ScalarCooperation,
    policy: Policy,
) -> BuiltinResult<ModelAxisNames> {
    let Value::Cell(axes) = value else {
        return Err(policy.invalid("IndexNames must be a cell with one entry per axis"));
    };
    axes.validate_layout()
        .map_err(|_| policy.invalid("invalid IndexNames layout"))?;
    if axes.data().len() != shape.len() {
        return Err(policy.invalid("IndexNames must have one entry per axis"));
    }
    let mut result = model_storage::array(cooperation, shape.len())?;
    for (value, &extent) in axes.data().iter().zip(shape) {
        let empty = matches!(value,Value::Cell(value) if value.data().is_empty())
            || matches!(value,Value::Tensor(value) if value.is_empty())
            || matches!(value,Value::CharArray(value) if value.data().is_empty())
            || matches!(value,Value::String(runmat_value::StringElement::Text(value)) if value.is_empty());
        if empty {
            result
                .try_push(None)
                .map_err(|_| policy.invalid("axis extent changed"))?;
        } else {
            let axis = if scalar_text(value) && extent == 1 {
                let mut labels = model_storage::array(cooperation, 1)?;
                labels
                    .try_push(policy.text(value, cooperation.runtime())?)
                    .map_err(|_| policy.invalid("axis extent changed"))?;
                Axis {
                    extent: 1,
                    names: Some(
                        labels
                            .try_finish()
                            .map_err(|_| policy.invalid("axis backing is incomplete"))?,
                    ),
                    row: true,
                }
            } else {
                names(value, cooperation, policy, ConstructorSyntax::Variable).await?
            };
            if axis.extent != extent {
                return Err(policy.invalid("IndexNames axis extent does not match the variable"));
            }
            result
                .try_push(axis.names)
                .map_err(|_| policy.invalid("axis extent changed"))?;
        }
        cooperation.completed(1).await?;
    }
    result
        .try_finish()
        .map_err(|_| policy.invalid("axis backing is incomplete"))
}

async fn all_scalar_text(
    values: &[Value],
    cooperation: &mut ScalarCooperation,
) -> BuiltinResult<bool> {
    for value in values {
        let scalar = scalar_text(value);
        cooperation.completed(1).await?;
        if !scalar {
            return Ok(false);
        }
    }
    Ok(true)
}

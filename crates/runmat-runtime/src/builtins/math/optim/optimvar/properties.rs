//! Public getters and atomic property replacement of the original definition.
use super::super::common::ScalarCooperation;
use super::*;
use runmat_builtins::OptimizationVariableProperty as Property;
use runmat_value::optimization_variable::*;
use runmat_value::{CellArray, Tensor, Utf8Text};
mod write;
pub(super) use write::{ensure_writable, set};

fn source_axis(view: &View, axis: usize, position: usize) -> Option<usize> {
    let OptimizationVariableAxisSource::DefinitionAxis(source) =
        *view.selection().axis_sources().get(axis)?
    else {
        return None;
    };
    let output_stride = view
        .selection()
        .shape()
        .iter()
        .take(axis)
        .try_fold(1usize, |a, &b| a.checked_mul(b))?;
    let original_stride = view
        .definition()
        .shape()
        .iter()
        .take(source)
        .try_fold(1usize, |a, &b| a.checked_mul(b))?;
    let index = view
        .selection()
        .definition_index(position.checked_mul(output_stride)?)?;
    Some((index / original_stride) % view.definition().shape()[source])
}
pub(in crate::builtins::math::optim) fn axis_name(
    view: &View,
    axis: usize,
    position: usize,
) -> Option<&Utf8Text> {
    let OptimizationVariableAxisSource::DefinitionAxis(source) =
        *view.selection().axis_sources().get(axis)?
    else {
        return None;
    };
    let labels = view.definition().index_names().get(source)?.as_ref()?;
    labels.get(source_axis(view, axis, position)?)
}
pub(super) async fn get(
    view: &View,
    property: Property,
    cooperation: &mut ScalarCooperation,
) -> BuiltinResult<RootedValue> {
    let value = match property {
        Property::Name => Value::string(view.definition().name_owner().clone()),
        Property::Type => Value::string(crate::call::argument_text::copy_str(
            view.definition().domain().name(),
            cooperation.runtime(),
        )?),
        Property::IndexNames => {
            let mut axes = runmat_value::exact_capacity_vec(view.selection().shape().len())
                .map_err(data_error)?;
            for (axis, &extent) in view.selection().shape().iter().enumerate() {
                let mut names = Vec::new();
                if extent > 0 && axis_name(view, axis, 0).is_some() {
                    names = runmat_value::exact_capacity_vec(extent).map_err(data_error)?;
                    for position in 0..extent {
                        names.push(Value::string(
                            axis_name(view, axis, position)
                                .ok_or_else(|| invalid("invalid reference index-name projection"))?
                                .clone(),
                        ));
                        cooperation.completed(1).await?;
                    }
                }
                let columns = names.len();
                axes.push(Value::Cell(
                    CellArray::new(names, if columns == 0 { 0 } else { 1 }, columns)
                        .map_err(invalid)?,
                ));
                cooperation.completed(1).await?;
            }
            let count = axes.len();
            Value::Cell(CellArray::new(axes, 1, count).map_err(invalid)?)
        }
        Property::LowerBound | Property::UpperBound => {
            let role = if property == Property::LowerBound {
                OptimizationBoundRole::Lower
            } else {
                OptimizationBoundRole::Upper
            };
            let count = view.selection().element_count();
            let mut values = runmat_value::exact_capacity_vec(count).map_err(data_error)?;
            for position in 0..count {
                let index = view
                    .selection()
                    .definition_index(position)
                    .ok_or_else(|| invalid("invalid variable reference index"))?;
                values.push(
                    view.definition()
                        .bound_value(role, index)
                        .ok_or_else(|| invalid("original bound is incomplete"))?,
                );
                cooperation.completed(1).await?;
            }
            Value::Tensor(Tensor::new(values, view.selection().shape().clone()).map_err(invalid)?)
        }
    };
    RootedValue::in_optional_context(value, cooperation.runtime())
}

pub(super) fn preflight(property: Property, value: &Value) -> BuiltinResult<()> {
    let text = |value: &Value| {
        matches!(
            value,
            Value::String(_) | Value::StringArray(_) | Value::CharArray(_)
        )
    };
    let accepted = match property {
        Property::Name => false,
        Property::Type => text(value),
        Property::LowerBound | Property::UpperBound => {
            matches!(value, Value::Num(_))
                || matches!(value,Value::Tensor(value) if value.as_f64_slice().is_some())
        }
        Property::IndexNames => {
            text(value)
                || matches!(value,Value::Cell(cell) if cell.data().iter().all(|value|text(value)||matches!(value,Value::Cell(axis) if axis.data().iter().all(text))))
        }
    };
    if accepted {
        Ok(())
    } else {
        Err(invalid(
            "assigned value has an unsupported property carrier",
        ))
    }
}

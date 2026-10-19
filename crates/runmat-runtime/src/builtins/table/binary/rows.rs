use std::cmp::Ordering;

use runmat_value::{NumericScalar, ObjectInstance, Value};

pub(super) fn align_right_to_left(
    left: &ObjectInstance,
    right: &ObjectInstance,
) -> Result<Vec<usize>, String> {
    let left_height = super::super::table_height(left).map_err(display)?;
    let right_height = super::super::table_height(right).map_err(display)?;
    if left_height != right_height {
        return Err(format!(
            "tabular operands have different heights ({left_height} and {right_height})"
        ));
    }
    if left.is_class(super::super::TIMETABLE_CLASS) {
        return align_row_times(left, right, left_height);
    }
    align_row_names(left, right, left_height)
}

fn align_row_names(
    left: &ObjectInstance,
    right: &ObjectInstance,
    height: usize,
) -> Result<Vec<usize>, String> {
    let all_rows = (0..height).collect::<Vec<_>>();
    let left_names = super::super::selected_row_names(left, &all_rows).map_err(display)?;
    let right_names = super::super::selected_row_names(right, &all_rows).map_err(display)?;
    match (left_names, right_names) {
        (None, None) => Ok(all_rows),
        (Some(left), Some(right)) => permutation(&left, &right, "row names"),
        _ => Err("tabular operands must either both define row names or both omit them".into()),
    }
}

fn align_row_times(
    left: &ObjectInstance,
    right: &ObjectInstance,
    height: usize,
) -> Result<Vec<usize>, String> {
    let left = super::super::timetable_row_times(left)
        .map_err(display)?
        .ok_or_else(|| "left timetable does not define row times".to_string())?;
    let right = super::super::timetable_row_times(right)
        .map_err(display)?
        .ok_or_else(|| "right timetable does not define row times".to_string())?;
    let left = row_time_values(&left)?;
    let right = row_time_values(&right)?;
    if left.len() != height || right.len() != height {
        return Err("timetable row-time count must match table height".into());
    }
    let mut used = vec![false; right.len()];
    let mut order = Vec::with_capacity(left.len());
    for expected in left {
        let Some(index) = right.iter().enumerate().position(|(index, actual)| {
            !used[index]
                && crate::builtins::logical::rel::integer_comparison::compare_numeric_scalars_exact(
                    expected, *actual,
                ) == Some(Ordering::Equal)
        }) else {
            return Err("timetable operands must contain the same row times".into());
        };
        used[index] = true;
        order.push(index);
    }
    Ok(order)
}

fn row_time_values(value: &Value) -> Result<Vec<NumericScalar>, String> {
    match value {
        Value::Tensor(tensor) => (0..tensor.len())
            .map(|index| {
                tensor
                    .numeric_value_at(index)
                    .ok_or_else(|| "timetable row time is missing".to_string())
            })
            .collect(),
        Value::Num(value) => Ok(vec![NumericScalar::F64(*value)]),
        Value::Int(value) => Ok(vec![NumericScalar::from(value.clone())]),
        Value::Object(object) if object.is_class(runmat_types::standard::DATETIME) => {
            let tensor =
                crate::builtins::datetime::serials_from_datetime_value(value).map_err(display)?;
            numeric_values(&tensor)
        }
        Value::Object(object) if object.is_class(runmat_types::standard::DURATION) => {
            let tensor = crate::builtins::duration::duration_tensor_from_duration_value(value)
                .map_err(display)?;
            numeric_values(&tensor)
        }
        other => Err(format!("unsupported timetable row-time value {other:?}")),
    }
}

fn numeric_values(tensor: &runmat_value::Tensor) -> Result<Vec<NumericScalar>, String> {
    (0..tensor.len())
        .map(|index| {
            tensor
                .numeric_value_at(index)
                .ok_or_else(|| "timetable row-time storage is inconsistent".to_string())
        })
        .collect()
}

fn permutation<T: Eq>(left: &[T], right: &[T], identity: &str) -> Result<Vec<usize>, String> {
    let mut used = vec![false; right.len()];
    let mut order = Vec::with_capacity(left.len());
    for expected in left {
        let Some(index) = right
            .iter()
            .enumerate()
            .position(|(index, actual)| !used[index] && actual == expected)
        else {
            return Err(format!("tabular operands must contain the same {identity}"));
        };
        used[index] = true;
        order.push(index);
    }
    Ok(order)
}

fn display(error: impl std::fmt::Display) -> String {
    error.to_string()
}

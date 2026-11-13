use runmat_value::{ObjectInstance, Value};

use crate::builtins::common::binary::BinaryInputPlan;
use crate::execution::{RootedValue, RootedValues, RootedValuesBuilder};

pub(crate) struct TabularBinaryPlan {
    pub(crate) source: ObjectInstance,
    pub(crate) variables: Vec<(String, Value)>,
    pub(crate) right_values: RootedValues,
}

pub(crate) fn plan(
    left: Value,
    right: Value,
) -> Result<BinaryInputPlan<TabularBinaryPlan>, String> {
    let left_tabular =
        matches!(&left, Value::Object(object) if super::super::is_tabular_object(object));
    let right_tabular =
        matches!(&right, Value::Object(object) if super::super::is_tabular_object(object));
    if !left_tabular && !right_tabular {
        return Ok(BinaryInputPlan::Values(Box::new((left, right))));
    }

    match (left, right) {
        (Value::Object(left), Value::Object(right))
            if super::super::is_tabular_object(&left)
                && super::super::is_tabular_object(&right) =>
        {
            pair_tabular(left, right).map(BinaryInputPlan::Structured)
        }
        (Value::Object(source), other) if super::super::is_tabular_object(&source) => {
            let fields = super::super::table_variables(&source)
                .map_err(|error| error.to_string())?
                .fields;
            let mut variables = Vec::with_capacity(fields.len());
            let mut right_values = RootedValuesBuilder::with_capacity(fields.len());
            for (name, left) in fields.iter() {
                right_values
                    .push(other.clone())
                    .map_err(|error| error.to_string())?;
                variables.push((name.to_owned(), left.clone()));
            }
            Ok(BinaryInputPlan::Structured(TabularBinaryPlan {
                source,
                variables,
                right_values: right_values.finish().map_err(|error| error.to_string())?,
            }))
        }
        (other, Value::Object(source)) if super::super::is_tabular_object(&source) => {
            let fields = super::super::table_variables(&source)
                .map_err(|error| error.to_string())?
                .fields;
            let mut variables = Vec::with_capacity(fields.len());
            let mut right_values = RootedValuesBuilder::with_capacity(fields.len());
            for (name, right) in fields.iter() {
                right_values
                    .push(right.clone())
                    .map_err(|error| error.to_string())?;
                variables.push((name.to_owned(), other.clone()));
            }
            Ok(BinaryInputPlan::Structured(TabularBinaryPlan {
                source,
                variables,
                right_values: right_values.finish().map_err(|error| error.to_string())?,
            }))
        }
        (left, right) => Ok(BinaryInputPlan::Values(Box::new((left, right)))),
    }
}

fn pair_tabular(left: ObjectInstance, right: ObjectInstance) -> Result<TabularBinaryPlan, String> {
    if left.class_name() != right.class_name() {
        return Err("tabular operands must have the same container class".into());
    }
    let left_variables = super::super::table_variables(&left).map_err(|error| error.to_string())?;
    let right_variables =
        super::super::table_variables(&right).map_err(|error| error.to_string())?;
    if left_variables.fields.len() != right_variables.fields.len()
        || left_variables
            .fields
            .keys()
            .any(|name| !right_variables.fields.contains_key(name))
    {
        return Err("tabular operands must contain the same variable names".into());
    }
    let right_rows = super::rows::align_right_to_left(&left, &right)?;
    let identity_rows = right_rows.iter().copied().eq(0..right_rows.len());
    let mut variables = Vec::with_capacity(left_variables.fields.len());
    let mut right_values = RootedValuesBuilder::with_capacity(left_variables.fields.len());
    for (name, left_value) in left_variables.into_fields_unbounded() {
        let right_value = right_variables
            .fields
            .get(&name)
            .cloned()
            .ok_or_else(|| format!("right tabular operand is missing variable '{name}'"))?;
        if identity_rows {
            right_values
                .push(right_value)
                .map_err(|error| error.to_string())?;
        } else {
            // `select_rows` is synchronous and currently retires no
            // temporary managed root internally. Publish its newly selected
            // graph immediately, before another field selection can allocate.
            let selected = super::super::select_rows(&right_value, &right_rows)
                .map_err(|error| error.to_string())?;
            let selected = RootedValue::new(selected).map_err(|error| error.to_string())?;
            right_values.push_owned(selected);
        }
        variables.push((name, left_value));
    }
    Ok(TabularBinaryPlan {
        source: left,
        variables,
        right_values: right_values.finish().map_err(|error| error.to_string())?,
    })
}

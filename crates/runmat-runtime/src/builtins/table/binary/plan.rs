use runmat_value::{ObjectInstance, Value};

use crate::builtins::common::binary::BinaryInputPlan;

pub(crate) struct TabularBinaryPlan {
    source: ObjectInstance,
    variables: Vec<(String, Value, Value)>,
}

impl TabularBinaryPlan {
    pub(crate) fn variables(self) -> (ObjectInstance, Vec<(String, Value, Value)>) {
        (self.source, self.variables)
    }
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
            let variables = super::super::table_variables(&source)
                .map_err(|error| error.to_string())?
                .fields
                .into_iter()
                .map(|(name, left)| (name, left, other.clone()))
                .collect();
            Ok(BinaryInputPlan::Structured(TabularBinaryPlan {
                source,
                variables,
            }))
        }
        (other, Value::Object(source)) if super::super::is_tabular_object(&source) => {
            let variables = super::super::table_variables(&source)
                .map_err(|error| error.to_string())?
                .fields
                .into_iter()
                .map(|(name, right)| (name, other.clone(), right))
                .collect();
            Ok(BinaryInputPlan::Structured(TabularBinaryPlan {
                source,
                variables,
            }))
        }
        (left, right) => Ok(BinaryInputPlan::Values(Box::new((left, right)))),
    }
}

fn pair_tabular(left: ObjectInstance, right: ObjectInstance) -> Result<TabularBinaryPlan, String> {
    if left.class_name != right.class_name {
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
    for (name, left_value) in left_variables.fields {
        let right_value = right_variables
            .fields
            .get(&name)
            .cloned()
            .ok_or_else(|| format!("right tabular operand is missing variable '{name}'"))?;
        let right_value = if identity_rows {
            right_value
        } else {
            super::super::select_rows(&right_value, &right_rows)
                .map_err(|error| error.to_string())?
        };
        variables.push((name, left_value, right_value));
    }
    Ok(TabularBinaryPlan {
        source: left,
        variables,
    })
}

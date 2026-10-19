use runmat_value::Value;

use crate::builtins::table::{
    is_tabular_object, table_variable_names_from_object, table_variables,
};
use crate::BuiltinResult;

use super::error;
use super::groups::{Groups, SplitAxis};

pub(super) struct Input {
    pub(super) function: Value,
    pub(super) data: Vec<Value>,
    pub(super) groups: Groups,
}

impl Input {
    pub(super) fn prepare(
        function: Value,
        first_data: Value,
        rest: Vec<Value>,
    ) -> BuiltinResult<Self> {
        let (group_numbers, data_tail) = rest.split_last().ok_or_else(|| {
            error::invalid("splitapply: expected at least one data input followed by group numbers")
        })?;
        super::extensions::validate(group_numbers)?;
        let groups = Groups::parse(group_numbers)?;
        let mut data = Vec::with_capacity(data_tail.len() + 1);
        expand_data(first_data, groups.axis, &mut data)?;
        for value in data_tail {
            expand_data(value.clone(), groups.axis, &mut data)?;
        }
        if data.is_empty() {
            return Err(error::invalid(
                "splitapply: expected at least one data input",
            ));
        }
        for value in &data {
            super::slice::validate_observation_count(value, &groups)?;
        }
        Ok(Self {
            function,
            data,
            groups,
        })
    }
}

fn expand_data(value: Value, axis: SplitAxis, output: &mut Vec<Value>) -> BuiltinResult<()> {
    let Value::Object(object) = &value else {
        output.push(value);
        return Ok(());
    };
    if !is_tabular_object(object) {
        output.push(value);
        return Ok(());
    }
    if axis == SplitAxis::Columns {
        return Err(error::invalid(
            "splitapply: table data requires a column group vector",
        ));
    }
    let names = table_variable_names_from_object(object)?;
    let variables = table_variables(object)?;
    for name in names {
        output.push(variables.fields.get(&name).cloned().ok_or_else(|| {
            error::invalid(format!("splitapply: missing table variable '{name}'"))
        })?);
    }
    Ok(())
}

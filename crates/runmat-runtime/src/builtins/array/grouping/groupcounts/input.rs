use runmat_value::Value;

use crate::builtins::array::grouping::variables::{columns_from_value, GroupColumn};
use crate::builtins::table::{
    is_tabular_object, parse_variable_selector_for_object, table_height,
    table_variable_names_from_object, table_variables,
};
use crate::BuiltinResult;

use super::bins::{self, BinLevels};
use super::error;
use super::options::GroupCountOptions;

pub(super) enum OutputForm {
    Array,
    Table { names: Vec<String> },
}

pub(super) struct GroupCountInput {
    pub(super) columns: Vec<GroupColumn>,
    pub(super) rows: usize,
    pub(super) options: GroupCountOptions,
    pub(super) bins: Option<BinLevels>,
    pub(super) output: OutputForm,
}

impl GroupCountInput {
    pub(super) fn prepare(first: Value, rest: Vec<Value>) -> BuiltinResult<Self> {
        let (positional, options) = GroupCountOptions::split_and_parse(rest)?;
        if matches!(&first, Value::Object(object) if is_tabular_object(object)) {
            table(first, positional, options)
        } else {
            array(first, positional, options)
        }
    }
}

fn array(
    first: Value,
    positional: Vec<Value>,
    options: GroupCountOptions,
) -> BuiltinResult<GroupCountInput> {
    if positional.len() > 1 {
        return Err(error::invalid(
            "groupcounts: array input accepts at most one bin specification",
        ));
    }
    let mut columns = columns_from_value("A", first, true).map_err(error::invalid)?;
    let rows = matching_rows(&columns)?;
    let bins = bins::apply(&mut columns, &positional, options.included_edge)?;
    Ok(GroupCountInput {
        columns,
        rows,
        options,
        bins,
        output: OutputForm::Array,
    })
}

fn table(
    first: Value,
    positional: Vec<Value>,
    options: GroupCountOptions,
) -> BuiltinResult<GroupCountInput> {
    if positional.is_empty() || positional.len() > 2 {
        return Err(error::invalid(
            "groupcounts: table input requires groupvars and accepts one optional groupbins value",
        ));
    }
    let Value::Object(object) = first else {
        unreachable!("table form was checked by the caller")
    };
    let all_names = table_variable_names_from_object(&object)?;
    let names = parse_variable_selector_for_object(positional.first(), &object, &all_names)?;
    let variables = table_variables(&object)?;
    let rows = table_height(&object)?;
    let mut columns = names
        .iter()
        .map(|name| {
            let value =
                variables.fields.get(name).cloned().ok_or_else(|| {
                    error::invalid(format!("groupcounts: unknown variable '{name}'"))
                })?;
            let column = GroupColumn::vector(name.clone(), value).map_err(error::invalid)?;
            if column.rows != rows {
                return Err(error::invalid(format!(
                    "groupcounts: table variable '{name}' has {} rows; expected {rows}",
                    column.rows
                )));
            }
            Ok(column)
        })
        .collect::<BuiltinResult<Vec<_>>>()?;
    let bins = bins::apply(&mut columns, &positional[1..], options.included_edge)?;
    Ok(GroupCountInput {
        columns,
        rows,
        options,
        bins,
        output: OutputForm::Table { names },
    })
}

fn matching_rows(columns: &[GroupColumn]) -> BuiltinResult<usize> {
    let rows = columns
        .first()
        .map(|column| column.rows)
        .ok_or_else(|| error::invalid("groupcounts: expected at least one grouping variable"))?;
    if columns.iter().any(|column| column.rows != rows) {
        return Err(error::invalid(
            "groupcounts: grouping variables must have matching row counts",
        ));
    }
    Ok(rows)
}

use runmat_value::{CellArray, Value};

use super::{categorical, numeric, observations, temporal, text, VariableResult};
use crate::builtins::array::grouping::keys::KeyAtom;

#[derive(Clone)]
pub(crate) struct GroupColumn {
    pub(crate) name: String,
    pub(crate) value: Value,
    pub(crate) rows: usize,
    atoms: Vec<KeyAtom>,
}

impl GroupColumn {
    pub(crate) fn vector(name: impl Into<String>, value: Value) -> VariableResult<Self> {
        let rows = observations::validate_and_count(&value)?;
        let atoms = observations::atoms(&value, rows)?;
        Ok(Self {
            name: name.into(),
            value,
            rows,
            atoms,
        })
    }

    pub(crate) fn atom(&self, row: usize) -> VariableResult<&KeyAtom> {
        self.atoms
            .get(row)
            .ok_or_else(|| "grouping row is out of bounds".to_string())
    }

    pub(crate) fn atoms(&self) -> &[KeyAtom] {
        &self.atoms
    }

    pub(crate) fn replace_value(&mut self, value: Value) -> VariableResult<()> {
        let rows = observations::validate_and_count(&value)?;
        if rows != self.rows {
            return Err(format!(
                "replacement grouping variable has {rows} rows; expected {}",
                self.rows
            ));
        }
        self.atoms = observations::atoms(&value, rows)?;
        self.value = value;
        Ok(())
    }
}

pub(crate) fn columns_from_arguments(
    values: Vec<Value>,
    split_matrices: bool,
) -> VariableResult<Vec<GroupColumn>> {
    let mut columns = Vec::new();
    for (index, value) in values.into_iter().enumerate() {
        columns.extend(columns_from_value(
            &format!("Var{}", index + 1),
            value,
            split_matrices,
        )?);
    }
    Ok(columns)
}

pub(crate) fn columns_from_value(
    base_name: &str,
    value: Value,
    split_matrices: bool,
) -> VariableResult<Vec<GroupColumn>> {
    match value {
        Value::Tensor(value) => numeric::tensor_columns(base_name, value, split_matrices),
        Value::LogicalArray(value) => numeric::logical_columns(base_name, value, split_matrices),
        Value::StringArray(value) => text::string_columns(base_name, value, split_matrices),
        Value::Cell(value) if is_group_vector_list(&value) => {
            let mut columns = Vec::with_capacity(value.data.len());
            for (index, value) in value.data.into_iter().enumerate() {
                columns.extend(columns_from_value(
                    &format!("{base_name}{}", index + 1),
                    value,
                    false,
                )?);
            }
            Ok(columns)
        }
        Value::Cell(value) => Ok(vec![GroupColumn::vector(base_name, Value::Cell(value))?]),
        Value::Object(value)
            if categorical::is_supported_object(&value)
                || temporal::is_supported_object(&value) =>
        {
            Ok(vec![GroupColumn::vector(base_name, Value::Object(value))?])
        }
        value => Ok(vec![GroupColumn::vector(base_name, value)?]),
    }
}

fn is_group_vector_list(value: &CellArray) -> bool {
    !value.data.is_empty()
        && value.data.iter().all(|value| {
            matches!(
                value,
                Value::Tensor(_)
                    | Value::LogicalArray(_)
                    | Value::StringArray(_)
                    | Value::Object(_)
                    | Value::Cell(_)
            )
        })
}

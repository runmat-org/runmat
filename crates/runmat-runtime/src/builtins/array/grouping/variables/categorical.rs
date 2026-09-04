use runmat_value::{ObjectInstance, Value};

use super::VariableResult;
use crate::builtins::array::grouping::keys::KeyAtom;

pub(super) fn is_supported_object(value: &ObjectInstance) -> bool {
    value.is_class(runmat_types::standard::CATEGORICAL)
}

pub(super) fn atoms(value: &Value, rows: usize) -> VariableResult<Vec<KeyAtom>> {
    let Value::Object(object) = value else {
        return Err("categorical grouping storage is not an object".into());
    };
    let labels = crate::builtins::table::categorical_observation_labels(object)
        .map_err(|error| error.to_string())?;
    if labels.len() != rows {
        return Err(format!(
            "categorical grouping storage has {} observations; expected {rows}",
            labels.len()
        ));
    }
    Ok(labels
        .into_iter()
        .map(|label| label.map(KeyAtom::Text).unwrap_or(KeyAtom::Missing))
        .collect())
}

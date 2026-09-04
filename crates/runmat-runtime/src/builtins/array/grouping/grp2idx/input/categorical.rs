use runmat_value::{ObjectInstance, Value};

use crate::builtins::table::{categorical_categories, categorical_observation_labels};
use crate::BuiltinResult;

use super::{GroupingInput, KeyAtom, KeyOrder};

pub(super) fn prepare(value: ObjectInstance) -> BuiltinResult<GroupingInput> {
    let categories = categorical_categories(&value)?;
    let rows = categorical_observation_labels(&value)?
        .into_iter()
        .map(|label| label.map(|label| vec![KeyAtom::Text(label)]))
        .collect();
    let order = KeyOrder::Explicit(
        categories
            .iter()
            .map(|label| vec![KeyAtom::Text(label.clone())])
            .collect(),
    );
    let mut input = GroupingInput::new(Value::Object(value), rows, order);
    input.categorical = true;
    Ok(input)
}

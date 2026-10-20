use super::{assign_into_value, assign_with_selector, errors};
use crate::builtins::structs::core::field_path::FieldStep;
use crate::execution::RootedValue;
use crate::BuiltinResult;
use runmat_value::{StructValue, Value};

#[async_recursion::async_recursion(?Send)]
pub(super) async fn assign_into_struct(
    runtime: Option<&crate::context::RuntimeContext>,
    struct_value: StructValue,
    steps: &[FieldStep],
    rhs: Value,
) -> BuiltinResult<RootedValue> {
    let _aggregate_root =
        super::super::super::construction::input(runtime, &Value::Struct(struct_value.clone()))?;
    let _rhs_root = super::super::super::construction::input(runtime, &rhs)?;
    let (first, rest) = steps
        .split_first()
        .expect("assignment path is validated before traversal");
    if rest.is_empty() {
        let updated = if let Some(selector) = &first.index {
            let current = struct_value
                .fields
                .get(&first.name)
                .cloned()
                .ok_or_else(|| missing_field(&first.name))?;
            let updated = assign_with_selector(runtime, current, selector, rest, rhs).await?;
            return replace_field(runtime, &struct_value, &first.name, updated.value());
        } else {
            rhs
        };
        return replace_field(runtime, &struct_value, &first.name, &updated);
    }
    if let Some(selector) = &first.index {
        let current = struct_value
            .fields
            .get(&first.name)
            .cloned()
            .ok_or_else(|| missing_field(&first.name))?;
        let updated = assign_with_selector(runtime, current, selector, rest, rhs).await?;
        return replace_field(runtime, &struct_value, &first.name, updated.value());
    }
    let current = struct_value
        .fields
        .get(&first.name)
        .cloned()
        .unwrap_or_else(|| Value::Struct(StructValue::new()));
    let updated = assign_into_value(runtime, current, rest, rhs).await?;
    replace_field(runtime, &struct_value, &first.name, updated.value())
}

fn missing_field(name: &str) -> crate::RuntimeError {
    errors::with_message(
        format!("Reference to non-existent field '{name}'."),
        &runmat_builtins::SETFIELD_ERROR_MISSING_FIELD,
    )
}

fn replace_field(
    runtime: Option<&crate::context::RuntimeContext>,
    structure: &StructValue,
    name: &str,
    value: &Value,
) -> BuiltinResult<RootedValue> {
    let fields = structure
        .fields
        .iter()
        .map(|(field, previous)| (field, if field == name { value } else { previous }));
    let appended = (!structure.fields.contains_key(name)).then_some((name, value));
    super::super::super::construction::scalar(runtime, fields.chain(appended))
}

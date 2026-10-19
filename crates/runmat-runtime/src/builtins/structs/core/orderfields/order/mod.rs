mod empty;
mod names;
mod permutation;
mod reference;
mod validation;

use runmat_value::Value;

pub(super) struct FieldOrder(Vec<String>);

impl FieldOrder {
    pub(super) fn as_slice(&self) -> &[String] {
        self.0.as_slice()
    }
}

pub(super) fn resolve(
    current: &[String],
    argument: Option<&Value>,
) -> crate::BuiltinResult<FieldOrder> {
    let Some(argument) = argument else {
        let mut names = current.to_vec();
        names.sort();
        return Ok(FieldOrder(names));
    };
    if current.is_empty() {
        return empty::resolve(argument);
    }
    if let Some(names) = reference::parse(argument)? {
        validation::exact_field_set(current, names.as_slice())?;
        return Ok(FieldOrder(names));
    }
    if let Some(names) = names::parse(argument)? {
        validation::exact_field_set(current, names.as_slice())?;
        return Ok(FieldOrder(names));
    }
    if let Some(names) = permutation::parse(current, argument)? {
        return Ok(FieldOrder(names));
    }
    Err(super::error::invalid_order())
}

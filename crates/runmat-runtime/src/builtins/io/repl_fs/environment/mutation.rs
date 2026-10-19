use std::panic::{self, AssertUnwindSafe};

use super::values::EnvironmentValue;

const MAX_NAME_CHARACTERS: usize = 32_766;

#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct EnvironmentMutation {
    name: String,
    value: EnvironmentValue,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) enum MutationError {
    EmptyName,
    NameContainsEquals,
    NameContainsNull,
    NameTooLong,
    ValueContainsNull,
    Host(String),
}

pub(super) fn plan(
    names: &[String],
    values: Vec<EnvironmentValue>,
) -> Result<Vec<EnvironmentMutation>, MutationError> {
    names
        .iter()
        .cloned()
        .zip(values)
        .map(|(name, value)| {
            validate(&name, &value)?;
            Ok(EnvironmentMutation { name, value })
        })
        .collect()
}

pub(super) fn removals(names: &[String]) -> Result<Vec<EnvironmentMutation>, MutationError> {
    plan(names, vec![EnvironmentValue::Remove; names.len()])
}

fn validate(name: &str, value: &EnvironmentValue) -> Result<(), MutationError> {
    if name.is_empty() {
        return Err(MutationError::EmptyName);
    }
    if name.contains('=') {
        return Err(MutationError::NameContainsEquals);
    }
    if name.contains('\0') {
        return Err(MutationError::NameContainsNull);
    }
    if name.chars().count() > MAX_NAME_CHARACTERS {
        return Err(MutationError::NameTooLong);
    }
    if matches!(value, EnvironmentValue::Set(value) if value.contains('\0')) {
        return Err(MutationError::ValueContainsNull);
    }
    Ok(())
}

pub(super) fn apply(mutations: &[EnvironmentMutation]) -> Result<(), MutationError> {
    for mutation in mutations {
        let result = panic::catch_unwind(AssertUnwindSafe(|| match &mutation.value {
            EnvironmentValue::Set(value) => set(&mutation.name, value),
            EnvironmentValue::Remove => crate::builtins::common::env::remove_var(&mutation.name),
        }));
        if let Err(payload) = result {
            return Err(MutationError::Host(panic_message(payload)));
        }
    }
    Ok(())
}

fn set(name: &str, value: &str) {
    #[cfg(target_os = "windows")]
    if value.is_empty() {
        crate::builtins::common::env::remove_var(name);
        return;
    }
    crate::builtins::common::env::set_var(name, value);
}

fn panic_message(payload: Box<dyn std::any::Any + Send>) -> String {
    match payload.downcast::<String>() {
        Ok(message) => *message,
        Err(payload) => match payload.downcast::<&'static str>() {
            Ok(message) => (*message).to_string(),
            Err(_) => "environment operation failed".to_string(),
        },
    }
}

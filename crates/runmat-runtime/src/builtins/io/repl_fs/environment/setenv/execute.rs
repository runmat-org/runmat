use runmat_builtins::{
    SETENV_ERROR_ARITY, SETENV_ERROR_DICTIONARY, SETENV_ERROR_NAME, SETENV_ERROR_SHAPE,
    SETENV_ERROR_TOO_MANY_OUTPUTS, SETENV_ERROR_VALUE, SETENV_STATUS_OUTPUT_EXTENSION,
};
use runmat_value::Value;

use super::super::{dictionary, error, mutation, names::*, values::*};
use super::result::SetenvResult;

const IDENTITY: &str = "setenv";

pub(super) fn run(args: Vec<Value>) -> crate::BuiltinResult<Value> {
    let requested = crate::output_count::current_output_count();
    if requested.is_some_and(|count| count > 2) {
        return Err(error::builtin(IDENTITY, &SETENV_ERROR_TOO_MANY_OUTPUTS));
    }
    if requested.is_some_and(|count| count > 0) {
        crate::compatibility::ensure_builtin_extension_enabled(
            &SETENV_STATUS_OUTPUT_EXTENSION,
            IDENTITY,
        )?;
    }
    Ok(evaluate(&args)?.render(requested))
}

pub(super) fn evaluate(args: &[Value]) -> crate::BuiltinResult<SetenvResult> {
    let prepared = match args {
        [Value::Object(object)] if object.is_class(runmat_types::standard::DICTIONARY) => {
            dictionary_plan(object)?
        }
        [name] => names_plan(name, &EnvironmentValues::empty_scalar())?,
        [name, value] => names_plan(
            name,
            &EnvironmentValues::decode(value).map_err(value_error)?,
        )?,
        _ => return Err(error::builtin(IDENTITY, &SETENV_ERROR_ARITY)),
    };
    let plan = match prepared {
        Prepared::Ready(plan) => plan,
        Prepared::Rejected(failure) => return Ok(SetenvResult::failure(mutation_message(failure))),
    };
    Ok(match mutation::apply(&plan) {
        Ok(()) => SetenvResult::success(),
        Err(error) => SetenvResult::failure(mutation_message(error)),
    })
}

enum Prepared {
    Ready(Vec<mutation::EnvironmentMutation>),
    Rejected(mutation::MutationError),
}

fn names_plan(name: &Value, values: &EnvironmentValues) -> crate::BuiltinResult<Prepared> {
    let names = EnvironmentNames::decode(name).map_err(name_error)?;
    if names.features().next().is_some() {
        return Err(error::builtin(IDENTITY, &SETENV_ERROR_NAME));
    }
    let values = values
        .align(names.len(), names.shape().as_deref())
        .map_err(|()| error::builtin(IDENTITY, &SETENV_ERROR_SHAPE))?;
    Ok(match mutation::plan(names.names(), values) {
        Ok(plan) => Prepared::Ready(plan),
        Err(failure) => Prepared::Rejected(failure),
    })
}

fn dictionary_plan(object: &runmat_value::ObjectInstance) -> crate::BuiltinResult<Prepared> {
    let pairs = dictionary::entries(object)
        .map_err(|()| error::builtin(IDENTITY, &SETENV_ERROR_DICTIONARY))?;
    let mut names = Vec::with_capacity(pairs.len());
    let mut values = Vec::with_capacity(pairs.len());
    for (name, value) in pairs {
        let decoded_name = EnvironmentNames::decode(&name).map_err(name_error)?;
        if decoded_name.len() != 1 || decoded_name.features().next().is_some() {
            return Err(error::builtin(IDENTITY, &SETENV_ERROR_NAME));
        }
        names.push(decoded_name.names()[0].clone());
        values.push(
            EnvironmentValues::decode(&value)
                .map_err(value_error)?
                .scalar_value()
                .ok_or_else(|| error::builtin(IDENTITY, &SETENV_ERROR_VALUE))?,
        );
    }
    Ok(match mutation::plan(&names, values) {
        Ok(plan) => Prepared::Ready(plan),
        Err(failure) => Prepared::Rejected(failure),
    })
}

fn name_error(_failure: NameError) -> crate::RuntimeError {
    error::builtin(IDENTITY, &SETENV_ERROR_NAME)
}

fn value_error(_failure: ValueError) -> crate::RuntimeError {
    error::builtin(IDENTITY, &SETENV_ERROR_VALUE)
}

fn mutation_message(failure: mutation::MutationError) -> String {
    match failure {
        mutation::MutationError::EmptyName => "Environment variable name must not be empty.".into(),
        mutation::MutationError::NameContainsEquals => {
            "Environment variable names must not contain '='.".into()
        }
        mutation::MutationError::NameContainsNull => {
            "Environment variable names must not contain null characters.".into()
        }
        mutation::MutationError::NameTooLong => {
            "Environment variable name exceeds the supported length.".into()
        }
        mutation::MutationError::ValueContainsNull => {
            "Environment variable values must not contain null characters.".into()
        }
        mutation::MutationError::Host(message) => {
            format!("Unable to update environment variable: {message}")
        }
    }
}

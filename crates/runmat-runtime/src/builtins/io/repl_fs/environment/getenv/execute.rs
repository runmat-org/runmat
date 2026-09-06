use runmat_builtins::{
    GETENV_CELL_STRING_EXTENSION, GETENV_CHAR_MATRIX_EXTENSION, GETENV_ERROR_CELL_ELEMENT,
    GETENV_ERROR_INTERNAL, GETENV_ERROR_INVALID_NAME, GETENV_ERROR_TOO_MANY_INPUTS,
};
use runmat_value::Value;

use super::super::{dictionary, error, names::*};

const IDENTITY: &str = "getenv";

pub(super) fn run(args: Vec<Value>) -> crate::BuiltinResult<Value> {
    match args.as_slice() {
        [] => all(),
        [value] => one(value),
        _ => Err(error::builtin(IDENTITY, &GETENV_ERROR_TOO_MANY_INPUTS)),
    }
}

fn all() -> crate::BuiltinResult<Value> {
    let mut entries = crate::builtins::common::env::vars();
    entries.sort_unstable_by(|left, right| left.0.cmp(&right.0));
    dictionary::from_pairs(entries).map_err(|detail| {
        error::message(
            IDENTITY,
            &GETENV_ERROR_INTERNAL,
            format!("{}: {detail}", GETENV_ERROR_INTERNAL.message),
        )
    })
}

fn one(value: &Value) -> crate::BuiltinResult<Value> {
    let names = EnvironmentNames::decode(value).map_err(|kind| match kind {
        NameError::InvalidType => error::builtin(IDENTITY, &GETENV_ERROR_INVALID_NAME),
        NameError::InvalidCellElement => error::builtin(IDENTITY, &GETENV_ERROR_CELL_ELEMENT),
    })?;
    for feature in names.features() {
        let extension = match feature {
            NameFeature::CharacterMatrix => &GETENV_CHAR_MATRIX_EXTENSION,
            NameFeature::StringInCell => &GETENV_CELL_STRING_EXTENSION,
        };
        crate::compatibility::ensure_builtin_extension_enabled(extension, IDENTITY)?;
    }
    let values = names
        .names()
        .iter()
        .map(|name| crate::builtins::common::env::var(name).unwrap_or_default())
        .collect();
    names.text_output(values).map_err(|detail| {
        error::message(
            IDENTITY,
            &GETENV_ERROR_INTERNAL,
            format!("{}: {detail}", GETENV_ERROR_INTERNAL.message),
        )
    })
}

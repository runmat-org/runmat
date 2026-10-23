use runmat_builtins::{
    GETENV_CELL_STRING_EXTENSION, GETENV_CHAR_MATRIX_EXTENSION, GETENV_ERROR_CELL_ELEMENT,
    GETENV_ERROR_INTERNAL, GETENV_ERROR_INVALID_NAME, GETENV_ERROR_TOO_MANY_INPUTS,
};
use runmat_value::Value;

use super::super::{dictionary, error, names::*};

const IDENTITY: &str = "getenv";

pub(super) fn run(
    runtime: Option<&crate::context::RuntimeContext>,
    args: crate::call::BuiltinArguments,
) -> crate::BuiltinResult<crate::execution::RootedValue> {
    crate::output_count::validate_fixed_outputs(
        &runmat_builtins::GETENV_DESCRIPTOR,
        &runmat_builtins::GETENV_ERROR_TOO_MANY_OUTPUTS,
        IDENTITY,
    )?;
    match args.len() {
        0 => all(runtime),
        1 => crate::execution::RootedValue::new(one(args.first().expect("one getenv argument"))?)
            .map_err(|source| source.into_runtime_error()),
        _ => Err(error::builtin(IDENTITY, &GETENV_ERROR_TOO_MANY_INPUTS)),
    }
}

fn all(
    runtime: Option<&crate::context::RuntimeContext>,
) -> crate::BuiltinResult<crate::execution::RootedValue> {
    let mut entries = crate::builtins::common::env::try_vars().map_err(|detail| {
        error::message(
            IDENTITY,
            &GETENV_ERROR_INTERNAL,
            format!("{}: {detail}", GETENV_ERROR_INTERNAL.message),
        )
    })?;
    entries.sort_unstable_by(|left, right| left.0.cmp(&right.0));
    dictionary::from_pairs(runtime, entries).map_err(|detail| {
        error::message(
            IDENTITY,
            &GETENV_ERROR_INTERNAL,
            format!("{}: {}", GETENV_ERROR_INTERNAL.message, detail.message()),
        )
    })
}

fn one(value: &Value) -> crate::BuiltinResult<Value> {
    let names = EnvironmentNames::decode(value).map_err(|kind| match kind {
        NameError::InvalidType => error::builtin(IDENTITY, &GETENV_ERROR_INVALID_NAME),
        NameError::InvalidCellElement => error::builtin(IDENTITY, &GETENV_ERROR_CELL_ELEMENT),
        NameError::Encoding(source) => {
            let mut error = error::builtin(IDENTITY, &GETENV_ERROR_INVALID_NAME);
            error.source = Some(Box::new(source));
            error
        }
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

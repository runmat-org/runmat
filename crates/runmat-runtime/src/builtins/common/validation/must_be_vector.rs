//! Identity-owned mustBeVector admission and execution.
use super::{BuiltinResult, Value};
use runmat_builtins::{MUST_BE_VECTOR_CATALOG_ENTRY, MUST_BE_VECTOR_ERROR_OUTPUTS};
#[runmat_macros::runtime_builtin(
    name = "mustBeVector",
    binding_variant = "default",
    builtin_path = "crate::builtins::common::validation::must_be_vector"
)]
fn builtin(args: crate::call::BuiltinArguments) -> BuiltinResult<Value> {
    evaluate(args)
}

pub(super) fn evaluate(args: crate::call::BuiltinArguments) -> BuiltinResult<Value> {
    super::admission::inputs(
        &MUST_BE_VECTOR_CATALOG_ENTRY,
        &MUST_BE_VECTOR_ERROR_OUTPUTS,
        args.view(),
    )?;
    let allow_all_empties = match args.get(1) {
        None => false,
        Some(option)
            if super::text_scalar_arg(MUST_BE_VECTOR_CATALOG_ENTRY.identity.name, option)?
                .eq_ignore_ascii_case("allow-all-empties") =>
        {
            true
        }
        Some(_) => {
            return Err(super::invalid_argument_error(
                MUST_BE_VECTOR_CATALOG_ENTRY.identity.name,
                "option must be 'allow-all-empties'",
            ))
        }
    };
    super::validate_resident_metadata(&args[0])?;
    super::check_validator(
        MUST_BE_VECTOR_CATALOG_ENTRY.identity.name,
        super::value_satisfies_vector_validator(&args[0], allow_all_empties)?,
    )
}

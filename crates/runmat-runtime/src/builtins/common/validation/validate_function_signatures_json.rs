//! Identity-owned validateFunctionSignaturesJSON admission and execution.
use super::{BuiltinResult, Value};
use runmat_builtins::{
    VALIDATE_FUNCTION_SIGNATURES_JSON_CATALOG_ENTRY,
    VALIDATE_FUNCTION_SIGNATURES_JSON_ERROR_OUTPUTS,
};
#[runmat_macros::runtime_builtin(
    name = "validateFunctionSignaturesJSON",
    binding_variant = "default",
    builtin_path = "crate::builtins::common::validation::validate_function_signatures_json"
)]
fn builtin(args: crate::call::BuiltinArguments) -> BuiltinResult<Value> {
    evaluate(args)
}

pub(super) fn evaluate(args: crate::call::BuiltinArguments) -> BuiltinResult<Value> {
    super::admission::inputs(
        &VALIDATE_FUNCTION_SIGNATURES_JSON_CATALOG_ENTRY,
        &VALIDATE_FUNCTION_SIGNATURES_JSON_ERROR_OUTPUTS,
        args.view(),
    )?;
    super::admission::reject_resident_text(
        &VALIDATE_FUNCTION_SIGNATURES_JSON_CATALOG_ENTRY,
        &args[0],
    )?;
    crate::compatibility::ensure_builtin_extension_enabled(
        &runmat_builtins::VALIDATE_FUNCTION_SIGNATURES_JSON_EXTENSION,
        VALIDATE_FUNCTION_SIGNATURES_JSON_CATALOG_ENTRY
            .identity
            .name,
    )?;
    super::validate_function_signatures_json(&args[0])?;
    super::pass()
}

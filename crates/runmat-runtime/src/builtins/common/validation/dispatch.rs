//! Retained legacy callable routing pending identity-owned cutover.

use super::{
    check_validator, ensure_resident_extension, invalid_argument_error, range_inclusivity_arg,
    require_arg_count, require_args, require_exact_arg_count, validate_resident_metadata,
    value_is_finite_async, value_is_greater_than_or_equal_values_async,
    value_is_greater_than_values_async, value_is_in_range_documented_async, value_is_integer_async,
    value_is_less_than_or_equal_values_async, value_is_less_than_values_async,
    value_is_member_async, value_is_negative_async, value_is_non_nan_async,
    value_is_nonmissing_async, value_is_nonnegative_async, value_is_nonpositive_async,
    value_is_nonzero_async, value_is_positive_async, value_is_real_async, value_texts,
    BuiltinResult, Path, Value,
};

pub fn dispatch_validator(builtin: &str, args: Vec<Value>) -> BuiltinResult<Value> {
    futures::executor::block_on(dispatch_validator_async(builtin, args))
}

pub async fn dispatch_validator_async(builtin: &str, args: Vec<Value>) -> BuiltinResult<Value> {
    // Retained public string API forwards migrated identities to their sole domain owner.
    match builtin {
        "mustBeA" => return super::must_be_a::evaluate(args),
        "mustBeColumn" => return super::must_be_column::evaluate(args),
        "mustBeFloat" => return super::must_be_float::evaluate(args),
        "mustBeNonempty" => return super::must_be_nonempty::evaluate(args),
        "mustBeNonsparse" => return super::must_be_nonsparse::evaluate(args),
        "mustBeNumeric" => return super::must_be_numeric::evaluate(args),
        "mustBeNumericOrLogical" => return super::must_be_numeric_or_logical::evaluate(args),
        "mustBeScalarOrEmpty" => return super::must_be_scalar_or_empty::evaluate(args),
        "mustBeSparse" => return super::must_be_sparse::evaluate(args),
        "mustBeText" => return super::must_be_text::evaluate(args),
        "mustBeTextScalar" => return super::must_be_text_scalar::evaluate(args),
        "mustBeNonzeroLengthText" => return super::must_be_nonzero_length_text::evaluate(args),
        "mustBeUnderlyingType" => return super::must_be_underlying_type::evaluate(args),
        "mustBeValidVariableName" => return super::must_be_valid_variable_name::evaluate(args),
        "mustBeVector" => return super::must_be_vector::evaluate(args),
        "validateFunctionSignaturesJSON" => {
            return super::validate_function_signatures_json::evaluate(args)
        }
        _ => {}
    }
    let value = require_args(builtin, &args, 1, usize::MAX)?;
    validate_resident_metadata(value)?;
    match builtin {
        "mustBeFile" => check_validator(builtin, {
            require_exact_arg_count(builtin, &args, 1)?;
            value_texts(value)?.iter().all(|p| Path::new(p).is_file())
        }),
        "mustBeFinite" => {
            require_exact_arg_count(builtin, &args, 1)?;
            ensure_resident_extension(value, builtin)?;
            check_validator(builtin, value_is_finite_async(value).await?)
        }
        "mustBeFolder" => check_validator(builtin, {
            require_exact_arg_count(builtin, &args, 1)?;
            value_texts(value)?.iter().all(|p| Path::new(p).is_dir())
        }),
        "mustBeGreaterThan" => {
            require_exact_arg_count(builtin, &args, 2)?;
            check_validator(
                builtin,
                value_is_greater_than_values_async(value, &args[1]).await?,
            )
        }
        "mustBeGreaterThanOrEqual" => {
            require_exact_arg_count(builtin, &args, 2)?;
            check_validator(
                builtin,
                value_is_greater_than_or_equal_values_async(value, &args[1]).await?,
            )
        }
        "mustBeInRange" => {
            require_arg_count(builtin, &args, 3, 5)?;
            let inclusivity = range_inclusivity_arg(builtin, &args[3..])?;
            check_validator(
                builtin,
                value_is_in_range_documented_async(value, &args[1], &args[2], inclusivity).await?,
            )
        }
        "mustBeInteger" => {
            require_exact_arg_count(builtin, &args, 1)?;
            ensure_resident_extension(value, builtin)?;
            check_validator(builtin, value_is_integer_async(value).await?)
        }
        "mustBeLessThan" => {
            require_exact_arg_count(builtin, &args, 2)?;
            check_validator(
                builtin,
                value_is_less_than_values_async(value, &args[1]).await?,
            )
        }
        "mustBeLessThanOrEqual" => {
            require_exact_arg_count(builtin, &args, 2)?;
            check_validator(
                builtin,
                value_is_less_than_or_equal_values_async(value, &args[1]).await?,
            )
        }
        "mustBeMember" => {
            require_exact_arg_count(builtin, &args, 2)?;
            check_validator(builtin, value_is_member_async(value, &args[1]).await?)
        }
        "mustBeNegative" => {
            require_exact_arg_count(builtin, &args, 1)?;
            check_validator(builtin, value_is_negative_async(value).await?)
        }
        "mustBeNonmissing" => {
            require_exact_arg_count(builtin, &args, 1)?;
            check_validator(builtin, value_is_nonmissing_async(value).await?)
        }
        "mustBeNonNan" => {
            require_exact_arg_count(builtin, &args, 1)?;
            ensure_resident_extension(value, builtin)?;
            check_validator(builtin, value_is_non_nan_async(value).await?)
        }
        "mustBeNonnegative" => {
            require_exact_arg_count(builtin, &args, 1)?;
            check_validator(builtin, value_is_nonnegative_async(value).await?)
        }
        "mustBeNonpositive" => {
            require_exact_arg_count(builtin, &args, 1)?;
            check_validator(builtin, value_is_nonpositive_async(value).await?)
        }
        "mustBeNonzero" => {
            require_exact_arg_count(builtin, &args, 1)?;
            ensure_resident_extension(value, builtin)?;
            check_validator(builtin, value_is_nonzero_async(value).await?)
        }
        "mustBePositive" => {
            require_exact_arg_count(builtin, &args, 1)?;
            check_validator(builtin, value_is_positive_async(value).await?)
        }
        "mustBeReal" => {
            require_exact_arg_count(builtin, &args, 1)?;
            check_validator(builtin, value_is_real_async(value).await?)
        }
        _ => Err(invalid_argument_error(builtin, "unknown validator").into()),
    }
}

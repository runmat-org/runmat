//! Retained legacy callable routing pending identity-owned cutover.

use super::{
    build_runtime_error, check_validator, ensure_resident_extension, invalid_argument_error,
    is_valid_varname, must_be_a, pass, range_inclusivity_arg, require_arg_count, require_args,
    require_exact_arg_count, text_scalar_arg, type_names_arg, validate_function_signatures_json,
    validate_resident_metadata, value_is_column, value_is_empty, value_is_finite_async,
    value_is_float, value_is_greater_than_or_equal_values_async,
    value_is_greater_than_values_async, value_is_in_range_documented_async, value_is_integer_async,
    value_is_less_than_or_equal_values_async, value_is_less_than_values_async,
    value_is_member_async, value_is_negative_async, value_is_non_nan_async,
    value_is_nonmissing_async, value_is_nonnegative_async, value_is_nonpositive_async,
    value_is_nonzero_async, value_is_nonzero_length_text, value_is_numeric,
    value_is_numeric_or_logical, value_is_positive_async, value_is_real_async,
    value_is_scalar_or_empty, value_is_text, value_is_text_scalar,
    value_satisfies_vector_validator, value_texts, value_underlying_type_matches, BuiltinResult,
    Path, SparseTensor, Value,
};

pub fn dispatch_validator(builtin: &str, args: Vec<Value>) -> BuiltinResult<Value> {
    futures::executor::block_on(dispatch_validator_async(builtin, args))
}

pub async fn dispatch_validator_async(builtin: &str, args: Vec<Value>) -> BuiltinResult<Value> {
    let value = require_args(builtin, &args, 1, usize::MAX)?;
    if matches!(value, Value::GpuTensor(_))
        && matches!(
            builtin,
            "mustBeText"
                | "mustBeTextScalar"
                | "mustBeValidVariableName"
                | "validateFunctionSignaturesJSON"
        )
    {
        require_exact_arg_count(builtin, &args, 1)?;
        return Err(
            build_runtime_error(format!("{builtin}: value does not satisfy validator"))
                .with_builtin(builtin)
                .with_identifier(format!("RunMat:{builtin}:ValidationFailed"))
                .with_gpu_gather_retry(crate::GpuGatherRetry::Never)
                .build()
                .into(),
        );
    }
    validate_resident_metadata(value)?;
    match builtin {
        "mustBeA" => {
            require_exact_arg_count(builtin, &args, 2)?;
            check_validator(builtin, must_be_a(value, type_names_arg(&args, 1)?)?)
        }
        "mustBeColumn" => {
            require_exact_arg_count(builtin, &args, 1)?;
            check_validator(builtin, value_is_column(value))
        }
        "mustBeFile" => check_validator(builtin, {
            require_exact_arg_count(builtin, &args, 1)?;
            value_texts(value)?.iter().all(|p| Path::new(p).is_file())
        }),
        "mustBeFinite" => {
            require_exact_arg_count(builtin, &args, 1)?;
            ensure_resident_extension(value, builtin)?;
            check_validator(builtin, value_is_finite_async(value).await?)
        }
        "mustBeFloat" => {
            require_exact_arg_count(builtin, &args, 1)?;
            check_validator(builtin, value_is_float(value))
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
        "mustBeNonempty" => {
            require_exact_arg_count(builtin, &args, 1)?;
            check_validator(builtin, !value_is_empty(value))
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
        "mustBeNonsparse" => {
            require_exact_arg_count(builtin, &args, 1)?;
            check_validator(builtin, !matches!(value, Value::SparseTensor(_)))
        }
        "mustBeNonzero" => {
            require_exact_arg_count(builtin, &args, 1)?;
            ensure_resident_extension(value, builtin)?;
            check_validator(builtin, value_is_nonzero_async(value).await?)
        }
        "mustBeNonzeroLengthText" => {
            require_exact_arg_count(builtin, &args, 1)?;
            check_validator(builtin, value_is_nonzero_length_text(value))
        }
        "mustBeNumeric" => {
            require_exact_arg_count(builtin, &args, 1)?;
            check_validator(builtin, value_is_numeric(value))
        }
        "mustBeNumericOrLogical" => {
            require_exact_arg_count(builtin, &args, 1)?;
            check_validator(builtin, value_is_numeric_or_logical(value))
        }
        "mustBePositive" => {
            require_exact_arg_count(builtin, &args, 1)?;
            check_validator(builtin, value_is_positive_async(value).await?)
        }
        "mustBeReal" => {
            require_exact_arg_count(builtin, &args, 1)?;
            check_validator(builtin, value_is_real_async(value).await?)
        }
        "mustBeScalarOrEmpty" => {
            require_exact_arg_count(builtin, &args, 1)?;
            check_validator(builtin, value_is_scalar_or_empty(value))
        }
        "mustBeSparse" => {
            require_exact_arg_count(builtin, &args, 1)?;
            check_validator(
                builtin,
                value_is_empty(value) || matches!(value, Value::SparseTensor(_)),
            )
        }
        "mustBeText" => {
            require_exact_arg_count(builtin, &args, 1)?;
            check_validator(builtin, value_is_text(value))
        }
        "mustBeTextScalar" => {
            require_exact_arg_count(builtin, &args, 1)?;
            check_validator(builtin, value_is_text_scalar(value))
        }
        "mustBeUnderlyingType" => {
            require_exact_arg_count(builtin, &args, 2)?;
            check_validator(
                builtin,
                value_underlying_type_matches(value, type_names_arg(&args, 1)?)?,
            )
        }
        "mustBeValidVariableName" => {
            require_exact_arg_count(builtin, &args, 1)?;
            check_validator(
                builtin,
                value_texts(value)?
                    .iter()
                    .all(|name| is_valid_varname(name)),
            )
        }
        "mustBeVector" => {
            require_arg_count(builtin, &args, 1, 2)?;
            let allow_all_empties = match args.get(1) {
                None => false,
                Some(flag)
                    if text_scalar_arg(builtin, flag)?
                        .eq_ignore_ascii_case("allow-all-empties") =>
                {
                    true
                }
                Some(_) => {
                    return Err(invalid_argument_error(
                        builtin,
                        "option must be 'allow-all-empties'",
                    )
                    .into())
                }
            };
            check_validator(
                builtin,
                value_satisfies_vector_validator(value, allow_all_empties)?,
            )
        }
        "validateFunctionSignaturesJSON" => {
            require_exact_arg_count(builtin, &args, 1)?;
            validate_function_signatures_json(value)?;
            pass()
        }
        _ => Err(invalid_argument_error(builtin, "unknown validator").into()),
    }
}

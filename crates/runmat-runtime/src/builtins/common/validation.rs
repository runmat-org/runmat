//! Shared MATLAB argument-validation helpers and callable `mustBe*` builtins.

use runmat_builtins::{
    BuiltinExtensionDescriptor, BuiltinExtensionMode, BuiltinIntegerAuditDescriptor,
    BuiltinIntegerAuditKind, BuiltinIntegerBackendRule, BuiltinIntegerCapabilityDescriptor,
    BuiltinIntegerComputationDomain, BuiltinIntegerInputAvailability,
    BuiltinIntegerInputCapability, BuiltinIntegerOutputClassRule, BuiltinIntegerOverflowRule,
    BuiltinIntegerOverloadKind, BuiltinIntegerScalarDoubleRule,
};
use std::cmp::Ordering;
use std::path::Path;

use runmat_accelerate_api::{
    handle_integer_type, handle_is_logical, handle_storage, GpuTensorStorage,
};
use runmat_builtins::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor, BuiltinOutputMode,
    BuiltinParamArity, BuiltinParamDescriptor, BuiltinParamType, BuiltinSignatureDescriptor,
};
use runmat_macros::runtime_builtin;
use runmat_value::{
    CellArray, CharArray, ComplexTensor, IntValue, IntegerStorage, NumericDType, SparseTensor,
    Value,
};

use crate::builtins::common::identifiers::is_valid_varname;
use crate::builtins::common::tensor;
use crate::builtins::introspection::class::class_name_for_value;
use crate::builtins::logical::rel::integer_comparison::integer_f64_order;
use crate::builtins::logical::rel::integer_comparison::{
    try_complex_ordering_comparison, try_real_ordering_comparison, IntegerComparisonError,
    IntegerComparisonOp,
};
use crate::{build_runtime_error, BuiltinResult, RuntimeError};

// Preserve the shared validation API while callable identities migrate independently.
mod types;
pub use types::{RangeInclusivity, ValidationAtom};
mod contracts;
pub use contracts::{
    ISVARNAME_DESCRIPTOR, ISVARNAME_INTEGER_AUDIT, MUST_BE_A_INTEGER_CAPABILITIES,
    MUST_BE_COLUMN_INTEGER_CAPABILITIES, MUST_BE_FILE_INTEGER_AUDIT, MUST_BE_FINITE_EXTENSIONS,
    MUST_BE_FINITE_INTEGER_CAPABILITIES, MUST_BE_FLOAT_INTEGER_CAPABILITIES,
    MUST_BE_FOLDER_INTEGER_AUDIT, MUST_BE_GREATER_THAN_INTEGER_CAPABILITIES,
    MUST_BE_GREATER_THAN_OR_EQUAL_INTEGER_CAPABILITIES, MUST_BE_INTEGER_EXTENSIONS,
    MUST_BE_INTEGER_INTEGER_CAPABILITIES, MUST_BE_IN_RANGE_INTEGER_CAPABILITIES,
    MUST_BE_LESS_THAN_INTEGER_CAPABILITIES, MUST_BE_LESS_THAN_OR_EQUAL_INTEGER_CAPABILITIES,
    MUST_BE_MEMBER_INTEGER_CAPABILITIES, MUST_BE_NEGATIVE_INTEGER_CAPABILITIES,
    MUST_BE_NONEMPTY_INTEGER_CAPABILITIES, MUST_BE_NONMISSING_INTEGER_CAPABILITIES,
    MUST_BE_NONNEGATIVE_INTEGER_CAPABILITIES, MUST_BE_NONPOSITIVE_INTEGER_CAPABILITIES,
    MUST_BE_NONSPARSE_INTEGER_CAPABILITIES, MUST_BE_NONZERO_EXTENSIONS,
    MUST_BE_NONZERO_INTEGER_CAPABILITIES, MUST_BE_NONZERO_LENGTH_TEXT_INTEGER_AUDIT,
    MUST_BE_NON_NAN_EXTENSIONS, MUST_BE_NON_NAN_INTEGER_CAPABILITIES,
    MUST_BE_NUMERIC_INTEGER_CAPABILITIES, MUST_BE_NUMERIC_OR_LOGICAL_INTEGER_CAPABILITIES,
    MUST_BE_POSITIVE_INTEGER_CAPABILITIES, MUST_BE_REAL_INTEGER_CAPABILITIES,
    MUST_BE_SCALAR_OR_EMPTY_INTEGER_CAPABILITIES, MUST_BE_SPARSE_INTEGER_CAPABILITIES,
    MUST_BE_TEXT_INTEGER_AUDIT, MUST_BE_TEXT_SCALAR_INTEGER_AUDIT,
    MUST_BE_UNDERLYING_TYPE_INTEGER_CAPABILITIES, MUST_BE_VALID_VARIABLE_NAME_INTEGER_AUDIT,
    MUST_BE_VECTOR_INTEGER_CAPABILITIES, NAMEDARGS2CELL_DESCRIPTOR, NAMEDARGS2CELL_INTEGER_AUDIT,
    VALIDATE_FUNCTION_SIGNATURES_JSON_INTEGER_AUDIT, VALIDATOR_DESCRIPTOR,
};
use contracts::{
    MUST_BE_FINITE_RESIDENT_EXTENSION, MUST_BE_INTEGER_RESIDENT_EXTENSION,
    MUST_BE_NONZERO_RESIDENT_EXTENSION, MUST_BE_NON_NAN_RESIDENT_EXTENSION,
    VALIDATION_ERROR_PROVIDER_OWNERSHIP, VALIDATION_ERROR_PROVIDER_PAYLOAD,
};
mod errors;
pub use errors::validation_error;
use errors::{
    check_validator, invalid_argument_error, pass, require_arg_count, require_args,
    require_exact_arg_count,
};
mod dispatch;
pub use dispatch::{dispatch_validator, dispatch_validator_async};
mod shape;
pub use shape::{
    value_is_column, value_is_empty, value_is_scalar_or_empty, value_is_vector,
    value_satisfies_vector_validator, value_shape_2d,
};
mod class;
pub use class::{
    must_be_a, value_is_float, value_is_numeric, value_is_numeric_or_logical, value_matches_class,
    value_underlying_type_matches,
};
mod text;
use text::{chars_to_string, value_texts};
pub use text::{
    isvarname_value, namedargs2cell_value, validate_function_signatures_json,
    value_is_nonzero_length_text, value_is_text, value_is_text_scalar,
};
mod numeric;
pub use numeric::{
    value_is_finite, value_is_finite_async, value_is_greater_than, value_is_greater_than_or_equal,
    value_is_in_range, value_is_integer, value_is_integer_async, value_is_less_than,
    value_is_less_than_or_equal, value_is_negative, value_is_negative_async, value_is_non_nan,
    value_is_non_nan_async, value_is_nonmissing, value_is_nonmissing_async, value_is_nonnegative,
    value_is_nonnegative_async, value_is_nonpositive, value_is_nonpositive_async, value_is_nonzero,
    value_is_nonzero_async, value_is_positive, value_is_positive_async, value_is_real,
    value_is_real_async,
};
mod comparison;
use comparison::{numeric_class_identity, value_is_ordered_against_zero_async};
pub use comparison::{
    value_is_greater_than_or_equal_values_async, value_is_greater_than_values_async,
    value_is_in_range_documented_async, value_is_in_range_values_async,
    value_is_less_than_or_equal_values_async, value_is_less_than_values_async,
};
mod membership;
pub use membership::{
    atoms, value_is_member, value_is_member_async, value_is_member_atoms,
    value_is_member_atoms_async,
};
mod integer;
pub use integer::{
    ensure_runmat_integer_f64_boundary, integer_is_exact_f64, is_typed_complex_integer,
    native_integer_value_is_exact_f64, native_integer_value_is_exact_f64_async,
    reject_typed_complex_integer, reject_typed_complex_integer_tensor, value_contains_explicit_gpu,
    value_contains_native_integer_class, value_has_logical_class, value_has_native_integer_class,
};
mod elements;
use elements::{
    complex_real_values_all, exact_integer_values_all, int_f64_matches, int_is_negative,
    int_is_nonnegative, int_is_nonpositive, int_is_positive, numeric_values_all,
};
mod controls;
use controls::{range_inclusivity_arg, text_scalar_arg, type_names_arg};
mod resident;
use resident::host_value;
pub use resident::{ensure_resident_extension, validate_resident_metadata};

fn bool_type(
    _: &[runmat_builtins::Type],
    _: &runmat_builtins::ResolveContext,
) -> runmat_builtins::Type {
    runmat_builtins::Type::Bool
}

fn any_type(
    _: &[runmat_builtins::Type],
    _: &runmat_builtins::ResolveContext,
) -> runmat_builtins::Type {
    runmat_builtins::Type::Unknown
}

#[runtime_builtin(
    name = "isvarname",
    category = "argument-validation",
    summary = "Return true when text is a valid MATLAB variable name.",
    type_resolver(bool_type),
    descriptor(self::ISVARNAME_DESCRIPTOR),
    integer_audit(self::ISVARNAME_INTEGER_AUDIT),
    builtin_path = "crate::builtins::common::validation"
)]
fn isvarname_builtin(value: Value) -> BuiltinResult<Value> {
    Ok(Value::Bool(isvarname_value(&value)))
}

#[runtime_builtin(
    name = "namedargs2cell",
    category = "argument-validation",
    summary = "Convert a scalar name-value struct to an alternating name/value cell row.",
    type_resolver(any_type),
    descriptor(self::NAMEDARGS2CELL_DESCRIPTOR),
    integer_audit(self::NAMEDARGS2CELL_INTEGER_AUDIT),
    builtin_path = "crate::builtins::common::validation"
)]
fn namedargs2cell_builtin(value: Value) -> BuiltinResult<Value> {
    namedargs2cell_value(value)
}

macro_rules! validator_builtin {
    ($func:ident, $name:literal, capabilities = $capabilities:path) => {
        #[runtime_builtin(
            name = $name,
            category = "argument-validation",
            summary = "Validate an input argument and throw if the constraint is not satisfied.",
            sink = true,
            suppress_auto_output = true,
            descriptor(self::VALIDATOR_DESCRIPTOR),
            integer_capabilities($capabilities),
            builtin_path = "crate::builtins::common::validation"
        )]
        async fn $func(args: Vec<Value>) -> BuiltinResult<Value> {
            dispatch_validator_async($name, args).await
        }
    };
    ($func:ident, $name:literal, capabilities = $capabilities:path, extensions = $extensions:path) => {
        #[runtime_builtin(
            name = $name,
            category = "argument-validation",
            summary = "Validate an input argument and throw if the constraint is not satisfied.",
            sink = true,
            suppress_auto_output = true,
            descriptor(self::VALIDATOR_DESCRIPTOR),
            extensions($extensions),
            integer_capabilities($capabilities),
            builtin_path = "crate::builtins::common::validation"
        )]
        async fn $func(args: Vec<Value>) -> BuiltinResult<Value> {
            dispatch_validator_async($name, args).await
        }
    };
    ($func:ident, $name:literal, audit = $audit:path) => {
        #[runtime_builtin(
            name = $name,
            category = "argument-validation",
            summary = "Validate an input argument and throw if the constraint is not satisfied.",
            sink = true,
            suppress_auto_output = true,
            descriptor(self::VALIDATOR_DESCRIPTOR),
            integer_audit($audit),
            builtin_path = "crate::builtins::common::validation"
        )]
        async fn $func(args: Vec<Value>) -> BuiltinResult<Value> {
            dispatch_validator_async($name, args).await
        }
    };
    ($func:ident, $name:literal) => {
        #[runtime_builtin(
            name = $name,
            category = "argument-validation",
            summary = "Validate an input argument and throw if the constraint is not satisfied.",
            sink = true,
            suppress_auto_output = true,
            descriptor(self::VALIDATOR_DESCRIPTOR),
            builtin_path = "crate::builtins::common::validation"
        )]
        fn $func(args: Vec<Value>) -> BuiltinResult<Value> {
            dispatch_validator($name, args)
        }
    };
}

validator_builtin!(
    must_be_a_builtin,
    "mustBeA",
    capabilities = self::MUST_BE_A_INTEGER_CAPABILITIES
);
validator_builtin!(
    must_be_column_builtin,
    "mustBeColumn",
    capabilities = self::MUST_BE_COLUMN_INTEGER_CAPABILITIES
);
validator_builtin!(
    must_be_file_builtin,
    "mustBeFile",
    audit = self::MUST_BE_FILE_INTEGER_AUDIT
);
validator_builtin!(
    must_be_finite_builtin,
    "mustBeFinite",
    capabilities = self::MUST_BE_FINITE_INTEGER_CAPABILITIES,
    extensions = self::MUST_BE_FINITE_EXTENSIONS
);
validator_builtin!(
    must_be_float_builtin,
    "mustBeFloat",
    capabilities = self::MUST_BE_FLOAT_INTEGER_CAPABILITIES
);
validator_builtin!(
    must_be_folder_builtin,
    "mustBeFolder",
    audit = self::MUST_BE_FOLDER_INTEGER_AUDIT
);
validator_builtin!(
    must_be_greater_than_builtin,
    "mustBeGreaterThan",
    capabilities = self::MUST_BE_GREATER_THAN_INTEGER_CAPABILITIES
);
validator_builtin!(
    must_be_greater_than_or_equal_builtin,
    "mustBeGreaterThanOrEqual",
    capabilities = self::MUST_BE_GREATER_THAN_OR_EQUAL_INTEGER_CAPABILITIES
);
validator_builtin!(
    must_be_in_range_builtin,
    "mustBeInRange",
    capabilities = self::MUST_BE_IN_RANGE_INTEGER_CAPABILITIES
);
validator_builtin!(
    must_be_integer_builtin,
    "mustBeInteger",
    capabilities = self::MUST_BE_INTEGER_INTEGER_CAPABILITIES,
    extensions = self::MUST_BE_INTEGER_EXTENSIONS
);
validator_builtin!(
    must_be_less_than_builtin,
    "mustBeLessThan",
    capabilities = self::MUST_BE_LESS_THAN_INTEGER_CAPABILITIES
);
validator_builtin!(
    must_be_less_than_or_equal_builtin,
    "mustBeLessThanOrEqual",
    capabilities = self::MUST_BE_LESS_THAN_OR_EQUAL_INTEGER_CAPABILITIES
);
validator_builtin!(
    must_be_member_builtin,
    "mustBeMember",
    capabilities = self::MUST_BE_MEMBER_INTEGER_CAPABILITIES
);
validator_builtin!(
    must_be_negative_builtin,
    "mustBeNegative",
    capabilities = self::MUST_BE_NEGATIVE_INTEGER_CAPABILITIES
);
validator_builtin!(
    must_be_nonempty_builtin,
    "mustBeNonempty",
    capabilities = self::MUST_BE_NONEMPTY_INTEGER_CAPABILITIES
);
validator_builtin!(
    must_be_nonmissing_builtin,
    "mustBeNonmissing",
    capabilities = self::MUST_BE_NONMISSING_INTEGER_CAPABILITIES
);
validator_builtin!(
    must_be_non_nan_builtin,
    "mustBeNonNan",
    capabilities = self::MUST_BE_NON_NAN_INTEGER_CAPABILITIES,
    extensions = self::MUST_BE_NON_NAN_EXTENSIONS
);
validator_builtin!(
    must_be_nonnegative_builtin,
    "mustBeNonnegative",
    capabilities = self::MUST_BE_NONNEGATIVE_INTEGER_CAPABILITIES
);
validator_builtin!(
    must_be_nonpositive_builtin,
    "mustBeNonpositive",
    capabilities = self::MUST_BE_NONPOSITIVE_INTEGER_CAPABILITIES
);
validator_builtin!(
    must_be_nonsparse_builtin,
    "mustBeNonsparse",
    capabilities = self::MUST_BE_NONSPARSE_INTEGER_CAPABILITIES
);
validator_builtin!(
    must_be_nonzero_builtin,
    "mustBeNonzero",
    capabilities = self::MUST_BE_NONZERO_INTEGER_CAPABILITIES,
    extensions = self::MUST_BE_NONZERO_EXTENSIONS
);
validator_builtin!(
    must_be_nonzero_length_text_builtin,
    "mustBeNonzeroLengthText",
    audit = self::MUST_BE_NONZERO_LENGTH_TEXT_INTEGER_AUDIT
);
validator_builtin!(
    must_be_numeric_builtin,
    "mustBeNumeric",
    capabilities = self::MUST_BE_NUMERIC_INTEGER_CAPABILITIES
);
validator_builtin!(
    must_be_numeric_or_logical_builtin,
    "mustBeNumericOrLogical",
    capabilities = self::MUST_BE_NUMERIC_OR_LOGICAL_INTEGER_CAPABILITIES
);
validator_builtin!(
    must_be_positive_builtin,
    "mustBePositive",
    capabilities = self::MUST_BE_POSITIVE_INTEGER_CAPABILITIES
);
validator_builtin!(
    must_be_real_builtin,
    "mustBeReal",
    capabilities = self::MUST_BE_REAL_INTEGER_CAPABILITIES
);
validator_builtin!(
    must_be_scalar_or_empty_builtin,
    "mustBeScalarOrEmpty",
    capabilities = self::MUST_BE_SCALAR_OR_EMPTY_INTEGER_CAPABILITIES
);
validator_builtin!(
    must_be_sparse_builtin,
    "mustBeSparse",
    capabilities = self::MUST_BE_SPARSE_INTEGER_CAPABILITIES
);
validator_builtin!(
    must_be_text_builtin,
    "mustBeText",
    audit = self::MUST_BE_TEXT_INTEGER_AUDIT
);
validator_builtin!(
    must_be_text_scalar_builtin,
    "mustBeTextScalar",
    audit = self::MUST_BE_TEXT_SCALAR_INTEGER_AUDIT
);
validator_builtin!(
    must_be_underlying_type_builtin,
    "mustBeUnderlyingType",
    capabilities = self::MUST_BE_UNDERLYING_TYPE_INTEGER_CAPABILITIES
);
validator_builtin!(
    must_be_valid_variable_name_builtin,
    "mustBeValidVariableName",
    audit = self::MUST_BE_VALID_VARIABLE_NAME_INTEGER_AUDIT
);
validator_builtin!(
    must_be_vector_builtin,
    "mustBeVector",
    capabilities = self::MUST_BE_VECTOR_INTEGER_CAPABILITIES
);
validator_builtin!(
    validate_function_signatures_json_builtin,
    "validateFunctionSignaturesJSON",
    audit = self::VALIDATE_FUNCTION_SIGNATURES_JSON_INTEGER_AUDIT
);

#[cfg(test)]
mod tests;

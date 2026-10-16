//! Shared MATLAB argument-validation helpers and callable `mustBe*` builtins.

use std::cmp::Ordering;

use runmat_accelerate_api::{
    handle_integer_type, handle_is_logical, handle_storage, GpuTensorStorage,
};
use runmat_value::{
    CellArray, CharArray, ComplexTensor, IntValue, IntegerStorage, NumericDType, SparseTensor,
    Value,
};

use crate::builtins::common::identifiers::is_valid_varname;
use crate::builtins::common::tensor;
use crate::builtins::introspection::class_identity::class_name_for_value;
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
mod datetime_format;
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
    VALIDATE_FUNCTION_SIGNATURES_JSON_INTEGER_AUDIT,
};
use contracts::{VALIDATION_ERROR_PROVIDER_OWNERSHIP, VALIDATION_ERROR_PROVIDER_PAYLOAD};
pub(crate) use datetime_format::{
    datetime_format_to_strftime, parse_datetime_with_input_format, supported_datetime_input_format,
};
mod errors;
pub use errors::validation_error;
use errors::{check_validator, invalid_argument_error, pass, require_arg_count};
mod shape;
pub use shape::{
    value_is_column, value_is_empty, value_is_scalar_or_empty, value_is_vector,
    value_satisfies_nonsparse_validator, value_satisfies_sparse_validator,
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
    value_is_nonzero_length_text, value_is_text, value_is_text_scalar, value_text_scalar_string,
};
mod numeric;
mod cell_leaves;
mod path;
pub use numeric::{
    value_is_finite, value_is_finite_async, value_is_greater_than, value_is_greater_than_or_equal,
    value_is_in_range, value_is_integer, value_is_integer_async, value_is_less_than,
    value_is_less_than_or_equal, value_is_negative, value_is_negative_async, value_is_non_nan,
    value_is_non_nan_async, value_is_nonmissing, value_is_nonmissing_async, value_is_nonnegative,
    value_is_nonnegative_async, value_is_nonpositive, value_is_nonpositive_async, value_is_nonzero,
    value_is_nonzero_async, value_is_positive, value_is_positive_async, value_is_real,
    value_is_real_async,
};
pub use path::{value_is_file_async, value_is_folder_async};
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
    ensure_runmat_integer_f64_boundary, ensure_runmat_real_integer_f64_boundary,
    integer_is_exact_f64, is_typed_complex_integer, native_integer_value_is_exact_f64,
    native_integer_value_is_exact_f64_async, prepare_direct_integer_f64_boundary,
    reject_typed_complex_integer, reject_typed_complex_integer_for,
    reject_typed_complex_integer_tensor, value_contains_explicit_gpu,
    value_contains_native_integer_class, value_has_logical_class, value_has_native_integer_class,
    IntegerBoundaryFailure, PreparedIntegerBoundary, PreparedIntegerPayload,
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

mod admission;
pub(crate) mod isvarname;
pub(crate) mod must_be_a;
pub(crate) mod must_be_column;
pub(crate) mod must_be_float;
pub(crate) mod must_be_nonempty;
pub(crate) mod must_be_nonsparse;
pub(crate) mod must_be_numeric;
pub(crate) mod must_be_numeric_or_logical;
pub(crate) mod must_be_scalar_or_empty;
pub(crate) mod must_be_sparse;
pub(crate) mod must_be_text;
pub(crate) mod must_be_text_scalar;
pub(crate) mod must_be_nonzero_length_text;
pub(crate) mod must_be_underlying_type;
pub(crate) mod must_be_valid_variable_name;
pub(crate) mod must_be_vector;
pub(crate) mod namedargs2cell;
pub(crate) mod validate_function_signatures_json;

#[cfg(test)]
mod tests;

pub(crate) mod must_be_file;
pub(crate) mod must_be_folder;
pub(crate) mod must_be_finite;
pub(crate) mod must_be_integer;
pub(crate) mod must_be_non_nan;
pub(crate) mod must_be_nonzero;
pub(crate) mod must_be_nonmissing;
pub(crate) mod must_be_real;
pub(crate) mod must_be_positive;
pub(crate) mod must_be_negative;
pub(crate) mod must_be_nonnegative;
pub(crate) mod must_be_nonpositive;
pub(crate) mod must_be_greater_than;
pub(crate) mod must_be_greater_than_or_equal;
pub(crate) mod must_be_less_than;
pub(crate) mod must_be_less_than_or_equal;
pub(crate) mod must_be_in_range;
pub(crate) mod must_be_member;

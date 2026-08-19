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
use crate::builtins::introspection::underlying_type::underlying_type_matches;
use crate::builtins::logical::rel::integer_comparison::integer_f64_order;
use crate::builtins::logical::rel::integer_comparison::{
    try_complex_ordering_comparison, try_real_ordering_comparison, IntegerComparisonError,
    IntegerComparisonOp,
};
use crate::{build_runtime_error, BuiltinResult, RuntimeError};

/// MATLAB stores complex integer arrays, but does not support arithmetic on
/// them. Arithmetic builtins use this before selecting floating or provider
/// execution paths so exact integer components are never coerced to `f64`.
pub fn is_typed_complex_integer(value: &Value) -> bool {
    matches!(value, Value::ComplexTensor(tensor) if tensor.integer_storage().is_some())
        || matches!(value, Value::GpuTensor(handle)
            if handle_integer_type(handle).is_some()
                && handle_storage(handle) == GpuTensorStorage::ComplexInterleaved)
}

/// Reject a value that would otherwise enter a floating complex operation.
pub fn reject_typed_complex_integer(value: &Value, builtin: &str) -> BuiltinResult<()> {
    if is_typed_complex_integer(value) {
        return Err(build_runtime_error(format!(
            "{builtin}: operations involving complex numbers with integer types are not supported"
        ))
        .build());
    }
    Ok(())
}

/// Reject operations that would consume the lossy `f64` compatibility view of
/// a typed complex integer tensor. MATLAB permits storage/inspection of these
/// values, but not operations on them.
pub fn reject_typed_complex_integer_tensor(
    tensor: &ComplexTensor,
    builtin: &str,
) -> BuiltinResult<()> {
    if tensor.integer_storage().is_some() {
        return Err(build_runtime_error(format!(
            "{builtin}: operations involving complex numbers with integer types are not supported"
        ))
        .build());
    }
    Ok(())
}

#[derive(Debug, Clone, PartialEq)]
pub enum ValidationAtom {
    Number(f64),
    Integer(IntValue),
    ComplexNumber(f64, f64),
    ComplexInteger(IntValue, IntValue),
    Text(String),
    Bool(bool),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct RangeInclusivity {
    pub lower: bool,
    pub upper: bool,
}

impl RangeInclusivity {
    pub const CLOSED: Self = Self {
        lower: true,
        upper: true,
    };

    pub const OPEN: Self = Self {
        lower: false,
        upper: false,
    };

    pub const OPEN_LEFT: Self = Self {
        lower: false,
        upper: true,
    };

    pub const OPEN_RIGHT: Self = Self {
        lower: true,
        upper: false,
    };
}

const VALUE_INPUT: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "A",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Value to validate.",
};

const EXTRA_INPUT: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "B",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Variadic,
    default: None,
    description: "Additional validator-specific argument.",
};

const PREDICATE_OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "tf",
    ty: BuiltinParamType::LogicalArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Validation result.",
}];

const VALIDATOR_INPUTS: [BuiltinParamDescriptor; 2] = [VALUE_INPUT, EXTRA_INPUT];
const VALIDATOR_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "mustBe*(A, ...)",
    inputs: &VALIDATOR_INPUTS,
    outputs: &[],
}];

const PREDICATE_INPUTS: [BuiltinParamDescriptor; 1] = [VALUE_INPUT];
const ISVARNAME_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "tf = isvarname(S)",
    inputs: &PREDICATE_INPUTS,
    outputs: &PREDICATE_OUTPUT,
}];

const NAMEDARGS_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "C = namedargs2cell(S)",
    inputs: &PREDICATE_INPUTS,
    outputs: &[BuiltinParamDescriptor {
        name: "C",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Cell row vector of alternating field names and values.",
    }],
}];

const VALIDATION_ERROR_FAILED: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.ARGUMENT_VALIDATION.FAILED",
    identifier: Some("RunMat:validators:ValidationFailed"),
    when: "A value does not satisfy the requested validator.",
    message: "argument validation failed",
};

const VALIDATION_ERROR_INVALID_ARGUMENT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.ARGUMENT_VALIDATION.INVALID_ARGUMENT",
    identifier: Some("RunMat:validators:InvalidArgument"),
    when: "A validator receives an unsupported argument count or argument type.",
    message: "invalid argument validation input",
};

const VALIDATION_ERROR_PROVIDER_OWNERSHIP: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.ARGUMENT_VALIDATION.PROVIDER_OWNERSHIP_MISMATCH",
    identifier: Some("RunMat:validators:ProviderOwnershipMismatch"),
    when: "A resident value has no exact owning provider.",
    message: "argument validation: no acceleration provider owns the input",
};

const VALIDATION_ERROR_PROVIDER_PAYLOAD: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.ARGUMENT_VALIDATION.PROVIDER_PAYLOAD_MISMATCH",
    identifier: Some("RunMat:validators:ProviderPayloadMismatch"),
    when: "A resident value carries contradictory physical class metadata.",
    message: "argument validation: resident input has contradictory physical class metadata",
};

const VALIDATION_ERRORS: [BuiltinErrorDescriptor; 4] = [
    VALIDATION_ERROR_FAILED,
    VALIDATION_ERROR_INVALID_ARGUMENT,
    VALIDATION_ERROR_PROVIDER_OWNERSHIP,
    VALIDATION_ERROR_PROVIDER_PAYLOAD,
];

pub const VALIDATOR_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &VALIDATOR_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &VALIDATION_ERRORS,
};

const ALL_INTEGER_CLASSES: &[runmat_builtins::BuiltinIntegerClass] =
    &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES;

const VALIDATOR_INTEGER_VALUE: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "A",
        classes: ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "All eight integer classes are validated directly from authoritative class, shape, or exact element storage.",
    }];

const VALIDATOR_INTEGER_VALUE_AND_BOUND: [BuiltinIntegerInputCapability; 2] = [
    BuiltinIntegerInputCapability {
        name: "A",
        classes: ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "All eight integer classes participate in exact mixed numeric comparisons without conversion through binary64.",
    },
    BuiltinIntegerInputCapability {
        name: "bound",
        classes: ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "Integer, logical, single, and double compatible-size bounds retain their native values during comparison.",
    },
];

const VALIDATOR_INTEGER_RANGE_INPUTS: [BuiltinIntegerInputCapability; 3] = [
    BuiltinIntegerInputCapability {
        name: "A",
        classes: ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::Rejected,
        notes: "All eight integer classes are accepted, and the documented form requires both range bounds to use the same class as A.",
    },
    BuiltinIntegerInputCapability {
        name: "lower",
        classes: ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::Rejected,
        notes: "The lower bound must share A's integer class and is compared exactly.",
    },
    BuiltinIntegerInputCapability {
        name: "upper",
        classes: ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::Rejected,
        notes: "The upper bound must share A's integer class and is compared exactly.",
    },
];

const VALIDATOR_INTEGER_MEMBER_INPUTS: [BuiltinIntegerInputCapability; 2] = [
    BuiltinIntegerInputCapability {
        name: "A",
        classes: ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "Membership reads authoritative integer elements exactly.",
    },
    BuiltinIntegerInputCapability {
        name: "S",
        classes: ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "Unlike nondouble numeric inputs normally share a class; double retains the documented cross-class exception.",
    },
];

macro_rules! validator_integer_capability {
    ($constant:ident, $form:literal, $inputs:expr, $backend:expr, $overload:expr, $notes:literal) => {
        pub const $constant: [BuiltinIntegerCapabilityDescriptor; 1] =
            [BuiltinIntegerCapabilityDescriptor {
                form: $form,
                inputs: $inputs,
                computation_domain: BuiltinIntegerComputationDomain::Predicate,
                output_class: BuiltinIntegerOutputClassRule::NotApplicable,
                overflow: BuiltinIntegerOverflowRule::NotApplicable,
                backend: $backend,
                overload: $overload,
                notes: $notes,
            }];
    };
}

validator_integer_capability!(MUST_BE_A_INTEGER_CAPABILITIES, "mustBeA(integer_A, class_names)", &VALIDATOR_INTEGER_VALUE, BuiltinIntegerBackendRule::HostAndGpu, BuiltinIntegerOverloadKind::FunctionSpecific, "Class validation uses wrapper identity for explicit gpuArray values and the underlying integer class for ordinary host or internally automatic-resident values; payload data is never converted.");
validator_integer_capability!(MUST_BE_COLUMN_INTEGER_CAPABILITIES, "mustBeColumn(integer_A)", &VALIDATOR_INTEGER_VALUE, BuiltinIntegerBackendRule::HostAndGpu, BuiltinIntegerOverloadKind::StructuralParameter, "Column validation reads shape metadata only and supports resident integer arrays without gathering.");
validator_integer_capability!(MUST_BE_FINITE_INTEGER_CAPABILITIES, "mustBeFinite(integer_A)", &VALIDATOR_INTEGER_VALUE, BuiltinIntegerBackendRule::HostAndGpu, BuiltinIntegerOverloadKind::ElementwiseShapePreserving, "Every native integer value is finite; compatibility is decided from exact class metadata without floating conversion.");
validator_integer_capability!(MUST_BE_FLOAT_INTEGER_CAPABILITIES, "mustBeFloat(integer_A)", &VALIDATOR_INTEGER_VALUE, BuiltinIntegerBackendRule::HostAndGpu, BuiltinIntegerOverloadKind::FunctionSpecific, "Every native integer class fails isfloat-style validation from metadata without gathering or conversion.");
validator_integer_capability!(MUST_BE_GREATER_THAN_INTEGER_CAPABILITIES, "mustBeGreaterThan(integer_A, numeric_B)", &VALIDATOR_INTEGER_VALUE_AND_BOUND, BuiltinIntegerBackendRule::GatherFallback, BuiltinIntegerOverloadKind::BroadcastCompatible, "The compatibility target's compatible-size comparison is exact across mixed integer and floating classes; resident values use an owner-preserving predicate path.");
validator_integer_capability!(MUST_BE_GREATER_THAN_OR_EQUAL_INTEGER_CAPABILITIES, "mustBeGreaterThanOrEqual(integer_A, numeric_B)", &VALIDATOR_INTEGER_VALUE_AND_BOUND, BuiltinIntegerBackendRule::GatherFallback, BuiltinIntegerOverloadKind::BroadcastCompatible, "The compatibility target's compatible-size comparison is exact across mixed integer and floating classes; resident values use an owner-preserving predicate path.");
validator_integer_capability!(MUST_BE_IN_RANGE_INTEGER_CAPABILITIES, "mustBeInRange(integer_A, integer_lower, integer_upper, flags...)", &VALIDATOR_INTEGER_RANGE_INPUTS, BuiltinIntegerBackendRule::GatherFallback, BuiltinIntegerOverloadKind::BroadcastCompatible, "Documented same-class bounds and selectable open/closed endpoints compare exactly without binary64 materialization.");
validator_integer_capability!(MUST_BE_INTEGER_INTEGER_CAPABILITIES, "mustBeInteger(integer_A)", &VALIDATOR_INTEGER_VALUE, BuiltinIntegerBackendRule::HostAndGpu, BuiltinIntegerOverloadKind::ElementwiseShapePreserving, "Every native integer class is integral by construction; this validator checks value integrality rather than integer storage class.");
validator_integer_capability!(
    MUST_BE_LESS_THAN_INTEGER_CAPABILITIES,
    "mustBeLessThan(integer_A, numeric_B)",
    &VALIDATOR_INTEGER_VALUE_AND_BOUND,
    BuiltinIntegerBackendRule::GatherFallback,
    BuiltinIntegerOverloadKind::BroadcastCompatible,
    "The compatibility target's compatible-size comparison is exact across mixed integer and floating classes."
);
validator_integer_capability!(
    MUST_BE_LESS_THAN_OR_EQUAL_INTEGER_CAPABILITIES,
    "mustBeLessThanOrEqual(integer_A, numeric_B)",
    &VALIDATOR_INTEGER_VALUE_AND_BOUND,
    BuiltinIntegerBackendRule::GatherFallback,
    BuiltinIntegerOverloadKind::BroadcastCompatible,
    "The compatibility target's compatible-size comparison is exact across mixed integer and floating classes."
);
validator_integer_capability!(MUST_BE_MEMBER_INTEGER_CAPABILITIES, "mustBeMember(integer_A, integer_or_double_S)", &VALIDATOR_INTEGER_MEMBER_INPUTS, BuiltinIntegerBackendRule::GatherFallback, BuiltinIntegerOverloadKind::Multiple, "Exact membership preserves the documented same-class rule and double cross-class exception without rounding wide integers.");
validator_integer_capability!(MUST_BE_NEGATIVE_INTEGER_CAPABILITIES, "mustBeNegative(integer_A)", &VALIDATOR_INTEGER_VALUE, BuiltinIntegerBackendRule::GatherFallback, BuiltinIntegerOverloadKind::ElementwiseShapePreserving, "All elements compare exactly against zero; unsigned nonempty arrays fail and empty arrays pass.");
validator_integer_capability!(
    MUST_BE_NON_NAN_INTEGER_CAPABILITIES,
    "mustBeNonNan(integer_A)",
    &VALIDATOR_INTEGER_VALUE,
    BuiltinIntegerBackendRule::HostAndGpu,
    BuiltinIntegerOverloadKind::ElementwiseShapePreserving,
    "Native integer storage cannot contain NaN and passes from metadata without conversion."
);
validator_integer_capability!(
    MUST_BE_NONEMPTY_INTEGER_CAPABILITIES,
    "mustBeNonempty(integer_A)",
    &VALIDATOR_INTEGER_VALUE,
    BuiltinIntegerBackendRule::HostAndGpu,
    BuiltinIntegerOverloadKind::StructuralParameter,
    "Emptiness is decided from shape metadata without payload access."
);
validator_integer_capability!(MUST_BE_NONMISSING_INTEGER_CAPABILITIES, "mustBeNonmissing(integer_A)", &VALIDATOR_INTEGER_VALUE, BuiltinIntegerBackendRule::HostAndGpu, BuiltinIntegerOverloadKind::ElementwiseShapePreserving, "Native integer classes have no standard missing representation and therefore pass without conversion; public validator and delegated anymissing type lists are not fully synchronized.");
validator_integer_capability!(
    MUST_BE_NONNEGATIVE_INTEGER_CAPABILITIES,
    "mustBeNonnegative(integer_A)",
    &VALIDATOR_INTEGER_VALUE,
    BuiltinIntegerBackendRule::GatherFallback,
    BuiltinIntegerOverloadKind::ElementwiseShapePreserving,
    "All elements compare exactly against zero."
);
validator_integer_capability!(
    MUST_BE_NONPOSITIVE_INTEGER_CAPABILITIES,
    "mustBeNonpositive(integer_A)",
    &VALIDATOR_INTEGER_VALUE,
    BuiltinIntegerBackendRule::GatherFallback,
    BuiltinIntegerOverloadKind::ElementwiseShapePreserving,
    "All elements compare exactly against zero."
);
validator_integer_capability!(MUST_BE_NONSPARSE_INTEGER_CAPABILITIES, "mustBeNonsparse(integer_A)", &VALIDATOR_INTEGER_VALUE, BuiltinIntegerBackendRule::HostAndGpu, BuiltinIntegerOverloadKind::StructuralParameter, "Dense host and resident integer arrays pass from storage metadata; sparse integer arrays fail.");
validator_integer_capability!(
    MUST_BE_NONZERO_INTEGER_CAPABILITIES,
    "mustBeNonzero(integer_A)",
    &VALIDATOR_INTEGER_VALUE,
    BuiltinIntegerBackendRule::GatherFallback,
    BuiltinIntegerOverloadKind::ElementwiseShapePreserving,
    "All integer elements compare exactly against zero without a floating compatibility view."
);
validator_integer_capability!(
    MUST_BE_NUMERIC_INTEGER_CAPABILITIES,
    "mustBeNumeric(integer_A)",
    &VALIDATOR_INTEGER_VALUE,
    BuiltinIntegerBackendRule::HostAndGpu,
    BuiltinIntegerOverloadKind::FunctionSpecific,
    "All native integer classes satisfy isnumeric-style validation from class metadata."
);
validator_integer_capability!(
    MUST_BE_NUMERIC_OR_LOGICAL_INTEGER_CAPABILITIES,
    "mustBeNumericOrLogical(integer_A)",
    &VALIDATOR_INTEGER_VALUE,
    BuiltinIntegerBackendRule::HostAndGpu,
    BuiltinIntegerOverloadKind::FunctionSpecific,
    "All native integer classes satisfy the numeric branch from class metadata."
);
validator_integer_capability!(
    MUST_BE_POSITIVE_INTEGER_CAPABILITIES,
    "mustBePositive(integer_A)",
    &VALIDATOR_INTEGER_VALUE,
    BuiltinIntegerBackendRule::GatherFallback,
    BuiltinIntegerOverloadKind::ElementwiseShapePreserving,
    "All elements compare exactly against zero; empty arrays pass."
);
validator_integer_capability!(MUST_BE_REAL_INTEGER_CAPABILITIES, "mustBeReal(integer_A)", &VALIDATOR_INTEGER_VALUE, BuiltinIntegerBackendRule::HostAndGpu, BuiltinIntegerOverloadKind::FunctionSpecific, "All eight native integer classes are real by construction; host and resident values pass from authoritative class/storage metadata without floating conversion or payload access.");
validator_integer_capability!(MUST_BE_SCALAR_OR_EMPTY_INTEGER_CAPABILITIES, "mustBeScalarOrEmpty(integer_A)", &VALIDATOR_INTEGER_VALUE, BuiltinIntegerBackendRule::HostAndGpu, BuiltinIntegerOverloadKind::StructuralParameter, "Scalar-or-empty validation reads shape metadata only, including all documented empty integer shapes, without gathering resident payloads.");
validator_integer_capability!(MUST_BE_SPARSE_INTEGER_CAPABILITIES, "mustBeSparse(integer_A)", &VALIDATOR_INTEGER_VALUE, BuiltinIntegerBackendRule::HostAndGpu, BuiltinIntegerOverloadKind::StructuralParameter, "Every empty integer array passes as documented; nonempty dense or resident integer arrays fail from storage metadata, while RunMat-native sparse integer values pass without materialization.");
validator_integer_capability!(MUST_BE_UNDERLYING_TYPE_INTEGER_CAPABILITIES, "mustBeUnderlyingType(integer_A, typenames)", &VALIDATOR_INTEGER_VALUE, BuiltinIntegerBackendRule::HostAndGpu, BuiltinIntegerOverloadKind::FunctionSpecific, "One or more requested type names are compared with the authoritative signedness and width of host or resident integer storage without reading payload values.");

pub const MUST_BE_VECTOR_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 2] = [
    BuiltinIntegerCapabilityDescriptor {
        form: "mustBeVector(integer_A)",
        inputs: &VALIDATOR_INTEGER_VALUE,
        computation_domain: BuiltinIntegerComputationDomain::Predicate,
        output_class: BuiltinIntegerOutputClassRule::NotApplicable,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::HostAndGpu,
        overload: BuiltinIntegerOverloadKind::StructuralParameter,
        notes: "The documented 1-by-N or N-by-1 rule, including only vector-shaped empties, is decided from host or resident shape metadata without payload access.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "mustBeVector(integer_A, \"allow-all-empties\")",
        inputs: &VALIDATOR_INTEGER_VALUE,
        computation_domain: BuiltinIntegerComputationDomain::Predicate,
        output_class: BuiltinIntegerOutputClassRule::NotApplicable,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::HostAndGpu,
        overload: BuiltinIntegerOverloadKind::StructuralParameter,
        notes: "The compatibility target's option additionally accepts every empty integer shape while preserving the ordinary vector rule for nonempty values; callable and arguments-block paths share the same metadata-only predicate.",
    },
];

pub const MUST_BE_FILE_INTEGER_AUDIT: BuiltinIntegerAuditDescriptor = BuiltinIntegerAuditDescriptor { kind: BuiltinIntegerAuditKind::NotApplicable, canonical_builtin: None, notes: "mustBeFile is a text/path validator; integer host or resident values reject without numeric conversion or provider access." };
pub const MUST_BE_FOLDER_INTEGER_AUDIT: BuiltinIntegerAuditDescriptor = BuiltinIntegerAuditDescriptor { kind: BuiltinIntegerAuditKind::NotApplicable, canonical_builtin: None, notes: "mustBeFolder is a text/path validator; integer host or resident values reject without numeric conversion or provider access." };
pub const MUST_BE_NONZERO_LENGTH_TEXT_INTEGER_AUDIT: BuiltinIntegerAuditDescriptor = BuiltinIntegerAuditDescriptor { kind: BuiltinIntegerAuditKind::NotApplicable, canonical_builtin: None, notes: "mustBeNonzeroLengthText is a text validator; integer host or resident values fail without numeric conversion or provider access." };
pub const MUST_BE_TEXT_INTEGER_AUDIT: BuiltinIntegerAuditDescriptor = BuiltinIntegerAuditDescriptor { kind: BuiltinIntegerAuditKind::NotApplicable, canonical_builtin: None, notes: "mustBeText is a text validator; integer host or resident values fail before numeric conversion, payload access, or provider lookup." };
pub const MUST_BE_TEXT_SCALAR_INTEGER_AUDIT: BuiltinIntegerAuditDescriptor = BuiltinIntegerAuditDescriptor { kind: BuiltinIntegerAuditKind::NotApplicable, canonical_builtin: None, notes: "mustBeTextScalar is a text-shape validator; integer host or resident values fail before numeric conversion, payload access, or provider lookup." };
pub const MUST_BE_VALID_VARIABLE_NAME_INTEGER_AUDIT: BuiltinIntegerAuditDescriptor = BuiltinIntegerAuditDescriptor { kind: BuiltinIntegerAuditKind::NotApplicable, canonical_builtin: None, notes: "mustBeValidVariableName accepts text names only; integer host or resident values fail before numeric conversion, payload access, or provider lookup." };
pub const NAMEDARGS2CELL_INTEGER_AUDIT: BuiltinIntegerAuditDescriptor = BuiltinIntegerAuditDescriptor { kind: BuiltinIntegerAuditKind::NotApplicable, canonical_builtin: None, notes: "namedargs2cell accepts only a scalar structure. A top-level integer host or resident value rejects without conversion or provider access, while integer values stored in valid structure fields are preserved exactly as ordinary payloads." };
pub const VALIDATE_FUNCTION_SIGNATURES_JSON_INTEGER_AUDIT: BuiltinIntegerAuditDescriptor = BuiltinIntegerAuditDescriptor { kind: BuiltinIntegerAuditKind::NotApplicable, canonical_builtin: None, notes: "validateFunctionSignaturesJSON accepts text paths in the compatibility target and a text JSON payload in the current RunMat implementation; integer and resident numeric values are not valid in either form and reject without conversion or provider access." };

const MUST_BE_INTEGER_RESIDENT_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "mustBeInteger.resident-input",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "explicit gpuArray validation is not documented for mustBeInteger",
    error_identifier: Some("RunMat:compatibility:mustBeIntegerResidentInput"),
};
const MUST_BE_FINITE_RESIDENT_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "mustBeFinite.resident-input",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "explicit gpuArray validation is not documented for mustBeFinite",
    error_identifier: Some("RunMat:compatibility:mustBeFiniteResidentInput"),
};
const MUST_BE_NON_NAN_RESIDENT_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "mustBeNonNan.resident-input",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "explicit gpuArray validation is not documented for mustBeNonNan",
    error_identifier: Some("RunMat:compatibility:mustBeNonNanResidentInput"),
};
const MUST_BE_NONZERO_RESIDENT_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "mustBeNonzero.resident-input",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "explicit gpuArray validation is not documented for mustBeNonzero",
    error_identifier: Some("RunMat:compatibility:mustBeNonzeroResidentInput"),
};
pub const MUST_BE_INTEGER_EXTENSIONS: [BuiltinExtensionDescriptor; 1] =
    [MUST_BE_INTEGER_RESIDENT_EXTENSION];
pub const MUST_BE_FINITE_EXTENSIONS: [BuiltinExtensionDescriptor; 1] =
    [MUST_BE_FINITE_RESIDENT_EXTENSION];
pub const MUST_BE_NON_NAN_EXTENSIONS: [BuiltinExtensionDescriptor; 1] =
    [MUST_BE_NON_NAN_RESIDENT_EXTENSION];
pub const MUST_BE_NONZERO_EXTENSIONS: [BuiltinExtensionDescriptor; 1] =
    [MUST_BE_NONZERO_RESIDENT_EXTENSION];

pub const ISVARNAME_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &ISVARNAME_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &[],
};
pub const ISVARNAME_INTEGER_AUDIT: BuiltinIntegerAuditDescriptor =
    BuiltinIntegerAuditDescriptor {
        kind: BuiltinIntegerAuditKind::NotApplicable,
        canonical_builtin: None,
        notes: "isvarname is a text predicate; integer host or resident values return scalar false without numeric conversion or provider access.",
    };

pub const NAMEDARGS2CELL_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &NAMEDARGS_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &VALIDATION_ERRORS,
};

pub fn validation_error(builtin: &str, detail: impl AsRef<str>) -> RuntimeError {
    let detail = detail.as_ref();
    let message = if detail.is_empty() {
        format!("{builtin}: validation failed")
    } else {
        format!("{builtin}: {detail}")
    };
    build_runtime_error(message)
        .with_builtin(builtin)
        .with_identifier(format!("RunMat:{builtin}:ValidationFailed"))
        .build()
}

fn invalid_argument_error(builtin: &str, detail: impl AsRef<str>) -> RuntimeError {
    let detail = detail.as_ref();
    let message = if detail.is_empty() {
        format!("{builtin}: invalid argument")
    } else {
        format!("{builtin}: {detail}")
    };
    build_runtime_error(message)
        .with_builtin(builtin)
        .with_identifier(format!("RunMat:{builtin}:InvalidArgument"))
        .build()
}

fn pass() -> BuiltinResult<Value> {
    Ok(Value::Num(0.0))
}

fn require_args<'a>(
    builtin: &str,
    args: &'a [Value],
    min: usize,
    max: usize,
) -> BuiltinResult<&'a Value> {
    if args.len() < min || args.len() > max {
        return Err(invalid_argument_error(builtin, "invalid number of inputs").into());
    }
    args.first()
        .ok_or_else(|| invalid_argument_error(builtin, "missing value").into())
}

fn require_arg_count(builtin: &str, args: &[Value], min: usize, max: usize) -> BuiltinResult<()> {
    if args.len() < min || args.len() > max {
        return Err(invalid_argument_error(builtin, "invalid number of inputs").into());
    }
    Ok(())
}

fn require_exact_arg_count(builtin: &str, args: &[Value], expected: usize) -> BuiltinResult<()> {
    require_arg_count(builtin, args, expected, expected)
}

fn check_validator(builtin: &str, ok: bool) -> BuiltinResult<Value> {
    if ok {
        pass()
    } else {
        Err(validation_error(builtin, "value does not satisfy validator").into())
    }
}

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

pub fn value_shape_2d(value: &Value) -> (usize, usize) {
    match value {
        Value::Tensor(t) => (t.rows, t.cols),
        Value::SparseTensor(t) => (t.rows, t.cols),
        Value::ComplexTensor(t) => (t.rows, t.cols),
        Value::LogicalArray(a) => {
            let rows = a.shape.first().copied().unwrap_or(0);
            let cols = a.shape.get(1).copied().unwrap_or(1);
            (rows, cols)
        }
        Value::Cell(c) => (c.rows, c.cols),
        Value::CharArray(c) => (c.rows, c.cols),
        Value::StringArray(s) => (s.rows, s.cols),
        Value::GpuTensor(handle) => {
            let rows = handle.shape.first().copied().unwrap_or(1);
            let cols = handle.shape.get(1).copied().unwrap_or(1);
            (rows, cols)
        }
        _ => (1, 1),
    }
}

pub fn value_is_empty(value: &Value) -> bool {
    match value {
        Value::Tensor(t) => t.is_empty(),
        Value::SparseTensor(t) => t.rows == 0 || t.cols == 0,
        Value::ComplexTensor(t) => tensor::complex_tensor_element_len(t) == 0,
        Value::LogicalArray(a) => a.data.is_empty(),
        Value::StringArray(s) => s.data.is_empty(),
        Value::CharArray(c) => c.rows == 0 || c.cols == 0,
        Value::Cell(c) => c.data.is_empty(),
        Value::GpuTensor(handle) => handle.shape.contains(&0),
        _ => false,
    }
}

pub fn value_is_finite(value: &Value) -> bool {
    match value {
        Value::Num(v) => v.is_finite(),
        Value::Int(_) | Value::Bool(_) => true,
        Value::Complex(re, im) => re.is_finite() && im.is_finite(),
        Value::Tensor(t) if t.integer_storage().is_some() => true,
        Value::Tensor(t) => tensor::tensor_values_f64_cow(t)
            .iter()
            .all(|v| v.is_finite()),
        Value::SparseTensor(t) if t.integer_storage().is_some() => true,
        Value::SparseTensor(t) => t.materialize_f64().iter().all(|v| v.is_finite()),
        Value::ComplexTensor(t) if t.integer_storage().is_some() => true,
        Value::ComplexTensor(t) => t
            .materialize_f64()
            .iter()
            .all(|(re, im)| re.is_finite() && im.is_finite()),
        Value::LogicalArray(_) | Value::CharArray(_) => true,
        Value::GpuTensor(_) => true,
        _ => false,
    }
}

pub fn value_is_numeric(value: &Value) -> bool {
    match value {
        Value::Num(_)
        | Value::Int(_)
        | Value::Complex(_, _)
        | Value::Tensor(_)
        | Value::SparseTensor(_)
        | Value::ComplexTensor(_) => true,
        Value::GpuTensor(handle) => !handle_is_logical(handle),
        _ => false,
    }
}

pub fn value_is_float(value: &Value) -> bool {
    match value {
        Value::Num(_) | Value::Complex(_, _) => true,
        Value::ComplexTensor(tensor) => tensor.integer_storage().is_none(),
        Value::Tensor(t) => matches!(t.numeric_dtype(), NumericDType::F64 | NumericDType::F32),
        Value::SparseTensor(tensor) => tensor.integer_storage().is_none(),
        Value::GpuTensor(handle) => {
            !handle_is_logical(handle) && handle_integer_type(handle).is_none()
        }
        _ => false,
    }
}

pub fn value_is_numeric_or_logical(value: &Value) -> bool {
    value_is_numeric(value) || value_has_logical_class(value)
}

pub fn value_is_text(value: &Value) -> bool {
    match value {
        Value::String(_) | Value::StringArray(_) => true,
        Value::CharArray(chars) => chars.rows == 1,
        Value::Cell(cell) => cell.data.iter().all(value_is_text),
        _ => false,
    }
}

pub fn value_is_text_scalar(value: &Value) -> bool {
    match value {
        Value::String(_) => true,
        Value::StringArray(strings) => {
            strings.data.len() == 1 && strings.rows == 1 && strings.cols == 1
        }
        Value::CharArray(chars) => chars.rows == 1,
        _ => false,
    }
}

pub fn value_is_nonzero_length_text(value: &Value) -> bool {
    if !value_is_text(value) {
        return false;
    }
    match value {
        Value::String(s) => !s.is_empty(),
        Value::StringArray(s) => s.data.iter().all(|value| !value.is_empty()),
        Value::CharArray(c) => c.rows == 1 && c.cols > 0,
        Value::Cell(c) => c.data.iter().all(value_is_nonzero_length_text),
        _ => false,
    }
}

pub fn value_is_scalar_or_empty(value: &Value) -> bool {
    let (rows, cols) = value_shape_2d(value);
    (rows == 1 && cols == 1) || rows == 0 || cols == 0
}

pub fn value_is_real(value: &Value) -> bool {
    if value_is_empty(value) {
        return true;
    }
    match value {
        Value::Complex(_, im) => *im == 0.0,
        Value::ComplexTensor(t) if t.integer_storage().is_some() => t
            .integer_storage()
            .as_ref()
            .expect("checked integer complex storage")
            .imag
            .exact_values()
            .iter()
            .all(IntValue::is_zero),
        Value::ComplexTensor(t) => t.materialize_f64().iter().all(|(_, im)| *im == 0.0),
        Value::GpuTensor(handle) => handle_storage(handle) == GpuTensorStorage::Real,
        Value::Num(_)
        | Value::Int(_)
        | Value::Bool(_)
        | Value::Tensor(_)
        | Value::SparseTensor(_)
        | Value::LogicalArray(_)
        | Value::CharArray(_) => true,
        _ => false,
    }
}

pub async fn value_is_real_async(value: &Value) -> BuiltinResult<bool> {
    if matches!(value, Value::GpuTensor(handle) if handle_storage(handle) == GpuTensorStorage::ComplexInterleaved)
    {
        return Ok(value_is_real(&host_value(value).await?));
    }
    Ok(value_is_real(value))
}

pub fn value_is_integer(value: &Value) -> bool {
    match value {
        Value::Int(_) => true,
        Value::Bool(_) | Value::LogicalArray(_) | Value::CharArray(_) => true,
        Value::Num(v) => v.is_finite() && v.fract() == 0.0,
        Value::Tensor(t) if t.integer_storage().is_some() => true,
        Value::Tensor(t) => tensor::tensor_values_f64_cow(t)
            .iter()
            .all(|v| v.is_finite() && v.fract() == 0.0),
        Value::SparseTensor(t) if t.integer_storage().is_some() => true,
        Value::SparseTensor(t) => t
            .materialize_f64()
            .iter()
            .all(|v| v.is_finite() && v.fract() == 0.0),
        Value::Complex(re, im) => {
            re.is_finite() && re.fract() == 0.0 && im.is_finite() && im.fract() == 0.0
        }
        Value::ComplexTensor(t) if t.integer_storage().is_some() => true,
        Value::ComplexTensor(t) => t.materialize_f64().iter().all(|(re, im)| {
            re.is_finite() && re.fract() == 0.0 && im.is_finite() && im.fract() == 0.0
        }),
        Value::GpuTensor(handle) => {
            handle_is_logical(handle) || handle_integer_type(handle).is_some()
        }
        _ => false,
    }
}

pub fn value_is_non_nan(value: &Value) -> bool {
    match value {
        Value::Num(v) => !v.is_nan(),
        Value::Complex(re, im) => !re.is_nan() && !im.is_nan(),
        Value::Tensor(t) if t.integer_storage().is_some() => true,
        Value::Tensor(t) => tensor::tensor_values_f64_cow(t).iter().all(|v| !v.is_nan()),
        Value::SparseTensor(t) if t.integer_storage().is_some() => true,
        Value::SparseTensor(t) => t.materialize_f64().iter().all(|v| !v.is_nan()),
        Value::ComplexTensor(t) if t.integer_storage().is_some() => true,
        Value::ComplexTensor(t) => t
            .materialize_f64()
            .iter()
            .all(|(re, im)| !re.is_nan() && !im.is_nan()),
        Value::Cell(c) => c.data.iter().all(value_is_non_nan),
        _ => true,
    }
}

pub fn value_is_nonmissing(value: &Value) -> bool {
    value_is_non_nan(value)
}

pub fn value_is_positive(value: &Value) -> bool {
    if let Some(result) = complex_real_values_all(value, |v| v > 0.0, int_is_positive) {
        return result;
    }
    if let Some(result) = exact_integer_values_all(value, int_is_positive) {
        return result;
    }
    numeric_values_all(value, |v| v > 0.0)
}

pub fn value_is_negative(value: &Value) -> bool {
    if let Some(result) = complex_real_values_all(value, |v| v < 0.0, int_is_negative) {
        return result;
    }
    if let Some(result) = exact_integer_values_all(value, int_is_negative) {
        return result;
    }
    numeric_values_all(value, |v| v < 0.0)
}

pub fn value_is_nonnegative(value: &Value) -> bool {
    if let Some(result) = complex_real_values_all(value, |v| v >= 0.0, int_is_nonnegative) {
        return result;
    }
    if let Some(result) = exact_integer_values_all(value, int_is_nonnegative) {
        return result;
    }
    numeric_values_all(value, |v| v >= 0.0)
}

pub fn value_is_nonpositive(value: &Value) -> bool {
    if let Some(result) = complex_real_values_all(value, |v| v <= 0.0, int_is_nonpositive) {
        return result;
    }
    if let Some(result) = exact_integer_values_all(value, int_is_nonpositive) {
        return result;
    }
    numeric_values_all(value, |v| v <= 0.0)
}

pub fn value_is_nonzero(value: &Value) -> bool {
    match value {
        Value::Complex(re, im) => *re != 0.0 || *im != 0.0,
        Value::ComplexTensor(t) if t.integer_storage().is_some() => {
            let integer_data = t.integer_storage().expect("checked integer data");
            (0..integer_data.len()).all(|index| integer_data.is_nonzero_at(index).unwrap_or(false))
        }
        Value::ComplexTensor(t) => t
            .materialize_f64()
            .iter()
            .all(|(re, im)| *re != 0.0 || *im != 0.0),
        _ => {
            if let Some(result) = exact_integer_values_all(value, |integer| !integer.is_zero()) {
                return result;
            }
            numeric_values_all(value, |v| v != 0.0)
        }
    }
}

pub fn value_is_greater_than_or_equal(value: &Value, threshold: f64) -> bool {
    if let Some(result) = exact_integer_values_all(value, |integer| {
        int_f64_matches(integer, threshold, |ordering| ordering >= Ordering::Equal)
    }) {
        return result;
    }
    numeric_values_all(value, |v| v.is_finite() && v >= threshold)
}

pub fn value_is_less_than_or_equal(value: &Value, threshold: f64) -> bool {
    if let Some(result) = exact_integer_values_all(value, |integer| {
        int_f64_matches(integer, threshold, |ordering| ordering <= Ordering::Equal)
    }) {
        return result;
    }
    numeric_values_all(value, |v| v.is_finite() && v <= threshold)
}

pub fn value_is_greater_than(value: &Value, threshold: f64) -> bool {
    if let Some(result) = exact_integer_values_all(value, |integer| {
        int_f64_matches(integer, threshold, |ordering| ordering == Ordering::Greater)
    }) {
        return result;
    }
    numeric_values_all(value, |v| v.is_finite() && v > threshold)
}

pub fn value_is_less_than(value: &Value, threshold: f64) -> bool {
    if let Some(result) = exact_integer_values_all(value, |integer| {
        int_f64_matches(integer, threshold, |ordering| ordering == Ordering::Less)
    }) {
        return result;
    }
    numeric_values_all(value, |v| v.is_finite() && v < threshold)
}

pub fn value_is_in_range(
    value: &Value,
    lower: f64,
    upper: f64,
    inclusivity: RangeInclusivity,
) -> bool {
    if let Some(result) = exact_integer_values_all(value, |integer| {
        let lower_ok = int_f64_matches(integer, lower, |ordering| {
            if inclusivity.lower {
                ordering >= Ordering::Equal
            } else {
                ordering == Ordering::Greater
            }
        });
        let upper_ok = int_f64_matches(integer, upper, |ordering| {
            if inclusivity.upper {
                ordering <= Ordering::Equal
            } else {
                ordering == Ordering::Less
            }
        });
        lower_ok && upper_ok
    }) {
        return result;
    }
    numeric_values_all(value, |v| {
        v.is_finite()
            && if inclusivity.lower {
                v >= lower
            } else {
                v > lower
            }
            && if inclusivity.upper {
                v <= upper
            } else {
                v < upper
            }
    })
}

async fn resident_host_value(value: &Value) -> BuiltinResult<Option<Value>> {
    let Value::GpuTensor(handle) = value else {
        return Ok(None);
    };
    let owner = crate::builtins::common::gpu_helpers::exact_provider_for_handle(handle)
        .ok_or_else(|| {
            build_runtime_error(VALIDATION_ERROR_PROVIDER_OWNERSHIP.message)
                .with_gpu_gather_retry(crate::GpuGatherRetry::Never)
                .build()
        })?;
    crate::builtins::common::gpu_helpers::download_value_preserving_residency_async(owner, handle)
        .await
        .map(Some)
}

async fn host_value(value: &Value) -> BuiltinResult<Value> {
    validate_resident_metadata(value)?;
    Ok(match resident_host_value(value).await? {
        Some(value) => value,
        None => value.clone(),
    })
}

fn explicit_resident(value: &Value) -> bool {
    matches!(value, Value::GpuTensor(handle) if runmat_accelerate_api::handle_provenance(handle) == Some(runmat_accelerate_api::GpuHandleProvenance::Explicit))
}

pub fn ensure_resident_extension(value: &Value, builtin: &str) -> BuiltinResult<()> {
    if !explicit_resident(value) {
        return Ok(());
    }
    let extension = match builtin {
        "mustBeFinite" => &MUST_BE_FINITE_RESIDENT_EXTENSION,
        "mustBeInteger" => &MUST_BE_INTEGER_RESIDENT_EXTENSION,
        "mustBeNonNan" => &MUST_BE_NON_NAN_RESIDENT_EXTENSION,
        "mustBeNonzero" => &MUST_BE_NONZERO_RESIDENT_EXTENSION,
        _ => return Ok(()),
    };
    crate::compatibility::ensure_builtin_extension_enabled(extension, builtin)
}

pub fn validate_resident_metadata(value: &Value) -> BuiltinResult<()> {
    let Value::GpuTensor(handle) = value else {
        return Ok(());
    };
    let owner = crate::builtins::common::gpu_helpers::exact_provider_for_handle(handle)
        .ok_or_else(|| {
            build_runtime_error(VALIDATION_ERROR_PROVIDER_OWNERSHIP.message)
                .with_identifier(
                    VALIDATION_ERROR_PROVIDER_OWNERSHIP
                        .identifier
                        .expect("validator provider-ownership descriptor identifier"),
                )
                .with_gpu_gather_retry(crate::GpuGatherRetry::Never)
                .build()
        })?;
    let storage = handle_storage(handle);
    let integer = handle_integer_type(handle);
    let logical = handle_is_logical(handle);
    let precision = runmat_accelerate_api::handle_precision(handle);
    let expected_class =
        crate::builtins::common::gpu_helpers::expected_gpu_class_name(precision, integer, logical);
    let class_valid = runmat_accelerate_api::handle_class_name(handle)
        .as_deref()
        .is_none_or(|class_name| Some(class_name) == expected_class);
    let physical_valid = if integer.is_some() {
        storage == GpuTensorStorage::Real && precision.is_none() && !logical
    } else if logical {
        storage == GpuTensorStorage::Real && precision == Some(owner.precision())
    } else {
        precision == Some(owner.precision())
    };
    if !physical_valid || !class_valid {
        return Err(
            build_runtime_error(VALIDATION_ERROR_PROVIDER_PAYLOAD.message)
                .with_identifier(
                    VALIDATION_ERROR_PROVIDER_PAYLOAD
                        .identifier
                        .expect("validator provider-payload descriptor identifier"),
                )
                .with_gpu_gather_retry(crate::GpuGatherRetry::Never)
                .build()
                .into(),
        );
    }
    Ok(())
}

pub async fn value_is_finite_async(value: &Value) -> BuiltinResult<bool> {
    if matches!(value, Value::GpuTensor(handle) if handle_integer_type(handle).is_some() || handle_is_logical(handle))
    {
        return Ok(true);
    }
    let value = host_value(value).await?;
    Ok(value_is_finite(&value))
}

pub async fn value_is_integer_async(value: &Value) -> BuiltinResult<bool> {
    if matches!(value, Value::GpuTensor(handle) if handle_integer_type(handle).is_some() || handle_is_logical(handle))
    {
        return Ok(true);
    }
    let value = host_value(value).await?;
    Ok(value_is_integer(&value))
}

pub async fn value_is_non_nan_async(value: &Value) -> BuiltinResult<bool> {
    if matches!(value, Value::GpuTensor(handle) if handle_integer_type(handle).is_some() || handle_is_logical(handle))
    {
        return Ok(true);
    }
    let value = host_value(value).await?;
    Ok(value_is_non_nan(&value))
}

pub async fn value_is_nonmissing_async(value: &Value) -> BuiltinResult<bool> {
    if matches!(value, Value::GpuTensor(handle) if handle_integer_type(handle).is_some() || handle_is_logical(handle))
    {
        return Ok(true);
    }
    let value = host_value(value).await?;
    Ok(value_is_nonmissing(&value))
}

pub async fn value_is_positive_async(value: &Value) -> BuiltinResult<bool> {
    value_is_ordered_against_zero_async(value, IntegerComparisonOp::Gt).await
}

pub async fn value_is_negative_async(value: &Value) -> BuiltinResult<bool> {
    value_is_ordered_against_zero_async(value, IntegerComparisonOp::Lt).await
}

pub async fn value_is_nonnegative_async(value: &Value) -> BuiltinResult<bool> {
    value_is_ordered_against_zero_async(value, IntegerComparisonOp::Ge).await
}

pub async fn value_is_nonpositive_async(value: &Value) -> BuiltinResult<bool> {
    value_is_ordered_against_zero_async(value, IntegerComparisonOp::Le).await
}

pub async fn value_is_nonzero_async(value: &Value) -> BuiltinResult<bool> {
    let value = host_value(value).await?;
    Ok(value_is_nonzero(&value))
}

async fn value_is_ordered_against_zero_async(
    value: &Value,
    operation: IntegerComparisonOp,
) -> BuiltinResult<bool> {
    let value = host_value(value).await?;
    comparison_all_normalized(&value, &Value::Num(0.0), operation)
}

pub async fn value_is_greater_than_values_async(
    value: &Value,
    bound: &Value,
) -> BuiltinResult<bool> {
    compare_values_async(value, bound, IntegerComparisonOp::Gt).await
}

pub async fn value_is_greater_than_or_equal_values_async(
    value: &Value,
    bound: &Value,
) -> BuiltinResult<bool> {
    compare_values_async(value, bound, IntegerComparisonOp::Ge).await
}

pub async fn value_is_less_than_values_async(value: &Value, bound: &Value) -> BuiltinResult<bool> {
    compare_values_async(value, bound, IntegerComparisonOp::Lt).await
}

pub async fn value_is_less_than_or_equal_values_async(
    value: &Value,
    bound: &Value,
) -> BuiltinResult<bool> {
    compare_values_async(value, bound, IntegerComparisonOp::Le).await
}

async fn compare_values_async(
    value: &Value,
    bound: &Value,
    operation: IntegerComparisonOp,
) -> BuiltinResult<bool> {
    let value = host_value(value).await?;
    let bound = host_value(bound).await?;
    comparison_all_normalized(&value, &bound, operation)
}

pub async fn value_is_in_range_values_async(
    value: &Value,
    lower: &Value,
    upper: &Value,
    inclusivity: RangeInclusivity,
) -> BuiltinResult<bool> {
    let value = host_value(value).await?;
    let lower = host_value(lower).await?;
    let upper = host_value(upper).await?;
    let lower_op = if inclusivity.lower {
        IntegerComparisonOp::Ge
    } else {
        IntegerComparisonOp::Gt
    };
    let upper_op = if inclusivity.upper {
        IntegerComparisonOp::Le
    } else {
        IntegerComparisonOp::Lt
    };
    Ok(comparison_all_normalized(&value, &lower, lower_op)?
        && comparison_all_normalized(&value, &upper, upper_op)?)
}

pub async fn value_is_in_range_documented_async(
    value: &Value,
    lower: &Value,
    upper: &Value,
    inclusivity: RangeInclusivity,
) -> BuiltinResult<bool> {
    ensure_same_numeric_class("mustBeInRange", value, lower)?;
    ensure_same_numeric_class("mustBeInRange", value, upper)?;
    value_is_in_range_values_async(value, lower, upper, inclusivity).await
}

fn comparison_all_normalized(
    lhs: &Value,
    rhs: &Value,
    operation: IntegerComparisonOp,
) -> BuiltinResult<bool> {
    if let Value::SparseTensor(sparse) = lhs {
        if let Some(rhs) = scalar_numeric_value(rhs) {
            return sparse_scalar_comparison_all(sparse, &rhs, true, operation);
        }
    }
    if let Value::SparseTensor(sparse) = rhs {
        if let Some(lhs) = scalar_numeric_value(lhs) {
            return sparse_scalar_comparison_all(sparse, &lhs, false, operation);
        }
    }
    let lhs_dense;
    let rhs_dense;
    let lhs = if let Value::SparseTensor(sparse) = lhs {
        lhs_dense = Value::Tensor(
            sparse
                .to_dense()
                .map_err(|error| invalid_argument_error("argumentValidation", error))?,
        );
        &lhs_dense
    } else {
        lhs
    };
    let rhs = if let Value::SparseTensor(sparse) = rhs {
        rhs_dense = Value::Tensor(
            sparse
                .to_dense()
                .map_err(|error| invalid_argument_error("argumentValidation", error))?,
        );
        &rhs_dense
    } else {
        rhs
    };
    comparison_all(lhs, rhs, operation)
}

fn comparison_all(lhs: &Value, rhs: &Value, operation: IntegerComparisonOp) -> BuiltinResult<bool> {
    let map_error = |error| match error {
        IntegerComparisonError::SizeMismatch => invalid_argument_error(
            "argumentValidation",
            "numeric inputs are not compatible for implicit expansion",
        ),
        IntegerComparisonError::Internal => {
            invalid_argument_error("argumentValidation", "numeric comparison failed")
        }
    };
    let result = if let Some(result) =
        try_real_ordering_comparison(lhs, rhs, operation).map_err(map_error)?
    {
        result
    } else {
        try_complex_ordering_comparison(lhs, rhs, operation)
            .map_err(map_error)?
            .ok_or_else(|| {
                invalid_argument_error("argumentValidation", "expected comparable numeric inputs")
            })?
    };
    logical_value_all_true(&result)
}

fn scalar_numeric_value(value: &Value) -> Option<Value> {
    match value {
        Value::Num(_) | Value::Int(_) | Value::Bool(_) => Some(value.clone()),
        Value::Complex(real, _) => Some(Value::Num(*real)),
        Value::Tensor(tensor) if tensor::tensor_element_len(tensor) == 1 => tensor
            .numeric_value_at(0)
            .map(|value| match value.into_int_value() {
                Some(value) => Value::Int(value),
                None => Value::Num(floating_numeric_scalar_to_f64(value)),
            }),
        Value::LogicalArray(array) if array.data.len() == 1 => {
            Some(Value::Bool(array.data[0] != 0))
        }
        Value::ComplexTensor(tensor) if tensor::complex_tensor_element_len(tensor) == 1 => {
            tensor.numeric_value_at(0).map(|(real, _)| {
                if let Some(real) = real.into_int_value() {
                    Value::Int(real)
                } else {
                    Value::Num(floating_numeric_scalar_to_f64(real))
                }
            })
        }
        _ => None,
    }
}

fn sparse_scalar_comparison_all(
    sparse: &SparseTensor,
    scalar: &Value,
    sparse_is_left: bool,
    operation: IntegerComparisonOp,
) -> BuiltinResult<bool> {
    for index in 0..sparse.nnz() {
        let value = sparse.numeric_value_at(index).ok_or_else(|| {
            invalid_argument_error("argumentValidation", "invalid sparse storage")
        })?;
        let value = match value.into_int_value() {
            Some(value) => Value::Int(value),
            None => Value::Num(floating_numeric_scalar_to_f64(value)),
        };
        let matches = if sparse_is_left {
            comparison_all(&value, scalar, operation)?
        } else {
            comparison_all(scalar, &value, operation)?
        };
        if !matches {
            return Ok(false);
        }
    }
    if sparse.nnz() < sparse.rows.saturating_mul(sparse.cols) {
        let zero = Value::Num(0.0);
        return if sparse_is_left {
            comparison_all(&zero, scalar, operation)
        } else {
            comparison_all(scalar, &zero, operation)
        };
    }
    Ok(true)
}

fn floating_numeric_scalar_to_f64(value: runmat_value::NumericScalar) -> f64 {
    match value {
        runmat_value::NumericScalar::F64(value) => value,
        runmat_value::NumericScalar::F32(value) => f64::from(value),
        _ => unreachable!("integer numeric scalar was handled before floating conversion"),
    }
}

fn logical_value_all_true(value: &Value) -> BuiltinResult<bool> {
    match value {
        Value::Bool(value) => Ok(*value),
        Value::LogicalArray(array) => Ok(array.data.iter().all(|value| *value != 0)),
        _ => Err(invalid_argument_error(
            "argumentValidation",
            "comparison did not return logical data",
        )
        .into()),
    }
}

fn ensure_same_numeric_class(builtin: &str, lhs: &Value, rhs: &Value) -> BuiltinResult<()> {
    let lhs = numeric_class_name(lhs)
        .ok_or_else(|| invalid_argument_error(builtin, "expected numeric input"))?;
    let rhs = numeric_class_name(rhs)
        .ok_or_else(|| invalid_argument_error(builtin, "expected numeric bound"))?;
    if lhs != rhs {
        return Err(invalid_argument_error(
            builtin,
            "bounds must have the same numeric class as the value",
        )
        .into());
    }
    Ok(())
}

fn numeric_class_name(value: &Value) -> Option<String> {
    match value {
        Value::GpuTensor(handle) if handle_is_logical(handle) => Some("logical".into()),
        Value::GpuTensor(handle) if handle_integer_type(handle).is_some() => {
            crate::builtins::common::gpu_helpers::expected_gpu_class_name(
                None,
                handle_integer_type(handle),
                false,
            )
            .map(str::to_owned)
        }
        Value::GpuTensor(handle) => crate::builtins::common::gpu_helpers::expected_gpu_class_name(
            runmat_accelerate_api::handle_precision(handle),
            None,
            false,
        )
        .map(str::to_owned),
        _ if value_is_numeric_or_logical(value) => {
            Some(class_name_for_value(value).to_ascii_lowercase())
        }
        _ => None,
    }
}

pub fn value_is_column(value: &Value) -> bool {
    let shape: &[usize] = match value {
        Value::Tensor(value) => &value.shape,
        Value::ComplexTensor(value) => &value.shape,
        Value::LogicalArray(value) => &value.shape,
        Value::StringArray(value) => &value.shape,
        Value::Cell(value) => &value.shape,
        Value::CharArray(value) => &value.shape,
        Value::GpuTensor(handle) => &handle.shape,
        _ => {
            let (_, cols) = value_shape_2d(value);
            return cols == 1;
        }
    };
    shape.get(1).copied().unwrap_or(1) == 1 && shape.iter().skip(2).all(|extent| *extent == 1)
}

pub fn value_is_vector(value: &Value) -> Result<bool, RuntimeError> {
    fn shape_is_vector(shape: &[usize]) -> bool {
        let mut dimensions = shape.len();
        while dimensions > 2 && shape[dimensions - 1] == 1 {
            dimensions -= 1;
        }
        if dimensions > 2 {
            return false;
        }
        let rows = shape.first().copied().unwrap_or(1);
        let cols = shape.get(1).copied().unwrap_or(1);
        rows == 1 || cols == 1
    }

    Ok(match value {
        Value::Tensor(value) => shape_is_vector(&value.shape),
        Value::ComplexTensor(value) => shape_is_vector(&value.shape),
        Value::LogicalArray(value) => shape_is_vector(&value.shape),
        Value::StringArray(value) => shape_is_vector(&value.shape),
        Value::Cell(value) => shape_is_vector(&value.shape),
        Value::CharArray(value) => shape_is_vector(&value.shape),
        Value::GpuTensor(handle) => shape_is_vector(&handle.shape),
        Value::SparseTensor(value) => value.rows == 1 || value.cols == 1,
        _ => true,
    })
}

pub fn value_satisfies_vector_validator(
    value: &Value,
    allow_all_empties: bool,
) -> Result<bool, RuntimeError> {
    Ok(value_is_vector(value)? || (allow_all_empties && value_is_empty(value)))
}

pub fn value_matches_class(value: &Value, class_name: &str) -> bool {
    let requested = class_name.trim();
    if requested.is_empty() {
        return false;
    }
    if matches!(value, Value::GpuTensor(handle) if runmat_accelerate_api::handle_is_explicit(handle))
    {
        return requested.eq_ignore_ascii_case("gpuarray");
    }
    match requested.to_ascii_lowercase().as_str() {
        "numeric" => value_is_numeric(value),
        "float" => value_is_float(value),
        "integer" => value_has_native_integer_class(value),
        "logical" => value_has_logical_class(value),
        "char" => matches!(value, Value::CharArray(_)),
        "string" => matches!(value, Value::String(_) | Value::StringArray(_)),
        "cell" => matches!(value, Value::Cell(_)),
        "struct" => matches!(value, Value::Struct(_)),
        "sparse" => matches!(value, Value::SparseTensor(_)),
        "double" => {
            matches!(value, Value::Num(_) | Value::Complex(_, _))
                || matches!(value, Value::Tensor(t) if t.numeric_dtype() == NumericDType::F64)
                || matches!(value, Value::SparseTensor(t) if t.integer_storage().is_none())
                || matches!(value, Value::ComplexTensor(t) if t.numeric_dtype() == NumericDType::F64)
                || matches!(value, Value::GpuTensor(handle) if !handle_is_logical(handle) && handle_integer_type(handle).is_none() && runmat_accelerate_api::handle_precision(handle) == Some(runmat_accelerate_api::ProviderPrecision::F64))
        }
        "single" => {
            matches!(value, Value::Tensor(t) if t.numeric_dtype() == NumericDType::F32)
                || matches!(value, Value::ComplexTensor(t) if t.numeric_dtype() == NumericDType::F32)
                || matches!(value, Value::GpuTensor(handle) if !handle_is_logical(handle) && handle_integer_type(handle).is_none() && runmat_accelerate_api::handle_precision(handle) == Some(runmat_accelerate_api::ProviderPrecision::F32))
        }
        "gpuarray" => {
            matches!(value, Value::GpuTensor(handle) if runmat_accelerate_api::handle_is_explicit(handle))
        }
        _ => class_name_for_value(value).eq_ignore_ascii_case(requested),
    }
}

pub fn value_has_native_integer_class(value: &Value) -> bool {
    match value {
        Value::Int(_) => true,
        Value::Tensor(tensor) => tensor.integer_storage().is_some(),
        Value::SparseTensor(tensor) => tensor.integer_storage().is_some(),
        Value::ComplexTensor(tensor) => tensor.integer_storage().is_some(),
        Value::GpuTensor(handle) => handle_integer_type(handle).is_some(),
        _ => false,
    }
}

/// Returns whether `value`, including aggregate payloads, contains a native
/// integer class. Use this at compatibility boundaries before a recursive
/// gather can erase resident class metadata.
pub fn value_contains_native_integer_class(value: &Value) -> bool {
    match value {
        Value::Cell(cell) => cell.data.iter().any(value_contains_native_integer_class),
        Value::Struct(value) => value
            .fields
            .values()
            .any(value_contains_native_integer_class),
        Value::Object(value) => value
            .properties
            .values()
            .any(value_contains_native_integer_class),
        Value::Closure(value) => value
            .captures
            .iter()
            .any(value_contains_native_integer_class),
        Value::OutputList(values) => values.iter().any(value_contains_native_integer_class),
        _ => value_has_native_integer_class(value),
    }
}

/// Returns whether `value`, including aggregate payloads, contains a handle
/// created through explicit `gpuArray` intent rather than automatic residency.
pub fn value_contains_explicit_gpu(value: &Value) -> bool {
    match value {
        Value::GpuTensor(handle) => runmat_accelerate_api::handle_is_explicit(handle),
        Value::Cell(cell) => cell.data.iter().any(value_contains_explicit_gpu),
        Value::Struct(value) => value.fields.values().any(value_contains_explicit_gpu),
        Value::Object(value) => value.properties.values().any(value_contains_explicit_gpu),
        Value::Closure(value) => value.captures.iter().any(value_contains_explicit_gpu),
        Value::OutputList(values) => values.iter().any(value_contains_explicit_gpu),
        _ => false,
    }
}

pub fn value_has_logical_class(value: &Value) -> bool {
    matches!(value, Value::Bool(_) | Value::LogicalArray(_))
        || matches!(value, Value::GpuTensor(handle) if handle_is_logical(handle))
}

pub fn native_integer_value_is_exact_f64(value: &Value) -> bool {
    let exact = crate::builtins::math::trigonometry::cos::integer_is_exact_f64;
    match value {
        Value::Int(value) => exact(value),
        Value::Tensor(tensor) => tensor
            .integer_storage()
            .is_none_or(|storage| storage.exact_values().iter().all(exact)),
        Value::SparseTensor(tensor) => tensor
            .integer_storage()
            .is_none_or(|storage| storage.exact_values().iter().all(exact)),
        Value::ComplexTensor(tensor) => tensor.integer_storage().is_none_or(|storage| {
            storage.real.exact_values().iter().all(exact)
                && storage.imag.exact_values().iter().all(exact)
        }),
        Value::Cell(cell) => cell.data.iter().all(native_integer_value_is_exact_f64),
        Value::Struct(value) => value.fields.values().all(native_integer_value_is_exact_f64),
        Value::OutputList(values) => values.iter().all(native_integer_value_is_exact_f64),
        Value::GpuTensor(handle) => runmat_accelerate_api::handle_integer_type(handle)
            .is_none_or(|element_type| element_type.element_size() <= 4),
        _ => true,
    }
}

/// Check a native integer value against a binary64 boundary without rejecting a
/// resident 64-bit value solely because its contents are not visible in handle
/// metadata. Compatibility admission must run before calling this helper.
pub async fn native_integer_value_is_exact_f64_async(value: &Value) -> Result<bool, RuntimeError> {
    if native_integer_value_is_exact_f64(value) {
        return Ok(true);
    }
    if matches!(value, Value::GpuTensor(handle) if handle_integer_type(handle).is_some()) {
        let Value::GpuTensor(handle) = value else {
            unreachable!();
        };
        let provider = crate::builtins::common::gpu_helpers::exact_provider_for_handle(handle)
            .ok_or_else(|| {
                build_runtime_error(
                    "integer exactness check: no acceleration provider owns the input handle",
                )
                .with_gpu_gather_retry(crate::GpuGatherRetry::Never)
                .build()
            })?;
        let expected_type = handle_integer_type(handle).expect("integer type checked above");
        if runmat_accelerate_api::handle_storage(handle)
            != runmat_accelerate_api::GpuTensorStorage::Real
            || handle_is_logical(handle)
            || runmat_accelerate_api::handle_precision(handle).is_some()
            || !crate::builtins::common::gpu_helpers::gpu_class_metadata_matches(
                handle,
                None,
                Some(expected_type),
                false,
            )
        {
            return Err(build_runtime_error(
                "integer exactness check: input handle has contradictory integer metadata",
            )
            .with_gpu_gather_retry(crate::GpuGatherRetry::Never)
            .build());
        }
        let snapshot = crate::builtins::common::gpu_helpers::snapshot_handle_metadata(handle);
        let gathered = provider.download_integer(handle).await.map_err(|error| {
            build_runtime_error(format!(
                "integer exactness check: owner-preserving download failed: {error}"
            ))
            .with_gpu_gather_retry(crate::GpuGatherRetry::Never)
            .build()
        });
        crate::builtins::common::gpu_helpers::restore_handle_metadata(handle, &snapshot);
        let gathered = gathered?;
        if gathered.shape != handle.shape || gathered.data.element_type() != expected_type {
            return Err(build_runtime_error(
                "integer exactness check: provider returned contradictory integer payload metadata",
            )
            .with_gpu_gather_retry(crate::GpuGatherRetry::Never)
            .build());
        }
        let exact = crate::builtins::math::trigonometry::cos::integer_is_exact_f64;
        let values_exact = match &gathered.data {
            runmat_accelerate_api::HostIntegerDataOwned::I8(values) => values
                .iter()
                .all(|value| exact(&IntValue::I64(i64::from(*value)))),
            runmat_accelerate_api::HostIntegerDataOwned::I16(values) => values
                .iter()
                .all(|value| exact(&IntValue::I64(i64::from(*value)))),
            runmat_accelerate_api::HostIntegerDataOwned::I32(values) => values
                .iter()
                .all(|value| exact(&IntValue::I64(i64::from(*value)))),
            runmat_accelerate_api::HostIntegerDataOwned::I64(values) => {
                values.iter().all(|value| exact(&IntValue::I64(*value)))
            }
            runmat_accelerate_api::HostIntegerDataOwned::U8(values) => values
                .iter()
                .all(|value| exact(&IntValue::U64(u64::from(*value)))),
            runmat_accelerate_api::HostIntegerDataOwned::U16(values) => values
                .iter()
                .all(|value| exact(&IntValue::U64(u64::from(*value)))),
            runmat_accelerate_api::HostIntegerDataOwned::U32(values) => values
                .iter()
                .all(|value| exact(&IntValue::U64(u64::from(*value)))),
            runmat_accelerate_api::HostIntegerDataOwned::U64(values) => {
                values.iter().all(|value| exact(&IntValue::U64(*value)))
            }
        };
        return Ok(values_exact);
    }
    Ok(false)
}

/// Gate a native-integer RunMat extension before provider lookup or gathering,
/// then prove that its authoritative values can cross a binary64 computation
/// boundary without rounding.
pub async fn ensure_runmat_integer_f64_boundary(
    value: &Value,
    extension: &'static BuiltinExtensionDescriptor,
    builtin: &'static str,
    role: &str,
) -> BuiltinResult<()> {
    if !value_has_native_integer_class(value) {
        return Ok(());
    }
    crate::compatibility::ensure_builtin_extension_enabled(extension, builtin)?;
    if !native_integer_value_is_exact_f64_async(value).await? {
        return Err(build_runtime_error(format!(
            "{builtin}: integer {role} values must be exactly representable as double"
        ))
        .with_builtin(builtin)
        .with_gpu_gather_retry(crate::GpuGatherRetry::Never)
        .build());
    }
    Ok(())
}

pub fn must_be_a(value: &Value, class_names: Vec<String>) -> Result<bool, RuntimeError> {
    Ok(class_names
        .iter()
        .any(|class_name| value_matches_class(value, class_name)))
}

pub fn value_underlying_type_matches(
    value: &Value,
    class_names: Vec<String>,
) -> Result<bool, RuntimeError> {
    Ok(class_names
        .iter()
        .any(|class_name| underlying_type_matches(value, class_name)))
}

pub fn value_is_member(value: &Value, set: &Value) -> Result<bool, RuntimeError> {
    let values = atoms(value)?;
    let allowed = atoms(set)?;
    value_is_member_atoms_inner(&values, &allowed)
}

pub async fn value_is_member_async(value: &Value, set: &Value) -> BuiltinResult<bool> {
    let value = host_value(value).await?;
    let set = host_value(set).await?;
    ensure_member_class_compatibility(&value, &set)?;
    value_is_member(&value, &set).map_err(Into::into)
}

fn ensure_member_class_compatibility(value: &Value, set: &Value) -> BuiltinResult<()> {
    let Some(value_class) = numeric_class_name(value) else {
        return Ok(());
    };
    let Some(set_class) = numeric_class_name(set) else {
        return Ok(());
    };
    if value_class == set_class || value_class == "double" || set_class == "double" {
        return Ok(());
    }
    Err(invalid_argument_error(
        "mustBeMember",
        "unlike nondouble numeric inputs must have the same class",
    )
    .into())
}

pub fn value_is_member_atoms(
    value: &Value,
    allowed: &[ValidationAtom],
) -> Result<bool, RuntimeError> {
    let values = atoms(value)?;
    value_is_member_atoms_inner(&values, allowed)
}

pub async fn value_is_member_atoms_async(
    value: &Value,
    allowed: &[ValidationAtom],
) -> BuiltinResult<bool> {
    let value = host_value(value).await?;
    value_is_member_atoms(&value, allowed).map_err(Into::into)
}

fn value_is_member_atoms_inner(
    values: &[ValidationAtom],
    allowed: &[ValidationAtom],
) -> Result<bool, RuntimeError> {
    Ok(values
        .iter()
        .all(|value| allowed.iter().any(|allowed| atom_eq(value, allowed))))
}

pub fn atoms(value: &Value) -> Result<Vec<ValidationAtom>, RuntimeError> {
    match value {
        Value::Num(v) => Ok(vec![ValidationAtom::Number(*v)]),
        Value::Int(v) => Ok(vec![ValidationAtom::Integer(v.clone())]),
        Value::Bool(v) => Ok(vec![ValidationAtom::Bool(*v)]),
        Value::Complex(re, im) => Ok(vec![ValidationAtom::ComplexNumber(*re, *im)]),
        Value::String(s) => Ok(vec![ValidationAtom::Text(s.clone())]),
        Value::CharArray(c) if c.rows == 1 => Ok(vec![ValidationAtom::Text(chars_to_string(c))]),
        Value::Tensor(t) => {
            if let Some(storage) = t.integer_storage() {
                return Ok(storage
                    .exact_values()
                    .into_iter()
                    .map(ValidationAtom::Integer)
                    .collect());
            }
            Ok(tensor::tensor_values_f64_cow(t)
                .iter()
                .copied()
                .map(ValidationAtom::Number)
                .collect())
        }
        Value::ComplexTensor(t) => {
            if let Some(storage) = t.integer_storage() {
                return Ok((0..storage.len())
                    .map(|index| {
                        ValidationAtom::ComplexInteger(
                            storage
                                .real
                                .value_at(index)
                                .expect("validated real storage"),
                            storage
                                .imag
                                .value_at(index)
                                .expect("validated imaginary storage"),
                        )
                    })
                    .collect());
            }
            Ok(t.materialize_f64()
                .iter()
                .map(|(re, im)| ValidationAtom::ComplexNumber(*re, *im))
                .collect())
        }
        Value::SparseTensor(t) => sparse_atoms(t),
        Value::LogicalArray(a) => Ok(a
            .data
            .iter()
            .map(|v| ValidationAtom::Bool(*v != 0))
            .collect()),
        Value::StringArray(s) => Ok(s.data.iter().cloned().map(ValidationAtom::Text).collect()),
        Value::Cell(c) => {
            let mut out = Vec::new();
            for entry in &c.data {
                out.extend(atoms(entry)?);
            }
            Ok(out)
        }
        _ => Err(invalid_argument_error(
            "mustBeMember",
            "unsupported member value type",
        )),
    }
}

fn sparse_atoms(t: &SparseTensor) -> Result<Vec<ValidationAtom>, RuntimeError> {
    let numel = t.rows.saturating_mul(t.cols);
    let mut out = Vec::with_capacity(numel.min(t.nnz().saturating_add(1)));
    if let Some(storage) = t.integer_storage() {
        out.extend(
            storage
                .exact_values()
                .into_iter()
                .map(ValidationAtom::Integer),
        );
        if storage.len() < numel {
            let zero = storage
                .zeros_like(1)
                .value_at(0)
                .expect("one-element zero storage");
            out.push(ValidationAtom::Integer(zero));
        }
        return Ok(out);
    }
    let values = t.materialize_f64();
    out.extend(values.iter().copied().map(ValidationAtom::Number));
    if values.len() < numel {
        out.push(ValidationAtom::Number(0.0));
    }
    Ok(out)
}

fn atom_eq(left: &ValidationAtom, right: &ValidationAtom) -> bool {
    match (left, right) {
        (ValidationAtom::Number(a), ValidationAtom::Number(b)) => a == b,
        (ValidationAtom::Integer(a), ValidationAtom::Integer(b)) => a == b,
        (ValidationAtom::Integer(a), ValidationAtom::Number(b))
        | (ValidationAtom::Number(b), ValidationAtom::Integer(a)) => {
            integer_f64_order(a.clone(), *b) == Some(Ordering::Equal)
        }
        (ValidationAtom::ComplexNumber(ar, ai), ValidationAtom::ComplexNumber(br, bi)) => {
            ar == br && ai == bi
        }
        (ValidationAtom::ComplexInteger(ar, ai), ValidationAtom::ComplexInteger(br, bi)) => {
            ar == br && ai == bi
        }
        (ValidationAtom::ComplexInteger(ar, ai), ValidationAtom::ComplexNumber(br, bi))
        | (ValidationAtom::ComplexNumber(br, bi), ValidationAtom::ComplexInteger(ar, ai)) => {
            integer_f64_order(ar.clone(), *br) == Some(Ordering::Equal)
                && integer_f64_order(ai.clone(), *bi) == Some(Ordering::Equal)
        }
        (ValidationAtom::ComplexNumber(re, im), ValidationAtom::Number(value))
        | (ValidationAtom::Number(value), ValidationAtom::ComplexNumber(re, im)) => {
            *im == 0.0 && re == value
        }
        (ValidationAtom::ComplexInteger(re, im), ValidationAtom::Integer(value))
        | (ValidationAtom::Integer(value), ValidationAtom::ComplexInteger(re, im)) => {
            im.is_zero() && re == value
        }
        (ValidationAtom::ComplexInteger(re, im), ValidationAtom::Number(value))
        | (ValidationAtom::Number(value), ValidationAtom::ComplexInteger(re, im)) => {
            im.is_zero() && integer_f64_order(re.clone(), *value) == Some(Ordering::Equal)
        }
        (ValidationAtom::ComplexNumber(re, im), ValidationAtom::Integer(value))
        | (ValidationAtom::Integer(value), ValidationAtom::ComplexNumber(re, im)) => {
            *im == 0.0 && integer_f64_order(value.clone(), *re) == Some(Ordering::Equal)
        }
        (ValidationAtom::Text(a), ValidationAtom::Text(b)) => a == b,
        (ValidationAtom::Bool(a), ValidationAtom::Bool(b)) => a == b,
        (ValidationAtom::Bool(value), ValidationAtom::Number(number))
        | (ValidationAtom::Number(number), ValidationAtom::Bool(value)) => {
            *number == f64::from(*value)
        }
        (ValidationAtom::Bool(value), ValidationAtom::Integer(integer))
        | (ValidationAtom::Integer(integer), ValidationAtom::Bool(value)) => {
            integer_f64_order(integer.clone(), f64::from(*value)) == Some(Ordering::Equal)
        }
        _ => false,
    }
}

fn complex_real_values_all(
    value: &Value,
    float_pred: impl Fn(f64) -> bool + Copy,
    integer_pred: impl Fn(&IntValue) -> bool + Copy,
) -> Option<bool> {
    match value {
        Value::Complex(real, _) => Some(float_pred(*real)),
        Value::ComplexTensor(tensor) => {
            if let Some(storage) = tensor.integer_storage() {
                Some(integer_storage_all(&storage.real, integer_pred))
            } else {
                Some(
                    tensor
                        .materialize_f64()
                        .iter()
                        .all(|(real, _)| float_pred(*real)),
                )
            }
        }
        _ => None,
    }
}

fn exact_integer_values_all(
    value: &Value,
    pred: impl Fn(&IntValue) -> bool + Copy,
) -> Option<bool> {
    match value {
        Value::Int(value) => Some(pred(value)),
        Value::Tensor(tensor) => tensor
            .integer_storage()
            .map(|storage| integer_storage_all(storage, pred)),
        Value::SparseTensor(tensor) => tensor.integer_storage().map(|storage| {
            let numel = tensor.rows.saturating_mul(tensor.cols);
            integer_storage_all(storage, pred)
                && (storage.len() >= numel || integer_storage_zero_satisfies(storage, pred))
        }),
        Value::ComplexTensor(tensor) => tensor.integer_storage().map(|storage| {
            (0..storage.len()).all(|index| {
                let Some(real) = storage.real.value_at(index) else {
                    return false;
                };
                let Some(imag) = storage.imag.value_at(index) else {
                    return false;
                };
                imag.is_zero() && pred(&real)
            })
        }),
        _ => None,
    }
}

fn integer_storage_all(storage: &IntegerStorage, pred: impl Fn(&IntValue) -> bool + Copy) -> bool {
    (0..storage.len()).all(|index| {
        storage
            .value_at(index)
            .as_ref()
            .is_some_and(|value| pred(value))
    })
}

fn integer_storage_zero_satisfies(
    storage: &IntegerStorage,
    pred: impl Fn(&IntValue) -> bool,
) -> bool {
    storage.zeros_like(1).value_at(0).as_ref().is_some_and(pred)
}

fn int_f64_matches(integer: &IntValue, threshold: f64, pred: impl Fn(Ordering) -> bool) -> bool {
    integer_f64_order(integer.clone(), threshold).is_some_and(pred)
}

fn int_is_positive(value: &IntValue) -> bool {
    match value {
        IntValue::I8(value) => *value > 0,
        IntValue::I16(value) => *value > 0,
        IntValue::I32(value) => *value > 0,
        IntValue::I64(value) => *value > 0,
        IntValue::U8(value) => *value > 0,
        IntValue::U16(value) => *value > 0,
        IntValue::U32(value) => *value > 0,
        IntValue::U64(value) => *value > 0,
    }
}

fn int_is_negative(value: &IntValue) -> bool {
    match value {
        IntValue::I8(value) => *value < 0,
        IntValue::I16(value) => *value < 0,
        IntValue::I32(value) => *value < 0,
        IntValue::I64(value) => *value < 0,
        IntValue::U8(_) | IntValue::U16(_) | IntValue::U32(_) | IntValue::U64(_) => false,
    }
}

fn int_is_nonnegative(value: &IntValue) -> bool {
    !int_is_negative(value)
}

fn int_is_nonpositive(value: &IntValue) -> bool {
    !int_is_positive(value)
}

fn numeric_values_all(value: &Value, pred: impl Fn(f64) -> bool) -> bool {
    match value {
        Value::Num(v) => pred(*v),
        Value::Int(v) => pred(v.to_f64()),
        Value::Bool(v) => pred(if *v { 1.0 } else { 0.0 }),
        Value::LogicalArray(a) => a.data.iter().map(|v| f64::from(*v != 0)).all(pred),
        Value::Tensor(t) => tensor::tensor_values_f64(t).into_iter().all(pred),
        Value::SparseTensor(t) if t.integer_storage().is_some() => {
            let storage = t
                .integer_storage()
                .expect("integer storage was checked above");
            let numel = t.rows.saturating_mul(t.cols);
            (0..storage.len()).all(|index| {
                pred(
                    storage
                        .value_at(index)
                        .expect("sparse integer storage length is consistent")
                        .to_f64(),
                )
            }) && (storage.len() >= numel || pred(0.0))
        }
        Value::SparseTensor(t) => {
            let numel = t.rows.saturating_mul(t.cols);
            let values = t.materialize_f64();
            values.iter().copied().all(&pred) && (values.len() >= numel || pred(0.0))
        }
        Value::Complex(re, im) => *im == 0.0 && pred(*re),
        Value::ComplexTensor(t) if t.integer_storage().is_some() => {
            let storage = t
                .integer_storage()
                .expect("integer storage was checked above");
            (0..storage.len()).all(|index| {
                let real = storage
                    .real
                    .value_at(index)
                    .expect("complex integer real storage length is consistent");
                let imag = storage
                    .imag
                    .value_at(index)
                    .expect("complex integer imaginary storage length is consistent");
                imag.is_zero() && pred(real.to_f64())
            })
        }
        Value::ComplexTensor(t) => t
            .materialize_f64()
            .iter()
            .all(|(re, im)| *im == 0.0 && pred(*re)),
        _ => false,
    }
}

fn type_names_arg(args: &[Value], index: usize) -> Result<Vec<String>, RuntimeError> {
    match args.get(index) {
        Some(value) => value_texts(value),
        None => Err(invalid_argument_error(
            "argumentValidation",
            "missing type name argument",
        )),
    }
}

fn range_inclusivity_arg(builtin: &str, args: &[Value]) -> Result<RangeInclusivity, RuntimeError> {
    match args {
        [] => Ok(RangeInclusivity::CLOSED),
        [flag] => range_inclusivity_single_flag(builtin, text_scalar_arg(builtin, flag)?.as_str()),
        [lower, upper] => {
            let lower =
                range_bound_inclusive_flag(builtin, text_scalar_arg(builtin, lower)?.as_str())?;
            let upper =
                range_bound_inclusive_flag(builtin, text_scalar_arg(builtin, upper)?.as_str())?;
            Ok(RangeInclusivity { lower, upper })
        }
        _ => Err(invalid_argument_error(
            builtin,
            "invalid range inclusivity flags",
        )),
    }
}

fn range_inclusivity_single_flag(
    builtin: &str,
    flag: &str,
) -> Result<RangeInclusivity, RuntimeError> {
    match flag.trim().to_ascii_lowercase().as_str() {
        "inclusive" => Ok(RangeInclusivity::CLOSED),
        "exclusive" => Ok(RangeInclusivity::OPEN),
        "exclude-lower" | "openleft" | "open-left" => Ok(RangeInclusivity::OPEN_LEFT),
        "exclude-upper" | "openright" | "open-right" => Ok(RangeInclusivity::OPEN_RIGHT),
        _ => Err(invalid_argument_error(
            builtin,
            "range flag must be 'inclusive', 'exclusive', 'exclude-lower', or 'exclude-upper'",
        )),
    }
}

fn range_bound_inclusive_flag(builtin: &str, flag: &str) -> Result<bool, RuntimeError> {
    match flag.trim().to_ascii_lowercase().as_str() {
        "inclusive" => Ok(true),
        "exclusive" => Ok(false),
        _ => Err(invalid_argument_error(
            builtin,
            "range bound flag must be 'inclusive' or 'exclusive'",
        )),
    }
}

fn text_scalar_arg(builtin: &str, value: &Value) -> Result<String, RuntimeError> {
    let texts = value_texts(value)?;
    match texts.as_slice() {
        [text] => Ok(text.clone()),
        _ => Err(invalid_argument_error(builtin, "expected text scalar")),
    }
}

fn value_texts(value: &Value) -> Result<Vec<String>, RuntimeError> {
    match value {
        Value::String(s) => Ok(vec![s.clone()]),
        Value::StringArray(s) => Ok(s.data.clone()),
        Value::CharArray(c) if c.rows == 1 => Ok(vec![chars_to_string(c)]),
        Value::Cell(c) => {
            let mut out = Vec::with_capacity(c.data.len());
            for entry in &c.data {
                out.extend(value_texts(entry)?);
            }
            Ok(out)
        }
        other => Err(invalid_argument_error(
            "argumentValidation",
            format!("expected text, got {}", class_name_for_value(other)),
        )),
    }
}

fn chars_to_string(chars: &CharArray) -> String {
    chars.data.iter().collect()
}

pub fn isvarname_value(value: &Value) -> bool {
    value_texts(value)
        .map(|names| names.iter().all(|name| is_valid_varname(name)))
        .unwrap_or(false)
}

pub fn namedargs2cell_value(value: Value) -> BuiltinResult<Value> {
    let Value::Struct(struct_value) = value else {
        return Err(
            invalid_argument_error("namedargs2cell", "input must be a scalar struct").into(),
        );
    };
    let mut data = Vec::with_capacity(struct_value.fields.len().saturating_mul(2));
    for (field, value) in struct_value.fields {
        data.push(Value::String(field));
        data.push(value);
    }
    let cols = data.len();
    let cell = CellArray::new(data, 1, cols)
        .map_err(|err| invalid_argument_error("namedargs2cell", err))?;
    Ok(Value::Cell(cell))
}

pub fn validate_function_signatures_json(value: &Value) -> BuiltinResult<()> {
    for text in value_texts(value)? {
        serde_json::from_str::<serde_json::Value>(&text).map_err(|err| {
            invalid_argument_error(
                "validateFunctionSignaturesJSON",
                format!("invalid JSON signature payload: {err}"),
            )
        })?;
    }
    Ok(())
}

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
mod tests {
    use super::*;
    use crate::builtins::common::identifiers::MATLAB_NAME_LENGTH_MAX;
    use crate::builtins::common::test_support;
    use runmat_value::{
        ComplexTensor, IntValue, IntegerComplexStorage, IntegerStorage, LogicalArray, StringArray,
        StructValue, Tensor,
    };

    fn ok(builtin: &str, args: Vec<Value>) {
        dispatch_validator(builtin, args).unwrap_or_else(|err| {
            panic!("{builtin} unexpectedly failed: {err}");
        });
    }

    fn err(builtin: &str, args: Vec<Value>) {
        assert!(
            dispatch_validator(builtin, args).is_err(),
            "{builtin} unexpectedly passed"
        );
    }

    fn tensor(data: Vec<f64>, rows: usize, cols: usize) -> Value {
        Value::Tensor(Tensor::new_2d(data, rows, cols).unwrap())
    }

    fn sparse(values: Vec<f64>) -> Value {
        Value::SparseTensor(SparseTensor::new(2, 2, vec![0, 1, 1], vec![0], values).unwrap())
    }

    #[test]
    fn resident_integer_exactness_is_class_conservative() {
        use runmat_accelerate_api::{GpuTensorHandle, IntegerElementType};

        let narrow = GpuTensorHandle {
            shape: vec![1, 1],
            device_id: u32::MAX,
            buffer_id: u64::MAX - 1,
            descriptor: Default::default(),
        }
        .with_numeric_descriptor(
            IntegerElementType::U32.into(),
            runmat_accelerate_api::GpuTensorStorage::Real,
        );
        assert!(native_integer_value_is_exact_f64(&Value::GpuTensor(narrow)));
        let wide = GpuTensorHandle {
            shape: vec![1, 1],
            device_id: u32::MAX,
            buffer_id: u64::MAX - 2,
            descriptor: Default::default(),
        }
        .with_numeric_descriptor(
            IntegerElementType::I64.into(),
            runmat_accelerate_api::GpuTensorStorage::Real,
        );
        assert!(!native_integer_value_is_exact_f64(&Value::GpuTensor(wide)));
    }

    #[test]
    fn resident_wide_integer_exactness_is_decided_from_gathered_values() {
        use crate::builtins::common::test_support;
        use futures::executor::block_on;
        use runmat_accelerate_api::{HostIntegerDataView, HostIntegerTensorView};

        test_support::with_test_provider(|provider| {
            for (value, expected) in [
                (9_007_199_254_740_992_u64, true),
                (9_007_199_254_740_993_u64, false),
            ] {
                let handle = provider
                    .upload_integer(&HostIntegerTensorView {
                        data: HostIntegerDataView::U64(std::slice::from_ref(&value)),
                        shape: &[1, 1],
                    })
                    .expect("upload resident uint64");
                assert_eq!(
                    block_on(native_integer_value_is_exact_f64_async(&Value::GpuTensor(
                        handle.clone()
                    )))
                    .expect("exactness check"),
                    expected
                );
                provider.free(&handle).expect("free resident uint64");
                runmat_accelerate_api::clear_handle_metadata(&handle);
            }
        });
    }

    fn integer_tensor(storage: IntegerStorage, shape: Vec<usize>) -> Value {
        Value::Tensor(Tensor::new_integer(storage, shape).unwrap())
    }

    #[test]
    fn structural_validators_cover_all_native_integer_classes_without_conversion() {
        let cases = [
            (IntegerStorage::I8(vec![1]), "int8"),
            (IntegerStorage::I16(vec![1]), "int16"),
            (IntegerStorage::I32(vec![1]), "int32"),
            (IntegerStorage::I64(vec![1]), "int64"),
            (IntegerStorage::U8(vec![1]), "uint8"),
            (IntegerStorage::U16(vec![1]), "uint16"),
            (IntegerStorage::U32(vec![1]), "uint32"),
            (IntegerStorage::U64(vec![1]), "uint64"),
        ];

        for (storage, class_name) in cases {
            let value = integer_tensor(storage, vec![1, 1]);
            ok("mustBeReal", vec![value.clone()]);
            ok("mustBeScalarOrEmpty", vec![value.clone()]);
            err("mustBeSparse", vec![value.clone()]);
            ok(
                "mustBeUnderlyingType",
                vec![value.clone(), Value::String(class_name.into())],
            );
            ok("mustBeVector", vec![value.clone()]);
            err("mustBeText", vec![value.clone()]);
            err("mustBeTextScalar", vec![value.clone()]);
            err("mustBeValidVariableName", vec![value]);
        }
    }

    #[test]
    fn sparse_and_vector_validators_apply_documented_empty_shape_rules() {
        let empty = integer_tensor(IntegerStorage::U16(vec![]), vec![0, 3]);
        ok("mustBeScalarOrEmpty", vec![empty.clone()]);
        ok("mustBeSparse", vec![empty.clone()]);
        err("mustBeVector", vec![empty.clone()]);
        ok(
            "mustBeVector",
            vec![empty, Value::String("allow-all-empties".into())],
        );

        let empty_vector = integer_tensor(IntegerStorage::I8(vec![]), vec![0, 1]);
        ok("mustBeVector", vec![empty_vector]);

        let multidimensional = integer_tensor(IntegerStorage::U32(vec![1, 2]), vec![1, 1, 2]);
        err("mustBeVector", vec![multidimensional]);
        let trailing_singleton = integer_tensor(IntegerStorage::U32(vec![1, 2]), vec![1, 2, 1]);
        ok("mustBeVector", vec![trailing_singleton]);

        err(
            "mustBeVector",
            vec![
                Value::Int(IntValue::U8(1)),
                Value::String("unsupported".into()),
            ],
        );
    }

    #[test]
    fn sparse_integer_storage_satisfies_sparse_validation_without_materialization() {
        let sparse = SparseTensor::new_integer(
            2,
            2,
            vec![0, 1, 1],
            vec![0],
            IntegerStorage::I64(vec![9_007_199_254_740_993]),
        )
        .expect("sparse integer");
        ok("mustBeSparse", vec![Value::SparseTensor(sparse)]);
    }

    #[test]
    fn resident_text_validators_reject_before_provider_lookup() {
        use runmat_accelerate_api::{GpuTensorHandle, IntegerElementType};

        let handle = GpuTensorHandle {
            shape: vec![1, 1],
            device_id: u32::MAX - 1,
            buffer_id: u64::MAX - 2,
            descriptor: Default::default(),
        }
        .with_numeric_descriptor(
            IntegerElementType::I32.into(),
            runmat_accelerate_api::GpuTensorStorage::Real,
        );
        for builtin in ["mustBeText", "mustBeTextScalar", "mustBeValidVariableName"] {
            let error = dispatch_validator(builtin, vec![Value::GpuTensor(handle.clone())])
                .expect_err("resident integer must fail text validation");
            let expected_identifier = format!("RunMat:{builtin}:ValidationFailed");
            assert_eq!(error.identifier(), Some(expected_identifier.as_str()));
            assert_eq!(error.gpu_gather_retry(), crate::GpuGatherRetry::Never);
        }
        runmat_accelerate_api::clear_handle_metadata(&handle);
    }

    #[test]
    fn resident_integer_structural_validators_do_not_read_freed_payloads() {
        test_support::with_test_provider(|provider| {
            let tensor = Tensor::new_integer(IntegerStorage::U64(vec![1, u64::MAX]), vec![1, 2])
                .expect("resident integer");
            let handle = crate::builtins::common::gpu_helpers::upload_tensor(provider, &tensor)
                .expect("upload resident integer");
            provider
                .free(&handle)
                .expect("free payload before predicates");
            runmat_accelerate_api::set_handle_logical(&handle, false);
            let value = Value::GpuTensor(handle.clone());

            ok("mustBeReal", vec![value.clone()]);
            err("mustBeScalarOrEmpty", vec![value.clone()]);
            err("mustBeSparse", vec![value.clone()]);
            ok(
                "mustBeUnderlyingType",
                vec![value.clone(), Value::String("uint64".into())],
            );
            ok("mustBeVector", vec![value]);
            runmat_accelerate_api::clear_handle_metadata(&handle);
        });
    }

    #[test]
    fn numeric_validators_check_all_elements() {
        let ok = Tensor::new(vec![1.0, 2.0], vec![1, 2]).unwrap();
        assert!(dispatch_validator("mustBePositive", vec![Value::Tensor(ok)]).is_ok());

        let bad = Tensor::new(vec![1.0, 0.0], vec![1, 2]).unwrap();
        assert!(dispatch_validator("mustBePositive", vec![Value::Tensor(bad)]).is_err());
    }

    #[test]
    fn numeric_validators_read_typed_integer_storage_exactly() {
        let positive =
            Tensor::new_integer(IntegerStorage::U16(vec![1, 2]), vec![1, 2]).expect("positive");
        ok("mustBePositive", vec![Value::Tensor(positive)]);

        let negative =
            Tensor::new_integer(IntegerStorage::I16(vec![-1, -2]), vec![1, 2]).expect("negative");
        ok("mustBeNegative", vec![Value::Tensor(negative)]);

        let zero = Tensor::new_integer(IntegerStorage::I16(vec![0]), vec![1, 1]).expect("zero");
        err("mustBeNonzero", vec![Value::Tensor(zero)]);

        let wide = 9_007_199_254_740_993_u64;
        let adjacent = wide - 1;
        assert_eq!(wide as f64, adjacent as f64);
        let wide_nonzero =
            Tensor::new_integer(IntegerStorage::U64(vec![wide]), vec![1, 1]).expect("wide");
        ok("mustBeNonzero", vec![Value::Tensor(wide_nonzero)]);

        let complex_nonzero = ComplexTensor::new_integer(
            IntegerComplexStorage::new(
                IntegerStorage::U64(vec![0]),
                IntegerStorage::U64(vec![wide]),
            )
            .expect("complex integer storage"),
            vec![1, 1],
        )
        .expect("complex integer tensor");
        ok("mustBeNonzero", vec![Value::ComplexTensor(complex_nonzero)]);
    }

    #[test]
    fn value_is_empty_uses_typed_integer_storage_length() {
        let scalar = Tensor::new_integer(IntegerStorage::U16(vec![7]), vec![1, 1]).expect("scalar");
        assert!(!value_is_empty(&Value::Tensor(scalar)));

        let empty =
            Tensor::new_integer(IntegerStorage::U16(Vec::new()), vec![0, 0]).expect("empty tensor");
        assert!(value_is_empty(&Value::Tensor(empty)));

        let complex = ComplexTensor::new_integer(
            IntegerComplexStorage::new(IntegerStorage::I16(vec![1]), IntegerStorage::I16(vec![0]))
                .expect("complex integer storage"),
            vec![1, 1],
        )
        .expect("complex integer tensor");
        assert!(!value_is_empty(&Value::ComplexTensor(complex)));

        let empty_complex = ComplexTensor::new_integer(
            IntegerComplexStorage::new(IntegerStorage::I16(vec![]), IntegerStorage::I16(vec![]))
                .expect("empty complex integer storage"),
            vec![0, 0],
        )
        .expect("empty complex integer tensor");
        assert!(value_is_empty(&Value::ComplexTensor(empty_complex)));
    }

    #[test]
    fn finite_integer_and_nan_predicates_read_typed_integer_storage_exactly() {
        let tensor = Tensor::new_integer(IntegerStorage::I16(vec![-1, 0, 2]), vec![1, 3]).unwrap();
        let value = Value::Tensor(tensor);

        assert!(value_is_finite(&value));
        assert!(value_is_integer(&value));
        assert!(value_is_non_nan(&value));
        ok("mustBeFinite", vec![value.clone()]);
        ok("mustBeInteger", vec![value.clone()]);
        ok("mustBeNonNan", vec![value]);

        let sparse = Value::SparseTensor(
            SparseTensor::new_integer(
                2,
                2,
                vec![0, 1, 2],
                vec![0, 1],
                IntegerStorage::U8(vec![1, 2]),
            )
            .unwrap(),
        );
        assert!(value_is_finite(&sparse));
        assert!(value_is_integer(&sparse));
        assert!(value_is_non_nan(&sparse));
    }

    #[test]
    fn real_and_integer_predicates_read_authoritative_complex_integer_storage() {
        let real_storage = IntegerStorage::I16(vec![1, -2]);
        let zero_imag = IntegerStorage::I16(vec![0, 0]);
        let real_complex = ComplexTensor::new_integer(
            IntegerComplexStorage::new(real_storage, zero_imag).unwrap(),
            vec![1, 2],
        )
        .unwrap();
        let value = Value::ComplexTensor(real_complex);
        assert!(value_is_finite(&value));
        assert!(value_is_real(&value));
        assert!(value_is_integer(&value));
        assert!(value_is_non_nan(&value));

        let nonreal_complex = ComplexTensor::new_integer(
            IntegerComplexStorage::new(IntegerStorage::I16(vec![1]), IntegerStorage::I16(vec![1]))
                .unwrap(),
            vec![1, 1],
        )
        .unwrap();
        let value = Value::ComplexTensor(nonreal_complex);
        assert!(!value_is_real(&value));
        assert!(value_is_integer(&value));
        ok("mustBeInteger", vec![value]);
    }

    #[test]
    fn complex_validators_use_component_integrality_real_ordering_and_exact_membership() {
        ok("mustBeInteger", vec![Value::Complex(1.0, 2.0)]);
        err("mustBeInteger", vec![Value::Complex(1.0, 2.5)]);
        ok("mustBeNegative", vec![Value::Complex(-1.0, 9.0)]);
        ok("mustBePositive", vec![Value::Complex(1.0, -9.0)]);
        ok(
            "mustBeMember",
            vec![Value::Complex(1.0, 2.0), Value::Complex(1.0, 2.0)],
        );

        let wide = Value::ComplexTensor(
            ComplexTensor::new_integer(
                IntegerComplexStorage::new(
                    IntegerStorage::U64(vec![9_007_199_254_740_993]),
                    IntegerStorage::U64(vec![7]),
                )
                .expect("integer components"),
                vec![1, 1],
            )
            .expect("complex integer tensor"),
        );
        ok("mustBeMember", vec![wide.clone(), wide]);
        ok("mustBeMember", vec![Value::Bool(true), Value::Num(1.0)]);
        ok("mustBeMember", vec![Value::Num(0.0), Value::Bool(false)]);
        ok(
            "mustBeInteger",
            vec![Value::CharArray(
                CharArray::new(vec!['a', 'b'], 1, 2).expect("character row"),
            )],
        );
        ok("mustBeNonzero", vec![Value::Num(f64::INFINITY)]);
        ok("mustBeNonzero", vec![Value::Num(f64::NAN)]);
        ok("mustBePositive", vec![Value::Num(f64::INFINITY)]);
        ok("mustBeNegative", vec![Value::Num(f64::NEG_INFINITY)]);
        ok(
            "mustBeNonnegative",
            vec![Value::Complex(f64::INFINITY, 7.0)],
        );
        ok(
            "mustBeNonpositive",
            vec![Value::Complex(f64::NEG_INFINITY, 7.0)],
        );
    }

    #[test]
    fn resident_secondary_validator_operands_require_coherent_metadata() {
        test_support::with_test_provider(|provider| {
            let value =
                Tensor::new_integer(IntegerStorage::U8(vec![2]), vec![1, 1]).expect("value");
            let lower =
                Tensor::new_integer(IntegerStorage::U8(vec![1]), vec![1, 1]).expect("lower");
            let value_handle =
                crate::builtins::common::gpu_helpers::upload_tensor(provider, &value)
                    .expect("upload value");
            let mut lower_handle =
                crate::builtins::common::gpu_helpers::upload_tensor(provider, &lower)
                    .expect("upload lower");
            lower_handle.descriptor.storage =
                Some(runmat_accelerate_api::GpuTensorStorage::ComplexInterleaved);
            let error = dispatch_validator(
                "mustBeGreaterThan",
                vec![
                    Value::GpuTensor(value_handle.clone()),
                    Value::GpuTensor(lower_handle.clone()),
                ],
            )
            .expect_err("contradictory bound metadata must reject");
            assert_eq!(
                error.identifier(),
                Some("RunMat:validators:ProviderPayloadMismatch")
            );
            provider.free(&value_handle).ok();
            provider.free(&lower_handle).ok();
            runmat_accelerate_api::clear_handle_metadata(&value_handle);
            runmat_accelerate_api::clear_handle_metadata(&lower_handle);
        });
    }

    #[test]
    fn threshold_validators_read_typed_sparse_and_complex_integer_storage() {
        let sparse =
            SparseTensor::new_integer(1, 2, vec![0, 1, 1], vec![0], IntegerStorage::U8(vec![1]))
                .unwrap();
        ok("mustBeNonnegative", vec![Value::SparseTensor(sparse)]);

        let complex = ComplexTensor::new_integer(
            IntegerComplexStorage::new(IntegerStorage::I16(vec![2]), IntegerStorage::I16(vec![0]))
                .unwrap(),
            vec![1, 1],
        )
        .unwrap();
        ok("mustBePositive", vec![Value::ComplexTensor(complex)]);
    }

    #[test]
    fn member_validator_accepts_numeric_and_text_sets() {
        let allowed = Tensor::new(vec![1.0, 3.0, 5.0], vec![1, 3]).unwrap();
        assert!(dispatch_validator(
            "mustBeMember",
            vec![Value::Num(3.0), Value::Tensor(allowed)]
        )
        .is_ok());

        let allowed = StringArray::new(vec!["on".into(), "off".into()], vec![1, 2]).unwrap();
        assert!(dispatch_validator(
            "mustBeMember",
            vec![Value::String("on".into()), Value::StringArray(allowed)]
        )
        .is_ok());
    }

    #[test]
    fn class_validators_use_native_integer_storage_metadata() {
        let dense = integer_tensor(
            IntegerStorage::U64(vec![u64::MAX, 9_007_199_254_740_993]),
            vec![1, 2],
        );
        ok(
            "mustBeA",
            vec![dense.clone(), Value::String("integer".into())],
        );
        err("mustBeFloat", vec![dense.clone()]);
        err("mustBeA", vec![dense, Value::String("double".into())]);

        let sparse = Value::SparseTensor(
            SparseTensor::new_integer(
                2,
                2,
                vec![0, 1, 1],
                vec![1],
                IntegerStorage::I64(vec![i64::MIN]),
            )
            .unwrap(),
        );
        ok(
            "mustBeA",
            vec![sparse.clone(), Value::String("integer".into())],
        );
        err("mustBeFloat", vec![sparse.clone()]);
        err("mustBeA", vec![sparse, Value::String("double".into())]);

        let typed_complex = Value::ComplexTensor(
            ComplexTensor::new_integer(
                IntegerComplexStorage::new(
                    IntegerStorage::I16(vec![1]),
                    IntegerStorage::I16(vec![2]),
                )
                .unwrap(),
                vec![1, 1],
            )
            .unwrap(),
        );
        ok(
            "mustBeA",
            vec![typed_complex.clone(), Value::String("integer".into())],
        );
        err("mustBeFloat", vec![typed_complex]);
    }

    #[test]
    fn class_validators_use_gpu_integer_metadata_without_gather() {
        use crate::builtins::common::test_support;
        use runmat_accelerate_api::{HostIntegerDataView, HostIntegerTensorView};

        test_support::with_test_provider(|provider| {
            let values = [u64::MAX, 9_007_199_254_740_993];
            let shape = [1usize, 2usize];
            let handle = provider
                .upload_integer(&HostIntegerTensorView {
                    data: HostIntegerDataView::U64(&values),
                    shape: &shape,
                })
                .expect("upload integer gpu tensor");
            let gpu = Value::GpuTensor(handle);

            ok(
                "mustBeA",
                vec![gpu.clone(), Value::String("integer".into())],
            );
            ok("mustBeInteger", vec![gpu.clone()]);
            err("mustBeFloat", vec![gpu.clone()]);
            err("mustBeA", vec![gpu, Value::String("double".into())]);
        });
    }

    #[test]
    fn must_be_a_distinguishes_explicit_gpuarray_from_automatic_residency() {
        test_support::with_test_provider(|provider| {
            let tensor = Tensor::new_integer(IntegerStorage::U16(vec![7]), vec![1, 1])
                .expect("integer source");
            let handle = crate::builtins::common::gpu_helpers::upload_tensor(provider, &tensor)
                .expect("upload integer source");
            let handle =
                handle.with_provenance(runmat_accelerate_api::GpuHandleProvenance::Automatic);
            ok(
                "mustBeA",
                vec![
                    Value::GpuTensor(handle.clone()),
                    Value::String("uint16".into()),
                ],
            );
            err(
                "mustBeA",
                vec![
                    Value::GpuTensor(handle.clone()),
                    Value::String("gpuArray".into()),
                ],
            );
            let handle =
                handle.with_provenance(runmat_accelerate_api::GpuHandleProvenance::Explicit);
            ok(
                "mustBeA",
                vec![
                    Value::GpuTensor(handle.clone()),
                    Value::String("gpuArray".into()),
                ],
            );
            err(
                "mustBeA",
                vec![
                    Value::GpuTensor(handle.clone()),
                    Value::String("uint16".into()),
                ],
            );
            provider.free(&handle).expect("free resident source");
            runmat_accelerate_api::clear_handle_metadata(&handle);
        });
    }

    #[test]
    fn member_validator_compares_native_integers_exactly() {
        let wide = 9_007_199_254_740_993_u64;
        let adjacent = wide - 1;
        assert_eq!(wide as f64, adjacent as f64);

        let allowed = Tensor::new_integer(IntegerStorage::U64(vec![wide]), vec![1, 1]).unwrap();
        ok(
            "mustBeMember",
            vec![
                Value::Int(IntValue::U64(wide)),
                Value::Tensor(allowed.clone()),
            ],
        );
        err(
            "mustBeMember",
            vec![Value::Int(IntValue::U64(adjacent)), Value::Tensor(allowed)],
        );

        let sparse_allowed = SparseTensor::new_integer(
            2,
            2,
            vec![0, 1, 1],
            vec![1],
            IntegerStorage::U64(vec![wide]),
        )
        .unwrap();
        ok(
            "mustBeMember",
            vec![
                Value::Int(IntValue::U64(0)),
                Value::SparseTensor(sparse_allowed),
            ],
        );
    }

    #[test]
    fn member_validator_does_not_equate_wide_integer_with_rounded_double() {
        let wide = 9_007_199_254_740_993_u64;
        let rounded = (wide - 1) as f64;
        assert_eq!(wide as f64, rounded);

        err(
            "mustBeMember",
            vec![Value::Int(IntValue::U64(wide)), Value::Num(rounded)],
        );
    }

    #[test]
    fn text_and_varname_validators_follow_core_shapes() {
        assert!(dispatch_validator(
            "mustBeNonzeroLengthText",
            vec![Value::CharArray(CharArray::new_row("alpha"))]
        )
        .is_ok());
        assert!(isvarname_value(&Value::String("alpha_1".into())));
        assert!(!isvarname_value(&Value::String("1alpha".into())));
    }

    #[test]
    fn isvarname_returns_false_for_all_integer_classes() {
        for value in [
            IntValue::I8(-1),
            IntValue::I16(-2),
            IntValue::I32(-3),
            IntValue::I64(i64::MIN),
            IntValue::U8(1),
            IntValue::U16(2),
            IntValue::U32(3),
            IntValue::U64(u64::MAX),
        ] {
            assert!(!isvarname_value(&Value::Int(value)));
        }
    }

    #[test]
    fn isvarname_returns_false_for_resident_integer_without_gather() {
        test_support::with_test_provider(|provider| {
            let tensor = Tensor::new_integer(IntegerStorage::U64(vec![u64::MAX]), vec![1, 1])
                .expect("integer tensor");
            let handle = crate::builtins::common::gpu_helpers::upload_tensor(provider, &tensor)
                .expect("upload integer");
            assert!(!isvarname_value(&Value::GpuTensor(handle)));
        });
    }

    #[test]
    fn namedargs2cell_preserves_field_order() {
        let mut st = StructValue::new();
        st.insert("Name", Value::String("Ada".into()));
        st.insert("Value", Value::Num(7.0));
        let out = namedargs2cell_value(Value::Struct(st)).expect("namedargs2cell");
        let Value::Cell(cell) = out else {
            panic!("expected cell");
        };
        assert_eq!(cell.rows, 1);
        assert_eq!(cell.cols, 4);
        assert_eq!(cell.data[0], Value::String("Name".into()));
        assert_eq!(cell.data[2], Value::String("Value".into()));
    }

    #[test]
    fn namedargs2cell_rejects_top_level_integers_and_preserves_integer_fields_exactly() {
        let cases = [
            IntValue::I8(i8::MIN),
            IntValue::I16(i16::MIN),
            IntValue::I32(i32::MIN),
            IntValue::I64(i64::MIN),
            IntValue::U8(u8::MAX),
            IntValue::U16(u16::MAX),
            IntValue::U32(u32::MAX),
            IntValue::U64(u64::MAX),
        ];
        for value in cases {
            assert!(namedargs2cell_value(Value::Int(value.clone())).is_err());
            let mut structure = StructValue::new();
            structure.insert("Exact", Value::Int(value.clone()));
            let output = namedargs2cell_value(Value::Struct(structure))
                .expect("scalar struct with integer field");
            let Value::Cell(cell) = output else {
                panic!("expected name-value cell");
            };
            assert_eq!(cell.data[1], Value::Int(value));
        }

        let resident = Value::GpuTensor(runmat_accelerate_api::GpuTensorHandle {
            shape: vec![1, 1],
            device_id: 0,
            buffer_id: 9_419_006,
            descriptor: Default::default(),
        });
        assert!(namedargs2cell_value(resident).is_err());
    }

    #[test]
    fn validator_surface_accepts_and_rejects_representative_values() {
        let temp_dir = tempfile::tempdir().unwrap();
        let file_path = temp_dir.path().join("data.txt");
        std::fs::write(&file_path, "ok").unwrap();
        let dir_text = Value::String(temp_dir.path().to_string_lossy().into_owned());
        let file_text = Value::String(file_path.to_string_lossy().into_owned());

        ok(
            "mustBeA",
            vec![Value::Num(1.0), Value::String("double".into())],
        );
        err(
            "mustBeA",
            vec![Value::String("x".into()), Value::String("double".into())],
        );
        ok("mustBeColumn", vec![tensor(vec![1.0, 2.0], 2, 1)]);
        err("mustBeColumn", vec![tensor(vec![1.0, 2.0], 1, 2)]);
        ok("mustBeFile", vec![file_text.clone()]);
        err("mustBeFile", vec![dir_text.clone()]);
        ok("mustBeFolder", vec![dir_text.clone()]);
        err("mustBeFolder", vec![file_text.clone()]);
        ok("mustBeFinite", vec![tensor(vec![1.0, 2.0], 1, 2)]);
        err("mustBeFinite", vec![Value::Num(f64::INFINITY)]);
        ok("mustBeFloat", vec![Value::Num(1.0)]);
        err("mustBeFloat", vec![Value::Int(IntValue::I32(1))]);
        ok("mustBeInteger", vec![tensor(vec![1.0, 2.0], 1, 2)]);
        ok("mustBeInteger", vec![Value::Bool(true)]);
        ok(
            "mustBeInteger",
            vec![Value::LogicalArray(
                LogicalArray::new(vec![1, 0], vec![1, 2]).unwrap(),
            )],
        );
        err("mustBeInteger", vec![Value::Num(1.5)]);
        ok("mustBeNumeric", vec![Value::Complex(1.0, 2.0)]);
        err("mustBeNumeric", vec![Value::String("1".into())]);
        ok(
            "mustBeNumericOrLogical",
            vec![Value::LogicalArray(
                LogicalArray::new(vec![1, 0], vec![1, 2]).unwrap(),
            )],
        );
        err("mustBeNumericOrLogical", vec![Value::String("true".into())]);
        ok("mustBeReal", vec![Value::Complex(1.0, 0.0)]);
        err("mustBeReal", vec![Value::Complex(1.0, 1.0)]);
        ok("mustBeVector", vec![tensor(vec![1.0, 2.0], 1, 2)]);
        err("mustBeVector", vec![tensor(vec![1.0, 2.0, 3.0, 4.0], 2, 2)]);
        ok("mustBeScalarOrEmpty", vec![Value::Num(1.0)]);
        err("mustBeScalarOrEmpty", vec![tensor(vec![1.0, 2.0], 1, 2)]);
        ok("mustBeSparse", vec![sparse(vec![1.0])]);
        err("mustBeSparse", vec![Value::Num(1.0)]);
        ok("mustBeNonsparse", vec![Value::Num(1.0)]);
        err("mustBeNonsparse", vec![sparse(vec![1.0])]);
        ok(
            "mustBeText",
            vec![Value::CharArray(CharArray::new_row("abc"))],
        );
        err("mustBeText", vec![Value::Num(1.0)]);
        ok("mustBeTextScalar", vec![Value::String("abc".into())]);
        err(
            "mustBeTextScalar",
            vec![Value::StringArray(
                StringArray::new_2d(vec!["a".into(), "b".into()], 1, 2).unwrap(),
            )],
        );
        ok(
            "mustBeNonzeroLengthText",
            vec![Value::StringArray(
                StringArray::new_2d(vec!["a".into(), "b".into()], 1, 2).unwrap(),
            )],
        );
        err(
            "mustBeNonzeroLengthText",
            vec![Value::String(String::new())],
        );
        ok("mustBeNonempty", vec![Value::String("x".into())]);
        err(
            "mustBeNonempty",
            vec![Value::StringArray(
                StringArray::new_2d(vec![], 0, 0).unwrap(),
            )],
        );
        ok("mustBeNonmissing", vec![Value::Num(1.0)]);
        err("mustBeNonmissing", vec![Value::Num(f64::NAN)]);
        ok("mustBeNonNan", vec![Value::Complex(1.0, 0.0)]);
        err("mustBeNonNan", vec![Value::Complex(f64::NAN, 0.0)]);
        ok(
            "mustBeUnderlyingType",
            vec![Value::Int(IntValue::I16(1)), Value::String("int16".into())],
        );
        err(
            "mustBeUnderlyingType",
            vec![Value::Bool(true), Value::String("double".into())],
        );
        ok(
            "mustBeValidVariableName",
            vec![Value::String("alpha_1".into())],
        );
        err(
            "mustBeValidVariableName",
            vec![Value::String("_alpha".into())],
        );
        ok(
            "mustBeMember",
            vec![
                Value::String("on".into()),
                Value::Cell(
                    CellArray::new(
                        vec![Value::String("on".into()), Value::String("off".into())],
                        1,
                        2,
                    )
                    .unwrap(),
                ),
            ],
        );
        err(
            "mustBeMember",
            vec![
                Value::String("bad".into()),
                Value::Cell(
                    CellArray::new(
                        vec![Value::String("on".into()), Value::String("off".into())],
                        1,
                        2,
                    )
                    .unwrap(),
                ),
            ],
        );
    }

    #[test]
    fn numeric_threshold_validators_cover_boundaries() {
        ok("mustBePositive", vec![Value::Num(1.0)]);
        ok("mustBePositive", vec![Value::Bool(true)]);
        err("mustBePositive", vec![Value::Num(0.0)]);
        ok("mustBeNegative", vec![Value::Num(-1.0)]);
        err("mustBeNegative", vec![Value::Num(0.0)]);
        ok("mustBeNonnegative", vec![Value::Num(0.0)]);
        ok(
            "mustBeNonnegative",
            vec![Value::LogicalArray(
                LogicalArray::new(vec![1, 0], vec![1, 2]).unwrap(),
            )],
        );
        err("mustBeNonnegative", vec![Value::Num(-1.0)]);
        ok("mustBeNonpositive", vec![Value::Num(0.0)]);
        err("mustBeNonpositive", vec![Value::Num(1.0)]);
        ok("mustBeNonzero", vec![Value::Complex(0.0, 1.0)]);
        err("mustBeNonzero", vec![Value::Num(0.0)]);
        ok("mustBeGreaterThan", vec![Value::Num(2.0), Value::Num(1.0)]);
        err("mustBeGreaterThan", vec![Value::Num(1.0), Value::Num(1.0)]);
        ok(
            "mustBeGreaterThanOrEqual",
            vec![Value::Num(1.0), Value::Num(1.0)],
        );
        err(
            "mustBeGreaterThanOrEqual",
            vec![Value::Num(0.0), Value::Num(1.0)],
        );
        ok("mustBeLessThan", vec![Value::Num(0.0), Value::Num(1.0)]);
        err("mustBeLessThan", vec![Value::Num(1.0), Value::Num(1.0)]);
        ok(
            "mustBeLessThanOrEqual",
            vec![Value::Num(1.0), Value::Num(1.0)],
        );
        err(
            "mustBeLessThanOrEqual",
            vec![Value::Num(2.0), Value::Num(1.0)],
        );
    }

    #[test]
    fn threshold_validators_read_typed_integer_storage_exactly() {
        let lower = Tensor::new_integer(IntegerStorage::U16(vec![1]), vec![1, 1]).expect("lower");
        ok(
            "mustBeGreaterThan",
            vec![Value::Num(2.0), Value::Tensor(lower)],
        );

        let upper = Tensor::new_integer(IntegerStorage::U16(vec![3]), vec![1, 1]).expect("upper");
        ok(
            "mustBeLessThan",
            vec![Value::Num(2.0), Value::Tensor(upper)],
        );

        let range_lower =
            Tensor::new_integer(IntegerStorage::U16(vec![1]), vec![1, 1]).expect("range lower");
        let range_upper =
            Tensor::new_integer(IntegerStorage::U16(vec![3]), vec![1, 1]).expect("range upper");
        ok(
            "mustBeInRange",
            vec![
                Value::Tensor(
                    Tensor::new_integer(IntegerStorage::U16(vec![2]), vec![1, 1])
                        .expect("range value"),
                ),
                Value::Tensor(range_lower),
                Value::Tensor(range_upper),
            ],
        );

        let wide = 9_007_199_254_740_993_u64;
        let adjacent = wide - 1;
        let rounded = adjacent as f64;
        assert_eq!(wide as f64, rounded);

        let wide_value =
            Tensor::new_integer(IntegerStorage::U64(vec![wide]), vec![1, 1]).expect("wide value");
        ok(
            "mustBeGreaterThan",
            vec![Value::Tensor(wide_value.clone()), Value::Num(rounded)],
        );
        err(
            "mustBeLessThanOrEqual",
            vec![Value::Tensor(wide_value.clone()), Value::Num(rounded)],
        );
        err(
            "mustBeInRange",
            vec![
                Value::Tensor(wide_value.clone()),
                Value::Tensor(
                    Tensor::new_integer(IntegerStorage::U64(vec![0]), vec![1, 1])
                        .expect("wide lower"),
                ),
                Value::Tensor(
                    Tensor::new_integer(IntegerStorage::U64(vec![adjacent]), vec![1, 1])
                        .expect("wide upper"),
                ),
            ],
        );

        let complex_value = ComplexTensor::new_integer(
            IntegerComplexStorage::new(
                IntegerStorage::U64(vec![wide]),
                IntegerStorage::U64(vec![0]),
            )
            .expect("complex integer storage"),
            vec![1, 1],
        )
        .expect("complex integer tensor");
        ok(
            "mustBeGreaterThan",
            vec![Value::ComplexTensor(complex_value), Value::Num(rounded)],
        );

        let sparse_value = Value::SparseTensor(
            SparseTensor::new_integer(1, 1, vec![0, 1], vec![0], IntegerStorage::U64(vec![wide]))
                .expect("sparse integer value"),
        );
        ok(
            "mustBeGreaterThan",
            vec![sparse_value.clone(), Value::Num(rounded)],
        );
        err(
            "mustBeInRange",
            vec![
                sparse_value,
                Value::SparseTensor(
                    SparseTensor::new_integer(
                        1,
                        1,
                        vec![0, 0],
                        vec![],
                        IntegerStorage::U64(vec![]),
                    )
                    .expect("sparse lower"),
                ),
                Value::SparseTensor(
                    SparseTensor::new_integer(
                        1,
                        1,
                        vec![0, 1],
                        vec![0],
                        IntegerStorage::U64(vec![adjacent]),
                    )
                    .expect("sparse upper"),
                ),
            ],
        );
    }

    #[test]
    fn in_range_supports_interval_flags_and_rejects_extra_inputs() {
        ok(
            "mustBeInRange",
            vec![Value::Num(1.0), Value::Num(1.0), Value::Num(2.0)],
        );
        err(
            "mustBeInRange",
            vec![
                Value::Num(1.0),
                Value::Num(1.0),
                Value::Num(2.0),
                Value::String("exclusive".into()),
            ],
        );
        ok(
            "mustBeInRange",
            vec![
                Value::Num(1.5),
                Value::Num(1.0),
                Value::Num(2.0),
                Value::String("exclusive".into()),
            ],
        );
        err(
            "mustBeInRange",
            vec![
                Value::Num(1.0),
                Value::Num(1.0),
                Value::Num(2.0),
                Value::String("exclusive".into()),
                Value::String("inclusive".into()),
            ],
        );
        ok(
            "mustBeInRange",
            vec![
                Value::Num(2.0),
                Value::Num(1.0),
                Value::Num(2.0),
                Value::String("exclude-lower".into()),
            ],
        );
        err(
            "mustBeInRange",
            vec![
                Value::Num(2.0),
                Value::Num(1.0),
                Value::Num(2.0),
                Value::String("inclusive".into()),
                Value::String("exclusive".into()),
                Value::String("extra".into()),
            ],
        );
    }

    #[test]
    fn varname_rules_reject_keywords_underscores_and_overlong_names() {
        assert!(isvarname_value(&Value::String("alpha_1".into())));
        assert!(!isvarname_value(&Value::String("_alpha".into())));
        assert!(!isvarname_value(&Value::String("1alpha".into())));
        assert!(!isvarname_value(&Value::String("for".into())));
        assert!(!isvarname_value(&Value::String("end".into())));
        assert!(isvarname_value(&Value::String(
            "a".repeat(MATLAB_NAME_LENGTH_MAX)
        )));
        assert!(!isvarname_value(&Value::String(
            "a".repeat(MATLAB_NAME_LENGTH_MAX + 1)
        )));
    }

    #[test]
    fn callable_validators_reject_unexpected_extra_arguments() {
        err("mustBePositive", vec![Value::Num(1.0), Value::Num(2.0)]);
        err("mustBeMember", vec![Value::String("on".into())]);
        err(
            "mustBeA",
            vec![
                Value::Num(1.0),
                Value::String("double".into()),
                Value::String("extra".into()),
            ],
        );
    }

    #[test]
    fn ordered_validators_support_exact_compatible_integer_bounds() {
        let value = integer_tensor(
            IntegerStorage::U64(vec![9_007_199_254_740_993, 4]),
            vec![2, 1],
        );
        let bounds = integer_tensor(
            IntegerStorage::U64(vec![9_007_199_254_740_992, 3]),
            vec![2, 1],
        );
        ok("mustBeGreaterThan", vec![value.clone(), bounds]);
        err(
            "mustBeLessThanOrEqual",
            vec![value, Value::Num(9_007_199_254_740_992.0)],
        );

        let matrix = integer_tensor(IntegerStorage::I16(vec![1, 2]), vec![2, 1]);
        let row_bounds = integer_tensor(IntegerStorage::I16(vec![0, 0]), vec![1, 2]);
        ok("mustBeGreaterThan", vec![matrix, row_bounds]);

        let huge_sparse = SparseTensor::new_integer(
            1_000_000,
            1_000_000,
            {
                let mut pointers = vec![0; 1_000_001];
                pointers[1..].fill(1);
                pointers
            },
            vec![0],
            IntegerStorage::I64(vec![1]),
        )
        .expect("huge sparse sentinel");
        ok("mustBeNonnegative", vec![Value::SparseTensor(huge_sparse)]);
    }

    #[test]
    fn must_be_in_range_requires_same_class_and_compares_wide_integers_exactly() {
        let value = integer_tensor(IntegerStorage::U64(vec![9_007_199_254_740_993]), vec![1, 1]);
        let lower = integer_tensor(IntegerStorage::U64(vec![9_007_199_254_740_993]), vec![1, 1]);
        let upper = integer_tensor(IntegerStorage::U64(vec![u64::MAX]), vec![1, 1]);
        ok("mustBeInRange", vec![value.clone(), lower, upper]);
        err(
            "mustBeInRange",
            vec![value, Value::Num(0.0), Value::Num(f64::INFINITY)],
        );
    }

    #[test]
    fn must_be_member_enforces_nondouble_class_compatibility_without_rounding() {
        let wide = integer_tensor(IntegerStorage::U64(vec![9_007_199_254_740_993]), vec![1, 1]);
        let rounded_double = tensor(vec![9_007_199_254_740_992.0], 1, 1);
        err("mustBeMember", vec![wide.clone(), rounded_double]);
        let exact = integer_tensor(IntegerStorage::U64(vec![9_007_199_254_740_993]), vec![1, 1]);
        ok("mustBeMember", vec![wide.clone(), exact]);
        let unlike = integer_tensor(IntegerStorage::I64(vec![9]), vec![1, 1]);
        err("mustBeMember", vec![wide, unlike]);
    }

    #[test]
    fn resident_content_validators_download_exactly_and_preserve_source() {
        use futures::executor::block_on;

        test_support::with_test_provider(|provider| {
            let tensor = Tensor::new_integer(IntegerStorage::I64(vec![-1, 2]), vec![2, 1])
                .expect("resident integer");
            let handle = crate::builtins::common::gpu_helpers::upload_tensor(provider, &tensor)
                .expect("upload resident integer");
            let handle =
                handle.with_provenance(runmat_accelerate_api::GpuHandleProvenance::Automatic);
            let value = Value::GpuTensor(handle.clone());
            assert!(block_on(dispatch_validator_async(
                "mustBeNonzero",
                vec![value.clone()],
            ))
            .is_ok());
            assert!(block_on(dispatch_validator_async("mustBePositive", vec![value],)).is_err());
            assert!(
                crate::builtins::common::gpu_helpers::exact_provider_for_handle(&handle).is_some()
            );
            let gathered = block_on(provider.download_integer(&handle)).expect("source survives");
            assert_eq!(
                gathered.data,
                runmat_accelerate_api::HostIntegerDataOwned::I64(vec![-1, 2])
            );
            provider.free(&handle).expect("free resident source");
            runmat_accelerate_api::clear_handle_metadata(&handle);
        });
    }

    #[cfg(feature = "wgpu")]
    #[test]
    fn wgpu_resident_integer_validators_preserve_exact_source_and_provenance() {
        use futures::executor::block_on;
        use runmat_accelerate_api::AccelProvider;

        let _lock = test_support::accel_test_lock();
        let provider = runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
            runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
        )
        .expect("register WGPU provider for integer validator coverage");
        let _provider = runmat_accelerate_api::ThreadProviderGuard::set(Some(provider));
        let tensor = Tensor::new_integer(
            IntegerStorage::U64(vec![9_007_199_254_740_993, u64::MAX]),
            vec![1, 2],
        )
        .expect("wide integer source");
        let handle = crate::builtins::common::gpu_helpers::upload_tensor(provider, &tensor)
            .expect("upload wide integer source");
        let handle = handle.with_provenance(runmat_accelerate_api::GpuHandleProvenance::Automatic);
        let value = Value::GpuTensor(handle.clone());

        for name in [
            "mustBeFinite",
            "mustBeInteger",
            "mustBeNonNan",
            "mustBeNonzero",
            "mustBePositive",
        ] {
            block_on(dispatch_validator_async(name, vec![value.clone()]))
                .unwrap_or_else(|error| panic!("{name} unexpectedly failed: {error}"));
        }
        let handle = handle.with_provenance(runmat_accelerate_api::GpuHandleProvenance::Explicit);
        let value = Value::GpuTensor(handle.clone());
        let _strict = crate::compatibility::push_runmat_extensions_enabled(false);
        for name in ["mustBeReal", "mustBeVector"] {
            block_on(dispatch_validator_async(name, vec![value.clone()]))
                .unwrap_or_else(|error| panic!("{name} unexpectedly failed: {error}"));
        }
        block_on(dispatch_validator_async(
            "mustBeUnderlyingType",
            vec![value.clone(), Value::String("uint64".into())],
        ))
        .expect("resident underlying type metadata");
        assert!(block_on(dispatch_validator_async(
            "mustBeScalarOrEmpty",
            vec![value.clone()],
        ))
        .is_err());
        assert!(block_on(dispatch_validator_async(
            "mustBeSparse",
            vec![value.clone()],
        ))
        .is_err());
        for name in ["mustBeText", "mustBeTextScalar", "mustBeValidVariableName"] {
            let error = block_on(dispatch_validator_async(name, vec![value.clone()]))
                .expect_err("resident integer must reject text validation");
            assert_eq!(error.gpu_gather_retry(), crate::GpuGatherRetry::Never);
        }
        assert_eq!(
            runmat_accelerate_api::handle_provenance(&handle),
            Some(runmat_accelerate_api::GpuHandleProvenance::Explicit)
        );
        let gathered = block_on(provider.download_integer(&handle)).expect("source survives");
        assert_eq!(
            gathered.data,
            runmat_accelerate_api::HostIntegerDataOwned::U64(
                vec![9_007_199_254_740_993, u64::MAX,]
            )
        );
        provider.free(&handle).expect("free source");
        runmat_accelerate_api::clear_handle_metadata(&handle);
    }

    #[test]
    fn undocumented_explicit_resident_validators_are_mode_gated() {
        test_support::with_test_provider(|provider| {
            let tensor = Tensor::new_integer(IntegerStorage::U8(vec![1]), vec![1, 1])
                .expect("resident integer");
            let handle = crate::builtins::common::gpu_helpers::upload_tensor(provider, &tensor)
                .expect("upload resident integer");
            let handle =
                handle.with_provenance(runmat_accelerate_api::GpuHandleProvenance::Explicit);
            let value = Value::GpuTensor(handle.clone());
            let _strict = crate::compatibility::push_runmat_extensions_enabled(false);
            for name in [
                "mustBeFinite",
                "mustBeInteger",
                "mustBeNonNan",
                "mustBeNonzero",
            ] {
                let error = dispatch_validator(name, vec![value.clone()])
                    .expect_err("explicit resident extension must reject in compatibility mode");
                assert_eq!(error.gpu_gather_retry(), crate::GpuGatherRetry::Never);
            }
            drop(_strict);
            let _runmat = crate::compatibility::push_runmat_extensions_enabled(true);
            for name in [
                "mustBeFinite",
                "mustBeInteger",
                "mustBeNonNan",
                "mustBeNonzero",
            ] {
                ok(name, vec![value.clone()]);
            }
            provider.free(&handle).expect("free resident source");
            runmat_accelerate_api::clear_handle_metadata(&handle);
        });
    }

    #[test]
    fn validate_function_signatures_json_checks_json_syntax() {
        assert_eq!(
            VALIDATE_FUNCTION_SIGNATURES_JSON_INTEGER_AUDIT.kind,
            BuiltinIntegerAuditKind::NotApplicable
        );
        ok(
            "validateFunctionSignaturesJSON",
            vec![Value::String(r#"{"functions":[]}"#.into())],
        );
        err(
            "validateFunctionSignaturesJSON",
            vec![Value::String("{not json}".into())],
        );
        err(
            "validateFunctionSignaturesJSON",
            vec![Value::Int(IntValue::U64(u64::MAX))],
        );
        let resident = Value::GpuTensor(runmat_accelerate_api::GpuTensorHandle {
            shape: vec![1, 1],
            device_id: u32::MAX,
            buffer_id: u64::MAX,
            descriptor: Default::default(),
        });
        let error = dispatch_validator("validateFunctionSignaturesJSON", vec![resident])
            .expect_err("resident numeric input must reject as invalid text");
        assert_eq!(error.gpu_gather_retry(), crate::GpuGatherRetry::Never);
    }
}

//! Retained legacy validator descriptors and integer capability declarations.

use super::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor, BuiltinExtensionDescriptor,
    BuiltinExtensionMode, BuiltinIntegerAuditDescriptor, BuiltinIntegerAuditKind,
    BuiltinIntegerBackendRule, BuiltinIntegerCapabilityDescriptor, BuiltinIntegerComputationDomain,
    BuiltinIntegerInputAvailability, BuiltinIntegerInputCapability, BuiltinIntegerOutputClassRule,
    BuiltinIntegerOverflowRule, BuiltinIntegerOverloadKind, BuiltinIntegerScalarDoubleRule,
    BuiltinOutputMode, BuiltinParamArity, BuiltinParamDescriptor, BuiltinParamType,
    BuiltinSignatureDescriptor,
};

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

pub(super) const VALIDATION_ERROR_PROVIDER_OWNERSHIP: BuiltinErrorDescriptor =
    BuiltinErrorDescriptor {
        code: "RM.ARGUMENT_VALIDATION.PROVIDER_OWNERSHIP_MISMATCH",
        identifier: Some("RunMat:validators:ProviderOwnershipMismatch"),
        when: "A resident value has no exact owning provider.",
        message: "argument validation: no acceleration provider owns the input",
    };

pub(super) const VALIDATION_ERROR_PROVIDER_PAYLOAD: BuiltinErrorDescriptor =
    BuiltinErrorDescriptor {
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

pub(super) const MUST_BE_INTEGER_RESIDENT_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "mustBeInteger.resident-input",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "explicit gpuArray validation is not documented for mustBeInteger",
        error_identifier: Some("RunMat:compatibility:mustBeIntegerResidentInput"),
    };
pub(super) const MUST_BE_FINITE_RESIDENT_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "mustBeFinite.resident-input",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "explicit gpuArray validation is not documented for mustBeFinite",
        error_identifier: Some("RunMat:compatibility:mustBeFiniteResidentInput"),
    };
pub(super) const MUST_BE_NON_NAN_RESIDENT_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "mustBeNonNan.resident-input",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "explicit gpuArray validation is not documented for mustBeNonNan",
        error_identifier: Some("RunMat:compatibility:mustBeNonNanResidentInput"),
    };
pub(super) const MUST_BE_NONZERO_RESIDENT_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
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

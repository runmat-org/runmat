use super::super::*;
use super::core::{FILLMISSING_SIGNATURES, RMMISSING_SIGNATURES, STANDARDIZE_SIGNATURES};

pub(in crate::builtins::missing::domain) const RMMISSING_INTEGER_DIM_EXTENSION:
    BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "rmmissing-integer-dimension",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "rmmissing accepts a typed-integer dimension control as a RunMat extension",
    error_identifier: Some("RunMat:compatibility:RmmissingIntegerDimensionExtension"),
};
pub const RMMISSING_EXTENSIONS: [BuiltinExtensionDescriptor; 1] = [RMMISSING_INTEGER_DIM_EXTENSION];
const RMMISSING_INTEGER_DATA_INPUTS: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "A",
        classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "The R2022a-and-later surface accepts datatypes without a standard missing definition; all eight integer classes therefore remain unchanged.",
    }];
const RMMISSING_INTEGER_DIM_INPUTS: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "dim",
        classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "RunMat additionally accepts a typed integer dimension scalar and reads it exactly as a structural control.",
    }];
pub const RMMISSING_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 2] = [
    BuiltinIntegerCapabilityDescriptor {
        form: "[R,TF] = rmmissing(integer_A, ...)",
        inputs: &RMMISSING_INTEGER_DATA_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::Structural,
        output_class: BuiltinIntegerOutputClassRule::PreserveInput,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::Multiple,
        notes: "Integer A is an exact no-op because it has no standard missing value; R preserves class and storage, TF is all-false logical, and documented resident outputs return through the owning provider.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "R = rmmissing(A, integer_dim)",
        inputs: &RMMISSING_INTEGER_DIM_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::Structural,
        output_class: BuiltinIntegerOutputClassRule::PreserveInput,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::StructuralParameter,
        notes: "The dimension extension is mode-gated before provider access and never crosses a floating boundary.",
    },
];

pub(in crate::builtins::missing::domain) const FILLMISSING_INTEGER_DATA_EXTENSION:
    BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "fillmissing-integer-data",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "fillmissing with typed-integer input data is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:FillmissingIntegerDataExtension"),
};
pub(in crate::builtins::missing::domain) const FILLMISSING_AGGREGATE_INTEGER_DATA_EXTENSION:
    BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "fillmissing-aggregate-integer-data",
    mode: BuiltinExtensionMode::RunMatOnly,
    description:
        "fillmissing with integer data nested in a table or cell array is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:FillmissingAggregateIntegerDataExtension"),
};
pub const FILLMISSING_EXTENSIONS: [BuiltinExtensionDescriptor; 2] = [
    FILLMISSING_INTEGER_DATA_EXTENSION,
    FILLMISSING_AGGREGATE_INTEGER_DATA_EXTENSION,
];
const FILLMISSING_INTEGER_INPUTS: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "A",
        classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "Integer arrays have no standard missing value. RunMat mode preserves authoritative same-class storage and returns an all-false filled-entry mask.",
    }];
const FILLMISSING_AGGREGATE_INTEGER_INPUTS: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "table variables or nested cell contents",
        classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "The aggregate is recursively classified before any resident child can be gathered; integer children retain exact same-class storage and contribute false entries to the filled mask.",
    }];
pub const FILLMISSING_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 2] = [
    BuiltinIntegerCapabilityDescriptor {
        form: "[F, TF] = fillmissing(integer_A, method, ...)",
        inputs: &FILLMISSING_INTEGER_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::Structural,
        output_class: BuiltinIntegerOutputClassRule::PreserveInput,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::Multiple,
        notes: "The RunMat-only integer form is an exact no-op because integer classes have no default missing representation; TF is logical false with the input shape.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "[F, TF] = fillmissing(table_or_cell_with_integer_data, method, ...)",
        inputs: &FILLMISSING_AGGREGATE_INTEGER_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::Structural,
        output_class: BuiltinIntegerOutputClassRule::FunctionSpecific,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::Multiple,
        notes: "Nested table/cell integer data is a separately declared RunMat-only aggregate extension and is classified recursively before provider access.",
    },
];
descriptor!(
    FILLMISSING_DESCRIPTOR,
    FILLMISSING_SIGNATURES,
    BuiltinOutputMode::ByRequestedOutputCount
);
descriptor!(
    RMMISSING_DESCRIPTOR,
    RMMISSING_SIGNATURES,
    BuiltinOutputMode::ByRequestedOutputCount
);
descriptor!(
    STANDARDIZE_MISSING_DESCRIPTOR,
    STANDARDIZE_SIGNATURES,
    BuiltinOutputMode::Fixed
);

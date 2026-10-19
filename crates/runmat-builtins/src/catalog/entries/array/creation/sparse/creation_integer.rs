//! Integer input and extension contracts for sparse creation forms.
use crate::*;

const NONZEROS_INTEGER_INPUTS: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "A",
        classes: &ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "All eight integer classes are documented for full and sparse inputs; authoritative elements are filtered without numeric conversion.",
    }];
pub const NONZEROS_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 1] =
    [BuiltinIntegerCapabilityDescriptor { form: "v = nonzeros(integer_A)", inputs: &NONZEROS_INTEGER_INPUTS, computation_domain: BuiltinIntegerComputationDomain::ExactInteger, output_class: BuiltinIntegerOutputClassRule::PreserveInput, overflow: BuiltinIntegerOverflowRule::NotApplicable, backend: BuiltinIntegerBackendRule::GatherFallback, overload: BuiltinIntegerOverloadKind::FunctionSpecific, notes: "Stored nonzero values are copied in column order into a full same-class column. Host, sparse, scalar, and supported real gpuArray inputs preserve exact integer class; automatic residency may gather transparently and explicit residency is restored through the exact owner." }];

const SPEYE_INTEGER_SIZE_INPUT: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "n, [m n], or m and n",
        classes: &ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "Every built-in integer class is documented for sparse identity dimensions and is decoded directly into platform extents.",
    }];
pub const SPEYE_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 1] =
    [BuiltinIntegerCapabilityDescriptor {
        form: "S = speye(integer_dimensions, typename?)",
        inputs: &SPEYE_INTEGER_SIZE_INPUT,
        computation_domain: BuiltinIntegerComputationDomain::Structural,
        output_class: BuiltinIntegerOutputClassRule::OptionDependent,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::HostOnly,
        overload: BuiltinIntegerOverloadKind::StructuralParameter,
        notes: "Dimensions affect shape only; typename selects documented double, single, or logical sparse storage.",
    }];

pub const SPONES_INTEGER_INPUT_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "spones-integer-input",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "spones with typed-integer input data is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:SponesIntegerInputExtension"),
};
pub const SPONES_EXTENSIONS: [BuiltinExtensionDescriptor; 1] = [SPONES_INTEGER_INPUT_EXTENSION];
const SPONES_INTEGER_INPUT: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "S",
        classes: &ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "MATLAB sparse storage does not establish typed-integer classes; RunMat admits them only to inspect the exact nonzero pattern.",
    }];
pub const SPONES_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 1] =
    [BuiltinIntegerCapabilityDescriptor {
        form: "S = spones(integer_A)",
        inputs: &SPONES_INTEGER_INPUT,
        computation_domain: BuiltinIntegerComputationDomain::ExactInteger,
        output_class: BuiltinIntegerOutputClassRule::Double,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::FunctionSpecific,
        notes: "The nonzero pattern is read exactly and emitted as a host sparse double matrix of ones; documented GPU input may use the gather fallback.",
    }];

pub const SPRAND_INTEGER_CONTROL_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "sprand-integer-numeric-control",
    mode: BuiltinExtensionMode::RunMatOnly,
    description:
        "sprand with typed-integer dimensions, density, or condition profile is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:SprandIntegerNumericControlExtension"),
};
pub const SPRAND_EXTENSIONS: [BuiltinExtensionDescriptor; 1] = [SPRAND_INTEGER_CONTROL_EXTENSION];
const SPRAND_INTEGER_PATTERN_INPUT: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "S",
        classes: &ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "All built-in integer classes are documented for the input whose nonzero pattern is copied.",
    }];
const SPRAND_INTEGER_CONTROL_INPUTS: [BuiltinIntegerInputCapability; 2] = [
    BuiltinIntegerInputCapability {
        name: "m and n",
        classes: &ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "Public documentation requires nonnegative integer values but does not establish typed storage; RunMat gates typed dimensions and decodes them exactly.",
    },
    BuiltinIntegerInputCapability {
        name: "density or rc",
        classes: &ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "Public documentation does not establish typed storage for density or condition values; RunMat gates them and requires exact binary64 admission.",
    },
];
pub const SPRAND_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 2] = [
    BuiltinIntegerCapabilityDescriptor { form: "R = sprand(integer_S, typename?)", inputs: &SPRAND_INTEGER_PATTERN_INPUT, computation_domain: BuiltinIntegerComputationDomain::ExactInteger, output_class: BuiltinIntegerOutputClassRule::OptionDependent, overflow: BuiltinIntegerOverflowRule::NotApplicable, backend: BuiltinIntegerBackendRule::GatherFallback, overload: BuiltinIntegerOverloadKind::FunctionSpecific, notes: "Only S's exact nonzero pattern is consumed; random values use double unless typename selects single." },
    BuiltinIntegerCapabilityDescriptor { form: "R = sprand(integer_m, integer_n, integer_density/rc, typename?)", inputs: &SPRAND_INTEGER_CONTROL_INPUTS, computation_domain: BuiltinIntegerComputationDomain::FloatingPoint, output_class: BuiltinIntegerOutputClassRule::OptionDependent, overflow: BuiltinIntegerOverflowRule::Error, backend: BuiltinIntegerBackendRule::GatherFallback, overload: BuiltinIntegerOverloadKind::StructuralParameter, notes: "Typed controls are gated RunMat extensions: dimensions remain exact structural values, while density and condition values require exact binary64 admission." },
];

pub const SPDIAGS_INTEGER_DATA_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "spdiags-integer-data",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "spdiags with typed-integer matrix data is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:SpdiagsIntegerDataExtension"),
};
pub const SPDIAGS_INTEGER_CONTROL_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "spdiags-integer-control",
        mode: BuiltinExtensionMode::RunMatOnly,
        description:
            "spdiags with typed-integer diagonal or dimension controls is a RunMat extension",
        error_identifier: Some("RunMat:compatibility:SpdiagsIntegerControlExtension"),
    };
pub const SPDIAGS_EXTENSIONS: [BuiltinExtensionDescriptor; 2] = [
    SPDIAGS_INTEGER_DATA_EXTENSION,
    SPDIAGS_INTEGER_CONTROL_EXTENSION,
];
const SPDIAGS_INTEGER_DATA_INPUT: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "A, B, or Bin",
        classes: &ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "Documented matrix storage is single, double, or logical; RunMat gates typed matrix values and processes them directly in their integer class.",
    }];
const SPDIAGS_INTEGER_CONTROL_INPUTS: [BuiltinIntegerInputCapability; 2] = [
    BuiltinIntegerInputCapability { name: "d", classes: &ALL_INTEGER_CLASSES, availability: BuiltinIntegerInputAvailability::RunMatOnly, scalar_double: BuiltinIntegerScalarDoubleRule::Allowed, notes: "Diagonal offsets are documented as integer-valued but typed storage is not established; RunMat gates typed offsets and decodes them exactly." },
    BuiltinIntegerInputCapability { name: "m and n", classes: &ALL_INTEGER_CLASSES, availability: BuiltinIntegerInputAvailability::RunMatOnly, scalar_double: BuiltinIntegerScalarDoubleRule::Allowed, notes: "Output dimensions are documented as nonnegative integers but typed storage is not established; RunMat gates typed dimensions and decodes them exactly." },
];
pub const SPDIAGS_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 2] = [
    BuiltinIntegerCapabilityDescriptor { form: "B/S = spdiags(integer_A/B/Bin, ...)", inputs: &SPDIAGS_INTEGER_DATA_INPUT, computation_domain: BuiltinIntegerComputationDomain::FunctionSpecific, output_class: BuiltinIntegerOutputClassRule::FunctionSpecific, overflow: BuiltinIntegerOverflowRule::Saturate, backend: BuiltinIntegerBackendRule::GatherFallback, overload: BuiltinIntegerOverloadKind::FunctionSpecific, notes: "Typed matrix data is a gated extension. Extraction and construction preserve the integer class exactly; replacement follows the target class, and duplicate same-class integer diagonals add with saturation." },
    BuiltinIntegerCapabilityDescriptor { form: "B/S = spdiags(A/B, integer_d, integer_m?, integer_n?)", inputs: &SPDIAGS_INTEGER_CONTROL_INPUTS, computation_domain: BuiltinIntegerComputationDomain::Structural, output_class: BuiltinIntegerOutputClassRule::FunctionSpecific, overflow: BuiltinIntegerOverflowRule::Error, backend: BuiltinIntegerBackendRule::GatherFallback, overload: BuiltinIntegerOverloadKind::StructuralParameter, notes: "Typed offsets and dimensions are gated RunMat extensions decoded exactly before diagonal selection or allocation." },
];

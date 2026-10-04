use crate::*;

const FREAD_INTEGER_ID_INPUTS: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "fileID",
        classes: &crate::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "The compatibility target documents double file identifiers; typed integer identifiers are independently gated and checked exactly.",
    }];
const FREAD_INTEGER_SIZE_INPUTS: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "sizeA",
        classes: &crate::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "The compatibility target documents double size values; typed integer scalar and two-element size controls are independently gated.",
    }];
const FREAD_INTEGER_SKIP_INPUTS: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "skip",
        classes: &crate::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "The compatibility target documents a double skip scalar; typed integer skip values are independently gated.",
    }];
const FREAD_INTEGER_PROTOTYPE_INPUTS: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "prototype",
        classes: &crate::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::Rejected,
        notes: "A typed integer host or resident prototype selects the exact output class; this like form is a RunMat extension.",
    }];
pub const FREAD_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 5] = [
    BuiltinIntegerCapabilityDescriptor {
        form: "A = fread(fileID, ..., integer_output_precision)",
        inputs: &[],
        computation_domain: BuiltinIntegerComputationDomain::ExactInteger,
        output_class: BuiltinIntegerOutputClassRule::FunctionSpecific,
        overflow: BuiltinIntegerOverflowRule::Saturate,
        backend: BuiltinIntegerBackendRule::HostOnly,
        overload: BuiltinIntegerOverloadKind::FunctionSpecific,
        notes: "Documented source=>output and *source precision forms produce the requested exact integer storage; the count output remains double.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "A = fread(integer_fileID, ...)",
        inputs: &FREAD_INTEGER_ID_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::Structural,
        output_class: BuiltinIntegerOutputClassRule::FunctionSpecific,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::ScalarOnly,
        notes: "The registry identifier is validated exactly as an integer.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "A = fread(fileID, integer_sizeA, ...)",
        inputs: &FREAD_INTEGER_SIZE_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::Structural,
        output_class: BuiltinIntegerOutputClassRule::FunctionSpecific,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::StructuralParameter,
        notes: "Scalar and two-element sizes are decoded exactly and bounded before allocation or file access.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "A = fread(fileID, ..., integer_skip, ...)",
        inputs: &FREAD_INTEGER_SKIP_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::Structural,
        output_class: BuiltinIntegerOutputClassRule::FunctionSpecific,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::ScalarOnly,
        notes: "The skip is decoded exactly and bounded to the seek domain.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "A = fread(fileID, ..., \"like\", integer_prototype)",
        inputs: &FREAD_INTEGER_PROTOTYPE_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::ExactInteger,
        output_class: BuiltinIntegerOutputClassRule::PreserveInput,
        overflow: BuiltinIntegerOverflowRule::Saturate,
        backend: BuiltinIntegerBackendRule::FunctionSpecific,
        overload: BuiltinIntegerOverloadKind::FunctionSpecific,
        notes: "The prototype determines one of all eight integer output classes. Host prototypes return exact host storage; resident prototypes upload through their actual owning provider after saturating conversion and strict output validation.",
    },
];

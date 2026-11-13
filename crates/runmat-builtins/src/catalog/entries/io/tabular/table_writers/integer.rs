use crate::*;

pub const WRITETABLE_BYTES_OUTPUT_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "writetable-bytes-written-output",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "Request a bytes-written output from writetable",
        error_identifier: Some("RunMat:compatibility:WritetableBytesOutputExtension"),
    };
pub const WRITETABLE_EXPLICIT_GPU_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "writetable-explicit-gpu-input",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "Pass explicit gpuArray input to writetable",
        error_identifier: Some("RunMat:compatibility:WritetableExplicitGpuInputExtension"),
    };
pub const WRITETABLE_INTEGER_LOGICAL_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "writetable-integer-logical-option",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "Use a typed-integer WriteVariableNames value with writetable",
        error_identifier: Some("RunMat:compatibility:WritetableIntegerLogicalOptionExtension"),
    };
pub const WRITETABLE_EXTENSIONS: [BuiltinExtensionDescriptor; 3] = [
    WRITETABLE_BYTES_OUTPUT_EXTENSION,
    WRITETABLE_EXPLICIT_GPU_EXTENSION,
    WRITETABLE_INTEGER_LOGICAL_EXTENSION,
];
pub const WRITETIMETABLE_BYTES_OUTPUT_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "writetimetable-bytes-written-output",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "Request a bytes-written output from writetimetable",
        error_identifier: Some("RunMat:compatibility:WritetimetableBytesOutputExtension"),
    };
pub const WRITETIMETABLE_EXPLICIT_GPU_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "writetimetable-explicit-gpu-input",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "Pass explicit gpuArray input to writetimetable",
        error_identifier: Some("RunMat:compatibility:WritetimetableExplicitGpuInputExtension"),
    };
pub const WRITETIMETABLE_INTEGER_LOGICAL_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "writetimetable-integer-logical-option",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "Use a typed-integer WriteVariableNames value with writetimetable",
        error_identifier: Some("RunMat:compatibility:WritetimetableIntegerLogicalOptionExtension"),
    };
pub const WRITETIMETABLE_EXTENSIONS: [BuiltinExtensionDescriptor; 3] = [
    WRITETIMETABLE_BYTES_OUTPUT_EXTENSION,
    WRITETIMETABLE_EXPLICIT_GPU_EXTENSION,
    WRITETIMETABLE_INTEGER_LOGICAL_EXTENSION,
];

const TABLE_WRITE_INTEGER_VARIABLE_INPUT: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "integer table variables",
        classes: &ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "Numeric table variables from all eight integer classes serialize from authoritative integer storage.",
    }];
const TABLE_WRITE_INTEGER_LOGICAL_INPUT: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "WriteVariableNames",
        classes: &ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "RunMat mode preserves scalar integer truth conversion: zero is false and nonzero is true, without floating conversion. Native scalar integer tensors accept exact zero or one.",
    }];
pub const WRITETABLE_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 2] = [
    BuiltinIntegerCapabilityDescriptor { form: "writetable(T_with_integer_variables, filename, ___)", inputs: &TABLE_WRITE_INTEGER_VARIABLE_INPUT, computation_domain: BuiltinIntegerComputationDomain::ExactInteger, output_class: BuiltinIntegerOutputClassRule::NotApplicable, overflow: BuiltinIntegerOverflowRule::NotApplicable, backend: BuiltinIntegerBackendRule::GatherFallback, overload: BuiltinIntegerOverloadKind::FunctionSpecific, notes: "Integer cells retain their exact decimal value through delimited-text serialization. Automatically resident values gather transparently; explicit device input is separately gated." },
    BuiltinIntegerCapabilityDescriptor { form: "writetable(T, filename, 'WriteVariableNames', typed_integer)", inputs: &TABLE_WRITE_INTEGER_LOGICAL_INPUT, computation_domain: BuiltinIntegerComputationDomain::Structural, output_class: BuiltinIntegerOutputClassRule::NotApplicable, overflow: BuiltinIntegerOverflowRule::Error, backend: BuiltinIntegerBackendRule::GatherFallback, overload: BuiltinIntegerOverloadKind::StructuralParameter, notes: "Strict mode rejects the typed-integer alias before file access. RunMat scalar integers preserve exact zero/nonzero truth conversion; scalar native integer tensors require zero or one." },
];
pub const WRITETIMETABLE_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 2] = [
    BuiltinIntegerCapabilityDescriptor { form: "writetimetable(TT_with_integer_variables, filename, ___)", inputs: &TABLE_WRITE_INTEGER_VARIABLE_INPUT, computation_domain: BuiltinIntegerComputationDomain::ExactInteger, output_class: BuiltinIntegerOutputClassRule::NotApplicable, overflow: BuiltinIntegerOverflowRule::NotApplicable, backend: BuiltinIntegerBackendRule::GatherFallback, overload: BuiltinIntegerOverloadKind::FunctionSpecific, notes: "Integer cells retain their exact decimal value through delimited-text serialization. Automatically resident values gather transparently; explicit device input is separately gated." },
    BuiltinIntegerCapabilityDescriptor { form: "writetimetable(TT, filename, 'WriteVariableNames', typed_integer)", inputs: &TABLE_WRITE_INTEGER_LOGICAL_INPUT, computation_domain: BuiltinIntegerComputationDomain::Structural, output_class: BuiltinIntegerOutputClassRule::NotApplicable, overflow: BuiltinIntegerOverflowRule::Error, backend: BuiltinIntegerBackendRule::GatherFallback, overload: BuiltinIntegerOverloadKind::StructuralParameter, notes: "Strict mode rejects the typed-integer alias before file access. RunMat scalar integers preserve exact zero/nonzero truth conversion; scalar native integer tensors require zero or one." },
];

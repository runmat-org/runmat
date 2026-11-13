use crate::*;

pub const TABLE_GPU_INPUT_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "table-gpu-input",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "table with an explicit interactive GPU variable is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:TableGpuInputExtension"),
};
pub const TABLE_EXTENSIONS: [BuiltinExtensionDescriptor; 1] = [TABLE_GPU_INPUT_EXTENSION];
const TABLE_INTEGER_VARIABLE_INPUT: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "var1,...,varN",
        classes: &ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "Any table variable may use one of the eight real integer classes and retains its own class and authoritative payload.",
    }];
const TABLE_INTEGER_SIZE_INPUT: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "sz",
        classes: &ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "The documented two-element numeric size vector is decoded exactly from native storage and its second element must match VariableTypes.",
    }];
pub const TABLE_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 2] = [
    BuiltinIntegerCapabilityDescriptor {
        form: "T = table(integer_var1, ..., integer_varN, Name=Value...)",
        inputs: &TABLE_INTEGER_VARIABLE_INPUT,
        computation_domain: BuiltinIntegerComputationDomain::Structural,
        output_class: BuiltinIntegerOutputClassRule::PreserveInput,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::Multiple,
        notes: "Host and automatically resident variables become host table variables with exact classes and values preserved; unsupported explicit GPU intent is independently gated before gather.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "T = table(Size=integer_sz, VariableTypes=integer_type_names, Name=Value...)",
        inputs: &TABLE_INTEGER_SIZE_INPUT,
        computation_domain: BuiltinIntegerComputationDomain::Structural,
        output_class: BuiltinIntegerOutputClassRule::OptionDependent,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::HostOnly,
        overload: BuiltinIntegerOverloadKind::StructuralParameter,
        notes: "int8 through uint64 VariableTypes allocate zero-filled authoritative native columns directly; floating VariableTypes allocate native f64 or f32 columns.",
    },
];

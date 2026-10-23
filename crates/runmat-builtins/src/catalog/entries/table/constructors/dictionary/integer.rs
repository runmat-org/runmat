use crate::*;

pub const DICTIONARY_GPU_INPUT_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "dictionary-gpu-input",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "dictionary with an interactive resident argument is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:DictionaryGpuInputExtension"),
};
pub const DICTIONARY_EXTENSIONS: [BuiltinExtensionDescriptor; 1] = [DICTIONARY_GPU_INPUT_EXTENSION];
const DICTIONARY_INTEGER_INPUTS: [BuiltinIntegerInputCapability; 2] = [
    BuiltinIntegerInputCapability {
        name: "keys",
        classes: &ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "Dictionary keys accept every numeric integer class and retain exact configured-class identity, including wide uint64 values.",
    },
    BuiltinIntegerInputCapability {
        name: "values",
        classes: &ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "Dictionary values accept every numeric integer class and retain exact configured-class storage.",
    },
];
const DICTIONARY_RESIDENT_INTEGER_INPUT: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "resident keys or values",
        classes: &ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "Interactive resident input is not a documented gpuArray overload and is gated before exact gather into the host dictionary object.",
    }];
pub const DICTIONARY_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 2] = [
    BuiltinIntegerCapabilityDescriptor {
        form: "d = dictionary(integer_keys, integer_values)",
        inputs: &DICTIONARY_INTEGER_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::ExactInteger,
        output_class: BuiltinIntegerOutputClassRule::FunctionSpecific,
        overflow: BuiltinIntegerOverflowRule::FunctionSpecific,
        backend: BuiltinIntegerBackendRule::HostOnly,
        overload: BuiltinIntegerOverloadKind::Multiple,
        notes: "Construction, duplicate resolution, lookup, assignment, and removal compare configured-class integer values exactly; the result is a host dictionary object.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "d = dictionary(gpuArray(integer_keys_or_values), ...)",
        inputs: &DICTIONARY_RESIDENT_INTEGER_INPUT,
        computation_domain: BuiltinIntegerComputationDomain::ExactInteger,
        output_class: BuiltinIntegerOutputClassRule::FunctionSpecific,
        overflow: BuiltinIntegerOverflowRule::FunctionSpecific,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::Multiple,
        notes: "RunMat mode gathers admitted resident inputs through the owning provider and preserves exact typed values in the host object.",
    },
];

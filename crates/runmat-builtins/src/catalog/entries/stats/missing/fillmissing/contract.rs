use crate::*;

const INPUTS: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "A",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Numeric, string, cell, or table data.",
    },
    BuiltinParamDescriptor {
        name: "method_and_options",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Variadic,
        default: None,
        description: "Supported fill method, value, and optional dimension.",
    },
];
const OUTPUTS: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "F",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Filled data with the input shape.",
    },
    BuiltinParamDescriptor {
        name: "TF",
        ty: BuiltinParamType::LogicalArray,
        arity: BuiltinParamArity::Optional,
        default: None,
        description: "Entries actually filled.",
    },
];
const VALUE_OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "F",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Filled data with the input shape.",
}];
pub const FILLMISSING_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &[
        BuiltinSignatureDescriptor {
            label: "F = fillmissing(A, method, ...)",
            inputs: &INPUTS,
            outputs: &VALUE_OUTPUT,
        },
        BuiltinSignatureDescriptor {
            label: "[F, TF] = fillmissing(A, method, ...)",
            inputs: &INPUTS,
            outputs: &OUTPUTS,
        },
    ],
    output_mode: BuiltinOutputMode::ByRequestedOutputCount,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &[
        BuiltinErrorDescriptor {
            code: "RM.MISSING.INVALID_ARGUMENT",
            identifier: Some("RunMat:missing:InvalidArgument"),
            when: "The method, dimension, output count, or resident input is invalid.",
            message: "missing-value builtin: invalid argument",
        },
        BuiltinErrorDescriptor {
            code: "RM.MISSING.UNSUPPORTED_TYPE",
            identifier: Some("RunMat:missing:UnsupportedType"),
            when: "The input and method have no supported fill representation.",
            message: "missing-value builtin: unsupported input type",
        },
        BuiltinErrorDescriptor {
            code: "RM.MISSING.INTERNAL",
            identifier: Some("RunMat:missing:InternalError"),
            when: "Array or table materialization fails.",
            message: "missing-value builtin: internal error",
        },
    ],
};

pub const FILLMISSING_INTEGER_DATA_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "fillmissing-integer-data",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "fillmissing with typed-integer input data is a RunMat extension",
        error_identifier: Some("RunMat:compatibility:FillmissingIntegerDataExtension"),
    };
pub const FILLMISSING_AGGREGATE_INTEGER_DATA_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
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

const INTEGER_DATA: [BuiltinIntegerInputCapability; 1] = [BuiltinIntegerInputCapability {
    name: "A",
    classes: &ALL_INTEGER_CLASSES,
    availability: BuiltinIntegerInputAvailability::RunMatOnly,
    scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
    notes: "Integer arrays have no standard missing element; RunMat mode keeps authoritative integer storage unchanged.",
}];
const INTEGER_AGGREGATE: [BuiltinIntegerInputCapability; 1] = [BuiltinIntegerInputCapability {
    name: "table variables or nested cell contents",
    classes: &ALL_INTEGER_CLASSES,
    availability: BuiltinIntegerInputAvailability::RunMatOnly,
    scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
    notes:
        "Nested integer data is classified before any resident child is gathered and remains exact.",
}];
pub const FILLMISSING_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 2] = [
    BuiltinIntegerCapabilityDescriptor {
        form: "[F, TF] = fillmissing(integer_A, method, ...)",
        inputs: &INTEGER_DATA,
        computation_domain: BuiltinIntegerComputationDomain::Structural,
        output_class: BuiltinIntegerOutputClassRule::PreserveInput,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::Multiple,
        notes: "F preserves exact same-class storage and TF is all false.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "[F, TF] = fillmissing(table_or_cell_with_integer_data, method, ...)",
        inputs: &INTEGER_AGGREGATE,
        computation_domain: BuiltinIntegerComputationDomain::Structural,
        output_class: BuiltinIntegerOutputClassRule::FunctionSpecific,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::Multiple,
        notes:
            "Each nested integer value retains exact storage and contributes false mask entries.",
    },
];

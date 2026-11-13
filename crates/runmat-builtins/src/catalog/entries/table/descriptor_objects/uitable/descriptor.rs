use crate::*;

pub const UITABLE_EXPLICIT_GPU_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "uitable-explicit-gpu-data",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "uitable with explicitly GPU-resident data is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:UitableExplicitGpuDataExtension"),
};
pub const UITABLE_EXTENSIONS: [BuiltinExtensionDescriptor; 1] = [UITABLE_EXPLICIT_GPU_EXTENSION];
const UITABLE_INTEGER_DATA_INPUT: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "Data",
        classes: &ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "The Data property accepts every numeric type and retains the exact integer class, shape, and values in the UI component object.",
    }];
pub const UITABLE_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 1] =
    [BuiltinIntegerCapabilityDescriptor {
        form: "ui = uitable(..., Data=integer_data, ...)",
        inputs: &UITABLE_INTEGER_DATA_INPUT,
        computation_domain: BuiltinIntegerComputationDomain::Structural,
        output_class: BuiltinIntegerOutputClassRule::FunctionSpecific,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::Multiple,
        notes: "Host integer data remains exact in the Data property. Automatically resident data gathers transparently through its owner; explicit gpuArray data is a separately gated extension.",
    }];

pub const UITABLE_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &[
        BuiltinSignatureDescriptor {
            label: "u = uitable()",
            inputs: &[],
            outputs: &[BuiltinParamDescriptor {
                name: "descriptor",
                ty: BuiltinParamType::Any,
                arity: BuiltinParamArity::Required,
                default: None,
                description: "Scalar host value descriptor; not a live iterator or UI handle.",
            }],
        },
        BuiltinSignatureDescriptor {
            label: "u = uitable(A)",
            inputs: &[BuiltinParamDescriptor {
                name: "A",
                ty: BuiltinParamType::Any,
                arity: BuiltinParamArity::Required,
                default: None,
                description: "Fallback positional data.",
            }],
            outputs: &[BuiltinParamDescriptor {
                name: "descriptor",
                ty: BuiltinParamType::Any,
                arity: BuiltinParamArity::Required,
                default: None,
                description: "Scalar host value descriptor; not a live iterator or UI handle.",
            }],
        },
        BuiltinSignatureDescriptor {
            label: "u = uitable('Data', A)",
            inputs: &[
                BuiltinParamDescriptor {
                    name: "Data",
                    ty: BuiltinParamType::StringScalar,
                    arity: BuiltinParamArity::Required,
                    default: None,
                    description: "Data property name.",
                },
                BuiltinParamDescriptor {
                    name: "A",
                    ty: BuiltinParamType::Any,
                    arity: BuiltinParamArity::Required,
                    default: None,
                    description: "Data retained in the host descriptor; no UI is rendered.",
                },
            ],
            outputs: &[BuiltinParamDescriptor {
                name: "descriptor",
                ty: BuiltinParamType::Any,
                arity: BuiltinParamArity::Required,
                default: None,
                description: "Scalar host value descriptor; not a live iterator or UI handle.",
            }],
        },
    ],
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &[super::super::super::shared::TABLE_ERROR_INVALID_ARGUMENT],
};

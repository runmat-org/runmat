use crate::*;

pub const GCA_FIGURE_ARGUMENT_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "gca-figure-argument",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "gca with an explicit figure argument is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:GcaFigureArgumentExtension"),
};
pub const GCA_STRUCT_OUTPUT_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "gca-struct-output",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "gca('struct') is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:GcaStructOutputExtension"),
};
pub const GCA_INTEGER_FIGURE_ALIAS_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "gca-integer-figure-alias",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "typed-integer numeric figure aliases for gca are a RunMat extension",
        error_identifier: Some("RunMat:compatibility:GcaIntegerFigureAliasExtension"),
    };
pub const GCA_EXTENSIONS: [BuiltinExtensionDescriptor; 3] = [
    GCA_FIGURE_ARGUMENT_EXTENSION,
    GCA_STRUCT_OUTPUT_EXTENSION,
    GCA_INTEGER_FIGURE_ALIAS_EXTENSION,
];

const GCA_INTEGER_FIGURE_INPUTS: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "fig",
        classes: &crate::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "The compatibility target's gca contract has no input form and returns a graphics object; typed integers are exact aliases for RunMat's numeric figure registry only.",
    }];
pub const GCA_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 1] =
    [BuiltinIntegerCapabilityDescriptor {
        form: "ax = gca(integer_fig_alias)",
        inputs: &GCA_INTEGER_FIGURE_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::Structural,
        output_class: BuiltinIntegerOutputClassRule::FunctionSpecific,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::HostOnly,
        overload: BuiltinIntegerOverloadKind::ScalarOnly,
        notes: "The alias is decoded exactly and gated before graphics lookup or mutation. RunMat currently returns its numeric axes representation rather than a MATLAB graphics object.",
    }];

const GCA_OUTPUT_HANDLE: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "ax",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description:
        "Current axes graphics object; RunMat currently returns its encoded numeric axes handle.",
}];

const GCA_OUTPUT_STRUCT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "s",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Axes state struct with handle/figure/rows/cols/index fields.",
}];

const GCA_INPUTS_NONE: [BuiltinParamDescriptor; 0] = [];

const GCA_INPUTS_MODE: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "mode",
    ty: BuiltinParamType::StringScalar,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Only 'struct' is supported.",
}];

const GCA_INPUTS_FIGURE: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "fig",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Exact native Figure receiver or numeric Figure compatibility handle whose current axes should be returned.",
}];

const GCA_SIGNATURES: [BuiltinSignatureDescriptor; 3] = [
    BuiltinSignatureDescriptor {
        label: "ax = gca()",
        inputs: &GCA_INPUTS_NONE,
        outputs: &GCA_OUTPUT_HANDLE,
    },
    BuiltinSignatureDescriptor {
        label: "ax = gca(fig)",
        inputs: &GCA_INPUTS_FIGURE,
        outputs: &GCA_OUTPUT_HANDLE,
    },
    BuiltinSignatureDescriptor {
        label: "s = gca('struct')",
        inputs: &GCA_INPUTS_MODE,
        outputs: &GCA_OUTPUT_STRUCT,
    },
];

const GCA_ERROR_INVALID_ARGUMENT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.GCA.INVALID_ARGUMENT",
    identifier: Some("RunMat:gca:InvalidArgument"),
    when: "Unsupported arguments are provided.",
    message: "gca: unsupported arguments",
};

const GCA_ERRORS: [BuiltinErrorDescriptor; 2] =
    [GCA_ERROR_INVALID_ARGUMENT, GCA_ERROR_TOO_MANY_OUTPUTS];

pub const GCA_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &GCA_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &GCA_ERRORS,
};
pub const GCA_ERROR_TOO_MANY_OUTPUTS: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.GCA.TOO_MANY_OUTPUTS",
    identifier: Some("RunMat:gca:TooManyOutputs"),
    when: "More than one output is requested.",
    message: "gca: at most one output is supported",
};

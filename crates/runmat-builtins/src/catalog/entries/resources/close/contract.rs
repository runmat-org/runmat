use crate::*;

const STATUS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "status",
    ty: BuiltinParamType::NumericScalar,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Scalar double: graphics completion, or whether a TCP resource was closed.",
}];
const TARGET: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "target",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Figure number, sealed TCP receiver, or an admitted close selector.",
}];
const TARGETS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "targets",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Variadic,
    default: None,
    description: "RunMat-only separate close targets.",
}];
const SIGNATURES: [BuiltinSignatureDescriptor; 4] = [
    BuiltinSignatureDescriptor {
        label: "close",
        inputs: &[],
        outputs: &[],
    },
    BuiltinSignatureDescriptor {
        label: "status = close()",
        inputs: &[],
        outputs: &STATUS,
    },
    BuiltinSignatureDescriptor {
        label: "status = close(target)",
        inputs: &TARGET,
        outputs: &STATUS,
    },
    BuiltinSignatureDescriptor {
        label: "status = close(targets...)",
        inputs: &TARGETS,
        outputs: &STATUS,
    },
];

pub const CLOSE_ERROR_TOO_MANY_OUTPUTS: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.CLOSE.TOO_MANY_OUTPUTS",
    identifier: Some("RunMat:close:TooManyOutputs"),
    when: "More than one output is requested.",
    message: "close: too many output arguments",
};
pub const CLOSE_ERROR_INVALID_ARGUMENT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.CLOSE.INVALID_ARGUMENT",
    identifier: Some("RunMat:close:InvalidArgument"),
    when: "A graphics target or TCP selector is invalid.",
    message: "close: invalid argument",
};
pub const CLOSE_ERROR_INVALID_HANDLE: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.CLOSE.INVALID_HANDLE",
    identifier: Some("RunMat:close:InvalidHandle"),
    when: "A TCP-looking receiver has no sealed socket authority.",
    message: "close: invalid handle",
};
pub const CLOSE_ERROR_TCP: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.CLOSE.TCP",
    identifier: None,
    when: "The session-owned TCP service rejects or fails a close operation.",
    message: "close: TCP close failed",
};
pub const CLOSE_ERROR_CANCELLED: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.CLOSE.CANCELLED",
    identifier: Some("RunMat:parallel:Cancelled"),
    when: "An invocation is cancelled while a TCP close is pending.",
    message: "close: cancelled",
};
pub const CLOSE_ERROR_GRAPHICS: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.CLOSE.GRAPHICS",
    identifier: None,
    when: "A graphics target cannot be parsed or the selected figure cannot be closed.",
    message: "close: graphics close failed",
};
pub const CLOSE_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &[
        CLOSE_ERROR_TOO_MANY_OUTPUTS,
        CLOSE_ERROR_INVALID_ARGUMENT,
        CLOSE_ERROR_INVALID_HANDLE,
        CLOSE_ERROR_TCP,
        CLOSE_ERROR_CANCELLED,
        CLOSE_ERROR_GRAPHICS,
    ],
};

pub const CLOSE_INTEGER_FIGURE_NUMBER_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "close-integer-figure-number",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "A typed integer figure number is a RunMat extension.",
        error_identifier: Some("RunMat:compatibility:CloseIntegerFigureNumberExtension"),
    };
pub const CLOSE_VARIADIC_TARGETS_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "close-variadic-targets",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "Separate variadic targets are a RunMat extension.",
        error_identifier: Some("RunMat:compatibility:CloseVariadicTargetsExtension"),
    };
pub const CLOSE_NETWORK_SELECTOR_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "close-network-selector",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "The clients and servers TCP scope selectors are RunMat extensions.",
        error_identifier: Some("RunMat:compatibility:CloseNetworkSelectorExtension"),
    };
pub const CLOSE_TCP_CELL_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "close-tcp-cell",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "A cell containing TCP close targets is a RunMat extension.",
    error_identifier: Some("RunMat:compatibility:CloseTcpCellExtension"),
};
pub const CLOSE_EXTENSIONS: [BuiltinExtensionDescriptor; 4] = [
    CLOSE_INTEGER_FIGURE_NUMBER_EXTENSION,
    CLOSE_VARIADIC_TARGETS_EXTENSION,
    CLOSE_NETWORK_SELECTOR_EXTENSION,
    CLOSE_TCP_CELL_EXTENSION,
];

const INTEGER_INPUT: [BuiltinIntegerInputCapability; 1] = [BuiltinIntegerInputCapability {
    name: "fig",
    classes: &ALL_INTEGER_CLASSES,
    availability: BuiltinIntegerInputAvailability::RunMatOnly,
    scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
    notes: "Figure identity uses exact host integer storage, not a Double mirror.",
}];
pub const CLOSE_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 1] = [BuiltinIntegerCapabilityDescriptor {
    form: "status = close(integer_fig)",
    inputs: &INTEGER_INPUT,
    computation_domain: BuiltinIntegerComputationDomain::Structural,
    output_class: BuiltinIntegerOutputClassRule::Double,
    overflow: BuiltinIntegerOverflowRule::NotApplicable,
    backend: BuiltinIntegerBackendRule::HostOnly,
    overload: BuiltinIntegerOverloadKind::StructuralParameter,
    notes: "Positive figure numbers must fit the u32 graphics identifier; resident numeric controls reject before provider dispatch.",
}];

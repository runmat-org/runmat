use crate::*;

pub const DRAWNOW_STATUS_OUTPUT_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "drawnow-status-output",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "Request a logical host-flush completion status",
        error_identifier: Some("RunMat:compatibility:DrawNowStatusOutputExtension"),
    };
pub const DRAWNOW_EXTENSIONS: [BuiltinExtensionDescriptor; 1] = [DRAWNOW_STATUS_OUTPUT_EXTENSION];

const NO_INPUT: [BuiltinParamDescriptor; 0] = [];
const ONE_CONTROL: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "control",
    ty: BuiltinParamType::StringScalar,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "limitrate, nocallbacks, update, or expose.",
}];
const TWO_CONTROLS: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "limitrate",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Limit presentation to at most 20 updates per second.",
    },
    BuiltinParamDescriptor {
        name: "nocallbacks",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Defer pending callbacks until a later full flush.",
    },
];
const STATUS_OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "ok",
    ty: BuiltinParamType::LogicalArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "RunMat-only status for a completed host flush.",
}];
const SIGNATURES: [BuiltinSignatureDescriptor; 6] = [
    BuiltinSignatureDescriptor {
        label: "drawnow",
        inputs: &NO_INPUT,
        outputs: &[],
    },
    BuiltinSignatureDescriptor {
        label: "drawnow control",
        inputs: &ONE_CONTROL,
        outputs: &[],
    },
    BuiltinSignatureDescriptor {
        label: "drawnow limitrate nocallbacks",
        inputs: &TWO_CONTROLS,
        outputs: &[],
    },
    BuiltinSignatureDescriptor {
        label: "ok = drawnow",
        inputs: &NO_INPUT,
        outputs: &STATUS_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "ok = drawnow control",
        inputs: &ONE_CONTROL,
        outputs: &STATUS_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "ok = drawnow limitrate nocallbacks",
        inputs: &TWO_CONTROLS,
        outputs: &STATUS_OUTPUT,
    },
];

pub const DRAWNOW_ERROR_INVALID_CONTROL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.DRAWNOW.INVALID_CONTROL",
    identifier: Some("RunMat:drawnow:InvalidControl"),
    when: "A control token is unknown, duplicated, conflicting, or in an unsupported order.",
    message: "drawnow: invalid control",
};
pub const DRAWNOW_ERROR_TOO_MANY_OUTPUTS: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.DRAWNOW.TOO_MANY_OUTPUTS",
    identifier: Some("RunMat:drawnow:TooManyOutputs"),
    when: "More than one output is requested.",
    message: "drawnow: too many output arguments",
};
pub const DRAWNOW_ERROR_PRESENTATION: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.DRAWNOW.PRESENTATION",
    identifier: Some("RunMat:drawnow:PresentationFailed"),
    when: "The selected host fails to present a figure or process callbacks.",
    message: "drawnow: graphics presentation failed",
};
pub const DRAWNOW_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.DRAWNOW.INTERNAL",
    identifier: None,
    when: "Operand materialization or runtime control flow fails with its original cause.",
    message: "drawnow: internal error",
};
const ERRORS: [BuiltinErrorDescriptor; 4] = [
    DRAWNOW_ERROR_INVALID_CONTROL,
    DRAWNOW_ERROR_TOO_MANY_OUTPUTS,
    DRAWNOW_ERROR_PRESENTATION,
    DRAWNOW_ERROR_INTERNAL,
];
pub const DRAWNOW_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ERRORS,
};

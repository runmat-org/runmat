use crate::*;

const SYS: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "sys",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Scalar SISO tf model.",
};
const W: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "w",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Optional,
    default: None,
    description: "Finite nonnegative angular-frequency vector.",
};
const MAG: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "mag",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Absolute magnitude, 1×1×N.",
};
const PHASE: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "phase",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Unwrapped phase in degrees, 1×1×N.",
};
const WOUT: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "wout",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Angular-frequency column vector.",
};
const SYS_INPUT: [BuiltinParamDescriptor; 1] = [SYS];
const SYS_W_INPUT: [BuiltinParamDescriptor; 2] = [SYS, W];
const MAG_OUTPUT: [BuiltinParamDescriptor; 1] = [MAG];
const MAG_PHASE_OUTPUT: [BuiltinParamDescriptor; 2] = [MAG, PHASE];
const ALL_OUTPUTS: [BuiltinParamDescriptor; 3] = [MAG, PHASE, WOUT];
const SIGNATURES: [BuiltinSignatureDescriptor; 6] = [
    BuiltinSignatureDescriptor {
        label: "bode(sys)",
        inputs: &SYS_INPUT,
        outputs: &[],
    },
    BuiltinSignatureDescriptor {
        label: "bode(sys,w)",
        inputs: &SYS_W_INPUT,
        outputs: &[],
    },
    BuiltinSignatureDescriptor {
        label: "mag = bode(sys,w)",
        inputs: &SYS_W_INPUT,
        outputs: &MAG_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "[mag,phase] = bode(sys,w)",
        inputs: &SYS_W_INPUT,
        outputs: &MAG_PHASE_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "[mag,phase,wout] = bode(sys,w)",
        inputs: &SYS_W_INPUT,
        outputs: &ALL_OUTPUTS,
    },
    BuiltinSignatureDescriptor {
        label: "[mag,phase,wout] = bode(sys)",
        inputs: &SYS_INPUT,
        outputs: &ALL_OUTPUTS,
    },
];
const ERRORS: [BuiltinErrorDescriptor; 5] = [
    BuiltinErrorDescriptor { code: "RM.BODE.INVALID_ARGUMENT", identifier: Some("RunMat:bode:InvalidArgument"), when: "Invocation or frequency vector is invalid, singular, or cannot produce a finite response.", message: "bode: invalid argument" },
    BuiltinErrorDescriptor { code: "RM.BODE.INVALID_MODEL", identifier: Some("RunMat:bode:InvalidModel"), when: "A transfer-function property is malformed.", message: "bode: invalid model" },
    BuiltinErrorDescriptor { code: "RM.BODE.UNSUPPORTED_MODEL", identifier: Some("RunMat:bode:UnsupportedModel"), when: "The model is outside the supported zero-delay SISO tf domain.", message: "bode: unsupported model" },
    BuiltinErrorDescriptor { code: "RM.BODE.INTERNAL", identifier: Some("RunMat:bode:Internal"), when: "Response array materialization fails.", message: "bode: internal error" },
    BuiltinErrorDescriptor { code: "RM.BODE.PLOT_FAILED", identifier: Some("RunMat:bode:PlotFailed"), when: "Statement-form presentation fails outside known nonfatal setup conditions.", message: "bode: plotting failed" },
];
pub const BODE_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &SIGNATURES,
    output_mode: BuiltinOutputMode::ByRequestedOutputCount,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ERRORS,
};

use crate::*;

const DYNAMIC_WORKSPACE_OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "varargout",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Variadic,
    default: None,
    description: "Value(s) produced by evaluated source text.",
}];
const EVAL_INPUTS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "source",
    ty: BuiltinParamType::StringScalar,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Source text to evaluate in the current workspace.",
}];
const EVALIN_INPUTS: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "workspace",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "`base` or `caller` workspace selector.",
    },
    BuiltinParamDescriptor {
        name: "source",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Source text to evaluate in the selected workspace.",
    },
];
const EVALC_OUTPUT_RESULTS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "results",
    ty: BuiltinParamType::StringScalar,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Captured Command Window text.",
}];
const EVALC_OUTPUTS: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "results",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Captured Command Window text.",
    },
    BuiltinParamDescriptor {
        name: "varargout",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Variadic,
        default: None,
        description: "Value(s) produced by evaluated source text.",
    },
];
const ASSIGNIN_INPUTS: [BuiltinParamDescriptor; 3] = [
    BuiltinParamDescriptor {
        name: "workspace",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "`base` or `caller` workspace selector.",
    },
    BuiltinParamDescriptor {
        name: "name",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Workspace variable name.",
    },
    BuiltinParamDescriptor {
        name: "value",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Value to assign into the selected workspace.",
    },
];

const EVAL_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "[varargout] = eval(source)",
    inputs: &EVAL_INPUTS,
    outputs: &DYNAMIC_WORKSPACE_OUTPUT,
}];
const EVALC_SIGNATURES: [BuiltinSignatureDescriptor; 2] = [
    BuiltinSignatureDescriptor {
        label: "results = evalc(expression)",
        inputs: &EVAL_INPUTS,
        outputs: &EVALC_OUTPUT_RESULTS,
    },
    BuiltinSignatureDescriptor {
        label: "[results,varargout] = evalc(expression)",
        inputs: &EVAL_INPUTS,
        outputs: &EVALC_OUTPUTS,
    },
];
const EVALIN_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "[varargout] = evalin(workspace, source)",
    inputs: &EVALIN_INPUTS,
    outputs: &DYNAMIC_WORKSPACE_OUTPUT,
}];
const ASSIGNIN_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "assignin(workspace, name, value)",
    inputs: &ASSIGNIN_INPUTS,
    outputs: &[],
}];

pub const DYNAMIC_WORKSPACE_ERROR_REQUIRES_VM: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.DYNAMIC_WORKSPACE.REQUIRES_VM",
    identifier: Some("RunMat:DynamicWorkspaceRequiresVm"),
    when:
        "A dynamic workspace builtin is dispatched without an active workspace execution context.",
    message: "dynamic workspace builtin requires an active workspace execution context",
};
pub const DYNAMIC_WORKSPACE_ERRORS: [BuiltinErrorDescriptor; 1] =
    [DYNAMIC_WORKSPACE_ERROR_REQUIRES_VM];
pub const ASSIGNIN_ERROR_TOO_MANY_OUTPUTS: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.ASSIGNIN.TOO_MANY_OUTPUTS",
    identifier: Some("RunMat:assignin:TooManyOutputs"),
    when: "One or more outputs are requested from assignin.",
    message: "assignin: too many output arguments",
};
pub const ASSIGNIN_ERROR_INVALID_NAME: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.ASSIGNIN.INVALID_NAME",
    identifier: Some("RunMat:DynamicWorkspaceName"),
    when: "The variable-name argument is not a valid MATLAB variable name.",
    message: "assignin: invalid workspace variable name",
};
pub const ASSIGNIN_ERRORS: [BuiltinErrorDescriptor; 3] = [
    DYNAMIC_WORKSPACE_ERROR_REQUIRES_VM,
    ASSIGNIN_ERROR_TOO_MANY_OUTPUTS,
    ASSIGNIN_ERROR_INVALID_NAME,
];

pub const EVAL_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &EVAL_SIGNATURES,
    output_mode: BuiltinOutputMode::ByRequestedOutputCount,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &DYNAMIC_WORKSPACE_ERRORS,
};
pub const EVALC_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &EVALC_SIGNATURES,
    output_mode: BuiltinOutputMode::ByRequestedOutputCount,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &DYNAMIC_WORKSPACE_ERRORS,
};
pub const EVALIN_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &EVALIN_SIGNATURES,
    output_mode: BuiltinOutputMode::ByRequestedOutputCount,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &DYNAMIC_WORKSPACE_ERRORS,
};
pub const ASSIGNIN_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &ASSIGNIN_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ASSIGNIN_ERRORS,
};

const ASSIGNIN_INTEGER_VALUE_INPUT: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "value",
        classes: &ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "The assigned value may use any real integer class; assignin performs no numeric conversion.",
    }];
pub const ASSIGNIN_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 1] =
    [BuiltinIntegerCapabilityDescriptor {
        form: "assignin(workspace, name, integer_value)",
        inputs: &ASSIGNIN_INTEGER_VALUE_INPUT,
        computation_domain: BuiltinIntegerComputationDomain::Structural,
        output_class: BuiltinIntegerOutputClassRule::NotApplicable,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::HostAndGpu,
        overload: BuiltinIntegerOverloadKind::Multiple,
        notes: "The already evaluated value retains its integer class and storage owner; assignin has no output.",
    }];

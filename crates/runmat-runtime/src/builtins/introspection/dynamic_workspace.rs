use runmat_builtins::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor, BuiltinOutputMode,
    BuiltinParamArity, BuiltinParamDescriptor, BuiltinParamType, BuiltinSignatureDescriptor, Value,
};
use runmat_hir::{ASSIGNIN_BUILTIN_NAME, EVALIN_BUILTIN_NAME, EVAL_BUILTIN_NAME};

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
    when: "A dynamic workspace builtin is dispatched outside the VM workspace frame.",
    message: "dynamic workspace builtin requires VM workspace context",
};

pub const DYNAMIC_WORKSPACE_ERRORS: [BuiltinErrorDescriptor; 1] =
    [DYNAMIC_WORKSPACE_ERROR_REQUIRES_VM];

pub const EVAL_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &EVAL_SIGNATURES,
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
    errors: &DYNAMIC_WORKSPACE_ERRORS,
};

fn requires_vm_workspace_context(builtin: &'static str) -> crate::BuiltinResult<Value> {
    Err(crate::runtime_descriptor_error(
        builtin,
        &DYNAMIC_WORKSPACE_ERROR_REQUIRES_VM,
    ))
}

pub(crate) fn dispatch_eval(_args: Vec<Value>) -> crate::BuiltinResult<Value> {
    requires_vm_workspace_context(EVAL_BUILTIN_NAME)
}

pub(crate) fn dispatch_evalin(_args: Vec<Value>) -> crate::BuiltinResult<Value> {
    requires_vm_workspace_context(EVALIN_BUILTIN_NAME)
}

pub(crate) fn dispatch_assignin(_args: Vec<Value>) -> crate::BuiltinResult<Value> {
    requires_vm_workspace_context(ASSIGNIN_BUILTIN_NAME)
}

#[runmat_macros::runtime_builtin(
    name = "eval",
    category = "introspection",
    summary = "Evaluate source text in the current workspace.",
    sink = true,
    descriptor(self::EVAL_DESCRIPTOR),
    builtin_path = "crate::builtins::introspection::dynamic_workspace"
)]
pub fn eval_builtin_registered(args: Vec<Value>) -> crate::BuiltinResult<Value> {
    dispatch_eval(args)
}

#[runmat_macros::runtime_builtin(
    name = "evalin",
    category = "introspection",
    summary = "Evaluate source text in a selected workspace.",
    sink = true,
    descriptor(self::EVALIN_DESCRIPTOR),
    builtin_path = "crate::builtins::introspection::dynamic_workspace"
)]
pub fn evalin_builtin_registered(args: Vec<Value>) -> crate::BuiltinResult<Value> {
    dispatch_evalin(args)
}

#[runmat_macros::runtime_builtin(
    name = "assignin",
    category = "introspection",
    summary = "Assign a value in a selected workspace.",
    sink = true,
    suppress_auto_output = true,
    descriptor(self::ASSIGNIN_DESCRIPTOR),
    builtin_path = "crate::builtins::introspection::dynamic_workspace"
)]
pub fn assignin_builtin_registered(args: Vec<Value>) -> crate::BuiltinResult<Value> {
    dispatch_assignin(args)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn runtime_dynamic_workspace_fallback_requires_vm_context() {
        for dispatch in [
            dispatch_eval as fn(Vec<Value>) -> crate::BuiltinResult<Value>,
            dispatch_evalin,
            dispatch_assignin,
        ] {
            let err = dispatch(Vec::new()).expect_err("runtime fallback should fail");
            assert_eq!(err.identifier(), Some("RunMat:DynamicWorkspaceRequiresVm"));
        }
    }
}

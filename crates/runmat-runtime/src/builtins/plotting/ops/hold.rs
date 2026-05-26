//! MATLAB-compatible `hold` builtin.

use runmat_builtins::Value;
use runmat_builtins::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor, BuiltinOutputMode,
    BuiltinParamArity, BuiltinParamDescriptor, BuiltinParamType, BuiltinSignatureDescriptor,
};
use runmat_macros::runtime_builtin;

use super::op_common::cmd_parsing::parse_hold_mode;
use super::state::{set_hold, HoldMode};
use crate::builtins::plotting::type_resolvers::bool_type;

const HOLD_OUTPUT_STATUS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "enabled",
    ty: BuiltinParamType::LogicalArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "True when hold is enabled after command execution.",
}];

const HOLD_INPUTS_NONE: [BuiltinParamDescriptor; 0] = [];

const HOLD_INPUTS_MODE: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "mode",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Optional,
    default: Some("\"toggle\""),
    description: "Hold mode ('on'|'off'|'all') or numeric/logical scalar.",
}];

const HOLD_SIGNATURES: [BuiltinSignatureDescriptor; 2] = [
    BuiltinSignatureDescriptor {
        label: "enabled = hold()",
        inputs: &HOLD_INPUTS_NONE,
        outputs: &HOLD_OUTPUT_STATUS,
    },
    BuiltinSignatureDescriptor {
        label: "enabled = hold(mode)",
        inputs: &HOLD_INPUTS_MODE,
        outputs: &HOLD_OUTPUT_STATUS,
    },
];

const HOLD_ERROR_INVALID_ARGUMENT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.HOLD.INVALID_ARGUMENT",
    identifier: Some("RunMat:hold:InvalidArgument"),
    when: "Hold mode argument is unsupported.",
    message: "hold: invalid argument",
};

const HOLD_ERRORS: [BuiltinErrorDescriptor; 1] = [HOLD_ERROR_INVALID_ARGUMENT];

pub const HOLD_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &HOLD_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &HOLD_ERRORS,
};

#[runtime_builtin(
    name = "hold",
    category = "plotting",
    summary = "Toggle whether plots replace or append to the current axes.",
    keywords = "hold,plotting",
    suppress_auto_output = true,
    type_resolver(bool_type),
    descriptor(crate::builtins::plotting::hold::HOLD_DESCRIPTOR),
    builtin_path = "crate::builtins::plotting::hold"
)]
pub fn hold_builtin(rest: Vec<Value>) -> crate::BuiltinResult<bool> {
    let mode = if rest.is_empty() {
        HoldMode::Toggle
    } else {
        parse_hold_mode(&rest[0])?
    };
    let enabled = set_hold(mode);
    Ok(enabled)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hold_descriptor_signatures_cover_core_forms() {
        let labels: Vec<&str> = HOLD_DESCRIPTOR
            .signatures
            .iter()
            .map(|sig| sig.label)
            .collect();
        assert!(labels.contains(&"enabled = hold()"));
        assert!(labels.contains(&"enabled = hold(mode)"));
    }

    #[test]
    fn hold_toggle_and_explicit_modes_work() {
        let _ = hold_builtin(Vec::new()).unwrap();
        let on = hold_builtin(vec![Value::String("on".into())]).unwrap();
        assert!(on);
        let off = hold_builtin(vec![Value::String("off".into())]).unwrap();
        assert!(!off);
    }
}

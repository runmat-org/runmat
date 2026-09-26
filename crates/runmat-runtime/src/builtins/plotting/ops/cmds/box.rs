use super::error::cmd_error_with_message;
use crate::builtins::plotting::op_common::cmd_parsing::parse_on_off;
use crate::builtins::plotting::state::{set_box_enabled, toggle_box};
use crate::builtins::plotting::type_resolvers::bool_type;
use runmat_builtins::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor, BuiltinOutputMode,
    BuiltinParamArity, BuiltinParamDescriptor, BuiltinParamType, BuiltinSignatureDescriptor,
};
use runmat_macros::runtime_builtin;
use runmat_value::Value;
const BOX_OUTPUT_ENABLED: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "enabled",
    ty: BuiltinParamType::LogicalArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Box outline enabled state after command execution.",
}];
const BOX_INPUTS_NONE: [BuiltinParamDescriptor; 0] = [];
const BOX_INPUTS_MODE: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "mode",
    ty: BuiltinParamType::StringScalar,
    arity: BuiltinParamArity::Optional,
    default: Some("\"toggle\""),
    description: "Box mode token ('on'|'off').",
}];
const BOX_SIGNATURES: [BuiltinSignatureDescriptor; 2] = [
    BuiltinSignatureDescriptor {
        label: "enabled = box()",
        inputs: &BOX_INPUTS_NONE,
        outputs: &BOX_OUTPUT_ENABLED,
    },
    BuiltinSignatureDescriptor {
        label: "enabled = box(mode)",
        inputs: &BOX_INPUTS_MODE,
        outputs: &BOX_OUTPUT_ENABLED,
    },
];
const BOX_ERROR_INVALID_ARGUMENT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.BOX.INVALID_ARGUMENT",
    identifier: Some("RunMat:box:InvalidArgument"),
    when: "Box mode argument is unsupported.",
    message: "box: invalid argument",
};
const BOX_ERRORS: [BuiltinErrorDescriptor; 1] = [BOX_ERROR_INVALID_ARGUMENT];
pub const BOX_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &BOX_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &BOX_ERRORS,
};

#[runtime_builtin(
    name = "box",
    category = "plotting",
    summary = "Toggle axes box outlines.",
    keywords = "box,plotting",
    suppress_auto_output = true,
    type_resolver(bool_type),
    descriptor(crate::builtins::plotting::cmds::BOX_DESCRIPTOR),
    builtin_path = "crate::builtins::plotting::cmds"
)]
pub fn box_builtin(args: Vec<Value>) -> crate::BuiltinResult<bool> {
    match parse_on_off("box", args.first()).map_err(|err| {
        cmd_error_with_message(
            "box",
            format!("{}: {}", BOX_ERROR_INVALID_ARGUMENT.message, err.message()),
            &BOX_ERROR_INVALID_ARGUMENT,
        )
    })? {
        Some(enabled) => {
            set_box_enabled(enabled);
            Ok(enabled)
        }
        None => {
            let enabled = toggle_box();
            Ok(enabled)
        }
    }
}

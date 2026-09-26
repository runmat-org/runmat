use super::error::cmd_error_with_message;
use crate::builtins::plotting::op_common::cmd_parsing::parse_on_off;
use crate::builtins::plotting::state::{set_colorbar_enabled, toggle_colorbar};
use crate::builtins::plotting::type_resolvers::bool_type;
use runmat_builtins::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor, BuiltinOutputMode,
    BuiltinParamArity, BuiltinParamDescriptor, BuiltinParamType, BuiltinSignatureDescriptor,
};
use runmat_macros::runtime_builtin;
use runmat_value::Value;
#[cfg(test)]
#[path = "colorbar/tests.rs"]
mod tests;
const COLORBAR_OUTPUT_ENABLED: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "enabled",
    ty: BuiltinParamType::LogicalArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Colorbar enabled state after command execution.",
}];
const COLORBAR_INPUTS_NONE: [BuiltinParamDescriptor; 0] = [];
const COLORBAR_INPUTS_MODE: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "mode",
    ty: BuiltinParamType::StringScalar,
    arity: BuiltinParamArity::Optional,
    default: Some("\"toggle\""),
    description: "Colorbar mode token ('on'|'off').",
}];
const COLORBAR_SIGNATURES: [BuiltinSignatureDescriptor; 2] = [
    BuiltinSignatureDescriptor {
        label: "enabled = colorbar()",
        inputs: &COLORBAR_INPUTS_NONE,
        outputs: &COLORBAR_OUTPUT_ENABLED,
    },
    BuiltinSignatureDescriptor {
        label: "enabled = colorbar(mode)",
        inputs: &COLORBAR_INPUTS_MODE,
        outputs: &COLORBAR_OUTPUT_ENABLED,
    },
];
const COLORBAR_ERROR_INVALID_ARGUMENT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.COLORBAR.INVALID_ARGUMENT",
    identifier: Some("RunMat:colorbar:InvalidArgument"),
    when: "Colorbar mode argument is unsupported.",
    message: "colorbar: invalid argument",
};
const COLORBAR_ERRORS: [BuiltinErrorDescriptor; 1] = [COLORBAR_ERROR_INVALID_ARGUMENT];
pub const COLORBAR_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &COLORBAR_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &COLORBAR_ERRORS,
};

#[runtime_builtin(
    name = "colorbar",
    category = "plotting",
    summary = "Show, hide, or toggle colorbars.",
    keywords = "colorbar,plotting",
    suppress_auto_output = true,
    type_resolver(bool_type),
    descriptor(crate::builtins::plotting::cmds::COLORBAR_DESCRIPTOR),
    builtin_path = "crate::builtins::plotting::cmds"
)]
pub fn colorbar_builtin(args: Vec<Value>) -> crate::BuiltinResult<bool> {
    if args.len() > 1 {
        return Err(cmd_error_with_message(
            "colorbar",
            "colorbar: expected at most one mode argument",
            &COLORBAR_ERROR_INVALID_ARGUMENT,
        ));
    }
    match parse_on_off("colorbar", args.first()).map_err(|err| {
        cmd_error_with_message(
            "colorbar",
            format!(
                "{}: {}",
                COLORBAR_ERROR_INVALID_ARGUMENT.message,
                err.message()
            ),
            &COLORBAR_ERROR_INVALID_ARGUMENT,
        )
    })? {
        Some(enabled) => {
            set_colorbar_enabled(enabled);
            Ok(enabled)
        }
        None => {
            let enabled = toggle_colorbar();
            Ok(enabled)
        }
    }
}

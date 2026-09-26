use super::error::cmd_error_with_message;
use crate::builtins::plotting::op_common::cmd_parsing::as_lower_str;
use crate::builtins::plotting::state::{
    set_grid_and_minor_grid_enabled, toggle_grid, toggle_minor_grid,
};
use crate::builtins::plotting::type_resolvers::bool_type;
use runmat_builtins::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor, BuiltinOutputMode,
    BuiltinParamArity, BuiltinParamDescriptor, BuiltinParamType, BuiltinSignatureDescriptor,
};
use runmat_macros::runtime_builtin;
use runmat_value::Value;
const GRID_OUTPUT_ENABLED: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "enabled",
    ty: BuiltinParamType::LogicalArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Grid enabled state after command execution.",
}];
const GRID_INPUTS_NONE: [BuiltinParamDescriptor; 0] = [];
const GRID_INPUTS_MODE: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "mode",
    ty: BuiltinParamType::StringScalar,
    arity: BuiltinParamArity::Optional,
    default: Some("\"toggle\""),
    description: "Grid mode token ('on'|'off'|'minor').",
}];
const GRID_SIGNATURES: [BuiltinSignatureDescriptor; 2] = [
    BuiltinSignatureDescriptor {
        label: "enabled = grid()",
        inputs: &GRID_INPUTS_NONE,
        outputs: &GRID_OUTPUT_ENABLED,
    },
    BuiltinSignatureDescriptor {
        label: "enabled = grid(mode)",
        inputs: &GRID_INPUTS_MODE,
        outputs: &GRID_OUTPUT_ENABLED,
    },
];
const GRID_ERROR_INVALID_ARGUMENT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.GRID.INVALID_ARGUMENT",
    identifier: Some("RunMat:grid:InvalidArgument"),
    when: "Grid mode argument is unsupported.",
    message: "grid: invalid argument",
};
const GRID_ERRORS: [BuiltinErrorDescriptor; 1] = [GRID_ERROR_INVALID_ARGUMENT];
pub const GRID_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &GRID_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &GRID_ERRORS,
};

enum GridMode {
    ToggleMajor,
    Major(bool),
    ToggleMinor,
}

#[runtime_builtin(
    name = "grid",
    category = "plotting",
    summary = "Toggle axes grid lines.",
    keywords = "grid,plotting",
    suppress_auto_output = true,
    type_resolver(bool_type),
    descriptor(crate::builtins::plotting::cmds::GRID_DESCRIPTOR),
    builtin_path = "crate::builtins::plotting::cmds"
)]
pub fn grid_builtin(args: Vec<Value>) -> crate::BuiltinResult<bool> {
    match parse_grid_mode(&args)? {
        GridMode::ToggleMajor => {
            let enabled = toggle_grid();
            Ok(enabled)
        }
        GridMode::Major(enabled) => {
            set_grid_and_minor_grid_enabled(enabled, if enabled { None } else { Some(false) });
            Ok(enabled)
        }
        GridMode::ToggleMinor => {
            let enabled = toggle_minor_grid();
            Ok(enabled)
        }
    }
}

fn parse_grid_mode(args: &[Value]) -> crate::BuiltinResult<GridMode> {
    if args.len() > 1 {
        return Err(cmd_error_with_message(
            "grid",
            format!(
                "{}: expected at most one mode argument",
                GRID_ERROR_INVALID_ARGUMENT.message
            ),
            &GRID_ERROR_INVALID_ARGUMENT,
        ));
    }

    let Some(arg) = args.first() else {
        return Ok(GridMode::ToggleMajor);
    };
    let Some(mode) = as_lower_str(arg) else {
        return Err(cmd_error_with_message(
            "grid",
            format!(
                "{}: expected string argument",
                GRID_ERROR_INVALID_ARGUMENT.message
            ),
            &GRID_ERROR_INVALID_ARGUMENT,
        ));
    };
    match mode.trim() {
        "on" => Ok(GridMode::Major(true)),
        "off" => Ok(GridMode::Major(false)),
        "minor" => Ok(GridMode::ToggleMinor),
        other => Err(cmd_error_with_message(
            "grid",
            format!(
                "{}: expected 'on', 'off', or 'minor' (got '{other}')",
                GRID_ERROR_INVALID_ARGUMENT.message
            ),
            &GRID_ERROR_INVALID_ARGUMENT,
        )),
    }
}
#[cfg(test)]
#[path = "grid/tests.rs"]
mod tests;

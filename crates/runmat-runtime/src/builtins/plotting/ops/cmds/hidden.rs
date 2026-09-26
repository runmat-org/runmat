use super::error::cmd_error_with_message;
use crate::builtins::plotting::op_common::cmd_parsing::parse_on_off;
use crate::builtins::plotting::properties::{resolve_plot_handle, PlotHandle};
use crate::builtins::plotting::state::{
    axes_metadata_snapshot, current_axes_state, current_figure_handle,
    set_hidden_line_removal_for_axes,
};
use crate::builtins::plotting::type_resolvers::bool_type;
use runmat_builtins::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor, BuiltinOutputMode,
    BuiltinParamArity, BuiltinParamDescriptor, BuiltinParamType, BuiltinSignatureDescriptor,
};
use runmat_macros::runtime_builtin;
use runmat_value::Value;
const HIDDEN_OUTPUT_ENABLED: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "enabled",
    ty: BuiltinParamType::LogicalArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Hidden-line-removal state after command execution.",
}];
const HIDDEN_INPUTS_NONE: [BuiltinParamDescriptor; 0] = [];
const HIDDEN_INPUTS_MODE: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "mode",
    ty: BuiltinParamType::StringScalar,
    arity: BuiltinParamArity::Optional,
    default: Some("\"toggle\""),
    description: "Hidden-line-removal mode token ('on'|'off').",
}];
const HIDDEN_INPUTS_AX: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "ax",
    ty: BuiltinParamType::AxesHandle,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Target axes handle.",
}];
const HIDDEN_INPUTS_AX_MODE: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "ax",
        ty: BuiltinParamType::AxesHandle,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Target axes handle.",
    },
    BuiltinParamDescriptor {
        name: "mode",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Optional,
        default: Some("\"toggle\""),
        description: "Hidden-line-removal mode token ('on'|'off').",
    },
];
const HIDDEN_SIGNATURES: [BuiltinSignatureDescriptor; 4] = [
    BuiltinSignatureDescriptor {
        label: "enabled = hidden()",
        inputs: &HIDDEN_INPUTS_NONE,
        outputs: &HIDDEN_OUTPUT_ENABLED,
    },
    BuiltinSignatureDescriptor {
        label: "enabled = hidden(mode)",
        inputs: &HIDDEN_INPUTS_MODE,
        outputs: &HIDDEN_OUTPUT_ENABLED,
    },
    BuiltinSignatureDescriptor {
        label: "enabled = hidden(ax)",
        inputs: &HIDDEN_INPUTS_AX,
        outputs: &HIDDEN_OUTPUT_ENABLED,
    },
    BuiltinSignatureDescriptor {
        label: "enabled = hidden(ax, mode)",
        inputs: &HIDDEN_INPUTS_AX_MODE,
        outputs: &HIDDEN_OUTPUT_ENABLED,
    },
];
const HIDDEN_ERROR_INVALID_ARGUMENT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.HIDDEN.INVALID_ARGUMENT",
    identifier: Some("RunMat:hidden:InvalidArgument"),
    when: "Hidden-line-removal mode or axes target is unsupported.",
    message: "hidden: invalid argument",
};
const HIDDEN_ERRORS: [BuiltinErrorDescriptor; 1] = [HIDDEN_ERROR_INVALID_ARGUMENT];
pub const HIDDEN_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &HIDDEN_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &HIDDEN_ERRORS,
};
type HiddenAxesTarget = Option<(crate::builtins::plotting::state::FigureHandle, usize)>;

#[runtime_builtin(
    name = "hidden",
    category = "plotting",
    summary = "Set axes hidden-line-removal state.",
    keywords = "hidden,hiddenline,plotting,surface,mesh",
    suppress_auto_output = true,
    type_resolver(bool_type),
    descriptor(crate::builtins::plotting::cmds::HIDDEN_DESCRIPTOR),
    builtin_path = "crate::builtins::plotting::cmds"
)]
pub fn hidden_builtin(args: Vec<Value>) -> crate::BuiltinResult<bool> {
    let (target, rest) = parse_hidden_target(args)?;
    let (handle, axes_index) = target.unwrap_or_else(|| {
        let axes = current_axes_state();
        (current_figure_handle(), axes.active_index)
    });
    let requested = parse_on_off("hidden", rest.first()).map_err(|err| {
        cmd_error_with_message(
            "hidden",
            format!(
                "{}: {}",
                HIDDEN_ERROR_INVALID_ARGUMENT.message,
                err.message()
            ),
            &HIDDEN_ERROR_INVALID_ARGUMENT,
        )
    })?;
    if rest.len() > 1 {
        return Err(cmd_error_with_message(
            "hidden",
            "hidden: expected at most one mode argument after optional axes handle",
            &HIDDEN_ERROR_INVALID_ARGUMENT,
        ));
    }
    let current = axes_metadata_snapshot(handle, axes_index)
        .map_err(|err| {
            cmd_error_with_message(
                "hidden",
                format!("{}: {err}", HIDDEN_ERROR_INVALID_ARGUMENT.message),
                &HIDDEN_ERROR_INVALID_ARGUMENT,
            )
        })?
        .hidden_line_removal;
    let enabled = requested.unwrap_or(!current);
    set_hidden_line_removal_for_axes(handle, axes_index, enabled).map_err(|err| {
        cmd_error_with_message(
            "hidden",
            format!("{}: {err}", HIDDEN_ERROR_INVALID_ARGUMENT.message),
            &HIDDEN_ERROR_INVALID_ARGUMENT,
        )
    })?;
    Ok(enabled)
}

fn parse_hidden_target(args: Vec<Value>) -> crate::BuiltinResult<(HiddenAxesTarget, Vec<Value>)> {
    let mut iter = args.into_iter();
    let Some(first) = iter.next() else {
        return Ok((None, Vec::new()));
    };
    if let Ok(PlotHandle::Axes(handle, axes_index)) = resolve_plot_handle(&first, "hidden") {
        return Ok((Some((handle, axes_index)), iter.collect()));
    }
    let mut rest = Vec::with_capacity(iter.size_hint().0 + 1);
    rest.push(first);
    rest.extend(iter);
    Ok((None, rest))
}
#[cfg(test)]
#[path = "hidden/tests.rs"]
mod tests;

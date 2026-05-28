use runmat_builtins::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor, BuiltinOutputMode,
    BuiltinParamArity, BuiltinParamDescriptor, BuiltinParamType, BuiltinSignatureDescriptor, Value,
};
use runmat_macros::runtime_builtin;

use super::op_common::{map_figure_error, parse_text_command};
use super::state::set_ylabel_for_axes;
use crate::builtins::plotting::type_resolvers::handle_scalar_type;

const YLABEL_OUTPUT_HANDLE: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "h",
    ty: BuiltinParamType::NumericScalar,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Handle to the created/updated ylabel object.",
}];

const YLABEL_INPUTS_TEXT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "txt",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Label text (string/char/cellstr-like multiline forms).",
}];

const YLABEL_INPUTS_AX_TEXT: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "ax",
        ty: BuiltinParamType::AxesHandle,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Target axes handle.",
    },
    BuiltinParamDescriptor {
        name: "txt",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Label text (string/char/cellstr-like multiline forms).",
    },
];

const YLABEL_INPUTS_TEXT_PROPS: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "txt",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Label text (string/char/cellstr-like multiline forms).",
    },
    BuiltinParamDescriptor {
        name: "props",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Variadic,
        default: None,
        description: "Property/value pairs (Color, FontSize, FontWeight, etc.).",
    },
];

const YLABEL_INPUTS_AX_TEXT_PROPS: [BuiltinParamDescriptor; 3] = [
    BuiltinParamDescriptor {
        name: "ax",
        ty: BuiltinParamType::AxesHandle,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Target axes handle.",
    },
    BuiltinParamDescriptor {
        name: "txt",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Label text (string/char/cellstr-like multiline forms).",
    },
    BuiltinParamDescriptor {
        name: "props",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Variadic,
        default: None,
        description: "Property/value pairs (Color, FontSize, FontWeight, etc.).",
    },
];

const YLABEL_SIGNATURES: [BuiltinSignatureDescriptor; 4] = [
    BuiltinSignatureDescriptor {
        label: "h = ylabel(txt)",
        inputs: &YLABEL_INPUTS_TEXT,
        outputs: &YLABEL_OUTPUT_HANDLE,
    },
    BuiltinSignatureDescriptor {
        label: "h = ylabel(ax, txt)",
        inputs: &YLABEL_INPUTS_AX_TEXT,
        outputs: &YLABEL_OUTPUT_HANDLE,
    },
    BuiltinSignatureDescriptor {
        label: "h = ylabel(txt, Name, Value, ...)",
        inputs: &YLABEL_INPUTS_TEXT_PROPS,
        outputs: &YLABEL_OUTPUT_HANDLE,
    },
    BuiltinSignatureDescriptor {
        label: "h = ylabel(ax, txt, Name, Value, ...)",
        inputs: &YLABEL_INPUTS_AX_TEXT_PROPS,
        outputs: &YLABEL_OUTPUT_HANDLE,
    },
];

const YLABEL_ERROR_INVALID_ARGUMENT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.YLABEL.INVALID_ARGUMENT",
    identifier: Some("RunMat:ylabel:InvalidArgument"),
    when: "Axes handle, text payload, or property/value arguments are invalid.",
    message: "ylabel: invalid argument",
};

const YLABEL_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.YLABEL.INTERNAL",
    identifier: Some("RunMat:ylabel:Internal"),
    when: "Internal plotting state update fails.",
    message: "ylabel: internal operation failed",
};

const YLABEL_ERRORS: [BuiltinErrorDescriptor; 2] =
    [YLABEL_ERROR_INVALID_ARGUMENT, YLABEL_ERROR_INTERNAL];

pub const YLABEL_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &YLABEL_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &YLABEL_ERRORS,
};

#[runtime_builtin(
    name = "ylabel",
    category = "plotting",
    summary = "Set the current axes y-axis label.",
    keywords = "ylabel,plotting",
    suppress_auto_output = true,
    type_resolver(handle_scalar_type),
    descriptor(crate::builtins::plotting::ylabel::YLABEL_DESCRIPTOR),
    builtin_path = "crate::builtins::plotting::ylabel"
)]
pub fn ylabel_builtin(args: Vec<Value>) -> crate::BuiltinResult<f64> {
    let command = parse_text_command("ylabel", &args)?;
    set_ylabel_for_axes(
        command.target.0,
        command.target.1,
        &command.text,
        command.style,
    )
    .map_err(|err| map_figure_error("ylabel", err))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::builtins::plotting::tests::{ensure_plot_test_env, lock_plot_registry};
    use crate::builtins::plotting::{clear_figure, reset_hold_state_for_run};

    #[test]
    fn ylabel_descriptor_signatures_cover_core_forms() {
        let labels: Vec<&str> = YLABEL_DESCRIPTOR
            .signatures
            .iter()
            .map(|sig| sig.label)
            .collect();
        assert!(labels.contains(&"h = ylabel(txt)"));
        assert!(labels.contains(&"h = ylabel(ax, txt)"));
        assert!(labels.contains(&"h = ylabel(txt, Name, Value, ...)"));
    }

    #[test]
    fn ylabel_rejects_invalid_axes_handle() {
        let _guard = lock_plot_registry();
        ensure_plot_test_env();
        reset_hold_state_for_run();
        let _ = clear_figure(None);

        let err = ylabel_builtin(vec![
            Value::Num(crate::builtins::plotting::state::encode_axes_handle(
                crate::builtins::plotting::current_figure_handle(),
                42,
            )),
            Value::String("Amp".into()),
        ])
        .unwrap_err();
        assert!(err.message.contains("invalid axes") || err.message.contains("out of range"));
    }
}

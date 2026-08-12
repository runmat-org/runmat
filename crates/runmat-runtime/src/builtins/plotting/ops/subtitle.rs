use runmat_builtins::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor, BuiltinOutputMode,
    BuiltinParamArity, BuiltinParamDescriptor, BuiltinParamType, BuiltinSignatureDescriptor,
};
use runmat_macros::runtime_builtin;
use runmat_value::Value;

use super::op_common::{map_figure_error, parse_text_command};
use super::state::set_figure_subtitle_for_axes;
use crate::builtins::plotting::type_resolvers::handle_scalar_type;

const SUBTITLE_OUTPUT_HANDLE: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "h",
    ty: BuiltinParamType::NumericScalar,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Handle to the created/updated subtitle object.",
}];

const SUBTITLE_INPUTS_TEXT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "txt",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Subtitle text (string/char/cellstr-like multiline forms).",
}];

const SUBTITLE_INPUTS_AX_TEXT: [BuiltinParamDescriptor; 2] = [
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
        description: "Subtitle text (string/char/cellstr-like multiline forms).",
    },
];

const SUBTITLE_INPUTS_TEXT_PROPS: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "txt",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Subtitle text (string/char/cellstr-like multiline forms).",
    },
    BuiltinParamDescriptor {
        name: "props",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Variadic,
        default: None,
        description: "Property/value pairs (Color, FontSize, FontWeight, etc.).",
    },
];

const SUBTITLE_INPUTS_AX_TEXT_PROPS: [BuiltinParamDescriptor; 3] = [
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
        description: "Subtitle text (string/char/cellstr-like multiline forms).",
    },
    BuiltinParamDescriptor {
        name: "props",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Variadic,
        default: None,
        description: "Property/value pairs (Color, FontSize, FontWeight, etc.).",
    },
];

const SUBTITLE_SIGNATURES: [BuiltinSignatureDescriptor; 4] = [
    BuiltinSignatureDescriptor {
        label: "h = subtitle(txt)",
        inputs: &SUBTITLE_INPUTS_TEXT,
        outputs: &SUBTITLE_OUTPUT_HANDLE,
    },
    BuiltinSignatureDescriptor {
        label: "h = subtitle(ax, txt)",
        inputs: &SUBTITLE_INPUTS_AX_TEXT,
        outputs: &SUBTITLE_OUTPUT_HANDLE,
    },
    BuiltinSignatureDescriptor {
        label: "h = subtitle(txt, Name, Value, ...)",
        inputs: &SUBTITLE_INPUTS_TEXT_PROPS,
        outputs: &SUBTITLE_OUTPUT_HANDLE,
    },
    BuiltinSignatureDescriptor {
        label: "h = subtitle(ax, txt, Name, Value, ...)",
        inputs: &SUBTITLE_INPUTS_AX_TEXT_PROPS,
        outputs: &SUBTITLE_OUTPUT_HANDLE,
    },
];

const SUBTITLE_ERROR_INVALID_ARGUMENT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.SUBTITLE.INVALID_ARGUMENT",
    identifier: Some("RunMat:subtitle:InvalidArgument"),
    when: "Axes handle, text payload, or property/value arguments are invalid.",
    message: "subtitle: invalid argument",
};

const SUBTITLE_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.SUBTITLE.INTERNAL",
    identifier: Some("RunMat:subtitle:Internal"),
    when: "Internal plotting state update fails.",
    message: "subtitle: internal operation failed",
};

const SUBTITLE_ERRORS: [BuiltinErrorDescriptor; 2] =
    [SUBTITLE_ERROR_INVALID_ARGUMENT, SUBTITLE_ERROR_INTERNAL];

pub const SUBTITLE_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &SUBTITLE_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &SUBTITLE_ERRORS,
};

#[runtime_builtin(
    name = "subtitle",
    category = "plotting",
    summary = "Set subtitle text for the current or specified axes.",
    keywords = "subtitle,plotting,title",
    suppress_auto_output = true,
    type_resolver(handle_scalar_type),
    descriptor(crate::builtins::plotting::subtitle::SUBTITLE_DESCRIPTOR),
    builtin_path = "crate::builtins::plotting::subtitle"
)]
pub fn subtitle_builtin(args: Vec<Value>) -> crate::BuiltinResult<f64> {
    let command = parse_text_command("subtitle", &args)?;
    set_figure_subtitle_for_axes(
        command.target.0,
        command.target.1,
        &command.text,
        command.style,
    )
    .map_err(|err| map_figure_error("subtitle", err))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::builtins::plotting::get::get_builtin;
    use crate::builtins::plotting::state::PlotObjectKind;
    use crate::builtins::plotting::tests::{ensure_plot_test_env, lock_plot_registry};
    use crate::builtins::plotting::{
        clear_figure, clone_figure, current_figure_handle, reset_hold_state_for_run,
    };
    use runmat_value::{CellArray, Value};

    fn setup_plot_tests() -> crate::builtins::plotting::state::PlotTestLockGuard {
        let guard = lock_plot_registry();
        ensure_plot_test_env();
        reset_hold_state_for_run();
        let _ = clear_figure(None);
        guard
    }

    #[test]
    fn subtitle_descriptor_signatures_cover_core_forms() {
        let labels: Vec<&str> = SUBTITLE_DESCRIPTOR
            .signatures
            .iter()
            .map(|sig| sig.label)
            .collect();
        assert!(labels.contains(&"h = subtitle(txt)"));
        assert!(labels.contains(&"h = subtitle(ax, txt)"));
        assert!(labels.contains(&"h = subtitle(txt, Name, Value, ...)"));
        assert!(labels.contains(&"h = subtitle(ax, txt, Name, Value, ...)"));
    }

    #[test]
    fn subtitle_returns_text_handle_and_updates_active_axes() {
        let _guard = setup_plot_tests();
        let handle = subtitle_builtin(vec![Value::String("Slope = 2".into())]).unwrap();
        let (figure, axes, kind) =
            crate::builtins::plotting::state::decode_plot_object_handle(handle).unwrap();
        assert_eq!(figure, current_figure_handle());
        assert_eq!(axes, 0);
        assert_eq!(kind, PlotObjectKind::Subtitle);

        let fig = clone_figure(figure).unwrap();
        assert_eq!(
            fig.axes_metadata(0).and_then(|m| m.subtitle.as_deref()),
            Some("Slope = 2")
        );
        assert!(fig.has_any_titles());
    }

    #[test]
    fn subtitle_accepts_axes_target_multiline_text_and_properties() {
        let _guard = setup_plot_tests();
        let ax = crate::builtins::plotting::subplot::subplot_builtin(
            Value::Num(1.0),
            Value::Num(2.0),
            Value::Num(2.0),
        )
        .unwrap();
        let lines = Value::Cell(
            CellArray::new(
                vec![Value::String("left".into()), Value::String("right".into())],
                1,
                2,
            )
            .unwrap(),
        );
        subtitle_builtin(vec![
            Value::Num(ax),
            lines,
            Value::String("FontSize".into()),
            Value::Num(13.0),
            Value::String("FontWeight".into()),
            Value::String("bold".into()),
            Value::String("Visible".into()),
            Value::Bool(false),
        ])
        .unwrap();

        let fig = clone_figure(current_figure_handle()).unwrap();
        let meta = fig.axes_metadata(1).unwrap();
        assert_eq!(meta.subtitle.as_deref(), Some("left\nright"));
        assert_eq!(meta.subtitle_style.font_size, Some(13.0));
        assert_eq!(meta.subtitle_style.font_weight.as_deref(), Some("bold"));
        assert!(!meta.subtitle_style.visible);
    }

    #[test]
    fn get_axes_and_subtitle_handle_expose_subtitle() {
        let _guard = setup_plot_tests();
        let handle = subtitle_builtin(vec![Value::String("Details".into())]).unwrap();
        let ax = crate::builtins::plotting::gca::gca_builtin(Vec::new()).unwrap();

        let subtitle_prop =
            get_builtin(vec![ax.clone(), Value::String("Subtitle".into())]).unwrap();
        assert_eq!(subtitle_prop, Value::Num(handle));

        let string_prop =
            get_builtin(vec![Value::Num(handle), Value::String("String".into())]).unwrap();
        assert_eq!(string_prop, Value::String("Details".into()));
    }

    #[test]
    fn subtitle_rejects_bad_properties() {
        let _guard = setup_plot_tests();
        let err = subtitle_builtin(vec![
            Value::String("Oops".into()),
            Value::String("FontSize".into()),
            Value::String("large".into()),
        ])
        .unwrap_err();
        assert!(err.message.contains("FontSize must be numeric"));
    }
}

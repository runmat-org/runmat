use runmat_builtins::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor, BuiltinOutputMode,
    BuiltinParamArity, BuiltinParamDescriptor, BuiltinParamType, BuiltinSignatureDescriptor, Value,
};
use runmat_macros::runtime_builtin;

use super::op_common::{map_figure_error, parse_text_command};
use super::state::set_figure_title_for_axes;
use crate::builtins::plotting::type_resolvers::handle_scalar_type;

const TITLE_OUTPUT_HANDLE: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "h",
    ty: BuiltinParamType::NumericScalar,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Handle to the created/updated title object.",
}];

const TITLE_INPUTS_TEXT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "txt",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Title text (string/char/cellstr-like multiline forms).",
}];

const TITLE_INPUTS_AX_TEXT: [BuiltinParamDescriptor; 2] = [
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
        description: "Title text (string/char/cellstr-like multiline forms).",
    },
];

const TITLE_INPUTS_TEXT_PROPS: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "txt",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Title text (string/char/cellstr-like multiline forms).",
    },
    BuiltinParamDescriptor {
        name: "props",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Variadic,
        default: None,
        description: "Property/value pairs (Color, FontSize, FontWeight, etc.).",
    },
];

const TITLE_INPUTS_AX_TEXT_PROPS: [BuiltinParamDescriptor; 3] = [
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
        description: "Title text (string/char/cellstr-like multiline forms).",
    },
    BuiltinParamDescriptor {
        name: "props",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Variadic,
        default: None,
        description: "Property/value pairs (Color, FontSize, FontWeight, etc.).",
    },
];

const TITLE_SIGNATURES: [BuiltinSignatureDescriptor; 4] = [
    BuiltinSignatureDescriptor {
        label: "h = title(txt)",
        inputs: &TITLE_INPUTS_TEXT,
        outputs: &TITLE_OUTPUT_HANDLE,
    },
    BuiltinSignatureDescriptor {
        label: "h = title(ax, txt)",
        inputs: &TITLE_INPUTS_AX_TEXT,
        outputs: &TITLE_OUTPUT_HANDLE,
    },
    BuiltinSignatureDescriptor {
        label: "h = title(txt, Name, Value, ...)",
        inputs: &TITLE_INPUTS_TEXT_PROPS,
        outputs: &TITLE_OUTPUT_HANDLE,
    },
    BuiltinSignatureDescriptor {
        label: "h = title(ax, txt, Name, Value, ...)",
        inputs: &TITLE_INPUTS_AX_TEXT_PROPS,
        outputs: &TITLE_OUTPUT_HANDLE,
    },
];

const TITLE_ERROR_INVALID_ARGUMENT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.TITLE.INVALID_ARGUMENT",
    identifier: Some("RunMat:title:InvalidArgument"),
    when: "Axes handle, text payload, or property/value arguments are invalid.",
    message: "title: invalid argument",
};

const TITLE_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.TITLE.INTERNAL",
    identifier: Some("RunMat:title:Internal"),
    when: "Internal plotting state update fails.",
    message: "title: internal operation failed",
};

const TITLE_ERRORS: [BuiltinErrorDescriptor; 2] =
    [TITLE_ERROR_INVALID_ARGUMENT, TITLE_ERROR_INTERNAL];

pub const TITLE_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &TITLE_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &TITLE_ERRORS,
};

#[runtime_builtin(
    name = "title",
    category = "plotting",
    summary = "Set the current axes title.",
    keywords = "title,plotting",
    suppress_auto_output = true,
    type_resolver(handle_scalar_type),
    descriptor(crate::builtins::plotting::title::TITLE_DESCRIPTOR),
    builtin_path = "crate::builtins::plotting::title"
)]
pub fn title_builtin(args: Vec<Value>) -> crate::BuiltinResult<f64> {
    let command = parse_text_command("title", &args)?;
    set_figure_title_for_axes(
        command.target.0,
        command.target.1,
        &command.text,
        command.style,
    )
    .map_err(|err| map_figure_error("title", err))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::builtins::plotting::state::PlotTestLockGuard;
    use crate::builtins::plotting::state::{decode_plot_object_handle, PlotObjectKind};
    use crate::builtins::plotting::tests::{ensure_plot_test_env, lock_plot_registry};
    use crate::builtins::plotting::{
        clear_figure, clone_figure, current_figure_handle, reset_hold_state_for_run,
    };
    use runmat_builtins::{CellArray, CharArray, StringArray, Value};
    use runmat_plot::plots::Figure;

    fn setup_plot_tests() -> PlotTestLockGuard {
        let guard = lock_plot_registry();
        ensure_plot_test_env();
        reset_hold_state_for_run();
        let _ = clear_figure(None);
        guard
    }

    #[test]
    fn title_descriptor_signatures_cover_core_forms() {
        let labels: Vec<&str> = TITLE_DESCRIPTOR
            .signatures
            .iter()
            .map(|sig| sig.label)
            .collect();
        assert!(labels.contains(&"h = title(txt)"));
        assert!(labels.contains(&"h = title(ax, txt)"));
        assert!(labels.contains(&"h = title(txt, Name, Value, ...)"));
    }

    #[test]
    fn title_returns_plot_object_handle_and_updates_active_axes() {
        let _guard = setup_plot_tests();
        let handle = title_builtin(vec![Value::String("Signal".into())]).unwrap();
        let (figure, axes, kind) = decode_plot_object_handle(handle).unwrap();
        assert_eq!(figure, current_figure_handle());
        assert_eq!(axes, 0);
        assert_eq!(kind, PlotObjectKind::Title);

        let fig = clone_figure(figure).unwrap();
        assert_eq!(
            fig.axes_metadata(0).and_then(|m| m.title.as_deref()),
            Some("Signal")
        );
    }

    #[test]
    fn title_accepts_axes_target_and_char_array_with_properties() {
        let _guard = setup_plot_tests();
        let mut figure = Figure::new();
        figure.set_subplot_grid(1, 2);
        let handle = crate::builtins::plotting::state::import_figure(figure);
        let ax = Value::Num(crate::builtins::plotting::state::encode_axes_handle(
            handle, 1,
        ));
        let title = Value::CharArray(CharArray {
            data: "Right".chars().collect(),
            rows: 1,
            cols: 5,
        });
        title_builtin(vec![
            ax,
            title,
            Value::String("FontSize".into()),
            Value::Num(18.0),
            Value::String("Visible".into()),
            Value::Bool(false),
        ])
        .unwrap();

        let fig = clone_figure(handle).unwrap();
        assert_eq!(
            fig.axes_metadata(1).and_then(|m| m.title.as_deref()),
            Some("Right")
        );
        let meta = fig.axes_metadata(1).unwrap();
        assert_eq!(meta.title_style.font_size, Some(18.0));
        assert!(!meta.title_style.visible);
    }

    #[test]
    fn title_rejects_invalid_axes_handle_and_bad_properties() {
        let _guard = setup_plot_tests();

        let invalid_handle = Value::Num(crate::builtins::plotting::state::encode_axes_handle(
            current_figure_handle(),
            99,
        ));
        let err = title_builtin(vec![invalid_handle, Value::String("Oops".into())]).unwrap_err();
        assert!(err.message.contains("invalid axes") || err.message.contains("out of range"));

        let err = title_builtin(vec![
            Value::String("Oops".into()),
            Value::String("Bogus".into()),
            Value::Num(1.0),
        ])
        .unwrap_err();
        assert!(err.message.contains("unsupported property"));

        let err = title_builtin(vec![
            Value::String("Oops".into()),
            Value::String("FontSize".into()),
            Value::String("big".into()),
        ])
        .unwrap_err();
        assert!(err.message.contains("FontSize must be numeric"));
    }

    #[test]
    fn title_accepts_multiline_text_inputs() {
        let _guard = setup_plot_tests();

        title_builtin(vec![Value::StringArray(StringArray {
            data: vec!["Line 1".into(), "Line 2".into()],
            shape: vec![1, 2],
            rows: 1,
            cols: 2,
        })])
        .unwrap();
        let fig = clone_figure(current_figure_handle()).unwrap();
        assert_eq!(
            fig.axes_metadata(0).and_then(|m| m.title.as_deref()),
            Some("Line 1\nLine 2")
        );

        let cell = CellArray::new(
            vec![Value::String("A".into()), Value::String("B".into())],
            1,
            2,
        )
        .unwrap();
        title_builtin(vec![Value::Cell(cell)]).unwrap();
        let fig = clone_figure(current_figure_handle()).unwrap();
        assert_eq!(
            fig.axes_metadata(0).and_then(|m| m.title.as_deref()),
            Some("A\nB")
        );
    }
}

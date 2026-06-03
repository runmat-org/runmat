use runmat_builtins::Value;
use runmat_builtins::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor, BuiltinOutputMode,
    BuiltinParamArity, BuiltinParamDescriptor, BuiltinParamType, BuiltinSignatureDescriptor,
};
use runmat_macros::runtime_builtin;

use super::op_common::parse_legend_command;
use super::state::{set_legend_for_axes, FigureError};
use crate::builtins::plotting::type_resolvers::handle_scalar_type;
use crate::{build_runtime_error, RuntimeError};

const BUILTIN_NAME: &str = "legend";

const LEGEND_OUTPUT_HANDLE: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "h",
    ty: BuiltinParamType::NumericScalar,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Legend object handle.",
}];

const LEGEND_INPUTS_NONE: [BuiltinParamDescriptor; 0] = [];

const LEGEND_INPUTS_AX: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "ax",
    ty: BuiltinParamType::AxesHandle,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Target axes handle.",
}];

const LEGEND_INPUTS_MODE: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "mode",
    ty: BuiltinParamType::StringScalar,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Legend mode token: 'on'|'off'|'show'|'hide'|'boxon'|'boxoff'.",
}];

const LEGEND_INPUTS_AX_MODE: [BuiltinParamDescriptor; 2] = [
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
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Legend mode token: 'on'|'off'|'show'|'hide'|'boxon'|'boxoff'.",
    },
];

const LEGEND_INPUTS_LABELS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "labels_or_pairs",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Variadic,
    default: None,
    description:
        "Labels (string/cell/string-array) with optional trailing Name/Value legend style pairs.",
}];

const LEGEND_INPUTS_AX_LABELS: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "ax",
        ty: BuiltinParamType::AxesHandle,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Target axes handle.",
    },
    BuiltinParamDescriptor {
        name: "labels_or_pairs",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Variadic,
        default: None,
        description:
            "Labels (string/cell/string-array) with optional trailing Name/Value legend style pairs.",
    },
];

const LEGEND_SIGNATURES: [BuiltinSignatureDescriptor; 6] = [
    BuiltinSignatureDescriptor {
        label: "h = legend()",
        inputs: &LEGEND_INPUTS_NONE,
        outputs: &LEGEND_OUTPUT_HANDLE,
    },
    BuiltinSignatureDescriptor {
        label: "h = legend(ax)",
        inputs: &LEGEND_INPUTS_AX,
        outputs: &LEGEND_OUTPUT_HANDLE,
    },
    BuiltinSignatureDescriptor {
        label: "h = legend(mode)",
        inputs: &LEGEND_INPUTS_MODE,
        outputs: &LEGEND_OUTPUT_HANDLE,
    },
    BuiltinSignatureDescriptor {
        label: "h = legend(ax, mode)",
        inputs: &LEGEND_INPUTS_AX_MODE,
        outputs: &LEGEND_OUTPUT_HANDLE,
    },
    BuiltinSignatureDescriptor {
        label: "h = legend(labels..., Name, Value, ...)",
        inputs: &LEGEND_INPUTS_LABELS,
        outputs: &LEGEND_OUTPUT_HANDLE,
    },
    BuiltinSignatureDescriptor {
        label: "h = legend(ax, labels..., Name, Value, ...)",
        inputs: &LEGEND_INPUTS_AX_LABELS,
        outputs: &LEGEND_OUTPUT_HANDLE,
    },
];

const LEGEND_ERROR_INVALID_ARGUMENT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.LEGEND.INVALID_ARGUMENT",
    identifier: Some("RunMat:legend:InvalidArgument"),
    when: "Legend arguments are malformed, unsupported, or have invalid labels/properties.",
    message: "legend: invalid argument",
};

const LEGEND_ERROR_INVALID_AXES: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.LEGEND.INVALID_AXES",
    identifier: Some("RunMat:legend:InvalidAxes"),
    when: "Resolved axes target is invalid or out of range.",
    message: "legend: invalid axes target",
};

const LEGEND_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.LEGEND.INTERNAL",
    identifier: Some("RunMat:legend:Internal"),
    when: "Internal legend state operation fails.",
    message: "legend: internal operation failed",
};

const LEGEND_ERRORS: [BuiltinErrorDescriptor; 3] = [
    LEGEND_ERROR_INVALID_ARGUMENT,
    LEGEND_ERROR_INVALID_AXES,
    LEGEND_ERROR_INTERNAL,
];

pub const LEGEND_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &LEGEND_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &LEGEND_ERRORS,
};

fn legend_error(error: &'static BuiltinErrorDescriptor) -> RuntimeError {
    legend_error_with_message(error.message, error)
}

fn legend_error_with_message(
    message: impl Into<String>,
    error: &'static BuiltinErrorDescriptor,
) -> RuntimeError {
    let mut builder = build_runtime_error(message).with_builtin(BUILTIN_NAME);
    if let Some(identifier) = error.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

fn map_legend_figure_error(err: FigureError) -> RuntimeError {
    match err {
        FigureError::InvalidHandle(_)
        | FigureError::InvalidAxesHandle
        | FigureError::InvalidSubplotIndex { .. } => legend_error(&LEGEND_ERROR_INVALID_AXES),
        other => legend_error_with_message(
            format!("{}: {}", LEGEND_ERROR_INTERNAL.message, other),
            &LEGEND_ERROR_INTERNAL,
        ),
    }
}

#[runtime_builtin(
    name = "legend",
    category = "plotting",
    summary = "Show, hide, or configure the current axes legend.",
    keywords = "legend,plotting",
    suppress_auto_output = true,
    type_resolver(handle_scalar_type),
    descriptor(crate::builtins::plotting::legend::LEGEND_DESCRIPTOR),
    builtin_path = "crate::builtins::plotting::legend"
)]
pub fn legend_builtin(args: Vec<Value>) -> crate::BuiltinResult<f64> {
    let command = parse_legend_command(BUILTIN_NAME, &args).map_err(|err| {
        legend_error_with_message(
            format!(
                "{}: {}",
                LEGEND_ERROR_INVALID_ARGUMENT.message,
                err.message()
            ),
            &LEGEND_ERROR_INVALID_ARGUMENT,
        )
    })?;
    set_legend_for_axes(
        command.target.0,
        command.target.1,
        command.enabled,
        command.labels.as_deref(),
        Some(command.style),
    )
    .map_err(map_legend_figure_error)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::builtins::plotting::op_common::vec4_eq;
    use crate::builtins::plotting::state::PlotTestLockGuard;
    use crate::builtins::plotting::state::{decode_plot_object_handle, PlotObjectKind};
    use crate::builtins::plotting::tests::{ensure_plot_test_env, lock_plot_registry};
    use crate::builtins::plotting::{clear_figure, clone_figure, reset_hold_state_for_run};
    use glam::Vec4;
    use runmat_builtins::{CellArray, StringArray, Value};
    use runmat_plot::plots::{Figure, LinePlot};

    fn setup_plot_tests() -> PlotTestLockGuard {
        let guard = lock_plot_registry();
        ensure_plot_test_env();
        reset_hold_state_for_run();
        let _ = clear_figure(None);
        guard
    }

    #[test]
    fn legend_returns_handle_and_uses_display_names() {
        let _guard = setup_plot_tests();
        let line = LinePlot::new(vec![0.0, 1.0], vec![1.0, 2.0])
            .unwrap()
            .with_label("Signal");
        let mut figure = Figure::new();
        figure.add_line_plot(line);
        let figure = crate::builtins::plotting::state::import_figure(figure);

        let handle = legend_builtin(vec![Value::Num(
            crate::builtins::plotting::state::encode_axes_handle(figure, 0),
        )])
        .unwrap();
        let (decoded_figure, axes, kind) = decode_plot_object_handle(handle).unwrap();
        assert_eq!(decoded_figure, figure);
        assert_eq!(axes, 0);
        assert_eq!(kind, PlotObjectKind::Legend);

        let fig = clone_figure(figure).unwrap();
        let entries = fig.legend_entries_for_axes(0);
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].label, "Signal");
        assert!(fig.axes_metadata(0).unwrap().legend_enabled);
    }

    #[test]
    fn legend_supports_default_show_and_off_modes() {
        let _guard = setup_plot_tests();
        let mut figure = Figure::new();
        figure.add_line_plot(LinePlot::new(vec![0.0, 1.0], vec![1.0, 2.0]).unwrap());
        let figure = crate::builtins::plotting::state::import_figure(figure);
        let ax = Value::Num(crate::builtins::plotting::state::encode_axes_handle(
            figure, 0,
        ));

        legend_builtin(vec![ax.clone(), Value::String("off".into())]).unwrap();
        let fig = clone_figure(figure).unwrap();
        assert!(!fig.axes_metadata(0).unwrap().legend_enabled);

        legend_builtin(vec![ax.clone(), Value::String("show".into())]).unwrap();
        let fig = clone_figure(figure).unwrap();
        assert!(fig.axes_metadata(0).unwrap().legend_enabled);

        legend_builtin(vec![ax]).unwrap();
        let fig = clone_figure(figure).unwrap();
        assert!(fig.axes_metadata(0).unwrap().legend_enabled);

        legend_builtin(vec![
            Value::Num(crate::builtins::plotting::state::encode_axes_handle(
                figure, 0,
            )),
            Value::String("boxoff".into()),
        ])
        .unwrap();
        let fig = clone_figure(figure).unwrap();
        assert_eq!(
            fig.axes_metadata(0).unwrap().legend_style.box_visible,
            Some(false)
        );

        legend_builtin(vec![
            Value::Num(crate::builtins::plotting::state::encode_axes_handle(
                figure, 0,
            )),
            Value::String("boxon".into()),
        ])
        .unwrap();
        let fig = clone_figure(figure).unwrap();
        assert_eq!(
            fig.axes_metadata(0).unwrap().legend_style.box_visible,
            Some(true)
        );
    }

    #[test]
    fn legend_is_subplot_local_and_supports_labels_and_properties() {
        let _guard = setup_plot_tests();
        let left = LinePlot::new(vec![0.0, 1.0], vec![1.0, 2.0]).unwrap();
        let right = LinePlot::new(vec![0.0, 1.0], vec![2.0, 3.0]).unwrap();
        let mut figure = Figure::new();
        figure.set_subplot_grid(1, 2);
        figure.add_line_plot_on_axes(left, 0);
        figure.add_line_plot_on_axes(right, 1);
        let figure = crate::builtins::plotting::state::import_figure(figure);
        let left_ax = Value::Num(crate::builtins::plotting::state::encode_axes_handle(
            figure, 0,
        ));

        legend_builtin(vec![
            Value::Num(crate::builtins::plotting::state::encode_axes_handle(
                figure, 1,
            )),
            Value::String("Right".into()),
            Value::String("Location".into()),
            Value::String("southwest".into()),
            Value::String("TextColor".into()),
            Value::String("r".into()),
        ])
        .unwrap();
        legend_builtin(vec![left_ax, Value::String("hide".into())]).unwrap();

        let fig = clone_figure(figure).unwrap();
        assert!(!fig.axes_metadata(0).unwrap().legend_enabled);
        let right_meta = fig.axes_metadata(1).unwrap();
        assert!(right_meta.legend_enabled);
        assert_eq!(
            right_meta.legend_style.location.as_deref(),
            Some("southwest")
        );
        assert!(vec4_eq(
            right_meta.legend_style.text_color,
            Vec4::new(1.0, 0.0, 0.0, 1.0)
        ));
        let right_entries = fig.legend_entries_for_axes(1);
        assert_eq!(right_entries[0].label, "Right");
    }

    #[test]
    fn legend_accepts_cell_and_string_array_labels() {
        let _guard = setup_plot_tests();
        let mut figure = Figure::new();
        figure.add_line_plot(LinePlot::new(vec![0.0, 1.0], vec![1.0, 2.0]).unwrap());
        figure.add_line_plot(LinePlot::new(vec![0.0, 1.0], vec![2.0, 3.0]).unwrap());
        let figure = crate::builtins::plotting::state::import_figure(figure);
        let labels = StringArray {
            data: vec!["A".into(), "B".into()],
            shape: vec![1, 2],
            rows: 1,
            cols: 2,
        };
        let ax = Value::Num(crate::builtins::plotting::state::encode_axes_handle(
            figure, 0,
        ));
        legend_builtin(vec![ax.clone(), Value::StringArray(labels)]).unwrap();
        let fig = clone_figure(figure).unwrap();
        let entries = fig.legend_entries_for_axes(0);
        assert_eq!(entries[0].label, "A");
        assert_eq!(entries[1].label, "B");

        let cell = CellArray::new(
            vec![Value::String("C".into()), Value::String("D".into())],
            1,
            2,
        )
        .unwrap();
        legend_builtin(vec![ax, Value::Cell(cell)]).unwrap();
        let fig = clone_figure(figure).unwrap();
        let entries = fig.legend_entries_for_axes(0);
        assert_eq!(entries[0].label, "C");
        assert_eq!(entries[1].label, "D");
    }

    #[test]
    fn legend_accepts_separate_string_labels() {
        let _guard = setup_plot_tests();
        let mut figure = Figure::new();
        figure.add_line_plot(LinePlot::new(vec![0.0, 1.0], vec![1.0, 2.0]).unwrap());
        figure.add_line_plot(LinePlot::new(vec![0.0, 1.0], vec![2.0, 3.0]).unwrap());
        let figure = crate::builtins::plotting::state::import_figure(figure);
        let ax = Value::Num(crate::builtins::plotting::state::encode_axes_handle(
            figure, 0,
        ));

        legend_builtin(vec![
            ax,
            Value::String("Left".into()),
            Value::String("Right".into()),
        ])
        .unwrap();
        let fig = clone_figure(figure).unwrap();
        let entries = fig.legend_entries_for_axes(0);
        assert_eq!(entries[0].label, "Left");
        assert_eq!(entries[1].label, "Right");
    }

    #[test]
    fn legend_accepts_labels_plus_trailing_properties() {
        let _guard = setup_plot_tests();
        let mut figure = Figure::new();
        figure.add_line_plot(LinePlot::new(vec![0.0, 1.0], vec![1.0, 2.0]).unwrap());
        figure.add_line_plot(LinePlot::new(vec![0.0, 1.0], vec![2.0, 3.0]).unwrap());
        let figure = crate::builtins::plotting::state::import_figure(figure);
        let ax = Value::Num(crate::builtins::plotting::state::encode_axes_handle(
            figure, 0,
        ));

        legend_builtin(vec![
            ax,
            Value::String("Left".into()),
            Value::String("Right".into()),
            Value::String("Location".into()),
            Value::String("northwest".into()),
            Value::String("Orientation".into()),
            Value::String("horizontal".into()),
        ])
        .unwrap();

        let fig = clone_figure(figure).unwrap();
        let entries = fig.legend_entries_for_axes(0);
        assert_eq!(entries[0].label, "Left");
        assert_eq!(entries[1].label, "Right");
        let meta = fig.axes_metadata(0).unwrap();
        assert_eq!(meta.legend_style.location.as_deref(), Some("northwest"));
        assert_eq!(meta.legend_style.orientation.as_deref(), Some("horizontal"));
    }

    #[test]
    fn legend_rejects_invalid_inputs() {
        let _guard = setup_plot_tests();
        let mut figure = Figure::new();
        figure.add_line_plot(LinePlot::new(vec![0.0, 1.0], vec![1.0, 2.0]).unwrap());
        let figure = crate::builtins::plotting::state::import_figure(figure);

        let err = legend_builtin(vec![
            Value::Num(crate::builtins::plotting::state::encode_axes_handle(
                figure, 99,
            )),
            Value::String("A".into()),
        ])
        .unwrap_err();
        assert!(err.message.contains("invalid axes") || err.message.contains("out of range"));

        let ax = Value::Num(crate::builtins::plotting::state::encode_axes_handle(
            figure, 0,
        ));
        let err = legend_builtin(vec![
            ax.clone(),
            Value::String("Location".into()),
            Value::Num(1.0),
        ])
        .unwrap_err();
        assert!(err.message.contains("Location must be a string"));

        let err = legend_builtin(vec![
            ax.clone(),
            Value::String("Bogus".into()),
            Value::Num(1.0),
        ])
        .unwrap_err();
        assert!(
            err.message.contains("labels must be strings")
                || err.message.contains("unsupported property")
        );

        let err = legend_builtin(vec![ax, Value::Num(3.0)]).unwrap_err();
        assert!(err.message.contains("labels must be strings"));
    }

    #[test]
    fn legend_descriptor_signatures_cover_core_forms() {
        let labels: Vec<&str> = LEGEND_DESCRIPTOR
            .signatures
            .iter()
            .map(|sig| sig.label)
            .collect();
        assert!(labels.contains(&"h = legend()"));
        assert!(labels.contains(&"h = legend(ax)"));
        assert!(labels.contains(&"h = legend(mode)"));
        assert!(labels.contains(&"h = legend(ax, mode)"));
        assert!(labels.contains(&"h = legend(labels..., Name, Value, ...)"));
        assert!(labels.contains(&"h = legend(ax, labels..., Name, Value, ...)"));
    }
}

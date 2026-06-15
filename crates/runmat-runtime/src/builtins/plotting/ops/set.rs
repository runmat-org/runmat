use runmat_builtins::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor, BuiltinOutputMode,
    BuiltinParamArity, BuiltinParamDescriptor, BuiltinParamType, BuiltinSignatureDescriptor, Value,
};
use runmat_macros::runtime_builtin;

use super::properties::{resolve_plot_handle, set_properties};
use crate::builtins::plotting::type_resolvers::set_type;
use crate::{build_runtime_error, RuntimeError};

const BUILTIN_NAME: &str = "set";

const SET_OUTPUT_STATUS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "status",
    ty: BuiltinParamType::StringScalar,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Returns \"ok\" when property updates succeed.",
}];

const SET_INPUTS_HANDLE_PAIRS: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "h",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Graphics handle (figure, axes, plot object, text, legend, etc.).",
    },
    BuiltinParamDescriptor {
        name: "pairs",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Variadic,
        default: None,
        description: "Property/value pairs to assign (for example 'LineWidth', 2).",
    },
];

const SET_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "status = set(h, property, value, ...)",
    inputs: &SET_INPUTS_HANDLE_PAIRS,
    outputs: &SET_OUTPUT_STATUS,
}];

const SET_ERROR_INVALID_ARGUMENT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.SET.INVALID_ARGUMENT",
    identifier: Some("RunMat:set:InvalidArgument"),
    when: "Handle value is invalid or property/value pairs are missing/malformed/unsupported.",
    message: "set: invalid argument",
};

const SET_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.SET.INTERNAL",
    identifier: Some("RunMat:set:Internal"),
    when: "Internal handle/property mutation fails unexpectedly.",
    message: "set: internal operation failed",
};

const SET_ERRORS: [BuiltinErrorDescriptor; 2] = [SET_ERROR_INVALID_ARGUMENT, SET_ERROR_INTERNAL];

pub const SET_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &SET_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &SET_ERRORS,
};

fn set_error_with_detail(
    error: &'static BuiltinErrorDescriptor,
    detail: impl AsRef<str>,
) -> RuntimeError {
    let mut builder = build_runtime_error(format!("{}: {}", error.message, detail.as_ref()))
        .with_builtin(BUILTIN_NAME);
    if let Some(identifier) = error.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

fn map_set_error(err: RuntimeError) -> RuntimeError {
    if err.identifier().is_some() {
        return err;
    }
    set_error_with_detail(&SET_ERROR_INVALID_ARGUMENT, err.message)
}

#[runtime_builtin(
    name = "set",
    category = "plotting",
    summary = "Set properties on plotting and graphics handles.",
    keywords = "set,plotting,handle,property",
    suppress_auto_output = true,
    type_resolver(set_type),
    descriptor(crate::builtins::plotting::set::SET_DESCRIPTOR),
    builtin_path = "crate::builtins::plotting::set"
)]
pub fn set_builtin(args: Vec<Value>) -> crate::BuiltinResult<String> {
    if args.len() < 3 {
        return Err(set_error_with_detail(
            &SET_ERROR_INVALID_ARGUMENT,
            "expected a plotting handle followed by property/value pairs",
        ));
    }
    let handle = resolve_plot_handle(&args[0], BUILTIN_NAME).map_err(map_set_error)?;
    set_properties(handle, &args[1..], BUILTIN_NAME).map_err(map_set_error)?;
    Ok("ok".to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::builtins::plotting::get::get_builtin;
    use crate::builtins::plotting::legend::legend_builtin;
    use crate::builtins::plotting::tests::{ensure_plot_test_env, lock_plot_registry};
    use crate::builtins::plotting::title::title_builtin;
    use crate::builtins::plotting::{clear_figure, clone_figure, reset_hold_state_for_run};
    use runmat_builtins::Value;
    use runmat_plot::plots::{Figure, LinePlot};

    fn setup() -> crate::builtins::plotting::state::PlotTestLockGuard {
        let guard = lock_plot_registry();
        ensure_plot_test_env();
        reset_hold_state_for_run();
        let _ = clear_figure(None);
        guard
    }

    #[test]
    fn set_descriptor_signatures_cover_core_forms() {
        let labels: Vec<&str> = SET_DESCRIPTOR
            .signatures
            .iter()
            .map(|sig| sig.label)
            .collect();
        assert!(labels.contains(&"status = set(h, property, value, ...)"));
    }

    #[test]
    fn set_missing_pairs_uses_stable_identifier() {
        let _guard = setup();
        let err = set_builtin(vec![Value::Num(1.0)])
            .expect_err("expected missing property/value pairs to fail");
        assert_eq!(err.identifier(), SET_ERROR_INVALID_ARGUMENT.identifier);
    }

    #[test]
    fn set_updates_text_handle_properties() {
        let _guard = setup();
        let h = title_builtin(vec![Value::String("Signal".into())]).unwrap();
        set_builtin(vec![
            Value::Num(h),
            Value::String("String".into()),
            Value::String("Updated".into()),
            Value::String("Visible".into()),
            Value::Bool(false),
        ])
        .unwrap();

        let title = get_builtin(vec![Value::Num(h), Value::String("String".into())]).unwrap();
        assert_eq!(title, Value::String("Updated".into()));
        let visible = get_builtin(vec![Value::Num(h), Value::String("Visible".into())]).unwrap();
        assert_eq!(visible, Value::Bool(false));
    }

    #[test]
    fn set_updates_extended_text_and_legend_properties() {
        let _guard = setup();
        let h = title_builtin(vec![Value::String("Signal".into())]).unwrap();
        set_builtin(vec![
            Value::Num(h),
            Value::String("FontWeight".into()),
            Value::String("bold".into()),
            Value::String("FontAngle".into()),
            Value::String("italic".into()),
        ])
        .unwrap();
        assert_eq!(
            get_builtin(vec![Value::Num(h), Value::String("FontWeight".into())]).unwrap(),
            Value::String("bold".into())
        );
        assert_eq!(
            get_builtin(vec![Value::Num(h), Value::String("FontAngle".into())]).unwrap(),
            Value::String("italic".into())
        );

        let mut figure = Figure::new();
        figure.add_line_plot(
            LinePlot::new(vec![0.0, 1.0], vec![1.0, 2.0])
                .unwrap()
                .with_label("A"),
        );
        let figure = crate::builtins::plotting::import_figure(figure);
        let ax = crate::builtins::plotting::state::encode_axes_handle(figure, 0);
        let legend = legend_builtin(vec![Value::Num(ax)]).unwrap();
        set_builtin(vec![
            Value::Num(legend),
            Value::String("FontWeight".into()),
            Value::String("bold".into()),
            Value::String("FontAngle".into()),
            Value::String("italic".into()),
            Value::String("Interpreter".into()),
            Value::String("none".into()),
            Value::String("Box".into()),
            Value::Bool(false),
            Value::String("Orientation".into()),
            Value::String("horizontal".into()),
        ])
        .unwrap();
        let fig = clone_figure(figure).unwrap();
        let meta = fig.axes_metadata(0).unwrap();
        assert_eq!(meta.legend_style.font_weight.as_deref(), Some("bold"));
        assert_eq!(meta.legend_style.font_angle.as_deref(), Some("italic"));
        assert_eq!(meta.legend_style.interpreter.as_deref(), Some("none"));
        assert_eq!(meta.legend_style.box_visible, Some(false));
        assert_eq!(meta.legend_style.orientation.as_deref(), Some("horizontal"));
    }

    #[test]
    fn set_updates_legend_handle_properties() {
        let _guard = setup();
        let mut figure = Figure::new();
        figure.add_line_plot(
            LinePlot::new(vec![0.0, 1.0], vec![1.0, 2.0])
                .unwrap()
                .with_label("A"),
        );
        let figure = crate::builtins::plotting::import_figure(figure);
        let ax = crate::builtins::plotting::state::encode_axes_handle(figure, 0);
        let legend = legend_builtin(vec![Value::Num(ax)]).unwrap();

        set_builtin(vec![
            Value::Num(legend),
            Value::String("Location".into()),
            Value::String("southwest".into()),
            Value::String("Visible".into()),
            Value::Bool(false),
        ])
        .unwrap();

        let fig = clone_figure(figure).unwrap();
        let meta = fig.axes_metadata(0).unwrap();
        assert_eq!(meta.legend_style.location.as_deref(), Some("southwest"));
        assert!(!meta.legend_enabled);
    }

    #[test]
    fn set_updates_axes_handle_alias_properties() {
        let _guard = setup();
        let h = title_builtin(vec![Value::String("Signal".into())]).unwrap();
        let (figure, axes, _) =
            crate::builtins::plotting::state::decode_plot_object_handle(h).unwrap();
        let ax = crate::builtins::plotting::state::encode_axes_handle(figure, axes);

        set_builtin(vec![
            Value::Num(ax),
            Value::String("Title".into()),
            Value::String("Updated Title".into()),
            Value::String("LegendVisible".into()),
            Value::Bool(false),
        ])
        .unwrap();

        let fig = clone_figure(figure).unwrap();
        let meta = fig.axes_metadata(axes).unwrap();
        assert_eq!(meta.title.as_deref(), Some("Updated Title"));
        assert!(!meta.legend_enabled);
    }

    #[test]
    fn set_updates_axes_local_properties() {
        let _guard = setup();
        let ax = crate::builtins::plotting::subplot::subplot_builtin(
            Value::Num(1.0),
            Value::Num(2.0),
            Value::Num(2.0),
        )
        .unwrap();
        set_builtin(vec![
            Value::Num(ax),
            Value::String("YLim".into()),
            Value::Tensor(runmat_builtins::Tensor {
                rows: 1,
                cols: 2,
                shape: vec![1, 2],
                data: vec![2.0, 8.0],
                dtype: runmat_builtins::NumericDType::F64,
            }),
            Value::String("Colorbar".into()),
            Value::Bool(true),
            Value::String("Colormap".into()),
            Value::String("hot".into()),
            Value::String("FontSize".into()),
            Value::Num(14.0),
        ])
        .unwrap();

        let fig = clone_figure(crate::builtins::plotting::current_figure_handle()).unwrap();
        let meta = fig.axes_metadata(1).unwrap();
        assert_eq!(meta.y_limits, Some((2.0, 8.0)));
        assert!(meta.colorbar_enabled);
        assert_eq!(format!("{:?}", meta.colormap), "Hot");
        assert_eq!(meta.axes_style.font_size, Some(14.0));
        assert_eq!(
            get_builtin(vec![Value::Num(ax), Value::String("FontSize".into())]).unwrap(),
            Value::Num(14.0)
        );
    }

    #[test]
    fn set_updates_figure_current_axes() {
        let _guard = setup();
        let fig =
            crate::builtins::plotting::figure::figure_builtin(vec![Value::Num(4321.0)]).unwrap();
        let ax = crate::builtins::plotting::subplot::subplot_builtin(
            Value::Num(1.0),
            Value::Num(2.0),
            Value::Num(2.0),
        )
        .unwrap();
        set_builtin(vec![
            Value::Num(fig),
            Value::String("CurrentAxes".into()),
            Value::Num(ax),
        ])
        .unwrap();
        let current =
            get_builtin(vec![Value::Num(fig), Value::String("CurrentAxes".into())]).unwrap();
        assert_eq!(current, Value::Num(ax));
    }

    #[test]
    fn set_updates_figure_sgtitle() {
        let _guard = setup();
        let fig =
            crate::builtins::plotting::figure::figure_builtin(vec![Value::Num(7777.0)]).unwrap();
        set_builtin(vec![
            Value::Num(fig),
            Value::String("SGTitle".into()),
            Value::String("Overview".into()),
        ])
        .unwrap();
        let figure = clone_figure(crate::builtins::plotting::state::FigureHandle::from(7777))
            .expect("figure should exist");
        assert_eq!(figure.sg_title.as_deref(), Some("Overview"));
    }

    #[test]
    fn set_updates_figure_window_properties() {
        let _guard = setup();
        let fig =
            crate::builtins::plotting::figure::figure_builtin(vec![Value::Num(8888.0)]).unwrap();
        set_builtin(vec![
            Value::Num(fig),
            Value::String("Name".into()),
            Value::String("demo".into()),
            Value::String("NumberTitle".into()),
            Value::String("off".into()),
            Value::String("Visible".into()),
            Value::String("off".into()),
            Value::String("Color".into()),
            Value::String("k".into()),
        ])
        .unwrap();

        let figure = clone_figure(crate::builtins::plotting::state::FigureHandle::from(8888))
            .expect("figure should exist");
        assert_eq!(figure.name.as_deref(), Some("demo"));
        assert!(!figure.number_title);
        assert!(!figure.visible);
        assert_eq!(figure.background_color, glam::Vec4::new(0.0, 0.0, 0.0, 1.0));
        assert_eq!(
            get_builtin(vec![Value::Num(fig), Value::String("Name".into())]).unwrap(),
            Value::String("demo".into())
        );
    }

    #[test]
    fn set_figure_prevalidates_all_pairs_before_mutating() {
        let _guard = setup();
        let fig =
            crate::builtins::plotting::figure::figure_builtin(vec![Value::Num(8898.0)]).unwrap();
        let handle = crate::builtins::plotting::state::FigureHandle::from(8898);
        let before = clone_figure(handle).expect("figure should exist before failed set");

        let err = set_builtin(vec![
            Value::Num(fig),
            Value::String("Name".into()),
            Value::String("mutated".into()),
            Value::String("Color".into()),
            Value::String("banana".into()),
        ])
        .expect_err("invalid later pair should fail before any mutation");
        assert_eq!(err.identifier(), SET_ERROR_INVALID_ARGUMENT.identifier);

        let after = clone_figure(handle).expect("figure should still exist after failed set");
        assert_eq!(after.name, before.name);
        assert_eq!(after.background_color, before.background_color);
    }

    #[test]
    fn set_visible_figure_display_properties_request_presentation() {
        let _guard = setup();
        let fig =
            crate::builtins::plotting::figure::figure_builtin(vec![Value::Num(8899.0)]).unwrap();
        let ax = crate::builtins::plotting::subplot::subplot_builtin(
            Value::Num(1.0),
            Value::Num(2.0),
            Value::Num(2.0),
        )
        .unwrap();

        let _host_guard = crate::builtins::plotting::disable_host_managed_plot_env_for_tests();
        for args in [
            vec![
                Value::Num(fig),
                Value::String("Name".into()),
                Value::String("demo".into()),
            ],
            vec![
                Value::Num(fig),
                Value::String("NumberTitle".into()),
                Value::String("off".into()),
            ],
            vec![
                Value::Num(fig),
                Value::String("SGTitle".into()),
                Value::String("Overview".into()),
            ],
            vec![
                Value::Num(fig),
                Value::String("CurrentAxes".into()),
                Value::Num(ax),
            ],
        ] {
            let err = set_builtin(args).expect_err("visible display property should present");
            assert!(
                err.message().contains("Plotting is unavailable"),
                "unexpected error: {err:?}"
            );
        }

        set_builtin(vec![
            Value::Num(fig),
            Value::String("Visible".into()),
            Value::String("off".into()),
        ])
        .expect("visible off should present as an update without rendering");
        let figure = clone_figure(crate::builtins::plotting::state::FigureHandle::from(8899))
            .expect("figure should exist");
        assert!(!figure.visible);
    }

    #[test]
    fn set_updates_stem_properties() {
        let _guard = setup();
        let handle = crate::builtins::plotting::stem::stem_builtin(vec![Value::Tensor(
            runmat_builtins::Tensor {
                rows: 2,
                cols: 1,
                shape: vec![2],
                data: vec![1.0, 2.0],
                dtype: runmat_builtins::NumericDType::F64,
            },
        )])
        .unwrap();
        set_builtin(vec![
            Value::Num(handle),
            Value::String("BaseValue".into()),
            Value::Num(-2.0),
            Value::String("Filled".into()),
            Value::Bool(true),
        ])
        .unwrap();
        let base =
            get_builtin(vec![Value::Num(handle), Value::String("BaseValue".into())]).unwrap();
        let filled = get_builtin(vec![Value::Num(handle), Value::String("Filled".into())]).unwrap();
        assert_eq!(base, Value::Num(-2.0));
        assert_eq!(filled, Value::Bool(true));
    }

    #[test]
    fn set_rejects_invalid_property_assignments() {
        let _guard = setup();
        let h = title_builtin(vec![Value::String("Signal".into())]).unwrap();
        let err = set_builtin(vec![
            Value::Num(h),
            Value::String("Bogus".into()),
            Value::Num(1.0),
        ])
        .unwrap_err();
        assert!(err.message.contains("unsupported property"));

        let ax = crate::builtins::plotting::gca::gca_builtin(Vec::new()).unwrap();
        for invalid in [0.0, -1.0, f64::NAN, 1.0e6] {
            let err = set_builtin(vec![
                ax.clone(),
                Value::String("FontSize".into()),
                Value::Num(invalid),
            ])
            .unwrap_err();
            assert!(err
                .message
                .contains("FontSize must be a positive finite value"));
        }
    }
}

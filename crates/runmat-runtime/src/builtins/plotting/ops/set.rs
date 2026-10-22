#[cfg(test)]
use runmat_builtins::SET_DESCRIPTOR;
use runmat_builtins::{BuiltinErrorDescriptor, SET_ERROR_INTERNAL, SET_ERROR_INVALID_ARGUMENT};
use runmat_macros::runtime_builtin;
#[cfg(test)]
use runmat_value::Value;

use super::properties::resolve_plot_handle;
use crate::{build_runtime_error, RuntimeError};

const BUILTIN_NAME: &str = "set";

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
    let span = err.span;
    let retry = err.gpu_gather_retry();
    let descriptor = if crate::builtins::plotting::error::plot_data_export_failure(&err) {
        &SET_ERROR_INTERNAL
    } else {
        &SET_ERROR_INVALID_ARGUMENT
    };
    let mut builder = build_runtime_error(format!("{}: {}", descriptor.message, err.message))
        .with_builtin(BUILTIN_NAME)
        .with_gpu_gather_retry(retry)
        .with_source(err);
    if let Some(identifier) = descriptor.identifier {
        builder = builder.with_identifier(identifier);
    }
    if let Some(span) = span {
        builder = builder.with_span(span);
    }
    builder.build()
}

#[runtime_builtin(
    name = "set",
    binding_variant = "default",
    context = true,
    builtin_path = "crate::builtins::plotting::set"
)]
pub async fn set_builtin_registered(
    runtime: Option<crate::context::RuntimeContext>,
    args: crate::call::BuiltinArguments,
) -> crate::BuiltinResult<String> {
    set_builtin_with_runtime(runtime.as_ref(), args).await
}

#[cfg(test)]
pub async fn set_builtin_async(args: Vec<Value>) -> crate::BuiltinResult<String> {
    set_builtin_with_runtime(
        None,
        crate::execution::RootedValues::new(args)
            .map_err(crate::execution::ExecutionServiceError::into_runtime_error)?
            .into(),
    )
    .await
}

async fn set_builtin_with_runtime(
    runtime: Option<&crate::context::RuntimeContext>,
    args: crate::call::BuiltinArguments,
) -> crate::BuiltinResult<String> {
    if args.len() < 3 || args.len().is_multiple_of(2) {
        return Err(set_error_with_detail(
            &SET_ERROR_INVALID_ARGUMENT,
            "expected a plotting handle followed by property/value pairs",
        ));
    }
    if let Some(()) = super::zoom::set_zoom_object_properties_in_context(
        runtime,
        &args[0],
        args.view().range(1..)?,
        BUILTIN_NAME,
    )? {
        return Ok("ok".to_string());
    }
    if let Some(()) = super::pan::set_pan_object_properties_in_context(
        runtime,
        &args[0],
        args.view().range(1..)?,
        BUILTIN_NAME,
    )? {
        return Ok("ok".to_string());
    }
    if let Some(()) = super::datacursormode::set_data_cursor_object_properties_in_context(
        runtime,
        &args[0],
        args.view().range(1..)?,
        BUILTIN_NAME,
    )? {
        return Ok("ok".to_string());
    }
    if let Some(()) = super::properties::set_receiver_properties_with_runtime(
        runtime,
        &args[0],
        args.view().range(1..)?,
        BUILTIN_NAME,
    )
    .await
    .map_err(map_set_error)?
    {
        return Ok("ok".to_string());
    }
    let handle = resolve_plot_handle(&args[0], BUILTIN_NAME).map_err(map_set_error)?;
    super::properties::set_properties_async_with_runtime(
        runtime,
        handle,
        args.view().range(1..)?,
        BUILTIN_NAME,
    )
    .await
    .map_err(map_set_error)?;
    Ok("ok".to_string())
}

#[cfg(test)]
pub fn set_builtin(args: Vec<Value>) -> crate::BuiltinResult<String> {
    futures::executor::block_on(set_builtin_async(args))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::builtins::plotting::get::get_builtin;
    use crate::builtins::plotting::legend::legend_builtin;
    use crate::builtins::plotting::tests::{ensure_plot_test_env, lock_plot_registry};
    use crate::builtins::plotting::title::title_builtin;
    use crate::builtins::plotting::{clear_figure, clone_figure, reset_hold_state_for_run};
    use runmat_plot::plots::{Figure, LinePlot};
    use runmat_value::Value;

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
    fn set_rejects_unpaired_control_before_mutation() {
        let _guard = setup();
        let error = set_builtin(vec![
            Value::Num(0.0),
            Value::String("Visible".into()),
            Value::Bool(false),
            Value::String("dangling".into()),
        ])
        .expect_err("unpaired property must fail");
        assert_eq!(error.identifier(), SET_ERROR_INVALID_ARGUMENT.identifier);
    }

    #[test]
    fn set_property_name_admits_scalar_string_array_and_rejects_multirow_characters() {
        let _guard = setup();
        let root = Value::Num(0.0);
        set_builtin(vec![
            root.clone(),
            Value::StringArray(
                runmat_value::StringArray::new_2d(vec!["ShowHiddenHandles".into()], 1, 1)
                    .expect("scalar text"),
            ),
            Value::String("off".into()),
        ])
        .expect("scalar string property name");
        let error = set_builtin(vec![
            root,
            Value::CharArray(
                runmat_value::CharArray::from_unicode_scalars(vec!['G', 'r', 'i', 'd'], 2, 2)
                    .expect("two character rows"),
            ),
            Value::Bool(true),
        ])
        .expect_err("multirow property name");
        assert_eq!(error.identifier(), SET_ERROR_INVALID_ARGUMENT.identifier);
    }

    #[test]
    fn interaction_control_properties_share_scalar_text_name_boundary() {
        let _guard = setup();
        let controls = [
            crate::builtins::plotting::zoom::zoom_builtin(
                None,
                crate::builtins::common::test_support::builtin_arguments(Vec::new()),
            )
            .expect("zoom object"),
            crate::builtins::plotting::pan::pan_builtin(
                None,
                crate::builtins::common::test_support::builtin_arguments(Vec::new()),
            )
            .expect("pan object"),
            crate::builtins::plotting::datacursormode::datacursormode_builtin(
                None,
                crate::builtins::common::test_support::builtin_arguments(Vec::new()),
            )
            .expect("data cursor object"),
        ];
        for control in controls {
            let scalar_name = || {
                Value::StringArray(
                    runmat_value::StringArray::new_2d(vec!["Enable".into()], 1, 1)
                        .expect("scalar property name"),
                )
            };
            set_builtin(vec![
                control.clone(),
                scalar_name(),
                Value::String("off".into()),
            ])
            .expect("scalar string array property name");
            assert_eq!(
                get_builtin(vec![control.clone(), scalar_name()]).expect("get scalar name"),
                Value::String("off".into())
            );
            set_builtin(vec![
                control.clone(),
                Value::String("NotAProperty".into()),
                Value::String("on".into()),
            ])
            .expect_err("unknown property must not mutate control state");
            for invalid_name in [
                Value::StringArray(
                    runmat_value::StringArray::new_2d(vec!["Enable".into(), "Motion".into()], 1, 2)
                        .expect("two strings"),
                ),
                Value::CharArray(
                    runmat_value::CharArray::from_unicode_scalars(vec!['E', 'n', 'a', 'b'], 2, 2)
                        .expect("two character rows"),
                ),
            ] {
                set_builtin(vec![
                    control.clone(),
                    invalid_name.clone(),
                    Value::String("on".into()),
                ])
                .expect_err("non-scalar property name must fail before mutation");
                get_builtin(vec![control.clone(), invalid_name])
                    .expect_err("get must reject the same non-scalar name");
                assert_eq!(
                    get_builtin(vec![control.clone(), scalar_name()]).expect("state unchanged"),
                    Value::String("off".into())
                );
            }
        }
    }

    #[test]
    fn set_updates_text_handle_properties() {
        let _guard = setup();
        let h = title_builtin(crate::builtins::common::test_support::builtin_arguments(
            vec![Value::String("Signal".into())],
        ))
        .unwrap();
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
        let h = title_builtin(crate::builtins::common::test_support::builtin_arguments(
            vec![Value::String("Signal".into())],
        ))
        .unwrap();
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
        let legend = legend_builtin(crate::builtins::common::test_support::builtin_arguments(
            vec![Value::Num(ax)],
        ))
        .unwrap();
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
        let legend = legend_builtin(crate::builtins::common::test_support::builtin_arguments(
            vec![Value::Num(ax)],
        ))
        .unwrap();

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
        let h = title_builtin(crate::builtins::common::test_support::builtin_arguments(
            vec![Value::String("Signal".into())],
        ))
        .unwrap();
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
            Value::Tensor(runmat_value::Tensor::new(vec![2.0, 8.0], vec![1, 2]).expect("y limits")),
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
        let fig = crate::builtins::plotting::figure::figure_builtin(
            None,
            crate::builtins::common::test_support::builtin_arguments(vec![Value::Num(4321.0)]),
        )
        .unwrap();
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
        let fig = crate::builtins::plotting::figure::figure_builtin(
            None,
            crate::builtins::common::test_support::builtin_arguments(vec![Value::Num(7777.0)]),
        )
        .unwrap();
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
        let fig = crate::builtins::plotting::figure::figure_builtin(
            None,
            crate::builtins::common::test_support::builtin_arguments(vec![Value::Num(8888.0)]),
        )
        .unwrap();
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
        let fig = crate::builtins::plotting::figure::figure_builtin(
            None,
            crate::builtins::common::test_support::builtin_arguments(vec![Value::Num(8898.0)]),
        )
        .unwrap();
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
    #[cfg(not(target_arch = "wasm32"))]
    fn set_visible_figure_display_properties_request_presentation() {
        let _guard = setup();
        let fig = crate::builtins::plotting::figure::figure_builtin(
            None,
            crate::builtins::common::test_support::builtin_arguments(vec![Value::Num(8899.0)]),
        )
        .unwrap();
        let ax = crate::builtins::plotting::subplot::subplot_builtin(
            Value::Num(1.0),
            Value::Num(2.0),
            Value::Num(2.0),
        )
        .unwrap();

        let _host_guard = _guard.disable_host_managed_plot_env();
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
        let handle = futures::executor::block_on(crate::builtins::plotting::stem::stem_builtin(
            None,
            crate::builtins::common::test_support::builtin_arguments(vec![Value::Tensor(
                runmat_value::Tensor::new(vec![1.0, 2.0], vec![2]).expect("stem values"),
            )]),
        ))
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
        let h = title_builtin(crate::builtins::common::test_support::builtin_arguments(
            vec![Value::String("Signal".into())],
        ))
        .unwrap();
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

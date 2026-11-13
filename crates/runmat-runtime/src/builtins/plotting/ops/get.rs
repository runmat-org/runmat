use runmat_builtins::BuiltinErrorDescriptor;
use runmat_macros::runtime_builtin;
use runmat_value::Value;

use super::properties::{get_properties_async, property_name_text, resolve_plot_handle};
use crate::{build_runtime_error, RuntimeError};
#[cfg(test)]
use runmat_builtins::{GET_DESCRIPTOR, GET_INTEGER_CAPABILITIES};
use runmat_builtins::{
    GET_ERROR_INTERNAL, GET_ERROR_INVALID_ARGUMENT, GET_INTEGER_HANDLE_ALIAS_EXTENSION,
};

const BUILTIN_NAME: &str = "get";

fn get_error_with_detail(
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

fn map_get_error(err: RuntimeError) -> RuntimeError {
    if err.identifier().is_some() {
        return err;
    }
    let span = err.span;
    let retry = err.gpu_gather_retry();
    let descriptor = if crate::builtins::plotting::error::plot_data_export_failure(&err) {
        &GET_ERROR_INTERNAL
    } else {
        &GET_ERROR_INVALID_ARGUMENT
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
    name = "get",
    binding_variant = "default",
    builtin_path = "crate::builtins::plotting::get"
)]
pub async fn get_builtin_async(args: crate::call::BuiltinArguments) -> crate::BuiltinResult<Value> {
    if args.is_empty() || args.len() > 2 {
        return Err(get_error_with_detail(
            &GET_ERROR_INVALID_ARGUMENT,
            "expected a plotting handle and an optional property name",
        ));
    }
    let typed_integer_alias = is_typed_integer_handle_alias(&args[0]);
    if typed_integer_alias {
        crate::compatibility::ensure_builtin_extension_enabled(
            &GET_INTEGER_HANDLE_ALIAS_EXTENSION,
            BUILTIN_NAME,
        )?;
    }
    let handle_arg = if typed_integer_alias {
        Value::Num(
            super::op_common::handles::numeric_handle_scalar(&args[0]).ok_or_else(|| {
                get_error_with_detail(
                    &GET_ERROR_INVALID_ARGUMENT,
                    "typed-integer graphics alias must be exactly representable",
                )
            })?,
        )
    } else {
        args[0].clone()
    };
    let property = args
        .get(1)
        .map(|value| property_name_text(value, BUILTIN_NAME).map_err(map_get_error))
        .transpose()?;
    if let Some(value) =
        super::zoom::get_zoom_object_property(&handle_arg, property.as_deref(), BUILTIN_NAME)?
    {
        return Ok(value);
    }
    if let Some(value) =
        super::pan::get_pan_object_property(&handle_arg, property.as_deref(), BUILTIN_NAME)?
    {
        return Ok(value);
    }
    if let Some(value) = super::datacursormode::get_data_cursor_object_property(
        &handle_arg,
        property.as_deref(),
        BUILTIN_NAME,
    )? {
        return Ok(value);
    }
    if let Some(value) =
        super::properties::get_receiver_properties(&handle_arg, property.as_deref(), BUILTIN_NAME)
            .await
            .map_err(map_get_error)?
    {
        return Ok(value);
    }
    let handle = resolve_plot_handle(&handle_arg, BUILTIN_NAME).map_err(map_get_error)?;
    get_properties_async(handle, property.as_deref(), BUILTIN_NAME)
        .await
        .map_err(map_get_error)
}

#[cfg(test)]
pub fn get_builtin(args: Vec<Value>) -> crate::BuiltinResult<Value> {
    futures::executor::block_on(get_builtin_async(
        crate::builtins::common::test_support::builtin_arguments(args),
    ))
}

fn is_typed_integer_handle_alias(value: &Value) -> bool {
    matches!(value, Value::Int(_))
        || matches!(value, Value::Tensor(tensor) if tensor.integer_storage().is_some())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::builtins::plotting::isgraphics::isgraphics_builtin;
    use crate::builtins::plotting::ishandle::ishandle_builtin;
    use crate::builtins::plotting::legend::legend_builtin;
    use crate::builtins::plotting::tests::{ensure_plot_test_env, lock_plot_registry};
    use crate::builtins::plotting::title::title_builtin;
    use crate::builtins::plotting::{clear_figure, reset_hold_state_for_run};
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
    fn get_descriptor_signatures_cover_core_forms() {
        let labels: Vec<&str> = GET_DESCRIPTOR
            .signatures
            .iter()
            .map(|sig| sig.label)
            .collect();
        assert!(labels.contains(&"value = get(h)"));
        assert!(labels.contains(&"value = get(h, property)"));
    }

    #[test]
    fn get_missing_handle_uses_stable_identifier() {
        let _guard = setup();
        let err = get_builtin(vec![]).expect_err("expected missing handle to fail");
        assert_eq!(err.identifier(), GET_ERROR_INVALID_ARGUMENT.identifier);
    }

    #[test]
    fn get_rejects_extra_and_nontext_property_arguments() {
        let _guard = setup();
        for args in [
            vec![Value::Num(0.0), Value::Num(1.0)],
            vec![
                Value::Num(0.0),
                Value::String("Type".into()),
                Value::String("extra".into()),
            ],
        ] {
            let error = get_builtin(args).expect_err("invalid property invocation");
            assert_eq!(error.identifier(), GET_ERROR_INVALID_ARGUMENT.identifier);
        }
    }

    #[test]
    fn get_property_name_accepts_scalar_string_array_but_not_multirow_text() {
        let _guard = setup();
        let scalar = Value::StringArray(
            runmat_value::StringArray::new_2d(vec!["Type".into()], 1, 1).expect("scalar text"),
        );
        assert_eq!(
            get_builtin(vec![Value::Num(0.0), scalar]).expect("scalar string property"),
            Value::String("root".into())
        );
        let invalid = [
            Value::StringArray(
                runmat_value::StringArray::new_2d(vec!["Type".into(), "Parent".into()], 1, 2)
                    .expect("two strings"),
            ),
            Value::CharArray(
                runmat_value::CharArray::from_unicode_scalars(vec!['T', 'y', 'p', 'e'], 2, 2)
                    .expect("two character rows"),
            ),
        ];
        for property in invalid {
            let error =
                get_builtin(vec![Value::Num(0.0), property]).expect_err("non-scalar property name");
            assert_eq!(error.identifier(), GET_ERROR_INVALID_ARGUMENT.identifier);
        }
    }

    #[test]
    fn control_object_type_is_not_a_supported_property() {
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
            let error = get_builtin(vec![control.clone(), Value::String("Type".into())])
                .expect_err("control Type property is not exposed");
            assert!(error.message.contains("unsupported"));
            assert!(matches!(
                get_builtin(vec![control, Value::String("Enable".into())])
                    .expect("shared supported control property"),
                Value::String(value) if value == "on" || value == "off"
            ));
        }
    }

    #[test]
    fn get_typed_integer_graphics_aliases_are_gated_before_lookup() {
        assert_eq!(GET_INTEGER_CAPABILITIES[0].inputs[0].classes.len(), 8);
        let _matlab = crate::compatibility::push_runmat_extensions_enabled(false);
        for integer in [
            runmat_value::IntValue::I8(1),
            runmat_value::IntValue::I16(1),
            runmat_value::IntValue::I32(1),
            runmat_value::IntValue::I64(1),
            runmat_value::IntValue::U8(1),
            runmat_value::IntValue::U16(1),
            runmat_value::IntValue::U32(1),
            runmat_value::IntValue::U64(1),
        ] {
            let error = get_builtin(vec![Value::Int(integer)]).unwrap_err();
            assert_eq!(
                error.identifier(),
                Some("RunMat:compatibility:GetIntegerHandleAliasExtension")
            );
        }
        let tensor = runmat_value::Tensor::new_integer(
            runmat_value::IntegerStorage::U32(vec![1]),
            vec![1, 1],
        )
        .expect("alias");
        let error = get_builtin(vec![Value::Tensor(tensor)]).unwrap_err();
        assert_eq!(
            error.identifier(),
            Some("RunMat:compatibility:GetIntegerHandleAliasExtension")
        );
    }

    #[test]
    fn get_rejects_inexact_wide_integer_aliases_before_lookup() {
        let _matlab = crate::compatibility::push_runmat_extensions_enabled(true);
        let error = get_builtin(vec![Value::Int(runmat_value::IntValue::U64(
            (1_u64 << 53) + 1,
        ))])
        .expect_err("wide scalar alias must not round");
        assert_eq!(error.identifier(), GET_ERROR_INVALID_ARGUMENT.identifier);

        let tensor = runmat_value::Tensor::new_integer(
            runmat_value::IntegerStorage::U64(vec![(1_u64 << 53) + 1]),
            vec![1, 1],
        )
        .unwrap();
        let error =
            get_builtin(vec![Value::Tensor(tensor)]).expect_err("wide tensor alias must not round");
        assert_eq!(error.identifier(), GET_ERROR_INVALID_ARGUMENT.identifier);
    }

    #[test]
    fn get_reads_text_handle_properties() {
        let _guard = setup();
        let h = title_builtin(crate::builtins::common::test_support::builtin_arguments(
            vec![
                Value::String("Signal".into()),
                Value::String("FontSize".into()),
                Value::Num(16.0),
            ],
        ))
        .unwrap();

        let value = get_builtin(vec![Value::Num(h)]).unwrap();
        let Value::Struct(st) = value else {
            panic!("expected struct");
        };
        assert_eq!(
            st.fields.get("String"),
            Some(&Value::String("Signal".into()))
        );
        assert_eq!(st.fields.get("FontSize"), Some(&Value::Num(16.0)));

        let string_value =
            get_builtin(vec![Value::Num(h), Value::String("String".into())]).unwrap();
        assert_eq!(string_value, Value::String("Signal".into()));

        let _ = crate::builtins::plotting::set::set_builtin(vec![
            Value::Num(h),
            Value::String("FontWeight".into()),
            Value::String("bold".into()),
        ])
        .unwrap();
        let weight = get_builtin(vec![Value::Num(h), Value::String("FontWeight".into())]).unwrap();
        assert_eq!(weight, Value::String("bold".into()));

        let _ = crate::builtins::plotting::set::set_builtin(vec![
            Value::Num(h),
            Value::String("String".into()),
            Value::StringArray(
                runmat_value::StringArray::new(vec!["Top".into(), "Bottom".into()], vec![1, 2])
                    .unwrap(),
            ),
        ])
        .unwrap();
        let multiline = get_builtin(vec![Value::Num(h), Value::String("String".into())]).unwrap();
        assert!(matches!(multiline, Value::StringArray(_)));
    }

    #[test]
    fn get_reads_axes_and_legend_properties() {
        let _guard = setup();
        let mut figure = Figure::new();
        figure.add_line_plot(
            LinePlot::new(vec![0.0, 1.0], vec![1.0, 2.0])
                .unwrap()
                .with_label("A"),
        );
        let handle = crate::builtins::plotting::import_figure(figure);
        let ax = crate::builtins::plotting::state::encode_axes_handle(handle, 0);
        let legend = legend_builtin(crate::builtins::common::test_support::builtin_arguments(
            vec![Value::Num(ax)],
        ))
        .unwrap();

        let axes_value = get_builtin(vec![Value::Num(ax)]).unwrap();
        assert!(matches!(axes_value, Value::Struct(_)));
        let title_handle =
            get_builtin(vec![Value::Num(ax), Value::String("Title".into())]).unwrap();
        assert!(matches!(title_handle, Value::Num(_)));

        let legend_value = get_builtin(vec![Value::Num(legend)]).unwrap();
        let Value::Struct(legend_struct) = legend_value else {
            panic!("expected struct");
        };
        assert!(legend_struct.fields.contains_key("String"));
        assert!(legend_struct.fields.contains_key("Visible"));

        let visible =
            get_builtin(vec![Value::Num(ax), Value::String("LegendVisible".into())]).unwrap();
        assert_eq!(visible, Value::Bool(true));

        let _ = crate::builtins::plotting::set::set_builtin(vec![
            Value::Num(legend),
            Value::String("Orientation".into()),
            Value::String("horizontal".into()),
        ])
        .unwrap();
        let orientation = get_builtin(vec![
            Value::Num(legend),
            Value::String("Orientation".into()),
        ])
        .unwrap();
        assert_eq!(orientation, Value::String("horizontal".into()));

        let _ = crate::builtins::plotting::set::set_builtin(vec![
            Value::Num(legend),
            Value::String("Box".into()),
            Value::Bool(false),
        ])
        .unwrap();
        let box_value = get_builtin(vec![Value::Num(legend), Value::String("Box".into())]).unwrap();
        assert_eq!(box_value, Value::Bool(false));
    }

    #[test]
    fn get_reads_axes_local_limit_and_scale_properties() {
        let _guard = setup();
        let ax = crate::builtins::plotting::subplot::subplot_builtin(
            Value::Num(1.0),
            Value::Num(2.0),
            Value::Num(2.0),
        )
        .unwrap();
        crate::builtins::plotting::set::set_builtin(vec![
            Value::Num(ax),
            Value::String("XLim".into()),
            Value::Tensor(runmat_value::Tensor::new(vec![1.0, 5.0], vec![1, 2]).expect("x limits")),
            Value::String("XScale".into()),
            Value::String("log".into()),
            Value::String("Grid".into()),
            Value::Bool(false),
        ])
        .unwrap();
        let xlim = get_builtin(vec![Value::Num(ax), Value::String("XLim".into())]).unwrap();
        let scale = get_builtin(vec![Value::Num(ax), Value::String("XScale".into())]).unwrap();
        let grid = get_builtin(vec![Value::Num(ax), Value::String("Grid".into())]).unwrap();
        assert_eq!(
            runmat_value::Tensor::try_from(&xlim)
                .unwrap()
                .materialize_f64(),
            vec![1.0, 5.0]
        );
        assert_eq!(scale, Value::String("log".into()));
        assert_eq!(grid, Value::Bool(false));
    }

    #[test]
    fn get_reads_figure_properties() {
        let _guard = setup();
        let fig = crate::builtins::plotting::figure::figure_builtin(
            None,
            crate::builtins::common::test_support::builtin_arguments(vec![
                Value::Num(1234.0),
                Value::String("Name".into()),
                Value::String("demo".into()),
                Value::String("NumberTitle".into()),
                Value::String("off".into()),
                Value::String("Visible".into()),
                Value::String("off".into()),
                Value::String("Color".into()),
                Value::String("k".into()),
            ]),
        )
        .unwrap();
        let ax = crate::builtins::plotting::subplot::subplot_builtin(
            Value::Num(1.0),
            Value::Num(2.0),
            Value::Num(2.0),
        )
        .unwrap();
        let value = get_builtin(vec![Value::Num(fig)]).unwrap();
        let Value::Struct(st) = value else {
            panic!("expected struct");
        };
        assert_eq!(st.fields.get("Number"), Some(&Value::Num(1234.0)));
        assert_eq!(st.fields.get("Type"), Some(&Value::String("figure".into())));
        assert_eq!(st.fields.get("Name"), Some(&Value::String("demo".into())));
        assert_eq!(st.fields.get("NumberTitle"), Some(&Value::Bool(false)));
        assert_eq!(st.fields.get("Visible"), Some(&Value::Bool(false)));
        assert_eq!(
            st.fields.get("Color"),
            Some(&Value::Tensor(
                runmat_value::Tensor::new(vec![0.0, 0.0, 0.0], vec![1, 3]).unwrap()
            ))
        );
        assert_eq!(st.fields.get("CurrentAxes"), Some(&Value::Num(ax)));
        assert!(matches!(st.fields.get("Children"), Some(Value::Tensor(_))));
        let parent = st.fields.get("Parent").expect("parent property");
        let Value::Num(v) = parent else {
            panic!("expected numeric parent");
        };
        assert!(v.is_nan());
        let sgtitle = get_builtin(vec![Value::Num(fig), Value::String("SGTitle".into())]).unwrap();
        assert!(matches!(sgtitle, Value::Num(_)));
        let name = get_builtin(vec![Value::Num(fig), Value::String("Name".into())]).unwrap();
        assert_eq!(name, Value::String("demo".into()));
    }

    #[test]
    fn hierarchy_and_query_semantics_exist() {
        let _guard = setup();
        let fig = crate::builtins::plotting::figure::figure_builtin(
            None,
            crate::builtins::common::test_support::builtin_arguments(vec![Value::Num(5555.0)]),
        )
        .unwrap();
        let ax = crate::builtins::plotting::subplot::subplot_builtin(
            Value::Num(1.0),
            Value::Num(1.0),
            Value::Num(1.0),
        )
        .unwrap();
        let title = crate::builtins::plotting::title::title_builtin(
            crate::builtins::common::test_support::builtin_arguments(vec![Value::String(
                "Signal".into(),
            )]),
        )
        .unwrap();

        let axes_value = get_builtin(vec![Value::Num(ax)]).unwrap();
        let Value::Struct(axes_struct) = axes_value else {
            panic!("expected struct");
        };
        assert_eq!(
            axes_struct.fields.get("Type"),
            Some(&Value::String("axes".into()))
        );
        assert_eq!(axes_struct.fields.get("Parent"), Some(&Value::Num(fig)));
        assert!(matches!(
            axes_struct.fields.get("Children"),
            Some(Value::Tensor(_))
        ));

        let text_value = get_builtin(vec![Value::Num(title)]).unwrap();
        let Value::Struct(text_struct) = text_value else {
            panic!("expected struct");
        };
        assert_eq!(
            text_struct.fields.get("Type"),
            Some(&Value::String("text".into()))
        );
        assert_eq!(text_struct.fields.get("Parent"), Some(&Value::Num(ax)));

        assert!(matches!(
            ishandle_builtin(
                None,
                crate::builtins::common::test_support::builtin_arguments(vec![Value::Num(fig)])
            )
            .unwrap(),
            Value::Bool(true)
        ));
        assert!(matches!(
            isgraphics_builtin(
                None,
                crate::builtins::common::test_support::builtin_arguments(vec![Value::Num(ax)])
            )
            .unwrap(),
            Value::Bool(true)
        ));
        assert!(matches!(
            ishandle_builtin(
                None,
                crate::builtins::common::test_support::builtin_arguments(vec![Value::Num(-1.0)])
            )
            .unwrap(),
            Value::Bool(false)
        ));
    }

    #[test]
    fn stem_handle_exposes_runtime_properties() {
        let _guard = setup();
        let handle = futures::executor::block_on(crate::builtins::plotting::stem::stem_builtin(
            None,
            crate::builtins::common::test_support::builtin_arguments(vec![
                Value::Tensor(
                    runmat_value::Tensor::new(vec![1.0, 2.0], vec![2]).expect("stem values"),
                ),
                Value::String("DisplayName".into()),
                Value::String("Impulse".into()),
                Value::String("BaseValue".into()),
                Value::Num(-1.0),
                Value::String("filled".into()),
            ]),
        ))
        .unwrap();
        let props = get_builtin(vec![Value::Num(handle)]).unwrap();
        let Value::Struct(st) = props else {
            panic!("expected struct");
        };
        assert_eq!(st.fields.get("Type"), Some(&Value::String("stem".into())));
        assert_eq!(st.fields.get("BaseValue"), Some(&Value::Num(-1.0)));
        assert_eq!(st.fields.get("Filled"), Some(&Value::Bool(true)));
    }
}

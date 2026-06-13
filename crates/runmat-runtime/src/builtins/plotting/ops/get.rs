use runmat_builtins::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor, BuiltinOutputMode,
    BuiltinParamArity, BuiltinParamDescriptor, BuiltinParamType, BuiltinSignatureDescriptor, Value,
};
use runmat_macros::runtime_builtin;

use super::properties::{get_properties, resolve_plot_handle};
use crate::builtins::plotting::type_resolvers::get_type;
use crate::{build_runtime_error, RuntimeError};

const BUILTIN_NAME: &str = "get";

const GET_OUTPUT_VALUE: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "value",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Property value or a struct of all readable properties.",
}];

const GET_INPUTS_HANDLE: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "h",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Graphics handle (figure, axes, plot object, text, legend, etc.).",
}];

const GET_INPUTS_HANDLE_PROPERTY: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "h",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Graphics handle (figure, axes, plot object, text, legend, etc.).",
    },
    BuiltinParamDescriptor {
        name: "property",
        ty: BuiltinParamType::PropertyName,
        arity: BuiltinParamArity::Optional,
        default: None,
        description: "Property name to query (case-insensitive).",
    },
];

const GET_SIGNATURES: [BuiltinSignatureDescriptor; 2] = [
    BuiltinSignatureDescriptor {
        label: "value = get(h)",
        inputs: &GET_INPUTS_HANDLE,
        outputs: &GET_OUTPUT_VALUE,
    },
    BuiltinSignatureDescriptor {
        label: "value = get(h, property)",
        inputs: &GET_INPUTS_HANDLE_PROPERTY,
        outputs: &GET_OUTPUT_VALUE,
    },
];

const GET_ERROR_INVALID_ARGUMENT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.GET.INVALID_ARGUMENT",
    identifier: Some("RunMat:get:InvalidArgument"),
    when:
        "Handle value or property selector is missing/invalid, or property lookup fails validation.",
    message: "get: invalid argument",
};

const GET_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.GET.INTERNAL",
    identifier: Some("RunMat:get:Internal"),
    when: "Internal handle/property retrieval fails unexpectedly.",
    message: "get: internal operation failed",
};

const GET_ERRORS: [BuiltinErrorDescriptor; 2] = [GET_ERROR_INVALID_ARGUMENT, GET_ERROR_INTERNAL];

pub const GET_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &GET_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &GET_ERRORS,
};

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
    get_error_with_detail(&GET_ERROR_INVALID_ARGUMENT, err.message)
}

#[runtime_builtin(
    name = "get",
    category = "plotting",
    summary = "Get graphics object properties.",
    keywords = "get,plotting,handle,property",
    type_resolver(get_type),
    descriptor(crate::builtins::plotting::get::GET_DESCRIPTOR),
    builtin_path = "crate::builtins::plotting::get"
)]
pub fn get_builtin(args: Vec<Value>) -> crate::BuiltinResult<Value> {
    if args.is_empty() {
        return Err(get_error_with_detail(
            &GET_ERROR_INVALID_ARGUMENT,
            "expected a plotting handle",
        ));
    }
    let handle = resolve_plot_handle(&args[0], BUILTIN_NAME).map_err(map_get_error)?;
    let property = args
        .get(1)
        .and_then(|v| crate::builtins::plotting::style::value_as_string(v));
    get_properties(handle, property.as_deref(), BUILTIN_NAME).map_err(map_get_error)
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
    fn get_reads_text_handle_properties() {
        let _guard = setup();
        let h = title_builtin(vec![
            Value::String("Signal".into()),
            Value::String("FontSize".into()),
            Value::Num(16.0),
        ])
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
            Value::StringArray(runmat_builtins::StringArray {
                data: vec!["Top".into(), "Bottom".into()],
                shape: vec![1, 2],
                rows: 1,
                cols: 2,
            }),
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
        let legend = legend_builtin(vec![Value::Num(ax)]).unwrap();

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
            Value::Tensor(runmat_builtins::Tensor {
                rows: 1,
                cols: 2,
                shape: vec![1, 2],
                data: vec![1.0, 5.0],
                dtype: runmat_builtins::NumericDType::F64,
            }),
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
            runmat_builtins::Tensor::try_from(&xlim).unwrap().data,
            vec![1.0, 5.0]
        );
        assert_eq!(scale, Value::String("log".into()));
        assert_eq!(grid, Value::Bool(false));
    }

    #[test]
    fn get_reads_figure_properties() {
        let _guard = setup();
        let fig = crate::builtins::plotting::figure::figure_builtin(vec![
            Value::Num(1234.0),
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
        assert_eq!(st.fields.get("Color"), Some(&Value::String("k".into())));
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
        let fig =
            crate::builtins::plotting::figure::figure_builtin(vec![Value::Num(5555.0)]).unwrap();
        let ax = crate::builtins::plotting::subplot::subplot_builtin(
            Value::Num(1.0),
            Value::Num(1.0),
            Value::Num(1.0),
        )
        .unwrap();
        let title =
            crate::builtins::plotting::title::title_builtin(vec![Value::String("Signal".into())])
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

        assert!(ishandle_builtin(vec![Value::Num(fig)]).unwrap());
        assert!(isgraphics_builtin(vec![Value::Num(ax)]).unwrap());
        assert!(!ishandle_builtin(vec![Value::Num(-1.0)]).unwrap());
    }

    #[test]
    fn stem_handle_exposes_runtime_properties() {
        let _guard = setup();
        let handle = crate::builtins::plotting::stem::stem_builtin(vec![
            Value::Tensor(runmat_builtins::Tensor {
                rows: 2,
                cols: 1,
                shape: vec![2],
                data: vec![1.0, 2.0],
                dtype: runmat_builtins::NumericDType::F64,
            }),
            Value::String("DisplayName".into()),
            Value::String("Impulse".into()),
            Value::String("BaseValue".into()),
            Value::Num(-1.0),
            Value::String("filled".into()),
        ])
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

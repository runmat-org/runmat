//! MATLAB-compatible `gca` builtin.

use runmat_builtins::{
    GCA_DESCRIPTOR, GCA_ERROR_TOO_MANY_OUTPUTS, GCA_FIGURE_ARGUMENT_EXTENSION,
    GCA_INTEGER_FIGURE_ALIAS_EXTENSION, GCA_STRUCT_OUTPUT_EXTENSION,
};
use runmat_macros::runtime_builtin;
use runmat_value::Value;

use super::op_common::handles::{handle_from_integer, handle_from_scalar};
use super::plotting_error;
use super::properties::{map_figure_error, resolve_plot_handle, PlotHandle};
use super::state::{encode_axes_handle, FigureAxesState, FigureHandle};
use super::style::value_as_string;
use crate::builtins::common::tensor as tensor_utils;

#[cfg(test)]
use runmat_builtins::GCA_INTEGER_CAPABILITIES;

const BUILTIN_NAME: &str = "gca";

#[runtime_builtin(
    name = "gca",
    binding_variant = "default",
    context = true,
    builtin_path = "crate::builtins::plotting::gca"
)]
pub fn gca_builtin_registered(
    runtime: Option<crate::context::RuntimeContext>,
    rest: crate::call::BuiltinArguments,
) -> crate::BuiltinResult<Value> {
    gca_with_runtime(runtime.as_ref(), rest)
}

#[cfg(test)]
pub fn gca_builtin(rest: Vec<Value>) -> crate::BuiltinResult<Value> {
    gca_with_runtime(
        None,
        crate::builtins::common::test_support::builtin_arguments(rest),
    )
}

fn gca_with_runtime(
    runtime: Option<&crate::context::RuntimeContext>,
    rest: crate::call::BuiltinArguments,
) -> crate::BuiltinResult<Value> {
    crate::output_count::validate_fixed_outputs(
        &GCA_DESCRIPTOR,
        &GCA_ERROR_TOO_MANY_OUTPUTS,
        "gca",
    )?;
    if rest.is_empty() {
        let target = super::op_common::axes_target::PreparedAxesTarget::current(runtime)
            .resolve(BUILTIN_NAME)?;
        let (handle, index) = target.coordinates();
        return Ok(Value::Num(encode_axes_handle(handle, index)));
    }

    if rest.len() == 1 {
        if let Some(mode) = value_as_string(&rest[0]).map_err(|source| {
            crate::builtins::plotting::plotting_error_with_source("gca", source.to_string(), source)
        })? {
            if mode.trim().eq_ignore_ascii_case("struct") {
                crate::compatibility::ensure_builtin_extension_enabled(
                    &GCA_STRUCT_OUTPUT_EXTENSION,
                    BUILTIN_NAME,
                )?;
                let target = super::op_common::axes_target::PreparedAxesTarget::current(runtime)
                    .resolve(BUILTIN_NAME)?;
                let (handle, index) = target.coordinates();
                let state = super::state::axes_state_snapshot_for_property(
                    target.authority(),
                    handle,
                    index,
                )
                .map_err(|err| map_figure_error("gca", err))?;
                return Ok(axes_struct_response(state));
            }
        }
        if matches!(rest[0], Value::Object(_) | Value::HandleObject(_)) {
            if let Some(prepared) = super::op_common::axes_target::prepare_graphics_target(
                runtime,
                BUILTIN_NAME,
                &rest[0],
            )? {
                if matches!(prepared.handle(), PlotHandle::Figure(_)) {
                    crate::compatibility::ensure_builtin_extension_enabled(
                        &GCA_FIGURE_ARGUMENT_EXTENSION,
                        BUILTIN_NAME,
                    )?;
                    let target = prepared.resolve_axes(BUILTIN_NAME)?;
                    let (handle, index) = target.coordinates();
                    return Ok(Value::Num(encode_axes_handle(handle, index)));
                }
            }
        }
        if is_typed_integer_figure_alias(&rest[0]) {
            crate::compatibility::ensure_builtin_extension_enabled(
                &GCA_INTEGER_FIGURE_ALIAS_EXTENSION,
                BUILTIN_NAME,
            )?;
        }
        if let Some(handle) = figure_handle_arg(&rest[0])? {
            crate::compatibility::ensure_builtin_extension_enabled(
                &GCA_FIGURE_ARGUMENT_EXTENSION,
                BUILTIN_NAME,
            )?;
            let target = super::op_common::axes_target::PreparedGraphicsTarget::figure_alias(
                runtime, handle,
            )
            .resolve_axes(BUILTIN_NAME)?;
            let (handle, index) = target.coordinates();
            return Ok(Value::Num(encode_axes_handle(handle, index)));
        }
    }

    Err(plotting_error(
        "gca",
        "gca: unsupported arguments (pass no inputs, a figure handle, or 'struct')",
    ))
}

fn is_typed_integer_figure_alias(value: &Value) -> bool {
    matches!(value, Value::Int(_))
        || matches!(value, Value::Tensor(tensor) if tensor.integer_storage().is_some())
}

fn axes_struct_response(state: FigureAxesState) -> Value {
    let mut st = runmat_value::UnboundedStructValueBuilder::new();
    st.insert(
        "handle",
        Value::Num(encode_axes_handle(state.handle, state.active_index)),
    );
    st.insert("figure", Value::Num(state.handle.as_u32() as f64));
    st.insert("rows", Value::Num(state.rows as f64));
    st.insert("cols", Value::Num(state.cols as f64));
    st.insert("index", Value::Num((state.active_index + 1) as f64));
    Value::Struct(st.finish())
}

fn figure_handle_arg(value: &Value) -> crate::BuiltinResult<Option<FigureHandle>> {
    match value {
        Value::Int(integer) => return Ok(Some(handle_from_integer(integer, "gca")?)),
        Value::Tensor(tensor)
            if tensor_utils::is_scalar_tensor(tensor) && tensor.integer_storage().is_some() =>
        {
            return Ok(Some(handle_from_integer(
                &tensor
                    .integer_storage()
                    .and_then(|storage| storage.value_at(0))
                    .expect("one-element integer storage"),
                "gca",
            )?));
        }
        _ => {}
    }
    if let Ok(handle) = resolve_plot_handle(value, "gca") {
        return match handle {
            PlotHandle::Figure(handle) => Ok(Some(handle)),
            PlotHandle::Root
            | PlotHandle::Axes(_, _)
            | PlotHandle::Ruler(_, _, _)
            | PlotHandle::Text(_, _, _)
            | PlotHandle::Legend(_, _)
            | PlotHandle::PlotChild(_, _) => Err(plotting_error(
                "gca",
                "gca: expected a figure handle, got a non-figure graphics handle",
            )),
        };
    }

    match value {
        Value::Num(v) => Ok(Some(handle_from_scalar(*v, "gca")?)),
        Value::Tensor(tensor) if tensor_utils::is_scalar_tensor(tensor) => Ok(Some(
            handle_from_scalar(tensor_utils::tensor_value_f64(tensor, 0), "gca")?,
        )),
        _ => Ok(None),
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::builtins::plotting::tests::{ensure_plot_test_env, lock_plot_registry};

    fn setup_plot_tests() {
        ensure_plot_test_env();
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn gca_descriptor_signatures_cover_core_forms() {
        let labels: Vec<&str> = GCA_DESCRIPTOR
            .signatures
            .iter()
            .map(|sig| sig.label)
            .collect();
        assert!(labels.contains(&"ax = gca()"));
        assert!(labels.contains(&"ax = gca(fig)"));
        assert!(labels.contains(&"s = gca('struct')"));
    }

    #[test]
    fn gca_figure_handle_arg_reads_typed_integer_storage_exactly() {
        let tensor = runmat_value::Tensor::new_integer(
            runmat_value::IntegerStorage::U32(vec![7]),
            vec![1, 1],
        )
        .unwrap();

        assert_eq!(
            figure_handle_arg(&Value::Tensor(tensor)).unwrap(),
            Some(FigureHandle::from(7))
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn default_returns_scalar_handle() {
        setup_plot_tests();
        let handle = gca_builtin(Vec::new()).unwrap();
        match handle {
            Value::Num(v) => assert!(v > 0.0),
            other => panic!("expected scalar handle, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn struct_mode_returns_struct() {
        setup_plot_tests();
        let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
        let value = gca_builtin(vec![Value::String("struct".to_string().into())]).unwrap();
        assert!(matches!(value, Value::Struct(_)));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn figure_handle_returns_that_figures_current_axes() {
        setup_plot_tests();
        let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
        let value = gca_builtin(vec![Value::Num(42.0)]).unwrap();
        assert!(matches!(value, Value::Num(v) if v > 0.0));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn invalid_arguments_return_plotting_error() {
        setup_plot_tests();
        let err = gca_builtin(vec![Value::String("unsupported".to_string().into())]).unwrap_err();
        let text = err.to_string();
        assert!(
            text.contains("gca: unsupported arguments"),
            "unexpected error: {text}"
        );
        assert!(
            text.contains("pass no inputs, a figure handle, or 'struct'"),
            "unexpected error: {text}"
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn axes_handle_argument_is_rejected() {
        let _guard = lock_plot_registry();
        setup_plot_tests();
        crate::builtins::plotting::reset_plot_state();
        let _compat = crate::compatibility::push_runmat_extensions_enabled(true);

        let axes = gca_builtin(Vec::new()).unwrap();
        let err = gca_builtin(vec![axes]).expect_err("axes handle should not be accepted");
        let text = err.to_string();
        assert!(
            text.contains("expected a figure handle"),
            "unexpected error: {text}"
        );
    }

    #[test]
    fn gca_runmat_only_forms_and_integer_alias_are_independently_gated() {
        assert_eq!(GCA_INTEGER_CAPABILITIES[0].inputs[0].classes.len(), 8);
        let _matlab = crate::compatibility::push_runmat_extensions_enabled(false);
        let struct_error = gca_builtin(vec![Value::from("struct")]).unwrap_err();
        assert_eq!(
            struct_error.identifier(),
            Some("RunMat:compatibility:GcaStructOutputExtension")
        );
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
            let error = gca_builtin(vec![Value::Int(integer)]).unwrap_err();
            assert_eq!(
                error.identifier(),
                Some("RunMat:compatibility:GcaIntegerFigureAliasExtension")
            );
        }
        let figure_error = gca_builtin(vec![Value::Num(1.0)]).unwrap_err();
        assert_eq!(
            figure_error.identifier(),
            Some("RunMat:compatibility:GcaFigureArgumentExtension")
        );
    }

    #[test]
    fn gca_rejects_wide_integer_figure_alias_without_lossy_lookup() {
        setup_plot_tests();
        let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
        let error = gca_builtin(vec![Value::Int(runmat_value::IntValue::U64(
            (1_u64 << 53) + 1,
        ))])
        .expect_err("wide figure identifier is not rounded into the registry");
        assert!(error.message().contains("too large"));

        let tensor = runmat_value::Tensor::new_integer(
            runmat_value::IntegerStorage::U64(vec![(1_u64 << 53) + 1]),
            vec![1, 1],
        )
        .unwrap();
        let error = gca_builtin(vec![Value::Tensor(tensor)])
            .expect_err("wide tensor figure identifier is not rounded into the registry");
        assert!(error.message().contains("too large"));
    }
}

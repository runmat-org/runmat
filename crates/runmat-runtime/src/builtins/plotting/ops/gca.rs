//! MATLAB-compatible `gca` builtin.

use runmat_builtins::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor, BuiltinExtensionDescriptor,
    BuiltinExtensionMode, BuiltinIntegerBackendRule, BuiltinIntegerCapabilityDescriptor,
    BuiltinIntegerComputationDomain, BuiltinIntegerInputAvailability,
    BuiltinIntegerInputCapability, BuiltinIntegerOutputClassRule, BuiltinIntegerOverflowRule,
    BuiltinIntegerOverloadKind, BuiltinIntegerScalarDoubleRule, BuiltinOutputMode,
    BuiltinParamArity, BuiltinParamDescriptor, BuiltinParamType, BuiltinSignatureDescriptor,
};
use runmat_builtins::{StructValue, Value};
use runmat_macros::runtime_builtin;

use super::op_common::handles::{handle_from_integer, handle_from_scalar};
use super::plotting_error;
use super::properties::{map_figure_error, resolve_plot_handle, PlotHandle};
use super::state::{
    current_axes_handle_for_figure, current_axes_state, encode_axes_handle, FigureAxesState,
    FigureHandle,
};
use super::style::value_as_string;
use crate::builtins::common::tensor as tensor_utils;
use crate::builtins::plotting::type_resolvers::gca_type;

const BUILTIN_NAME: &str = "gca";

const GCA_FIGURE_ARGUMENT_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "gca-figure-argument",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "gca with an explicit figure argument is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:GcaFigureArgumentExtension"),
};
const GCA_STRUCT_OUTPUT_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "gca-struct-output",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "gca('struct') is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:GcaStructOutputExtension"),
};
const GCA_INTEGER_FIGURE_ALIAS_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "gca-integer-figure-alias",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "typed-integer numeric figure aliases for gca are a RunMat extension",
    error_identifier: Some("RunMat:compatibility:GcaIntegerFigureAliasExtension"),
};
pub const GCA_EXTENSIONS: [BuiltinExtensionDescriptor; 3] = [
    GCA_FIGURE_ARGUMENT_EXTENSION,
    GCA_STRUCT_OUTPUT_EXTENSION,
    GCA_INTEGER_FIGURE_ALIAS_EXTENSION,
];

const GCA_INTEGER_FIGURE_INPUTS: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "fig",
        classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "R2026a gca has no input form and returns a graphics object; typed integers are exact aliases for RunMat's numeric figure registry only.",
    }];
pub const GCA_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 1] =
    [BuiltinIntegerCapabilityDescriptor {
        form: "ax = gca(integer_fig_alias)",
        inputs: &GCA_INTEGER_FIGURE_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::Structural,
        output_class: BuiltinIntegerOutputClassRule::FunctionSpecific,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::HostOnly,
        overload: BuiltinIntegerOverloadKind::ScalarOnly,
        notes: "The alias is decoded exactly and gated before graphics lookup or mutation. RunMat currently returns its numeric axes representation rather than a MATLAB graphics object.",
    }];

const GCA_OUTPUT_HANDLE: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "ax",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description:
        "Current axes graphics object; RunMat currently returns its encoded numeric axes handle.",
}];

const GCA_OUTPUT_STRUCT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "s",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Axes state struct with handle/figure/rows/cols/index fields.",
}];

const GCA_INPUTS_NONE: [BuiltinParamDescriptor; 0] = [];

const GCA_INPUTS_MODE: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "mode",
    ty: BuiltinParamType::StringScalar,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Only 'struct' is supported.",
}];

const GCA_INPUTS_FIGURE: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "fig",
    ty: BuiltinParamType::NumericScalar,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Figure handle whose current axes should be returned.",
}];

const GCA_SIGNATURES: [BuiltinSignatureDescriptor; 3] = [
    BuiltinSignatureDescriptor {
        label: "ax = gca()",
        inputs: &GCA_INPUTS_NONE,
        outputs: &GCA_OUTPUT_HANDLE,
    },
    BuiltinSignatureDescriptor {
        label: "ax = gca(fig)",
        inputs: &GCA_INPUTS_FIGURE,
        outputs: &GCA_OUTPUT_HANDLE,
    },
    BuiltinSignatureDescriptor {
        label: "s = gca('struct')",
        inputs: &GCA_INPUTS_MODE,
        outputs: &GCA_OUTPUT_STRUCT,
    },
];

const GCA_ERROR_INVALID_ARGUMENT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.GCA.INVALID_ARGUMENT",
    identifier: Some("RunMat:gca:InvalidArgument"),
    when: "Unsupported arguments are provided.",
    message: "gca: unsupported arguments",
};

const GCA_ERRORS: [BuiltinErrorDescriptor; 1] = [GCA_ERROR_INVALID_ARGUMENT];

pub const GCA_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &GCA_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &GCA_ERRORS,
};

#[runtime_builtin(
    name = "gca",
    category = "plotting",
    summary = "Return the current axes handle.",
    keywords = "gca,axes,plotting",
    suppress_auto_output = true,
    type_resolver(gca_type),
    descriptor(crate::builtins::plotting::gca::GCA_DESCRIPTOR),
    extensions(crate::builtins::plotting::gca::GCA_EXTENSIONS),
    integer_capabilities(crate::builtins::plotting::gca::GCA_INTEGER_CAPABILITIES),
    builtin_path = "crate::builtins::plotting::gca"
)]
pub fn gca_builtin(rest: Vec<Value>) -> crate::BuiltinResult<Value> {
    if rest.is_empty() {
        let state = current_axes_state();
        return Ok(Value::Num(encode_axes_handle(
            state.handle,
            state.active_index,
        )));
    }

    if rest.len() == 1 {
        if let Some(mode) = value_as_string(&rest[0]) {
            if mode.trim().eq_ignore_ascii_case("struct") {
                crate::compatibility::ensure_builtin_extension_enabled(
                    &GCA_STRUCT_OUTPUT_EXTENSION,
                    BUILTIN_NAME,
                )?;
                let state = current_axes_state();
                return Ok(axes_struct_response(state));
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
            let axes = current_axes_handle_for_figure(handle)
                .map_err(|err| map_figure_error("gca", err))?;
            return Ok(Value::Num(axes));
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
    let mut st = StructValue::new();
    st.insert(
        "handle",
        Value::Num(encode_axes_handle(state.handle, state.active_index)),
    );
    st.insert("figure", Value::Num(state.handle.as_u32() as f64));
    st.insert("rows", Value::Num(state.rows as f64));
    st.insert("cols", Value::Num(state.cols as f64));
    st.insert("index", Value::Num((state.active_index + 1) as f64));
    Value::Struct(st)
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
    use runmat_builtins::{ResolveContext, Type};

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
        let tensor = runmat_builtins::Tensor::new_integer(
            runmat_builtins::IntegerStorage::U32(vec![7]),
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
        let value = gca_builtin(vec![Value::String("struct".to_string())]).unwrap();
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
        let err = gca_builtin(vec![Value::String("unsupported".to_string())]).unwrap_err();
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
            runmat_builtins::IntValue::I8(1),
            runmat_builtins::IntValue::I16(1),
            runmat_builtins::IntValue::I32(1),
            runmat_builtins::IntValue::I64(1),
            runmat_builtins::IntValue::U8(1),
            runmat_builtins::IntValue::U16(1),
            runmat_builtins::IntValue::U32(1),
            runmat_builtins::IntValue::U64(1),
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
        let error = gca_builtin(vec![Value::Int(runmat_builtins::IntValue::U64(
            (1_u64 << 53) + 1,
        ))])
        .expect_err("wide figure identifier is not rounded into the registry");
        assert!(error.message().contains("too large"));

        let tensor = runmat_builtins::Tensor::new_integer(
            runmat_builtins::IntegerStorage::U64(vec![(1_u64 << 53) + 1]),
            vec![1, 1],
        )
        .unwrap();
        let error = gca_builtin(vec![Value::Tensor(tensor)])
            .expect_err("wide tensor figure identifier is not rounded into the registry");
        assert!(error.message().contains("too large"));
    }

    #[test]
    fn gca_type_no_args_returns_num() {
        assert_eq!(gca_type(&[], &ResolveContext::new(Vec::new())), Type::Num);
    }

    #[test]
    fn gca_type_with_string_arg_returns_struct() {
        let out = gca_type(&[Type::String], &ResolveContext::new(Vec::new()));
        assert!(matches!(out, Type::Struct { .. }));
    }

    #[test]
    fn gca_type_with_figure_arg_returns_num() {
        assert_eq!(
            gca_type(&[Type::Num], &ResolveContext::new(Vec::new())),
            Type::Num
        );
    }

    #[test]
    fn gca_type_with_multi_args_returns_unknown() {
        assert_eq!(
            gca_type(&[Type::Num, Type::String], &ResolveContext::new(Vec::new())),
            Type::Unknown
        );
    }

    #[test]
    fn gca_type_with_scalar_tensor_arg_returns_num() {
        assert_eq!(
            gca_type(
                &[Type::Tensor {
                    shape: Some(vec![Some(1), Some(1)])
                }],
                &ResolveContext::new(Vec::new())
            ),
            Type::Num
        );
    }

    #[test]
    fn gca_type_with_non_scalar_tensor_arg_returns_unknown() {
        assert_eq!(
            gca_type(
                &[Type::Tensor {
                    shape: Some(vec![Some(1), Some(2)])
                }],
                &ResolveContext::new(Vec::new())
            ),
            Type::Unknown
        );
    }
}

//! MATLAB-compatible `gca` builtin.

use runmat_builtins::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor, BuiltinOutputMode,
    BuiltinParamArity, BuiltinParamDescriptor, BuiltinParamType, BuiltinSignatureDescriptor,
};
use runmat_builtins::{StructValue, Value};
use runmat_macros::runtime_builtin;

use super::plotting_error;
use super::state::{current_axes_state, encode_axes_handle, FigureAxesState};
use super::style::value_as_string;
use crate::builtins::plotting::type_resolvers::gca_type;

const GCA_OUTPUT_HANDLE: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "ax",
    ty: BuiltinParamType::NumericScalar,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Current axes handle.",
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

const GCA_SIGNATURES: [BuiltinSignatureDescriptor; 2] = [
    BuiltinSignatureDescriptor {
        label: "ax = gca()",
        inputs: &GCA_INPUTS_NONE,
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
    summary = "Return the handle for the current axes.",
    keywords = "gca,axes,plotting",
    suppress_auto_output = true,
    type_resolver(gca_type),
    descriptor(crate::builtins::plotting::gca::GCA_DESCRIPTOR),
    builtin_path = "crate::builtins::plotting::gca"
)]
pub fn gca_builtin(rest: Vec<Value>) -> crate::BuiltinResult<Value> {
    let state = current_axes_state();
    if rest.is_empty() {
        return Ok(Value::Num(encode_axes_handle(
            state.handle,
            state.active_index,
        )));
    }

    if rest.len() == 1 {
        if let Some(mode) = value_as_string(&rest[0]) {
            if mode.trim().eq_ignore_ascii_case("struct") {
                return Ok(axes_struct_response(state));
            }
        }
    }

    Err(plotting_error(
        "gca",
        "gca: unsupported arguments (pass no inputs or 'struct')",
    ))
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

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::builtins::plotting::tests::ensure_plot_test_env;
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
        assert!(labels.contains(&"s = gca('struct')"));
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
        let value = gca_builtin(vec![Value::String("struct".to_string())]).unwrap();
        assert!(matches!(value, Value::Struct(_)));
    }

    #[test]
    fn gca_type_no_args_returns_num() {
        assert_eq!(gca_type(&[], &ResolveContext::new(Vec::new())), Type::Num);
    }

    #[test]
    fn gca_type_with_args_returns_struct() {
        let out = gca_type(&[Type::String], &ResolveContext::new(Vec::new()));
        assert!(matches!(out, Type::Struct { .. }));
    }
}

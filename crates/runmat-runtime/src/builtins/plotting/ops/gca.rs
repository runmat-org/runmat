//! MATLAB-compatible `gca` builtin.

use runmat_builtins::{StructValue, Value};
use runmat_macros::runtime_builtin;

use super::state::{current_axes_state, encode_axes_handle, FigureAxesState};
use super::style::value_as_string;

#[runtime_builtin(
    name = "gca",
    category = "plotting",
    summary = "Return the handle for the current axes.",
    keywords = "gca,axes,plotting",
    builtin_path = "crate::builtins::plotting::gca"
)]
pub fn gca_builtin(rest: Vec<Value>) -> Result<Value, String> {
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

    Err("gca: unsupported arguments (pass no inputs or 'struct')".to_string())
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
    #[ctor::ctor]
    fn init_plot_test_env() {
        crate::builtins::plotting::state::disable_rendering_for_tests();
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn default_returns_scalar_handle() {
        let handle = gca_builtin(Vec::new()).unwrap();
        match handle {
            Value::Num(v) => assert!(v > 0.0),
            other => panic!("expected scalar handle, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn struct_mode_returns_struct() {
        let value = gca_builtin(vec![Value::String("struct".to_string())]).unwrap();
        assert!(matches!(value, Value::Struct(_)));
    }
}

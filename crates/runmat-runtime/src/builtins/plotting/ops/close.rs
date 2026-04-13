//! MATLAB-compatible `close` builtin.

use std::collections::BTreeSet;

use runmat_builtins::Value;
use runmat_macros::runtime_builtin;

use super::op_common::figure_actions::{parse_close_action, FigureAction};
use super::state::{close_figure, close_figure_with_builtin, figure_handles, FigureHandle};
use crate::builtins::plotting::type_resolvers::handle_scalar_type;

#[runtime_builtin(
    name = "close",
    category = "plotting",
    summary = "Close figures by handle or the active figure.",
    keywords = "close,figure,plotting",
    sink = true,
    suppress_auto_output = true,
    type_resolver(handle_scalar_type),
    builtin_path = "crate::builtins::plotting::close"
)]
pub fn close_builtin(rest: Vec<Value>) -> crate::BuiltinResult<f64> {
    match parse_close_action(&rest)? {
        FigureAction::Current => {
            let closed = close_figure_with_builtin("close", None)?;
            Ok(closed.as_u32() as f64)
        }
        FigureAction::Handles(handles) => {
            let unique: BTreeSet<u32> = handles.into_iter().map(|h| h.as_u32()).collect();
            if unique.is_empty() {
                let closed = close_figure_with_builtin("close", None)?;
                return Ok(closed.as_u32() as f64);
            }
            let mut closed = Vec::new();
            for id in unique {
                let handle = FigureHandle::from(id);
                close_figure_with_builtin("close", Some(handle))?;
                closed.push(id);
            }
            if closed.len() == 1 {
                Ok(closed[0] as f64)
            } else {
                Ok(closed.len() as f64)
            }
        }
        FigureAction::All => {
            let handles = figure_handles();
            if handles.is_empty() {
                return Ok(0.0);
            }
            let count = handles.len();
            for handle in handles {
                let _ = close_figure(Some(handle));
            }
            Ok(count as f64)
        }
    }
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
    fn parse_defaults_to_current() {
        setup_plot_tests();
        assert!(matches!(
            parse_close_action(&[]).unwrap(),
            FigureAction::Current
        ));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn parse_numeric_handles() {
        setup_plot_tests();
        let values = vec![Value::Num(3.0), Value::Num(1.0)];
        match parse_close_action(&values).unwrap() {
            FigureAction::Handles(handles) => {
                assert_eq!(handles.len(), 2);
                assert_eq!(handles[0].as_u32(), 3);
                assert_eq!(handles[1].as_u32(), 1);
            }
            other => panic!("unexpected variant: {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn parse_all_flag() {
        setup_plot_tests();
        let values = vec![Value::String("all".to_string())];
        assert!(matches!(
            parse_close_action(&values).unwrap(),
            FigureAction::All
        ));
    }

    #[test]
    fn close_type_is_numeric_handle() {
        assert_eq!(
            handle_scalar_type(&[Type::tensor()], &ResolveContext::new(Vec::new())),
            Type::Num
        );
    }
}

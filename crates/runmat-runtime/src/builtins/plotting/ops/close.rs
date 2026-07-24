//! MATLAB-compatible `close` builtin.

use std::collections::BTreeSet;

use runmat_builtins::Value;

use super::op_common::figure_actions::{parse_close_action, FigureAction};
use super::state::{close_figure, close_figure_with_builtin, figure_handles, FigureHandle};
#[cfg(test)]
use crate::builtins::plotting::type_resolvers::handle_scalar_type;

pub(crate) fn close_plot_targets(rest: &[Value]) -> crate::BuiltinResult<f64> {
    match parse_close_action(rest)? {
        FigureAction::Current => {
            if figure_handles().is_empty() {
                return Ok(0.0);
            }
            let closed = close_figure_with_builtin("close", None)?;
            Ok(closed.as_u32() as f64)
        }
        FigureAction::Handles(handles) => {
            let unique: BTreeSet<u32> = handles.into_iter().map(|h| h.as_u32()).collect();
            if unique.is_empty() {
                if figure_handles().is_empty() {
                    return Ok(0.0);
                }
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
    use crate::builtins::plotting::{
        lock_plot_test_context, reset_hold_state_for_run, reset_plot_state,
        tests::ensure_plot_test_env, PlotTestLockGuard,
    };
    use runmat_builtins::{ResolveContext, Type};

    fn setup_plot_tests() -> PlotTestLockGuard {
        let guard = lock_plot_test_context();
        ensure_plot_test_env();
        reset_plot_state();
        reset_hold_state_for_run();
        guard
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn parse_defaults_to_current() {
        let _guard = setup_plot_tests();
        assert!(matches!(
            parse_close_action(&[]).unwrap(),
            FigureAction::Current
        ));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn parse_numeric_handles() {
        let _guard = setup_plot_tests();
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
        let _guard = setup_plot_tests();
        let values = vec![Value::String("all".to_string())];
        assert!(matches!(
            parse_close_action(&values).unwrap(),
            FigureAction::All
        ));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn close_current_is_noop_when_no_figures_exist() {
        let _guard = setup_plot_tests();
        assert!(figure_handles().is_empty());

        let result = close_plot_targets(&[]).expect("bare close should be safe with no figures");
        assert_eq!(result, 0.0);
        assert!(figure_handles().is_empty());
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn close_explicit_missing_handle_still_errors() {
        let _guard = setup_plot_tests();
        assert!(figure_handles().is_empty());

        let err = close_plot_targets(&[Value::Num(99.0)]).expect_err("explicit missing handle");
        assert!(
            err.message().contains("figure handle 99 does not exist"),
            "unexpected error: {}",
            err.message()
        );
    }

    #[test]
    fn close_type_is_numeric_handle() {
        assert_eq!(
            handle_scalar_type(&[Type::tensor()], &ResolveContext::new(Vec::new())),
            Type::Num
        );
    }
}

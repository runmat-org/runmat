//! MATLAB-compatible `close` builtin.

use std::collections::BTreeSet;

use runmat_value::Value;

use super::op_common::figure_actions::{parse_close_action, FigureAction};
use super::state::{close_figure, close_figure_with_builtin, figure_handles, FigureHandle};
#[cfg(test)]
use crate::builtins::plotting::type_resolvers::handle_scalar_type;

pub(crate) fn close_plot_targets(rest: &[Value]) -> crate::BuiltinResult<f64> {
    match parse_close_action(rest)? {
        FigureAction::Current => {
            if figure_handles().is_empty() {
                return Ok(1.0);
            }
            close_figure_with_builtin("close", None)?;
            Ok(1.0)
        }
        FigureAction::Handles(handles) => {
            let unique: BTreeSet<u32> = handles.into_iter().map(|h| h.as_u32()).collect();
            if unique.is_empty() {
                if figure_handles().is_empty() {
                    return Ok(1.0);
                }
                close_figure_with_builtin("close", None)?;
                return Ok(1.0);
            }
            for id in unique {
                let handle = FigureHandle::from(id);
                close_figure_with_builtin("close", Some(handle))?;
            }
            Ok(1.0)
        }
        FigureAction::All => {
            let handles = figure_handles();
            if handles.is_empty() {
                return Ok(1.0);
            }
            for handle in handles {
                let _ = close_figure(Some(handle));
            }
            Ok(1.0)
        }
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::builtins::plotting::{
        figure::figure_builtin, lock_plot_test_context, reset_hold_state_for_run, reset_plot_state,
        tests::ensure_plot_test_env, PlotTestLockGuard,
    };
    use runmat_builtins::{ResolveContext, Type};
    use runmat_value::{IntegerStorage, Tensor};

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

    #[test]
    fn parse_all_integer_figure_number_classes_exactly() {
        let storages = [
            IntegerStorage::I8(vec![3]),
            IntegerStorage::I16(vec![3]),
            IntegerStorage::I32(vec![3]),
            IntegerStorage::I64(vec![3]),
            IntegerStorage::U8(vec![3]),
            IntegerStorage::U16(vec![3]),
            IntegerStorage::U32(vec![3]),
            IntegerStorage::U64(vec![3]),
        ];

        for storage in storages {
            let value = Value::Tensor(Tensor::new_integer(storage, vec![1, 1]).expect("figure"));
            let FigureAction::Handles(handles) = parse_close_action(&[value]).unwrap() else {
                panic!("expected integer figure target");
            };
            assert_eq!(handles, vec![FigureHandle::from(3)]);
        }
    }

    #[test]
    fn integer_figure_numbers_enforce_positive_u32_bounds() {
        for storage in [
            IntegerStorage::I64(vec![-1]),
            IntegerStorage::U64(vec![u32::MAX as u64 + 1]),
        ] {
            let value = Value::Tensor(Tensor::new_integer(storage, vec![1, 1]).expect("figure"));
            assert!(parse_close_action(&[value]).is_err());
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
        assert_eq!(result, 1.0);
        assert!(figure_handles().is_empty());
    }

    #[test]
    fn close_explicit_and_all_forms_return_scalar_one_on_success() {
        let _guard = setup_plot_tests();
        figure_builtin(vec![Value::Num(7.0)]).expect("figure 7");
        assert_eq!(close_plot_targets(&[Value::Num(7.0)]).unwrap(), 1.0);

        figure_builtin(vec![Value::Num(8.0)]).expect("figure 8");
        figure_builtin(vec![Value::Num(9.0)]).expect("figure 9");
        assert_eq!(
            close_plot_targets(&[Value::String("all".to_string())]).unwrap(),
            1.0
        );
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

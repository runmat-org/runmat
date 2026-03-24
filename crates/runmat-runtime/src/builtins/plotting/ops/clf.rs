//! MATLAB-compatible `clf` builtin.

use std::collections::BTreeSet;

use runmat_builtins::Value;
use runmat_macros::runtime_builtin;

use super::op_common::figure_actions::{parse_clf_action, FigureAction};
use super::state::{clear_figure, clear_figure_with_builtin, figure_handles, FigureHandle};
use crate::builtins::plotting::type_resolvers::string_type;

#[runtime_builtin(
    name = "clf",
    category = "plotting",
    summary = "Clear figure contents, optionally targeting specific handles.",
    keywords = "clf,clear figure,plotting",
    sink = true,
    suppress_auto_output = true,
    type_resolver(string_type),
    builtin_path = "crate::builtins::plotting::clf"
)]
pub fn clf_builtin(rest: Vec<Value>) -> crate::BuiltinResult<String> {
    let (action, _reset) = parse_clf_action(&rest)?;
    match action {
        FigureAction::Current => {
            let cleared = clear_figure_with_builtin("clf", None)?;
            Ok(format!("Cleared figure {}", cleared.as_u32()))
        }
        FigureAction::Handles(handles) => {
            let ordered: BTreeSet<u32> = handles.into_iter().map(|h| h.as_u32()).collect();
            if ordered.is_empty() {
                let cleared = clear_figure_with_builtin("clf", None)?;
                return Ok(format!("Cleared figure {}", cleared.as_u32()));
            }
            for id in &ordered {
                clear_figure_with_builtin("clf", Some(FigureHandle::from(*id)))?;
            }
            if ordered.len() == 1 {
                Ok(format!("Cleared figure {}", ordered.iter().next().unwrap()))
            } else {
                Ok(format!("Cleared {} figures", ordered.len()))
            }
        }
        FigureAction::All => {
            let handles = figure_handles();
            if handles.is_empty() {
                return Ok("clf: no figures to clear".to_string());
            }
            for handle in handles {
                let _ = clear_figure(Some(handle));
            }
            Ok("Cleared all figures".to_string())
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
    fn defaults_to_current() {
        setup_plot_tests();
        assert!(matches!(
            parse_clf_action(&[]).unwrap(),
            (FigureAction::Current, false)
        ));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn parses_all_flag() {
        setup_plot_tests();
        let values = vec![Value::String("all".to_string())];
        assert!(matches!(
            parse_clf_action(&values).unwrap(),
            (FigureAction::All, false)
        ));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn parses_handles() {
        setup_plot_tests();
        let values = vec![Value::Num(2.0)];
        match parse_clf_action(&values).unwrap() {
            (FigureAction::Handles(handles), _) => {
                assert_eq!(handles.len(), 1);
                assert_eq!(handles[0].as_u32(), 2);
            }
            other => panic!("unexpected: {other:?}"),
        }
    }

    #[test]
    fn clf_type_is_string() {
        assert_eq!(
            string_type(&[Type::tensor()], &ResolveContext::new(Vec::new())),
            Type::String
        );
    }
}

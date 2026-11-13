//! MATLAB-compatible `clf` builtin.

use std::collections::BTreeSet;

use runmat_builtins::{
    CLF_ALL_SELECTOR_EXTENSION, CLF_DESCRIPTOR, CLF_ERROR_TOO_MANY_OUTPUTS,
    CLF_INTEGER_FIGURE_NUMBER_EXTENSION, CLF_VARIADIC_TARGETS_EXTENSION,
};

use runmat_macros::runtime_builtin;
use runmat_value::Value;

use super::op_common::figure_actions::{parse_clf_action, FigureAction};
use super::state::{clear_figure_with_builtin, figure_handles, FigureHandle};

#[runtime_builtin(
    name = "clf",
    binding_variant = "default",
    context = true,
    builtin_path = "crate::builtins::plotting::clf"
)]
pub fn clf_builtin(
    runtime: Option<crate::context::RuntimeContext>,
    arguments: crate::call::BuiltinArguments,
) -> crate::BuiltinResult<f64> {
    let _scope = runtime.as_ref().map(|runtime| runtime.enter());
    crate::output_count::validate_fixed_outputs(
        &CLF_DESCRIPTOR,
        &CLF_ERROR_TOO_MANY_OUTPUTS,
        "clf",
    )
    .map_err(super::coordinate_admission::terminal)?;
    let work = super::coordinate_admission::CoordinateWork::admit(runtime.as_ref(), "clf")?;
    work.finish(clf_impl(arguments.view(), &work))
}

fn clf_impl(
    rest: crate::call::FunctionArgumentView<'_>,
    work: &super::coordinate_admission::CoordinateWork<'_>,
) -> crate::BuiltinResult<f64> {
    if requests_variadic_targets(rest) {
        crate::compatibility::ensure_builtin_extension_enabled(
            &CLF_VARIADIC_TARGETS_EXTENSION,
            "clf",
        )?;
    }
    if rest.iter().any(is_all_selector) {
        crate::compatibility::ensure_builtin_extension_enabled(&CLF_ALL_SELECTOR_EXTENSION, "clf")?;
    }
    if rest.iter().any(is_typed_integer_value) {
        crate::compatibility::ensure_builtin_extension_enabled(
            &CLF_INTEGER_FIGURE_NUMBER_EXTENSION,
            "clf",
        )?;
    }
    let (action, _reset) = parse_clf_action(rest)?;
    work.check()?;
    match action {
        FigureAction::Current => {
            let cleared = clear_figure_with_builtin("clf", None)?;
            Ok(cleared.as_u32() as f64)
        }
        FigureAction::Handles(handles) => {
            let ordered: BTreeSet<u32> = handles.into_iter().map(|h| h.as_u32()).collect();
            if ordered.is_empty() {
                let cleared = clear_figure_with_builtin("clf", None)?;
                return Ok(cleared.as_u32() as f64);
            }
            for id in &ordered {
                work.check()?;
                clear_figure_with_builtin("clf", Some(FigureHandle::from(*id)))?;
            }
            if ordered.len() == 1 {
                Ok(*ordered.iter().next().unwrap() as f64)
            } else {
                Ok(ordered.len() as f64)
            }
        }
        FigureAction::All => {
            let handles = figure_handles();
            if handles.is_empty() {
                return Ok(0.0);
            }
            let count = handles.len();
            for handle in handles {
                work.check()?;
                clear_figure_with_builtin("clf", Some(handle))?;
            }
            Ok(count as f64)
        }
    }
}

fn is_typed_integer_value(value: &Value) -> bool {
    matches!(value, Value::Int(_))
        || matches!(value, Value::Tensor(tensor) if tensor.integer_storage().is_some())
}

fn is_all_selector(value: &Value) -> bool {
    crate::builtins::common::validation::value_text_scalar_string(value)
        .is_some_and(|text| text.trim().eq_ignore_ascii_case("all"))
}

fn requests_variadic_targets(values: crate::call::FunctionArgumentView<'_>) -> bool {
    values
        .iter()
        .filter(|value| !is_reset_selector(value) && !is_all_selector(value))
        .count()
        > 1
}

fn is_reset_selector(value: &Value) -> bool {
    crate::builtins::common::validation::value_text_scalar_string(value)
        .is_some_and(|text| text.trim().eq_ignore_ascii_case("reset"))
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::builtins::plotting::tests::ensure_plot_test_env;
    use runmat_builtins::{
        BuiltinIntegerBackendRule, BuiltinIntegerComputationDomain, BuiltinIntegerOutputClassRule,
    };
    use runmat_builtins::{
        CLF_ALL_SELECTOR_EXTENSION, CLF_DESCRIPTOR, CLF_INTEGER_CAPABILITIES,
        CLF_VARIADIC_TARGETS_EXTENSION,
    };
    use runmat_value::{IntegerStorage, Tensor};

    fn setup_plot_tests() {
        ensure_plot_test_env();
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn clf_descriptor_signatures_cover_core_forms() {
        let labels: Vec<&str> = CLF_DESCRIPTOR
            .signatures
            .iter()
            .map(|sig| sig.label)
            .collect();
        assert!(labels.contains(&"result = clf()"));
        assert!(labels.contains(&"result = clf(targets...)"));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn defaults_to_current() {
        setup_plot_tests();
        assert!(matches!(
            parse_clf_action(crate::call::FunctionArgumentView::from_slice(&[])).unwrap(),
            (FigureAction::Current, false)
        ));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn parses_all_flag() {
        setup_plot_tests();
        let values = vec![Value::String("all".to_string().into())];
        assert!(matches!(
            parse_clf_action(crate::call::FunctionArgumentView::from_slice(&values)).unwrap(),
            (FigureAction::All, false)
        ));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn parses_handles() {
        setup_plot_tests();
        let values = vec![Value::Num(2.0)];
        match parse_clf_action(crate::call::FunctionArgumentView::from_slice(&values)).unwrap() {
            (FigureAction::Handles(handles), _) => {
                assert_eq!(handles.len(), 1);
                assert_eq!(handles[0].as_u32(), 2);
            }
            other => panic!("unexpected: {other:?}"),
        }
    }

    #[test]
    fn parses_all_integer_figure_number_classes_exactly() {
        let storages = [
            IntegerStorage::I8(vec![2]),
            IntegerStorage::I16(vec![2]),
            IntegerStorage::I32(vec![2]),
            IntegerStorage::I64(vec![2]),
            IntegerStorage::U8(vec![2]),
            IntegerStorage::U16(vec![2]),
            IntegerStorage::U32(vec![2]),
            IntegerStorage::U64(vec![2]),
        ];

        for storage in storages {
            let value = Value::Tensor(Tensor::new_integer(storage, vec![1, 1]).expect("figure"));
            let (FigureAction::Handles(handles), false) =
                parse_clf_action(crate::call::FunctionArgumentView::from_slice(&[value])).unwrap()
            else {
                panic!("expected integer figure target");
            };
            assert_eq!(handles, vec![FigureHandle::from(2)]);
        }
    }

    #[test]
    fn integer_figure_numbers_enforce_positive_u32_bounds() {
        for storage in [
            IntegerStorage::I64(vec![-1]),
            IntegerStorage::U64(vec![u32::MAX as u64 + 1]),
        ] {
            let value = Value::Tensor(Tensor::new_integer(storage, vec![1, 1]).expect("figure"));
            assert!(
                parse_clf_action(crate::call::FunctionArgumentView::from_slice(&[value])).is_err()
            );
        }
    }

    #[test]
    fn compatibility_mode_rejects_all_integer_figure_number_classes() {
        let _compat = crate::compatibility::push_runmat_extensions_enabled(false);
        let storages = [
            IntegerStorage::I8(vec![2]),
            IntegerStorage::I16(vec![2]),
            IntegerStorage::I32(vec![2]),
            IntegerStorage::I64(vec![2]),
            IntegerStorage::U8(vec![2]),
            IntegerStorage::U16(vec![2]),
            IntegerStorage::U32(vec![2]),
            IntegerStorage::U64(vec![2]),
        ];

        for storage in storages {
            let value = Value::Tensor(Tensor::new_integer(storage, vec![1, 1]).expect("figure"));
            let err = clf_builtin(
                None,
                crate::builtins::common::test_support::builtin_arguments(vec![value]),
            )
            .expect_err("typed integer extension must be gated");
            assert_eq!(
                err.identifier(),
                Some("RunMat:compatibility:ClfIntegerFigureNumberExtension")
            );
        }
    }

    #[test]
    fn compatibility_mode_rejects_runmat_all_selector_before_plot_state_access() {
        let _compat = crate::compatibility::push_runmat_extensions_enabled(false);
        let err = clf_builtin(
            None,
            crate::builtins::common::test_support::builtin_arguments(vec![Value::String(
                "all".into(),
            )]),
        )
        .expect_err("RunMat-only all selector");
        assert_eq!(
            err.identifier(),
            CLF_ALL_SELECTOR_EXTENSION.error_identifier
        );
    }

    #[test]
    fn compatibility_mode_rejects_separate_variadic_targets() {
        let _compat = crate::compatibility::push_runmat_extensions_enabled(false);
        let err = clf_builtin(
            None,
            crate::builtins::common::test_support::builtin_arguments(vec![
                Value::Num(1.0),
                Value::Num(2.0),
            ]),
        )
        .expect_err("RunMat-only variadic targets");
        assert_eq!(
            err.identifier(),
            CLF_VARIADIC_TARGETS_EXTENSION.error_identifier
        );
    }

    #[test]
    fn resident_numeric_targets_are_rejected_without_provider_dispatch() {
        let resident = Value::GpuTensor(
            runmat_accelerate_api::test_support::TestAllocationOwner::new()
                .allocate_zeroed(
                    vec![1, 1],
                    runmat_accelerate_api::NumericElementType::F64,
                    runmat_accelerate_api::GpuTensorStorage::Real,
                )
                .unwrap()
                .handle,
        );
        assert!(
            parse_clf_action(crate::call::FunctionArgumentView::from_slice(&[resident])).is_err()
        );
    }

    #[test]
    fn clf_integer_capability_is_structural_and_host_only() {
        let capability = &CLF_INTEGER_CAPABILITIES[0];
        assert_eq!(capability.inputs[0].classes.len(), 8);
        assert_eq!(
            capability.computation_domain,
            BuiltinIntegerComputationDomain::Structural
        );
        assert_eq!(capability.backend, BuiltinIntegerBackendRule::HostOnly);
        assert_eq!(
            capability.output_class,
            BuiltinIntegerOutputClassRule::NotApplicable
        );
    }
}

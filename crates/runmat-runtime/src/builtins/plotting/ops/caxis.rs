use runmat_builtins::{CAXIS_DESCRIPTOR, CAXIS_ERROR_TOO_MANY_OUTPUTS};
use runmat_macros::runtime_builtin;
use runmat_value::Value;

use super::op_common::color_limits::apply_color_limits;

#[runtime_builtin(
    name = "caxis",
    binding_variant = "default",
    context = true,
    builtin_path = "crate::builtins::plotting::caxis"
)]
pub fn caxis_builtin_registered(
    runtime: Option<crate::context::RuntimeContext>,
    args: crate::call::BuiltinArguments,
) -> crate::BuiltinResult<Value> {
    caxis_with_runtime(runtime.as_ref(), args)
}

#[cfg(test)]
pub fn caxis_builtin(args: Vec<Value>) -> crate::BuiltinResult<Value> {
    caxis_with_runtime(
        None,
        crate::builtins::common::test_support::builtin_arguments(args),
    )
}

fn caxis_with_runtime(
    runtime: Option<&crate::context::RuntimeContext>,
    args: crate::call::BuiltinArguments,
) -> crate::BuiltinResult<Value> {
    crate::output_count::validate_fixed_outputs(
        &CAXIS_DESCRIPTOR,
        &CAXIS_ERROR_TOO_MANY_OUTPUTS,
        "caxis",
    )?;
    apply_color_limits(runtime, "caxis", args)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::builtins::plotting::tests::{ensure_plot_test_env, lock_plot_registry};
    use crate::builtins::plotting::{clear_figure, reset_hold_state_for_run};
    use runmat_builtins::{
        BuiltinIntegerAuditKind, CAXIS_INTEGER_AUDIT, CLIM_INTEGER_LIMITS_EXTENSION,
    };
    use runmat_value::{IntegerStorage, Tensor};

    #[test]
    fn caxis_descriptor_signatures_cover_core_forms() {
        let labels: Vec<&str> = CAXIS_DESCRIPTOR
            .signatures
            .iter()
            .map(|sig| sig.label)
            .collect();
        assert!(labels.contains(&"limits = caxis()"));
        assert!(labels.contains(&"caxis([cmin cmax])"));
        assert!(labels.contains(&"caxis('auto'|'manual')"));
    }

    #[test]
    fn caxis_queries_and_sets_limits() {
        let _guard = lock_plot_registry();
        ensure_plot_test_env();
        reset_hold_state_for_run();
        let _ = clear_figure(None);

        let _ = caxis_builtin(vec![Value::Tensor(
            Tensor::new(vec![0.1, 0.9], vec![1, 2]).expect("caxis limits"),
        )])
        .unwrap();

        let queried = caxis_builtin(Vec::new()).unwrap();
        let tensor = Tensor::try_from(&queried).unwrap();
        assert_eq!(tensor.materialize_f64(), vec![0.1, 0.9]);
    }

    #[test]
    fn caxis_errors_use_caxis_context() {
        let _guard = lock_plot_registry();
        ensure_plot_test_env();
        reset_hold_state_for_run();
        let _ = clear_figure(None);

        let err = caxis_builtin(vec![Value::String("badmode".into())]).unwrap_err();
        assert!(err.message.contains("caxis"));
        assert!(!err.message.contains("clim"));
    }

    #[test]
    fn caxis_is_audited_as_the_clim_integer_alias() {
        assert_eq!(CAXIS_INTEGER_AUDIT.kind, BuiltinIntegerAuditKind::AliasOf);
        assert_eq!(CAXIS_INTEGER_AUDIT.canonical_builtin, Some("clim"));
    }

    #[test]
    fn caxis_integer_limits_are_mode_gated_for_every_class() {
        let _guard = lock_plot_registry();
        ensure_plot_test_env();
        reset_hold_state_for_run();
        let _ = clear_figure(None);

        for storage in [
            IntegerStorage::I8(vec![-1, 2]),
            IntegerStorage::I16(vec![-1, 2]),
            IntegerStorage::I32(vec![-1, 2]),
            IntegerStorage::I64(vec![-1, 2]),
            IntegerStorage::U8(vec![1, 2]),
            IntegerStorage::U16(vec![1, 2]),
            IntegerStorage::U32(vec![1, 2]),
            IntegerStorage::U64(vec![1, 2]),
        ] {
            let limits = Value::Tensor(Tensor::new_integer(storage, vec![1, 2]).unwrap());
            {
                let _compat = crate::compatibility::push_runmat_extensions_enabled(false);
                let error = caxis_builtin(vec![limits.clone()])
                    .expect_err("MATLAB mode must reject typed integer color limits");
                assert_eq!(
                    error.identifier(),
                    CLIM_INTEGER_LIMITS_EXTENSION.error_identifier
                );
            }
            {
                let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
                let result = caxis_builtin(vec![limits]).unwrap();
                let tensor = Tensor::try_from(&result).unwrap();
                assert_eq!(tensor.numeric_dtype(), runmat_value::NumericDType::F64);
            }
        }
    }

    #[test]
    fn caxis_rejects_equal_limits_and_undocumented_tight_mode() {
        let _guard = lock_plot_registry();
        ensure_plot_test_env();
        reset_hold_state_for_run();
        let _ = clear_figure(None);

        assert!(caxis_builtin(vec![Value::Tensor(
            Tensor::new(vec![1.0, 1.0], vec![1, 2]).unwrap()
        )])
        .is_err());
        assert!(caxis_builtin(vec![Value::String("tight".into())]).is_err());
    }
}

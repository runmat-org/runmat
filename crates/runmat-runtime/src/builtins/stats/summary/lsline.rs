//! Least-squares line plotting compatibility helper.

#[cfg(test)]
use runmat_builtins::LSLINE_DESCRIPTOR;
use runmat_builtins::{BuiltinErrorDescriptor, ReferenceLineOwner, LSLINE_ERROR_INVALID_ARGUMENT};
use runmat_macros::runtime_builtin;
use runmat_value::Value;

use crate::builtins::plotting::properties::{resolve_plot_handle, PlotHandle};
use crate::{build_runtime_error, BuiltinResult, RuntimeError};

const NAME: &str = "lsline";

fn error(descriptor: &'static BuiltinErrorDescriptor, message: impl Into<String>) -> RuntimeError {
    let mut builder = build_runtime_error(message).with_builtin(NAME);
    if let Some(identifier) = descriptor.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

fn invalid_argument(message: impl Into<String>) -> RuntimeError {
    error(&LSLINE_ERROR_INVALID_ARGUMENT, message)
}

fn invalid_axes_source(source: RuntimeError) -> RuntimeError {
    let span = source.span;
    let mut builder = build_runtime_error("lsline: expected an axes handle")
        .with_builtin(NAME)
        .with_gpu_gather_retry(source.gpu_gather_retry())
        .with_source(source);
    if let Some(identifier) = LSLINE_ERROR_INVALID_ARGUMENT.identifier {
        builder = builder.with_identifier(identifier);
    }
    if let Some(span) = span {
        builder = builder.with_span(span);
    }
    builder.build()
}

fn validate_lsline_args(args: crate::call::FunctionArgumentView<'_>) -> BuiltinResult<()> {
    match (args.len(), args.first()) {
        (0, _) => Ok(()),
        (1, Some(arg)) => match resolve_plot_handle(arg, NAME) {
            Ok(PlotHandle::Axes(_, _)) => Ok(()),
            Ok(_) => Err(invalid_argument("lsline: expected an axes handle")),
            Err(err) => Err(invalid_axes_source(err)),
        },
        _ => Err(invalid_argument(
            "lsline: expected zero inputs or one axes handle",
        )),
    }
}

#[runtime_builtin(
    name = "lsline",
    binding_variant = "default",
    context = true,
    builtin_path = "crate::builtins::stats::summary::lsline"
)]
pub(crate) async fn lsline_builtin(
    runtime: Option<crate::context::RuntimeContext>,
    args: crate::call::BuiltinArguments,
) -> BuiltinResult<Value> {
    validate_lsline_args(args.view())?;
    super::reference_lines::run_reference_lines(runtime, args, ReferenceLineOwner::Lsline).await
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::builtins::plotting::get::get_builtin;
    use crate::builtins::plotting::scatter::scatter_builtin;
    use crate::builtins::plotting::state::{encode_axes_handle, PlotTestLockGuard};
    use crate::builtins::plotting::tests::{ensure_plot_test_env, lock_plot_registry};
    use crate::builtins::plotting::{
        clear_figure, clone_figure, configure_subplot, current_figure_handle,
        reset_hold_state_for_run,
    };
    use futures::executor::block_on;
    use runmat_value::Tensor;

    fn setup() -> PlotTestLockGuard {
        let guard = lock_plot_registry();
        ensure_plot_test_env();
        reset_hold_state_for_run();
        let _ = clear_figure(None);
        guard
    }

    fn tensor(data: Vec<f64>, rows: usize, cols: usize) -> Value {
        Value::Tensor(Tensor::new(data, vec![rows, cols]).unwrap())
    }

    fn x_data(handle: f64) -> Vec<f64> {
        let value = get_builtin(vec![Value::Num(handle), Value::String("XData".into())]).unwrap();
        Tensor::try_from(&value).unwrap().materialize_f64()
    }

    fn y_data(handle: f64) -> Vec<f64> {
        let value = get_builtin(vec![Value::Num(handle), Value::String("YData".into())]).unwrap();
        Tensor::try_from(&value).unwrap().materialize_f64()
    }

    #[test]
    fn lsline_adds_least_squares_line_to_current_axes() {
        let _guard = setup();
        block_on(scatter_builtin(None, {
            let mut coordinates = vec![
                tensor(vec![1.0, 2.0, 3.0], 1, 3),
                tensor(vec![2.0, 4.0, 6.0], 1, 3),
            ];
            coordinates.extend(Vec::new());
            crate::builtins::common::test_support::builtin_arguments(coordinates)
        }))
        .unwrap();

        let handle = block_on(lsline_builtin(
            None,
            crate::builtins::common::test_support::builtin_arguments(Vec::new()),
        ))
        .unwrap();
        let Value::Num(handle) = handle else {
            panic!("expected line handle");
        };
        assert_eq!(x_data(handle), vec![1.0, 3.0]);
        assert_eq!(y_data(handle), vec![2.0, 6.0]);
    }

    #[test]
    fn lsline_targets_axes_and_returns_one_line_per_eligible_plot() {
        let _guard = setup();
        configure_subplot(1, 2, 0).unwrap();
        block_on(scatter_builtin(None, {
            let mut coordinates = vec![
                tensor(vec![100.0, 200.0], 1, 2),
                tensor(vec![10.0, 20.0], 1, 2),
            ];
            coordinates.extend(Vec::new());
            crate::builtins::common::test_support::builtin_arguments(coordinates)
        }))
        .unwrap();
        configure_subplot(1, 2, 1).unwrap();
        block_on(scatter_builtin(None, {
            let mut coordinates = vec![tensor(vec![2.0, 4.0], 1, 2), tensor(vec![1.0, 5.0], 1, 2)];
            coordinates.extend(Vec::new());
            crate::builtins::common::test_support::builtin_arguments(coordinates)
        }))
        .unwrap();

        let fig = current_figure_handle();
        let ax = encode_axes_handle(fig, 1);
        let handle = block_on(lsline_builtin(
            None,
            crate::builtins::common::test_support::builtin_arguments(vec![Value::Num(ax)]),
        ))
        .unwrap();
        let Value::Num(handle) = handle else {
            panic!("expected line handle");
        };
        assert_eq!(x_data(handle), vec![2.0, 4.0]);
        assert_eq!(y_data(handle), vec![1.0, 5.0]);

        let figure = clone_figure(fig).unwrap();
        assert_eq!(figure.len(), 3);
    }

    #[test]
    fn lsline_rejects_extra_arguments() {
        let _guard = setup();
        let err = block_on(lsline_builtin(
            None,
            crate::builtins::common::test_support::builtin_arguments(vec![
                Value::Num(1.0),
                Value::Num(2.0),
            ]),
        ))
        .expect_err("extra arguments should fail");
        assert_eq!(err.identifier(), LSLINE_ERROR_INVALID_ARGUMENT.identifier);
    }

    #[test]
    fn lsline_invalid_axes_keeps_its_own_diagnostic() {
        let _guard = setup();
        let err = block_on(lsline_builtin(
            None,
            crate::builtins::common::test_support::builtin_arguments(vec![Value::String(
                "not axes".into(),
            )]),
        ))
        .expect_err("text is not an axes handle");
        assert_eq!(err.identifier(), LSLINE_ERROR_INVALID_ARGUMENT.identifier);
        assert_eq!(err.context.builtin.as_deref(), Some("lsline"));
        assert!(std::error::Error::source(&err).is_some());
    }

    #[test]
    fn lsline_descriptor_signatures_cover_matlab_surface() {
        let labels: Vec<&str> = LSLINE_DESCRIPTOR
            .signatures
            .iter()
            .map(|sig| sig.label)
            .collect();
        assert!(labels.contains(&"lsline()"));
        assert!(labels.contains(&"lsline(ax)"));
        assert!(labels.contains(&"h = lsline()"));
        assert!(labels.contains(&"h = lsline(ax)"));
    }
}

//! Least-squares line plotting compatibility helper.

use runmat_builtins::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor, BuiltinOutputMode,
    BuiltinParamArity, BuiltinParamDescriptor, BuiltinParamType, BuiltinSignatureDescriptor, Type,
};
use runmat_macros::runtime_builtin;
use runmat_value::Value;

use crate::builtins::plotting::properties::{resolve_plot_handle, PlotHandle};
use crate::{build_runtime_error, BuiltinResult, RuntimeError};

const NAME: &str = "lsline";

const PARAM_AX: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "ax",
    ty: BuiltinParamType::AxesHandle,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Target axes handle.",
};

const OUTPUT_HANDLE: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "h",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Line graphics handle or vector of line handles.",
};

const INPUTS_EMPTY: [BuiltinParamDescriptor; 0] = [];
const INPUTS_AX: [BuiltinParamDescriptor; 1] = [PARAM_AX];
const OUTPUTS_HANDLE: [BuiltinParamDescriptor; 1] = [OUTPUT_HANDLE];

const SIGNATURES: [BuiltinSignatureDescriptor; 4] = [
    BuiltinSignatureDescriptor {
        label: "lsline()",
        inputs: &INPUTS_EMPTY,
        outputs: &[],
    },
    BuiltinSignatureDescriptor {
        label: "lsline(ax)",
        inputs: &INPUTS_AX,
        outputs: &[],
    },
    BuiltinSignatureDescriptor {
        label: "h = lsline()",
        inputs: &INPUTS_EMPTY,
        outputs: &OUTPUTS_HANDLE,
    },
    BuiltinSignatureDescriptor {
        label: "h = lsline(ax)",
        inputs: &INPUTS_AX,
        outputs: &OUTPUTS_HANDLE,
    },
];

pub const LSLINE_ERROR_INVALID_ARGUMENT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.LSLINE.INVALID_ARGUMENT",
    identifier: Some("RunMat:lsline:InvalidArgument"),
    when: "The optional axes handle is malformed or extra arguments are supplied.",
    message: "lsline: invalid argument",
};

pub const LSLINE_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.LSLINE.INTERNAL",
    identifier: Some("RunMat:lsline:Internal"),
    when: "RunMat cannot construct or register the least-squares line.",
    message: "lsline: internal error",
};

const ERRORS: [BuiltinErrorDescriptor; 2] = [LSLINE_ERROR_INVALID_ARGUMENT, LSLINE_ERROR_INTERNAL];

pub const LSLINE_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ERRORS,
};

fn lsline_type(_args: &[Type], _ctx: &runmat_builtins::ResolveContext) -> Type {
    Type::Unknown
}

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

fn internal_error(message: impl Into<String>) -> RuntimeError {
    error(&LSLINE_ERROR_INTERNAL, message)
}

fn validate_lsline_args(args: &[Value]) -> BuiltinResult<()> {
    match args {
        [] => Ok(()),
        [arg] => match resolve_plot_handle(arg, NAME) {
            Ok(PlotHandle::Axes(_, _)) => Ok(()),
            Ok(_) => Err(invalid_argument("lsline: expected an axes handle")),
            Err(err) => {
                if err.identifier().is_some() {
                    Err(invalid_argument("lsline: expected an axes handle"))
                } else {
                    Err(invalid_argument(err.message))
                }
            }
        },
        _ => Err(invalid_argument(
            "lsline: expected zero inputs or one axes handle",
        )),
    }
}

fn map_refline_error(err: RuntimeError) -> RuntimeError {
    let message = err.message.replace("refline", "lsline");
    if err
        .identifier()
        .is_some_and(|id| id.contains("InvalidArgument"))
    {
        invalid_argument(message)
    } else {
        internal_error(message)
    }
}

#[runtime_builtin(
    name = "lsline",
    category = "stats/summary",
    summary = "Add least-squares fit lines to scatter-style plots.",
    keywords = "lsline,least squares,statistics,plotting",
    sink = true,
    suppress_auto_output = true,
    type_resolver(lsline_type),
    descriptor(crate::builtins::stats::summary::lsline::LSLINE_DESCRIPTOR),
    builtin_path = "crate::builtins::stats::summary::lsline"
)]
pub(crate) async fn lsline_builtin(args: Vec<Value>) -> BuiltinResult<Value> {
    validate_lsline_args(&args)?;
    super::refline::refline_builtin(args)
        .await
        .map_err(map_refline_error)
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
        block_on(scatter_builtin(
            tensor(vec![1.0, 2.0, 3.0], 1, 3),
            tensor(vec![2.0, 4.0, 6.0], 1, 3),
            Vec::new(),
        ))
        .unwrap();

        let handle = block_on(lsline_builtin(Vec::new())).unwrap();
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
        block_on(scatter_builtin(
            tensor(vec![100.0, 200.0], 1, 2),
            tensor(vec![10.0, 20.0], 1, 2),
            Vec::new(),
        ))
        .unwrap();
        configure_subplot(1, 2, 1).unwrap();
        block_on(scatter_builtin(
            tensor(vec![2.0, 4.0], 1, 2),
            tensor(vec![1.0, 5.0], 1, 2),
            Vec::new(),
        ))
        .unwrap();

        let fig = current_figure_handle();
        let ax = encode_axes_handle(fig, 1);
        let handle = block_on(lsline_builtin(vec![Value::Num(ax)])).unwrap();
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
        let err = block_on(lsline_builtin(vec![Value::Num(1.0), Value::Num(2.0)]))
            .expect_err("extra arguments should fail");
        assert_eq!(err.identifier(), LSLINE_ERROR_INVALID_ARGUMENT.identifier);
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

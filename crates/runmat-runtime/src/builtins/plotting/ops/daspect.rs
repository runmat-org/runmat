//! MATLAB-compatible `daspect` builtin.

use runmat_builtins::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor, BuiltinOutputMode,
    BuiltinParamArity, BuiltinParamDescriptor, BuiltinParamType, BuiltinSignatureDescriptor,
    Tensor, Value,
};
use runmat_macros::runtime_builtin;

use super::op_common::axes_target::AxesTarget;
use super::properties::{
    data_aspect_ratio_from_value, data_aspect_ratio_mode_from_value, resolve_plot_handle,
    PlotHandle,
};
use super::state::{
    data_aspect_ratio_snapshot, data_aspect_ratio_snapshot_for_axes, set_data_aspect_ratio,
    set_data_aspect_ratio_for_axes,
};
use super::{plotting_error, plotting_error_with_source};
use crate::builtins::plotting::type_resolvers::daspect_type;
use crate::BuiltinResult;

const BUILTIN_NAME: &str = "daspect";

const DASPECT_OUTPUT_RATIO: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "ratio",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Current data aspect ratio [dx dy dz].",
}];

const DASPECT_OUTPUT_MODE: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "mode",
    ty: BuiltinParamType::StringScalar,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Current data aspect ratio mode, 'auto' or 'manual'.",
}];

const DASPECT_INPUTS_NONE: [BuiltinParamDescriptor; 0] = [];

const DASPECT_INPUTS_RATIO: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "ratio",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Three-element positive numeric data aspect ratio.",
}];

const DASPECT_INPUTS_MODE: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "mode",
    ty: BuiltinParamType::StringScalar,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Mode string: 'auto', 'manual', or 'mode'.",
}];

const DASPECT_INPUTS_AX_RATIO: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "ax",
        ty: BuiltinParamType::AxesHandle,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Target axes handle.",
    },
    BuiltinParamDescriptor {
        name: "ratio_or_mode",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Ratio vector or mode string.",
    },
];

const DASPECT_INPUTS_AX: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "ax",
    ty: BuiltinParamType::AxesHandle,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Target axes handle.",
}];

const DASPECT_SIGNATURES: [BuiltinSignatureDescriptor; 5] = [
    BuiltinSignatureDescriptor {
        label: "ratio = daspect()",
        inputs: &DASPECT_INPUTS_NONE,
        outputs: &DASPECT_OUTPUT_RATIO,
    },
    BuiltinSignatureDescriptor {
        label: "ratio = daspect(ratio)",
        inputs: &DASPECT_INPUTS_RATIO,
        outputs: &DASPECT_OUTPUT_RATIO,
    },
    BuiltinSignatureDescriptor {
        label: "mode = daspect(mode)",
        inputs: &DASPECT_INPUTS_MODE,
        outputs: &DASPECT_OUTPUT_MODE,
    },
    BuiltinSignatureDescriptor {
        label: "ratio = daspect(ax)",
        inputs: &DASPECT_INPUTS_AX,
        outputs: &DASPECT_OUTPUT_RATIO,
    },
    BuiltinSignatureDescriptor {
        label: "ratio = daspect(ax, ratio_or_mode)",
        inputs: &DASPECT_INPUTS_AX_RATIO,
        outputs: &DASPECT_OUTPUT_RATIO,
    },
];

const DASPECT_ERROR_INVALID_ARGUMENT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.DASPECT.INVALID_ARGUMENT",
    identifier: Some("RunMat:daspect:InvalidArgument"),
    when: "Argument count, axes handle, ratio vector, or mode string is invalid.",
    message: "daspect: invalid argument",
};

const DASPECT_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.DASPECT.INTERNAL",
    identifier: Some("RunMat:daspect:Internal"),
    when: "Internal plotting state update fails.",
    message: "daspect: internal operation failed",
};

const DASPECT_ERRORS: [BuiltinErrorDescriptor; 2] =
    [DASPECT_ERROR_INVALID_ARGUMENT, DASPECT_ERROR_INTERNAL];

pub const DASPECT_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &DASPECT_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &DASPECT_ERRORS,
};

#[runtime_builtin(
    name = "daspect",
    category = "plotting",
    summary = "Query or set axes data aspect ratio.",
    keywords = "daspect,data aspect ratio,plotting,axes",
    suppress_auto_output = true,
    type_resolver(daspect_type),
    descriptor(crate::builtins::plotting::daspect::DASPECT_DESCRIPTOR),
    builtin_path = "crate::builtins::plotting::daspect"
)]
pub fn daspect_builtin(args: Vec<Value>) -> BuiltinResult<Value> {
    let (target, args) = split_optional_axes_target(args)?;
    match args.as_slice() {
        [] => query_ratio(target),
        [value] => {
            if let Some(text) = value_as_string(value) {
                match text.trim().to_ascii_lowercase().as_str() {
                    "mode" => return query_mode(target),
                    "auto" | "manual" => {
                        let mode = data_aspect_ratio_mode_from_value(value, BUILTIN_NAME)?;
                        let ratio = current_ratio(target)?;
                        set_ratio_and_mode(target, ratio, mode)?;
                        return Ok(Value::String(mode.into()));
                    }
                    _ => {}
                }
            }
            let ratio = data_aspect_ratio_from_value(value, BUILTIN_NAME)?;
            set_ratio_and_mode(target, ratio, "manual")?;
            Ok(ratio_value(ratio))
        }
        _ => Err(daspect_err("expected zero or one data-aspect argument")),
    }
}

fn split_optional_axes_target(args: Vec<Value>) -> BuiltinResult<(AxesTarget, Vec<Value>)> {
    let mut iter = args.into_iter();
    let Some(first) = iter.next() else {
        return Ok((None, Vec::new()));
    };
    if let Ok(PlotHandle::Axes(handle, axes_index)) = resolve_plot_handle(&first, BUILTIN_NAME) {
        return Ok((Some((handle, axes_index)), iter.collect()));
    }
    let mut rest = Vec::with_capacity(iter.size_hint().0 + 1);
    rest.push(first);
    rest.extend(iter);
    Ok((None, rest))
}

fn query_ratio(target: AxesTarget) -> BuiltinResult<Value> {
    Ok(ratio_value(current_ratio(target)?))
}

fn query_mode(target: AxesTarget) -> BuiltinResult<Value> {
    let (_, mode) = match target {
        Some((handle, axes_index)) => data_aspect_ratio_snapshot_for_axes(handle, axes_index)
            .map_err(|err| {
                plotting_error_with_source(BUILTIN_NAME, format!("{BUILTIN_NAME}: {err}"), err)
            })?,
        None => data_aspect_ratio_snapshot(),
    };
    Ok(Value::String(mode))
}

fn current_ratio(target: AxesTarget) -> BuiltinResult<[f64; 3]> {
    let (ratio, _) = match target {
        Some((handle, axes_index)) => data_aspect_ratio_snapshot_for_axes(handle, axes_index)
            .map_err(|err| {
                plotting_error_with_source(BUILTIN_NAME, format!("{BUILTIN_NAME}: {err}"), err)
            })?,
        None => data_aspect_ratio_snapshot(),
    };
    Ok(ratio)
}

fn set_ratio_and_mode(target: AxesTarget, ratio: [f64; 3], mode: &str) -> BuiltinResult<()> {
    match target {
        Some((handle, axes_index)) => {
            set_data_aspect_ratio_for_axes(handle, axes_index, ratio, mode).map_err(|err| {
                plotting_error_with_source(BUILTIN_NAME, format!("{BUILTIN_NAME}: {err}"), err)
            })
        }
        None => {
            set_data_aspect_ratio(ratio, mode);
            Ok(())
        }
    }
}

fn ratio_value(ratio: [f64; 3]) -> Value {
    Value::Tensor(Tensor {
        rows: 1,
        cols: 3,
        shape: vec![1, 3],
        data: ratio.to_vec(),
        integer_data: None,
        dtype: runmat_builtins::NumericDType::F64,
    })
}

fn value_as_string(value: &Value) -> Option<String> {
    match value {
        Value::String(s) => Some(s.clone()),
        Value::CharArray(chars) => chars.row_string(),
        _ => None,
    }
}

fn daspect_err(detail: impl AsRef<str>) -> crate::RuntimeError {
    plotting_error(BUILTIN_NAME, detail.as_ref().to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::builtins::plotting::get::get_builtin;
    use crate::builtins::plotting::set::set_builtin;
    use crate::builtins::plotting::state::axis_display_bounds_snapshot_for_axes;
    use crate::builtins::plotting::tests::{ensure_plot_test_env, lock_plot_registry};
    use crate::builtins::plotting::{clear_figure, reset_hold_state_for_run};
    use futures::executor::block_on;

    fn setup() -> crate::builtins::plotting::state::PlotTestLockGuard {
        let guard = lock_plot_registry();
        ensure_plot_test_env();
        reset_hold_state_for_run();
        let _ = clear_figure(None);
        guard
    }

    #[test]
    fn daspect_queries_sets_ratio_and_mode() {
        let _guard = setup();
        let ax = crate::builtins::plotting::gca::gca_builtin(Vec::new()).unwrap();
        let ratio = daspect_builtin(vec![ratio(&[1.0, 2.0, 1.0])]).unwrap();
        assert_eq!(tensor_data(ratio), vec![1.0, 2.0, 1.0]);
        assert_eq!(
            daspect_builtin(vec![Value::String("mode".into())]).unwrap(),
            Value::String("manual".into())
        );
        assert_eq!(
            get_builtin(vec![
                ax.clone(),
                Value::String("DataAspectRatioMode".into())
            ])
            .unwrap(),
            Value::String("manual".into())
        );

        let prop = get_builtin(vec![ax, Value::String("DataAspectRatio".into())]).unwrap();
        assert_eq!(tensor_data(prop), vec![1.0, 2.0, 1.0]);

        assert_eq!(
            daspect_builtin(vec![Value::String("auto".into())]).unwrap(),
            Value::String("auto".into())
        );
    }

    #[test]
    fn daspect_explicit_axes_does_not_change_current_axes() {
        let _guard = setup();
        let ax1 = crate::builtins::plotting::subplot::subplot_builtin(
            Value::Num(1.0),
            Value::Num(2.0),
            Value::Num(1.0),
        )
        .unwrap();
        let ax2 = crate::builtins::plotting::subplot::subplot_builtin(
            Value::Num(1.0),
            Value::Num(2.0),
            Value::Num(2.0),
        )
        .unwrap();
        daspect_builtin(vec![Value::Num(ax1), ratio(&[3.0, 1.0, 1.0])]).unwrap();
        let current = daspect_builtin(Vec::new()).unwrap();
        assert_eq!(tensor_data(current), vec![1.0, 1.0, 1.0]);
        let left = daspect_builtin(vec![Value::Num(ax1)]).unwrap();
        assert_eq!(tensor_data(left), vec![3.0, 1.0, 1.0]);
        let right = daspect_builtin(vec![Value::Num(ax2)]).unwrap();
        assert_eq!(tensor_data(right), vec![1.0, 1.0, 1.0]);
    }

    #[test]
    fn daspect_round_trips_through_axes_properties_and_bounds() {
        let _guard = setup();
        let ax = crate::builtins::plotting::gca::gca_builtin(Vec::new()).unwrap();
        block_on(crate::builtins::plotting::plot::plot_builtin(vec![
            ratio(&[0.0, 10.0]),
            ratio(&[0.0, 1.0]),
        ]))
        .unwrap();
        set_builtin(vec![
            ax.clone(),
            Value::String("DataAspectRatio".into()),
            ratio(&[2.0, 1.0, 1.0]),
        ])
        .unwrap();
        let bounds = axis_display_bounds_snapshot_for_axes(
            crate::builtins::plotting::current_figure_handle(),
            0,
        )
        .unwrap()
        .unwrap();
        assert_eq!(bounds, (0.0, 10.0, -2.0, 3.0));
    }

    #[test]
    fn daspect_rejects_invalid_ratios() {
        let _guard = setup();
        let err = daspect_builtin(vec![ratio(&[1.0, 0.0, 1.0])]).unwrap_err();
        assert!(err.message.contains("positive"));
    }

    fn ratio(values: &[f64]) -> Value {
        Value::Tensor(Tensor {
            rows: 1,
            cols: values.len(),
            shape: vec![1, values.len()],
            data: values.to_vec(),
            integer_data: None,
            dtype: runmat_builtins::NumericDType::F64,
        })
    }

    fn tensor_data(value: Value) -> Vec<f64> {
        match value {
            Value::Tensor(tensor) => tensor.data,
            other => panic!("expected tensor, got {other:?}"),
        }
    }
}

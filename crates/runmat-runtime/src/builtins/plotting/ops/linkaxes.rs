//! MATLAB-compatible `linkaxes` builtin.

use runmat_builtins::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor, BuiltinOutputMode,
    BuiltinParamArity, BuiltinParamDescriptor, BuiltinParamType, BuiltinSignatureDescriptor,
};
use runmat_macros::runtime_builtin;
use runmat_value::Value;

use super::plotting_error;
use super::state::{
    axes_metadata_snapshot, decode_axes_handle, figure_handle_exists, link_axes, FigureHandle,
    LinkAxesMode,
};
use crate::builtins::common::tensor as tensor_utils;
use crate::builtins::plotting::op_common::handles::numeric_handle_from_integer;
use crate::builtins::plotting::style::value_as_string;
use crate::builtins::plotting::type_resolvers::set_type;

type AxesTarget = (FigureHandle, usize);
type ParsedLinkAxesArgs = (Vec<AxesTarget>, Option<LinkAxesMode>);

const LINKAXES_OUTPUTS_NONE: [BuiltinParamDescriptor; 0] = [];

const LINKAXES_INPUTS_AXES: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "ax",
    ty: BuiltinParamType::AxesHandle,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Axes handle or numeric array of axes handles to link.",
}];

const LINKAXES_INPUTS_AXES_OPTION: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "ax",
        ty: BuiltinParamType::AxesHandle,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Axes handle or numeric array of axes handles to link.",
    },
    BuiltinParamDescriptor {
        name: "option",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Link option: 'x', 'y', 'xy', or 'off'.",
    },
];

const LINKAXES_SIGNATURES: [BuiltinSignatureDescriptor; 2] = [
    BuiltinSignatureDescriptor {
        label: "linkaxes(ax)",
        inputs: &LINKAXES_INPUTS_AXES,
        outputs: &LINKAXES_OUTPUTS_NONE,
    },
    BuiltinSignatureDescriptor {
        label: "linkaxes(ax, option)",
        inputs: &LINKAXES_INPUTS_AXES_OPTION,
        outputs: &LINKAXES_OUTPUTS_NONE,
    },
];

const LINKAXES_ERROR_INVALID_ARGUMENT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.LINKAXES.INVALID_ARGUMENT",
    identifier: Some("RunMat:linkaxes:InvalidArgument"),
    when: "Axes handles, option, or argument count is invalid.",
    message: "linkaxes: invalid argument",
};

const LINKAXES_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.LINKAXES.INTERNAL",
    identifier: Some("RunMat:linkaxes:Internal"),
    when: "Internal plotting state update fails.",
    message: "linkaxes: internal operation failed",
};

const LINKAXES_ERRORS: [BuiltinErrorDescriptor; 2] =
    [LINKAXES_ERROR_INVALID_ARGUMENT, LINKAXES_ERROR_INTERNAL];

pub const LINKAXES_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &LINKAXES_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &LINKAXES_ERRORS,
};

#[runtime_builtin(
    name = "linkaxes",
    category = "plotting",
    summary = "Link axes limits across axes.",
    keywords = "linkaxes,plotting,axes,xlim,ylim",
    suppress_auto_output = true,
    type_resolver(set_type),
    descriptor(crate::builtins::plotting::linkaxes::LINKAXES_DESCRIPTOR),
    builtin_path = "crate::builtins::plotting::linkaxes"
)]
pub fn linkaxes_builtin(args: Vec<Value>) -> crate::BuiltinResult<String> {
    let (handles, mode) = parse_linkaxes_args(args)?;
    link_axes(handles, mode).map_err(|err| {
        crate::builtins::plotting::plotting_error_with_source(
            "linkaxes",
            format!("linkaxes: {err}"),
            err,
        )
    })?;
    Ok(String::new())
}

fn parse_linkaxes_args(args: Vec<Value>) -> crate::BuiltinResult<ParsedLinkAxesArgs> {
    match args.as_slice() {
        [axes] => Ok((axes_handles_from_value(axes)?, Some(LinkAxesMode::XY))),
        [axes, option] => Ok((
            axes_handles_from_value(axes)?,
            link_mode_from_value(option)?,
        )),
        _ => Err(plotting_error(
            "linkaxes",
            "linkaxes: expected axes handles and optional option",
        )),
    }
}

fn link_mode_from_value(value: &Value) -> crate::BuiltinResult<Option<LinkAxesMode>> {
    let option = value_as_string(value)
        .ok_or_else(|| plotting_error("linkaxes", "linkaxes: option must be a string"))?;
    match option.trim().to_ascii_lowercase().as_str() {
        "x" => Ok(Some(LinkAxesMode::X)),
        "y" => Ok(Some(LinkAxesMode::Y)),
        "xy" => Ok(Some(LinkAxesMode::XY)),
        "off" => Ok(None),
        _ => Err(plotting_error(
            "linkaxes",
            "linkaxes: option must be 'x', 'y', 'xy', or 'off'",
        )),
    }
}

fn axes_handles_from_value(value: &Value) -> crate::BuiltinResult<Vec<(FigureHandle, usize)>> {
    let raw = match value {
        Value::Num(handle) => vec![*handle],
        Value::Int(handle) => vec![numeric_handle_from_integer(handle).ok_or_else(|| {
            plotting_error("linkaxes", "linkaxes: axes must be valid axes handles")
        })?],
        Value::Tensor(tensor) => {
            if let Some(storage) = tensor.integer_storage() {
                (0..storage.len())
                    .map(|index| {
                        storage
                            .value_at(index)
                            .and_then(|value| numeric_handle_from_integer(&value))
                            .ok_or_else(|| {
                                plotting_error(
                                    "linkaxes",
                                    "linkaxes: axes must be valid axes handles",
                                )
                            })
                    })
                    .collect::<crate::BuiltinResult<Vec<_>>>()?
            } else {
                tensor_utils::tensor_values_f64(tensor)
            }
        }
        _ => {
            return Err(plotting_error(
                "linkaxes",
                "linkaxes: axes must be an axes handle or numeric array of axes handles",
            ))
        }
    };

    let mut axes = Vec::with_capacity(raw.len());
    for handle in raw {
        let (figure, axes_index) = decode_axes_handle(handle)
            .map_err(|_| plotting_error("linkaxes", "linkaxes: axes must be valid axes handles"))?;
        if !figure_handle_exists(figure) {
            return Err(plotting_error("linkaxes", "linkaxes: invalid axes handle"));
        }
        axes_metadata_snapshot(figure, axes_index).map_err(|err| {
            crate::builtins::plotting::plotting_error_with_source(
                "linkaxes",
                format!("linkaxes: {err}"),
                err,
            )
        })?;
        axes.push((figure, axes_index));
    }
    Ok(axes)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::builtins::plotting::get::get_builtin;
    use crate::builtins::plotting::set::set_builtin;
    use crate::builtins::plotting::subplot::subplot_builtin;
    use crate::builtins::plotting::tests::{ensure_plot_test_env, lock_plot_registry};
    use crate::builtins::plotting::{clear_figure, reset_hold_state_for_run};
    use runmat_value::Tensor;

    fn setup() -> crate::builtins::plotting::state::PlotTestLockGuard {
        let guard = lock_plot_registry();
        ensure_plot_test_env();
        reset_hold_state_for_run();
        let _ = clear_figure(None);
        guard
    }

    fn tensor(data: Vec<f64>, shape: Vec<usize>) -> Value {
        Value::Tensor(Tensor::new(data, shape).expect("linkaxes test tensor"))
    }

    fn numeric_vec(value: Value) -> Vec<f64> {
        let tensor = Tensor::try_from(&value).expect("tensor value");
        tensor.materialize_f64()
    }

    fn subplot(rows: f64, cols: f64, index: f64) -> f64 {
        subplot_builtin(Value::Num(rows), Value::Num(cols), Value::Num(index)).expect("subplot")
    }

    fn set_limits(ax: f64, property: &str, limits: [f64; 2]) {
        set_builtin(vec![
            Value::Num(ax),
            Value::String(property.into()),
            tensor(limits.to_vec(), vec![1, 2]),
        ])
        .expect("set limits");
    }

    fn get_limits(ax: f64, property: &str) -> Vec<f64> {
        numeric_vec(
            get_builtin(vec![Value::Num(ax), Value::String(property.into())]).expect("get limits"),
        )
    }

    #[test]
    fn linkaxes_handle_vector_reads_typed_integer_storage_exactly() {
        let _guard = setup();
        let first = subplot(1.0, 2.0, 1.0);
        let second = subplot(1.0, 2.0, 2.0);
        let handles = Tensor::new_integer(
            runmat_value::IntegerStorage::U32(vec![first as u32, second as u32]),
            vec![1, 2],
        )
        .expect("typed axes handles");
        let first_target = decode_axes_handle(first).expect("first axes handle");
        let second_target = decode_axes_handle(second).expect("second axes handle");

        assert_eq!(
            axes_handles_from_value(&Value::Tensor(handles)).expect("axes handles"),
            vec![first_target, second_target]
        );
    }

    #[test]
    fn linkaxes_descriptor_signatures_cover_core_forms() {
        let labels: Vec<&str> = LINKAXES_DESCRIPTOR
            .signatures
            .iter()
            .map(|sig| sig.label)
            .collect();
        assert!(labels.contains(&"linkaxes(ax)"));
        assert!(labels.contains(&"linkaxes(ax, option)"));
    }

    #[test]
    fn linkaxes_propagates_selected_axis_limits_and_unlinks() {
        let _guard = setup();
        let ax1 = subplot(2.0, 1.0, 1.0);
        let ax2 = subplot(2.0, 1.0, 2.0);
        let handles = tensor(vec![ax1, ax2], vec![1, 2]);

        linkaxes_builtin(vec![handles.clone(), Value::String("x".into())]).expect("link x");
        set_limits(ax1, "XLim", [0.0, 10.0]);
        set_limits(ax2, "YLim", [5.0, 9.0]);

        assert_eq!(get_limits(ax2, "XLim"), vec![0.0, 10.0]);
        assert_ne!(get_limits(ax1, "YLim"), vec![5.0, 9.0]);

        linkaxes_builtin(vec![handles.clone(), Value::String("xy".into())]).expect("link xy");
        set_limits(ax2, "YLim", [2.0, 4.0]);
        assert_eq!(get_limits(ax1, "YLim"), vec![2.0, 4.0]);

        let x_before_off = get_limits(ax2, "XLim");
        linkaxes_builtin(vec![handles, Value::String("off".into())]).expect("unlink");
        set_limits(ax1, "XLim", [20.0, 30.0]);
        assert_eq!(get_limits(ax2, "XLim"), x_before_off);
    }

    #[test]
    fn linkaxes_initial_sync_uses_union_of_current_limits() {
        let _guard = setup();
        let ax1 = subplot(2.0, 1.0, 1.0);
        let ax2 = subplot(2.0, 1.0, 2.0);
        set_limits(ax1, "XLim", [0.0, 2.0]);
        set_limits(ax2, "XLim", [5.0, 9.0]);

        linkaxes_builtin(vec![
            tensor(vec![ax1, ax2], vec![1, 2]),
            Value::String("x".into()),
        ])
        .expect("link x");

        assert_eq!(get_limits(ax1, "XLim"), vec![0.0, 9.0]);
        assert_eq!(get_limits(ax2, "XLim"), vec![0.0, 9.0]);
    }

    #[test]
    fn linkaxes_preserves_overlapping_axis_specific_groups() {
        let _guard = setup();
        let ax1 = subplot(3.0, 1.0, 1.0);
        let ax2 = subplot(3.0, 1.0, 2.0);
        let ax3 = subplot(3.0, 1.0, 3.0);

        linkaxes_builtin(vec![
            tensor(vec![ax1, ax2], vec![1, 2]),
            Value::String("x".into()),
        ])
        .expect("link x group");
        linkaxes_builtin(vec![
            tensor(vec![ax1, ax3], vec![1, 2]),
            Value::String("y".into()),
        ])
        .expect("link y group");

        set_limits(ax2, "XLim", [10.0, 12.0]);
        assert_eq!(get_limits(ax1, "XLim"), vec![10.0, 12.0]);
        assert_ne!(get_limits(ax3, "XLim"), vec![10.0, 12.0]);

        set_limits(ax3, "YLim", [20.0, 24.0]);
        assert_eq!(get_limits(ax1, "YLim"), vec![20.0, 24.0]);
        assert_ne!(get_limits(ax2, "YLim"), vec![20.0, 24.0]);
    }

    #[test]
    fn linkaxes_clear_figure_removes_stale_groups() {
        let _guard = setup();
        let ax1 = subplot(2.0, 1.0, 1.0);
        let ax2 = subplot(2.0, 1.0, 2.0);
        linkaxes_builtin(vec![
            tensor(vec![ax1, ax2], vec![1, 2]),
            Value::String("x".into()),
        ])
        .expect("link x");

        let _ = clear_figure(None).expect("clear figure");
        let ax1 = subplot(2.0, 1.0, 1.0);
        let ax2 = subplot(2.0, 1.0, 2.0);
        set_limits(ax1, "XLim", [30.0, 40.0]);

        assert_ne!(get_limits(ax2, "XLim"), vec![30.0, 40.0]);
    }
}

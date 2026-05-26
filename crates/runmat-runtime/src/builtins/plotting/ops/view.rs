use runmat_builtins::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor, BuiltinOutputMode,
    BuiltinParamArity, BuiltinParamDescriptor, BuiltinParamType, BuiltinSignatureDescriptor,
};
use runmat_builtins::{Tensor, Value};
use runmat_macros::runtime_builtin;

use super::op_common::current_axes_target;
use super::state::{set_view_for_axes, FigureError};
use crate::builtins::plotting::type_resolvers::get_type;
use crate::{build_runtime_error, RuntimeError};

const BUILTIN_NAME: &str = "view";

const VIEW_OUTPUT_ANGLES: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "angles",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Current or assigned view angles as [az el].",
}];

const VIEW_INPUTS_NONE: [BuiltinParamDescriptor; 0] = [];

const VIEW_INPUTS_AX: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "ax",
    ty: BuiltinParamType::AxesHandle,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Axes handle to query.",
}];

const VIEW_INPUTS_VECTOR: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "angles",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Two-element [az el] vector or preset value 2/3.",
}];

const VIEW_INPUTS_AZ_EL: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "az",
        ty: BuiltinParamType::NumericScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Azimuth in degrees.",
    },
    BuiltinParamDescriptor {
        name: "el",
        ty: BuiltinParamType::NumericScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Elevation in degrees.",
    },
];

const VIEW_INPUTS_AX_VECTOR: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "ax",
        ty: BuiltinParamType::AxesHandle,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Axes handle to set.",
    },
    BuiltinParamDescriptor {
        name: "angles",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Two-element [az el] vector or preset value 2/3.",
    },
];

const VIEW_INPUTS_AX_AZ_EL: [BuiltinParamDescriptor; 3] = [
    BuiltinParamDescriptor {
        name: "ax",
        ty: BuiltinParamType::AxesHandle,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Axes handle to set.",
    },
    BuiltinParamDescriptor {
        name: "az",
        ty: BuiltinParamType::NumericScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Azimuth in degrees.",
    },
    BuiltinParamDescriptor {
        name: "el",
        ty: BuiltinParamType::NumericScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Elevation in degrees.",
    },
];

const VIEW_SIGNATURES: [BuiltinSignatureDescriptor; 6] = [
    BuiltinSignatureDescriptor {
        label: "angles = view()",
        inputs: &VIEW_INPUTS_NONE,
        outputs: &VIEW_OUTPUT_ANGLES,
    },
    BuiltinSignatureDescriptor {
        label: "angles = view(ax)",
        inputs: &VIEW_INPUTS_AX,
        outputs: &VIEW_OUTPUT_ANGLES,
    },
    BuiltinSignatureDescriptor {
        label: "angles = view([az el] | preset)",
        inputs: &VIEW_INPUTS_VECTOR,
        outputs: &VIEW_OUTPUT_ANGLES,
    },
    BuiltinSignatureDescriptor {
        label: "angles = view(az, el)",
        inputs: &VIEW_INPUTS_AZ_EL,
        outputs: &VIEW_OUTPUT_ANGLES,
    },
    BuiltinSignatureDescriptor {
        label: "angles = view(ax, [az el] | preset)",
        inputs: &VIEW_INPUTS_AX_VECTOR,
        outputs: &VIEW_OUTPUT_ANGLES,
    },
    BuiltinSignatureDescriptor {
        label: "angles = view(ax, az, el)",
        inputs: &VIEW_INPUTS_AX_AZ_EL,
        outputs: &VIEW_OUTPUT_ANGLES,
    },
];

const VIEW_ERROR_INVALID_ARGUMENT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.VIEW.INVALID_ARGUMENT",
    identifier: Some("RunMat:view:InvalidArgument"),
    when: "Arguments are malformed, wrong arity, wrong shape, or unsupported preset.",
    message: "view: invalid argument",
};

const VIEW_ERROR_INVALID_AXES: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.VIEW.INVALID_AXES",
    identifier: Some("RunMat:view:InvalidAxes"),
    when: "Axes target is invalid or no longer exists.",
    message: "view: invalid axes target",
};

const VIEW_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.VIEW.INTERNAL",
    identifier: Some("RunMat:view:Internal"),
    when: "Internal plotting state operation fails.",
    message: "view: internal operation failed",
};

const VIEW_ERRORS: [BuiltinErrorDescriptor; 3] = [
    VIEW_ERROR_INVALID_ARGUMENT,
    VIEW_ERROR_INVALID_AXES,
    VIEW_ERROR_INTERNAL,
];

pub const VIEW_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &VIEW_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &VIEW_ERRORS,
};

fn view_error(error: &'static BuiltinErrorDescriptor) -> RuntimeError {
    view_error_with_message(error.message, error)
}

fn view_error_with_message(
    message: impl Into<String>,
    error: &'static BuiltinErrorDescriptor,
) -> RuntimeError {
    let mut builder = build_runtime_error(message).with_builtin(BUILTIN_NAME);
    if let Some(identifier) = error.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

fn map_view_figure_error(err: FigureError) -> RuntimeError {
    match err {
        FigureError::InvalidSubplotIndex { .. }
        | FigureError::InvalidAxesHandle
        | FigureError::InvalidHandle(_) => view_error(&VIEW_ERROR_INVALID_AXES),
        other => view_error_with_message(
            format!("{}: {}", VIEW_ERROR_INTERNAL.message, other),
            &VIEW_ERROR_INTERNAL,
        ),
    }
}

fn parse_view_target(
    args: &[Value],
) -> crate::BuiltinResult<(
    (crate::builtins::plotting::state::FigureHandle, usize),
    &[Value],
)> {
    if let Some(first) = args.first() {
        if let Ok(crate::builtins::plotting::properties::PlotHandle::Axes(fig, axes)) =
            crate::builtins::plotting::properties::resolve_plot_handle(first, "view")
        {
            return Ok(((fig, axes), &args[1..]));
        }
    }
    Ok((current_axes_target(), args))
}

fn parse_view_angles(args: &[Value]) -> crate::BuiltinResult<(f32, f32)> {
    match args.len() {
        1 => {
            let tensor = scalar_or_tensor(&args[0])?;
            if tensor.data.len() == 1 {
                match tensor.data[0] as i32 {
                    2 => Ok((0.0, 90.0)),
                    3 => Ok((-37.5, 30.0)),
                    _ => Err(view_error(&VIEW_ERROR_INVALID_ARGUMENT)),
                }
            } else if tensor.data.len() == 2 {
                Ok((tensor.data[0] as f32, tensor.data[1] as f32))
            } else {
                Err(view_error(&VIEW_ERROR_INVALID_ARGUMENT))
            }
        }
        2 => {
            let az = scalar_or_tensor(&args[0])?;
            let el = scalar_or_tensor(&args[1])?;
            if az.data.len() != 1 || el.data.len() != 1 {
                return Err(view_error(&VIEW_ERROR_INVALID_ARGUMENT));
            }
            Ok((az.data[0] as f32, el.data[0] as f32))
        }
        _ => Err(view_error(&VIEW_ERROR_INVALID_ARGUMENT)),
    }
}

fn scalar_or_tensor(value: &Value) -> crate::BuiltinResult<Tensor> {
    match value {
        Value::Num(v) => Ok(Tensor {
            rows: 1,
            cols: 1,
            shape: vec![1, 1],
            data: vec![*v],
            dtype: runmat_builtins::NumericDType::F64,
        }),
        Value::Int(i) => Ok(Tensor {
            rows: 1,
            cols: 1,
            shape: vec![1, 1],
            data: vec![i.to_f64()],
            dtype: runmat_builtins::NumericDType::F64,
        }),
        other => Tensor::try_from(other).map_err(|_| view_error(&VIEW_ERROR_INVALID_ARGUMENT)),
    }
}

#[runtime_builtin(
    name = "view",
    category = "plotting",
    summary = "Set or query the current 3-D view angles.",
    keywords = "view,plotting,3d,camera",
    suppress_auto_output = true,
    type_resolver(get_type),
    descriptor(crate::builtins::plotting::view::VIEW_DESCRIPTOR),
    builtin_path = "crate::builtins::plotting::view"
)]
pub fn view_builtin(args: Vec<Value>) -> crate::BuiltinResult<Value> {
    let (target, rest) = parse_view_target(&args)?;
    if rest.is_empty() {
        let meta = crate::builtins::plotting::state::axes_metadata_snapshot(target.0, target.1)
            .map_err(map_view_figure_error)?;
        let az = meta.view_azimuth_deg.unwrap_or(-37.5) as f64;
        let el = meta.view_elevation_deg.unwrap_or(30.0) as f64;
        return Ok(Value::Tensor(Tensor {
            rows: 1,
            cols: 2,
            shape: vec![1, 2],
            data: vec![az, el],
            dtype: runmat_builtins::NumericDType::F64,
        }));
    }
    let (az, el) = parse_view_angles(rest)?;
    set_view_for_axes(target.0, target.1, az, el).map_err(map_view_figure_error)?;
    Ok(Value::Tensor(Tensor {
        rows: 1,
        cols: 2,
        shape: vec![1, 2],
        data: vec![az as f64, el as f64],
        dtype: runmat_builtins::NumericDType::F64,
    }))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::builtins::plotting::get::get_builtin;
    use crate::builtins::plotting::set::set_builtin;
    use crate::builtins::plotting::tests::{ensure_plot_test_env, lock_plot_registry};
    use crate::builtins::plotting::{
        clear_figure, clone_figure, configure_subplot, current_figure_handle,
        reset_hold_state_for_run,
    };

    #[test]
    fn view_sets_axes_local_angles() {
        let _guard = lock_plot_registry();
        ensure_plot_test_env();
        reset_hold_state_for_run();
        let _ = clear_figure(None);
        let value = view_builtin(vec![Value::Num(45.0), Value::Num(20.0)]).unwrap();
        let t = Tensor::try_from(&value).unwrap();
        assert_eq!(t.data, vec![45.0, 20.0]);
        let fig = clone_figure(current_figure_handle()).unwrap();
        let meta = fig.axes_metadata(0).unwrap();
        assert_eq!(meta.view_azimuth_deg, Some(45.0));
        assert_eq!(meta.view_elevation_deg, Some(20.0));
    }

    #[test]
    fn view_is_subplot_local() {
        let _guard = lock_plot_registry();
        ensure_plot_test_env();
        reset_hold_state_for_run();
        let _ = clear_figure(None);
        configure_subplot(1, 2, 1).unwrap();
        let _ = view_builtin(vec![Value::Num(10.0), Value::Num(15.0)]).unwrap();
        let fig = clone_figure(current_figure_handle()).unwrap();
        assert_eq!(fig.axes_metadata(0).unwrap().view_azimuth_deg, None);
        assert_eq!(fig.axes_metadata(1).unwrap().view_azimuth_deg, Some(10.0));
    }

    #[test]
    fn view_supports_query_forms_and_presets() {
        let _guard = lock_plot_registry();
        ensure_plot_test_env();
        reset_hold_state_for_run();
        let _ = clear_figure(None);

        let ax = Value::Num(crate::builtins::plotting::state::encode_axes_handle(
            current_figure_handle(),
            0,
        ));
        let _ = view_builtin(vec![ax.clone(), Value::Num(2.0)]).unwrap();
        let v = view_builtin(vec![ax.clone()]).unwrap();
        let t = Tensor::try_from(&v).unwrap();
        assert_eq!(t.data, vec![0.0, 90.0]);

        let _ = view_builtin(vec![ax.clone(), Value::Num(3.0)]).unwrap();
        let v = view_builtin(vec![ax]).unwrap();
        let t = Tensor::try_from(&v).unwrap();
        assert_eq!(t.data, vec![-37.5, 30.0]);
    }

    #[test]
    fn view_handles_large_angles_and_property_roundtrip() {
        let _guard = lock_plot_registry();
        ensure_plot_test_env();
        reset_hold_state_for_run();
        let _ = clear_figure(None);

        let ax = Value::Num(crate::builtins::plotting::state::encode_axes_handle(
            current_figure_handle(),
            0,
        ));
        let _ = view_builtin(vec![ax.clone(), Value::Num(405.0), Value::Num(89.0)]).unwrap();
        let queried = get_builtin(vec![ax.clone(), Value::String("View".into())]).unwrap();
        let t = Tensor::try_from(&queried).unwrap();
        assert_eq!(t.data, vec![405.0, 89.0]);

        set_builtin(vec![
            ax,
            Value::String("View".into()),
            Value::Tensor(Tensor {
                rows: 1,
                cols: 2,
                shape: vec![1, 2],
                data: vec![180.0, -30.0],
                dtype: runmat_builtins::NumericDType::F64,
            }),
        ])
        .unwrap();
        let fig = clone_figure(current_figure_handle()).unwrap();
        assert_eq!(fig.axes_metadata(0).unwrap().view_azimuth_deg, Some(180.0));
        assert_eq!(
            fig.axes_metadata(0).unwrap().view_elevation_deg,
            Some(-30.0)
        );
    }

    #[test]
    fn view_descriptor_signatures_cover_query_and_set_forms() {
        let labels: Vec<&str> = VIEW_DESCRIPTOR
            .signatures
            .iter()
            .map(|sig| sig.label)
            .collect();
        assert!(labels.contains(&"angles = view()"));
        assert!(labels.contains(&"angles = view(ax)"));
        assert!(labels.contains(&"angles = view([az el] | preset)"));
        assert!(labels.contains(&"angles = view(az, el)"));
        assert!(labels.contains(&"angles = view(ax, [az el] | preset)"));
        assert!(labels.contains(&"angles = view(ax, az, el)"));
    }
}

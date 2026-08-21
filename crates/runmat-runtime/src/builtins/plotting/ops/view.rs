use runmat_builtins::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor, BuiltinExtensionDescriptor,
    BuiltinExtensionMode, BuiltinIntegerBackendRule, BuiltinIntegerCapabilityDescriptor,
    BuiltinIntegerComputationDomain, BuiltinIntegerInputAvailability,
    BuiltinIntegerInputCapability, BuiltinIntegerOutputClassRule, BuiltinIntegerOverflowRule,
    BuiltinIntegerOverloadKind, BuiltinIntegerScalarDoubleRule, BuiltinOutputMode,
    BuiltinParamArity, BuiltinParamDescriptor, BuiltinParamType, BuiltinSignatureDescriptor,
};
use runmat_macros::runtime_builtin;
use runmat_value::{IntValue, IntegerStorage, NumericScalar, Tensor, Value};

use super::op_common::current_axes_target;
use super::state::{set_view_for_axes, FigureError};
use crate::builtins::common::tensor;
use crate::builtins::plotting::type_resolvers::get_type;
use crate::{build_runtime_error, RuntimeError};

const BUILTIN_NAME: &str = "view";

const INTEGER_ANGLES_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "view-integer-angles",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "view with typed-integer angles or preset storage is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:ViewIntegerAnglesExtension"),
};
const INTEGER_AXES_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "view-integer-axes-handle",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "view with a typed-integer numeric axes alias is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:ViewIntegerAxesHandleExtension"),
};
pub const VIEW_EXTENSIONS: [BuiltinExtensionDescriptor; 2] =
    [INTEGER_ANGLES_EXTENSION, INTEGER_AXES_EXTENSION];

const INTEGER_ANGLE_INPUTS: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "az, el, v, or dim",
        classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "The public surface describes numbers and line-of-sight arrays without enumerating native integer storage; RunMat exposes typed integer angles and 2/3 presets behind one explicit extension gate.",
    }];
const INTEGER_AXES_INPUTS: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "ax",
        classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "The documented target is an Axes object; typed-integer numeric aliases are a gated RunMat graphics representation extension.",
    }];
pub const VIEW_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 2] = [
    BuiltinIntegerCapabilityDescriptor {
        form: "angles = view(integer_angles_or_preset)",
        inputs: &INTEGER_ANGLE_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::FloatingPoint,
        output_class: BuiltinIntegerOutputClassRule::Double,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::HostOnly,
        overload: BuiltinIntegerOverloadKind::Multiple,
        notes: "Host integer storage remains exact until each angle crosses the renderer's binary32 camera-state boundary; values that cannot be represented exactly reject before plot mutation.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "angles = view(integer_ax, ...)",
        inputs: &INTEGER_AXES_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::Structural,
        output_class: BuiltinIntegerOutputClassRule::Double,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::HostOnly,
        overload: BuiltinIntegerOverloadKind::StructuralParameter,
        notes: "The compatibility gate runs before numeric handle decoding or axes-state access; resident integer handles are not gathered.",
    },
];

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
            if tensor.len() == 1 {
                let value = view_value_f32(&tensor, 0)?;
                if value == 2.0 {
                    Ok((0.0, 90.0))
                } else if value == 3.0 {
                    Ok((-37.5, 30.0))
                } else {
                    Err(view_error(&VIEW_ERROR_INVALID_ARGUMENT))
                }
            } else if tensor.len() == 2 {
                Ok((view_value_f32(&tensor, 0)?, view_value_f32(&tensor, 1)?))
            } else {
                Err(view_error(&VIEW_ERROR_INVALID_ARGUMENT))
            }
        }
        2 => {
            let az = scalar_or_tensor(&args[0])?;
            let el = scalar_or_tensor(&args[1])?;
            if !tensor::is_scalar_tensor(&az) || !tensor::is_scalar_tensor(&el) {
                return Err(view_error(&VIEW_ERROR_INVALID_ARGUMENT));
            }
            Ok((view_value_f32(&az, 0)?, view_value_f32(&el, 0)?))
        }
        _ => Err(view_error(&VIEW_ERROR_INVALID_ARGUMENT)),
    }
}

fn scalar_or_tensor(value: &Value) -> crate::BuiltinResult<Tensor> {
    match value {
        Value::Num(v) => {
            Tensor::new(vec![*v], vec![1, 1]).map_err(|_| view_error(&VIEW_ERROR_INVALID_ARGUMENT))
        }
        Value::Int(i) => Tensor::new_integer(IntegerStorage::from_scalar(i.clone()), vec![1, 1])
            .map_err(|_| view_error(&VIEW_ERROR_INVALID_ARGUMENT)),
        other => Tensor::try_from(other).map_err(|_| view_error(&VIEW_ERROR_INVALID_ARGUMENT)),
    }
}

fn view_value_f32(tensor: &Tensor, index: usize) -> crate::BuiltinResult<f32> {
    let value = tensor
        .numeric_value_at(index)
        .ok_or_else(|| view_error(&VIEW_ERROR_INVALID_ARGUMENT))?;
    match value {
        NumericScalar::F64(value) => Ok(value as f32),
        NumericScalar::F32(value) => Ok(value),
        integer => {
            let integer = integer
                .into_int_value()
                .expect("nonfloating NumericScalar is an integer");
            exact_integer_f32(integer).ok_or_else(|| {
                view_error_with_message(
                    "view: integer angles must be exactly representable as single",
                    &VIEW_ERROR_INVALID_ARGUMENT,
                )
            })
        }
    }
}

fn exact_integer_f32(value: IntValue) -> Option<f32> {
    use num_traits::FromPrimitive;
    let converted = value.to_f64() as f32;
    let exact = match value {
        IntValue::I8(value) => num_bigint::BigInt::from(value),
        IntValue::I16(value) => num_bigint::BigInt::from(value),
        IntValue::I32(value) => num_bigint::BigInt::from(value),
        IntValue::I64(value) => num_bigint::BigInt::from(value),
        IntValue::U8(value) => num_bigint::BigInt::from(value),
        IntValue::U16(value) => num_bigint::BigInt::from(value),
        IntValue::U32(value) => num_bigint::BigInt::from(value),
        IntValue::U64(value) => num_bigint::BigInt::from(value),
    };
    (num_bigint::BigInt::from_f32(converted).as_ref() == Some(&exact)).then_some(converted)
}

fn is_typed_integer(value: &Value) -> bool {
    matches!(value, Value::Int(_))
        || matches!(value, Value::Tensor(tensor) if tensor.integer_storage().is_some())
        || matches!(value, Value::GpuTensor(handle) if runmat_accelerate_api::handle_integer_type(handle).is_some())
}

#[runtime_builtin(
    name = "view",
    category = "plotting",
    summary = "Set or query azimuth/elevation camera view angles.",
    keywords = "view,plotting,3d,camera",
    suppress_auto_output = true,
    type_resolver(get_type),
    descriptor(crate::builtins::plotting::view::VIEW_DESCRIPTOR),
    extensions(crate::builtins::plotting::view::VIEW_EXTENSIONS),
    integer_capabilities(crate::builtins::plotting::view::VIEW_INTEGER_CAPABILITIES),
    builtin_path = "crate::builtins::plotting::view"
)]
pub fn view_builtin(args: Vec<Value>) -> crate::BuiltinResult<Value> {
    if let Some(first) = args.first() {
        if is_typed_integer(first)
            && matches!(
                crate::builtins::plotting::properties::resolve_plot_handle(first, BUILTIN_NAME),
                Ok(crate::builtins::plotting::properties::PlotHandle::Axes(
                    _,
                    _
                ))
            )
        {
            crate::compatibility::ensure_builtin_extension_enabled(
                &INTEGER_AXES_EXTENSION,
                BUILTIN_NAME,
            )?;
        }
    }
    let (target, rest) = parse_view_target(&args)?;
    if rest.is_empty() {
        let meta = crate::builtins::plotting::state::axes_metadata_snapshot(target.0, target.1)
            .map_err(map_view_figure_error)?;
        let az = meta.view_azimuth_deg.unwrap_or(-37.5) as f64;
        let el = meta.view_elevation_deg.unwrap_or(30.0) as f64;
        return Ok(Value::Tensor(
            Tensor::new(vec![az, el], vec![1, 2]).expect("view vector"),
        ));
    }
    if rest.iter().any(is_typed_integer) {
        crate::compatibility::ensure_builtin_extension_enabled(
            &INTEGER_ANGLES_EXTENSION,
            BUILTIN_NAME,
        )?;
    }
    let (az, el) = parse_view_angles(rest)?;
    set_view_for_axes(target.0, target.1, az, el).map_err(map_view_figure_error)?;
    Ok(Value::Tensor(
        Tensor::new(vec![az as f64, el as f64], vec![1, 2]).expect("view vector"),
    ))
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
    use runmat_value::IntegerStorage;

    #[test]
    fn view_sets_axes_local_angles() {
        let _guard = lock_plot_registry();
        ensure_plot_test_env();
        reset_hold_state_for_run();
        let _ = clear_figure(None);
        let value = view_builtin(vec![Value::Num(45.0), Value::Num(20.0)]).unwrap();
        let t = Tensor::try_from(&value).unwrap();
        assert_eq!(t.materialize_f64(), vec![45.0, 20.0]);
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
        assert_eq!(t.materialize_f64(), vec![0.0, 90.0]);

        let _ = view_builtin(vec![ax.clone(), Value::Num(3.0)]).unwrap();
        let v = view_builtin(vec![ax]).unwrap();
        let t = Tensor::try_from(&v).unwrap();
        assert_eq!(t.materialize_f64(), vec![-37.5, 30.0]);
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
        assert_eq!(t.materialize_f64(), vec![405.0, 89.0]);

        set_builtin(vec![
            ax,
            Value::String("View".into()),
            Value::Tensor(Tensor::new(vec![180.0, -30.0], vec![1, 2]).expect("view vector")),
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
    fn view_accepts_typed_integer_angle_vector() {
        let _extensions = crate::compatibility::push_runmat_extensions_enabled(true);
        let _guard = lock_plot_registry();
        ensure_plot_test_env();
        reset_hold_state_for_run();
        let _ = clear_figure(None);

        let tensor =
            Tensor::new_integer(IntegerStorage::I16(vec![45, 20]), vec![1, 2]).expect("angles");
        let angles = Value::Tensor(tensor);
        let value = view_builtin(vec![angles]).unwrap();
        let t = Tensor::try_from(&value).unwrap();

        assert_eq!(t.materialize_f64(), vec![45.0, 20.0]);
    }

    #[test]
    fn view_accepts_separate_typed_integer_angle_scalars() {
        let _extensions = crate::compatibility::push_runmat_extensions_enabled(true);
        let _guard = lock_plot_registry();
        ensure_plot_test_env();
        reset_hold_state_for_run();
        let _ = clear_figure(None);

        let az = Tensor::new_integer(IntegerStorage::I16(vec![45]), vec![1, 1]).expect("azimuth");
        let el = Tensor::new_integer(IntegerStorage::I16(vec![20]), vec![1, 1]).expect("elevation");

        let value = view_builtin(vec![Value::Tensor(az), Value::Tensor(el)]).unwrap();
        let t = Tensor::try_from(&value).unwrap();

        assert_eq!(t.materialize_f64(), vec![45.0, 20.0]);
    }

    #[test]
    fn view_integer_angles_are_gated_and_exact_at_camera_boundary() {
        let _guard = lock_plot_registry();
        ensure_plot_test_env();
        reset_hold_state_for_run();
        let _ = clear_figure(None);

        let angles = Value::Tensor(
            Tensor::new_integer(IntegerStorage::I32(vec![45, 20]), vec![1, 2])
                .expect("integer angles"),
        );
        {
            let _compat = crate::compatibility::push_runmat_extensions_enabled(false);
            let error = view_builtin(vec![angles.clone()]).expect_err("compatibility gate");
            assert_eq!(
                error.identifier(),
                Some("RunMat:compatibility:ViewIntegerAnglesExtension")
            );
        }
        {
            let _runmat = crate::compatibility::push_runmat_extensions_enabled(true);
            let lossy = Value::Int(IntValue::I32(16_777_217));
            let error = view_builtin(vec![lossy, Value::Int(IntValue::I8(0))])
                .expect_err("lossy binary32 angle");
            assert!(error.message().contains("exactly representable as single"));
        }
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

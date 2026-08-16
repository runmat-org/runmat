use runmat_builtins::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor, BuiltinExtensionDescriptor,
    BuiltinExtensionMode, BuiltinIntegerBackendRule, BuiltinIntegerCapabilityDescriptor,
    BuiltinIntegerComputationDomain, BuiltinIntegerInputAvailability,
    BuiltinIntegerInputCapability, BuiltinIntegerOutputClassRule, BuiltinIntegerOverflowRule,
    BuiltinIntegerOverloadKind, BuiltinIntegerScalarDoubleRule, BuiltinOutputMode,
    BuiltinParamArity, BuiltinParamDescriptor, BuiltinParamType, BuiltinSignatureDescriptor,
    NumericStorage, Tensor, Value,
};
use runmat_macros::runtime_builtin;

use super::op_common::{map_figure_error, value_as_text_string};
use crate::builtins::plotting::properties::parse_text_style_pairs;
use crate::builtins::plotting::state::add_text_annotation_for_axes_with_source;
use crate::builtins::plotting::type_resolvers::handle_scalar_type;
use crate::{build_runtime_error, RuntimeError};

const BUILTIN_NAME: &str = "text";

const INTEGER_AXES_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "text-integer-axes-handle",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "text with a typed-integer numeric axes alias is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:TextIntegerAxesHandleExtension"),
};
const EXPLICIT_GPU_INPUT_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "text-explicit-gpu-input",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "text with explicit gpuArray coordinates is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:TextExplicitGpuInputExtension"),
};
pub const TEXT_EXTENSIONS: [BuiltinExtensionDescriptor; 2] =
    [INTEGER_AXES_EXTENSION, EXPLICIT_GPU_INPUT_EXTENSION];

const INTEGER_COORDINATE_INPUTS: [BuiltinIntegerInputCapability; 3] = [
    BuiltinIntegerInputCapability {
        name: "x",
        classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "The public Position property explicitly accepts all eight integer classes; native coordinate identity is retained before the renderer boundary.",
    },
    BuiltinIntegerInputCapability {
        name: "y",
        classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "Integer y coordinates are documented graphics data and cross to f32 only in the derived render cache.",
    },
    BuiltinIntegerInputCapability {
        name: "z",
        classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "Integer z coordinates are documented graphics data and cross to f32 only in the derived render cache.",
    },
];
const INTEGER_AXES_INPUT: [BuiltinIntegerInputCapability; 1] = [BuiltinIntegerInputCapability {
    name: "ax",
    classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES,
    availability: BuiltinIntegerInputAvailability::RunMatOnly,
    scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
    notes: "MATLAB axes are graphics objects; typed-integer numeric aliases are a gated RunMat representation extension.",
}];
pub const TEXT_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 2] = [
    BuiltinIntegerCapabilityDescriptor {
        form: "h = text(integer_x, integer_y [, integer_z], label, ...)",
        inputs: &INTEGER_COORDINATE_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::FunctionSpecific,
        output_class: BuiltinIntegerOutputClassRule::FunctionSpecific,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::HostOnly,
        overload: BuiltinIntegerOverloadKind::Multiple,
        notes: "Documented native coordinates remain the semantic source while the plotting backend receives an explicitly derived f32 position.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "h = text(integer_ax, x, y [, z], label, ...)",
        inputs: &INTEGER_AXES_INPUT,
        computation_domain: BuiltinIntegerComputationDomain::Structural,
        output_class: BuiltinIntegerOutputClassRule::FunctionSpecific,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::HostOnly,
        overload: BuiltinIntegerOverloadKind::StructuralParameter,
        notes: "The axes-alias extension gate runs before target selection or graphics mutation.",
    },
];

const TEXT_OUTPUT_HANDLE: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "h",
    ty: BuiltinParamType::NumericScalar,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Handle to the created text annotation.",
}];

const TEXT_INPUTS_X_Y_LABEL: [BuiltinParamDescriptor; 3] = [
    BuiltinParamDescriptor {
        name: "x",
        ty: BuiltinParamType::NumericScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "X coordinate.",
    },
    BuiltinParamDescriptor {
        name: "y",
        ty: BuiltinParamType::NumericScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Y coordinate.",
    },
    BuiltinParamDescriptor {
        name: "label",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Annotation text.",
    },
];

const TEXT_INPUTS_X_Y_Z_LABEL: [BuiltinParamDescriptor; 4] = [
    BuiltinParamDescriptor {
        name: "x",
        ty: BuiltinParamType::NumericScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "X coordinate.",
    },
    BuiltinParamDescriptor {
        name: "y",
        ty: BuiltinParamType::NumericScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Y coordinate.",
    },
    BuiltinParamDescriptor {
        name: "z",
        ty: BuiltinParamType::NumericScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Z coordinate.",
    },
    BuiltinParamDescriptor {
        name: "label",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Annotation text.",
    },
];

const TEXT_INPUTS_X_Y_LABEL_PROPS: [BuiltinParamDescriptor; 4] = [
    BuiltinParamDescriptor {
        name: "x",
        ty: BuiltinParamType::NumericScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "X coordinate.",
    },
    BuiltinParamDescriptor {
        name: "y",
        ty: BuiltinParamType::NumericScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Y coordinate.",
    },
    BuiltinParamDescriptor {
        name: "label",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Annotation text.",
    },
    BuiltinParamDescriptor {
        name: "props",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Variadic,
        default: None,
        description: "Name/value style options.",
    },
];

const TEXT_INPUTS_X_Y_Z_LABEL_PROPS: [BuiltinParamDescriptor; 5] = [
    BuiltinParamDescriptor {
        name: "x",
        ty: BuiltinParamType::NumericScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "X coordinate.",
    },
    BuiltinParamDescriptor {
        name: "y",
        ty: BuiltinParamType::NumericScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Y coordinate.",
    },
    BuiltinParamDescriptor {
        name: "z",
        ty: BuiltinParamType::NumericScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Z coordinate.",
    },
    BuiltinParamDescriptor {
        name: "label",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Annotation text.",
    },
    BuiltinParamDescriptor {
        name: "props",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Variadic,
        default: None,
        description: "Name/value style options.",
    },
];

const TEXT_INPUTS_AX_X_Y_LABEL: [BuiltinParamDescriptor; 4] = [
    BuiltinParamDescriptor {
        name: "ax",
        ty: BuiltinParamType::AxesHandle,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Target axes handle.",
    },
    BuiltinParamDescriptor {
        name: "x",
        ty: BuiltinParamType::NumericScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "X coordinate.",
    },
    BuiltinParamDescriptor {
        name: "y",
        ty: BuiltinParamType::NumericScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Y coordinate.",
    },
    BuiltinParamDescriptor {
        name: "label",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Annotation text.",
    },
];

const TEXT_INPUTS_AX_X_Y_Z_LABEL: [BuiltinParamDescriptor; 5] = [
    BuiltinParamDescriptor {
        name: "ax",
        ty: BuiltinParamType::AxesHandle,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Target axes handle.",
    },
    BuiltinParamDescriptor {
        name: "x",
        ty: BuiltinParamType::NumericScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "X coordinate.",
    },
    BuiltinParamDescriptor {
        name: "y",
        ty: BuiltinParamType::NumericScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Y coordinate.",
    },
    BuiltinParamDescriptor {
        name: "z",
        ty: BuiltinParamType::NumericScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Z coordinate.",
    },
    BuiltinParamDescriptor {
        name: "label",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Annotation text.",
    },
];

const TEXT_SIGNATURES: [BuiltinSignatureDescriptor; 6] = [
    BuiltinSignatureDescriptor {
        label: "h = text(x, y, label)",
        inputs: &TEXT_INPUTS_X_Y_LABEL,
        outputs: &TEXT_OUTPUT_HANDLE,
    },
    BuiltinSignatureDescriptor {
        label: "h = text(x, y, label, Name, Value, ...)",
        inputs: &TEXT_INPUTS_X_Y_LABEL_PROPS,
        outputs: &TEXT_OUTPUT_HANDLE,
    },
    BuiltinSignatureDescriptor {
        label: "h = text(x, y, z, label)",
        inputs: &TEXT_INPUTS_X_Y_Z_LABEL,
        outputs: &TEXT_OUTPUT_HANDLE,
    },
    BuiltinSignatureDescriptor {
        label: "h = text(x, y, z, label, Name, Value, ...)",
        inputs: &TEXT_INPUTS_X_Y_Z_LABEL_PROPS,
        outputs: &TEXT_OUTPUT_HANDLE,
    },
    BuiltinSignatureDescriptor {
        label: "h = text(ax, x, y, label)",
        inputs: &TEXT_INPUTS_AX_X_Y_LABEL,
        outputs: &TEXT_OUTPUT_HANDLE,
    },
    BuiltinSignatureDescriptor {
        label: "h = text(ax, x, y, z, label)",
        inputs: &TEXT_INPUTS_AX_X_Y_Z_LABEL,
        outputs: &TEXT_OUTPUT_HANDLE,
    },
];

const TEXT_ERROR_INVALID_ARGUMENT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.TEXT.INVALID_ARGUMENT",
    identifier: Some("RunMat:text:InvalidArgument"),
    when: "Coordinate, label, axes-target, or name/value text style arguments are invalid.",
    message: "text: invalid argument",
};

const TEXT_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.TEXT.INTERNAL",
    identifier: Some("RunMat:text:Internal"),
    when: "Internal plotting state update fails while creating annotation.",
    message: "text: internal operation failed",
};

const TEXT_ERRORS: [BuiltinErrorDescriptor; 2] = [TEXT_ERROR_INVALID_ARGUMENT, TEXT_ERROR_INTERNAL];

pub const TEXT_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &TEXT_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &TEXT_ERRORS,
};

fn text_error_with_detail(
    error: &'static BuiltinErrorDescriptor,
    detail: impl AsRef<str>,
) -> RuntimeError {
    let mut builder = build_runtime_error(format!("{}: {}", error.message, detail.as_ref()))
        .with_builtin(BUILTIN_NAME);
    if let Some(identifier) = error.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

fn map_text_invalid_argument(err: RuntimeError) -> RuntimeError {
    if err.identifier().is_some() {
        return err;
    }
    text_error_with_detail(&TEXT_ERROR_INVALID_ARGUMENT, err.message)
}

fn map_text_internal(err: RuntimeError) -> RuntimeError {
    if err.identifier().is_some() {
        return err;
    }
    text_error_with_detail(&TEXT_ERROR_INTERNAL, err.message)
}

#[runtime_builtin(
    name = "text",
    category = "plotting",
    summary = "Add text annotation at a 2-D or 3-D plot position.",
    keywords = "text,annotation,plotting",
    suppress_auto_output = true,
    type_resolver(handle_scalar_type),
    descriptor(crate::builtins::plotting::text::TEXT_DESCRIPTOR),
    extensions(crate::builtins::plotting::text::TEXT_EXTENSIONS),
    integer_capabilities(crate::builtins::plotting::text::TEXT_INTEGER_CAPABILITIES),
    builtin_path = "crate::builtins::plotting::text"
)]
pub fn text_builtin(args: Vec<Value>) -> crate::BuiltinResult<f64> {
    if args
        .iter()
        .any(crate::builtins::common::validation::value_contains_explicit_gpu)
    {
        crate::compatibility::ensure_builtin_extension_enabled(
            &EXPLICIT_GPU_INPUT_EXTENSION,
            BUILTIN_NAME,
        )?;
    }
    let typed_tensor_coordinate = matches!(
        args.first(),
        Some(Value::Tensor(tensor)) if tensor.integer_storage().is_some()
    );
    let (target, rest) = if typed_tensor_coordinate {
        (
            super::op_common::text::current_axes_target(),
            args.as_slice(),
        )
    } else {
        super::op_common::text::split_axes_target(BUILTIN_NAME, &args)
            .map_err(map_text_invalid_argument)?
    };
    if rest.len() != args.len()
        && args
            .first()
            .is_some_and(crate::builtins::common::validation::value_has_native_integer_class)
    {
        crate::compatibility::ensure_builtin_extension_enabled(
            &INTEGER_AXES_EXTENSION,
            BUILTIN_NAME,
        )?;
    }
    if rest.len() < 3 {
        return Err(text_error_with_detail(
            &TEXT_ERROR_INVALID_ARGUMENT,
            "expected text(x, y, label) or text(x, y, z, label)",
        ));
    }
    let (z, text_idx) = if let Some(text) = value_as_text_string(&rest[2]) {
        let (position, source) = position_source(&rest[0], &rest[1], None)?;
        let style =
            parse_text_style_pairs(BUILTIN_NAME, &rest[3..]).map_err(map_text_invalid_argument)?;
        return add_text_annotation_for_axes_with_source(
            target.0,
            target.1,
            position,
            &text,
            style,
            Some(source),
        )
        .map_err(|err| map_text_internal(map_figure_error(BUILTIN_NAME, err)));
    } else {
        (rest[2].clone(), 3usize)
    };

    if rest.len() <= text_idx {
        return Err(text_error_with_detail(
            &TEXT_ERROR_INVALID_ARGUMENT,
            "expected annotation string",
        ));
    }
    let text = value_as_text_string(&rest[text_idx]).ok_or_else(|| {
        text_error_with_detail(&TEXT_ERROR_INVALID_ARGUMENT, "label must be text")
    })?;
    let style = parse_text_style_pairs(BUILTIN_NAME, &rest[text_idx + 1..])
        .map_err(map_text_invalid_argument)?;
    let (position, source) = position_source(&rest[0], &rest[1], Some(&z))?;
    add_text_annotation_for_axes_with_source(
        target.0,
        target.1,
        position,
        &text,
        style,
        Some(source),
    )
    .map_err(|err| map_text_internal(map_figure_error(BUILTIN_NAME, err)))
}

fn position_source(
    x: &Value,
    y: &Value,
    z: Option<&Value>,
) -> crate::BuiltinResult<(glam::Vec3, runmat_plot::plots::NumericPlotData)> {
    let x = scalar_numeric_tensor(x, "x")?;
    let y = scalar_numeric_tensor(y, "y")?;
    let z = match z {
        Some(value) => scalar_numeric_tensor(value, "z")?,
        None => {
            Tensor::from_numeric_storage(NumericStorage::zeros(x.numeric_dtype(), 1), vec![1, 1])
                .map_err(|err| text_error_with_detail(&TEXT_ERROR_INTERNAL, err))?
        }
    };
    let tensors = [&x, &y, &z];
    let same_class = tensors
        .iter()
        .all(|tensor| tensor.numeric_dtype() == x.numeric_dtype());
    let storage = if same_class {
        let mut storage = NumericStorage::zeros(x.numeric_dtype(), 3);
        for (index, tensor) in tensors.into_iter().enumerate() {
            storage
                .set_value(
                    index,
                    tensor
                        .numeric_value_at(0)
                        .expect("validated scalar numeric tensor"),
                )
                .map_err(|err| text_error_with_detail(&TEXT_ERROR_INTERNAL, err))?;
        }
        storage
    } else {
        NumericStorage::F64(
            tensors
                .into_iter()
                .map(|tensor| crate::builtins::common::tensor::tensor_value_f64(tensor, 0))
                .collect(),
        )
    };
    let source = runmat_plot::plots::NumericPlotData::new(storage, vec![1, 3])
        .map_err(|err| text_error_with_detail(&TEXT_ERROR_INTERNAL, err))?;
    let rendered = source.materialize_f64();
    Ok((
        glam::Vec3::new(rendered[0] as f32, rendered[1] as f32, rendered[2] as f32),
        source,
    ))
}

fn scalar_numeric_tensor(value: &Value, role: &str) -> crate::BuiltinResult<Tensor> {
    let gathered;
    let value = if matches!(value, Value::GpuTensor(_)) {
        gathered = futures::executor::block_on(crate::gather_if_needed_async(value))
            .map_err(|err| text_error_with_detail(&TEXT_ERROR_INTERNAL, err.message()))?;
        &gathered
    } else {
        value
    };
    let tensor =
        crate::builtins::common::tensor::value_into_tensor_for(BUILTIN_NAME, value.clone())
            .map_err(|_| {
                text_error_with_detail(
                    &TEXT_ERROR_INVALID_ARGUMENT,
                    format!("{role} must be a real numeric scalar"),
                )
            })?;
    if !crate::builtins::common::tensor::is_scalar_tensor(&tensor) {
        return Err(text_error_with_detail(
            &TEXT_ERROR_INVALID_ARGUMENT,
            format!("{role} must be a real numeric scalar"),
        ));
    }
    Ok(tensor)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::builtins::plotting::get::get_builtin;
    use crate::builtins::plotting::set::set_builtin;
    use crate::builtins::plotting::state::PlotTestLockGuard;
    use crate::builtins::plotting::tests::{ensure_plot_test_env, lock_plot_registry};
    use crate::builtins::plotting::{
        clear_figure, current_figure_handle, reset_hold_state_for_run,
    };
    use runmat_builtins::Tensor;

    fn setup() -> PlotTestLockGuard {
        let guard = lock_plot_registry();
        ensure_plot_test_env();
        reset_hold_state_for_run();
        let _ = clear_figure(None);
        guard
    }

    #[test]
    fn text_descriptor_signatures_cover_core_forms() {
        let labels: Vec<&str> = TEXT_DESCRIPTOR
            .signatures
            .iter()
            .map(|sig| sig.label)
            .collect();
        assert!(labels.contains(&"h = text(x, y, label)"));
        assert!(labels.contains(&"h = text(x, y, z, label)"));
        assert!(labels.contains(&"h = text(ax, x, y, label)"));
    }

    #[test]
    fn text_missing_args_uses_stable_identifier() {
        let _guard = setup();
        let err = text_builtin(vec![]).expect_err("missing args should fail");
        assert_eq!(err.identifier(), TEXT_ERROR_INVALID_ARGUMENT.identifier);
    }

    #[test]
    fn text_creates_world_annotation_handle() {
        let _guard = setup();
        let handle = text_builtin(vec![
            Value::Num(1.0),
            Value::Num(2.0),
            Value::String("Hello".into()),
        ])
        .unwrap();
        let position =
            get_builtin(vec![Value::Num(handle), Value::String("Position".into())]).unwrap();
        let tensor = Tensor::try_from(&position).unwrap();
        assert_eq!(tensor.materialize_f64(), vec![1.0, 2.0, 0.0]);
    }

    #[test]
    fn text_supports_3d_form() {
        let _guard = setup();
        let handle = text_builtin(vec![
            Value::Num(1.0),
            Value::Num(2.0),
            Value::Num(3.0),
            Value::String("Hello".into()),
        ])
        .unwrap();
        let fig = crate::builtins::plotting::clone_figure(current_figure_handle()).unwrap();
        let annotation = fig.axes_text_annotations(0).first().unwrap();
        assert_eq!(annotation.text, "Hello");
        assert_eq!(annotation.position, glam::Vec3::new(1.0, 2.0, 3.0));
        let string = get_builtin(vec![Value::Num(handle), Value::String("String".into())]).unwrap();
        assert_eq!(string, Value::String("Hello".into()));
    }

    #[test]
    fn text_position_get_and_set_preserve_wide_integer_storage() {
        let _guard = setup();
        let wide = 9_007_199_254_740_993_u64;
        let scalar = |value| {
            Value::Tensor(
                Tensor::from_numeric_storage(NumericStorage::U64(vec![value]), vec![1, 1]).unwrap(),
            )
        };
        let handle = text_builtin(vec![
            scalar(wide),
            scalar(wide + 1),
            Value::String("exact".into()),
        ])
        .unwrap();
        let position = Tensor::try_from(
            &get_builtin(vec![Value::Num(handle), Value::String("Position".into())]).unwrap(),
        )
        .unwrap();
        assert_eq!(
            position.into_numeric_storage().unwrap(),
            NumericStorage::U64(vec![wide, wide + 1, 0])
        );

        let replacement = Tensor::from_numeric_storage(
            NumericStorage::I64(vec![-(wide as i64), 7, 9]),
            vec![1, 3],
        )
        .unwrap();
        set_builtin(vec![
            Value::Num(handle),
            Value::String("Position".into()),
            Value::Tensor(replacement),
        ])
        .unwrap();
        let position = Tensor::try_from(
            &get_builtin(vec![Value::Num(handle), Value::String("Position".into())]).unwrap(),
        )
        .unwrap();
        assert_eq!(
            position.into_numeric_storage().unwrap(),
            NumericStorage::I64(vec![-(wide as i64), 7, 9])
        );
    }

    #[test]
    fn text_documented_integer_coordinates_remain_available_in_compatibility_mode() {
        let _guard = setup();
        let _compat = crate::compatibility::push_runmat_extensions_enabled(false);
        let coordinate = |value| {
            Value::Tensor(
                Tensor::from_numeric_storage(NumericStorage::I32(vec![value]), vec![1, 1]).unwrap(),
            )
        };
        let handle = text_builtin(vec![
            coordinate(4),
            coordinate(-7),
            Value::String("documented".into()),
        ])
        .expect("documented integer coordinates");
        let position = Tensor::try_from(
            &get_builtin(vec![Value::Num(handle), Value::String("Position".into())]).unwrap(),
        )
        .unwrap();
        assert_eq!(
            position.into_numeric_storage().unwrap(),
            NumericStorage::I32(vec![4, -7, 0])
        );
    }

    #[test]
    fn text_annotations_clear_on_fresh_axes_replot() {
        let _guard = setup();
        let _ = text_builtin(vec![
            Value::Num(0.5),
            Value::Num(0.0),
            Value::String("midpoint".into()),
        ])
        .unwrap();
        futures::executor::block_on(crate::builtins::plotting::plot::plot_builtin(vec![
            Value::Tensor(Tensor::new_2d(vec![0.0, 1.0], 1, 2).unwrap()),
            Value::Tensor(Tensor::new_2d(vec![0.0, 1.0], 1, 2).unwrap()),
        ]))
        .unwrap();
        let fig = crate::builtins::plotting::clone_figure(current_figure_handle()).unwrap();
        assert!(fig.axes_text_annotations(0).is_empty());
    }
}

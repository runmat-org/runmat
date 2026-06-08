use runmat_builtins::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor, BuiltinOutputMode,
    BuiltinParamArity, BuiltinParamDescriptor, BuiltinParamType, BuiltinSignatureDescriptor, Value,
};
use runmat_macros::runtime_builtin;

use super::op_common::{map_figure_error, value_as_text_string};
use crate::builtins::plotting::properties::parse_text_style_pairs;
use crate::builtins::plotting::state::add_text_annotation_for_axes;
use crate::builtins::plotting::style::value_as_f64;
use crate::builtins::plotting::type_resolvers::handle_scalar_type;
use crate::{build_runtime_error, RuntimeError};

const BUILTIN_NAME: &str = "text";

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
    builtin_path = "crate::builtins::plotting::text"
)]
pub fn text_builtin(args: Vec<Value>) -> crate::BuiltinResult<f64> {
    let (target, rest) = super::op_common::text::split_axes_target(BUILTIN_NAME, &args)
        .map_err(map_text_invalid_argument)?;
    if rest.len() < 3 {
        return Err(text_error_with_detail(
            &TEXT_ERROR_INVALID_ARGUMENT,
            "expected text(x, y, label) or text(x, y, z, label)",
        ));
    }
    let x = value_as_f64(&rest[0])
        .ok_or_else(|| text_error_with_detail(&TEXT_ERROR_INVALID_ARGUMENT, "x must be numeric"))?;
    let y = value_as_f64(&rest[1])
        .ok_or_else(|| text_error_with_detail(&TEXT_ERROR_INVALID_ARGUMENT, "y must be numeric"))?;

    let (z, text_idx) = if let Some(text) = value_as_text_string(&rest[2]) {
        let style =
            parse_text_style_pairs(BUILTIN_NAME, &rest[3..]).map_err(map_text_invalid_argument)?;
        return add_text_annotation_for_axes(
            target.0,
            target.1,
            glam::Vec3::new(x as f32, y as f32, 0.0),
            &text,
            style,
        )
        .map_err(|err| map_text_internal(map_figure_error(BUILTIN_NAME, err)));
    } else {
        let z = value_as_f64(&rest[2]).ok_or_else(|| {
            text_error_with_detail(
                &TEXT_ERROR_INVALID_ARGUMENT,
                "z must be numeric for the 3-D form",
            )
        })?;
        (z, 3usize)
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
    add_text_annotation_for_axes(
        target.0,
        target.1,
        glam::Vec3::new(x as f32, y as f32, z as f32),
        &text,
        style,
    )
    .map_err(|err| map_text_internal(map_figure_error(BUILTIN_NAME, err)))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::builtins::plotting::get::get_builtin;
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
        assert_eq!(tensor.data, vec![1.0, 2.0, 0.0]);
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

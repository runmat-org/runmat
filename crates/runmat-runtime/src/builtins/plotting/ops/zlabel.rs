use runmat_builtins::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor, BuiltinOutputMode,
    BuiltinParamArity, BuiltinParamDescriptor, BuiltinParamType, BuiltinSignatureDescriptor,
};
use runmat_builtins::{
    BuiltinExtensionDescriptor, BuiltinExtensionMode, BuiltinIntegerBackendRule,
    BuiltinIntegerCapabilityDescriptor, BuiltinIntegerComputationDomain,
    BuiltinIntegerInputAvailability, BuiltinIntegerInputCapability, BuiltinIntegerOutputClassRule,
    BuiltinIntegerOverflowRule, BuiltinIntegerOverloadKind, BuiltinIntegerScalarDoubleRule,
};
use runmat_macros::runtime_builtin;
use runmat_value::Value;

use super::op_common::{map_figure_error, parse_numeric_text_command};
use super::state::set_zlabel_for_axes;
use crate::builtins::plotting::type_resolvers::handle_scalar_type;

const ZLABEL_INTEGER_AXES_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "zlabel-integer-axes-handle",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "Allow a typed-integer alias for an encoded axes handle",
    error_identifier: Some("RunMat:compatibility:ZlabelIntegerAxesHandleExtension"),
};
pub const ZLABEL_EXTENSIONS: [BuiltinExtensionDescriptor; 1] = [ZLABEL_INTEGER_AXES_EXTENSION];
const ZLABEL_INTEGER_TEXT_INPUT: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "txt",
        classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes:
            "A scalar integer label is formatted from its exact signed or unsigned decimal value.",
    }];
const ZLABEL_INTEGER_FONT_INPUT: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "FontSize",
        classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "All eight integer classes are documented for positive FontSize values.",
    }];
const ZLABEL_INTEGER_AXES_INPUT: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "ax",
        classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "Typed-integer aliases for encoded axes handles are separately gated.",
    }];
pub const ZLABEL_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 3] = [
    BuiltinIntegerCapabilityDescriptor {
        form: "h = zlabel(integer_txt)",
        inputs: &ZLABEL_INTEGER_TEXT_INPUT,
        computation_domain: BuiltinIntegerComputationDomain::Structural,
        output_class: BuiltinIntegerOutputClassRule::Double,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::HostOnly,
        overload: BuiltinIntegerOverloadKind::ScalarOnly,
        notes: "The exact decimal spelling becomes the stored String property.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "h = zlabel(..., 'FontSize', integer_size)",
        inputs: &ZLABEL_INTEGER_FONT_INPUT,
        computation_domain: BuiltinIntegerComputationDomain::FloatingPoint,
        output_class: BuiltinIntegerOutputClassRule::Double,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::HostOnly,
        overload: BuiltinIntegerOverloadKind::StructuralParameter,
        notes: "The validated positive size crosses the graphics scalar boundary.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "h = zlabel(integer_ax, txt, ...)",
        inputs: &ZLABEL_INTEGER_AXES_INPUT,
        computation_domain: BuiltinIntegerComputationDomain::Structural,
        output_class: BuiltinIntegerOutputClassRule::Double,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::HostOnly,
        overload: BuiltinIntegerOverloadKind::StructuralParameter,
        notes: "Strict mode rejects before graphics state access.",
    },
];

const ZLABEL_OUTPUT_HANDLE: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "h",
    ty: BuiltinParamType::NumericScalar,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Handle to the created/updated zlabel object.",
}];

const ZLABEL_INPUTS_TEXT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "txt",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Label text (string/char/cellstr-like multiline forms).",
}];

const ZLABEL_INPUTS_AX_TEXT: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "ax",
        ty: BuiltinParamType::AxesHandle,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Target axes handle.",
    },
    BuiltinParamDescriptor {
        name: "txt",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Label text (string/char/cellstr-like multiline forms).",
    },
];

const ZLABEL_INPUTS_TEXT_PROPS: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "txt",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Label text (string/char/cellstr-like multiline forms).",
    },
    BuiltinParamDescriptor {
        name: "props",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Variadic,
        default: None,
        description: "Property/value pairs (Color, FontSize, FontWeight, etc.).",
    },
];

const ZLABEL_INPUTS_AX_TEXT_PROPS: [BuiltinParamDescriptor; 3] = [
    BuiltinParamDescriptor {
        name: "ax",
        ty: BuiltinParamType::AxesHandle,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Target axes handle.",
    },
    BuiltinParamDescriptor {
        name: "txt",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Label text (string/char/cellstr-like multiline forms).",
    },
    BuiltinParamDescriptor {
        name: "props",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Variadic,
        default: None,
        description: "Property/value pairs (Color, FontSize, FontWeight, etc.).",
    },
];

const ZLABEL_SIGNATURES: [BuiltinSignatureDescriptor; 4] = [
    BuiltinSignatureDescriptor {
        label: "h = zlabel(txt)",
        inputs: &ZLABEL_INPUTS_TEXT,
        outputs: &ZLABEL_OUTPUT_HANDLE,
    },
    BuiltinSignatureDescriptor {
        label: "h = zlabel(ax, txt)",
        inputs: &ZLABEL_INPUTS_AX_TEXT,
        outputs: &ZLABEL_OUTPUT_HANDLE,
    },
    BuiltinSignatureDescriptor {
        label: "h = zlabel(txt, Name, Value, ...)",
        inputs: &ZLABEL_INPUTS_TEXT_PROPS,
        outputs: &ZLABEL_OUTPUT_HANDLE,
    },
    BuiltinSignatureDescriptor {
        label: "h = zlabel(ax, txt, Name, Value, ...)",
        inputs: &ZLABEL_INPUTS_AX_TEXT_PROPS,
        outputs: &ZLABEL_OUTPUT_HANDLE,
    },
];

const ZLABEL_ERROR_INVALID_ARGUMENT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.ZLABEL.INVALID_ARGUMENT",
    identifier: Some("RunMat:zlabel:InvalidArgument"),
    when: "Axes handle, text payload, or property/value arguments are invalid.",
    message: "zlabel: invalid argument",
};

const ZLABEL_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.ZLABEL.INTERNAL",
    identifier: Some("RunMat:zlabel:Internal"),
    when: "Internal plotting state update fails.",
    message: "zlabel: internal operation failed",
};

const ZLABEL_ERRORS: [BuiltinErrorDescriptor; 2] =
    [ZLABEL_ERROR_INVALID_ARGUMENT, ZLABEL_ERROR_INTERNAL];

pub const ZLABEL_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &ZLABEL_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ZLABEL_ERRORS,
};

#[runtime_builtin(
    name = "zlabel",
    category = "plotting",
    summary = "Set z-axis label text for current or specified axes.",
    keywords = "zlabel,plotting",
    suppress_auto_output = true,
    type_resolver(handle_scalar_type),
    descriptor(crate::builtins::plotting::zlabel::ZLABEL_DESCRIPTOR),
    extensions(crate::builtins::plotting::zlabel::ZLABEL_EXTENSIONS),
    integer_capabilities(crate::builtins::plotting::zlabel::ZLABEL_INTEGER_CAPABILITIES),
    builtin_path = "crate::builtins::plotting::zlabel"
)]
pub fn zlabel_builtin(args: Vec<Value>) -> crate::BuiltinResult<f64> {
    if args.len() >= 2
        && args.len().is_multiple_of(2)
        && args
            .first()
            .is_some_and(crate::builtins::common::validation::value_has_native_integer_class)
    {
        crate::compatibility::ensure_builtin_extension_enabled(
            &ZLABEL_INTEGER_AXES_EXTENSION,
            "zlabel",
        )?;
    }
    let command = parse_numeric_text_command("zlabel", &args)?;
    set_zlabel_for_axes(
        command.target.0,
        command.target.1,
        &command.text,
        command.style,
    )
    .map_err(|err| map_figure_error("zlabel", err))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::builtins::plotting::get::get_builtin;
    use crate::builtins::plotting::set::set_builtin;
    use crate::builtins::plotting::state::{decode_plot_object_handle, PlotObjectKind};
    use crate::builtins::plotting::tests::{ensure_plot_test_env, lock_plot_registry};
    use crate::builtins::plotting::{
        clear_figure, clone_figure, configure_subplot, current_figure_handle,
        reset_hold_state_for_run,
    };
    use runmat_value::{CellArray, StringArray};

    #[test]
    fn zlabel_descriptor_signatures_cover_core_forms() {
        let labels: Vec<&str> = ZLABEL_DESCRIPTOR
            .signatures
            .iter()
            .map(|sig| sig.label)
            .collect();
        assert!(labels.contains(&"h = zlabel(txt)"));
        assert!(labels.contains(&"h = zlabel(ax, txt)"));
        assert!(labels.contains(&"h = zlabel(txt, Name, Value, ...)"));
    }

    #[test]
    fn zlabel_sets_axes_local_metadata_and_returns_handle() {
        let _guard = lock_plot_registry();
        ensure_plot_test_env();
        reset_hold_state_for_run();
        let _ = clear_figure(None);

        let handle = zlabel_builtin(vec![Value::String("Height".into())]).unwrap();
        assert_eq!(
            decode_plot_object_handle(handle).unwrap().2,
            PlotObjectKind::ZLabel
        );
        let fig = clone_figure(current_figure_handle()).unwrap();
        assert_eq!(
            fig.axes_metadata(0).and_then(|m| m.z_label.as_deref()),
            Some("Height")
        );
    }

    #[test]
    fn zlabel_supports_axes_target_multiline_and_properties() {
        let _guard = lock_plot_registry();
        ensure_plot_test_env();
        reset_hold_state_for_run();
        let _ = clear_figure(None);
        configure_subplot(1, 2, 1).unwrap();
        let ax = Value::Num(crate::builtins::plotting::state::encode_axes_handle(
            current_figure_handle(),
            1,
        ));

        let handle = zlabel_builtin(vec![
            ax,
            Value::StringArray(StringArray {
                data: vec!["Height".into(), "(m)".into()],
                shape: vec![1, 2],
                rows: 1,
                cols: 2,
            }),
            Value::String("FontWeight".into()),
            Value::String("bold".into()),
        ])
        .unwrap();

        let fig = clone_figure(current_figure_handle()).unwrap();
        let meta = fig.axes_metadata(1).unwrap();
        assert_eq!(meta.z_label.as_deref(), Some("Height\n(m)"));
        assert_eq!(meta.z_label_style.font_weight.as_deref(), Some("bold"));

        let string = get_builtin(vec![Value::Num(handle), Value::String("String".into())]).unwrap();
        assert!(matches!(string, Value::StringArray(_)));
        set_builtin(vec![
            Value::Num(handle),
            Value::String("String".into()),
            Value::Cell(
                CellArray::new(
                    vec![Value::String("Depth".into()), Value::String("(km)".into())],
                    1,
                    2,
                )
                .unwrap(),
            ),
        ])
        .unwrap();
        let fig = clone_figure(current_figure_handle()).unwrap();
        assert_eq!(
            fig.axes_metadata(1).unwrap().z_label.as_deref(),
            Some("Depth\n(km)")
        );
    }
}

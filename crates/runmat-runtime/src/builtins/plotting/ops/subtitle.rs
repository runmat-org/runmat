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

use super::op_common::{map_figure_error, parse_text_command};
use super::state::set_figure_subtitle_for_axes;
use crate::builtins::plotting::type_resolvers::handle_scalar_type;

pub const SUBTITLE_INTEGER_AXES_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "subtitle-integer-axes-handle",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "Allow typed-integer aliases for encoded axes handles",
        error_identifier: Some("RunMat:compatibility:SubtitleIntegerAxesHandleExtension"),
    };
pub const SUBTITLE_EXTENSIONS: [BuiltinExtensionDescriptor; 1] = [SUBTITLE_INTEGER_AXES_EXTENSION];
const SUBTITLE_INTEGER_FONT_SIZE: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "FontSize",
        classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "The compatibility target documents every native integer class for the FontSize property.",
    }];
const SUBTITLE_INTEGER_AXES: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "ax",
        classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "RunMat graphics handles are internally encoded numeric scalars; typed-integer aliases are separately gated.",
    }];
pub const SUBTITLE_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 2] = [
    BuiltinIntegerCapabilityDescriptor {
        form: "h = subtitle(..., 'FontSize', integer_size)",
        inputs: &SUBTITLE_INTEGER_FONT_SIZE,
        computation_domain: BuiltinIntegerComputationDomain::FloatingPoint,
        output_class: BuiltinIntegerOutputClassRule::Double,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::HostOnly,
        overload: BuiltinIntegerOverloadKind::StructuralParameter,
        notes: "The documented property crosses the graphics font-size boundary after exact scalar classification; the returned text handle is a host double scalar.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "h = subtitle(integer_ax, txt, ...)",
        inputs: &SUBTITLE_INTEGER_AXES,
        computation_domain: BuiltinIntegerComputationDomain::Structural,
        output_class: BuiltinIntegerOutputClassRule::Double,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::HostOnly,
        overload: BuiltinIntegerOverloadKind::StructuralParameter,
        notes: "RunMat mode parses the gated scalar exactly before resolving the encoded axes handle; strict mode rejects before graphics state access.",
    },
];

const SUBTITLE_OUTPUT_HANDLE: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "h",
    ty: BuiltinParamType::NumericScalar,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Handle to the created/updated subtitle object.",
}];

const SUBTITLE_INPUTS_TEXT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "txt",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Subtitle text (string/char/cellstr-like multiline forms).",
}];

const SUBTITLE_INPUTS_AX_TEXT: [BuiltinParamDescriptor; 2] = [
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
        description: "Subtitle text (string/char/cellstr-like multiline forms).",
    },
];

const SUBTITLE_INPUTS_TEXT_PROPS: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "txt",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Subtitle text (string/char/cellstr-like multiline forms).",
    },
    BuiltinParamDescriptor {
        name: "props",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Variadic,
        default: None,
        description: "Property/value pairs (Color, FontSize, FontWeight, etc.).",
    },
];

const SUBTITLE_INPUTS_AX_TEXT_PROPS: [BuiltinParamDescriptor; 3] = [
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
        description: "Subtitle text (string/char/cellstr-like multiline forms).",
    },
    BuiltinParamDescriptor {
        name: "props",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Variadic,
        default: None,
        description: "Property/value pairs (Color, FontSize, FontWeight, etc.).",
    },
];

const SUBTITLE_SIGNATURES: [BuiltinSignatureDescriptor; 4] = [
    BuiltinSignatureDescriptor {
        label: "h = subtitle(txt)",
        inputs: &SUBTITLE_INPUTS_TEXT,
        outputs: &SUBTITLE_OUTPUT_HANDLE,
    },
    BuiltinSignatureDescriptor {
        label: "h = subtitle(ax, txt)",
        inputs: &SUBTITLE_INPUTS_AX_TEXT,
        outputs: &SUBTITLE_OUTPUT_HANDLE,
    },
    BuiltinSignatureDescriptor {
        label: "h = subtitle(txt, Name, Value, ...)",
        inputs: &SUBTITLE_INPUTS_TEXT_PROPS,
        outputs: &SUBTITLE_OUTPUT_HANDLE,
    },
    BuiltinSignatureDescriptor {
        label: "h = subtitle(ax, txt, Name, Value, ...)",
        inputs: &SUBTITLE_INPUTS_AX_TEXT_PROPS,
        outputs: &SUBTITLE_OUTPUT_HANDLE,
    },
];

const SUBTITLE_ERROR_INVALID_ARGUMENT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.SUBTITLE.INVALID_ARGUMENT",
    identifier: Some("RunMat:subtitle:InvalidArgument"),
    when: "Axes handle, text payload, or property/value arguments are invalid.",
    message: "subtitle: invalid argument",
};

const SUBTITLE_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.SUBTITLE.INTERNAL",
    identifier: Some("RunMat:subtitle:Internal"),
    when: "Internal plotting state update fails.",
    message: "subtitle: internal operation failed",
};

const SUBTITLE_ERRORS: [BuiltinErrorDescriptor; 2] =
    [SUBTITLE_ERROR_INVALID_ARGUMENT, SUBTITLE_ERROR_INTERNAL];

pub const SUBTITLE_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &SUBTITLE_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &SUBTITLE_ERRORS,
};

#[runtime_builtin(
    name = "subtitle",
    category = "plotting",
    summary = "Set subtitle text for the current or specified axes.",
    keywords = "subtitle,plotting,title",
    suppress_auto_output = true,
    type_resolver(handle_scalar_type),
    descriptor(crate::builtins::plotting::subtitle::SUBTITLE_DESCRIPTOR),
    extensions(crate::builtins::plotting::subtitle::SUBTITLE_EXTENSIONS),
    integer_capabilities(crate::builtins::plotting::subtitle::SUBTITLE_INTEGER_CAPABILITIES),
    builtin_path = "crate::builtins::plotting::subtitle"
)]
pub fn subtitle_builtin(args: Vec<Value>) -> crate::BuiltinResult<f64> {
    if args.len() >= 2
        && args
            .first()
            .is_some_and(crate::builtins::common::validation::value_has_native_integer_class)
    {
        crate::compatibility::ensure_builtin_extension_enabled(
            &SUBTITLE_INTEGER_AXES_EXTENSION,
            "subtitle",
        )?;
    }
    let command = parse_text_command("subtitle", &args)?;
    set_figure_subtitle_for_axes(
        command.target.0,
        command.target.1,
        &command.text,
        command.style,
    )
    .map_err(|err| map_figure_error("subtitle", err))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::builtins::plotting::get::get_builtin;
    use crate::builtins::plotting::state::PlotObjectKind;
    use crate::builtins::plotting::tests::{ensure_plot_test_env, lock_plot_registry};
    use crate::builtins::plotting::{
        clear_figure, clone_figure, current_figure_handle, reset_hold_state_for_run,
    };
    use runmat_value::IntValue;
    use runmat_value::{CellArray, Value};

    fn setup_plot_tests() -> crate::builtins::plotting::state::PlotTestLockGuard {
        let guard = lock_plot_registry();
        ensure_plot_test_env();
        reset_hold_state_for_run();
        let _ = clear_figure(None);
        guard
    }

    #[test]
    fn subtitle_descriptor_signatures_cover_core_forms() {
        let labels: Vec<&str> = SUBTITLE_DESCRIPTOR
            .signatures
            .iter()
            .map(|sig| sig.label)
            .collect();
        assert!(labels.contains(&"h = subtitle(txt)"));
        assert!(labels.contains(&"h = subtitle(ax, txt)"));
        assert!(labels.contains(&"h = subtitle(txt, Name, Value, ...)"));
        assert!(labels.contains(&"h = subtitle(ax, txt, Name, Value, ...)"));
    }

    #[test]
    fn subtitle_returns_text_handle_and_updates_active_axes() {
        let _guard = setup_plot_tests();
        let handle = subtitle_builtin(vec![Value::String("Slope = 2".into())]).unwrap();
        let (figure, axes, kind) =
            crate::builtins::plotting::state::decode_plot_object_handle(handle).unwrap();
        assert_eq!(figure, current_figure_handle());
        assert_eq!(axes, 0);
        assert_eq!(kind, PlotObjectKind::Subtitle);

        let fig = clone_figure(figure).unwrap();
        assert_eq!(
            fig.axes_metadata(0).and_then(|m| m.subtitle.as_deref()),
            Some("Slope = 2")
        );
        assert!(fig.has_any_titles());
    }

    #[test]
    fn subtitle_accepts_axes_target_multiline_text_and_properties() {
        let _guard = setup_plot_tests();
        let ax = crate::builtins::plotting::subplot::subplot_builtin(
            Value::Num(1.0),
            Value::Num(2.0),
            Value::Num(2.0),
        )
        .unwrap();
        let lines = Value::Cell(
            CellArray::new(
                vec![Value::String("left".into()), Value::String("right".into())],
                1,
                2,
            )
            .unwrap(),
        );
        subtitle_builtin(vec![
            Value::Num(ax),
            lines,
            Value::String("FontSize".into()),
            Value::Num(13.0),
            Value::String("FontWeight".into()),
            Value::String("bold".into()),
            Value::String("Visible".into()),
            Value::Bool(false),
        ])
        .unwrap();

        let fig = clone_figure(current_figure_handle()).unwrap();
        let meta = fig.axes_metadata(1).unwrap();
        assert_eq!(meta.subtitle.as_deref(), Some("left\nright"));
        assert_eq!(meta.subtitle_style.font_size, Some(13.0));
        assert_eq!(meta.subtitle_style.font_weight.as_deref(), Some("bold"));
        assert!(!meta.subtitle_style.visible);
    }

    #[test]
    fn get_axes_and_subtitle_handle_expose_subtitle() {
        let _guard = setup_plot_tests();
        let handle = subtitle_builtin(vec![Value::String("Details".into())]).unwrap();
        let ax = crate::builtins::plotting::gca::gca_builtin(Vec::new()).unwrap();

        let subtitle_prop =
            get_builtin(vec![ax.clone(), Value::String("Subtitle".into())]).unwrap();
        assert_eq!(subtitle_prop, Value::Num(handle));

        let string_prop =
            get_builtin(vec![Value::Num(handle), Value::String("String".into())]).unwrap();
        assert_eq!(string_prop, Value::String("Details".into()));
    }

    #[test]
    fn subtitle_rejects_bad_properties() {
        let _guard = setup_plot_tests();
        let err = subtitle_builtin(vec![
            Value::String("Oops".into()),
            Value::String("FontSize".into()),
            Value::String("large".into()),
        ])
        .unwrap_err();
        assert!(err.message.contains("FontSize must be numeric"));
    }

    #[test]
    fn subtitle_accepts_documented_integer_font_size() {
        let _guard = setup_plot_tests();
        let handle = subtitle_builtin(vec![
            Value::String("Integer font size".into()),
            Value::String("FontSize".into()),
            Value::Int(IntValue::U16(14)),
        ])
        .expect("integer FontSize");
        let fig = clone_figure(current_figure_handle()).expect("figure");
        let (_, axes, _) =
            crate::builtins::plotting::state::decode_plot_object_handle(handle).expect("handle");
        assert_eq!(
            fig.axes_metadata(axes)
                .expect("axes")
                .subtitle_style
                .font_size,
            Some(14.0)
        );
    }

    #[test]
    fn subtitle_typed_integer_axes_alias_is_explicitly_gated() {
        let _guard = setup_plot_tests();
        let axes_handle = crate::builtins::plotting::subplot::subplot_builtin(
            Value::Num(1.0),
            Value::Num(1.0),
            Value::Num(1.0),
        )
        .expect("axes handle") as u64;
        let subtitle_before = clone_figure(current_figure_handle()).and_then(|figure| {
            figure
                .axes_metadata(0)
                .and_then(|metadata| metadata.subtitle.clone())
        });
        let strict = crate::compatibility::push_runmat_extensions_enabled(false);
        let error = subtitle_builtin(vec![
            Value::Int(IntValue::U64(axes_handle)),
            Value::String("Blocked".into()),
        ])
        .expect_err("strict mode rejects typed integer axes aliases");
        assert_eq!(
            error.identifier(),
            SUBTITLE_INTEGER_AXES_EXTENSION.error_identifier
        );
        let subtitle_after = clone_figure(current_figure_handle()).and_then(|figure| {
            figure
                .axes_metadata(0)
                .and_then(|metadata| metadata.subtitle.clone())
        });
        assert_eq!(subtitle_after, subtitle_before);
        drop(strict);

        let _extensions = crate::compatibility::push_runmat_extensions_enabled(true);
        let handle = subtitle_builtin(vec![
            Value::Int(IntValue::U64(axes_handle)),
            Value::String("Allowed".into()),
        ])
        .expect("extension mode accepts a typed integer axes alias");
        assert!(handle.is_finite());
    }
}

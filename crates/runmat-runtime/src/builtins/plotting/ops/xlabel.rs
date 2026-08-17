use runmat_builtins::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor, BuiltinExtensionDescriptor,
    BuiltinExtensionMode, BuiltinIntegerBackendRule, BuiltinIntegerCapabilityDescriptor,
    BuiltinIntegerComputationDomain, BuiltinIntegerInputAvailability,
    BuiltinIntegerInputCapability, BuiltinIntegerOutputClassRule, BuiltinIntegerOverflowRule,
    BuiltinIntegerOverloadKind, BuiltinIntegerScalarDoubleRule, BuiltinOutputMode,
    BuiltinParamArity, BuiltinParamDescriptor, BuiltinParamType, BuiltinSignatureDescriptor, Value,
};
use runmat_macros::runtime_builtin;

use super::op_common::{map_figure_error, parse_numeric_text_command};
use super::state::set_xlabel_for_axes;
use crate::builtins::plotting::type_resolvers::handle_scalar_type;

pub const XLABEL_INTEGER_AXES_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "xlabel-integer-axes-handle",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "Allow typed-integer aliases for encoded axes handles",
    error_identifier: Some("RunMat:compatibility:XlabelIntegerAxesHandleExtension"),
};
pub const XLABEL_EXTENSIONS: [BuiltinExtensionDescriptor; 1] = [XLABEL_INTEGER_AXES_EXTENSION];

const XLABEL_INTEGER_TEXT_INPUT: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "txt",
        classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes:
            "A scalar integer label is formatted from its exact signed or unsigned decimal value.",
    }];
const XLABEL_INTEGER_FONT_SIZE_INPUT: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "FontSize",
        classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "All eight integer classes are documented for FontSize; accepted values are positive finite practical renderer sizes.",
    }];
const XLABEL_INTEGER_AXES_INPUT: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "ax",
        classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "Typed-integer aliases for RunMat's encoded axes handles are separately gated.",
    }];
pub const XLABEL_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 3] = [
    BuiltinIntegerCapabilityDescriptor { form: "h = xlabel(integer_txt)", inputs: &XLABEL_INTEGER_TEXT_INPUT, computation_domain: BuiltinIntegerComputationDomain::Structural, output_class: BuiltinIntegerOutputClassRule::Double, overflow: BuiltinIntegerOverflowRule::NotApplicable, backend: BuiltinIntegerBackendRule::HostOnly, overload: BuiltinIntegerOverloadKind::ScalarOnly, notes: "The exact decimal spelling becomes the stored String property; the graphics handle is returned as a double scalar." },
    BuiltinIntegerCapabilityDescriptor { form: "h = xlabel(..., 'FontSize', integer_size)", inputs: &XLABEL_INTEGER_FONT_SIZE_INPUT, computation_domain: BuiltinIntegerComputationDomain::FloatingPoint, output_class: BuiltinIntegerOutputClassRule::Double, overflow: BuiltinIntegerOverflowRule::Error, backend: BuiltinIntegerBackendRule::HostOnly, overload: BuiltinIntegerOverloadKind::StructuralParameter, notes: "FontSize crosses the graphics scalar boundary after shared positive finite size validation." },
    BuiltinIntegerCapabilityDescriptor { form: "h = xlabel(integer_ax, txt, ...)", inputs: &XLABEL_INTEGER_AXES_INPUT, computation_domain: BuiltinIntegerComputationDomain::Structural, output_class: BuiltinIntegerOutputClassRule::Double, overflow: BuiltinIntegerOverflowRule::Error, backend: BuiltinIntegerBackendRule::HostOnly, overload: BuiltinIntegerOverloadKind::StructuralParameter, notes: "RunMat mode parses the gated scalar before resolving the encoded axes handle; strict mode rejects before graphics state access." },
];

const XLABEL_OUTPUT_HANDLE: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "h",
    ty: BuiltinParamType::NumericScalar,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Handle to the created/updated xlabel object.",
}];

const XLABEL_INPUTS_TEXT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "txt",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Label text (string/char/cellstr-like multiline forms).",
}];

const XLABEL_INPUTS_AX_TEXT: [BuiltinParamDescriptor; 2] = [
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

const XLABEL_INPUTS_TEXT_PROPS: [BuiltinParamDescriptor; 2] = [
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

const XLABEL_INPUTS_AX_TEXT_PROPS: [BuiltinParamDescriptor; 3] = [
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

const XLABEL_SIGNATURES: [BuiltinSignatureDescriptor; 4] = [
    BuiltinSignatureDescriptor {
        label: "h = xlabel(txt)",
        inputs: &XLABEL_INPUTS_TEXT,
        outputs: &XLABEL_OUTPUT_HANDLE,
    },
    BuiltinSignatureDescriptor {
        label: "h = xlabel(ax, txt)",
        inputs: &XLABEL_INPUTS_AX_TEXT,
        outputs: &XLABEL_OUTPUT_HANDLE,
    },
    BuiltinSignatureDescriptor {
        label: "h = xlabel(txt, Name, Value, ...)",
        inputs: &XLABEL_INPUTS_TEXT_PROPS,
        outputs: &XLABEL_OUTPUT_HANDLE,
    },
    BuiltinSignatureDescriptor {
        label: "h = xlabel(ax, txt, Name, Value, ...)",
        inputs: &XLABEL_INPUTS_AX_TEXT_PROPS,
        outputs: &XLABEL_OUTPUT_HANDLE,
    },
];

const XLABEL_ERROR_INVALID_ARGUMENT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.XLABEL.INVALID_ARGUMENT",
    identifier: Some("RunMat:xlabel:InvalidArgument"),
    when: "Axes handle, text payload, or property/value arguments are invalid.",
    message: "xlabel: invalid argument",
};

const XLABEL_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.XLABEL.INTERNAL",
    identifier: Some("RunMat:xlabel:Internal"),
    when: "Internal plotting state update fails.",
    message: "xlabel: internal operation failed",
};

const XLABEL_ERRORS: [BuiltinErrorDescriptor; 2] =
    [XLABEL_ERROR_INVALID_ARGUMENT, XLABEL_ERROR_INTERNAL];

pub const XLABEL_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &XLABEL_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &XLABEL_ERRORS,
};

#[runtime_builtin(
    name = "xlabel",
    category = "plotting",
    summary = "Set the current axes x-axis label.",
    keywords = "xlabel,plotting",
    suppress_auto_output = true,
    type_resolver(handle_scalar_type),
    descriptor(crate::builtins::plotting::xlabel::XLABEL_DESCRIPTOR),
    extensions(crate::builtins::plotting::xlabel::XLABEL_EXTENSIONS),
    integer_capabilities(crate::builtins::plotting::xlabel::XLABEL_INTEGER_CAPABILITIES),
    builtin_path = "crate::builtins::plotting::xlabel"
)]
pub fn xlabel_builtin(args: Vec<Value>) -> crate::BuiltinResult<f64> {
    if args.len() >= 2
        && args.len().is_multiple_of(2)
        && args
            .first()
            .is_some_and(crate::builtins::common::validation::value_has_native_integer_class)
    {
        crate::compatibility::ensure_builtin_extension_enabled(
            &XLABEL_INTEGER_AXES_EXTENSION,
            "xlabel",
        )?;
    }
    let command = parse_numeric_text_command("xlabel", &args)?;
    set_xlabel_for_axes(
        command.target.0,
        command.target.1,
        &command.text,
        command.style,
    )
    .map_err(|err| map_figure_error("xlabel", err))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::builtins::plotting::state::PlotTestLockGuard;
    use crate::builtins::plotting::state::{decode_plot_object_handle, PlotObjectKind};
    use crate::builtins::plotting::tests::{ensure_plot_test_env, lock_plot_registry};
    use crate::builtins::plotting::ylabel::ylabel_builtin;
    use crate::builtins::plotting::{
        clear_figure, clone_figure, current_figure_handle, reset_hold_state_for_run,
    };
    use runmat_builtins::{CellArray, IntValue, IntegerStorage, StringArray};
    use runmat_plot::plots::Figure;

    fn setup_plot_tests() -> PlotTestLockGuard {
        let guard = lock_plot_registry();
        ensure_plot_test_env();
        reset_hold_state_for_run();
        let _ = clear_figure(None);
        guard
    }

    #[test]
    fn xlabel_descriptor_signatures_cover_core_forms() {
        let labels: Vec<&str> = XLABEL_DESCRIPTOR
            .signatures
            .iter()
            .map(|sig| sig.label)
            .collect();
        assert!(labels.contains(&"h = xlabel(txt)"));
        assert!(labels.contains(&"h = xlabel(ax, txt)"));
        assert!(labels.contains(&"h = xlabel(txt, Name, Value, ...)"));
    }

    #[test]
    fn x_and_y_labels_update_active_axes_metadata() {
        let _guard = setup_plot_tests();
        let xh = xlabel_builtin(vec![Value::String("Time".into())]).unwrap();
        let yh = ylabel_builtin(vec![
            Value::String("Amplitude".into()),
            Value::String("Color".into()),
            Value::String("g".into()),
        ])
        .unwrap();

        assert_eq!(
            decode_plot_object_handle(xh).unwrap().2,
            PlotObjectKind::XLabel
        );
        assert_eq!(
            decode_plot_object_handle(yh).unwrap().2,
            PlotObjectKind::YLabel
        );

        let fig = clone_figure(current_figure_handle()).unwrap();
        let meta = fig.axes_metadata(0).unwrap();
        assert_eq!(meta.x_label.as_deref(), Some("Time"));
        assert_eq!(meta.y_label.as_deref(), Some("Amplitude"));
        assert!(meta.y_label_style.color.is_some());
    }

    #[test]
    fn xlabel_and_ylabel_reject_invalid_property_values() {
        let _guard = setup_plot_tests();

        let err = xlabel_builtin(vec![
            Value::String("Time".into()),
            Value::String("Bogus".into()),
            Value::Num(1.0),
        ])
        .unwrap_err();
        assert!(err.message.contains("unsupported property"));

        let err = ylabel_builtin(vec![
            Value::String("Amp".into()),
            Value::String("Interpreter".into()),
            Value::Num(5.0),
        ])
        .unwrap_err();
        assert!(err.message.contains("Interpreter must be a string"));
    }

    #[test]
    fn xlabel_formats_native_integer_scalars_exactly() {
        let _guard = setup_plot_tests();
        xlabel_builtin(vec![
            Value::Tensor(
                runmat_builtins::Tensor::new_integer(
                    IntegerStorage::U64(vec![u64::MAX]),
                    vec![1, 1],
                )
                .expect("scalar integer label"),
            ),
            Value::String("FontSize".into()),
            Value::Int(IntValue::U8(14)),
        ])
        .expect("integer label");
        let fig = clone_figure(current_figure_handle()).unwrap();
        assert_eq!(
            fig.axes_metadata(0).unwrap().x_label.as_deref(),
            Some("18446744073709551615")
        );
        assert_eq!(
            fig.axes_metadata(0).unwrap().x_label_style.font_size,
            Some(14.0)
        );
    }

    #[test]
    fn xlabel_typed_integer_axes_alias_is_gated() {
        let _guard = setup_plot_tests();
        let axes_handle = crate::builtins::plotting::subplot::subplot_builtin(
            Value::Num(1.0),
            Value::Num(1.0),
            Value::Num(1.0),
        )
        .expect("axes handle") as u64;
        let strict = crate::compatibility::push_runmat_extensions_enabled(false);
        let error = xlabel_builtin(vec![
            Value::Int(IntValue::U64(axes_handle)),
            Value::String("Blocked".into()),
        ])
        .expect_err("strict mode rejects typed integer axes aliases");
        assert_eq!(
            error.identifier(),
            XLABEL_INTEGER_AXES_EXTENSION.error_identifier
        );
        drop(strict);
    }

    #[test]
    fn xlabel_and_ylabel_support_explicit_axes_targets_and_multiline_text() {
        let _guard = setup_plot_tests();
        let mut figure = Figure::new();
        figure.set_subplot_grid(1, 2);
        let figure = crate::builtins::plotting::state::import_figure(figure);
        let ax = Value::Num(crate::builtins::plotting::state::encode_axes_handle(
            figure, 1,
        ));

        let xhandle = xlabel_builtin(vec![
            ax.clone(),
            Value::StringArray(StringArray {
                data: vec!["Time".into(), "(s)".into()],
                shape: vec![1, 2],
                rows: 1,
                cols: 2,
            }),
            Value::String("FontWeight".into()),
            Value::String("bold".into()),
        ])
        .unwrap();
        let yhandle = ylabel_builtin(vec![
            ax,
            Value::Cell(
                CellArray::new(
                    vec![
                        Value::String("Amplitude".into()),
                        Value::String("(V)".into()),
                    ],
                    1,
                    2,
                )
                .unwrap(),
            ),
            Value::String("FontAngle".into()),
            Value::String("italic".into()),
        ])
        .unwrap();

        assert_eq!(decode_plot_object_handle(xhandle).unwrap().1, 1);
        assert_eq!(decode_plot_object_handle(yhandle).unwrap().1, 1);

        let fig = clone_figure(figure).unwrap();
        let meta = fig.axes_metadata(1).unwrap();
        assert_eq!(meta.x_label.as_deref(), Some("Time\n(s)"));
        assert_eq!(meta.y_label.as_deref(), Some("Amplitude\n(V)"));
        assert_eq!(meta.x_label_style.font_weight.as_deref(), Some("bold"));
        assert_eq!(meta.y_label_style.font_angle.as_deref(), Some("italic"));
    }
}

use runmat_builtins::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor, BuiltinOutputMode,
    BuiltinParamArity, BuiltinParamDescriptor, BuiltinParamType, BuiltinSignatureDescriptor, Value,
};
use runmat_macros::runtime_builtin;

use crate::builtins::plotting::type_resolvers::handle_scalar_type;

const SUPTITLE_OUTPUT_HANDLE: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "h",
    ty: BuiltinParamType::NumericScalar,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Handle to the created/updated super-title object.",
}];

const SUPTITLE_INPUTS_TEXT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "txt",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Super-title text (string/char/cellstr-like multiline forms; numeric scalars also accepted).",
}];

const SUPTITLE_INPUTS_FIG_TEXT: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "fig",
        ty: BuiltinParamType::NumericScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Target figure handle.",
    },
    BuiltinParamDescriptor {
        name: "txt",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Super-title text (string/char/cellstr-like multiline forms; numeric scalars also accepted).",
    },
];

const SUPTITLE_INPUTS_TEXT_PROPS: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "txt",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Super-title text (string/char/cellstr-like multiline forms; numeric scalars also accepted).",
    },
    BuiltinParamDescriptor {
        name: "props",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Variadic,
        default: None,
        description: "Property/value pairs (Color, FontSize, FontWeight, etc.).",
    },
];

const SUPTITLE_INPUTS_FIG_TEXT_PROPS: [BuiltinParamDescriptor; 3] = [
    BuiltinParamDescriptor {
        name: "fig",
        ty: BuiltinParamType::NumericScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Target figure handle.",
    },
    BuiltinParamDescriptor {
        name: "txt",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Super-title text (string/char/cellstr-like multiline forms; numeric scalars also accepted).",
    },
    BuiltinParamDescriptor {
        name: "props",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Variadic,
        default: None,
        description: "Property/value pairs (Color, FontSize, FontWeight, etc.).",
    },
];

const SUPTITLE_SIGNATURES: [BuiltinSignatureDescriptor; 4] = [
    BuiltinSignatureDescriptor {
        label: "h = suptitle(txt)",
        inputs: &SUPTITLE_INPUTS_TEXT,
        outputs: &SUPTITLE_OUTPUT_HANDLE,
    },
    BuiltinSignatureDescriptor {
        label: "h = suptitle(fig, txt)",
        inputs: &SUPTITLE_INPUTS_FIG_TEXT,
        outputs: &SUPTITLE_OUTPUT_HANDLE,
    },
    BuiltinSignatureDescriptor {
        label: "h = suptitle(txt, Name, Value, ...)",
        inputs: &SUPTITLE_INPUTS_TEXT_PROPS,
        outputs: &SUPTITLE_OUTPUT_HANDLE,
    },
    BuiltinSignatureDescriptor {
        label: "h = suptitle(fig, txt, Name, Value, ...)",
        inputs: &SUPTITLE_INPUTS_FIG_TEXT_PROPS,
        outputs: &SUPTITLE_OUTPUT_HANDLE,
    },
];

const SUPTITLE_ERROR_INVALID_ARGUMENT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.SUPTITLE.INVALID_ARGUMENT",
    identifier: Some("RunMat:suptitle:InvalidArgument"),
    when: "Figure handle, text payload, or property/value arguments are invalid.",
    message: "suptitle: invalid argument",
};

const SUPTITLE_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.SUPTITLE.INTERNAL",
    identifier: Some("RunMat:suptitle:Internal"),
    when: "Internal plotting state update fails.",
    message: "suptitle: internal operation failed",
};

const SUPTITLE_ERRORS: [BuiltinErrorDescriptor; 2] =
    [SUPTITLE_ERROR_INVALID_ARGUMENT, SUPTITLE_ERROR_INTERNAL];

pub const SUPTITLE_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &SUPTITLE_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &SUPTITLE_ERRORS,
};

#[runtime_builtin(
    name = "suptitle",
    category = "plotting",
    summary = "Set a centered figure-level title (legacy alias of `sgtitle`).",
    keywords = "suptitle,sgtitle,subplot,title,plotting",
    suppress_auto_output = true,
    type_resolver(handle_scalar_type),
    descriptor(crate::builtins::plotting::suptitle::SUPTITLE_DESCRIPTOR),
    builtin_path = "crate::builtins::plotting::suptitle"
)]
pub fn suptitle_builtin(args: Vec<Value>) -> crate::BuiltinResult<f64> {
    super::sgtitle::sgtitle_impl("suptitle", args)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::builtins::plotting::get::get_builtin;
    use crate::builtins::plotting::set::set_builtin;
    use crate::builtins::plotting::state::{decode_plot_object_handle, PlotObjectKind};
    use crate::builtins::plotting::tests::{ensure_plot_test_env, lock_plot_registry};
    use crate::builtins::plotting::{
        clear_figure, clone_figure, current_figure_handle, figure::figure_builtin,
        reset_hold_state_for_run,
    };

    fn setup() -> crate::builtins::plotting::state::PlotTestLockGuard {
        let guard = lock_plot_registry();
        ensure_plot_test_env();
        reset_hold_state_for_run();
        let _ = clear_figure(None);
        guard
    }

    #[test]
    fn suptitle_descriptor_signatures_cover_core_forms() {
        let labels: Vec<&str> = SUPTITLE_DESCRIPTOR
            .signatures
            .iter()
            .map(|sig| sig.label)
            .collect();
        assert!(labels.contains(&"h = suptitle(txt)"));
        assert!(labels.contains(&"h = suptitle(fig, txt)"));
        assert!(labels.contains(&"h = suptitle(txt, Name, Value, ...)"));
    }

    #[test]
    fn suptitle_returns_handle_and_updates_current_figure() {
        let _guard = setup();
        let handle = suptitle_builtin(vec![Value::String("Overview".into())]).unwrap();
        let (figure, axes, kind) = decode_plot_object_handle(handle).unwrap();
        assert_eq!(figure, current_figure_handle());
        assert_eq!(axes, 0);
        assert_eq!(kind, PlotObjectKind::SuperTitle);

        let fig = clone_figure(figure).unwrap();
        assert_eq!(fig.sg_title.as_deref(), Some("Overview"));
    }

    #[test]
    fn suptitle_accepts_explicit_figure_target() {
        let _guard = setup();
        let fig = figure_builtin(vec![Value::Num(421.0)]).unwrap();
        suptitle_builtin(vec![Value::Num(fig), Value::String("Target Figure".into())]).unwrap();

        let figure = clone_figure(crate::builtins::plotting::FigureHandle::from(421)).unwrap();
        assert_eq!(figure.sg_title.as_deref(), Some("Target Figure"));
    }

    #[test]
    fn suptitle_applies_text_style_pairs() {
        let _guard = setup();
        let fig = figure_builtin(vec![Value::Num(422.0)]).unwrap();
        suptitle_builtin(vec![
            Value::Num(fig),
            Value::String("Styled".into()),
            Value::String("FontSize".into()),
            Value::Num(20.0),
            Value::String("FontWeight".into()),
            Value::String("bold".into()),
            Value::String("Color".into()),
            Value::String("red".into()),
            Value::String("Interpreter".into()),
            Value::String("none".into()),
            Value::String("Visible".into()),
            Value::Bool(false),
        ])
        .unwrap();

        let figure = clone_figure(crate::builtins::plotting::FigureHandle::from(422)).unwrap();
        assert_eq!(figure.sg_title.as_deref(), Some("Styled"));
        assert_eq!(figure.sg_title_style.font_size, Some(20.0));
        assert_eq!(figure.sg_title_style.font_weight.as_deref(), Some("bold"));
        assert!(figure.sg_title_style.color.is_some());
        assert_eq!(figure.sg_title_style.interpreter.as_deref(), Some("none"));
        assert!(!figure.sg_title_style.visible);
    }

    #[test]
    fn suptitle_handle_supports_get_and_set() {
        let _guard = setup();
        let handle = suptitle_builtin(vec![Value::String("Initial".into())]).unwrap();

        let ty = get_builtin(vec![Value::Num(handle), Value::String("Type".into())]).unwrap();
        assert_eq!(ty, Value::String("text".into()));

        set_builtin(vec![
            Value::Num(handle),
            Value::String("String".into()),
            Value::String("Updated".into()),
            Value::String("FontWeight".into()),
            Value::String("bold".into()),
        ])
        .unwrap();

        let string = get_builtin(vec![Value::Num(handle), Value::String("String".into())]).unwrap();
        assert_eq!(string, Value::String("Updated".into()));
        let weight =
            get_builtin(vec![Value::Num(handle), Value::String("FontWeight".into())]).unwrap();
        assert_eq!(weight, Value::String("bold".into()));
    }

    #[test]
    fn suptitle_reports_errors_with_suptitle_context() {
        let _guard = setup();
        let err = suptitle_builtin(vec![]).unwrap_err();
        assert!(err.message.contains("suptitle"));
        assert!(!err.message.contains("sgtitle"));

        let err = suptitle_builtin(vec![
            Value::String("Top".into()),
            Value::String("Bogus".into()),
            Value::Num(1.0),
        ])
        .unwrap_err();
        assert!(err.message.contains("suptitle"));
        assert!(!err.message.contains("sgtitle"));
    }
}

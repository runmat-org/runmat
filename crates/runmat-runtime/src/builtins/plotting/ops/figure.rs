//! MATLAB-compatible `figure` builtin for selecting/creating plotting windows.

use runmat_builtins::Value;
use runmat_builtins::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor, BuiltinOutputMode,
    BuiltinParamArity, BuiltinParamDescriptor, BuiltinParamType, BuiltinSignatureDescriptor,
};
use runmat_macros::runtime_builtin;

use super::op_common::handles::parse_optional_figure_handle;
use super::properties::{set_properties, validate_figure_property_name, PlotHandle};
use super::state::{new_figure_handle, select_figure};
use crate::builtins::plotting::plotting_error;
use crate::builtins::plotting::type_resolvers::handle_scalar_type;

const FIGURE_OUTPUT_HANDLE: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "fig",
    ty: BuiltinParamType::NumericScalar,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Figure handle.",
}];

const FIGURE_INPUTS_NONE: [BuiltinParamDescriptor; 0] = [];

const FIGURE_INPUTS_HANDLE: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "h",
    ty: BuiltinParamType::NumericScalar,
    arity: BuiltinParamArity::Optional,
    default: Some("\"next\""),
    description: "Figure handle or 'next' to create/select the next figure.",
}];

const FIGURE_INPUTS_PAIRS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "properties",
    ty: BuiltinParamType::PropertyName,
    arity: BuiltinParamArity::Variadic,
    default: None,
    description:
        "Figure property/value pairs such as 'Name', 'NumberTitle', 'Visible', or 'Color'.",
}];

const FIGURE_INPUTS_HANDLE_PAIRS: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "h",
        ty: BuiltinParamType::NumericScalar,
        arity: BuiltinParamArity::Optional,
        default: Some("\"next\""),
        description: "Figure handle or 'next' to create/select the next figure.",
    },
    BuiltinParamDescriptor {
        name: "properties",
        ty: BuiltinParamType::PropertyName,
        arity: BuiltinParamArity::Variadic,
        default: None,
        description: "Figure property/value pairs to apply after selecting or creating the figure.",
    },
];

const FIGURE_SIGNATURES: [BuiltinSignatureDescriptor; 4] = [
    BuiltinSignatureDescriptor {
        label: "fig = figure()",
        inputs: &FIGURE_INPUTS_NONE,
        outputs: &FIGURE_OUTPUT_HANDLE,
    },
    BuiltinSignatureDescriptor {
        label: "fig = figure(h)",
        inputs: &FIGURE_INPUTS_HANDLE,
        outputs: &FIGURE_OUTPUT_HANDLE,
    },
    BuiltinSignatureDescriptor {
        label: "fig = figure(property, value, ...)",
        inputs: &FIGURE_INPUTS_PAIRS,
        outputs: &FIGURE_OUTPUT_HANDLE,
    },
    BuiltinSignatureDescriptor {
        label: "fig = figure(h, property, value, ...)",
        inputs: &FIGURE_INPUTS_HANDLE_PAIRS,
        outputs: &FIGURE_OUTPUT_HANDLE,
    },
];

const FIGURE_ERROR_INVALID_ARGUMENT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.FIGURE.INVALID_ARGUMENT",
    identifier: Some("RunMat:figure:InvalidArgument"),
    when: "Provided figure handle argument is invalid.",
    message: "figure: invalid argument",
};

const FIGURE_ERRORS: [BuiltinErrorDescriptor; 1] = [FIGURE_ERROR_INVALID_ARGUMENT];

pub const FIGURE_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &FIGURE_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &FIGURE_ERRORS,
};

#[runtime_builtin(
    name = "figure",
    category = "plotting",
    summary = "Create or select plotting figures.",
    keywords = "figure,plotting",
    suppress_auto_output = true,
    type_resolver(handle_scalar_type),
    descriptor(crate::builtins::plotting::figure::FIGURE_DESCRIPTOR),
    builtin_path = "crate::builtins::plotting::figure"
)]
pub fn figure_builtin(rest: Vec<Value>) -> crate::BuiltinResult<f64> {
    let (target_info, property_args) = if rest.is_empty() {
        (None, &rest[..])
    } else {
        match parse_optional_figure_target(&rest[0], rest.len())? {
            Some(handle) => (Some(FigureTarget::Existing(handle)), &rest[1..]),
            None if is_next_selector(&rest[0]) => (Some(FigureTarget::New), &rest[1..]),
            None => (Some(FigureTarget::New), &rest[..]),
        }
    };

    // Validate properties before any state modifications
    if !property_args.is_empty() {
        validate_figure_properties(property_args)?;
    }

    // Now that validation passed, create/select the figure
    let handle = match target_info {
        Some(FigureTarget::Existing(h)) => {
            select_figure(h);
            h
        }
        Some(FigureTarget::New) | None => new_figure_handle(),
    };

    // Apply properties after figure creation/selection
    if !property_args.is_empty() {
        set_properties(PlotHandle::Figure(handle), property_args, "figure")?;
    }
    Ok(handle.as_u32() as f64)
}

enum FigureTarget {
    Existing(super::state::FigureHandle),
    New,
}

fn validate_figure_properties(args: &[Value]) -> crate::BuiltinResult<()> {
    if !args.len().is_multiple_of(2) {
        return Err(crate::builtins::plotting::plotting_error(
            "figure",
            "figure: property arguments must be name/value pairs",
        ));
    }
    for pair in args.chunks_exact(2) {
        validate_figure_property_name(&pair[0], "figure")?;
    }
    Ok(())
}

fn parse_optional_figure_target(
    value: &Value,
    arg_count: usize,
) -> crate::BuiltinResult<Option<super::state::FigureHandle>> {
    match parse_optional_figure_handle(value, "figure") {
        Ok(target) => Ok(target),
        Err(_) if starts_property_pairs(value, arg_count) => Ok(None),
        Err(_) if is_text(value) && !arg_count.is_multiple_of(2) => Err(plotting_error(
            "figure",
            "figure: property/value arguments must come in pairs",
        )),
        Err(err) => Err(err),
    }
}

fn starts_property_pairs(value: &Value, arg_count: usize) -> bool {
    is_text(value) && arg_count.is_multiple_of(2)
}

fn is_text(value: &Value) -> bool {
    matches!(value, Value::CharArray(_) | Value::String(_))
}

fn is_next_selector(value: &Value) -> bool {
    match value {
        Value::String(text) => text.trim().eq_ignore_ascii_case("next"),
        Value::CharArray(chars) => chars
            .data
            .iter()
            .collect::<String>()
            .trim()
            .eq_ignore_ascii_case("next"),
        _ => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::builtins::plotting::tests::{ensure_plot_test_env, lock_plot_registry};
    use crate::builtins::plotting::{
        clear_figure, clone_figure, current_figure_handle, reset_hold_state_for_run,
    };

    fn setup() -> crate::builtins::plotting::state::PlotTestLockGuard {
        let guard = lock_plot_registry();
        ensure_plot_test_env();
        reset_hold_state_for_run();
        let _ = clear_figure(None);
        guard
    }

    #[test]
    fn figure_descriptor_signatures_cover_core_forms() {
        let labels: Vec<&str> = FIGURE_DESCRIPTOR
            .signatures
            .iter()
            .map(|sig| sig.label)
            .collect();
        assert!(labels.contains(&"fig = figure()"));
        assert!(labels.contains(&"fig = figure(h)"));
        assert!(labels.contains(&"fig = figure(property, value, ...)"));
        assert!(labels.contains(&"fig = figure(h, property, value, ...)"));
    }

    #[test]
    fn figure_creates_and_selects_handles() {
        let _guard = setup();
        let first = figure_builtin(Vec::new()).unwrap();
        assert!(first > 0.0);
        let selected = figure_builtin(vec![Value::Num(first)]).unwrap();
        assert_eq!(selected, first);
        assert_eq!(current_figure_handle().as_u32() as f64, first);
    }

    #[test]
    fn figure_accepts_property_pairs_without_handle() {
        let _guard = setup();
        let handle = figure_builtin(vec![
            Value::String("Name".into()),
            Value::String("demo".into()),
            Value::String("NumberTitle".into()),
            Value::String("off".into()),
            Value::String("Visible".into()),
            Value::String("off".into()),
            Value::String("Color".into()),
            Value::String("black".into()),
        ])
        .unwrap();
        let figure = clone_figure(crate::builtins::plotting::state::FigureHandle::from(
            handle as u32,
        ))
        .expect("figure should exist");
        assert_eq!(figure.name.as_deref(), Some("demo"));
        assert!(!figure.number_title);
        assert!(!figure.visible);
        assert_eq!(figure.background_color, glam::Vec4::new(0.0, 0.0, 0.0, 1.0));
    }

    #[test]
    fn figure_selects_explicit_handle_and_applies_property_pairs() {
        let _guard = setup();
        let handle = figure_builtin(vec![
            Value::Num(42.0),
            Value::String("Name".into()),
            Value::String("selected".into()),
        ])
        .unwrap();
        assert_eq!(handle, 42.0);
        assert_eq!(current_figure_handle().as_u32(), 42);
        let figure = clone_figure(crate::builtins::plotting::state::FigureHandle::from(42))
            .expect("figure should exist");
        assert_eq!(figure.name.as_deref(), Some("selected"));
    }

    #[test]
    fn figure_next_selector_accepts_property_pairs() {
        let _guard = setup();
        let first = figure_builtin(Vec::new()).unwrap();
        let second = figure_builtin(vec![
            Value::String("next".into()),
            Value::String("Name".into()),
            Value::String("next window".into()),
        ])
        .unwrap();
        assert_ne!(second, first);
        assert_eq!(current_figure_handle().as_u32() as f64, second);
        let figure = clone_figure(crate::builtins::plotting::state::FigureHandle::from(
            second as u32,
        ))
        .expect("figure should exist");
        assert_eq!(figure.name.as_deref(), Some("next window"));
    }

    #[test]
    fn figure_rejects_dangling_property_name() {
        let _guard = setup();
        let err = figure_builtin(vec![Value::String("Name".into())])
            .expect_err("dangling property should fail");
        assert!(err
            .message()
            .contains("property/value arguments must come in pairs"));
    }

    #[test]
    fn figure_rejects_invalid_color_name() {
        let _guard = setup();
        let err = figure_builtin(vec![
            Value::String("Color".into()),
            Value::String("banana".into()),
        ])
        .expect_err("invalid color should fail");
        assert!(err
            .message()
            .contains("unsupported color specification `banana`"));
    }

    #[test]
    fn figure_rejects_oversized_numeric_handle() {
        let _guard = setup();
        let err = figure_builtin(vec![
            Value::Num(u32::MAX as f64 + 1.0),
            Value::String("Name".into()),
            Value::String("too large".into()),
        ])
        .expect_err("oversized handle should fail");
        assert!(err.message().contains("figure handle is too large"));
    }
}

//! RunMat `figure` builtin for selecting and creating plotting windows.

use runmat_builtins::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor, BuiltinExtensionDescriptor,
    BuiltinExtensionMode, BuiltinIntegerBackendRule, BuiltinIntegerCapabilityDescriptor,
    BuiltinIntegerComputationDomain, BuiltinIntegerInputAvailability,
    BuiltinIntegerInputCapability, BuiltinIntegerOutputClassRule, BuiltinIntegerOverflowRule,
    BuiltinIntegerOverloadKind, BuiltinIntegerScalarDoubleRule, BuiltinOutputMode,
    BuiltinParamArity, BuiltinParamDescriptor, BuiltinParamType, BuiltinSignatureDescriptor,
    NumericDType, Value,
};
use runmat_macros::runtime_builtin;

use super::op_common::handles::parse_optional_figure_handle;
use super::properties::{set_properties, validate_figure_property_value, PlotHandle};
use super::state::{new_figure_handle, select_figure};
use crate::builtins::plotting::plotting_error;
use crate::builtins::plotting::type_resolvers::handle_scalar_type;

const FIGURE_OUTPUT_HANDLE: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "fig",
    ty: BuiltinParamType::NumericScalar,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "RunMat f64 graphics-handle encoding (not a MATLAB Figure object).",
}];

const FIGURE_INPUTS_NONE: [BuiltinParamDescriptor; 0] = [];

const FIGURE_INPUTS_HANDLE: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "h",
    ty: BuiltinParamType::NumericScalar,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "RunMat numeric figure identifier.",
}];

const FIGURE_INPUTS_PAIRS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "properties",
    ty: BuiltinParamType::PropertyName,
    arity: BuiltinParamArity::Variadic,
    default: None,
    description:
        "Figure property/value pairs such as 'Name', 'NumberTitle', 'Visible', 'Position', or 'Color'.",
}];

const FIGURE_INPUTS_HANDLE_PAIRS: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "h",
        ty: BuiltinParamType::NumericScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "RunMat numeric figure identifier.",
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

pub const FIGURE_INTEGER_TARGET_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "figure-integer-target",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "figure with a typed integer numeric target is a RunMat extension",
        error_identifier: Some("RunMat:compatibility:FigureIntegerTargetExtension"),
    };

pub const FIGURE_SINGLE_TARGET_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "figure-single-target",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "figure with a single-precision numeric target is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:FigureSingleTargetExtension"),
};

pub const FIGURE_NEXT_SELECTOR_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "figure-next-selector",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "figure('next') is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:FigureNextSelectorExtension"),
};

pub const FIGURE_INTEGER_PROPERTY_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "figure-integer-property-value",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "figure with typed integer property values is a RunMat extension",
        error_identifier: Some("RunMat:compatibility:FigureIntegerPropertyValueExtension"),
    };

pub const FIGURE_EXTENSIONS: [BuiltinExtensionDescriptor; 4] = [
    FIGURE_INTEGER_TARGET_EXTENSION,
    FIGURE_SINGLE_TARGET_EXTENSION,
    FIGURE_NEXT_SELECTOR_EXTENSION,
    FIGURE_INTEGER_PROPERTY_EXTENSION,
];

const FIGURE_INTEGER_TARGET_INPUTS: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "h",
        classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "A host scalar from any built-in integer class is an exact RunMat numeric figure identifier and is independently compatibility gated. Logical and resident values are rejected.",
    }];

const FIGURE_INTEGER_POSITION_INPUTS: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "Position",
        classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "A four-element host integer vector is a RunMat-only geometry property value and must be exactly representable at the f64 figure-state boundary.",
    }];

const FIGURE_INTEGER_COLOR_INPUTS: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "Color",
        classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "A three-element host integer RGB vector is a RunMat-only property value. The normal [0,1] color constraint means admitted integer components are exactly zero or one.",
    }];

const FIGURE_INTEGER_SWITCH_INPUTS: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "state",
        classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "RunMat-only NumberTitle and Visible integer scalars use an exact zero test; zero is off and every nonzero value is on without binary64 conversion.",
    }];

const FIGURE_INTEGER_OBJECT_PROPERTY_INPUTS: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "graphics_alias",
        classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "CurrentAxes and SgTitle typed integers are RunMat-only aliases for the current f64 graphics registry and require exact binary64 representation before object lookup.",
    }];

pub const FIGURE_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 6] = [
    BuiltinIntegerCapabilityDescriptor {
        form: "fig = figure(integer_h, ...)",
        inputs: &FIGURE_INTEGER_TARGET_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::Structural,
        output_class: BuiltinIntegerOutputClassRule::NotApplicable,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::HostOnly,
        overload: BuiltinIntegerOverloadKind::ScalarOnly,
        notes: "RunMat reads the authoritative host integer directly and requires the identifier to be positive and representable as u32. The returned f64 is RunMat's current opaque numeric graphics-handle encoding; MATLAB returns a Figure object, so this is a general graphics representation gap rather than numeric MATLAB equivalence.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "fig = figure(..., 'Position', integer_position, ...)",
        inputs: &FIGURE_INTEGER_POSITION_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::FloatingPoint,
        output_class: BuiltinIntegerOutputClassRule::NotApplicable,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::HostOnly,
        overload: BuiltinIntegerOverloadKind::StructuralParameter,
        notes: "This independently gated RunMat extension reads authoritative integer storage, validates the four-element vector and exact f64 conversion before figure creation or selection, then stores host-double window geometry.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "fig = figure(..., 'Color', integer_rgb, ...)",
        inputs: &FIGURE_INTEGER_COLOR_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::FloatingPoint,
        output_class: BuiltinIntegerOutputClassRule::NotApplicable,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::HostOnly,
        overload: BuiltinIntegerOverloadKind::StructuralParameter,
        notes: "This independently gated RunMat extension reads authoritative integer RGB storage before validation; the [0,1] range admits only exact zero/one components and renderer state is f32.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "fig = figure(..., 'NumberTitle', integer_state, ...)",
        inputs: &FIGURE_INTEGER_SWITCH_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::Predicate,
        output_class: BuiltinIntegerOutputClassRule::NotApplicable,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::HostOnly,
        overload: BuiltinIntegerOverloadKind::ScalarOnly,
        notes: "This independently gated RunMat extension evaluates authoritative scalar integer zero/nonzero state without floating conversion before graphics mutation.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "fig = figure(..., 'Visible', integer_state, ...)",
        inputs: &FIGURE_INTEGER_SWITCH_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::Predicate,
        output_class: BuiltinIntegerOutputClassRule::NotApplicable,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::HostOnly,
        overload: BuiltinIntegerOverloadKind::ScalarOnly,
        notes: "This independently gated RunMat extension evaluates authoritative scalar integer zero/nonzero state without floating conversion before graphics mutation.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "fig = figure(..., 'CurrentAxes'|'SgTitle', integer_graphics_alias, ...)",
        inputs: &FIGURE_INTEGER_OBJECT_PROPERTY_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::Structural,
        output_class: BuiltinIntegerOutputClassRule::NotApplicable,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::HostOnly,
        overload: BuiltinIntegerOverloadKind::ScalarOnly,
        notes: "These independently gated RunMat aliases require an exactly representable host integer before lookup in RunMat's f64 graphics-object registry; they do not make numeric aliases MATLAB graphics objects.",
    },
];

#[runtime_builtin(
    name = "figure",
    category = "plotting",
    summary = "Create or select plotting figures.",
    keywords = "figure,plotting",
    suppress_auto_output = true,
    type_resolver(handle_scalar_type),
    descriptor(crate::builtins::plotting::figure::FIGURE_DESCRIPTOR),
    extensions(crate::builtins::plotting::figure::FIGURE_EXTENSIONS),
    integer_capabilities(crate::builtins::plotting::figure::FIGURE_INTEGER_CAPABILITIES),
    builtin_path = "crate::builtins::plotting::figure"
)]
pub fn figure_builtin(rest: Vec<Value>) -> crate::BuiltinResult<f64> {
    ensure_figure_target_extensions(&rest)?;
    let (target_info, property_args) = if rest.is_empty() {
        (None, &rest[..])
    } else {
        match parse_optional_figure_target(&rest[0], rest.len())? {
            Some(handle) => (Some(FigureTarget::Existing(handle)), &rest[1..]),
            None if is_next_selector(&rest[0]) => (Some(FigureTarget::New), &rest[1..]),
            None => (Some(FigureTarget::New), &rest[..]),
        }
    };

    ensure_figure_integer_property_extensions(property_args)?;

    // Validate properties before any state modifications
    if !property_args.is_empty() {
        validate_figure_properties(
            property_args,
            target_info.as_ref().and_then(FigureTarget::target_figure),
        )?;
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

fn ensure_figure_target_extensions(args: &[Value]) -> crate::BuiltinResult<()> {
    let Some(target) = args.first() else {
        return Ok(());
    };
    if is_typed_integer_target(target) {
        crate::compatibility::ensure_builtin_extension_enabled(
            &FIGURE_INTEGER_TARGET_EXTENSION,
            "figure",
        )?;
    }
    if is_single_target(target) {
        crate::compatibility::ensure_builtin_extension_enabled(
            &FIGURE_SINGLE_TARGET_EXTENSION,
            "figure",
        )?;
    }
    if is_next_selector(target) {
        crate::compatibility::ensure_builtin_extension_enabled(
            &FIGURE_NEXT_SELECTOR_EXTENSION,
            "figure",
        )?;
    }
    Ok(())
}

fn is_typed_integer_target(value: &Value) -> bool {
    matches!(value, Value::Int(_))
        || matches!(value, Value::Tensor(tensor) if tensor.integer_storage().is_some())
}

fn is_single_target(value: &Value) -> bool {
    matches!(value, Value::Tensor(tensor) if tensor.numeric_dtype() == NumericDType::F32)
}

fn ensure_figure_integer_property_extensions(args: &[Value]) -> crate::BuiltinResult<()> {
    if !args.len().is_multiple_of(2) {
        return Ok(());
    }
    for pair in args.chunks_exact(2) {
        let Some(property) = figure_property_name(&pair[0]) else {
            continue;
        };
        let value = &pair[1];
        if !is_typed_integer_target(value) || !is_supported_integer_property_route(&property) {
            continue;
        }
        crate::compatibility::ensure_builtin_extension_enabled(
            &FIGURE_INTEGER_PROPERTY_EXTENSION,
            "figure",
        )?;
        if matches!(
            property.as_str(),
            "position" | "color" | "currentaxes" | "sgtitle"
        ) {
            ensure_integer_property_exact_f64(value, &property)?;
        }
    }
    Ok(())
}

fn figure_property_name(value: &Value) -> Option<String> {
    let text = match value {
        Value::String(text) => text.clone(),
        Value::CharArray(chars) if chars.rows == 1 => chars.data.iter().collect(),
        Value::StringArray(strings) if strings.data.len() == 1 => strings.data[0].clone(),
        _ => return None,
    };
    Some(text.trim().to_ascii_lowercase())
}

fn is_supported_integer_property_route(property: &str) -> bool {
    matches!(
        property,
        "position" | "color" | "numbertitle" | "visible" | "currentaxes" | "sgtitle"
    )
}

fn ensure_integer_property_exact_f64(value: &Value, property: &str) -> crate::BuiltinResult<()> {
    let exact = crate::builtins::math::trigonometry::cos::integer_is_exact_f64;
    let valid = match value {
        Value::Int(value) => exact(value),
        Value::Tensor(tensor) => tensor
            .integer_storage()
            .is_none_or(|storage| storage.exact_values().iter().all(exact)),
        _ => true,
    };
    if valid {
        Ok(())
    } else {
        Err(plotting_error(
            "figure",
            format!("figure: integer {property} values must be exactly representable as double"),
        ))
    }
}

enum FigureTarget {
    Existing(super::state::FigureHandle),
    New,
}

impl FigureTarget {
    fn target_figure(&self) -> Option<super::state::FigureHandle> {
        match self {
            FigureTarget::Existing(handle) => Some(*handle),
            FigureTarget::New => None,
        }
    }
}

fn validate_figure_properties(
    args: &[Value],
    target_figure: Option<super::state::FigureHandle>,
) -> crate::BuiltinResult<()> {
    if !args.len().is_multiple_of(2) {
        return Err(crate::builtins::plotting::plotting_error(
            "figure",
            "figure: property arguments must be name/value pairs",
        ));
    }
    for pair in args.chunks_exact(2) {
        validate_figure_property_value(&pair[0], &pair[1], target_figure, "figure")?;
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
        clear_figure, clone_figure, current_figure_handle, figure_handles, reset_hold_state_for_run,
    };
    use runmat_builtins::{IntValue, IntegerStorage, Tensor};

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
    fn figure_accepts_position_property_pair() {
        let _guard = setup();
        let position = Tensor::new(vec![100.0, 100.0, 1000.0, 700.0], vec![1, 4]).unwrap();
        let handle = figure_builtin(vec![
            Value::String("Position".into()),
            Value::Tensor(position),
        ])
        .unwrap();
        let figure = clone_figure(crate::builtins::plotting::state::FigureHandle::from(
            handle as u32,
        ))
        .expect("figure should exist");
        assert_eq!(figure.position, [100.0, 100.0, 1000.0, 700.0]);

        let value = crate::builtins::plotting::get::get_builtin(vec![
            Value::Num(handle),
            Value::String("Position".into()),
        ])
        .expect("get position");
        let tensor = Tensor::try_from(&value).expect("position tensor");
        assert_eq!(tensor.materialize_f64(), vec![100.0, 100.0, 1000.0, 700.0]);
    }

    #[test]
    fn figure_accepts_position_column_vector() {
        let _guard = setup();
        let position = Tensor::new(vec![10.0, 20.0, 300.0, 400.0], vec![4, 1]).unwrap();
        let handle = figure_builtin(vec![
            Value::String("Position".into()),
            Value::Tensor(position),
        ])
        .unwrap();
        let figure = clone_figure(crate::builtins::plotting::state::FigureHandle::from(
            handle as u32,
        ))
        .expect("figure should exist");
        assert_eq!(figure.position, [10.0, 20.0, 300.0, 400.0]);
    }

    #[test]
    fn figure_rejects_position_matrix_shape() {
        let _guard = setup();
        let position = Tensor::new(vec![10.0, 20.0, 300.0, 400.0], vec![2, 2]).unwrap();
        let err = figure_builtin(vec![
            Value::String("Position".into()),
            Value::Tensor(position),
        ])
        .expect_err("2-by-2 Position matrix should fail");
        assert!(
            err.to_string()
                .contains("Position must be a 4-element numeric vector"),
            "unexpected error: {err:?}"
        );
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
        let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
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
        let first = figure_builtin(Vec::new()).unwrap();
        let handles_before_error = figure_handles();
        let err = figure_builtin(vec![
            Value::String("Color".into()),
            Value::String("banana".into()),
        ])
        .expect_err("invalid color should fail");
        assert!(err
            .message()
            .contains("unsupported color specification `banana`"));
        assert_eq!(current_figure_handle().as_u32() as f64, first);
        assert_eq!(figure_handles(), handles_before_error);
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

    #[test]
    fn figure_typed_integer_targets_are_exact_runmat_extensions() {
        let _guard = setup();
        let _extensions = crate::compatibility::push_runmat_extensions_enabled(true);
        let values = [
            IntValue::I8(11),
            IntValue::I16(12),
            IntValue::I32(13),
            IntValue::I64(14),
            IntValue::U8(15),
            IntValue::U16(16),
            IntValue::U32(17),
            IntValue::U64(18),
        ];
        for value in values {
            let expected = value.try_to_u64().expect("positive identifier") as f64;
            assert_eq!(figure_builtin(vec![Value::Int(value)]).unwrap(), expected);
        }
    }

    #[test]
    fn figure_integer_target_gate_precedes_graphics_state_mutation() {
        let _guard = setup();
        let first = figure_builtin(Vec::new()).unwrap();
        let handles_before = figure_handles();
        let _strict = crate::compatibility::push_runmat_extensions_enabled(false);
        let err = figure_builtin(vec![Value::Int(IntValue::U64(91))])
            .expect_err("integer target extension");
        assert_eq!(
            err.identifier(),
            FIGURE_INTEGER_TARGET_EXTENSION.error_identifier
        );
        assert_eq!(current_figure_handle().as_u32() as f64, first);
        assert_eq!(figure_handles(), handles_before);
    }

    #[test]
    fn figure_single_target_gate_precedes_graphics_state_mutation() {
        let _guard = setup();
        let first = figure_builtin(Vec::new()).unwrap();
        let handles_before = figure_handles();
        let single = Tensor::new_with_dtype(vec![92.0], vec![1, 1], NumericDType::F32).unwrap();
        let _strict = crate::compatibility::push_runmat_extensions_enabled(false);
        let err = figure_builtin(vec![Value::Tensor(single)]).expect_err("single target extension");
        assert_eq!(
            err.identifier(),
            FIGURE_SINGLE_TARGET_EXTENSION.error_identifier
        );
        assert_eq!(current_figure_handle().as_u32() as f64, first);
        assert_eq!(figure_handles(), handles_before);
    }

    #[test]
    fn figure_next_gate_precedes_graphics_state_mutation() {
        let _guard = setup();
        let first = figure_builtin(Vec::new()).unwrap();
        let handles_before = figure_handles();
        let _strict = crate::compatibility::push_runmat_extensions_enabled(false);
        let err = figure_builtin(vec![Value::String("next".into())])
            .expect_err("next selector extension");
        assert_eq!(
            err.identifier(),
            FIGURE_NEXT_SELECTOR_EXTENSION.error_identifier
        );
        assert_eq!(current_figure_handle().as_u32() as f64, first);
        assert_eq!(figure_handles(), handles_before);
    }

    #[test]
    fn figure_rejects_logical_and_resident_targets_without_state_mutation() {
        let _guard = setup();
        let first = figure_builtin(Vec::new()).unwrap();
        let handles_before = figure_handles();
        assert!(figure_builtin(vec![Value::Bool(true)]).is_err());
        let resident = Value::GpuTensor(runmat_accelerate_api::GpuTensorHandle {
            shape: vec![1, 1],
            device_id: u32::MAX,
            buffer_id: u64::MAX,
        });
        assert!(figure_builtin(vec![resident]).is_err());
        assert_eq!(current_figure_handle().as_u32() as f64, first);
        assert_eq!(figure_handles(), handles_before);
    }

    #[test]
    fn figure_integer_capability_is_structural_and_host_only() {
        let capability = &FIGURE_INTEGER_CAPABILITIES[0];
        assert_eq!(capability.inputs[0].classes.len(), 8);
        assert_eq!(
            capability.computation_domain,
            BuiltinIntegerComputationDomain::Structural
        );
        assert_eq!(capability.backend, BuiltinIntegerBackendRule::HostOnly);
        assert_eq!(
            capability.output_class,
            BuiltinIntegerOutputClassRule::NotApplicable
        );
        assert!(capability.notes.contains("representation gap"));
    }

    #[test]
    fn figure_integer_property_routes_gate_before_graphics_state_mutation() {
        let _guard = setup();
        let first = figure_builtin(Vec::new()).unwrap();
        let handles_before = figure_handles();
        let cases = [
            (
                "Position",
                Value::Tensor(
                    Tensor::new_integer(IntegerStorage::U64(vec![1, 2, 3, 4]), vec![1, 4]).unwrap(),
                ),
            ),
            (
                "Color",
                Value::Tensor(
                    Tensor::new_integer(IntegerStorage::U8(vec![1, 0, 0]), vec![1, 3]).unwrap(),
                ),
            ),
            ("NumberTitle", Value::Int(IntValue::U64(u64::MAX))),
            ("Visible", Value::Int(IntValue::I64(i64::MIN))),
            ("CurrentAxes", Value::Int(IntValue::U32(1))),
            ("SgTitle", Value::Int(IntValue::U32(1))),
        ];
        let _strict = crate::compatibility::push_runmat_extensions_enabled(false);
        for (property, value) in cases {
            let err = figure_builtin(vec![Value::String(property.into()), value])
                .expect_err("integer property extension");
            assert_eq!(
                err.identifier(),
                FIGURE_INTEGER_PROPERTY_EXTENSION.error_identifier,
                "{property}"
            );
            assert_eq!(current_figure_handle().as_u32() as f64, first);
            assert_eq!(figure_handles(), handles_before);
        }
    }

    #[test]
    fn figure_documented_double_and_logical_properties_remain_strict_compatible() {
        let _guard = setup();
        let _strict = crate::compatibility::push_runmat_extensions_enabled(false);
        let position = Tensor::new(vec![10.0, 20.0, 300.0, 400.0], vec![1, 4]).unwrap();
        let color = Tensor::new(vec![0.0, 1.0, 0.0], vec![1, 3]).unwrap();
        let handle = figure_builtin(vec![
            Value::String("Position".into()),
            Value::Tensor(position),
            Value::String("Color".into()),
            Value::Tensor(color),
            Value::String("NumberTitle".into()),
            Value::Bool(false),
            Value::String("Visible".into()),
            Value::Bool(false),
        ])
        .unwrap();
        let figure = clone_figure(crate::builtins::plotting::state::FigureHandle::from(
            handle as u32,
        ))
        .unwrap();
        assert_eq!(figure.position, [10.0, 20.0, 300.0, 400.0]);
        assert_eq!(figure.background_color, glam::Vec4::new(0.0, 1.0, 0.0, 1.0));
        assert!(!figure.number_title);
        assert!(!figure.visible);
    }

    #[test]
    fn figure_wide_integer_floating_property_boundaries_reject_before_mutation() {
        let _guard = setup();
        let first = figure_builtin(Vec::new()).unwrap();
        let handles_before = figure_handles();
        let wide = 9_007_199_254_740_993_u64;
        let cases = [
            (
                "Position",
                Value::Tensor(
                    Tensor::new_integer(IntegerStorage::U64(vec![0, 0, wide, 1]), vec![1, 4])
                        .unwrap(),
                ),
            ),
            (
                "Color",
                Value::Tensor(
                    Tensor::new_integer(IntegerStorage::U64(vec![wide, 0, 0]), vec![1, 3]).unwrap(),
                ),
            ),
            ("CurrentAxes", Value::Int(IntValue::U64(wide))),
            ("SgTitle", Value::Int(IntValue::U64(wide))),
        ];
        let _extensions = crate::compatibility::push_runmat_extensions_enabled(true);
        for (property, value) in cases {
            let err = figure_builtin(vec![Value::String(property.into()), value])
                .expect_err("inexact integer property");
            assert!(
                err.message().contains("exactly representable as double"),
                "{property}: {}",
                err.message()
            );
            assert_eq!(current_figure_handle().as_u32() as f64, first);
            assert_eq!(figure_handles(), handles_before);
        }
    }

    #[test]
    fn figure_integer_position_color_and_switches_use_authoritative_values() {
        let _guard = setup();
        let _extensions = crate::compatibility::push_runmat_extensions_enabled(true);
        let position =
            Tensor::new_integer(IntegerStorage::U64(vec![10, 20, 300, 400]), vec![1, 4]).unwrap();
        let color = Tensor::new_integer(IntegerStorage::U64(vec![1, 0, 0]), vec![1, 3]).unwrap();
        let handle = figure_builtin(vec![
            Value::String("Position".into()),
            Value::Tensor(position),
            Value::String("Color".into()),
            Value::Tensor(color),
            Value::String("NumberTitle".into()),
            Value::Int(IntValue::U64(u64::MAX)),
            Value::String("Visible".into()),
            Value::Int(IntValue::I64(i64::MIN)),
        ])
        .unwrap();
        let figure = clone_figure(crate::builtins::plotting::state::FigureHandle::from(
            handle as u32,
        ))
        .unwrap();
        assert_eq!(figure.position, [10.0, 20.0, 300.0, 400.0]);
        assert_eq!(figure.background_color, glam::Vec4::new(1.0, 0.0, 0.0, 1.0));
        assert!(figure.number_title);
        assert!(figure.visible);
    }

    #[test]
    fn figure_integer_capabilities_classify_every_supported_property_route() {
        assert_eq!(FIGURE_INTEGER_CAPABILITIES.len(), 6);
        for property in [
            "Position",
            "Color",
            "NumberTitle",
            "Visible",
            "CurrentAxes",
            "SgTitle",
        ] {
            let capability = FIGURE_INTEGER_CAPABILITIES
                .iter()
                .find(|capability| capability.form.contains(property))
                .unwrap_or_else(|| panic!("missing {property} capability"));
            assert_eq!(capability.inputs[0].classes.len(), 8);
            assert_eq!(capability.backend, BuiltinIntegerBackendRule::HostOnly);
            assert_eq!(
                capability.inputs[0].availability,
                BuiltinIntegerInputAvailability::RunMatOnly
            );
        }
    }
}

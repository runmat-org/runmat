use runmat_builtins::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor, BuiltinIntegerBackendRule,
    BuiltinIntegerCapabilityDescriptor, BuiltinIntegerComputationDomain,
    BuiltinIntegerInputAvailability, BuiltinIntegerInputCapability, BuiltinIntegerOutputClassRule,
    BuiltinIntegerOverflowRule, BuiltinIntegerOverloadKind, BuiltinIntegerScalarDoubleRule,
    BuiltinOutputMode, BuiltinParamArity, BuiltinParamDescriptor, BuiltinParamType,
    BuiltinSignatureDescriptor, Value,
};
use runmat_macros::runtime_builtin;
use runmat_plot::plots::ReferenceLineOrientation;

use crate::builtins::plotting::type_resolvers::handle_scalar_type;
use crate::BuiltinResult;

const BUILTIN_NAME: &str = "yline";

const YLINE_INTEGER_COORDINATE_INPUT: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "y",
        classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "All eight integer classes are documented for reference-line coordinates.",
    }];
const YLINE_INTEGER_COLOR_INPUT: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "Color",
        classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "Integer RGB components are accepted when every component is zero or one.",
    }];
const YLINE_INTEGER_WIDTH_INPUT: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "LineWidth",
        classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "Integer line widths are accepted when positive.",
    }];
const YLINE_INTEGER_VISIBLE_INPUT: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "Visible",
        classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "Numeric visibility values must be exactly zero or one.",
    }];
pub const YLINE_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 4] = [
    BuiltinIntegerCapabilityDescriptor { form: "h = yline(integer_y, ...)", inputs: &YLINE_INTEGER_COORDINATE_INPUT, computation_domain: BuiltinIntegerComputationDomain::FloatingPoint, output_class: BuiltinIntegerOutputClassRule::Double, overflow: BuiltinIntegerOverflowRule::NotApplicable, backend: BuiltinIntegerBackendRule::HostOnly, overload: BuiltinIntegerOverloadKind::FunctionSpecific, notes: "Coordinates cross the graphics floating-point boundary; returned graphics handles are double." },
    BuiltinIntegerCapabilityDescriptor { form: "h = yline(..., 'Color', integer_rgb)", inputs: &YLINE_INTEGER_COLOR_INPUT, computation_domain: BuiltinIntegerComputationDomain::FloatingPoint, output_class: BuiltinIntegerOutputClassRule::Double, overflow: BuiltinIntegerOverflowRule::Error, backend: BuiltinIntegerBackendRule::HostOnly, overload: BuiltinIntegerOverloadKind::StructuralParameter, notes: "Validated RGB components cross the renderer color boundary." },
    BuiltinIntegerCapabilityDescriptor { form: "h = yline(..., 'LineWidth', integer_width)", inputs: &YLINE_INTEGER_WIDTH_INPUT, computation_domain: BuiltinIntegerComputationDomain::FloatingPoint, output_class: BuiltinIntegerOutputClassRule::Double, overflow: BuiltinIntegerOverflowRule::Error, backend: BuiltinIntegerBackendRule::HostOnly, overload: BuiltinIntegerOverloadKind::StructuralParameter, notes: "The validated positive width crosses the renderer scalar boundary." },
    BuiltinIntegerCapabilityDescriptor { form: "h = yline(..., 'Visible', integer_visible)", inputs: &YLINE_INTEGER_VISIBLE_INPUT, computation_domain: BuiltinIntegerComputationDomain::Structural, output_class: BuiltinIntegerOutputClassRule::Double, overflow: BuiltinIntegerOverflowRule::Error, backend: BuiltinIntegerBackendRule::HostOnly, overload: BuiltinIntegerOverloadKind::StructuralParameter, notes: "The integer is classified exactly as zero or one before graphics state is updated." },
];

const YLINE_OUTPUT_HANDLES: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "h",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description:
        "Reference line handle scalar or row vector when multiple coordinates are provided.",
}];

const YLINE_INPUTS_COORD: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "y",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Y-coordinate scalar or numeric vector of horizontal line positions.",
}];

const YLINE_INPUTS_COORD_STYLE: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "y",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Y-coordinate scalar or numeric vector of horizontal line positions.",
    },
    BuiltinParamDescriptor {
        name: "style_or_label",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Optional,
        default: None,
        description: "LineSpec token (for example '--r') or label text.",
    },
];

const YLINE_INPUTS_COORD_STYLE_LABEL: [BuiltinParamDescriptor; 3] = [
    BuiltinParamDescriptor {
        name: "y",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Y-coordinate scalar or numeric vector of horizontal line positions.",
    },
    BuiltinParamDescriptor {
        name: "linespec",
        ty: BuiltinParamType::StyleSpec,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Line style token (for example '--r').",
    },
    BuiltinParamDescriptor {
        name: "label",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Reference line label text.",
    },
];

const YLINE_INPUTS_COORD_PROPS: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "y",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Y-coordinate scalar or numeric vector of horizontal line positions.",
    },
    BuiltinParamDescriptor {
        name: "props",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Variadic,
        default: None,
        description:
            "Style arguments and Name/Value pairs (Color, LineWidth, LineStyle, LabelOrientation, Visible).",
    },
];

const YLINE_SIGNATURES: [BuiltinSignatureDescriptor; 4] = [
    BuiltinSignatureDescriptor {
        label: "h = yline(y)",
        inputs: &YLINE_INPUTS_COORD,
        outputs: &YLINE_OUTPUT_HANDLES,
    },
    BuiltinSignatureDescriptor {
        label: "h = yline(y, styleOrLabel)",
        inputs: &YLINE_INPUTS_COORD_STYLE,
        outputs: &YLINE_OUTPUT_HANDLES,
    },
    BuiltinSignatureDescriptor {
        label: "h = yline(y, LineSpec, label)",
        inputs: &YLINE_INPUTS_COORD_STYLE_LABEL,
        outputs: &YLINE_OUTPUT_HANDLES,
    },
    BuiltinSignatureDescriptor {
        label: "h = yline(y, Name, Value, ...)",
        inputs: &YLINE_INPUTS_COORD_PROPS,
        outputs: &YLINE_OUTPUT_HANDLES,
    },
];

const YLINE_ERROR_INVALID_ARGUMENT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.YLINE.INVALID_ARGUMENT",
    identifier: Some("RunMat:yline:InvalidArgument"),
    when: "Coordinate input, style arguments, or Name/Value pairs are invalid.",
    message: "yline: invalid argument",
};

const YLINE_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.YLINE.INTERNAL",
    identifier: Some("RunMat:yline:Internal"),
    when: "Internal plotting state update fails.",
    message: "yline: internal operation failed",
};

const YLINE_ERRORS: [BuiltinErrorDescriptor; 2] =
    [YLINE_ERROR_INVALID_ARGUMENT, YLINE_ERROR_INTERNAL];

pub const YLINE_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &YLINE_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &YLINE_ERRORS,
};

#[runtime_builtin(
    name = "yline",
    category = "plotting",
    summary = "Draw horizontal reference lines on current or specified axes.",
    keywords = "yline,reference,line,plotting",
    sink = true,
    suppress_auto_output = true,
    type_resolver(handle_scalar_type),
    descriptor(crate::builtins::plotting::yline::YLINE_DESCRIPTOR),
    integer_capabilities(crate::builtins::plotting::yline::YLINE_INTEGER_CAPABILITIES),
    builtin_path = "crate::builtins::plotting::yline"
)]
pub fn yline_builtin(args: Vec<Value>) -> BuiltinResult<Value> {
    super::xline::reference_line_builtin(BUILTIN_NAME, ReferenceLineOrientation::Horizontal, args)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::builtins::plotting::get::get_builtin;
    use crate::builtins::plotting::state::PlotTestLockGuard;
    use crate::builtins::plotting::tests::{ensure_plot_test_env, lock_plot_registry};
    use crate::builtins::plotting::{clear_figure, clone_figure, current_figure_handle};
    use runmat_builtins::Tensor;

    fn setup() -> PlotTestLockGuard {
        let guard = lock_plot_registry();
        ensure_plot_test_env();
        super::super::state::reset_hold_state_for_run();
        let _ = clear_figure(None);
        guard
    }

    #[test]
    fn yline_descriptor_signatures_cover_core_forms() {
        let labels: Vec<&str> = YLINE_DESCRIPTOR
            .signatures
            .iter()
            .map(|sig| sig.label)
            .collect();
        assert!(labels.contains(&"h = yline(y)"));
        assert!(labels.contains(&"h = yline(y, styleOrLabel)"));
        assert!(labels.contains(&"h = yline(y, Name, Value, ...)"));
    }

    #[test]
    fn yline_supports_user_repro() {
        let _guard = setup();
        let handle = yline_builtin(vec![
            Value::Num(0.0),
            Value::String("k".into()),
            Value::String("LineWidth".into()),
            Value::Num(1.0),
        ])
        .unwrap();
        let Value::Num(handle) = handle else {
            panic!("expected scalar handle");
        };
        assert_eq!(
            get_builtin(vec![Value::Num(handle), Value::String("Value".into())]).unwrap(),
            Value::Num(0.0)
        );
        let figure = clone_figure(current_figure_handle()).unwrap();
        assert_eq!(figure.len(), 1);
    }

    #[test]
    fn yline_rejects_nonfinite_coordinates() {
        let _guard = setup();
        let err = yline_builtin(vec![Value::Tensor(
            Tensor::new_2d(vec![0.0, f64::INFINITY], 1, 2).unwrap(),
        )])
        .expect_err("nonfinite coordinates should fail");
        assert!(err.to_string().contains("finite"));
    }
}

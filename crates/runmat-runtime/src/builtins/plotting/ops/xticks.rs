use runmat_builtins::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor, BuiltinExtensionDescriptor,
    BuiltinExtensionMode, BuiltinIntegerBackendRule, BuiltinIntegerCapabilityDescriptor,
    BuiltinIntegerComputationDomain, BuiltinIntegerInputAvailability,
    BuiltinIntegerInputCapability, BuiltinIntegerOutputClassRule, BuiltinIntegerOverflowRule,
    BuiltinIntegerOverloadKind, BuiltinIntegerScalarDoubleRule, BuiltinOutputMode,
    BuiltinParamArity, BuiltinParamDescriptor, BuiltinParamType, BuiltinSignatureDescriptor, Value,
};
use runmat_macros::runtime_builtin;

use super::axis_ticks::{axis_ticks_builtin, TickAxis};
use crate::builtins::plotting::type_resolvers::get_type;

const XTICKS_INTEGER_AXES_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "xticks-integer-axes-handle",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "Allow a typed-integer alias for an encoded axes handle",
    error_identifier: Some("RunMat:compatibility:XticksIntegerAxesHandleExtension"),
};
pub const XTICKS_EXTENSIONS: [BuiltinExtensionDescriptor; 1] = [XTICKS_INTEGER_AXES_EXTENSION];
const XTICKS_INTEGER_INPUT: [BuiltinIntegerInputCapability; 1] = [BuiltinIntegerInputCapability {
    name: "ticks",
    classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES,
    availability: BuiltinIntegerInputAvailability::Documented,
    scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
    notes: "All eight integer classes are documented for increasing tick vectors.",
}];
const XTICKS_INTEGER_AXES_INPUT: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "ax",
        classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "Typed-integer aliases for encoded axes handles are separately gated.",
    }];
pub const XTICKS_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 2] = [
    BuiltinIntegerCapabilityDescriptor { form: "ticks = xticks(integer_ticks)", inputs: &XTICKS_INTEGER_INPUT, computation_domain: BuiltinIntegerComputationDomain::FloatingPoint, output_class: BuiltinIntegerOutputClassRule::Double, overflow: BuiltinIntegerOverflowRule::Error, backend: BuiltinIntegerBackendRule::HostOnly, overload: BuiltinIntegerOverloadKind::FunctionSpecific, notes: "Integer ticks are checked in their native class for strict increase before crossing the graphics coordinate boundary; queried values are double." },
    BuiltinIntegerCapabilityDescriptor { form: "ticks = xticks(integer_ax, ticks)", inputs: &XTICKS_INTEGER_AXES_INPUT, computation_domain: BuiltinIntegerComputationDomain::Structural, output_class: BuiltinIntegerOutputClassRule::Double, overflow: BuiltinIntegerOverflowRule::Error, backend: BuiltinIntegerBackendRule::HostOnly, overload: BuiltinIntegerOverloadKind::StructuralParameter, notes: "Strict mode rejects the encoded-handle alias before graphics state access." },
];

const XTICKS_OUTPUT_TICKS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "ticks",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Current X-axis tick values.",
}];

const XTICKS_OUTPUT_MODE: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "mode",
    ty: BuiltinParamType::StringScalar,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Current X-axis tick mode, 'auto' or 'manual'.",
}];

const XTICKS_INPUTS_NONE: [BuiltinParamDescriptor; 0] = [];

const XTICKS_INPUTS_TICKS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "ticks",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Increasing numeric vector of X-axis tick values; [] removes tick marks.",
}];

const XTICKS_INPUTS_MODE: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "mode",
    ty: BuiltinParamType::StringScalar,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Tick mode string: 'auto', 'manual', or 'mode'.",
}];

const XTICKS_INPUTS_AX_TICKS: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "ax",
        ty: BuiltinParamType::AxesHandle,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Target axes handle.",
    },
    BuiltinParamDescriptor {
        name: "ticks",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Increasing numeric vector of X-axis tick values.",
    },
];

const XTICKS_SIGNATURES: [BuiltinSignatureDescriptor; 4] = [
    BuiltinSignatureDescriptor {
        label: "ticks = xticks()",
        inputs: &XTICKS_INPUTS_NONE,
        outputs: &XTICKS_OUTPUT_TICKS,
    },
    BuiltinSignatureDescriptor {
        label: "ticks = xticks(ticks)",
        inputs: &XTICKS_INPUTS_TICKS,
        outputs: &XTICKS_OUTPUT_TICKS,
    },
    BuiltinSignatureDescriptor {
        label: "mode = xticks(mode)",
        inputs: &XTICKS_INPUTS_MODE,
        outputs: &XTICKS_OUTPUT_MODE,
    },
    BuiltinSignatureDescriptor {
        label: "ticks = xticks(ax, ticks)",
        inputs: &XTICKS_INPUTS_AX_TICKS,
        outputs: &XTICKS_OUTPUT_TICKS,
    },
];

const XTICKS_ERROR_INVALID_ARGUMENT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.XTICKS.INVALID_ARGUMENT",
    identifier: Some("RunMat:xticks:InvalidArgument"),
    when: "Argument count, tick vector, mode, or axes handle is invalid.",
    message: "xticks: invalid argument",
};

const XTICKS_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.XTICKS.INTERNAL",
    identifier: Some("RunMat:xticks:Internal"),
    when: "Internal plotting state update fails.",
    message: "xticks: internal operation failed",
};

const XTICKS_ERRORS: [BuiltinErrorDescriptor; 2] =
    [XTICKS_ERROR_INVALID_ARGUMENT, XTICKS_ERROR_INTERNAL];

pub const XTICKS_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &XTICKS_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &XTICKS_ERRORS,
};

#[runtime_builtin(
    name = "xticks",
    category = "plotting",
    summary = "Query or set X-axis tick values.",
    keywords = "xticks,plotting,axes,ticks",
    suppress_auto_output = true,
    type_resolver(get_type),
    descriptor(crate::builtins::plotting::xticks::XTICKS_DESCRIPTOR),
    extensions(crate::builtins::plotting::xticks::XTICKS_EXTENSIONS),
    integer_capabilities(crate::builtins::plotting::xticks::XTICKS_INTEGER_CAPABILITIES),
    builtin_path = "crate::builtins::plotting::xticks"
)]
pub fn xticks_builtin(args: Vec<Value>) -> crate::BuiltinResult<Value> {
    if args.len() == 2
        && args
            .first()
            .is_some_and(crate::builtins::common::validation::value_has_native_integer_class)
    {
        crate::compatibility::ensure_builtin_extension_enabled(
            &XTICKS_INTEGER_AXES_EXTENSION,
            "xticks",
        )?;
    }
    axis_ticks_builtin("xticks", TickAxis::X, args)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::builtins::plotting::get::get_builtin;
    use crate::builtins::plotting::tests::{ensure_plot_test_env, lock_plot_registry};
    use crate::builtins::plotting::{clear_figure, reset_hold_state_for_run};

    fn tensor(data: Vec<f64>) -> Value {
        let len = data.len();
        Value::Tensor(runmat_builtins::Tensor::new(data, vec![1, len]).expect("x tick row"))
    }

    #[test]
    fn xticks_set_query_mode_and_property_round_trip() {
        let _guard = lock_plot_registry();
        ensure_plot_test_env();
        reset_hold_state_for_run();
        let _ = clear_figure(None);

        let set = xticks_builtin(vec![tensor(vec![0.0, 2.5, 5.0])]).unwrap();
        assert_eq!(
            runmat_builtins::Tensor::try_from(&set)
                .unwrap()
                .materialize_f64(),
            vec![0.0, 2.5, 5.0]
        );
        assert_eq!(
            xticks_builtin(vec![Value::String("mode".into())]).unwrap(),
            Value::String("manual".into())
        );

        let queried = xticks_builtin(Vec::new()).unwrap();
        assert_eq!(
            runmat_builtins::Tensor::try_from(&queried)
                .unwrap()
                .materialize_f64(),
            vec![0.0, 2.5, 5.0]
        );
        let ax = crate::builtins::plotting::gca::gca_builtin(Vec::new()).unwrap();
        let prop = get_builtin(vec![ax, Value::String("XTick".into())]).unwrap();
        assert_eq!(
            runmat_builtins::Tensor::try_from(&prop)
                .unwrap()
                .materialize_f64(),
            vec![0.0, 2.5, 5.0]
        );

        let _ = xticks_builtin(vec![Value::String("auto".into())]).unwrap();
        assert_eq!(
            xticks_builtin(vec![Value::String("mode".into())]).unwrap(),
            Value::String("auto".into())
        );
    }

    #[test]
    fn xticks_axes_target_does_not_change_current_axes() {
        let _guard = lock_plot_registry();
        ensure_plot_test_env();
        reset_hold_state_for_run();
        let _ = clear_figure(None);

        let ax1 = crate::builtins::plotting::subplot::subplot_builtin(
            Value::Num(1.0),
            Value::Num(2.0),
            Value::Num(1.0),
        )
        .unwrap();
        let ax2 = crate::builtins::plotting::subplot::subplot_builtin(
            Value::Num(1.0),
            Value::Num(2.0),
            Value::Num(2.0),
        )
        .unwrap();

        let _ = xticks_builtin(vec![Value::Num(ax1), tensor(vec![0.0, 1.0, 2.0])]).unwrap();
        let current = crate::builtins::plotting::gca::gca_builtin(Vec::new()).unwrap();
        assert_eq!(current, Value::Num(ax2));
    }

    #[test]
    fn xticks_auto_query_uses_plotted_data_bounds() {
        let _guard = lock_plot_registry();
        ensure_plot_test_env();
        reset_hold_state_for_run();
        let _ = clear_figure(None);

        let _ = futures::executor::block_on(crate::builtins::plotting::plot::plot_builtin(vec![
            tensor(vec![10.0, 20.0, 30.0]),
            tensor(vec![1.0, 4.0, 9.0]),
        ]))
        .unwrap();
        let queried = xticks_builtin(Vec::new()).unwrap();
        let ticks = runmat_builtins::Tensor::try_from(&queried)
            .unwrap()
            .materialize_f64();

        assert_eq!(ticks.first().copied(), Some(10.0));
        assert_eq!(ticks.last().copied(), Some(30.0));
    }
}

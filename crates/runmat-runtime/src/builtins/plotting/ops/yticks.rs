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

const YTICKS_INTEGER_AXES_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "yticks-integer-axes-handle",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "Allow a typed-integer alias for an encoded axes handle",
    error_identifier: Some("RunMat:compatibility:YticksIntegerAxesHandleExtension"),
};
pub const YTICKS_EXTENSIONS: [BuiltinExtensionDescriptor; 1] = [YTICKS_INTEGER_AXES_EXTENSION];
const YTICKS_INTEGER_INPUT: [BuiltinIntegerInputCapability; 1] = [BuiltinIntegerInputCapability {
    name: "ticks",
    classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES,
    availability: BuiltinIntegerInputAvailability::Documented,
    scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
    notes: "All eight integer classes are documented for increasing tick vectors.",
}];
const YTICKS_INTEGER_AXES_INPUT: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "ax",
        classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "Typed-integer aliases for encoded axes handles are separately gated.",
    }];
pub const YTICKS_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 2] = [
    BuiltinIntegerCapabilityDescriptor { form: "ticks = yticks(integer_ticks)", inputs: &YTICKS_INTEGER_INPUT, computation_domain: BuiltinIntegerComputationDomain::FloatingPoint, output_class: BuiltinIntegerOutputClassRule::Double, overflow: BuiltinIntegerOverflowRule::Error, backend: BuiltinIntegerBackendRule::HostOnly, overload: BuiltinIntegerOverloadKind::FunctionSpecific, notes: "Integer ticks are checked in their native class for strict increase before crossing the graphics coordinate boundary; queried values are double." },
    BuiltinIntegerCapabilityDescriptor { form: "ticks = yticks(integer_ax, ticks)", inputs: &YTICKS_INTEGER_AXES_INPUT, computation_domain: BuiltinIntegerComputationDomain::Structural, output_class: BuiltinIntegerOutputClassRule::Double, overflow: BuiltinIntegerOverflowRule::Error, backend: BuiltinIntegerBackendRule::HostOnly, overload: BuiltinIntegerOverloadKind::StructuralParameter, notes: "Strict mode rejects the encoded-handle alias before graphics state access." },
];

const YTICKS_OUTPUT_TICKS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "ticks",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Current Y-axis tick values.",
}];

const YTICKS_OUTPUT_MODE: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "mode",
    ty: BuiltinParamType::StringScalar,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Current Y-axis tick mode, 'auto' or 'manual'.",
}];

const YTICKS_INPUTS_NONE: [BuiltinParamDescriptor; 0] = [];

const YTICKS_INPUTS_TICKS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "ticks",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Increasing numeric vector of Y-axis tick values; [] removes tick marks.",
}];

const YTICKS_INPUTS_MODE: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "mode",
    ty: BuiltinParamType::StringScalar,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Tick mode string: 'auto', 'manual', or 'mode'.",
}];

const YTICKS_INPUTS_AX_TICKS: [BuiltinParamDescriptor; 2] = [
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
        description: "Increasing numeric vector of Y-axis tick values.",
    },
];

const YTICKS_SIGNATURES: [BuiltinSignatureDescriptor; 4] = [
    BuiltinSignatureDescriptor {
        label: "ticks = yticks()",
        inputs: &YTICKS_INPUTS_NONE,
        outputs: &YTICKS_OUTPUT_TICKS,
    },
    BuiltinSignatureDescriptor {
        label: "ticks = yticks(ticks)",
        inputs: &YTICKS_INPUTS_TICKS,
        outputs: &YTICKS_OUTPUT_TICKS,
    },
    BuiltinSignatureDescriptor {
        label: "mode = yticks(mode)",
        inputs: &YTICKS_INPUTS_MODE,
        outputs: &YTICKS_OUTPUT_MODE,
    },
    BuiltinSignatureDescriptor {
        label: "ticks = yticks(ax, ticks)",
        inputs: &YTICKS_INPUTS_AX_TICKS,
        outputs: &YTICKS_OUTPUT_TICKS,
    },
];

const YTICKS_ERROR_INVALID_ARGUMENT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.YTICKS.INVALID_ARGUMENT",
    identifier: Some("RunMat:yticks:InvalidArgument"),
    when: "Argument count, tick vector, mode, or axes handle is invalid.",
    message: "yticks: invalid argument",
};

const YTICKS_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.YTICKS.INTERNAL",
    identifier: Some("RunMat:yticks:Internal"),
    when: "Internal plotting state update fails.",
    message: "yticks: internal operation failed",
};

const YTICKS_ERRORS: [BuiltinErrorDescriptor; 2] =
    [YTICKS_ERROR_INVALID_ARGUMENT, YTICKS_ERROR_INTERNAL];

pub const YTICKS_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &YTICKS_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &YTICKS_ERRORS,
};

#[runtime_builtin(
    name = "yticks",
    category = "plotting",
    summary = "Query or set Y-axis tick values.",
    keywords = "yticks,plotting,axes,ticks",
    suppress_auto_output = true,
    type_resolver(get_type),
    descriptor(crate::builtins::plotting::yticks::YTICKS_DESCRIPTOR),
    extensions(crate::builtins::plotting::yticks::YTICKS_EXTENSIONS),
    integer_capabilities(crate::builtins::plotting::yticks::YTICKS_INTEGER_CAPABILITIES),
    builtin_path = "crate::builtins::plotting::yticks"
)]
pub fn yticks_builtin(args: Vec<Value>) -> crate::BuiltinResult<Value> {
    if args.len() == 2
        && args
            .first()
            .is_some_and(crate::builtins::common::validation::value_has_native_integer_class)
    {
        crate::compatibility::ensure_builtin_extension_enabled(
            &YTICKS_INTEGER_AXES_EXTENSION,
            "yticks",
        )?;
    }
    axis_ticks_builtin("yticks", TickAxis::Y, args)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::builtins::plotting::get::get_builtin;
    use crate::builtins::plotting::set::set_builtin;
    use crate::builtins::plotting::tests::{ensure_plot_test_env, lock_plot_registry};
    use crate::builtins::plotting::{clear_figure, reset_hold_state_for_run};

    fn tensor(data: Vec<f64>) -> Value {
        let len = data.len();
        Value::Tensor(runmat_builtins::Tensor::new(data, vec![1, len]).expect("y tick row"))
    }

    #[test]
    fn yticks_supports_properties_and_manual_freeze() {
        let _guard = lock_plot_registry();
        ensure_plot_test_env();
        reset_hold_state_for_run();
        let _ = clear_figure(None);
        let ax = crate::builtins::plotting::gca::gca_builtin(Vec::new()).unwrap();

        let _ = set_builtin(vec![
            ax.clone(),
            Value::String("YTick".into()),
            tensor(vec![1.0, 3.0, 9.0]),
        ])
        .unwrap();
        let queried = yticks_builtin(Vec::new()).unwrap();
        assert_eq!(
            runmat_builtins::Tensor::try_from(&queried)
                .unwrap()
                .materialize_f64(),
            vec![1.0, 3.0, 9.0]
        );
        let mode = get_builtin(vec![ax.clone(), Value::String("YTickMode".into())]).unwrap();
        assert_eq!(mode, Value::String("manual".into()));

        let _ = set_builtin(vec![
            ax,
            Value::String("YTickMode".into()),
            Value::String("auto".into()),
        ])
        .unwrap();
        assert_eq!(
            yticks_builtin(vec![Value::String("mode".into())]).unwrap(),
            Value::String("auto".into())
        );
        let _ = yticks_builtin(vec![Value::String("manual".into())]).unwrap();
        assert_eq!(
            yticks_builtin(vec![Value::String("mode".into())]).unwrap(),
            Value::String("manual".into())
        );
    }
}

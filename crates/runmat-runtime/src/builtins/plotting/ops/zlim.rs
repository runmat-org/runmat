use runmat_builtins::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor, BuiltinOutputMode,
    BuiltinParamArity, BuiltinParamDescriptor, BuiltinParamType, BuiltinSignatureDescriptor,
};
use runmat_builtins::{
    BuiltinIntegerBackendRule, BuiltinIntegerCapabilityDescriptor, BuiltinIntegerComputationDomain,
    BuiltinIntegerInputAvailability, BuiltinIntegerInputCapability, BuiltinIntegerOutputClassRule,
    BuiltinIntegerOverflowRule, BuiltinIntegerOverloadKind, BuiltinIntegerScalarDoubleRule,
};
use runmat_macros::runtime_builtin;
use runmat_value::Value;

use super::op_common::limits::{
    limit_state_error, limit_value, prepare_limit_call, query_limit_method, selected_limit_axes,
    set_limit_method, LimitCommand,
};
use super::state::{
    axes_metadata_snapshot, axis_displayed_limits_snapshot_for_axes,
    axis_z_limits_snapshot_for_axes, set_axis_z_limits_preserving_selection,
};
use crate::builtins::plotting::type_resolvers::get_type;
use runmat_plot::plots::figure::AxisLimitAxis;

const ZLIM_INTEGER_INPUT: [BuiltinIntegerInputCapability; 1] = [BuiltinIntegerInputCapability {
    name: "limits",
    classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES,
    availability: BuiltinIntegerInputAvailability::Documented,
    scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
    notes: "All eight integer classes are documented for the two-element limits vector.",
}];
pub const ZLIM_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 1] = [BuiltinIntegerCapabilityDescriptor { form: "limits = zlim(integer_limits)", inputs: &ZLIM_INTEGER_INPUT, computation_domain: BuiltinIntegerComputationDomain::FloatingPoint, output_class: BuiltinIntegerOutputClassRule::Double, overflow: BuiltinIntegerOverflowRule::Error, backend: BuiltinIntegerBackendRule::HostOnly, overload: BuiltinIntegerOverloadKind::StructuralParameter, notes: "RunMat compares same-class integer endpoints exactly, then validates that they remain distinct in the graphics coordinate domain; queried limits are double." }];

const ZLIM_OUTPUT_LIMITS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "limits",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Two-element vector [zmin zmax] or [NaN NaN] when auto.",
}];

const ZLIM_INPUTS_NONE: [BuiltinParamDescriptor; 0] = [];

const ZLIM_INPUTS_LIMIT_VECTOR: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "limits",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Two-element numeric vector [zmin zmax].",
}];

const ZLIM_INPUTS_MODE: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "mode",
    ty: BuiltinParamType::StringScalar,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Limit mode or method ('auto', 'manual', 'mode', 'method', 'tight', 'padded', or 'tickaligned').",
}];

const ZLIM_SIGNATURES: [BuiltinSignatureDescriptor; 3] = [
    BuiltinSignatureDescriptor {
        label: "limits = zlim()",
        inputs: &ZLIM_INPUTS_NONE,
        outputs: &ZLIM_OUTPUT_LIMITS,
    },
    BuiltinSignatureDescriptor {
        label: "limits = zlim([zmin zmax])",
        inputs: &ZLIM_INPUTS_LIMIT_VECTOR,
        outputs: &ZLIM_OUTPUT_LIMITS,
    },
    BuiltinSignatureDescriptor {
        label: "limits = zlim(mode)",
        inputs: &ZLIM_INPUTS_MODE,
        outputs: &ZLIM_OUTPUT_LIMITS,
    },
];

const ZLIM_ERROR_INVALID_ARGUMENT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.ZLIM.INVALID_ARGUMENT",
    identifier: Some("RunMat:zlim:InvalidArgument"),
    when: "Argument count, mode, or limit vector is invalid.",
    message: "zlim: invalid argument",
};

const ZLIM_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.ZLIM.INTERNAL",
    identifier: Some("RunMat:zlim:Internal"),
    when: "Internal plotting state update fails.",
    message: "zlim: internal operation failed",
};

const ZLIM_ERRORS: [BuiltinErrorDescriptor; 2] = [ZLIM_ERROR_INVALID_ARGUMENT, ZLIM_ERROR_INTERNAL];

pub const ZLIM_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &ZLIM_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ZLIM_ERRORS,
};

#[runtime_builtin(
    name = "zlim",
    category = "plotting",
    summary = "Query or set Z-axis limits.",
    keywords = "zlim,plotting,axes",
    suppress_auto_output = true,
    type_resolver(get_type),
    descriptor(crate::builtins::plotting::zlim::ZLIM_DESCRIPTOR),
    integer_capabilities(crate::builtins::plotting::zlim::ZLIM_INTEGER_CAPABILITIES),
    builtin_path = "crate::builtins::plotting::zlim"
)]
pub fn zlim_builtin(args: Vec<Value>) -> crate::BuiltinResult<Value> {
    let call = prepare_limit_call("zlim", &args, false)?;
    let (handle, index) = selected_limit_axes(call.target);
    match call.command {
        LimitCommand::Query => {
            let limits = axis_displayed_limits_snapshot_for_axes(handle, index, AxisLimitAxis::Z)
                .map_err(|err| limit_state_error("zlim", err))?;
            Ok(limit_value(limits.or(Some((0.0, 1.0)))))
        }
        LimitCommand::QueryMode => {
            let metadata = axes_metadata_snapshot(handle, index)
                .map_err(|err| limit_state_error("zlim", err))?;
            Ok(Value::String(
                if metadata.z_limits.is_some() {
                    "manual"
                } else {
                    "auto"
                }
                .into(),
            ))
        }
        LimitCommand::QueryMethod => query_limit_method("zlim", AxisLimitAxis::Z, call.target),
        LimitCommand::SetMethod(method) => {
            set_limit_method("zlim", AxisLimitAxis::Z, method, call.target)
        }
        LimitCommand::Freeze => {
            let limits = axis_z_limits_snapshot_for_axes(handle, index)
                .map_err(|err| limit_state_error("zlim", err))?
                .unwrap_or((0.0, 1.0));
            set_axis_z_limits_preserving_selection(handle, index, Some(limits))
                .map_err(|err| limit_state_error("zlim", err))?;
            Ok(limit_value(Some(limits)))
        }
        LimitCommand::Set(limits) => {
            set_axis_z_limits_preserving_selection(handle, index, limits)
                .map_err(|err| limit_state_error("zlim", err))?;
            Ok(limit_value(limits))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::builtins::plotting::tests::{ensure_plot_test_env, lock_plot_registry};
    use crate::builtins::plotting::{clear_figure, reset_hold_state_for_run};

    #[test]
    fn zlim_descriptor_signatures_cover_core_forms() {
        let labels: Vec<&str> = ZLIM_DESCRIPTOR
            .signatures
            .iter()
            .map(|sig| sig.label)
            .collect();
        assert!(labels.contains(&"limits = zlim()"));
        assert!(labels.contains(&"limits = zlim([zmin zmax])"));
        assert!(labels.contains(&"limits = zlim(mode)"));
    }

    #[test]
    fn zlim_queries_and_sets_limits() {
        let _guard = lock_plot_registry();
        ensure_plot_test_env();
        reset_hold_state_for_run();
        let _ = clear_figure(None);

        let _ = zlim_builtin(vec![Value::Tensor(
            runmat_value::Tensor::new(vec![-1.0, 1.0], vec![1, 2]).expect("z limits"),
        )])
        .unwrap();
        let queried = zlim_builtin(Vec::new()).unwrap();
        let tensor = runmat_value::Tensor::try_from(&queried).unwrap();
        assert_eq!(tensor.materialize_f64(), vec![-1.0, 1.0]);
    }
}

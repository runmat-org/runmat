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
    axes_metadata_snapshot, axis_display_bounds_snapshot_for_axes,
    axis_displayed_limits_snapshot_for_axes, set_axis_manual_limits_for_axes, set_current_y_limits,
};
use crate::builtins::plotting::type_resolvers::get_type;
use runmat_plot::plots::figure::AxisLimitAxis;

const YLIM_INTEGER_INPUT: [BuiltinIntegerInputCapability; 1] = [BuiltinIntegerInputCapability {
    name: "limits",
    classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES,
    availability: BuiltinIntegerInputAvailability::Documented,
    scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
    notes: "All eight integer classes are documented for the two-element limits vector.",
}];
pub const YLIM_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 1] = [BuiltinIntegerCapabilityDescriptor { form: "limits = ylim(integer_limits)", inputs: &YLIM_INTEGER_INPUT, computation_domain: BuiltinIntegerComputationDomain::FloatingPoint, output_class: BuiltinIntegerOutputClassRule::Double, overflow: BuiltinIntegerOverflowRule::Error, backend: BuiltinIntegerBackendRule::HostOnly, overload: BuiltinIntegerOverloadKind::StructuralParameter, notes: "RunMat compares same-class integer endpoints exactly, then validates that they remain distinct in the graphics coordinate domain; queried limits are double." }];

const YLIM_OUTPUT_LIMITS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "limits",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Two-element vector [ymin ymax] or [NaN NaN] when auto.",
}];

const YLIM_INPUTS_NONE: [BuiltinParamDescriptor; 0] = [];

const YLIM_INPUTS_LIMIT_VECTOR: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "limits",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Two-element numeric vector [ymin ymax].",
}];

const YLIM_INPUTS_MODE: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "mode",
    ty: BuiltinParamType::StringScalar,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Limit mode or method ('auto', 'manual', 'mode', 'method', 'tight', 'padded', or 'tickaligned').",
}];

const YLIM_SIGNATURES: [BuiltinSignatureDescriptor; 3] = [
    BuiltinSignatureDescriptor {
        label: "limits = ylim()",
        inputs: &YLIM_INPUTS_NONE,
        outputs: &YLIM_OUTPUT_LIMITS,
    },
    BuiltinSignatureDescriptor {
        label: "limits = ylim([ymin ymax])",
        inputs: &YLIM_INPUTS_LIMIT_VECTOR,
        outputs: &YLIM_OUTPUT_LIMITS,
    },
    BuiltinSignatureDescriptor {
        label: "limits = ylim(mode)",
        inputs: &YLIM_INPUTS_MODE,
        outputs: &YLIM_OUTPUT_LIMITS,
    },
];

const YLIM_ERROR_INVALID_ARGUMENT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.YLIM.INVALID_ARGUMENT",
    identifier: Some("RunMat:ylim:InvalidArgument"),
    when: "Argument count, mode, or limit vector is invalid.",
    message: "ylim: invalid argument",
};

const YLIM_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.YLIM.INTERNAL",
    identifier: Some("RunMat:ylim:Internal"),
    when: "Internal plotting state update fails.",
    message: "ylim: internal operation failed",
};

const YLIM_ERRORS: [BuiltinErrorDescriptor; 2] = [YLIM_ERROR_INVALID_ARGUMENT, YLIM_ERROR_INTERNAL];

pub const YLIM_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &YLIM_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &YLIM_ERRORS,
};

#[runtime_builtin(
    name = "ylim",
    category = "plotting",
    summary = "Query or set Y-axis limits.",
    keywords = "ylim,plotting,axes",
    suppress_auto_output = true,
    type_resolver(get_type),
    descriptor(crate::builtins::plotting::ylim::YLIM_DESCRIPTOR),
    integer_capabilities(crate::builtins::plotting::ylim::YLIM_INTEGER_CAPABILITIES),
    builtin_path = "crate::builtins::plotting::ylim"
)]
pub fn ylim_builtin(args: Vec<Value>) -> crate::BuiltinResult<Value> {
    let call = prepare_limit_call("ylim", &args, false)?;
    match call.command {
        LimitCommand::Query => {
            let (handle, index) = selected_limit_axes(call.target);
            let limits = axis_displayed_limits_snapshot_for_axes(handle, index, AxisLimitAxis::Y)
                .map_err(|err| limit_state_error("ylim", err))?;
            Ok(limit_value(limits))
        }
        LimitCommand::QueryMode => {
            let (handle, index) = selected_limit_axes(call.target);
            let metadata = axes_metadata_snapshot(handle, index)
                .map_err(|err| limit_state_error("ylim", err))?;
            Ok(Value::String(
                if metadata.y_limits.is_some() {
                    "manual"
                } else {
                    "auto"
                }
                .into(),
            ))
        }
        LimitCommand::QueryMethod => query_limit_method("ylim", AxisLimitAxis::Y, call.target),
        LimitCommand::SetMethod(method) => {
            set_limit_method("ylim", AxisLimitAxis::Y, method, call.target)
        }
        LimitCommand::Freeze => {
            let (handle, index) = selected_limit_axes(call.target);
            let metadata = axes_metadata_snapshot(handle, index)
                .map_err(|err| limit_state_error("ylim", err))?;
            let bounds = axis_display_bounds_snapshot_for_axes(handle, index)
                .map_err(|err| limit_state_error("ylim", err))?;
            let limits = metadata
                .y_limits
                .or_else(|| bounds.map(|bounds| (bounds.2, bounds.3)))
                .unwrap_or((0.0, 1.0));
            set_axis_manual_limits_for_axes(handle, index, metadata.x_limits, Some(limits))
                .map_err(|err| limit_state_error("ylim", err))?;
            Ok(limit_value(Some(limits)))
        }
        LimitCommand::Set(limits) => {
            if let Some((handle, index)) = call.target {
                let metadata = axes_metadata_snapshot(handle, index)
                    .map_err(|err| limit_state_error("ylim", err))?;
                set_axis_manual_limits_for_axes(handle, index, metadata.x_limits, limits)
                    .map_err(|err| limit_state_error("ylim", err))?;
            } else {
                set_current_y_limits(limits);
            }
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
    fn ylim_descriptor_signatures_cover_core_forms() {
        let labels: Vec<&str> = YLIM_DESCRIPTOR
            .signatures
            .iter()
            .map(|sig| sig.label)
            .collect();
        assert!(labels.contains(&"limits = ylim()"));
        assert!(labels.contains(&"limits = ylim([ymin ymax])"));
        assert!(labels.contains(&"limits = ylim(mode)"));
    }

    #[test]
    fn ylim_supports_auto_reset() {
        let _guard = lock_plot_registry();
        ensure_plot_test_env();
        reset_hold_state_for_run();
        let _ = clear_figure(None);

        let _ = ylim_builtin(vec![Value::Tensor(
            runmat_value::Tensor::new(vec![1.0, 5.0], vec![1, 2]).expect("y limits"),
        )])
        .unwrap();
        let manual = ylim_builtin(Vec::new()).expect("manual limits on empty axes");
        assert_eq!(
            runmat_value::Tensor::try_from(&manual)
                .expect("manual limits tensor")
                .materialize_f64(),
            vec![1.0, 5.0]
        );
        let _ = ylim_builtin(vec![Value::String("auto".into())]).unwrap();
        let queried = ylim_builtin(Vec::new()).unwrap();
        let tensor = runmat_value::Tensor::try_from(&queried).unwrap();
        assert!(tensor.materialize_f64()[0].is_nan() && tensor.materialize_f64()[1].is_nan());
    }
}

use runmat_builtins::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor, BuiltinExtensionDescriptor,
    BuiltinExtensionMode, BuiltinIntegerBackendRule, BuiltinIntegerCapabilityDescriptor,
    BuiltinIntegerComputationDomain, BuiltinIntegerInputAvailability,
    BuiltinIntegerInputCapability, BuiltinIntegerOutputClassRule, BuiltinIntegerOverflowRule,
    BuiltinIntegerOverloadKind, BuiltinIntegerScalarDoubleRule, BuiltinOutputMode,
    BuiltinParamArity, BuiltinParamDescriptor, BuiltinParamType, BuiltinSignatureDescriptor,
};
use runmat_macros::runtime_builtin;
use runmat_value::Value;

use super::op_common::limits::{
    limit_state_error, limit_value, prepare_limit_call, selected_limit_axes, LimitCommand,
};
use super::state::{
    axis_displayed_color_limits_for_axes, set_axis_color_limits_preserving_selection,
};
use crate::builtins::plotting::type_resolvers::get_type;

const CLIM_OUTPUT_LIMITS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "limits",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Two-element color limit vector [cmin cmax] or [NaN NaN] when auto.",
}];

const CLIM_INPUTS_NONE: [BuiltinParamDescriptor; 0] = [];

const CLIM_INPUTS_LIMIT_VECTOR: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "limits",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Two-element numeric vector [cmin cmax].",
}];

const CLIM_INPUTS_MODE: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "mode",
    ty: BuiltinParamType::StringScalar,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Limit mode string ('auto', 'tight', or 'manual').",
}];

const CLIM_SIGNATURES: [BuiltinSignatureDescriptor; 3] = [
    BuiltinSignatureDescriptor {
        label: "limits = clim()",
        inputs: &CLIM_INPUTS_NONE,
        outputs: &CLIM_OUTPUT_LIMITS,
    },
    BuiltinSignatureDescriptor {
        label: "limits = clim([cmin cmax])",
        inputs: &CLIM_INPUTS_LIMIT_VECTOR,
        outputs: &CLIM_OUTPUT_LIMITS,
    },
    BuiltinSignatureDescriptor {
        label: "limits = clim(mode)",
        inputs: &CLIM_INPUTS_MODE,
        outputs: &CLIM_OUTPUT_LIMITS,
    },
];

const CLIM_ERROR_INVALID_ARGUMENT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.CLIM.INVALID_ARGUMENT",
    identifier: Some("RunMat:clim:InvalidArgument"),
    when: "Argument count, mode, or limit vector is invalid.",
    message: "clim: invalid argument",
};

const CLIM_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.CLIM.INTERNAL",
    identifier: Some("RunMat:clim:Internal"),
    when: "Internal plotting state update fails.",
    message: "clim: internal operation failed",
};

const CLIM_ERRORS: [BuiltinErrorDescriptor; 2] = [CLIM_ERROR_INVALID_ARGUMENT, CLIM_ERROR_INTERNAL];

pub const CLIM_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &CLIM_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &CLIM_ERRORS,
};

pub(crate) const CLIM_INTEGER_LIMITS_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "clim-integer-limits",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "clim/caxis with typed integer limits is a RunMat extension",
        error_identifier: Some("RunMat:compatibility:ClimIntegerLimitsExtension"),
    };

pub const CLIM_EXTENSIONS: [BuiltinExtensionDescriptor; 1] = [CLIM_INTEGER_LIMITS_EXTENSION];

const CLIM_INTEGER_LIMIT_INPUTS: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "limits",
        classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "Current public limits accept only single or double; RunMat mode admits every integer class before one explicit host f64 graphics-state conversion.",
    }];

pub const CLIM_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 1] =
    [BuiltinIntegerCapabilityDescriptor {
        form: "clim(integer_limits)",
        inputs: &CLIM_INTEGER_LIMIT_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::Structural,
        output_class: BuiltinIntegerOutputClassRule::Double,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::HostOnly,
        overload: BuiltinIntegerOverloadKind::StructuralParameter,
        notes: "The two-element vector is shape- and order-validated from authoritative integer storage, then stored and queried as double host graphics metadata.",
    }];

#[runtime_builtin(
    name = "clim",
    category = "plotting",
    summary = "Query or set color limits.",
    keywords = "clim,plotting,color",
    suppress_auto_output = true,
    type_resolver(get_type),
    descriptor(crate::builtins::plotting::clim::CLIM_DESCRIPTOR),
    extensions(crate::builtins::plotting::clim::CLIM_EXTENSIONS),
    integer_capabilities(crate::builtins::plotting::clim::CLIM_INTEGER_CAPABILITIES),
    builtin_path = "crate::builtins::plotting::clim"
)]
pub fn clim_builtin(args: Vec<Value>) -> crate::BuiltinResult<Value> {
    clim_impl("clim", args)
}

pub(super) fn clim_impl(builtin: &'static str, args: Vec<Value>) -> crate::BuiltinResult<Value> {
    if let Some(Value::Tensor(tensor)) = args.first() {
        if matches!(
            tensor.numeric_dtype(),
            runmat_value::NumericDType::I8
                | runmat_value::NumericDType::I16
                | runmat_value::NumericDType::I32
                | runmat_value::NumericDType::I64
                | runmat_value::NumericDType::U8
                | runmat_value::NumericDType::U16
                | runmat_value::NumericDType::U32
                | runmat_value::NumericDType::U64
        ) {
            crate::compatibility::ensure_builtin_extension_enabled(
                &CLIM_INTEGER_LIMITS_EXTENSION,
                builtin,
            )?;
        }
    }
    if args.len() == 1
        && crate::builtins::plotting::style::value_as_string(&args[0])
            .is_some_and(|mode| mode.trim().eq_ignore_ascii_case("tight"))
    {
        return Err(crate::builtins::plotting::plotting_error(
            builtin,
            format!("{builtin}: unsupported mode `tight`"),
        ));
    }
    let call = prepare_limit_call(builtin, &args, true)?;
    let (handle, index) = selected_limit_axes(call.target);
    match call.command {
        LimitCommand::Query => {
            let limits = axis_displayed_color_limits_for_axes(handle, index)
                .map_err(|err| limit_state_error(builtin, err))?;
            Ok(limit_value(limits))
        }
        LimitCommand::QueryMode | LimitCommand::QueryMethod | LimitCommand::SetMethod(_) => {
            Err(crate::builtins::plotting::plotting_error(
                builtin,
                format!("{builtin}: axis-limit mode or method is unsupported for color limits"),
            ))
        }
        LimitCommand::Freeze => {
            let limits = axis_displayed_color_limits_for_axes(handle, index)
                .map_err(|err| limit_state_error(builtin, err))?
                .ok_or_else(|| {
                    crate::builtins::plotting::plotting_error(
                        builtin,
                        format!("{builtin}: automatic color extent is unavailable for manual mode"),
                    )
                })?;
            set_axis_color_limits_preserving_selection(handle, index, Some(limits))
                .map_err(|err| limit_state_error(builtin, err))?;
            Ok(limit_value(Some(limits)))
        }
        LimitCommand::Set(limits) => {
            set_axis_color_limits_preserving_selection(handle, index, limits)
                .map_err(|err| limit_state_error(builtin, err))?;
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
    fn clim_descriptor_signatures_cover_core_forms() {
        let labels: Vec<&str> = CLIM_DESCRIPTOR
            .signatures
            .iter()
            .map(|sig| sig.label)
            .collect();
        assert!(labels.contains(&"limits = clim()"));
        assert!(labels.contains(&"limits = clim([cmin cmax])"));
        assert!(labels.contains(&"limits = clim(mode)"));
    }

    #[test]
    fn clim_queries_and_sets_limits() {
        let _guard = lock_plot_registry();
        ensure_plot_test_env();
        reset_hold_state_for_run();
        let _ = clear_figure(None);

        let _ = clim_builtin(vec![Value::Tensor(
            runmat_value::Tensor::new(vec![0.25, 0.75], vec![1, 2]).expect("color limits"),
        )])
        .unwrap();
        let queried = clim_builtin(Vec::new()).unwrap();
        let tensor = runmat_value::Tensor::try_from(&queried).unwrap();
        assert_eq!(tensor.materialize_f64(), vec![0.25, 0.75]);
    }

    #[test]
    fn named_axes_targets_preserve_current_selection_and_displayed_property() {
        let _guard = lock_plot_registry();
        ensure_plot_test_env();
        super::super::state::reset_plot_state();
        super::super::state::configure_subplot(1, 2, 0).expect("first axes");
        let first = super::super::state::current_axes_state();
        let target = Value::Num(super::super::state::encode_axes_handle(first.handle, 0));
        super::super::state::configure_subplot(1, 2, 1).expect("second axes");
        let active_before = super::super::state::current_axes_state().active_index;

        struct TargetLimitCase {
            property_name: &'static str,
            builtin: fn(Vec<Value>) -> crate::BuiltinResult<Value>,
            expected: [f64; 2],
        }
        let cases = [
            TargetLimitCase {
                property_name: "XLim",
                builtin: super::super::xlim::xlim_builtin,
                expected: [2.0, 8.0],
            },
            TargetLimitCase {
                property_name: "YLim",
                builtin: super::super::ylim::ylim_builtin,
                expected: [-2.0, 4.0],
            },
            TargetLimitCase {
                property_name: "ZLim",
                builtin: super::super::zlim::zlim_builtin,
                expected: [-4.0, 4.0],
            },
            TargetLimitCase {
                property_name: "CLim",
                builtin: clim_builtin,
                expected: [0.25, 0.75],
            },
            TargetLimitCase {
                property_name: "CLim",
                builtin: super::super::caxis::caxis_builtin,
                expected: [0.1, 0.9],
            },
        ];
        for TargetLimitCase {
            property_name,
            builtin,
            expected,
        } in cases
        {
            let limits = Value::Tensor(
                runmat_value::Tensor::new(expected.to_vec(), vec![1, 2]).expect("limits"),
            );
            builtin(vec![target.clone(), limits]).expect("target setter");
            assert_eq!(
                super::super::state::current_axes_state().active_index,
                active_before
            );
            let queried = builtin(vec![target.clone()]).expect("target query");
            assert_eq!(
                runmat_value::Tensor::try_from(&queried)
                    .expect("limits tensor")
                    .materialize_f64(),
                expected.to_vec()
            );
            let property = super::super::properties::get_properties(
                super::super::properties::PlotHandle::Axes(first.handle, 0),
                Some(property_name),
                "get",
            )
            .expect("target property");
            assert_eq!(
                runmat_value::Tensor::try_from(&property)
                    .expect("property tensor")
                    .materialize_f64(),
                expected.to_vec()
            );
        }
    }
}

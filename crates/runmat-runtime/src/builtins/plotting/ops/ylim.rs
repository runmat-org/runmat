use runmat_builtins::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor, BuiltinOutputMode,
    BuiltinParamArity, BuiltinParamDescriptor, BuiltinParamType, BuiltinSignatureDescriptor, Value,
};
use runmat_macros::runtime_builtin;

use super::op_common::limits::{limit_value, parse_limit_command, LimitCommand};
use super::state::{axis_limits_snapshot, set_axis_limits};
use crate::builtins::plotting::type_resolvers::get_type;

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
    description: "Limit mode string ('auto', 'tight', or 'manual').",
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
    builtin_path = "crate::builtins::plotting::ylim"
)]
pub fn ylim_builtin(args: Vec<Value>) -> crate::BuiltinResult<Value> {
    match parse_limit_command("ylim", &args)? {
        LimitCommand::Query => Ok(limit_value(axis_limits_snapshot().1)),
        LimitCommand::Set(limits) => {
            let (x, _) = axis_limits_snapshot();
            set_axis_limits(x, limits);
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

        let _ = ylim_builtin(vec![Value::Tensor(runmat_builtins::Tensor {
            data: vec![1.0, 5.0],
            shape: vec![1, 2],
            rows: 1,
            cols: 2,
            dtype: runmat_builtins::NumericDType::F64,
        })])
        .unwrap();
        let _ = ylim_builtin(vec![Value::String("auto".into())]).unwrap();
        let queried = ylim_builtin(Vec::new()).unwrap();
        let tensor = runmat_builtins::Tensor::try_from(&queried).unwrap();
        assert!(tensor.data[0].is_nan() && tensor.data[1].is_nan());
    }
}

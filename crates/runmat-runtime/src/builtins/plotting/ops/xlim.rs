use runmat_builtins::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor, BuiltinOutputMode,
    BuiltinParamArity, BuiltinParamDescriptor, BuiltinParamType, BuiltinSignatureDescriptor, Value,
};
use runmat_macros::runtime_builtin;

use super::op_common::limits::{limit_value, parse_limit_command, LimitCommand};
use super::state::{axis_limits_snapshot, set_axis_limits};
use crate::builtins::plotting::type_resolvers::get_type;

const XLIM_OUTPUT_LIMITS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "limits",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Two-element vector [xmin xmax] or [NaN NaN] when auto.",
}];

const XLIM_INPUTS_NONE: [BuiltinParamDescriptor; 0] = [];

const XLIM_INPUTS_LIMIT_VECTOR: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "limits",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Two-element numeric vector [xmin xmax].",
}];

const XLIM_INPUTS_MODE: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "mode",
    ty: BuiltinParamType::StringScalar,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Limit mode string ('auto', 'tight', or 'manual').",
}];

const XLIM_SIGNATURES: [BuiltinSignatureDescriptor; 3] = [
    BuiltinSignatureDescriptor {
        label: "limits = xlim()",
        inputs: &XLIM_INPUTS_NONE,
        outputs: &XLIM_OUTPUT_LIMITS,
    },
    BuiltinSignatureDescriptor {
        label: "limits = xlim([xmin xmax])",
        inputs: &XLIM_INPUTS_LIMIT_VECTOR,
        outputs: &XLIM_OUTPUT_LIMITS,
    },
    BuiltinSignatureDescriptor {
        label: "limits = xlim(mode)",
        inputs: &XLIM_INPUTS_MODE,
        outputs: &XLIM_OUTPUT_LIMITS,
    },
];

const XLIM_ERROR_INVALID_ARGUMENT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.XLIM.INVALID_ARGUMENT",
    identifier: Some("RunMat:xlim:InvalidArgument"),
    when: "Argument count, mode, or limit vector is invalid.",
    message: "xlim: invalid argument",
};

const XLIM_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.XLIM.INTERNAL",
    identifier: Some("RunMat:xlim:Internal"),
    when: "Internal plotting state update fails.",
    message: "xlim: internal operation failed",
};

const XLIM_ERRORS: [BuiltinErrorDescriptor; 2] = [XLIM_ERROR_INVALID_ARGUMENT, XLIM_ERROR_INTERNAL];

pub const XLIM_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &XLIM_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &XLIM_ERRORS,
};

#[runtime_builtin(
    name = "xlim",
    category = "plotting",
    summary = "Query or set X-axis limits.",
    keywords = "xlim,plotting,axes",
    suppress_auto_output = true,
    type_resolver(get_type),
    descriptor(crate::builtins::plotting::xlim::XLIM_DESCRIPTOR),
    builtin_path = "crate::builtins::plotting::xlim"
)]
pub fn xlim_builtin(args: Vec<Value>) -> crate::BuiltinResult<Value> {
    match parse_limit_command("xlim", &args)? {
        LimitCommand::Query => Ok(limit_value(axis_limits_snapshot().0)),
        LimitCommand::Set(limits) => {
            let (_, y) = axis_limits_snapshot();
            set_axis_limits(limits, y);
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
    fn xlim_descriptor_signatures_cover_core_forms() {
        let labels: Vec<&str> = XLIM_DESCRIPTOR
            .signatures
            .iter()
            .map(|sig| sig.label)
            .collect();
        assert!(labels.contains(&"limits = xlim()"));
        assert!(labels.contains(&"limits = xlim([xmin xmax])"));
        assert!(labels.contains(&"limits = xlim(mode)"));
    }

    #[test]
    fn xlim_queries_and_sets_limits() {
        let _guard = lock_plot_registry();
        ensure_plot_test_env();
        reset_hold_state_for_run();
        let _ = clear_figure(None);

        let result = xlim_builtin(vec![Value::Tensor(runmat_builtins::Tensor {
            data: vec![0.0, 10.0],
            shape: vec![1, 2],
            rows: 1,
            cols: 2,
            dtype: runmat_builtins::NumericDType::F64,
        })])
        .unwrap();
        assert!(matches!(result, Value::Tensor(_)));
        let queried = xlim_builtin(Vec::new()).unwrap();
        let tensor = runmat_builtins::Tensor::try_from(&queried).unwrap();
        assert_eq!(tensor.data, vec![0.0, 10.0]);
    }
}

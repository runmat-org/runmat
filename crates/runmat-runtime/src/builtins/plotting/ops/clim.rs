use runmat_builtins::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor, BuiltinOutputMode,
    BuiltinParamArity, BuiltinParamDescriptor, BuiltinParamType, BuiltinSignatureDescriptor, Value,
};
use runmat_macros::runtime_builtin;

use super::op_common::limits::{limit_value, parse_limit_command, LimitCommand};
use super::state::{color_limits_snapshot, set_color_limits_runtime};
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

#[runtime_builtin(
    name = "clim",
    category = "plotting",
    summary = "Query or set color limits.",
    keywords = "clim,plotting,color",
    suppress_auto_output = true,
    type_resolver(get_type),
    descriptor(crate::builtins::plotting::clim::CLIM_DESCRIPTOR),
    builtin_path = "crate::builtins::plotting::clim"
)]
pub fn clim_builtin(args: Vec<Value>) -> crate::BuiltinResult<Value> {
    clim_impl("clim", args)
}

pub(super) fn clim_impl(builtin: &'static str, args: Vec<Value>) -> crate::BuiltinResult<Value> {
    match parse_limit_command(builtin, &args)? {
        LimitCommand::Query => Ok(limit_value(color_limits_snapshot())),
        LimitCommand::Set(limits) => {
            set_color_limits_runtime(limits);
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

        let _ = clim_builtin(vec![Value::Tensor(runmat_builtins::Tensor {
            data: vec![0.25, 0.75],
            shape: vec![1, 2],
            rows: 1,
            cols: 2,
            dtype: runmat_builtins::NumericDType::F64,
        })])
        .unwrap();
        let queried = clim_builtin(Vec::new()).unwrap();
        let tensor = runmat_builtins::Tensor::try_from(&queried).unwrap();
        assert_eq!(tensor.data, vec![0.25, 0.75]);
    }
}

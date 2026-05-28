use runmat_builtins::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor, BuiltinOutputMode,
    BuiltinParamArity, BuiltinParamDescriptor, BuiltinParamType, BuiltinSignatureDescriptor, Value,
};
use runmat_macros::runtime_builtin;

use super::op_common::limits::{limit_value, parse_limit_command, LimitCommand};
use super::state::{set_z_limits, z_limits_snapshot};
use crate::builtins::plotting::type_resolvers::get_type;

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
    description: "Limit mode string ('auto', 'tight', or 'manual').",
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
    builtin_path = "crate::builtins::plotting::zlim"
)]
pub fn zlim_builtin(args: Vec<Value>) -> crate::BuiltinResult<Value> {
    match parse_limit_command("zlim", &args)? {
        LimitCommand::Query => Ok(limit_value(z_limits_snapshot())),
        LimitCommand::Set(limits) => {
            set_z_limits(limits);
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

        let _ = zlim_builtin(vec![Value::Tensor(runmat_builtins::Tensor {
            data: vec![-1.0, 1.0],
            shape: vec![1, 2],
            rows: 1,
            cols: 2,
            dtype: runmat_builtins::NumericDType::F64,
        })])
        .unwrap();
        let queried = zlim_builtin(Vec::new()).unwrap();
        let tensor = runmat_builtins::Tensor::try_from(&queried).unwrap();
        assert_eq!(tensor.data, vec![-1.0, 1.0]);
    }
}

use runmat_builtins::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor, BuiltinOutputMode,
    BuiltinParamArity, BuiltinParamDescriptor, BuiltinParamType, BuiltinSignatureDescriptor, Value,
};
use runmat_macros::runtime_builtin;

use super::clim::clim_impl;
use crate::builtins::plotting::type_resolvers::get_type;

const CAXIS_OUTPUT_LIMITS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "limits",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Two-element color limit vector [cmin cmax] or [NaN NaN] when auto.",
}];

const CAXIS_INPUTS_NONE: [BuiltinParamDescriptor; 0] = [];

const CAXIS_INPUTS_LIMIT_VECTOR: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "limits",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Two-element numeric vector [cmin cmax].",
}];

const CAXIS_INPUTS_MODE: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "mode",
    ty: BuiltinParamType::StringScalar,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Limit mode string ('auto', 'tight', or 'manual').",
}];

const CAXIS_SIGNATURES: [BuiltinSignatureDescriptor; 3] = [
    BuiltinSignatureDescriptor {
        label: "limits = caxis()",
        inputs: &CAXIS_INPUTS_NONE,
        outputs: &CAXIS_OUTPUT_LIMITS,
    },
    BuiltinSignatureDescriptor {
        label: "limits = caxis([cmin cmax])",
        inputs: &CAXIS_INPUTS_LIMIT_VECTOR,
        outputs: &CAXIS_OUTPUT_LIMITS,
    },
    BuiltinSignatureDescriptor {
        label: "limits = caxis(mode)",
        inputs: &CAXIS_INPUTS_MODE,
        outputs: &CAXIS_OUTPUT_LIMITS,
    },
];

const CAXIS_ERROR_INVALID_ARGUMENT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.CAXIS.INVALID_ARGUMENT",
    identifier: Some("RunMat:caxis:InvalidArgument"),
    when: "Argument count, mode, or limit vector is invalid.",
    message: "caxis: invalid argument",
};

const CAXIS_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.CAXIS.INTERNAL",
    identifier: Some("RunMat:caxis:Internal"),
    when: "Internal plotting state update fails.",
    message: "caxis: internal operation failed",
};

const CAXIS_ERRORS: [BuiltinErrorDescriptor; 2] =
    [CAXIS_ERROR_INVALID_ARGUMENT, CAXIS_ERROR_INTERNAL];

pub const CAXIS_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &CAXIS_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &CAXIS_ERRORS,
};

#[runtime_builtin(
    name = "caxis",
    category = "plotting",
    summary = "Query or set color limits.",
    keywords = "caxis,plotting,color",
    suppress_auto_output = true,
    type_resolver(get_type),
    descriptor(crate::builtins::plotting::caxis::CAXIS_DESCRIPTOR),
    builtin_path = "crate::builtins::plotting::caxis"
)]
pub fn caxis_builtin(args: Vec<Value>) -> crate::BuiltinResult<Value> {
    clim_impl("caxis", args)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::builtins::plotting::tests::{ensure_plot_test_env, lock_plot_registry};
    use crate::builtins::plotting::{clear_figure, reset_hold_state_for_run};
    use runmat_builtins::{NumericDType, Tensor};

    #[test]
    fn caxis_descriptor_signatures_cover_core_forms() {
        let labels: Vec<&str> = CAXIS_DESCRIPTOR
            .signatures
            .iter()
            .map(|sig| sig.label)
            .collect();
        assert!(labels.contains(&"limits = caxis()"));
        assert!(labels.contains(&"limits = caxis([cmin cmax])"));
        assert!(labels.contains(&"limits = caxis(mode)"));
    }

    #[test]
    fn caxis_queries_and_sets_limits() {
        let _guard = lock_plot_registry();
        ensure_plot_test_env();
        reset_hold_state_for_run();
        let _ = clear_figure(None);

        let _ = caxis_builtin(vec![Value::Tensor(Tensor {
            data: vec![0.1, 0.9],
            shape: vec![1, 2],
            rows: 1,
            cols: 2,
            dtype: NumericDType::F64,
        })])
        .unwrap();

        let queried = caxis_builtin(Vec::new()).unwrap();
        let tensor = Tensor::try_from(&queried).unwrap();
        assert_eq!(tensor.data, vec![0.1, 0.9]);
    }

    #[test]
    fn caxis_errors_use_caxis_context() {
        let _guard = lock_plot_registry();
        ensure_plot_test_env();
        reset_hold_state_for_run();
        let _ = clear_figure(None);

        let err = caxis_builtin(vec![Value::String("badmode".into())]).unwrap_err();
        assert!(err.message.contains("caxis"));
        assert!(!err.message.contains("clim"));
    }
}

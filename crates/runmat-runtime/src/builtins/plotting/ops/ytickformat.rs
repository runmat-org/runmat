use runmat_builtins::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor, BuiltinOutputMode,
    BuiltinParamArity, BuiltinParamDescriptor, BuiltinParamType, BuiltinSignatureDescriptor, Value,
};
use runmat_macros::runtime_builtin;

use super::axis_tick_format::axis_tick_format_builtin;
use super::axis_ticks::TickAxis;
use crate::builtins::plotting::type_resolvers::get_type;

const OUTPUT_FORMAT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "fmt",
    ty: BuiltinParamType::StringScalar,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Current Y-axis tick label format.",
}];
const NO_OUTPUTS: [BuiltinParamDescriptor; 0] = [];

const INPUTS_NONE: [BuiltinParamDescriptor; 0] = [];

const INPUTS_FORMAT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "fmt",
    ty: BuiltinParamType::StringScalar,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Numeric, date, or duration tick label format.",
}];

const INPUTS_AX: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "ax",
    ty: BuiltinParamType::AxesHandle,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Target axes handle or array of axes handles.",
}];

const INPUTS_AX_FORMAT: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "ax",
        ty: BuiltinParamType::AxesHandle,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Target axes handle or array of axes handles.",
    },
    BuiltinParamDescriptor {
        name: "fmt",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Numeric, date, or duration tick label format.",
    },
];

const SIGNATURES: [BuiltinSignatureDescriptor; 4] = [
    BuiltinSignatureDescriptor {
        label: "fmt = ytickformat()",
        inputs: &INPUTS_NONE,
        outputs: &OUTPUT_FORMAT,
    },
    BuiltinSignatureDescriptor {
        label: "ytickformat(fmt)",
        inputs: &INPUTS_FORMAT,
        outputs: &NO_OUTPUTS,
    },
    BuiltinSignatureDescriptor {
        label: "fmt = ytickformat(ax)",
        inputs: &INPUTS_AX,
        outputs: &OUTPUT_FORMAT,
    },
    BuiltinSignatureDescriptor {
        label: "ytickformat(ax, fmt)",
        inputs: &INPUTS_AX_FORMAT,
        outputs: &NO_OUTPUTS,
    },
];

const ERROR_INVALID_ARGUMENT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.YTICKFORMAT.INVALID_ARGUMENT",
    identifier: Some("RunMat:ytickformat:InvalidArgument"),
    when: "Argument count, format value, or axes handle is invalid.",
    message: "ytickformat: invalid argument",
};

const ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.YTICKFORMAT.INTERNAL",
    identifier: Some("RunMat:ytickformat:Internal"),
    when: "Internal plotting state update fails.",
    message: "ytickformat: internal operation failed",
};

const ERRORS: [BuiltinErrorDescriptor; 2] = [ERROR_INVALID_ARGUMENT, ERROR_INTERNAL];

pub const YTICKFORMAT_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ERRORS,
};

#[runtime_builtin(
    name = "ytickformat",
    category = "plotting",
    summary = "Query or set Y-axis tick label format.",
    keywords = "ytickformat,plotting,axes,tick format",
    suppress_auto_output = true,
    type_resolver(get_type),
    descriptor(crate::builtins::plotting::ytickformat::YTICKFORMAT_DESCRIPTOR),
    builtin_path = "crate::builtins::plotting::ytickformat"
)]
pub fn ytickformat_builtin(args: Vec<Value>) -> crate::BuiltinResult<Value> {
    axis_tick_format_builtin("ytickformat", TickAxis::Y, args)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::builtins::plotting::axis_tick_labels::{label_cell_texts, tensor};
    use crate::builtins::plotting::get::get_builtin;
    use crate::builtins::plotting::tests::{ensure_plot_test_env, lock_plot_registry};
    use crate::builtins::plotting::{clear_figure, reset_hold_state_for_run};

    #[test]
    fn ytickformat_sets_queries_and_formats_auto_labels() {
        let _guard = lock_plot_registry();
        ensure_plot_test_env();
        reset_hold_state_for_run();
        let _ = clear_figure(None);

        crate::builtins::plotting::yticks::yticks_builtin(vec![tensor(vec![0.0, 0.5, 1.0])])
            .unwrap();
        assert_eq!(
            ytickformat_builtin(vec![Value::String("percentage".into())]).unwrap(),
            Value::String("%g%%".into())
        );

        let labels =
            crate::builtins::plotting::yticklabels::yticklabels_builtin(Vec::new()).unwrap();
        assert_eq!(label_cell_texts(&labels), vec!["0%", "0.5%", "1%"]);

        let ax = crate::builtins::plotting::gca::gca_builtin(Vec::new()).unwrap();
        let y_axis = get_builtin(vec![ax, Value::String("YAxis".into())]).unwrap();
        let prop = get_builtin(vec![y_axis, Value::String("TickLabelFormat".into())]).unwrap();
        assert_eq!(prop, Value::String("%g%%".into()));
    }
}

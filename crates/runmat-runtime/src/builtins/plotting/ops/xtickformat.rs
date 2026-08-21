use runmat_builtins::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor, BuiltinOutputMode,
    BuiltinParamArity, BuiltinParamDescriptor, BuiltinParamType, BuiltinSignatureDescriptor,
};
use runmat_macros::runtime_builtin;
use runmat_value::Value;

use super::axis_tick_format::axis_tick_format_builtin;
use super::axis_ticks::TickAxis;
use crate::builtins::plotting::type_resolvers::get_type;

const OUTPUT_FORMAT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "fmt",
    ty: BuiltinParamType::StringScalar,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Current X-axis tick label format.",
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
        label: "fmt = xtickformat()",
        inputs: &INPUTS_NONE,
        outputs: &OUTPUT_FORMAT,
    },
    BuiltinSignatureDescriptor {
        label: "xtickformat(fmt)",
        inputs: &INPUTS_FORMAT,
        outputs: &NO_OUTPUTS,
    },
    BuiltinSignatureDescriptor {
        label: "fmt = xtickformat(ax)",
        inputs: &INPUTS_AX,
        outputs: &OUTPUT_FORMAT,
    },
    BuiltinSignatureDescriptor {
        label: "xtickformat(ax, fmt)",
        inputs: &INPUTS_AX_FORMAT,
        outputs: &NO_OUTPUTS,
    },
];

const ERROR_INVALID_ARGUMENT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.XTICKFORMAT.INVALID_ARGUMENT",
    identifier: Some("RunMat:xtickformat:InvalidArgument"),
    when: "Argument count, format value, or axes handle is invalid.",
    message: "xtickformat: invalid argument",
};

const ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.XTICKFORMAT.INTERNAL",
    identifier: Some("RunMat:xtickformat:Internal"),
    when: "Internal plotting state update fails.",
    message: "xtickformat: internal operation failed",
};

const ERRORS: [BuiltinErrorDescriptor; 2] = [ERROR_INVALID_ARGUMENT, ERROR_INTERNAL];

pub const XTICKFORMAT_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ERRORS,
};

#[runtime_builtin(
    name = "xtickformat",
    category = "plotting",
    summary = "Query or set X-axis tick label format.",
    keywords = "xtickformat,plotting,axes,tick format",
    suppress_auto_output = true,
    type_resolver(get_type),
    descriptor(crate::builtins::plotting::xtickformat::XTICKFORMAT_DESCRIPTOR),
    builtin_path = "crate::builtins::plotting::xtickformat"
)]
pub fn xtickformat_builtin(args: Vec<Value>) -> crate::BuiltinResult<Value> {
    axis_tick_format_builtin("xtickformat", TickAxis::X, args)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::builtins::plotting::axis_tick_labels::{label_cell_texts, tensor};
    use crate::builtins::plotting::get::get_builtin;
    use crate::builtins::plotting::set::set_builtin;
    use crate::builtins::plotting::tests::{ensure_plot_test_env, lock_plot_registry};
    use crate::builtins::plotting::{clear_figure, reset_hold_state_for_run};

    #[test]
    fn xtickformat_sets_queries_formats_and_properties() {
        let _guard = lock_plot_registry();
        ensure_plot_test_env();
        reset_hold_state_for_run();
        let _ = clear_figure(None);

        assert_eq!(
            xtickformat_builtin(Vec::new()).unwrap(),
            Value::String("%g".into())
        );
        assert_eq!(
            xtickformat_builtin(vec![Value::String("usd".into())]).unwrap(),
            Value::String("$%,.2f".into())
        );
        assert_eq!(
            xtickformat_builtin(Vec::new()).unwrap(),
            Value::String("$%,.2f".into())
        );

        let ax = crate::builtins::plotting::gca::gca_builtin(Vec::new()).unwrap();
        let x_axis = get_builtin(vec![ax.clone(), Value::String("XAxis".into())]).unwrap();
        let prop = get_builtin(vec![
            x_axis.clone(),
            Value::String("TickLabelFormat".into()),
        ])
        .unwrap();
        assert_eq!(prop, Value::String("$%,.2f".into()));

        set_builtin(vec![
            x_axis,
            Value::String("TickLabelFormat".into()),
            Value::String("%.1f GHz".into()),
        ])
        .unwrap();
        assert_eq!(
            xtickformat_builtin(Vec::new()).unwrap(),
            Value::String("%.1f GHz".into())
        );

        crate::builtins::plotting::xticks::xticks_builtin(vec![tensor(vec![1.0, 2.5])]).unwrap();
        let labels =
            crate::builtins::plotting::xticklabels::xticklabels_builtin(Vec::new()).unwrap();
        assert_eq!(label_cell_texts(&labels), vec!["1.0 GHz", "2.5 GHz"]);
    }

    #[test]
    fn xtickformat_axes_target_is_isolated() {
        let _guard = lock_plot_registry();
        ensure_plot_test_env();
        reset_hold_state_for_run();
        let _ = clear_figure(None);

        let ax1 = crate::builtins::plotting::subplot::subplot_builtin(
            Value::Num(1.0),
            Value::Num(2.0),
            Value::Num(1.0),
        )
        .unwrap();
        let ax2 = crate::builtins::plotting::subplot::subplot_builtin(
            Value::Num(1.0),
            Value::Num(2.0),
            Value::Num(2.0),
        )
        .unwrap();

        xtickformat_builtin(vec![Value::Num(ax1), Value::String("percentage".into())]).unwrap();
        assert_eq!(
            xtickformat_builtin(vec![Value::Num(ax1)]).unwrap(),
            Value::String("%g%%".into())
        );
        assert_eq!(
            xtickformat_builtin(vec![Value::Num(ax2)]).unwrap(),
            Value::String("%g".into())
        );
        assert_eq!(
            crate::builtins::plotting::gca::gca_builtin(Vec::new()).unwrap(),
            Value::Num(ax2)
        );
    }

    #[test]
    fn xtickformat_axes_array_sets_each_axes_and_queries_require_scalar() {
        let _guard = lock_plot_registry();
        ensure_plot_test_env();
        reset_hold_state_for_run();
        let _ = clear_figure(None);

        let ax1 = crate::builtins::plotting::subplot::subplot_builtin(
            Value::Num(1.0),
            Value::Num(2.0),
            Value::Num(1.0),
        )
        .unwrap();
        let ax2 = crate::builtins::plotting::subplot::subplot_builtin(
            Value::Num(1.0),
            Value::Num(2.0),
            Value::Num(2.0),
        )
        .unwrap();

        xtickformat_builtin(vec![
            tensor(vec![ax1, ax2]),
            Value::String("%.2f ms".into()),
        ])
        .unwrap();
        assert_eq!(
            xtickformat_builtin(vec![Value::Num(ax1)]).unwrap(),
            Value::String("%.2f ms".into())
        );
        assert_eq!(
            xtickformat_builtin(vec![Value::Num(ax2)]).unwrap(),
            Value::String("%.2f ms".into())
        );
        assert!(xtickformat_builtin(vec![tensor(vec![ax1, ax2])]).is_err());
    }

    #[test]
    fn xtickformat_preserves_literal_spacing_and_explicit_g_format() {
        let _guard = lock_plot_registry();
        ensure_plot_test_env();
        reset_hold_state_for_run();
        let _ = clear_figure(None);

        xtickformat_builtin(vec![Value::String(" %.1f s ".into())]).unwrap();
        assert_eq!(
            xtickformat_builtin(Vec::new()).unwrap(),
            Value::String(" %.1f s ".into())
        );
        crate::builtins::plotting::xticks::xticks_builtin(vec![tensor(vec![1.0])]).unwrap();
        let labels =
            crate::builtins::plotting::xticklabels::xticklabels_builtin(Vec::new()).unwrap();
        assert_eq!(label_cell_texts(&labels), vec![" 1.0 s "]);

        xtickformat_builtin(vec![Value::String("%g".into())]).unwrap();
        crate::builtins::plotting::xticks::xticks_builtin(vec![tensor(vec![0.0005])]).unwrap();
        let labels =
            crate::builtins::plotting::xticklabels::xticklabels_builtin(Vec::new()).unwrap();
        assert_eq!(label_cell_texts(&labels), vec!["0.0005"]);
    }
}

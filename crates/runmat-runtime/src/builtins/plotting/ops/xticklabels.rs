use runmat_builtins::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor, BuiltinOutputMode,
    BuiltinParamArity, BuiltinParamDescriptor, BuiltinParamType, BuiltinSignatureDescriptor, Value,
};
use runmat_macros::runtime_builtin;

use super::axis_tick_labels::axis_tick_labels_builtin;
use super::axis_ticks::TickAxis;
use crate::builtins::plotting::type_resolvers::get_type;

const XTICKLABELS_OUTPUT_LABELS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "labels",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Current X-axis tick labels as a cell array of character vectors.",
}];

const XTICKLABELS_OUTPUT_MODE: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "mode",
    ty: BuiltinParamType::StringScalar,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Current X-axis tick label mode, 'auto' or 'manual'.",
}];

const XTICKLABELS_INPUTS_NONE: [BuiltinParamDescriptor; 0] = [];

const XTICKLABELS_INPUTS_LABELS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "labels",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "String array or cell array of character vectors for X-axis tick labels.",
}];

const XTICKLABELS_INPUTS_MODE: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "mode",
    ty: BuiltinParamType::StringScalar,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Tick label mode string: 'auto', 'manual', or 'mode'.",
}];

const XTICKLABELS_INPUTS_AX_LABELS: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "ax",
        ty: BuiltinParamType::AxesHandle,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Target axes handle.",
    },
    BuiltinParamDescriptor {
        name: "labels",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "String array or cell array of character vectors for X-axis tick labels.",
    },
];

const XTICKLABELS_SIGNATURES: [BuiltinSignatureDescriptor; 4] = [
    BuiltinSignatureDescriptor {
        label: "labels = xticklabels()",
        inputs: &XTICKLABELS_INPUTS_NONE,
        outputs: &XTICKLABELS_OUTPUT_LABELS,
    },
    BuiltinSignatureDescriptor {
        label: "labels = xticklabels(labels)",
        inputs: &XTICKLABELS_INPUTS_LABELS,
        outputs: &XTICKLABELS_OUTPUT_LABELS,
    },
    BuiltinSignatureDescriptor {
        label: "mode = xticklabels(mode)",
        inputs: &XTICKLABELS_INPUTS_MODE,
        outputs: &XTICKLABELS_OUTPUT_MODE,
    },
    BuiltinSignatureDescriptor {
        label: "labels = xticklabels(ax, labels)",
        inputs: &XTICKLABELS_INPUTS_AX_LABELS,
        outputs: &XTICKLABELS_OUTPUT_LABELS,
    },
];

const XTICKLABELS_ERROR_INVALID_ARGUMENT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.XTICKLABELS.INVALID_ARGUMENT",
    identifier: Some("RunMat:xticklabels:InvalidArgument"),
    when: "Argument count, tick label values, mode, or axes handle is invalid.",
    message: "xticklabels: invalid argument",
};

const XTICKLABELS_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.XTICKLABELS.INTERNAL",
    identifier: Some("RunMat:xticklabels:Internal"),
    when: "Internal plotting state update fails.",
    message: "xticklabels: internal operation failed",
};

const XTICKLABELS_ERRORS: [BuiltinErrorDescriptor; 2] = [
    XTICKLABELS_ERROR_INVALID_ARGUMENT,
    XTICKLABELS_ERROR_INTERNAL,
];

pub const XTICKLABELS_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &XTICKLABELS_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &XTICKLABELS_ERRORS,
};

#[runtime_builtin(
    name = "xticklabels",
    category = "plotting",
    summary = "Query or set X-axis tick labels.",
    keywords = "xticklabels,plotting,axes,tick labels",
    suppress_auto_output = true,
    type_resolver(get_type),
    descriptor(crate::builtins::plotting::xticklabels::XTICKLABELS_DESCRIPTOR),
    builtin_path = "crate::builtins::plotting::xticklabels"
)]
pub fn xticklabels_builtin(args: Vec<Value>) -> crate::BuiltinResult<Value> {
    axis_tick_labels_builtin("xticklabels", TickAxis::X, args)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::builtins::plotting::axis_tick_labels::{label_cell_texts, tensor};
    use crate::builtins::plotting::get::get_builtin;
    use crate::builtins::plotting::tests::{ensure_plot_test_env, lock_plot_registry};
    use crate::builtins::plotting::{clear_figure, reset_hold_state_for_run};
    use runmat_builtins::{CellArray, CharArray, StringArray};

    fn setup_plot_tests() -> crate::builtins::plotting::state::PlotTestLockGuard {
        let guard = lock_plot_registry();
        ensure_plot_test_env();
        reset_hold_state_for_run();
        let _ = clear_figure(None);
        guard
    }

    #[test]
    fn xticklabels_set_query_mode_and_property_round_trip() {
        let _guard = setup_plot_tests();
        crate::builtins::plotting::xticks::xticks_builtin(vec![tensor(vec![0.0, 5.0, 10.0])])
            .unwrap();
        let labels = Value::Cell(
            CellArray::new(
                vec![
                    Value::CharArray(CharArray::new_row("zero")),
                    Value::CharArray(CharArray::new_row("five")),
                ],
                1,
                2,
            )
            .unwrap(),
        );
        let set = xticklabels_builtin(vec![labels]).unwrap();
        assert_eq!(label_cell_texts(&set), vec!["zero", "five", ""]);
        assert_eq!(
            xticklabels_builtin(vec![Value::String("mode".into())]).unwrap(),
            Value::String("manual".into())
        );

        let queried = xticklabels_builtin(Vec::new()).unwrap();
        assert_eq!(label_cell_texts(&queried), vec!["zero", "five", ""]);
        let ax = crate::builtins::plotting::gca::gca_builtin(Vec::new()).unwrap();
        let prop = get_builtin(vec![ax.clone(), Value::String("XTickLabel".into())]).unwrap();
        assert_eq!(label_cell_texts(&prop), vec!["zero", "five", ""]);
        let mode = get_builtin(vec![ax, Value::String("XTickLabelMode".into())]).unwrap();
        assert_eq!(mode, Value::String("manual".into()));
    }

    #[test]
    fn xticklabels_auto_generates_from_ticks_and_axes_target_isolated() {
        let _guard = setup_plot_tests();
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
        crate::builtins::plotting::xticks::xticks_builtin(vec![
            Value::Num(ax1),
            tensor(vec![1.0, 2.5]),
        ])
        .unwrap();

        let labels = Value::StringArray(
            StringArray::new(vec!["one".into(), "two point five".into()], vec![1, 2]).unwrap(),
        );
        xticklabels_builtin(vec![Value::Num(ax1), labels]).unwrap();
        assert_eq!(
            xticklabels_builtin(vec![Value::Num(ax1), Value::String("mode".into())]).unwrap(),
            Value::String("manual".into())
        );
        assert_eq!(
            label_cell_texts(&xticklabels_builtin(vec![Value::Num(ax1)]).unwrap()),
            vec!["one", "two point five"]
        );
        assert_eq!(
            crate::builtins::plotting::gca::gca_builtin(Vec::new()).unwrap(),
            Value::Num(ax2)
        );

        xticklabels_builtin(vec![Value::Num(ax1), Value::String("auto".into())]).unwrap();
        assert_eq!(
            xticklabels_builtin(vec![Value::Num(ax1), Value::String("mode".into())]).unwrap(),
            Value::String("auto".into())
        );
        assert_eq!(
            label_cell_texts(&xticklabels_builtin(vec![Value::Num(ax1)]).unwrap()),
            vec!["1", "2.5"]
        );
    }
}

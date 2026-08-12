use runmat_builtins::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor, BuiltinOutputMode,
    BuiltinParamArity, BuiltinParamDescriptor, BuiltinParamType, BuiltinSignatureDescriptor,
};
use runmat_macros::runtime_builtin;
use runmat_value::Value;

use super::axis_tick_labels::axis_tick_labels_builtin;
use super::axis_ticks::TickAxis;
use crate::builtins::plotting::type_resolvers::get_type;

const YTICKLABELS_OUTPUT_LABELS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "labels",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Current Y-axis tick labels as a cell array of character vectors.",
}];

const YTICKLABELS_OUTPUT_MODE: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "mode",
    ty: BuiltinParamType::StringScalar,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Current Y-axis tick label mode, 'auto' or 'manual'.",
}];

const YTICKLABELS_INPUTS_NONE: [BuiltinParamDescriptor; 0] = [];

const YTICKLABELS_INPUTS_LABELS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "labels",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "String array or cell array of character vectors for Y-axis tick labels.",
}];

const YTICKLABELS_INPUTS_MODE: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "mode",
    ty: BuiltinParamType::StringScalar,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Tick label mode string: 'auto', 'manual', or 'mode'.",
}];

const YTICKLABELS_INPUTS_AX_LABELS: [BuiltinParamDescriptor; 2] = [
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
        description: "String array or cell array of character vectors for Y-axis tick labels.",
    },
];

const YTICKLABELS_SIGNATURES: [BuiltinSignatureDescriptor; 4] = [
    BuiltinSignatureDescriptor {
        label: "labels = yticklabels()",
        inputs: &YTICKLABELS_INPUTS_NONE,
        outputs: &YTICKLABELS_OUTPUT_LABELS,
    },
    BuiltinSignatureDescriptor {
        label: "labels = yticklabels(labels)",
        inputs: &YTICKLABELS_INPUTS_LABELS,
        outputs: &YTICKLABELS_OUTPUT_LABELS,
    },
    BuiltinSignatureDescriptor {
        label: "mode = yticklabels(mode)",
        inputs: &YTICKLABELS_INPUTS_MODE,
        outputs: &YTICKLABELS_OUTPUT_MODE,
    },
    BuiltinSignatureDescriptor {
        label: "labels = yticklabels(ax, labels)",
        inputs: &YTICKLABELS_INPUTS_AX_LABELS,
        outputs: &YTICKLABELS_OUTPUT_LABELS,
    },
];

const YTICKLABELS_ERROR_INVALID_ARGUMENT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.YTICKLABELS.INVALID_ARGUMENT",
    identifier: Some("RunMat:yticklabels:InvalidArgument"),
    when: "Argument count, tick label values, mode, or axes handle is invalid.",
    message: "yticklabels: invalid argument",
};

const YTICKLABELS_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.YTICKLABELS.INTERNAL",
    identifier: Some("RunMat:yticklabels:Internal"),
    when: "Internal plotting state update fails.",
    message: "yticklabels: internal operation failed",
};

const YTICKLABELS_ERRORS: [BuiltinErrorDescriptor; 2] = [
    YTICKLABELS_ERROR_INVALID_ARGUMENT,
    YTICKLABELS_ERROR_INTERNAL,
];

pub const YTICKLABELS_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &YTICKLABELS_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &YTICKLABELS_ERRORS,
};

#[runtime_builtin(
    name = "yticklabels",
    category = "plotting",
    summary = "Query or set Y-axis tick labels.",
    keywords = "yticklabels,plotting,axes,tick labels",
    suppress_auto_output = true,
    type_resolver(get_type),
    descriptor(crate::builtins::plotting::yticklabels::YTICKLABELS_DESCRIPTOR),
    builtin_path = "crate::builtins::plotting::yticklabels"
)]
pub fn yticklabels_builtin(args: Vec<Value>) -> crate::BuiltinResult<Value> {
    axis_tick_labels_builtin("yticklabels", TickAxis::Y, args)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::builtins::plotting::axis_tick_labels::{label_cell_texts, tensor};
    use crate::builtins::plotting::get::get_builtin;
    use crate::builtins::plotting::set::set_builtin;
    use crate::builtins::plotting::tests::{ensure_plot_test_env, lock_plot_registry};
    use crate::builtins::plotting::{clear_figure, reset_hold_state_for_run};
    use runmat_value::{CellArray, CharArray};

    fn setup_plot_tests() -> crate::builtins::plotting::state::PlotTestLockGuard {
        let guard = lock_plot_registry();
        ensure_plot_test_env();
        reset_hold_state_for_run();
        let _ = clear_figure(None);
        guard
    }

    #[test]
    fn yticklabels_supports_set_get_properties_and_manual_freeze() {
        let _guard = setup_plot_tests();
        let ax = crate::builtins::plotting::gca::gca_builtin(Vec::new()).unwrap();
        crate::builtins::plotting::yticks::yticks_builtin(vec![tensor(vec![0.0, 50.0, 100.0])])
            .unwrap();

        let labels = Value::Cell(
            CellArray::new(
                vec![
                    Value::CharArray(CharArray::new_row("low")),
                    Value::CharArray(CharArray::new_row("mid")),
                    Value::CharArray(CharArray::new_row("high")),
                ],
                1,
                3,
            )
            .unwrap(),
        );
        yticklabels_builtin(vec![labels]).unwrap();
        assert_eq!(
            label_cell_texts(&yticklabels_builtin(Vec::new()).unwrap()),
            vec!["low", "mid", "high"]
        );
        assert_eq!(
            yticklabels_builtin(vec![Value::String("mode".into())]).unwrap(),
            Value::String("manual".into())
        );
        let prop = get_builtin(vec![ax.clone(), Value::String("YTickLabel".into())]).unwrap();
        assert_eq!(label_cell_texts(&prop), vec!["low", "mid", "high"]);
        let tick_mode = get_builtin(vec![ax, Value::String("YTickMode".into())]).unwrap();
        assert_eq!(tick_mode, Value::String("manual".into()));
    }

    #[test]
    fn yticklabels_property_set_and_auto_mode_round_trip() {
        let _guard = setup_plot_tests();
        let ax = crate::builtins::plotting::gca::gca_builtin(Vec::new()).unwrap();
        crate::builtins::plotting::yticks::yticks_builtin(vec![tensor(vec![1.0, 4.0])]).unwrap();

        set_builtin(vec![
            ax.clone(),
            Value::String("YTickLabel".into()),
            Value::Cell(
                CellArray::new(
                    vec![
                        Value::CharArray(CharArray::new_row("one")),
                        Value::CharArray(CharArray::new_row("four")),
                    ],
                    1,
                    2,
                )
                .unwrap(),
            ),
        ])
        .unwrap();
        assert_eq!(
            label_cell_texts(&yticklabels_builtin(Vec::new()).unwrap()),
            vec!["one", "four"]
        );

        set_builtin(vec![
            ax.clone(),
            Value::String("YTickLabelMode".into()),
            Value::String("auto".into()),
        ])
        .unwrap();
        assert_eq!(
            yticklabels_builtin(vec![Value::String("mode".into())]).unwrap(),
            Value::String("auto".into())
        );
        assert_eq!(
            label_cell_texts(&yticklabels_builtin(Vec::new()).unwrap()),
            vec!["1", "4"]
        );
    }
}

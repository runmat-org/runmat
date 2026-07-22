use runmat_builtins::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor, BuiltinOutputMode,
    BuiltinParamArity, BuiltinParamDescriptor, BuiltinParamType, BuiltinSignatureDescriptor, Value,
};
use runmat_macros::runtime_builtin;

use super::axis_scale::axis_scale_builtin;
use super::axis_ticks::TickAxis;
use crate::builtins::plotting::type_resolvers::get_type;

const YSCALE_OUTPUT_SCALE: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "scale",
    ty: BuiltinParamType::StringScalar,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Current Y-axis scale, 'linear' or 'log'.",
}];

const YSCALE_INPUTS_NONE: [BuiltinParamDescriptor; 0] = [];

const YSCALE_INPUTS_SCALE: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "scale",
    ty: BuiltinParamType::StringScalar,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Scale string: 'linear' or 'log'.",
}];

const YSCALE_INPUTS_AX_SCALE: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "ax",
        ty: BuiltinParamType::AxesHandle,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Target axes handle.",
    },
    BuiltinParamDescriptor {
        name: "scale",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Scale string: 'linear' or 'log'.",
    },
];

const YSCALE_OUTPUTS_NONE: [BuiltinParamDescriptor; 0] = [];

const YSCALE_SIGNATURES: [BuiltinSignatureDescriptor; 3] = [
    BuiltinSignatureDescriptor {
        label: "scale = yscale()",
        inputs: &YSCALE_INPUTS_NONE,
        outputs: &YSCALE_OUTPUT_SCALE,
    },
    BuiltinSignatureDescriptor {
        label: "yscale(scale)",
        inputs: &YSCALE_INPUTS_SCALE,
        outputs: &YSCALE_OUTPUTS_NONE,
    },
    BuiltinSignatureDescriptor {
        label: "yscale(ax, scale)",
        inputs: &YSCALE_INPUTS_AX_SCALE,
        outputs: &YSCALE_OUTPUTS_NONE,
    },
];

const YSCALE_ERROR_INVALID_ARGUMENT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.YSCALE.INVALID_ARGUMENT",
    identifier: Some("RunMat:yscale:InvalidArgument"),
    when: "Argument count, scale value, or axes handle is invalid.",
    message: "yscale: invalid argument",
};

const YSCALE_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.YSCALE.INTERNAL",
    identifier: Some("RunMat:yscale:Internal"),
    when: "Internal plotting state update fails.",
    message: "yscale: internal operation failed",
};

const YSCALE_ERRORS: [BuiltinErrorDescriptor; 2] =
    [YSCALE_ERROR_INVALID_ARGUMENT, YSCALE_ERROR_INTERNAL];

pub const YSCALE_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &YSCALE_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &YSCALE_ERRORS,
};

#[runtime_builtin(
    name = "yscale",
    category = "plotting",
    summary = "Query or set Y-axis scale.",
    keywords = "yscale,plotting,axes,log,linear",
    suppress_auto_output = true,
    type_resolver(get_type),
    descriptor(crate::builtins::plotting::yscale::YSCALE_DESCRIPTOR),
    builtin_path = "crate::builtins::plotting::yscale"
)]
pub fn yscale_builtin(args: Vec<Value>) -> crate::BuiltinResult<Value> {
    axis_scale_builtin("yscale", TickAxis::Y, args)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::builtins::plotting::get::get_builtin;
    use crate::builtins::plotting::set::set_builtin;
    use crate::builtins::plotting::tests::{ensure_plot_test_env, lock_plot_registry};
    use crate::builtins::plotting::{
        clear_figure, clone_figure, current_figure_handle, reset_hold_state_for_run,
    };

    fn setup_plot_tests() -> crate::builtins::plotting::state::PlotTestLockGuard {
        let guard = lock_plot_registry();
        ensure_plot_test_env();
        reset_hold_state_for_run();
        let _ = clear_figure(None);
        guard
    }

    #[test]
    fn yscale_queries_sets_and_round_trips_property() {
        let _guard = setup_plot_tests();
        assert_eq!(
            yscale_builtin(Vec::new()).unwrap(),
            Value::String("linear".into())
        );
        assert_eq!(
            yscale_builtin(vec![Value::String("log".into())]).unwrap(),
            Value::String("log".into())
        );
        assert_eq!(
            yscale_builtin(Vec::new()).unwrap(),
            Value::String("log".into())
        );

        let ax = crate::builtins::plotting::gca::gca_builtin(Vec::new()).unwrap();
        let prop = get_builtin(vec![ax.clone(), Value::String("YScale".into())]).unwrap();
        assert_eq!(prop, Value::String("log".into()));

        set_builtin(vec![
            ax,
            Value::String("YScale".into()),
            Value::String("linear".into()),
        ])
        .unwrap();
        assert_eq!(
            yscale_builtin(Vec::new()).unwrap(),
            Value::String("linear".into())
        );
    }

    #[test]
    fn yscale_explicit_axes_does_not_change_current_axes() {
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

        yscale_builtin(vec![Value::Num(ax1), Value::String("log".into())]).unwrap();
        assert_eq!(
            yscale_builtin(vec![Value::Num(ax1)]).unwrap(),
            Value::String("log".into())
        );
        assert_eq!(
            crate::builtins::plotting::gca::gca_builtin(Vec::new()).unwrap(),
            Value::Num(ax2)
        );

        let fig = clone_figure(current_figure_handle()).unwrap();
        assert!(fig.axes_metadata(0).unwrap().y_log);
        assert!(!fig.axes_metadata(1).unwrap().y_log);
    }

    #[test]
    fn yscale_rejects_unknown_scale() {
        let _guard = setup_plot_tests();
        let err = yscale_builtin(vec![Value::String("sqrt".into())]).unwrap_err();
        assert!(err.message.contains("scale must be 'linear' or 'log'"));
    }

    #[test]
    fn yscale_rejects_invalid_numeric_axes_handle() {
        let _guard = setup_plot_tests();
        let err =
            yscale_builtin(vec![Value::Num(9_999_000.0), Value::String("log".into())]).unwrap_err();
        assert!(err.message.contains("invalid axes handle"));
    }

    #[test]
    fn yscale_descriptor_only_advertises_query_output() {
        let signatures = YSCALE_DESCRIPTOR.signatures;
        assert_eq!(signatures[0].outputs.len(), 1);
        assert!(signatures[1].outputs.is_empty());
        assert!(signatures[2].outputs.is_empty());
    }
}

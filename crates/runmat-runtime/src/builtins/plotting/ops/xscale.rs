use runmat_builtins::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor, BuiltinOutputMode,
    BuiltinParamArity, BuiltinParamDescriptor, BuiltinParamType, BuiltinSignatureDescriptor, Value,
};
use runmat_macros::runtime_builtin;

use super::axis_scale::axis_scale_builtin;
use super::axis_ticks::TickAxis;
use crate::builtins::plotting::type_resolvers::get_type;

const XSCALE_OUTPUT_SCALE: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "scale",
    ty: BuiltinParamType::StringScalar,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Current X-axis scale, 'linear' or 'log'.",
}];

const XSCALE_INPUTS_NONE: [BuiltinParamDescriptor; 0] = [];

const XSCALE_INPUTS_SCALE: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "scale",
    ty: BuiltinParamType::StringScalar,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Scale string: 'linear' or 'log'.",
}];

const XSCALE_INPUTS_AX_SCALE: [BuiltinParamDescriptor; 2] = [
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

const XSCALE_OUTPUTS_NONE: [BuiltinParamDescriptor; 0] = [];

const XSCALE_SIGNATURES: [BuiltinSignatureDescriptor; 3] = [
    BuiltinSignatureDescriptor {
        label: "scale = xscale()",
        inputs: &XSCALE_INPUTS_NONE,
        outputs: &XSCALE_OUTPUT_SCALE,
    },
    BuiltinSignatureDescriptor {
        label: "xscale(scale)",
        inputs: &XSCALE_INPUTS_SCALE,
        outputs: &XSCALE_OUTPUTS_NONE,
    },
    BuiltinSignatureDescriptor {
        label: "xscale(ax, scale)",
        inputs: &XSCALE_INPUTS_AX_SCALE,
        outputs: &XSCALE_OUTPUTS_NONE,
    },
];

const XSCALE_ERROR_INVALID_ARGUMENT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.XSCALE.INVALID_ARGUMENT",
    identifier: Some("RunMat:xscale:InvalidArgument"),
    when: "Argument count, scale value, or axes handle is invalid.",
    message: "xscale: invalid argument",
};

const XSCALE_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.XSCALE.INTERNAL",
    identifier: Some("RunMat:xscale:Internal"),
    when: "Internal plotting state update fails.",
    message: "xscale: internal operation failed",
};

const XSCALE_ERRORS: [BuiltinErrorDescriptor; 2] =
    [XSCALE_ERROR_INVALID_ARGUMENT, XSCALE_ERROR_INTERNAL];

pub const XSCALE_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &XSCALE_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &XSCALE_ERRORS,
};

#[runtime_builtin(
    name = "xscale",
    category = "plotting",
    summary = "Query or set X-axis scale.",
    keywords = "xscale,plotting,axes,log,linear",
    suppress_auto_output = true,
    type_resolver(get_type),
    descriptor(crate::builtins::plotting::xscale::XSCALE_DESCRIPTOR),
    builtin_path = "crate::builtins::plotting::xscale"
)]
pub fn xscale_builtin(args: Vec<Value>) -> crate::BuiltinResult<Value> {
    axis_scale_builtin("xscale", TickAxis::X, args)
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
    fn xscale_queries_sets_and_round_trips_property() {
        let _guard = setup_plot_tests();
        assert_eq!(
            xscale_builtin(Vec::new()).unwrap(),
            Value::String("linear".into())
        );
        assert_eq!(
            xscale_builtin(vec![Value::String("log".into())]).unwrap(),
            Value::String("log".into())
        );
        assert_eq!(
            xscale_builtin(Vec::new()).unwrap(),
            Value::String("log".into())
        );

        let ax = crate::builtins::plotting::gca::gca_builtin(Vec::new()).unwrap();
        let prop = get_builtin(vec![ax.clone(), Value::String("XScale".into())]).unwrap();
        assert_eq!(prop, Value::String("log".into()));

        set_builtin(vec![
            ax,
            Value::String("XScale".into()),
            Value::String("linear".into()),
        ])
        .unwrap();
        assert_eq!(
            xscale_builtin(Vec::new()).unwrap(),
            Value::String("linear".into())
        );
    }

    #[test]
    fn xscale_explicit_axes_does_not_change_current_axes() {
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

        xscale_builtin(vec![Value::Num(ax1), Value::String("log".into())]).unwrap();
        assert_eq!(
            xscale_builtin(vec![Value::Num(ax1)]).unwrap(),
            Value::String("log".into())
        );
        assert_eq!(
            crate::builtins::plotting::gca::gca_builtin(Vec::new()).unwrap(),
            Value::Num(ax2)
        );

        let fig = clone_figure(current_figure_handle()).unwrap();
        assert!(fig.axes_metadata(0).unwrap().x_log);
        assert!(!fig.axes_metadata(1).unwrap().x_log);
    }

    #[test]
    fn xscale_rejects_unknown_scale() {
        let _guard = setup_plot_tests();
        let err = xscale_builtin(vec![Value::String("sqrt".into())]).unwrap_err();
        assert!(err.message.contains("scale must be 'linear' or 'log'"));
    }

    #[test]
    fn xscale_rejects_invalid_numeric_axes_handle() {
        let _guard = setup_plot_tests();
        let err =
            xscale_builtin(vec![Value::Num(9_999_000.0), Value::String("log".into())]).unwrap_err();
        assert!(err.message.contains("invalid axes handle"));
    }

    #[test]
    fn xscale_descriptor_only_advertises_query_output() {
        let signatures = XSCALE_DESCRIPTOR.signatures;
        assert_eq!(signatures[0].outputs.len(), 1);
        assert!(signatures[1].outputs.is_empty());
        assert!(signatures[2].outputs.is_empty());
    }
}

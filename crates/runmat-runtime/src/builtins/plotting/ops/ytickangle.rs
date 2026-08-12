use runmat_builtins::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor, BuiltinOutputMode,
    BuiltinParamArity, BuiltinParamDescriptor, BuiltinParamType, BuiltinSignatureDescriptor,
};
use runmat_macros::runtime_builtin;
use runmat_value::Value;

use super::axis_tick_angle::axis_tick_angle_builtin;
use super::axis_ticks::TickAxis;
use crate::builtins::plotting::type_resolvers::get_type;

const OUTPUT_ANGLE: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "angle",
    ty: BuiltinParamType::NumericScalar,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Current Y-axis tick label rotation in degrees.",
}];
const NO_OUTPUTS: [BuiltinParamDescriptor; 0] = [];

const INPUTS_NONE: [BuiltinParamDescriptor; 0] = [];

const INPUTS_ANGLE: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "angle",
    ty: BuiltinParamType::NumericScalar,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Y-axis tick label rotation in degrees.",
}];

const INPUTS_AX: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "ax",
    ty: BuiltinParamType::AxesHandle,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Scalar target axes handle.",
}];

const INPUTS_AX_ANGLE: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "ax",
        ty: BuiltinParamType::AxesHandle,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Target axes handle or array of axes handles.",
    },
    BuiltinParamDescriptor {
        name: "angle",
        ty: BuiltinParamType::NumericScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Y-axis tick label rotation in degrees.",
    },
];

const SIGNATURES: [BuiltinSignatureDescriptor; 4] = [
    BuiltinSignatureDescriptor {
        label: "angle = ytickangle()",
        inputs: &INPUTS_NONE,
        outputs: &OUTPUT_ANGLE,
    },
    BuiltinSignatureDescriptor {
        label: "ytickangle(angle)",
        inputs: &INPUTS_ANGLE,
        outputs: &NO_OUTPUTS,
    },
    BuiltinSignatureDescriptor {
        label: "angle = ytickangle(ax)",
        inputs: &INPUTS_AX,
        outputs: &OUTPUT_ANGLE,
    },
    BuiltinSignatureDescriptor {
        label: "ytickangle(ax, angle)",
        inputs: &INPUTS_AX_ANGLE,
        outputs: &NO_OUTPUTS,
    },
];

const ERROR_INVALID_ARGUMENT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.YTICKANGLE.INVALID_ARGUMENT",
    identifier: Some("RunMat:ytickangle:InvalidArgument"),
    when: "Argument count, angle value, or axes handle is invalid.",
    message: "ytickangle: invalid argument",
};

const ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.YTICKANGLE.INTERNAL",
    identifier: Some("RunMat:ytickangle:Internal"),
    when: "Internal plotting state update fails.",
    message: "ytickangle: internal operation failed",
};

const ERRORS: [BuiltinErrorDescriptor; 2] = [ERROR_INVALID_ARGUMENT, ERROR_INTERNAL];

pub const YTICKANGLE_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ERRORS,
};

#[runtime_builtin(
    name = "ytickangle",
    category = "plotting",
    summary = "Query or set Y-axis tick label rotation.",
    keywords = "ytickangle,plotting,axes,tick angle",
    suppress_auto_output = true,
    type_resolver(get_type),
    descriptor(crate::builtins::plotting::ytickangle::YTICKANGLE_DESCRIPTOR),
    builtin_path = "crate::builtins::plotting::ytickangle"
)]
pub fn ytickangle_builtin(args: Vec<Value>) -> crate::BuiltinResult<Value> {
    axis_tick_angle_builtin("ytickangle", TickAxis::Y, args)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::builtins::plotting::axis_tick_labels::tensor;
    use crate::builtins::plotting::get::get_builtin;
    use crate::builtins::plotting::set::set_builtin;
    use crate::builtins::plotting::tests::{ensure_plot_test_env, lock_plot_registry};
    use crate::builtins::plotting::{clear_figure, reset_hold_state_for_run};

    #[test]
    fn ytickangle_sets_queries_and_round_trips_properties() {
        let _guard = lock_plot_registry();
        ensure_plot_test_env();
        reset_hold_state_for_run();
        let _ = clear_figure(None);

        assert_eq!(ytickangle_builtin(Vec::new()).unwrap(), Value::Num(0.0));
        assert_eq!(
            ytickangle_builtin(vec![Value::Num(25.0)]).unwrap(),
            Value::Num(25.0)
        );
        assert_eq!(ytickangle_builtin(Vec::new()).unwrap(), Value::Num(25.0));

        let ax = crate::builtins::plotting::gca::gca_builtin(Vec::new()).unwrap();
        assert_eq!(
            get_builtin(vec![ax.clone(), Value::String("YTickLabelRotation".into())]).unwrap(),
            Value::Num(25.0)
        );
        let y_axis = get_builtin(vec![ax, Value::String("YAxis".into())]).unwrap();
        assert!(ytickangle_builtin(vec![y_axis.clone()]).is_err());
        assert_eq!(
            get_builtin(vec![
                y_axis.clone(),
                Value::String("TickLabelRotation".into())
            ])
            .unwrap(),
            Value::Num(25.0)
        );

        set_builtin(vec![
            y_axis,
            Value::String("TickLabelRotation".into()),
            Value::Num(-45.0),
        ])
        .unwrap();
        assert_eq!(ytickangle_builtin(Vec::new()).unwrap(), Value::Num(-45.0));
    }

    #[test]
    fn ytickangle_axes_array_sets_each_axes_and_rejects_array_query() {
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

        ytickangle_builtin(vec![tensor(vec![ax1, ax2]), Value::Num(35.0)]).unwrap();
        assert_eq!(
            ytickangle_builtin(vec![Value::Num(ax1)]).unwrap(),
            Value::Num(35.0)
        );
        assert_eq!(
            ytickangle_builtin(vec![Value::Num(ax2)]).unwrap(),
            Value::Num(35.0)
        );
        assert!(ytickangle_builtin(vec![tensor(vec![ax1, ax2])]).is_err());
        assert!(ytickangle_builtin(vec![tensor(vec![10.0, 20.0])]).is_err());
        assert!(ytickangle_builtin(vec![Value::Num(f64::INFINITY)]).is_err());
        assert!(set_builtin(vec![
            Value::Num(ax1),
            Value::String("YTickLabelRotation".into()),
            tensor(vec![10.0, 20.0]),
        ])
        .is_err());
    }
}

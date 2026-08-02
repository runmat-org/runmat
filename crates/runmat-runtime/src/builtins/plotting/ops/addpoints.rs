//! MATLAB-compatible `addpoints` builtin.

use runmat_builtins::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor, BuiltinOutputMode,
    BuiltinParamArity, BuiltinParamDescriptor, BuiltinParamType, BuiltinSignatureDescriptor,
    Tensor, Value,
};
use runmat_macros::runtime_builtin;

use super::op_common::line_inputs::NumericInput;
use super::plotting_error;
use super::properties::{resolve_plot_handle, PlotHandle};
use super::state::{append_points_to_animated_line, PlotChildHandleState};
use crate::builtins::common::tensor as tensor_utils;
use crate::builtins::plotting::type_resolvers::set_type;
use crate::BuiltinResult;

const BUILTIN_NAME: &str = "addpoints";

const ADDPOINTS_INPUTS_2D: [BuiltinParamDescriptor; 3] = [
    BuiltinParamDescriptor {
        name: "an",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Animated line handle.",
    },
    BuiltinParamDescriptor {
        name: "x",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "X coordinates.",
    },
    BuiltinParamDescriptor {
        name: "y",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Y coordinates.",
    },
];

const ADDPOINTS_INPUTS_3D: [BuiltinParamDescriptor; 4] = [
    BuiltinParamDescriptor {
        name: "an",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Animated line handle.",
    },
    BuiltinParamDescriptor {
        name: "x",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "X coordinates.",
    },
    BuiltinParamDescriptor {
        name: "y",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Y coordinates.",
    },
    BuiltinParamDescriptor {
        name: "z",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Z coordinates for 3-D animated lines.",
    },
];

const ADDPOINTS_SIGNATURES: [BuiltinSignatureDescriptor; 2] = [
    BuiltinSignatureDescriptor {
        label: "addpoints(an, x, y)",
        inputs: &ADDPOINTS_INPUTS_2D,
        outputs: &[],
    },
    BuiltinSignatureDescriptor {
        label: "addpoints(an, x, y, z)",
        inputs: &ADDPOINTS_INPUTS_3D,
        outputs: &[],
    },
];

const ADDPOINTS_ERROR_INVALID_ARGUMENT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.ADDPOINTS.INVALID_ARGUMENT",
    identifier: Some("RunMat:addpoints:InvalidArgument"),
    when: "The handle is not an animated line, coordinates are invalid, or vector lengths differ.",
    message: "addpoints: invalid argument",
};

const ADDPOINTS_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.ADDPOINTS.INTERNAL",
    identifier: Some("RunMat:addpoints:Internal"),
    when: "Internal plotting state update fails.",
    message: "addpoints: internal operation failed",
};

const ADDPOINTS_ERRORS: [BuiltinErrorDescriptor; 2] =
    [ADDPOINTS_ERROR_INVALID_ARGUMENT, ADDPOINTS_ERROR_INTERNAL];

pub const ADDPOINTS_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &ADDPOINTS_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ADDPOINTS_ERRORS,
};

#[runtime_builtin(
    name = "addpoints",
    category = "plotting",
    summary = "Append points to an animated line.",
    keywords = "addpoints,animatedline,plotting,graphics,animation",
    sink = true,
    suppress_auto_output = true,
    type_resolver(set_type),
    descriptor(crate::builtins::plotting::addpoints::ADDPOINTS_DESCRIPTOR),
    builtin_path = "crate::builtins::plotting::addpoints"
)]
pub async fn addpoints_builtin(args: Vec<Value>) -> BuiltinResult<String> {
    if args.len() != 3 && args.len() != 4 {
        return Err(addpoints_err(
            "expected addpoints(an, x, y) or addpoints(an, x, y, z)",
        ));
    }

    let state = match resolve_plot_handle(&args[0], BUILTIN_NAME)? {
        PlotHandle::PlotChild(_, child) => match *child {
            PlotChildHandleState::AnimatedLine(state) => state,
            _ => {
                return Err(addpoints_err(
                    "first argument must be an animatedline handle",
                ))
            }
        },
        _ => {
            return Err(addpoints_err(
                "first argument must be an animatedline handle",
            ))
        }
    };

    let x = numeric_vector(args[1].clone(), "X").await?;
    let y = numeric_vector(args[2].clone(), "Y").await?;
    let z = if args.len() == 4 {
        Some(numeric_vector(args[3].clone(), "Z").await?)
    } else {
        None
    };
    append_points_to_animated_line(&state, x, y, z)
        .map_err(|err| plotting_error(BUILTIN_NAME, err))?;
    Ok("ok".to_string())
}

async fn numeric_vector(value: Value, name: &str) -> BuiltinResult<Vec<f64>> {
    let tensor = NumericInput::from_value(value, BUILTIN_NAME)?
        .into_tensor_async(BUILTIN_NAME)
        .await?;
    if !is_vector_tensor(&tensor) {
        return Err(addpoints_err(format!("{name} must be a scalar or vector")));
    }
    Ok(tensor_utils::tensor_into_values_f64(tensor))
}

fn is_vector_tensor(tensor: &Tensor) -> bool {
    tensor.rows() == 1 || tensor.cols() == 1
}

fn addpoints_err(detail: impl AsRef<str>) -> crate::RuntimeError {
    plotting_error(BUILTIN_NAME, detail.as_ref().to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::builtins::plotting::animatedline::animatedline_builtin;
    use crate::builtins::plotting::get::get_builtin;
    use crate::builtins::plotting::state::current_figure_revision;
    use crate::builtins::plotting::tests::{ensure_plot_test_env, lock_plot_registry};
    use crate::builtins::plotting::{
        clear_figure, current_figure_handle, reset_hold_state_for_run,
    };
    use futures::executor::block_on;

    fn setup() -> crate::builtins::plotting::state::PlotTestLockGuard {
        let guard = lock_plot_registry();
        ensure_plot_test_env();
        reset_hold_state_for_run();
        let _ = clear_figure(None);
        guard
    }

    #[test]
    fn addpoints_appends_2d_points_and_trims_oldest() {
        let _guard = setup();
        let handle = block_on(animatedline_builtin(vec![
            Value::String("MaximumNumPoints".into()),
            Value::Num(3.0),
        ]))
        .unwrap();
        block_on(addpoints_builtin(vec![
            handle.clone(),
            vector(&[1.0, 2.0]),
            vector(&[10.0, 20.0]),
        ]))
        .unwrap();
        block_on(addpoints_builtin(vec![
            handle.clone(),
            vector(&[3.0, 4.0]),
            vector(&[30.0, 40.0]),
        ]))
        .unwrap();
        let x = get_builtin(vec![handle, Value::String("XData".into())]).unwrap();
        assert_eq!(tensor_data(x), vec![2.0, 3.0, 4.0]);
    }

    #[test]
    fn addpoints_supports_3d_coordinates() {
        let _guard = setup();
        let handle = block_on(animatedline_builtin(Vec::new())).unwrap();
        block_on(addpoints_builtin(vec![
            handle.clone(),
            vector(&[1.0, 2.0]),
            vector(&[3.0, 4.0]),
            vector(&[5.0, 6.0]),
        ]))
        .unwrap();
        let z = get_builtin(vec![handle, Value::String("ZData".into())]).unwrap();
        assert_eq!(tensor_data(z), vec![5.0, 6.0]);
    }

    #[test]
    fn addpoints_bumps_figure_revision_for_renderer_flush() {
        let _guard = setup();
        let handle = block_on(animatedline_builtin(Vec::new())).unwrap();
        let figure = current_figure_handle();
        let before = current_figure_revision(figure).unwrap();
        block_on(addpoints_builtin(vec![
            handle,
            vector(&[1.0]),
            vector(&[2.0]),
        ]))
        .unwrap();
        let after = current_figure_revision(figure).unwrap();
        assert!(after > before, "addpoints must publish a figure revision");
    }

    #[test]
    fn addpoints_rejects_3d_promotion_when_markers_would_be_lost() {
        let _guard = setup();
        let handle = block_on(animatedline_builtin(vec![
            Value::String("Marker".into()),
            Value::String("o".into()),
        ]))
        .unwrap();
        let err = block_on(addpoints_builtin(vec![
            handle,
            vector(&[1.0]),
            vector(&[2.0]),
            vector(&[3.0]),
        ]))
        .expect_err("3-D promotion with markers should fail");
        assert!(err.message.contains("marker"));
    }

    #[test]
    fn addpoints_rejects_non_animatedline_handles() {
        let _guard = setup();
        let err = block_on(addpoints_builtin(vec![
            Value::Num(1.0),
            vector(&[1.0]),
            vector(&[2.0]),
        ]))
        .expect_err("figure handle is not an animated line");
        assert!(err.message.contains("animatedline handle"));
    }

    #[test]
    fn addpoints_reads_typed_integer_coordinates_exactly() {
        let _guard = setup();
        let handle = block_on(animatedline_builtin(Vec::new())).unwrap();
        block_on(addpoints_builtin(vec![
            handle.clone(),
            integer_vector(&[1, 2]),
            integer_vector(&[10, 20]),
            integer_vector(&[100, 200]),
        ]))
        .unwrap();
        assert_eq!(
            tensor_data(get_builtin(vec![handle.clone(), Value::String("XData".into())]).unwrap()),
            vec![1.0, 2.0]
        );
        assert_eq!(
            tensor_data(get_builtin(vec![handle.clone(), Value::String("YData".into())]).unwrap()),
            vec![10.0, 20.0]
        );
        assert_eq!(
            tensor_data(get_builtin(vec![handle, Value::String("ZData".into())]).unwrap()),
            vec![100.0, 200.0]
        );
    }

    fn vector(values: &[f64]) -> Value {
        Value::Tensor(
            Tensor::new(values.to_vec(), vec![1, values.len()]).expect("addpoints row vector"),
        )
    }

    fn integer_vector(values: &[i16]) -> Value {
        let tensor = Tensor::new_integer(
            runmat_builtins::IntegerStorage::I16(values.to_vec()),
            vec![1, values.len()],
        )
        .unwrap();
        Value::Tensor(tensor)
    }

    fn tensor_data(value: Value) -> Vec<f64> {
        match value {
            Value::Tensor(tensor) => tensor.materialize_f64(),
            other => panic!("expected tensor, got {other:?}"),
        }
    }
}

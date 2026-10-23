//! MATLAB-compatible `addpoints` builtin.

use runmat_builtins::{ADDPOINTS_DESCRIPTOR, ADDPOINTS_ERROR_TOO_MANY_OUTPUTS};

use runmat_macros::runtime_builtin;
use runmat_value::{Tensor, Value};

use super::op_common::line_inputs::NumericInput;
use super::plotting_error;
use super::properties::{resolve_plot_handle, PlotHandle};
use super::state::{append_points_to_animated_line, PlotChildHandleState};
use crate::builtins::common::tensor as tensor_utils;
use crate::BuiltinResult;

const BUILTIN_NAME: &str = "addpoints";

#[runtime_builtin(
    name = "addpoints",
    binding_variant = "default",
    context = true,
    builtin_path = "crate::builtins::plotting::addpoints"
)]
pub async fn addpoints_builtin(
    runtime: Option<crate::context::RuntimeContext>,
    arguments: crate::call::BuiltinArguments,
) -> BuiltinResult<String> {
    {
        let _scope = runtime.as_ref().map(|runtime| runtime.enter());
        crate::output_count::validate_fixed_outputs(
            &ADDPOINTS_DESCRIPTOR,
            &ADDPOINTS_ERROR_TOO_MANY_OUTPUTS,
            "addpoints",
        )
        .map_err(super::coordinate_admission::terminal)?;
    }
    let work = super::coordinate_admission::CoordinateWork::admit(runtime.as_ref(), "addpoints")?;
    let result = if let Some(runtime) = runtime.as_ref() {
        runtime.scope(addpoints_impl(arguments.view(), &work)).await
    } else {
        addpoints_impl(arguments.view(), &work).await
    };
    work.finish(result)
}

async fn addpoints_impl(
    args: crate::call::FunctionArgumentView<'_>,
    work: &super::coordinate_admission::CoordinateWork<'_>,
) -> BuiltinResult<String> {
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

    let x = numeric_vector(args[1].clone(), "X", work).await?;
    let y = numeric_vector(args[2].clone(), "Y", work).await?;
    let z = if args.len() == 4 {
        Some(numeric_vector(args[3].clone(), "Z", work).await?)
    } else {
        None
    };
    work.check()?;
    append_points_to_animated_line(&state, x, y, z)
        .map_err(|err| plotting_error(BUILTIN_NAME, err))?;
    Ok("ok".to_string())
}

async fn numeric_vector(
    value: Value,
    name: &str,
    work: &super::coordinate_admission::CoordinateWork<'_>,
) -> BuiltinResult<Vec<f64>> {
    if matches!(value, Value::Bool(_) | Value::LogicalArray(_)) {
        return Err(addpoints_err(format!("{name} must not be logical data")));
    }
    let tensor = NumericInput::from_value(value, BUILTIN_NAME)?
        .into_tensor_async(BUILTIN_NAME)
        .await?;
    work.check()?;
    if !is_vector_tensor(&tensor) {
        return Err(addpoints_err(format!("{name} must be a scalar or vector")));
    }
    Ok(tensor_utils::tensor_into_values_f64(tensor))
}

fn is_vector_tensor(tensor: &Tensor) -> bool {
    (tensor.rows() == 1 || tensor.cols() == 1)
        && tensor.shape().iter().skip(2).all(|&extent| extent == 1)
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
    use runmat_builtins::ADDPOINTS_INTEGER_CAPABILITIES;
    use runmat_value::IntegerStorage;

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
        let handle = block_on(animatedline_builtin(
            crate::builtins::common::test_support::builtin_arguments(vec![
                Value::String("MaximumNumPoints".into()),
                Value::Num(3.0),
            ]),
        ))
        .unwrap();
        block_on(addpoints_builtin(
            None,
            crate::builtins::common::test_support::builtin_arguments(vec![
                handle.clone(),
                vector(&[1.0, 2.0]),
                vector(&[10.0, 20.0]),
            ]),
        ))
        .unwrap();
        block_on(addpoints_builtin(
            None,
            crate::builtins::common::test_support::builtin_arguments(vec![
                handle.clone(),
                vector(&[3.0, 4.0]),
                vector(&[30.0, 40.0]),
            ]),
        ))
        .unwrap();
        let x = get_builtin(vec![handle, Value::String("XData".into())]).unwrap();
        assert_eq!(tensor_data(x), vec![2.0, 3.0, 4.0]);
    }

    #[test]
    fn addpoints_supports_3d_coordinates() {
        let _guard = setup();
        let handle = block_on(animatedline_builtin(
            crate::builtins::common::test_support::builtin_arguments(Vec::new()),
        ))
        .unwrap();
        block_on(addpoints_builtin(
            None,
            crate::builtins::common::test_support::builtin_arguments(vec![
                handle.clone(),
                vector(&[1.0, 2.0]),
                vector(&[3.0, 4.0]),
                vector(&[5.0, 6.0]),
            ]),
        ))
        .unwrap();
        let z = get_builtin(vec![handle, Value::String("ZData".into())]).unwrap();
        assert_eq!(tensor_data(z), vec![5.0, 6.0]);
    }

    #[test]
    fn addpoints_bumps_figure_revision_for_renderer_flush() {
        let _guard = setup();
        let handle = block_on(animatedline_builtin(
            crate::builtins::common::test_support::builtin_arguments(Vec::new()),
        ))
        .unwrap();
        let figure = current_figure_handle();
        let before = current_figure_revision(figure).unwrap();
        block_on(addpoints_builtin(
            None,
            crate::builtins::common::test_support::builtin_arguments(vec![
                handle,
                vector(&[1.0]),
                vector(&[2.0]),
            ]),
        ))
        .unwrap();
        let after = current_figure_revision(figure).unwrap();
        assert!(after > before, "addpoints must publish a figure revision");
    }

    #[test]
    fn addpoints_descriptor_classifies_both_integer_coordinate_forms() {
        assert_eq!(ADDPOINTS_INTEGER_CAPABILITIES.len(), 2);
        assert_eq!(
            ADDPOINTS_INTEGER_CAPABILITIES[0].form,
            "addpoints(an, integer_x, integer_y)"
        );
        assert_eq!(
            ADDPOINTS_INTEGER_CAPABILITIES[1].form,
            "addpoints(an, integer_x, integer_y, integer_z)"
        );
    }

    #[test]
    fn addpoints_rejects_3d_promotion_when_markers_would_be_lost() {
        let _guard = setup();
        let handle = block_on(animatedline_builtin(
            crate::builtins::common::test_support::builtin_arguments(vec![
                Value::String("Marker".into()),
                Value::String("o".into()),
            ]),
        ))
        .unwrap();
        let err = block_on(addpoints_builtin(
            None,
            crate::builtins::common::test_support::builtin_arguments(vec![
                handle,
                vector(&[1.0]),
                vector(&[2.0]),
                vector(&[3.0]),
            ]),
        ))
        .expect_err("3-D promotion with markers should fail");
        assert!(err.message.contains("marker"));
    }

    #[test]
    fn addpoints_rejects_non_animatedline_handles() {
        assert!(!is_vector_tensor(
            &Tensor::new(vec![1.0, 2.0], vec![1, 1, 2]).unwrap()
        ));
        assert!(is_vector_tensor(
            &Tensor::new(vec![1.0, 2.0], vec![1, 2, 1]).unwrap()
        ));
        let _guard = setup();
        let figure = crate::builtins::plotting::state::new_figure_handle();
        let err = block_on(addpoints_builtin(
            None,
            crate::builtins::common::test_support::builtin_arguments(vec![
                Value::Num(f64::from(figure.as_u32())),
                vector(&[1.0]),
                vector(&[2.0]),
            ]),
        ))
        .expect_err("figure handle is not an animated line");
        assert!(err.message.contains("animatedline handle"));
    }

    #[test]
    fn addpoints_accepts_all_integer_coordinate_classes_and_mixed_double() {
        let _guard = setup();
        let coordinates = [
            IntegerStorage::I8(vec![1, 2]),
            IntegerStorage::I16(vec![1, 2]),
            IntegerStorage::I32(vec![1, 2]),
            IntegerStorage::I64(vec![1, 2]),
            IntegerStorage::U8(vec![1, 2]),
            IntegerStorage::U16(vec![1, 2]),
            IntegerStorage::U32(vec![1, 2]),
            IntegerStorage::U64(vec![1, 2]),
        ];
        for storage in coordinates {
            let handle = block_on(animatedline_builtin(
                crate::builtins::common::test_support::builtin_arguments(Vec::new()),
            ))
            .unwrap();
            block_on(addpoints_builtin(
                None,
                crate::builtins::common::test_support::builtin_arguments(vec![
                    handle.clone(),
                    integer_value(storage.clone()),
                    vector(&[10.0, 20.0]),
                    integer_value(storage),
                ]),
            ))
            .unwrap();
            assert_eq!(
                tensor_data(
                    get_builtin(vec![handle.clone(), Value::String("XData".into())]).unwrap()
                ),
                vec![1.0, 2.0]
            );
            assert_eq!(
                tensor_data(
                    get_builtin(vec![handle.clone(), Value::String("YData".into())]).unwrap()
                ),
                vec![10.0, 20.0]
            );
            assert_eq!(
                tensor_data(get_builtin(vec![handle, Value::String("ZData".into())]).unwrap()),
                vec![1.0, 2.0]
            );
        }
    }

    #[test]
    fn addpoints_rejects_undocumented_logical_coordinates() {
        let _guard = setup();
        let handle = block_on(animatedline_builtin(
            crate::builtins::common::test_support::builtin_arguments(Vec::new()),
        ))
        .unwrap();
        let err = block_on(addpoints_builtin(
            None,
            crate::builtins::common::test_support::builtin_arguments(vec![
                handle,
                Value::Bool(true),
                Value::Num(2.0),
            ]),
        ))
        .expect_err("logical coordinates are not documented");
        assert!(err.message.contains("must not be logical"));
    }

    fn vector(values: &[f64]) -> Value {
        Value::Tensor(
            Tensor::new(values.to_vec(), vec![1, values.len()]).expect("addpoints row vector"),
        )
    }

    fn integer_value(storage: IntegerStorage) -> Value {
        let len = storage.len();
        let tensor = Tensor::new_integer(storage, vec![1, len]).unwrap();
        Value::Tensor(tensor)
    }

    fn tensor_data(value: Value) -> Vec<f64> {
        match value {
            Value::Tensor(tensor) => tensor.materialize_f64(),
            other => panic!("expected tensor, got {other:?}"),
        }
    }
}

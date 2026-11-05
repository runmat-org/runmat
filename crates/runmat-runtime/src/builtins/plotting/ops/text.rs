use runmat_builtins::{
    BuiltinErrorDescriptor, INTEGER_AXES_EXTENSION, TEXT_ERROR_INTERNAL,
    TEXT_ERROR_INVALID_ARGUMENT, TEXT_EXPLICIT_GPU_INPUT_EXTENSION,
};
use runmat_builtins::{TEXT_DESCRIPTOR, TEXT_ERROR_TOO_MANY_OUTPUTS};
use runmat_macros::runtime_builtin;
use runmat_value::{NumericStorage, Tensor, Value};

use super::op_common::{map_figure_error, value_as_text_string};
use crate::builtins::plotting::properties::parse_text_style_pairs;
use crate::builtins::plotting::state::add_text_annotation_for_axes_with_source;
use crate::{build_runtime_error, RuntimeError};

const BUILTIN_NAME: &str = "text";

fn text_error_with_detail(
    error: &'static BuiltinErrorDescriptor,
    detail: impl AsRef<str>,
) -> RuntimeError {
    let mut builder = build_runtime_error(format!("{}: {}", error.message, detail.as_ref()))
        .with_builtin(BUILTIN_NAME);
    if let Some(identifier) = error.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

fn map_text_invalid_argument(err: RuntimeError) -> RuntimeError {
    if err.identifier().is_some() {
        return err;
    }
    text_error_with_source(&TEXT_ERROR_INVALID_ARGUMENT, err)
}

fn map_text_internal(err: RuntimeError) -> RuntimeError {
    if err.identifier().is_some() {
        return err;
    }
    text_error_with_source(&TEXT_ERROR_INTERNAL, err)
}

fn text_error_with_source(
    error: &'static BuiltinErrorDescriptor,
    source: RuntimeError,
) -> RuntimeError {
    let mut builder = build_runtime_error(format!("{}: {}", error.message, source.message))
        .with_builtin(BUILTIN_NAME)
        .with_source(source);
    if let Some(identifier) = error.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

#[runtime_builtin(
    name = "text",
    binding_variant = "default",
    context = true,
    builtin_path = "crate::builtins::plotting::text"
)]
pub async fn text_builtin(
    runtime: Option<crate::context::RuntimeContext>,
    args: crate::call::BuiltinArguments,
) -> crate::BuiltinResult<f64> {
    crate::output_count::validate_fixed_outputs(
        &TEXT_DESCRIPTOR,
        &TEXT_ERROR_TOO_MANY_OUTPUTS,
        "text",
    )?;
    if args.len() < 3 {
        return Err(text_error_with_detail(
            &TEXT_ERROR_INVALID_ARGUMENT,
            "expected text(x, y, label) or text(x, y, z, label)",
        ));
    }
    if args
        .iter()
        .any(crate::builtins::common::validation::value_contains_explicit_gpu)
    {
        crate::compatibility::ensure_builtin_extension_enabled(
            &TEXT_EXPLICIT_GPU_INPUT_EXTENSION,
            BUILTIN_NAME,
        )?;
    }
    let typed_tensor_coordinate = matches!(
        args.first(),
        Some(Value::Tensor(tensor)) if tensor.integer_storage().is_some()
    );
    let super::op_common::text::DeferredAxesArguments { target, rest } = if typed_tensor_coordinate
    {
        super::op_common::text::DeferredAxesArguments {
            target: None,
            rest: args.view(),
        }
    } else {
        super::op_common::text::split_axes_target_deferred(BUILTIN_NAME, args.view())
            .map_err(map_text_invalid_argument)?
    };
    if rest.len() != args.len()
        && args
            .first()
            .is_some_and(crate::builtins::common::validation::value_has_native_integer_class)
    {
        crate::compatibility::ensure_builtin_extension_enabled(
            &INTEGER_AXES_EXTENSION,
            BUILTIN_NAME,
        )?;
    }
    if rest.len() < 3 {
        return Err(text_error_with_detail(
            &TEXT_ERROR_INVALID_ARGUMENT,
            "expected text(x, y, label) or text(x, y, z, label)",
        ));
    }
    let mut cooperation = runtime.as_ref().map(|context| context.cpu_cooperation());
    let (z, text_idx) = if let Some(text) = value_as_text_string(&rest[2]).map_err(|source| {
        crate::builtins::plotting::plotting_error_with_source("text", source.to_string(), source)
    })? {
        let style = parse_text_style_pairs(BUILTIN_NAME, rest.range(3..)?)
            .map_err(map_text_invalid_argument)?;
        validate_coordinate(&rest[0], "x")?;
        validate_coordinate(&rest[1], "y")?;
        let (position, source) =
            position_source(&rest[0], &rest[1], None, cooperation.as_ref()).await?;
        if let Some(cooperation) = cooperation.as_mut() {
            cooperation
                .checkpoint(0)
                .await
                .map_err(map_execution_error)?;
        }
        let target = target.unwrap_or_else(super::op_common::text::current_axes_target);
        return add_text_annotation_for_axes_with_source(
            target.0,
            target.1,
            position,
            &text,
            style,
            Some(source),
        )
        .map_err(|err| map_text_internal(map_figure_error(BUILTIN_NAME, err)));
    } else {
        (rest[2].clone(), 3usize)
    };

    if rest.len() <= text_idx {
        return Err(text_error_with_detail(
            &TEXT_ERROR_INVALID_ARGUMENT,
            "expected annotation string",
        ));
    }
    let text = value_as_text_string(&rest[text_idx])
        .map_err(|source| {
            crate::builtins::plotting::plotting_error_with_source(
                "text",
                source.to_string(),
                source,
            )
        })?
        .ok_or_else(|| {
            text_error_with_detail(&TEXT_ERROR_INVALID_ARGUMENT, "label must be text")
        })?;
    let style = parse_text_style_pairs(BUILTIN_NAME, rest.range(text_idx + 1..)?)
        .map_err(map_text_invalid_argument)?;
    validate_coordinate(&rest[0], "x")?;
    validate_coordinate(&rest[1], "y")?;
    validate_coordinate(&z, "z")?;
    let (position, source) =
        position_source(&rest[0], &rest[1], Some(&z), cooperation.as_ref()).await?;
    if let Some(cooperation) = cooperation.as_mut() {
        cooperation
            .checkpoint(0)
            .await
            .map_err(map_execution_error)?;
    }
    let target = target.unwrap_or_else(super::op_common::text::current_axes_target);
    add_text_annotation_for_axes_with_source(
        target.0,
        target.1,
        position,
        &text,
        style,
        Some(source),
    )
    .map_err(|err| map_text_internal(map_figure_error(BUILTIN_NAME, err)))
}

fn map_execution_error(source: crate::execution::ExecutionServiceError) -> RuntimeError {
    let mut error = source.clone().into_runtime_error();
    error.source = Some(Box::new(source));
    error
}

async fn position_source(
    x: &Value,
    y: &Value,
    z: Option<&Value>,
    cooperation: Option<&crate::execution::CpuCooperationBudget>,
) -> crate::BuiltinResult<(glam::Vec3, runmat_plot::plots::NumericPlotData)> {
    let x = scalar_numeric_tensor(x, "x", cooperation).await?;
    let y = scalar_numeric_tensor(y, "y", cooperation).await?;
    let z = match z {
        Some(value) => scalar_numeric_tensor(value, "z", cooperation).await?,
        None => {
            Tensor::from_numeric_storage(NumericStorage::zeros(x.numeric_dtype(), 1), vec![1, 1])
                .map_err(|err| text_error_with_detail(&TEXT_ERROR_INTERNAL, err))?
        }
    };
    let tensors = [&x, &y, &z];
    let same_class = tensors
        .iter()
        .all(|tensor| tensor.numeric_dtype() == x.numeric_dtype());
    let storage = if same_class {
        let mut storage = NumericStorage::zeros(x.numeric_dtype(), 3);
        for (index, tensor) in tensors.into_iter().enumerate() {
            storage
                .set_value(
                    index,
                    tensor
                        .numeric_value_at(0)
                        .expect("validated scalar numeric tensor"),
                )
                .map_err(|err| text_error_with_detail(&TEXT_ERROR_INTERNAL, err))?;
        }
        storage
    } else {
        NumericStorage::F64(
            tensors
                .into_iter()
                .map(|tensor| crate::builtins::common::tensor::tensor_value_f64(tensor, 0))
                .collect(),
        )
    };
    let source = runmat_plot::plots::NumericPlotData::new(storage, vec![1, 3])
        .map_err(|err| text_error_with_detail(&TEXT_ERROR_INTERNAL, err))?;
    let rendered = source.materialize_f64();
    Ok((
        glam::Vec3::new(rendered[0] as f32, rendered[1] as f32, rendered[2] as f32),
        source,
    ))
}

fn validate_coordinate(value: &Value, role: &str) -> crate::BuiltinResult<()> {
    if let Value::GpuTensor(handle) = value {
        let numeric_real = handle.physical().storage()
            == runmat_accelerate_api::GpuTensorStorage::Real
            && !runmat_accelerate_api::handle_is_logical(handle);
        let scalar = handle
            .shape()
            .iter()
            .try_fold(1usize, |count, dim| count.checked_mul(*dim))
            == Some(1);
        if !numeric_real || !scalar {
            return Err(text_error_with_detail(
                &TEXT_ERROR_INVALID_ARGUMENT,
                format!("{role} must be a real numeric scalar"),
            ));
        }
        handle
            .allocation_lease()
            .validate_live()
            .map_err(|source| {
                crate::runtime_error::with_provider_execution_source(
                    text_error_with_detail(
                        &TEXT_ERROR_INTERNAL,
                        format!("{role} allocation is unavailable"),
                    ),
                    runmat_accelerate_api::ProviderExecutionError::from_source(source),
                )
            })?;
        if runmat_accelerate_api::provider_for_handle(handle).is_none() {
            return Err(text_error_with_detail(
                &TEXT_ERROR_INTERNAL,
                format!("{role} has no exact acceleration provider"),
            ));
        }
        return Ok(());
    }
    let tensor =
        crate::builtins::common::tensor::value_into_tensor_for(BUILTIN_NAME, value.clone())
            .map_err(|_| {
                text_error_with_detail(
                    &TEXT_ERROR_INVALID_ARGUMENT,
                    format!("{role} must be a real numeric scalar"),
                )
            })?;
    if crate::builtins::common::tensor::is_scalar_tensor(&tensor) {
        Ok(())
    } else {
        Err(text_error_with_detail(
            &TEXT_ERROR_INVALID_ARGUMENT,
            format!("{role} must be a real numeric scalar"),
        ))
    }
}

async fn scalar_numeric_tensor(
    value: &Value,
    role: &str,
    cooperation: Option<&crate::execution::CpuCooperationBudget>,
) -> crate::BuiltinResult<Tensor> {
    let gathered;
    let value = if matches!(value, Value::GpuTensor(_)) {
        gathered = if let Some(cooperation) = cooperation {
            cooperation
                .await_cancellable(crate::gather_if_needed_async(value))
                .await
                .map_err(map_execution_error)?
                .map_err(map_text_internal)?
        } else {
            crate::gather_if_needed_async(value)
                .await
                .map_err(map_text_internal)?
        };
        &gathered
    } else {
        value
    };
    let tensor =
        crate::builtins::common::tensor::value_into_tensor_for(BUILTIN_NAME, value.clone())
            .map_err(|_| {
                text_error_with_detail(
                    &TEXT_ERROR_INVALID_ARGUMENT,
                    format!("{role} must be a real numeric scalar"),
                )
            })?;
    if !crate::builtins::common::tensor::is_scalar_tensor(&tensor) {
        return Err(text_error_with_detail(
            &TEXT_ERROR_INVALID_ARGUMENT,
            format!("{role} must be a real numeric scalar"),
        ));
    }
    Ok(tensor)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::builtins::common::{gpu_helpers, test_support};
    use crate::builtins::plotting::get::get_builtin;
    use crate::builtins::plotting::set::set_builtin;
    use crate::builtins::plotting::state::PlotTestLockGuard;
    use crate::builtins::plotting::tests::{ensure_plot_test_env, lock_plot_registry};
    use crate::builtins::plotting::{
        clear_figure, current_figure_handle, reset_hold_state_for_run,
    };
    use runmat_value::Tensor;
    use std::error::Error;

    fn text_builtin(args: Vec<Value>) -> crate::BuiltinResult<f64> {
        futures::executor::block_on(super::text_builtin(
            None,
            crate::builtins::common::test_support::builtin_arguments(args),
        ))
    }

    fn setup() -> PlotTestLockGuard {
        let guard = lock_plot_registry();
        ensure_plot_test_env();
        reset_hold_state_for_run();
        let _ = clear_figure(None);
        guard
    }

    #[test]
    fn text_descriptor_signatures_cover_core_forms() {
        let labels: Vec<&str> = TEXT_DESCRIPTOR
            .signatures
            .iter()
            .map(|sig| sig.label)
            .collect();
        assert!(labels.contains(&"h = text(x, y, label)"));
        assert!(labels.contains(&"h = text(x, y, z, label)"));
        assert!(labels.contains(&"h = text(ax, x, y, label)"));
    }

    #[test]
    fn text_missing_args_uses_stable_identifier() {
        let _guard = setup();
        let err = text_builtin(vec![]).expect_err("missing args should fail");
        assert_eq!(err.identifier(), TEXT_ERROR_INVALID_ARGUMENT.identifier);
    }

    #[test]
    fn text_creates_world_annotation_handle() {
        let _guard = setup();
        let handle = text_builtin(vec![
            Value::Num(1.0),
            Value::Num(2.0),
            Value::String("Hello".into()),
        ])
        .unwrap();
        let position =
            get_builtin(vec![Value::Num(handle), Value::String("Position".into())]).unwrap();
        let tensor = Tensor::try_from(&position).unwrap();
        assert_eq!(tensor.materialize_f64(), vec![1.0, 2.0, 0.0]);
    }

    /// MATLAB's anchors: `text(x, y, s, 'HorizontalAlignment', 'left', 'VerticalAlignment',
    /// 'bottom')` keeps them on the annotation's style, which the overlay places by; a value
    /// outside MATLAB's set is refused.
    #[test]
    fn text_keeps_its_alignment() {
        let _guard = setup();
        text_builtin(vec![
            Value::Num(1.0),
            Value::Num(2.0),
            Value::String("f3db".into()),
            Value::String("HorizontalAlignment".into()),
            Value::String("Left".into()),
            Value::String("VerticalAlignment".into()),
            Value::String("bottom".into()),
        ])
        .unwrap();
        let figure = crate::builtins::plotting::clone_figure(current_figure_handle()).expect("figure");
        let annotation = figure.axes_text_annotation(0, 0).expect("annotation");
        assert_eq!(annotation.style.horizontal_alignment.as_deref(), Some("left"));
        assert_eq!(annotation.style.vertical_alignment.as_deref(), Some("bottom"));
        let err = text_builtin(vec![
            Value::Num(1.0),
            Value::Num(2.0),
            Value::String("x".into()),
            Value::String("HorizontalAlignment".into()),
            Value::String("middle".into()),
        ])
        .expect_err("middle is a vertical alignment");
        assert!(err.to_string().contains("HorizontalAlignment must be one of left, center, right"), "{err}");
    }

    #[test]
    fn text_supports_3d_form() {
        let _guard = setup();
        let handle = text_builtin(vec![
            Value::Num(1.0),
            Value::Num(2.0),
            Value::Num(3.0),
            Value::String("Hello".into()),
        ])
        .unwrap();
        let fig = crate::builtins::plotting::clone_figure(current_figure_handle()).unwrap();
        let annotation = fig.axes_text_annotations(0).first().unwrap();
        assert_eq!(annotation.text, "Hello");
        assert_eq!(annotation.position, glam::Vec3::new(1.0, 2.0, 3.0));
        let string = get_builtin(vec![Value::Num(handle), Value::String("String".into())]).unwrap();
        assert_eq!(string, Value::String("Hello".into()));
    }

    #[test]
    fn text_preserves_explicit_axes_two_and_three_dimensional_forms() {
        let _guard = setup();
        let mut figure = runmat_plot::plots::Figure::new();
        figure.set_subplot_grid(1, 2);
        let figure = crate::builtins::plotting::state::import_figure(figure);
        let axes = crate::builtins::plotting::state::encode_axes_handle(figure, 1);
        text_builtin(vec![
            Value::Num(axes),
            Value::Num(1.0),
            Value::Num(2.0),
            Value::String("two".into()),
        ])
        .unwrap();
        text_builtin(vec![
            Value::Num(axes),
            Value::Num(3.0),
            Value::Num(4.0),
            Value::Num(5.0),
            Value::String("three".into()),
        ])
        .unwrap();
        let figure = crate::builtins::plotting::clone_figure(figure).unwrap();
        let annotations = figure.axes_text_annotations(1);
        assert_eq!(annotations.len(), 2);
        assert_eq!(annotations[0].text, "two");
        assert_eq!(annotations[1].position, glam::Vec3::new(3.0, 4.0, 5.0));
    }

    #[test]
    fn text_position_get_and_set_preserve_wide_integer_storage() {
        let _guard = setup();
        let wide = 9_007_199_254_740_993_u64;
        let scalar = |value| {
            Value::Tensor(
                Tensor::from_numeric_storage(NumericStorage::U64(vec![value]), vec![1, 1]).unwrap(),
            )
        };
        let handle = text_builtin(vec![
            scalar(wide),
            scalar(wide + 1),
            Value::String("exact".into()),
        ])
        .unwrap();
        let position = Tensor::try_from(
            &get_builtin(vec![Value::Num(handle), Value::String("Position".into())]).unwrap(),
        )
        .unwrap();
        assert_eq!(
            position.into_numeric_storage().unwrap(),
            NumericStorage::U64(vec![wide, wide + 1, 0])
        );

        let replacement = Tensor::from_numeric_storage(
            NumericStorage::I64(vec![-(wide as i64), 7, 9]),
            vec![1, 3],
        )
        .unwrap();
        set_builtin(vec![
            Value::Num(handle),
            Value::String("Position".into()),
            Value::Tensor(replacement),
        ])
        .unwrap();
        let position = Tensor::try_from(
            &get_builtin(vec![Value::Num(handle), Value::String("Position".into())]).unwrap(),
        )
        .unwrap();
        assert_eq!(
            position.into_numeric_storage().unwrap(),
            NumericStorage::I64(vec![-(wide as i64), 7, 9])
        );
    }

    #[test]
    fn text_documented_integer_coordinates_remain_available_in_compatibility_mode() {
        let _guard = setup();
        let _compat = crate::compatibility::push_runmat_extensions_enabled(false);
        let coordinate = |value| {
            Value::Tensor(
                Tensor::from_numeric_storage(NumericStorage::I32(vec![value]), vec![1, 1]).unwrap(),
            )
        };
        let handle = text_builtin(vec![
            coordinate(4),
            coordinate(-7),
            Value::String("documented".into()),
        ])
        .expect("documented integer coordinates");
        let position = Tensor::try_from(
            &get_builtin(vec![Value::Num(handle), Value::String("Position".into())]).unwrap(),
        )
        .unwrap();
        assert_eq!(
            position.into_numeric_storage().unwrap(),
            NumericStorage::I32(vec![4, -7, 0])
        );
    }

    #[test]
    fn text_annotations_clear_on_fresh_axes_replot() {
        let _guard = setup();
        let _ = text_builtin(vec![
            Value::Num(0.5),
            Value::Num(0.0),
            Value::String("midpoint".into()),
        ])
        .unwrap();
        futures::executor::block_on(crate::builtins::plotting::plot::plot_builtin(vec![
            Value::Tensor(Tensor::new_2d(vec![0.0, 1.0], 1, 2).unwrap()),
            Value::Tensor(Tensor::new_2d(vec![0.0, 1.0], 1, 2).unwrap()),
        ]))
        .unwrap();
        let fig = crate::builtins::plotting::clone_figure(current_figure_handle()).unwrap();
        assert!(fig.axes_text_annotations(0).is_empty());
    }

    #[test]
    fn text_explicit_gpu_scalar_coordinates_materialize_without_releasing_inputs() {
        test_support::with_test_provider(|provider| {
            let _guard = setup();
            let x = gpu_helpers::upload_tensor(provider, &Tensor::new_2d(vec![2.0], 1, 1).unwrap())
                .unwrap();
            let y = gpu_helpers::upload_tensor(provider, &Tensor::new_2d(vec![3.0], 1, 1).unwrap())
                .unwrap();
            let handle = futures::executor::block_on(super::text_builtin(
                None,
                crate::builtins::common::test_support::builtin_arguments(vec![
                    Value::GpuTensor(x.clone()),
                    Value::GpuTensor(y.clone()),
                    Value::String("resident".into()),
                ]),
            ))
            .expect("explicit resident coordinates");
            let position = Tensor::try_from(
                &get_builtin(vec![Value::Num(handle), Value::String("Position".into())]).unwrap(),
            )
            .unwrap();
            assert_eq!(position.materialize_f64(), vec![2.0, 3.0, 0.0]);
            assert!(
                futures::executor::block_on(crate::gather_if_needed_async(&Value::GpuTensor(x)))
                    .is_ok()
            );
            assert!(
                futures::executor::block_on(crate::gather_if_needed_async(&Value::GpuTensor(y)))
                    .is_ok()
            );
        });
    }

    #[test]
    fn malformed_host_label_rejects_before_resident_coordinate_download() {
        test_support::with_test_provider(|provider| {
            let _guard = setup();
            let x = gpu_helpers::upload_tensor(provider, &Tensor::new_2d(vec![2.0], 1, 1).unwrap())
                .unwrap();
            let before = crate::builtins::plotting::clone_figure(current_figure_handle())
                .unwrap()
                .axes_text_annotations(0)
                .len();
            let error = futures::executor::block_on(super::text_builtin(
                None,
                crate::builtins::common::test_support::builtin_arguments(vec![
                    Value::GpuTensor(x),
                    Value::Num(3.0),
                    Value::Num(4.0),
                ]),
            ))
            .expect_err("text label is required");
            assert_eq!(error.identifier(), TEXT_ERROR_INVALID_ARGUMENT.identifier);
            let after = crate::builtins::plotting::clone_figure(current_figure_handle())
                .unwrap()
                .axes_text_annotations(0)
                .len();
            assert_eq!(before, after);
        });
    }

    #[test]
    fn cancelled_context_does_not_materialize_or_publish_resident_annotation() {
        test_support::with_test_provider(|provider| {
            let _guard = setup();
            let x = gpu_helpers::upload_tensor(provider, &Tensor::new_2d(vec![2.0], 1, 1).unwrap())
                .unwrap();
            let source = runmat_execution::cancellation::CancellationSource::new();
            source.cancel(runmat_execution::CancellationReason::User);
            let runtime = crate::context::RuntimeContext::with_cancellation(
                std::rc::Rc::new(crate::execution::RuntimeExecutionService::default()),
                source.token(),
            );
            let before = crate::builtins::plotting::clone_figure(current_figure_handle())
                .unwrap()
                .axes_text_annotations(0)
                .len();
            let error = futures::executor::block_on(super::text_builtin(
                Some(runtime),
                crate::builtins::common::test_support::builtin_arguments(vec![
                    Value::GpuTensor(x),
                    Value::Num(3.0),
                    Value::String("cancelled".into()),
                ]),
            ))
            .expect_err("cancelled materialization must not publish a text annotation");
            assert!(matches!(
                error.source().and_then(
                    |source| source.downcast_ref::<crate::execution::ExecutionServiceError>()
                ),
                Some(crate::execution::ExecutionServiceError::Cancelled)
            ));
            let after = crate::builtins::plotting::clone_figure(current_figure_handle())
                .unwrap()
                .axes_text_annotations(0)
                .len();
            assert_eq!(before, after);
        });
    }
}

#[cfg(test)]
mod output_contract_tests {
    fn text_builtin(args: Vec<runmat_value::Value>) -> crate::BuiltinResult<f64> {
        futures::executor::block_on(super::text_builtin(
            None,
            crate::builtins::common::test_support::builtin_arguments(args),
        ))
    }

    #[test]
    fn missing_required_input_uses_identity_diagnostic() {
        let error = text_builtin(vec![]).expect_err("missing annotation input");
        assert_eq!(error.identifier(), Some("RunMat:text:InvalidArgument"));
    }
    #[test]
    fn excessive_outputs_reject_before_argument_or_graphics_effects() {
        let _outputs = crate::output_count::push_output_count(Some(2)).expect("output-count scope");
        let error = text_builtin(vec![]).expect_err("excess outputs");
        assert_eq!(error.identifier(), Some("RunMat:text:TooManyOutputs"));
    }
}

#[cfg(test)]
#[path = "text/resident_tests.rs"]
mod resident_tests;

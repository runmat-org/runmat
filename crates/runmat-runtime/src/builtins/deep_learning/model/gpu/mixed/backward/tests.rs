use super::*;
use futures::executor::block_on;
use runmat_accelerate_api::{AccelProvider, HostTensorView, ProviderPrecision};
use runmat_value::Value;

fn upload(
    provider: &'static dyn AccelProvider,
    values: &[f64],
    shape: &[usize],
) -> GpuTensorHandle {
    provider
        .upload(&HostTensorView {
            data: values,
            shape,
        })
        .unwrap()
}

fn values(handle: GpuTensorHandle) -> Vec<f64> {
    crate::builtins::common::test_support::gather(Value::GpuTensor(handle))
        .unwrap()
        .materialize_f64()
}

#[test]
fn relu_and_elu_adjoints_use_native_stage_inputs_without_download() {
    crate::builtins::common::test_support::with_test_provider(|provider| {
        let shape = [2, 3];
        let x = [-2.0, 0.0, 0.5, 3.0, -0.25, 1.0];
        let dy = [1.0, -2.0, 0.75, 1.5, -3.0, 2.0];
        let input = upload(provider, &x, &shape);
        for (kind, expected) in [
            (ResidentStageKind::Relu, [0.0, 0.0, 0.75, 1.5, 0.0, 2.0]),
            (
                ResidentStageKind::Elu { alpha: 1.25 },
                [
                    1.25 * (-2.0f64).exp(),
                    -2.0,
                    0.75,
                    1.5,
                    -3.0 * 1.25 * (-0.25f64).exp(),
                    2.0,
                ],
            ),
        ] {
            let forward = match &kind {
                ResidentStageKind::Relu => block_on(provider.activation_relu(&input)).unwrap(),
                ResidentStageKind::Elu { alpha } => {
                    block_on(provider.activation_elu(&input, *alpha)).unwrap()
                }
                _ => unreachable!("activation fixture"),
            };
            let trace = ResidentTrace {
                stages: vec![super::super::trace::ResidentStage {
                    kind,
                    input: input.clone(),
                    output: forward.clone(),
                }],
                output: forward,
            };
            let seed = upload(provider, &dy, &shape);
            provider.reset_telemetry();
            let result = block_on(reverse(&trace, seed, &|| Ok(()))).unwrap();
            assert!(result.parameters.is_empty());
            assert_eq!(result.input.shape(), &shape);
            assert_eq!(
                runmat_accelerate_api::handle_precision(&result.input),
                Some(ProviderPrecision::F64)
            );
            assert_eq!(provider.telemetry_snapshot().download_bytes, 0);
            for (actual, expected) in values(result.input).into_iter().zip(expected) {
                assert!(
                    (actual - expected).abs() < 1e-12,
                    "{actual} versus {expected}"
                );
            }
        }
    });
}

#[test]
fn softmax_adjoint_non_square_rows_match_independent_jacobian() {
    crate::builtins::common::test_support::with_test_provider(|provider| {
        let shape = [2, 3];
        let x = [1.0, 2.0, 2.0, 0.0, 3.0, -1.0];
        let dy = [0.5, -1.0, -0.75, 0.2, 1.5, 0.3];
        let input = upload(provider, &x, &shape);
        let output = block_on(provider.activation_softmax_rows(&input)).unwrap();
        let trace = ResidentTrace {
            stages: vec![super::super::trace::ResidentStage {
                kind: ResidentStageKind::Softmax,
                input,
                output: output.clone(),
            }],
            output,
        };
        let seed = upload(provider, &dy, &shape);
        provider.reset_telemetry();
        let dx = block_on(reverse(&trace, seed, &|| Ok(()))).unwrap().input;
        assert_eq!(provider.telemetry_snapshot().download_bytes, 0);
        let actual = values(dx);
        for row in 0..shape[0] {
            let exponentials = (0..shape[1])
                .map(|col| (x[row + col * shape[0]] - 3.0).exp())
                .collect::<Vec<_>>();
            let total = exponentials.iter().sum::<f64>();
            let probabilities = exponentials.iter().map(|e| e / total).collect::<Vec<_>>();
            let weighted = probabilities
                .iter()
                .enumerate()
                .map(|(col, p)| p * dy[row + col * shape[0]])
                .sum::<f64>();
            for (col, p) in probabilities.iter().enumerate() {
                let index = row + col * shape[0];
                let expected = p * (dy[index] - weighted);
                assert!(
                    (actual[index] - expected).abs() < 1e-12,
                    "row {row}, col {col}"
                );
            }
        }
    });
}

#[test]
fn single_softmax_adjoint_restores_row_sum_before_subtraction_without_download() {
    crate::builtins::common::test_support::with_test_provider(|provider| {
        let shape = [2, 3];
        let input_values = [1.0_f32, 2.0, 2.0, 0.0, 3.0, -1.0];
        let seed_values = [0.5_f32, -1.0, -0.75, 0.2, 1.5, 0.3];
        let upload = |values: &[f32]| {
            provider
                .upload_numeric(&runmat_accelerate_api::HostNumericTensorView {
                    data: runmat_accelerate_api::HostNumericDataView::F32(values),
                    shape: &shape,
                    storage: GpuTensorStorage::Real,
                })
                .unwrap()
        };
        let input = upload(&input_values);
        let forward = block_on(provider.activation_softmax_rows(&input)).unwrap();
        let trace = ResidentTrace {
            stages: vec![super::super::trace::ResidentStage {
                kind: ResidentStageKind::Softmax,
                input: input.clone(),
                output: forward.clone(),
            }],
            output: forward.clone(),
        };
        let seed = upload(&seed_values);
        provider.reset_telemetry();
        let result = block_on(reverse(&trace, seed, &|| Ok(()))).unwrap();
        assert_eq!(
            runmat_accelerate_api::handle_precision(&result.input),
            Some(ProviderPrecision::F32)
        );
        assert_eq!(provider.telemetry_snapshot().download_bytes, 0);
        let actual = values(result.input);
        let probabilities = values(forward);
        for row in 0..shape[0] {
            let weighted = (0..shape[1])
                .map(|column| {
                    let index = row + column * shape[0];
                    probabilities[index] * f64::from(seed_values[index])
                })
                .sum::<f64>() as f32;
            for column in 0..shape[1] {
                let index = row + column * shape[0];
                let expected = probabilities[index] * f64::from(seed_values[index] - weighted);
                assert!((actual[index] - expected).abs() < 2e-6);
            }
        }
        input.allocation_lease().validate_live().unwrap();
    });
}

#[test]
fn single_dense_bias_adjoint_restores_parameter_precision_without_download() {
    crate::builtins::common::test_support::with_test_provider(|provider| {
        let upload = |values: &[f32], shape: &[usize]| {
            provider
                .upload_numeric(&runmat_accelerate_api::HostNumericTensorView {
                    data: runmat_accelerate_api::HostNumericDataView::F32(values),
                    shape,
                    storage: GpuTensorStorage::Real,
                })
                .unwrap()
        };
        let input = upload(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[3, 2]);
        let weights = upload(&[1.0, 0.0, 0.0, 1.0], &[2, 2]);
        let output = block_on(provider.matmul(&input, &weights)).unwrap();
        let trace = ResidentTrace {
            stages: vec![super::super::trace::ResidentStage {
                kind: ResidentStageKind::Dense {
                    layer_index: 0,
                    weights,
                },
                input: input.clone(),
                output: output.clone(),
            }],
            output,
        };
        let seed = upload(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[3, 2]);
        provider.reset_telemetry();
        let gradients = block_on(reverse(&trace, seed, &|| Ok(()))).unwrap();
        assert_eq!(provider.telemetry_snapshot().download_bytes, 0);
        assert_eq!(
            runmat_accelerate_api::handle_precision(&gradients.input),
            Some(ProviderPrecision::F32)
        );
        let bias = gradients
            .parameters
            .iter()
            .find(|(_, role, _)| *role == ParameterRole::Bias)
            .unwrap();
        assert_eq!(
            runmat_accelerate_api::handle_precision(&bias.2),
            Some(ProviderPrecision::F32)
        );
        assert_eq!(values(bias.2.clone()), vec![6.0, 15.0]);
        input.allocation_lease().validate_live().unwrap();
    });
}

#[test]
fn single_adjoint_rejects_a_double_reduction_disguised_as_single() {
    crate::builtins::common::test_support::with_test_provider(|provider| {
        let upload = |values: &[f32], shape: &[usize]| {
            provider
                .upload_numeric(&runmat_accelerate_api::HostNumericTensorView {
                    data: runmat_accelerate_api::HostNumericDataView::F32(values),
                    shape,
                    storage: GpuTensorStorage::Real,
                })
                .unwrap()
        };
        let input = upload(&[1.0, 2.0], &[2, 1]);
        let malformed = provider
            .upload_numeric(&runmat_accelerate_api::HostNumericTensorView {
                data: runmat_accelerate_api::HostNumericDataView::F64(&[3.0]),
                shape: &[1, 1],
                storage: GpuTensorStorage::Real,
            })
            .unwrap();
        provider.reset_telemetry();
        assert!(
            checked_native_sum_reduction(malformed.clone(), &input, provider, &[1, 1],).is_err()
        );
        assert_eq!(provider.telemetry_snapshot().download_bytes, 0);
        input.allocation_lease().validate_live().unwrap();
        malformed.allocation_lease().validate_live().unwrap();
    });
}

#[test]
fn cancellation_before_reverse_preserves_borrowed_stage_handles() {
    crate::builtins::common::test_support::with_test_provider(|provider| {
        let input = upload(provider, &[1.0, -2.0], &[1, 2]);
        let forward = block_on(provider.activation_relu(&input)).unwrap();
        let trace = ResidentTrace {
            stages: vec![super::super::trace::ResidentStage {
                kind: ResidentStageKind::Relu,
                input: input.clone(),
                output: forward.clone(),
            }],
            output: forward.clone(),
        };
        let seed = upload(provider, &[3.0, 4.0], &[1, 2]);
        let called = std::cell::Cell::new(0usize);
        let error = block_on(reverse(&trace, seed, &|| {
            called.set(called.get() + 1);
            Err(deep_learning_error(
                "dlgradient",
                "cancelled before provider work",
            ))
        }))
        .err()
        .expect("cancelled reverse must not dispatch");
        assert!(error.message().contains("cancelled before provider work"));
        assert_eq!(called.get(), 1);
        assert_eq!(values(input), vec![1.0, -2.0]);
        assert_eq!(values(forward), vec![1.0, 0.0]);
    });
}

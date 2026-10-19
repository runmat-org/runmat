#[path = "support/retirement.rs"]
mod retirement;
use futures::executor::block_on;
use runmat_accelerate_api::{AccelProvider, HostTensorView};
#[cfg(feature = "wgpu")]
#[path = "support/typed_fusion_plan.rs"]
mod fixture;

async fn exercise(provider: &dyn AccelProvider) {
    let left = provider
        .upload(&HostTensorView {
            data: &[7.0, -7.0],
            shape: &[2, 1],
        })
        .unwrap();
    let right = provider
        .upload(&HostTensorView {
            data: &[3.0, -3.0, 2.0],
            shape: &[1, 3],
        })
        .unwrap();
    for (modulus, expected) in [
        (false, vec![1.0, -1.0, 1.0, -1.0, 1.0, -1.0]),
        (true, vec![1.0, 2.0, -2.0, -1.0, 1.0, 1.0]),
    ] {
        let output = if modulus {
            provider.elem_mod(&left, &right)
        } else {
            provider.elem_rem(&left, &right)
        }
        .await
        .unwrap();
        assert_eq!(output.shape(), vec![2, 3]);
        assert_ne!(output.buffer_id(), left.buffer_id());
        assert_ne!(output.buffer_id(), right.buffer_id());
        assert_eq!(provider.download(&output).await.unwrap().data, expected);
        retirement::release_all([output]);
    }
    retirement::release_all([left]);
    retirement::release_all([right]);
    let source = provider
        .upload(&HostTensorView {
            data: &[7.0, -7.0, 8.0, -8.0, 9.0, -9.0],
            shape: &[2, 3],
        })
        .unwrap();
    let left = provider.transpose(&source).unwrap();
    let right = provider
        .upload(&HostTensorView {
            data: &[3.0],
            shape: &[1, 1],
        })
        .unwrap();
    let result = provider.elem_mod(&left, &right).await.unwrap();
    assert_eq!(result.shape(), vec![3, 2]);
    assert_eq!(
        provider.download(&result).await.unwrap().data,
        vec![1.0, 2.0, 0.0, 2.0, 1.0, 0.0]
    );
    retirement::release_all([result]);
    let source_weak = source.allocation_lease().downgrade();
    left.release().unwrap();
    assert!(source_weak.acquire().is_some());
    assert_eq!(
        provider.download(&source).await.unwrap().data,
        [7.0, -7.0, 8.0, -8.0, 9.0, -9.0]
    );
    retirement::release_all([source]);
    retirement::release_all([right]);
}

#[test]
fn in_process_floating_remainder_executes_real_broadcast_hooks() {
    block_on(exercise(
        &runmat_accelerate::simple_provider::InProcessProvider::new(),
    ));
}

#[cfg(feature = "wgpu")]
#[tokio::test]
#[ignore = "requires an actual compatible WGPU adapter"]
async fn wgpu_floating_remainder_executes_encoded_extremes_and_broadcasts() {
    use runmat_accelerate_api::ProviderPrecision;
    let provider = runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
        runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
    )
    .expect("floating remainder qualification requires a compatible WGPU adapter");
    exercise(provider).await;
    let single = provider.precision() == ProviderPrecision::F32;
    let f32_values = [1.0_f32];
    let f64_values = [1.0_f64];
    let mismatched = runmat_accelerate_api::upload_numeric_protecting(
        provider,
        &runmat_accelerate_api::HostNumericTensorView {
            data: if single {
                runmat_accelerate_api::HostNumericDataView::F64(&f64_values)
            } else {
                runmat_accelerate_api::HostNumericDataView::F32(&f32_values)
            },
            shape: &[1, 1],
            storage: runmat_accelerate_api::GpuTensorStorage::Real,
        },
        &[],
    )
    .expect("native physical mismatch fixture");
    for error in [
        provider
            .elem_mod(&mismatched, &mismatched)
            .await
            .unwrap_err(),
        provider
            .elem_rem(&mismatched, &mismatched)
            .await
            .unwrap_err(),
    ] {
        assert!(error.is_unsupported());
    }
    retirement::release_all([mismatched]);
    let (large, divisor, exact, maximum, tiny, tiny_divisor, tiny_exact) = if single {
        (
            1e9,
            7.0,
            6.0,
            f64::from(f32::MAX),
            f64::from(f32::from_bits(1)),
            11.0,
            10.0,
        )
    } else {
        (1e17, 11.0, 10.0, f64::MAX, f64::from_bits(1), 3.0, 2.0)
    };
    for (a, b, remainder, modulus) in [
        (large, divisor, exact, exact),
        (-large, divisor, -exact, divisor - exact),
        (
            maximum,
            tiny * tiny_divisor,
            tiny * tiny_exact,
            tiny * tiny_exact,
        ),
        (11.0 * tiny, 3.0 * tiny, 2.0 * tiny, 2.0 * tiny),
        (-11.0 * tiny, 3.0 * tiny, -2.0 * tiny, tiny),
        (2.0, 0.1, 0.0, 0.0),
        (3.0 * 0.7, 0.7, 0.0, 0.0),
        (-0.0, 0.0, f64::NAN, -0.0),
        (-1.0, f64::INFINITY, -1.0, f64::INFINITY),
        (f64::INFINITY, 1.0, f64::NAN, f64::NAN),
        (f64::NAN, 0.0, f64::NAN, f64::NAN),
    ] {
        let left = provider
            .upload(&HostTensorView {
                data: &[a],
                shape: &[1, 1],
            })
            .unwrap();
        let right = provider
            .upload(&HostTensorView {
                data: &[b],
                shape: &[1, 1],
            })
            .unwrap();
        for (is_modulus, expected) in [(false, remainder), (true, modulus)] {
            let output = if is_modulus {
                provider.elem_mod(&left, &right).await
            } else {
                provider.elem_rem(&left, &right).await
            }
            .unwrap();
            let mut operation = fixture::cast(if is_modulus { "mod" } else { "rem" }, 0, 2);
            if let runmat_accelerate::fusion::FusionOp::Builtin { inputs, .. } = &mut operation {
                inputs.push(1);
            }
            let mut plan = fixture::plan(vec![operation], 2);
            plan.inputs = vec![0, 1];
            plan.stack_pattern = vec![0, 1];
            plan.group.shape = runmat_accelerate::graph::ShapeInfo::Tensor(vec![Some(1), Some(1)]);
            let fused = runmat_accelerate::fusion_exec::execute_elementwise(
                runmat_accelerate::fusion_exec::FusionExecutionRequest {
                    plan: &plan,
                    inputs: vec![
                        runmat_value::Value::GpuTensor(left.clone()),
                        runmat_value::Value::GpuTensor(right.clone()),
                    ],
                    placement: None,
                    runtime: None,
                },
            )
            .await
            .expect("actual encoded floating remainder fusion");
            let runmat_value::Value::GpuTensor(fused) = fused.final_value else {
                panic!("resident fused remainder result")
            };
            for handle in [&output, &fused] {
                assert_eq!(handle.shape(), vec![1, 1]);
                assert_eq!(handle.device_id(), left.device_id());
                assert_eq!(
                    handle.descriptor().element_type,
                    left.descriptor().element_type
                );
                assert_ne!(handle.buffer_id(), left.buffer_id());
                assert_ne!(handle.buffer_id(), right.buffer_id());
                let actual = provider.download(handle).await.unwrap().data;
                assert_eq!(actual.len(), 1);
                if expected.is_nan() {
                    assert!(actual[0].is_nan());
                } else {
                    assert_eq!(
                        actual[0].to_bits(),
                        expected.to_bits(),
                        "a={a}, b={b}, modulus={is_modulus}"
                    );
                }
            }
            retirement::release_all([output]);
            retirement::release_all([fused]);
        }
        retirement::release_all([left]);
        retirement::release_all([right]);
    }
}

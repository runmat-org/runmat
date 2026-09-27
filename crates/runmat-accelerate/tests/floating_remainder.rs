use futures::executor::block_on;
use runmat_accelerate_api::{AccelProvider, HostTensorView};

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
        assert_eq!(output.shape, vec![2, 3]);
        assert_ne!(output.buffer_id, left.buffer_id);
        assert_ne!(output.buffer_id, right.buffer_id);
        assert_eq!(provider.download(&output).await.unwrap().data, expected);
        provider.free(&output).unwrap();
    }
    provider.free(&left).unwrap();
    provider.free(&right).unwrap();
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
            let actual = provider.download(&output).await.unwrap().data;
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
            provider.free(&output).unwrap();
        }
        provider.free(&left).unwrap();
        provider.free(&right).unwrap();
    }
}

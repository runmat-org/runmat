//! Independent numerical witnesses for the other composed resident routes.

use super::fixture::{assert_values, device, forward, read, upload};
use runmat_accelerate_api::{AccelProvider, GpuHandleProvenance, ThreadProviderGuard};
use runmat_runtime::call_builtin;
use runmat_runtime::execution::{RootedValue, RootedValuesBuilder};
use runmat_value::{Tensor, Value};

#[derive(Clone, Copy, Debug)]
enum Activation {
    Relu,
    Elu,
    Softmax,
}

impl Activation {
    fn layer(self) -> RootedValue {
        match self {
            Self::Relu => call_builtin("reluLayer", &[]),
            Self::Elu => call_builtin("eluLayer", &[Value::Num(1.5)]),
            Self::Softmax => call_builtin("softmaxLayer", &[]),
        }
        .unwrap()
    }

    fn pair(self, first: f64, second: f64) -> [f64; 2] {
        match self {
            Self::Relu => [first.max(0.0), second.max(0.0)],
            Self::Elu => [elu(first), elu(second)],
            Self::Softmax => {
                let peak = first.max(second);
                let a = (first - peak).exp();
                let b = (second - peak).exp();
                [a / (a + b), b / (a + b)]
            }
        }
    }
}

fn elu(value: f64) -> f64 {
    if value >= 0.0 {
        value
    } else {
        1.5 * (value.exp() - 1.0)
    }
}

#[test]
#[ignore = "actual WGPU device required; run separately with RUNMAT_WGPU_FORCE_PRECISION=f32/f64"]
fn nonsquare_lazy_cb_normalization_activation_preserves_layout_and_values() {
    let provider = device();
    let _selected = ThreadProviderGuard::set(Some(provider));
    let source = [1.0, 3.0, 2.0, 5.0, 5.0, 7.0];
    for lazy in [false, true] {
        let backing: &[f64] = if lazy {
            &[1.0, 2.0, 5.0, 3.0, 5.0, 7.0]
        } else {
            &source
        };
        let backing_shape: &[usize] = if lazy { &[3, 2] } else { &[2, 3] };
        let uploaded = upload(
            provider,
            backing,
            backing_shape,
            GpuHandleProvenance::Explicit,
        );
        let input = if lazy {
            provider.transpose(&uploaded).unwrap()
        } else {
            uploaded
        };
        assert_eq!(input.shape(), [2, 3]);
        for activation in [Activation::Relu, Activation::Elu, Activation::Softmax] {
            let mut layers = RootedValuesBuilder::with_capacity(3);
            layers.push_owned(call_builtin("featureInputLayer", &[Value::Num(2.0)]).unwrap());
            layers.push_owned(call_builtin("layerNormalizationLayer", &[]).unwrap());
            layers.push_owned(activation.layer());
            let layers = layers.finish().unwrap();
            let output = forward(
                provider,
                &input,
                "CB",
                "CB",
                layers,
                &[2, 3],
                GpuHandleProvenance::Explicit,
            );
            let mut expected = Vec::new();
            for spread in [1.0_f64, 1.5, 1.0] {
                let normalized = spread / (spread * spread + 1.0e-5).sqrt();
                expected.extend(activation.pair(-normalized, normalized));
            }
            assert_values(provider, &output, &expected);
            assert_eq!(
                read(provider, &input),
                source,
                "input was changed by {activation:?}"
            );
        }
    }
}

#[test]
#[ignore = "actual WGPU device required; run separately with RUNMAT_WGPU_FORCE_PRECISION=f32/f64"]
fn cbt_pool_dense_activation_keeps_exact_axes_and_consumed_parameter_intent() {
    let provider = device();
    let _selected = ThreadProviderGuard::set(Some(provider));
    // CBT storage is channel-major, then batch, then time. A four-step mean
    // gives channels (batch + 1.5, 2*batch - 0.75) for each batch.
    let mut source = Vec::new();
    for time in 0..4 {
        for batch in 0..3 {
            source.push(batch as f64 + time as f64);
            source.push(2.0 * batch as f64 - 0.5 * time as f64);
        }
    }
    for explicit_parameters in [false, true] {
        let input = upload(
            provider,
            &source,
            &[2, 3, 4],
            if explicit_parameters {
                GpuHandleProvenance::Automatic
            } else {
                GpuHandleProvenance::Explicit
            },
        );
        let resident = if explicit_parameters {
            Some((
                upload(
                    provider,
                    &[1.0, -1.0, -2.0, 2.0],
                    &[2, 2],
                    GpuHandleProvenance::Explicit,
                ),
                upload(
                    provider,
                    &[-0.25, 0.25],
                    &[2, 1],
                    GpuHandleProvenance::Explicit,
                ),
            ))
        } else {
            None
        };
        for activation in [Activation::Relu, Activation::Elu, Activation::Softmax] {
            let weights = resident.as_ref().map_or_else(
                || Value::Tensor(Tensor::new(vec![1.0, -1.0, -2.0, 2.0], vec![2, 2]).unwrap()),
                |(weights, _)| Value::GpuTensor(weights.clone()),
            );
            let bias = resident.as_ref().map_or_else(
                || Value::Tensor(Tensor::new(vec![-0.25, 0.25], vec![2, 1]).unwrap()),
                |(_, bias)| Value::GpuTensor(bias.clone()),
            );
            let mut layers = RootedValuesBuilder::with_capacity(4);
            layers.push_owned(call_builtin("sequenceInputLayer", &[Value::Num(2.0)]).unwrap());
            layers.push_owned(call_builtin("globalAveragePooling1dLayer", &[]).unwrap());
            layers.push_owned(
                call_builtin(
                    "fullyConnectedLayer",
                    &[
                        Value::Num(2.0),
                        Value::String("Weights".into()),
                        weights,
                        Value::String("Bias".into()),
                        bias,
                    ],
                )
                .unwrap(),
            );
            layers.push_owned(activation.layer());
            let layers = layers.finish().unwrap();
            let output = forward(
                provider,
                &input,
                "CBT",
                "CB",
                layers,
                &[2, 3],
                GpuHandleProvenance::Explicit,
            );
            let mut expected = Vec::new();
            for batch in 0..3 {
                let first = batch as f64 + 1.5;
                let second = 2.0 * batch as f64 - 0.75;
                let logit = first - 2.0 * second - 0.25;
                expected.extend(activation.pair(logit, -logit));
            }
            assert_values(provider, &output, &expected);
            assert_eq!(
                read(provider, &input),
                source,
                "input was changed by {activation:?}"
            );
            if let Some((weights, bias)) = &resident {
                assert_values(provider, weights, &[1.0, -1.0, -2.0, 2.0]);
                assert_values(provider, bias, &[-0.25, 0.25]);
            }
        }
    }
}

use futures::{executor::block_on, task::noop_waker};
use runmat_accelerate::simple_provider::InProcessProvider;
use runmat_accelerate_api::{
    AccelProvider, AccelProviderFuture, GpuTensorHandle, HostTensorView, ProviderPrecision,
    ThreadProviderGuard,
};
use runmat_value::{CharArray, Value};
use std::{
    future::Future,
    sync::{Mutex, OnceLock},
    task::{Context, Poll},
};

#[derive(Clone, Copy, Default)]
enum Fault {
    #[default]
    Pending,
    Failed,
    Alias,
    MissingStorage,
}
#[derive(Default)]
struct State {
    fault: Fault,
    calls: usize,
    freed: Vec<u64>,
}
struct Provider {
    inner: InProcessProvider,
    state: Mutex<State>,
}
impl Provider {
    fn binary<'a>(
        &'a self,
        left: &'a GpuTensorHandle,
        right: &'a GpuTensorHandle,
    ) -> AccelProviderFuture<'a, GpuTensorHandle> {
        Box::pin(async move {
            let fault = {
                let mut state = self.state.lock().unwrap();
                state.calls += 1;
                state.fault
            };
            runmat_accelerate_api::set_handle_logical(left, true);
            runmat_accelerate_api::set_handle_logical(right, true);
            match fault {
                Fault::Pending => std::future::pending().await,
                Fault::Failed => anyhow::bail!("injected remainder failure"),
                Fault::Alias => Ok(right.clone()),
                Fault::MissingStorage => {
                    let mut output = self.inner.upload_integer(
                        &runmat_accelerate_api::HostIntegerTensorView {
                            data: runmat_accelerate_api::HostIntegerDataView::I64(&[1, 1]),
                            shape: &[1, 2],
                        },
                    )?;
                    output.descriptor.storage = None;
                    Ok(output)
                }
            }
        })
    }
    fn invoke<'a>(
        &'a self,
        input: &'a GpuTensorHandle,
    ) -> AccelProviderFuture<'a, GpuTensorHandle> {
        Box::pin(async move {
            let fault = {
                let mut state = self.state.lock().unwrap();
                state.calls += 1;
                state.fault
            };
            runmat_accelerate_api::set_handle_logical(input, true);
            match fault {
                Fault::Pending => std::future::pending().await,
                Fault::Failed => anyhow::bail!("injected rounding failure"),
                Fault::Alias => Ok(input.clone()),
                Fault::MissingStorage => {
                    let mut output = self.inner.upload(&HostTensorView {
                        data: &[1.0, 4.0],
                        shape: &[1, 2],
                    })?;
                    output.descriptor.storage = None;
                    Ok(output)
                }
            }
        })
    }
}
impl AccelProvider for Provider {
    fn elem_mod<'a>(
        &'a self,
        left: &'a GpuTensorHandle,
        right: &'a GpuTensorHandle,
    ) -> AccelProviderFuture<'a, GpuTensorHandle> {
        self.binary(left, right)
    }
    fn elem_rem<'a>(
        &'a self,
        left: &'a GpuTensorHandle,
        right: &'a GpuTensorHandle,
    ) -> AccelProviderFuture<'a, GpuTensorHandle> {
        self.binary(left, right)
    }
    fn upload(&self, view: &HostTensorView) -> anyhow::Result<GpuTensorHandle> {
        self.inner.upload(view)
    }
    fn download<'a>(
        &'a self,
        handle: &'a GpuTensorHandle,
    ) -> runmat_accelerate_api::AccelDownloadFuture<'a> {
        self.inner.download(handle)
    }
    fn device_id(&self) -> u32 {
        self.inner.device_id()
    }
    fn device_info(&self) -> String {
        self.inner.device_info()
    }
    fn precision(&self) -> ProviderPrecision {
        self.inner.precision()
    }
    fn free(&self, handle: &GpuTensorHandle) -> anyhow::Result<()> {
        self.state.lock().unwrap().freed.push(handle.buffer_id);
        self.inner.free(handle)
    }
    fn unary_ceil<'a>(
        &'a self,
        input: &'a GpuTensorHandle,
    ) -> AccelProviderFuture<'a, GpuTensorHandle> {
        self.invoke(input)
    }
    fn unary_floor<'a>(
        &'a self,
        input: &'a GpuTensorHandle,
    ) -> AccelProviderFuture<'a, GpuTensorHandle> {
        self.invoke(input)
    }
    fn unary_fix<'a>(
        &'a self,
        input: &'a GpuTensorHandle,
    ) -> AccelProviderFuture<'a, GpuTensorHandle> {
        self.invoke(input)
    }
    fn unary_round<'a>(
        &'a self,
        input: &'a GpuTensorHandle,
    ) -> AccelProviderFuture<'a, GpuTensorHandle> {
        self.invoke(input)
    }
    fn round_digits<'a>(
        &'a self,
        input: &'a GpuTensorHandle,
        _digits: i32,
        _significant: bool,
    ) -> AccelProviderFuture<'a, GpuTensorHandle> {
        self.invoke(input)
    }
}

#[test]
fn rounding_dispatch_preserves_borrowed_metadata_and_releases_invalid_outputs() {
    static PROVIDER: OnceLock<Provider> = OnceLock::new();
    let provider = PROVIDER.get_or_init(|| Provider {
        inner: InProcessProvider::new(),
        state: Mutex::default(),
    });
    let _provider = ThreadProviderGuard::set(Some(provider));
    exercise_remainder_lifetimes(provider);
    for (name, controls) in [
        ("ceil", vec![]),
        ("floor", vec![]),
        ("fix", vec![]),
        ("round", vec![]),
        ("round", vec![Value::Num(2.0)]),
        (
            "round",
            vec![Value::Num(2.0), Value::String("significant".into())],
        ),
    ] {
        for fault in [
            Fault::Pending,
            Fault::Failed,
            Fault::Alias,
            Fault::MissingStorage,
        ] {
            *provider.state.lock().unwrap() = State {
                fault,
                ..State::default()
            };
            let input = provider
                .inner
                .upload(&HostTensorView {
                    data: &[1.25, 4.75],
                    shape: &[1, 2],
                })
                .unwrap();
            let mut args = vec![Value::GpuTensor(input.clone())];
            args.extend(controls.clone());
            let error = block_on(runmat_runtime::call_builtin_async_with_outputs(
                name, &args, 2,
            ))
            .unwrap_err();
            assert!(error.message().contains("output"));
            assert_eq!(provider.state.lock().unwrap().calls, 0);
            drop(runmat_runtime::call_builtin_async_with_outputs(
                name, &args, 1,
            ));
            assert_eq!(provider.state.lock().unwrap().calls, 0);
            let mut call = Box::pin(runmat_runtime::call_builtin_async_with_outputs(
                name, &args, 1,
            ));
            let result = call.as_mut().poll(&mut Context::from_waker(&noop_waker()));
            if matches!(fault, Fault::Pending) {
                assert!(matches!(result, Poll::Pending), "{name}");
            } else {
                assert!(matches!(result, Poll::Ready(Err(_))), "{name}");
            }
            drop(call);
            assert!(!runmat_accelerate_api::handle_is_logical(&input), "{name}");
            let state = provider.state.lock().unwrap();
            assert_eq!(
                state.calls, 1,
                "{name}: no retry after a real provider failure"
            );
            assert!(!state.freed.contains(&input.buffer_id), "{name}");
            assert_eq!(
                state.freed.len(),
                usize::from(matches!(fault, Fault::MissingStorage)),
                "{name}"
            );
            drop(state);
            assert_eq!(
                block_on(provider.inner.download(&input)).unwrap().data,
                vec![1.25, 4.75]
            );
            provider.inner.free(&input).unwrap();
        }
    }
}

fn exercise_remainder_lifetimes(provider: &'static Provider) {
    use runmat_accelerate_api::{HostIntegerDataView, HostIntegerTensorView};
    for name in ["mod", "rem"] {
        for fault in [
            Fault::Pending,
            Fault::Failed,
            Fault::Alias,
            Fault::MissingStorage,
        ] {
            *provider.state.lock().unwrap() = State {
                fault,
                ..State::default()
            };
            let left = provider
                .inner
                .upload_integer(&HostIntegerTensorView {
                    data: HostIntegerDataView::I64(&[7, -7]),
                    shape: &[1, 2],
                })
                .unwrap();
            let right = provider
                .inner
                .upload_integer(&HostIntegerTensorView {
                    data: HostIntegerDataView::I64(&[3, 3]),
                    shape: &[1, 2],
                })
                .unwrap();
            let args = [
                Value::GpuTensor(left.clone()),
                Value::GpuTensor(right.clone()),
            ];
            let mut call = Box::pin(runmat_runtime::call_builtin_async_with_outputs(
                name, &args, 1,
            ));
            let result = call.as_mut().poll(&mut Context::from_waker(&noop_waker()));
            if matches!(fault, Fault::Pending) {
                assert!(matches!(result, Poll::Pending), "{name}");
            } else {
                assert!(matches!(result, Poll::Ready(Err(_))), "{name}");
            }
            drop(call);
            assert!(!runmat_accelerate_api::handle_is_logical(&left));
            assert!(!runmat_accelerate_api::handle_is_logical(&right));
            let state = provider.state.lock().unwrap();
            assert_eq!(state.calls, 1);
            assert!(!state.freed.contains(&left.buffer_id));
            assert!(!state.freed.contains(&right.buffer_id));
            assert_eq!(
                state.freed.len(),
                usize::from(matches!(fault, Fault::MissingStorage))
            );
            drop(state);
            block_on(provider.inner.download_integer(&left)).unwrap();
            block_on(provider.inner.download_integer(&right)).unwrap();
            provider.inner.free(&left).unwrap();
            provider.inner.free(&right).unwrap();
        }
    }
}

#[test]
fn rounding_characters_retain_pages_and_semantic_element_order() {
    let input =
        CharArray::new_with_shape(vec!['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H'], vec![2, 2, 2])
            .unwrap();
    for name in ["ceil", "floor", "fix", "round"] {
        let Value::Tensor(output) = block_on(runmat_runtime::call_builtin_async_with_outputs(
            name,
            &[Value::CharArray(input.clone())],
            1,
        ))
        .unwrap() else {
            panic!("{name}: numeric character output")
        };
        assert_eq!(output.shape, input.shape);
        assert_eq!(
            output.as_f64_slice().unwrap(),
            input
                .to_column_major()
                .into_iter()
                .map(|ch| ch as u32 as f64)
                .collect::<Vec<_>>()
        );
    }
}

#[test]
fn round_scalar_ties_integral_extremes_and_signed_zero_have_independent_oracles() {
    let large = 4_503_599_627_370_497.0;
    for (input, expected) in [
        (0.5, 1.0_f64),
        (-0.5, -1.0),
        (2.5, 3.0),
        (-2.5, -3.0),
        (large, large),
        (-large, -large),
        (-0.0, -0.0),
        (-0.25, -0.0),
        (f64::INFINITY, f64::INFINITY),
        (f64::NEG_INFINITY, f64::NEG_INFINITY),
    ] {
        let Value::Num(actual) = block_on(runmat_runtime::call_builtin_async_with_outputs(
            "round",
            &[Value::Num(input)],
            1,
        ))
        .unwrap() else {
            panic!("round scalar")
        };
        assert_eq!(actual.to_bits(), expected.to_bits());
    }
    let Value::Num(actual) = block_on(runmat_runtime::call_builtin_async_with_outputs(
        "round",
        &[Value::Num(f64::NAN)],
        1,
    ))
    .unwrap() else {
        panic!("round NaN")
    };
    assert!(actual.is_nan());
}

#[test]
fn remainder_character_pages_are_not_scalarized_or_reordered() {
    for shape in [vec![1, 1, 2], vec![2, 2, 2]] {
        let count = shape.iter().product();
        let input = CharArray::new_with_shape(
            (0..count)
                .map(|index| char::from_u32(65 + index as u32).unwrap())
                .collect(),
            shape.clone(),
        )
        .unwrap();
        for name in ["mod", "rem"] {
            let Value::Tensor(output) = block_on(runmat_runtime::call_builtin_async_with_outputs(
                name,
                &[Value::CharArray(input.clone()), Value::Num(7.0)],
                1,
            ))
            .unwrap() else {
                panic!("{name}: pages")
            };
            assert_eq!(output.shape, shape);
            assert_eq!(
                output.as_f64_slice().unwrap(),
                input
                    .to_column_major()
                    .into_iter()
                    .map(|ch| (ch as u32 % 7) as f64)
                    .collect::<Vec<_>>()
            );
        }
    }
}

#[test]
fn finite_remainders_survive_overflowing_quotients() {
    for name in ["mod", "rem"] {
        for left in [f64::MAX, -f64::MAX] {
            for right in [0.5, -0.5, f64::MIN_POSITIVE] {
                let Value::Num(result) = block_on(runmat_runtime::call_builtin_async_with_outputs(
                    name,
                    &[Value::Num(left), Value::Num(right)],
                    1,
                ))
                .unwrap() else {
                    panic!("scalar remainder")
                };
                // MAX has no fractional bits and is a multiple of every
                // represented power of two below one; the exact remainder is 0.
                assert_eq!(result.to_bits(), 0.0_f64.to_bits(), "{name}");
            }
        }
        let left =
            runmat_value::Tensor::from_f32(vec![f32::MAX, -f32::MAX], vec![1, 1, 2]).unwrap();
        let right = runmat_value::Tensor::from_f32(vec![0.5], vec![1, 1]).unwrap();
        let Value::Tensor(result) = block_on(runmat_runtime::call_builtin_async_with_outputs(
            name,
            &[Value::Tensor(left), Value::Tensor(right)],
            1,
        ))
        .unwrap() else {
            panic!("single pages")
        };
        assert_eq!(result.shape, vec![1, 1, 2]);
        assert_eq!(result.as_f32_slice().unwrap(), &[0.0, 0.0]);
    }
    for (name, left, right, expected) in [
        ("mod", -5.5, 2.0, 0.5_f64),
        ("mod", 5.5, -2.0, -0.5),
        ("rem", -5.5, 2.0, -1.5),
        ("rem", 5.5, -2.0, 1.5),
    ] {
        let Value::Num(result) = block_on(runmat_runtime::call_builtin_async_with_outputs(
            name,
            &[Value::Num(left), Value::Num(right)],
            1,
        ))
        .unwrap() else {
            panic!("scalar remainder")
        };
        assert_eq!(result, expected);
    }
}

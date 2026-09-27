use super::*;
use crate::builtins::common::test_support;
use futures::{executor::block_on, task::noop_waker};
use runmat_accelerate::simple_provider::InProcessProvider;
use runmat_accelerate_api::{
    AccelDownloadFuture, AccelProvider, AccelProviderFuture, HostTensorView, ThreadProviderGuard,
};
use std::{
    sync::{Mutex, OnceLock},
    task::{Context, Poll},
};

fn atan_complex_template_cases(provider: &'static dyn AccelProvider, single: bool) {
    let _extensions = crate::compatibility::push_runmat_extensions_enabled(true);
    let prototype_tensor = if single {
        runmat_value::ComplexTensor::from_f32(vec![(0.0, 1.0)], vec![1, 1]).unwrap()
    } else {
        runmat_value::ComplexTensor::new(vec![(0.0, 1.0)], vec![1, 1]).unwrap()
    };
    let prototype = gpu_helpers::upload_complex_tensor(provider, &prototype_tensor).unwrap();
    for shape in [vec![1, 1], vec![1, 2, 1], vec![0, 2, 3]] {
        let len = shape.iter().product();
        for complex in [false, true] {
            let input = if complex {
                if single {
                    Value::ComplexTensor(
                        runmat_value::ComplexTensor::from_f32(
                            vec![(0.0, 0.25); len],
                            shape.clone(),
                        )
                        .unwrap(),
                    )
                } else {
                    Value::ComplexTensor(
                        runmat_value::ComplexTensor::new(vec![(0.0, 0.25); len], shape.clone())
                            .unwrap(),
                    )
                }
            } else if single {
                Value::Tensor(
                    runmat_value::Tensor::from_f32(vec![0.0; len], shape.clone()).unwrap(),
                )
            } else {
                Value::Tensor(runmat_value::Tensor::new(vec![0.0; len], shape.clone()).unwrap())
            };
            let expected = block_on(crate::dispatcher::call_builtin_async(
                "atan",
                &[
                    input.clone(),
                    Value::String("like".into()),
                    Value::Complex(0.0, 1.0),
                ],
            ))
            .unwrap();
            let expected_values = match expected {
                Value::Complex(real, imag) => vec![(real, imag)],
                Value::ComplexTensor(tensor) => tensor.materialize_f64(),
                other => panic!("complex host oracle: {other:?}"),
            };
            for resident in [false, true] {
                let source = if resident {
                    Some(match &input {
                        Value::Tensor(tensor) => {
                            gpu_helpers::upload_tensor(provider, tensor).unwrap()
                        }
                        Value::ComplexTensor(tensor) => {
                            gpu_helpers::upload_complex_tensor(provider, tensor).unwrap()
                        }
                        _ => unreachable!(),
                    })
                } else {
                    None
                };
                let value = source
                    .as_ref()
                    .map(|handle| Value::GpuTensor(handle.clone()))
                    .unwrap_or_else(|| input.clone());
                let arguments = [
                    value,
                    Value::String("like".into()),
                    Value::GpuTensor(prototype.clone()),
                ];
                let result =
                    block_on(crate::dispatcher::call_builtin_async("atan", &arguments)).unwrap();
                let Value::GpuTensor(output) = result else {
                    panic!("resident complex result")
                };
                assert_eq!(output.device_id, prototype.device_id);
                assert_eq!(output.shape, shape);
                assert_eq!(
                    output.descriptor.storage,
                    Some(runmat_accelerate_api::GpuTensorStorage::ComplexInterleaved)
                );
                assert_eq!(
                    runmat_accelerate_api::handle_precision(&output),
                    Some(if single {
                        runmat_accelerate_api::ProviderPrecision::F32
                    } else {
                        runmat_accelerate_api::ProviderPrecision::F64
                    })
                );
                let actual = block_on(gpu_helpers::gather_value_async(&Value::GpuTensor(
                    output.clone(),
                )))
                .unwrap();
                let actual_values = match actual {
                    Value::Complex(real, imag) => vec![(real, imag)],
                    Value::ComplexTensor(tensor) => tensor.materialize_f64(),
                    other => panic!("complex device result: {other:?}"),
                };
                assert_eq!(actual_values, expected_values);
                provider.free(&output).unwrap();
                if let Some(source) = source {
                    assert!(block_on(gpu_helpers::gather_value_async(&Value::GpuTensor(
                        source.clone()
                    )))
                    .is_ok());
                    provider.free(&source).unwrap();
                }
            }
        }
    }
    assert!(block_on(gpu_helpers::gather_value_async(&Value::GpuTensor(
        prototype.clone()
    )))
    .is_ok());
    provider.free(&prototype).unwrap();
}

#[test]
fn atan_complex_gpu_template_preserves_double_class_storage_owner_empty_and_nd() {
    test_support::with_test_provider(|provider| atan_complex_template_cases(provider, false));
}

#[test]
fn atan_complex_gpu_template_preserves_single_class_storage_owner_empty_and_nd() {
    test_support::with_f32_test_provider(|provider| atan_complex_template_cases(provider, true));
}

#[derive(Clone, Copy, Default)]
enum Download {
    #[default]
    Ready,
    Failure,
    Pending,
}
#[derive(Default)]
struct State {
    mode: Download,
    outputs: Vec<GpuTensorHandle>,
    freed: Vec<u64>,
    downloads: usize,
    hooks: usize,
}
struct Provider {
    inner: InProcessProvider,
    state: Mutex<State>,
}
impl Provider {
    fn track(&self, output: GpuTensorHandle) -> GpuTensorHandle {
        let mut state = self.state.lock().unwrap();
        state.hooks += 1;
        state.outputs.push(output.clone());
        output
    }
}
impl AccelProvider for Provider {
    fn upload(&self, host: &HostTensorView) -> anyhow::Result<GpuTensorHandle> {
        Ok(self.track(self.inner.upload(host)?))
    }
    fn device_id(&self) -> u32 {
        self.inner.device_id()
    }
    fn device_info(&self) -> String {
        self.inner.device_info()
    }
    fn precision(&self) -> runmat_accelerate_api::ProviderPrecision {
        self.inner.precision()
    }
    fn download<'a>(&'a self, handle: &'a GpuTensorHandle) -> AccelDownloadFuture<'a> {
        Box::pin(async move {
            let mode = {
                let mut state = self.state.lock().unwrap();
                state.downloads += 1;
                state.mode
            };
            match mode {
                Download::Ready => self.inner.download(handle).await,
                Download::Failure => anyhow::bail!("injected adaptation download failure"),
                Download::Pending => std::future::pending().await,
            }
        })
    }
    fn free(&self, handle: &GpuTensorHandle) -> anyhow::Result<()> {
        self.state.lock().unwrap().freed.push(handle.buffer_id);
        self.inner.free(handle)
    }
    fn unary_sin<'a>(
        &'a self,
        input: &'a GpuTensorHandle,
    ) -> AccelProviderFuture<'a, GpuTensorHandle> {
        Box::pin(async move { Ok(self.track(self.inner.unary_sin(input).await?)) })
    }
    fn unary_cos<'a>(
        &'a self,
        input: &'a GpuTensorHandle,
    ) -> AccelProviderFuture<'a, GpuTensorHandle> {
        Box::pin(async move { Ok(self.track(self.inner.unary_cos(input).await?)) })
    }
    fn unary_tan<'a>(
        &'a self,
        input: &'a GpuTensorHandle,
    ) -> AccelProviderFuture<'a, GpuTensorHandle> {
        Box::pin(async move { Ok(self.track(self.inner.unary_tan(input).await?)) })
    }
    fn unary_atan<'a>(
        &'a self,
        input: &'a GpuTensorHandle,
    ) -> AccelProviderFuture<'a, GpuTensorHandle> {
        Box::pin(async move { Ok(self.track(self.inner.unary_atan(input).await?)) })
    }
}
fn fixture(mode: Download) -> (&'static Provider, GpuTensorHandle, GpuTensorHandle) {
    static PROVIDER: OnceLock<Provider> = OnceLock::new();
    let provider = PROVIDER.get_or_init(|| Provider {
        inner: InProcessProvider::new(),
        state: Mutex::default(),
    });
    *provider.state.lock().unwrap() = State {
        mode,
        ..State::default()
    };
    let source = provider
        .inner
        .upload(&HostTensorView {
            data: &[0.0, 0.5],
            shape: &[1, 2],
        })
        .unwrap();
    let prototype = provider
        .inner
        .upload(&HostTensorView {
            data: &[1.0],
            shape: &[1, 1],
        })
        .unwrap();
    (provider, source, prototype)
}
fn args(source: &GpuTensorHandle, prototype: Value) -> [Value; 3] {
    [
        Value::GpuTensor(source.clone()),
        Value::String("like".into()),
        prototype,
    ]
}
fn borrowed_live(provider: &Provider, source: &GpuTensorHandle, prototype: &GpuTensorHandle) {
    for input in [source, prototype] {
        assert!(!provider
            .state
            .lock()
            .unwrap()
            .freed
            .contains(&input.buffer_id));
        assert!(block_on(provider.inner.download(input)).is_ok());
        provider.inner.free(input).unwrap();
    }
}

#[test]
fn trigonometric_like_actual_calls_free_replaced_intermediates_and_retain_published_results() {
    let _serial = test_support::accel_test_lock();
    let _extensions = crate::compatibility::push_runmat_extensions_enabled(true);
    for name in ["sin", "cos", "tan", "atan"] {
        for resident in [false, true] {
            let (provider, source, prototype) = fixture(Download::Ready);
            let _selected = ThreadProviderGuard::set(Some(provider));
            let prototype_value = if resident {
                Value::GpuTensor(prototype.clone())
            } else {
                Value::Num(0.0)
            };
            let result = block_on(crate::dispatcher::call_builtin_async(
                name,
                &args(&source, prototype_value),
            ))
            .unwrap();
            let state = provider.state.lock().unwrap();
            assert_eq!(state.outputs.len(), 1);
            assert_eq!(state.freed.len(), usize::from(!resident));
            drop(state);
            let gathered = block_on(gpu_helpers::gather_value_async(&result)).unwrap();
            let Value::Tensor(tensor) = gathered else {
                panic!("numeric result")
            };
            for (&value, input) in tensor.materialize_f64().iter().zip([0.0_f64, 0.5]) {
                let expected = match name {
                    "sin" => input.sin(),
                    "cos" => input.cos(),
                    "tan" => input.tan(),
                    "atan" => input.atan(),
                    _ => unreachable!(),
                };
                assert!((value - expected).abs() < 1e-12);
            }
            if let Value::GpuTensor(result) = result {
                provider.free(&result).unwrap();
            }
            borrowed_live(provider, &source, &prototype);
        }
    }
}

#[test]
fn trigonometric_like_actual_call_error_and_polled_drop_release_intermediates() {
    let _serial = test_support::accel_test_lock();
    let _extensions = crate::compatibility::push_runmat_extensions_enabled(true);
    for name in ["sin", "cos", "tan", "atan"] {
        for mode in [Download::Failure, Download::Pending] {
            let (provider, source, prototype) = fixture(mode);
            let _selected = ThreadProviderGuard::set(Some(provider));
            let inputs = args(&source, Value::Num(0.0));
            let mut call = Box::pin(crate::dispatcher::call_builtin_async(name, &inputs));
            if matches!(mode, Download::Pending) {
                assert!(matches!(
                    call.as_mut().poll(&mut Context::from_waker(&noop_waker())),
                    Poll::Pending
                ));
                drop(call);
            } else {
                assert!(block_on(call).is_err());
            }
            let state = provider.state.lock().unwrap();
            assert_eq!(state.hooks, 1);
            assert_eq!(state.downloads, 1);
            assert_eq!(state.freed, vec![state.outputs[0].buffer_id]);
            drop(state);
            borrowed_live(provider, &source, &prototype);
        }
    }
}

#[test]
fn trigonometric_like_invalid_prototype_rejects_before_provider_work() {
    let _serial = test_support::accel_test_lock();
    let _extensions = crate::compatibility::push_runmat_extensions_enabled(true);
    for name in ["sin", "cos", "tan", "atan"] {
        let (provider, source, mut prototype) = fixture(Download::Ready);
        let _selected = ThreadProviderGuard::set(Some(provider));
        let original = prototype.clone();
        prototype.descriptor.storage = None;
        for prototype in [Value::String("invalid".into()), Value::GpuTensor(prototype)] {
            assert!(block_on(crate::dispatcher::call_builtin_async(
                name,
                &args(&source, prototype)
            ))
            .is_err());
        }
        let state = provider.state.lock().unwrap();
        assert_eq!(state.hooks, 0);
        assert_eq!(state.downloads, 0);
        assert!(state.freed.is_empty());
        drop(state);
        borrowed_live(provider, &source, &original);
    }
}

#[test]
fn atan_complex_gpu_template_download_failure_or_abandonment_releases_real_intermediate() {
    let _serial = test_support::accel_test_lock();
    let _extensions = crate::compatibility::push_runmat_extensions_enabled(true);
    for mode in [Download::Failure, Download::Pending] {
        let (provider, source, spare) = fixture(mode);
        let _selected = ThreadProviderGuard::set(Some(provider));
        let prototype = gpu_helpers::upload_complex_tensor(
            &provider.inner,
            &runmat_value::ComplexTensor::new(vec![(0.0, 1.0)], vec![1, 1]).unwrap(),
        )
        .unwrap();
        let inputs = args(&source, Value::GpuTensor(prototype.clone()));
        let mut call = Box::pin(crate::dispatcher::call_builtin_async_with_outputs(
            "atan", &inputs, 0,
        ));
        if matches!(mode, Download::Pending) {
            assert!(matches!(
                call.as_mut().poll(&mut Context::from_waker(&noop_waker())),
                Poll::Pending
            ));
            drop(call);
        } else {
            assert!(block_on(call).is_err());
        }
        let state = provider.state.lock().unwrap();
        assert_eq!(state.hooks, 1);
        assert_eq!(state.downloads, 1);
        assert_eq!(state.freed, vec![state.outputs[0].buffer_id]);
        assert!(!state.freed.contains(&prototype.buffer_id));
        drop(state);
        provider.inner.free(&prototype).unwrap();
        borrowed_live(provider, &source, &spare);
    }
}

use super::*;
use crate::builtins::common::test_support;
use crate::builtins::math::trigonometry::test_owner;
use futures::{executor::block_on, task::noop_waker};
use runmat_accelerate_api::{allocation::WeakGpuTensorHandle, test_support::RetirementProbe};
use runmat_accelerate_api::{
    AccelDownloadFuture, AccelProvider, AccelProviderFuture, HostTensorView, ThreadProviderGuard,
};
use std::{
    sync::Mutex,
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
            let expected_values = match expected.value() {
                Value::Complex(real, imag) => vec![(*real, *imag)],
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
                let Value::GpuTensor(output) = result.value() else {
                    panic!("resident complex result")
                };
                assert_eq!(output.device_id(), prototype.device_id());
                assert_eq!(output.shape(), shape);
                assert_eq!(
                    output.descriptor().storage,
                    Some(runmat_accelerate_api::GpuTensorStorage::ComplexInterleaved)
                );
                assert_eq!(
                    runmat_accelerate_api::handle_precision(output),
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
                result.with_owned(|value| {
                    let Value::GpuTensor(output) = value else {
                        unreachable!()
                    };
                    output.release().unwrap();
                });
                if let Some(source) = source {
                    assert!(block_on(gpu_helpers::gather_value_async(&Value::GpuTensor(
                        source.clone()
                    )))
                    .is_ok());
                    source.release().unwrap();
                }
            }
        }
    }
    assert!(block_on(gpu_helpers::gather_value_async(&Value::GpuTensor(
        prototype.clone()
    )))
    .is_ok());
    prototype.release().unwrap();
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
    outputs: Vec<RetirementProbe>,
    downloads: usize,
    hooks: usize,
    hook_alias: Option<WeakGpuTensorHandle>,
    restore_alias: Option<WeakGpuTensorHandle>,
    unsupported_hook: bool,
}
struct Provider {
    inner: test_owner::Owner,
    state: Mutex<State>,
}
impl Provider {
    fn hook_override(&self) -> Option<anyhow::Result<GpuTensorHandle>> {
        let mut state = self.state.lock().unwrap();
        if let Some(handle) = state.hook_alias.clone() {
            state.hooks += 1;
            return Some(handle.acquire().map_err(anyhow::Error::from));
        }
        if state.unsupported_hook {
            state.hooks += 1;
            return Some(Err(runmat_accelerate_api::unsupported_provider_operation(
                "like test",
            )));
        }
        None
    }
    fn track(&self, output: GpuTensorHandle) -> GpuTensorHandle {
        let mut state = self.state.lock().unwrap();
        state.hooks += 1;
        state.outputs.push(self.inner.probe(&output));
        output
    }
}
impl AccelProvider for Provider {
    fn upload_numeric_raw(
        &self,
        host: &runmat_accelerate_api::HostNumericTensorView,
    ) -> runmat_accelerate_api::ProviderExecutionResult<GpuTensorHandle> {
        if let Some(handle) = self.state.lock().unwrap().restore_alias.clone() {
            return Ok(handle.acquire()?);
        }
        Ok(self.track(self.inner.upload_numeric(host)?))
    }
    fn upload(&self, host: &HostTensorView) -> anyhow::Result<GpuTensorHandle> {
        if let Some(handle) = self.state.lock().unwrap().restore_alias.clone() {
            return Ok(handle.acquire()?);
        }
        Ok(self.track(self.inner.upload(host)?))
    }
    fn device_id(&self) -> u32 {
        self.inner.device_id()
    }
    fn device_info(&self) -> String {
        "trigonometric like owning fixture".into()
    }
    fn precision(&self) -> runmat_accelerate_api::ProviderPrecision {
        runmat_accelerate_api::ProviderPrecision::F64
    }
    fn download<'a>(&'a self, handle: &'a GpuTensorHandle) -> AccelDownloadFuture<'a> {
        Box::pin(async move {
            let mode = {
                let mut state = self.state.lock().unwrap();
                state.downloads += 1;
                state.mode
            };
            match mode {
                Download::Ready => self.inner.download(handle),
                Download::Failure => anyhow::bail!("injected adaptation download failure"),
                Download::Pending => std::future::pending().await,
            }
        })
    }
    fn unary_sin<'a>(
        &'a self,
        input: &'a GpuTensorHandle,
    ) -> AccelProviderFuture<'a, GpuTensorHandle> {
        Box::pin(async move {
            if let Some(result) = self.hook_override() {
                return result;
            }
            let host = self.inner.download(input)?;
            assert_eq!(host.storage, runmat_accelerate_api::GpuTensorStorage::Real);
            let values: Vec<_> = host.data.into_iter().map(f64::sin).collect();
            Ok(self.track(self.inner.upload(&HostTensorView {
                data: &values,
                shape: &host.shape,
            })?))
        })
    }
    fn unary_cos<'a>(
        &'a self,
        input: &'a GpuTensorHandle,
    ) -> AccelProviderFuture<'a, GpuTensorHandle> {
        Box::pin(async move {
            if let Some(result) = self.hook_override() {
                return result;
            }
            let host = self.inner.download(input)?;
            assert_eq!(host.storage, runmat_accelerate_api::GpuTensorStorage::Real);
            let values: Vec<_> = host.data.into_iter().map(f64::cos).collect();
            Ok(self.track(self.inner.upload(&HostTensorView {
                data: &values,
                shape: &host.shape,
            })?))
        })
    }
    fn unary_tan<'a>(
        &'a self,
        input: &'a GpuTensorHandle,
    ) -> AccelProviderFuture<'a, GpuTensorHandle> {
        Box::pin(async move {
            if let Some(result) = self.hook_override() {
                return result;
            }
            let host = self.inner.download(input)?;
            assert_eq!(host.storage, runmat_accelerate_api::GpuTensorStorage::Real);
            let values: Vec<_> = host.data.into_iter().map(f64::tan).collect();
            Ok(self.track(self.inner.upload(&HostTensorView {
                data: &values,
                shape: &host.shape,
            })?))
        })
    }
    fn unary_atan<'a>(
        &'a self,
        input: &'a GpuTensorHandle,
    ) -> AccelProviderFuture<'a, GpuTensorHandle> {
        Box::pin(async move {
            if let Some(result) = self.hook_override() {
                return result;
            }
            let host = self.inner.download(input)?;
            assert_eq!(host.storage, runmat_accelerate_api::GpuTensorStorage::Real);
            let values: Vec<_> = host.data.into_iter().map(f64::atan).collect();
            Ok(self.track(self.inner.upload(&HostTensorView {
                data: &values,
                shape: &host.shape,
            })?))
        })
    }
}
fn fixture(mode: Download) -> (&'static Provider, GpuTensorHandle, GpuTensorHandle) {
    let provider = Box::leak(Box::new(Provider {
        inner: test_owner::Owner::new(),
        state: Mutex::default(),
    }));
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
        assert_eq!(provider.inner.probe(input).retirements(), 0);
        assert!(provider.inner.download(input).is_ok());
    }
}

#[test]
fn trigonometric_like_math_and_restore_never_adopt_or_free_borrowed_prototypes() {
    let _serial = test_support::accel_test_lock();
    let _extensions = crate::compatibility::push_runmat_extensions_enabled(true);
    for name in ["sin", "cos", "tan", "atan"] {
        for restoration in [false, true] {
            for malformed in [false, true] {
                let (provider, source, old_prototype) = fixture(Download::Ready);
                old_prototype.release().unwrap();
                let prototype = provider
                    .inner
                    .upload(&HostTensorView {
                        data: &[1.0, 2.0],
                        shape: &[1, 2],
                    })
                    .unwrap();
                let alias = if malformed {
                    prototype
                        .with_view(prototype.view().clone().with_shape(vec![2, 1]))
                        .unwrap()
                } else {
                    prototype.clone()
                };
                {
                    let mut state = provider.state.lock().unwrap();
                    if restoration {
                        state.unsupported_hook = true;
                        state.restore_alias = Some(alias.downgrade());
                    } else {
                        state.hook_alias = Some(alias.downgrade());
                    }
                }
                let _selected = ThreadProviderGuard::set(Some(provider));
                let error = block_on(crate::dispatcher::call_builtin_async(
                    name,
                    &args(&source, Value::GpuTensor(prototype.clone())),
                ))
                .expect_err("borrowed output must be rejected");
                assert!(error.to_string().contains(name), "{error}");
                let state = provider.state.lock().unwrap();
                assert_eq!(state.hooks, 1);
                assert_eq!(state.downloads, usize::from(restoration));
                assert!(state.outputs.is_empty());
                assert!(state.outputs.iter().all(|probe| probe.retirements() == 0));
                drop(state);
                borrowed_live(provider, &source, &prototype);
                source.release().unwrap();
                prototype.release().unwrap();
            }
        }
    }
}

#[test]
fn trigonometric_like_fallback_abandonment_preserves_all_borrowed_handles() {
    let _serial = test_support::accel_test_lock();
    let _extensions = crate::compatibility::push_runmat_extensions_enabled(true);
    for name in ["sin", "cos", "tan", "atan"] {
        let (provider, source, prototype) = fixture(Download::Pending);
        provider.state.lock().unwrap().unsupported_hook = true;
        let _selected = ThreadProviderGuard::set(Some(provider));
        let inputs = args(&source, Value::GpuTensor(prototype.clone()));
        let mut call = Box::pin(crate::dispatcher::call_builtin_async(name, &inputs));
        assert!(matches!(
            call.as_mut().poll(&mut Context::from_waker(&noop_waker())),
            Poll::Pending
        ));
        drop(call);
        let state = provider.state.lock().unwrap();
        assert_eq!(state.hooks, 1);
        assert_eq!(state.downloads, 1);
        assert!(state.outputs.is_empty());
        assert!(state.outputs.iter().all(|probe| probe.retirements() == 0));
        drop(state);
        borrowed_live(provider, &source, &prototype);
        source.release().unwrap();
        prototype.release().unwrap();
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
            assert_eq!(
                state
                    .outputs
                    .iter()
                    .map(|probe| probe.retirements())
                    .sum::<usize>(),
                usize::from(!resident)
            );
            drop(state);
            let gathered = block_on(gpu_helpers::gather_value_async(result.value())).unwrap();
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
            result.with_owned(|value| {
                if let Value::GpuTensor(result) = value {
                    result.release().unwrap();
                }
            });
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
            assert_eq!(state.outputs.len(), 1);
            assert_eq!(state.outputs[0].retirements(), 1);
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
        let (provider, source, prototype) = fixture(Download::Ready);
        let _selected = ThreadProviderGuard::set(Some(provider));
        let original = prototype.clone();
        assert!(prototype
            .with_view(prototype.view().clone().with_shape(vec![9, 9]))
            .is_err());
        assert!(block_on(crate::dispatcher::call_builtin_async(
            name,
            &args(&source, Value::String("invalid".into()))
        ))
        .is_err());
        let state = provider.state.lock().unwrap();
        assert_eq!(state.hooks, 0);
        assert_eq!(state.downloads, 0);
        assert!(state.outputs.iter().all(|probe| probe.retirements() == 0));
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
        let prototype = provider
            .inner
            .upload_numeric(&runmat_accelerate_api::HostNumericTensorView {
                data: runmat_accelerate_api::HostNumericDataView::F64(&[0.0, 1.0]),
                shape: &[1, 1],
                storage: runmat_accelerate_api::GpuTensorStorage::ComplexInterleaved,
            })
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
        assert_eq!(state.outputs.len(), 1);
        assert_eq!(state.outputs[0].retirements(), 1);
        assert_eq!(provider.inner.probe(&prototype).retirements(), 0);
        drop(state);
        prototype.release().unwrap();
        borrowed_live(provider, &source, &spare);
    }
}

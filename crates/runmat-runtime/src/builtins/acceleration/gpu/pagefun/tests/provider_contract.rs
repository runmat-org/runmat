use super::*;
use runmat_accelerate_api::{AccelProvider, GpuTensorStorage};
use std::sync::atomic::{AtomicUsize, Ordering};

#[derive(Clone, Copy)]
enum Response {
    Valid,
    WrongShape,
    WrongStorage,
    Alias,
    Failure,
    Unsupported,
}
struct Provider {
    inner: runmat_accelerate::simple_provider::InProcessProvider,
    response: Response,
    calls: AtomicUsize,
    frees: AtomicUsize,
}
impl AccelProvider for Provider {
    fn upload(&self, view: &HostTensorView) -> anyhow::Result<GpuTensorHandle> {
        self.inner.upload(view)
    }
    fn download<'a>(
        &'a self,
        handle: &'a GpuTensorHandle,
    ) -> runmat_accelerate_api::AccelDownloadFuture<'a> {
        self.inner.download(handle)
    }
    fn free(&self, handle: &GpuTensorHandle) -> anyhow::Result<()> {
        self.frees.fetch_add(1, Ordering::SeqCst);
        self.inner.free(handle)
    }
    fn device_id(&self) -> u32 {
        self.inner.device_id()
    }
    fn device_info(&self) -> String {
        "page-product contract fixture".into()
    }
    fn pagefun(&self, request: &PagefunRequest) -> anyhow::Result<GpuTensorHandle> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        match self.response {
            Response::Alias => return Ok(request.inputs[0].clone()),
            Response::Failure => anyhow::bail!("injected device allocation failure"),
            Response::Unsupported => {
                return Err(runmat_accelerate_api::unsupported_provider_operation(
                    "fixture unsupported",
                ))
            }
            _ => {}
        }
        let mut output = self.inner.upload(&HostTensorView {
            data: &[1., 0., 0., 1.],
            shape: &[2, 2],
        })?;
        match self.response {
            Response::WrongShape => output.shape = vec![1, 4],
            Response::WrongStorage => runmat_accelerate_api::set_handle_logical(&output, true),
            _ => {}
        }
        Ok(output)
    }
}
fn fixture(response: Response) -> &'static Provider {
    let provider = Box::leak(Box::new(Provider {
        inner: runmat_accelerate::simple_provider::InProcessProvider::new(),
        response,
        calls: AtomicUsize::new(0),
        frees: AtomicUsize::new(0),
    }));
    unsafe {
        runmat_accelerate_api::register_device_provider(provider);
    }
    provider
}

#[test]
fn result_contract_rejects_bad_metadata_without_freeing_borrowed_inputs() {
    let _lock = test_support::accel_test_lock();
    for response in [
        Response::WrongShape,
        Response::WrongStorage,
        Response::Alias,
    ] {
        let provider = fixture(response);
        let _active = runmat_accelerate_api::ThreadProviderGuard::set(Some(provider));
        let input = provider
            .upload(&HostTensorView {
                data: &[1., 0., 0., 1.],
                shape: &[2, 2],
            })
            .unwrap();
        let before = gpu_helpers::snapshot_handle_metadata(&input);
        let values = [
            Value::GpuTensor(input.clone()),
            Value::GpuTensor(input.clone()),
        ];
        let error = try_pagefun_gpu(&PageOperation::Mtimes, &values).unwrap_err();
        assert_eq!(error.identifier(), PAGEFUN_ERROR_INTERNAL.identifier);
        assert_eq!(provider.calls.load(Ordering::SeqCst), 1);
        assert_eq!(
            provider.frees.load(Ordering::SeqCst),
            usize::from(!matches!(response, Response::Alias))
        );
        assert_eq!(gpu_helpers::snapshot_handle_metadata(&input), before);
        assert!(block_on(provider.download(&input)).is_ok());
        provider.free(&input).unwrap();
    }
}

#[test]
fn unsupported_allows_fallback_but_execution_failure_does_not() {
    let _lock = test_support::accel_test_lock();
    for response in [Response::Unsupported, Response::Failure] {
        let provider = fixture(response);
        let _active = runmat_accelerate_api::ThreadProviderGuard::set(Some(provider));
        let input = provider
            .upload(&HostTensorView {
                data: &[1., 0., 0., 1.],
                shape: &[2, 2],
            })
            .unwrap();
        let values = [
            Value::GpuTensor(input.clone()),
            Value::GpuTensor(input.clone()),
        ];
        let result = try_pagefun_gpu(&PageOperation::Mtimes, &values);
        if matches!(response, Response::Unsupported) {
            assert!(result.unwrap().is_none());
        } else {
            assert!(result
                .unwrap_err()
                .message()
                .contains("injected device allocation failure"));
        }
        assert_eq!(provider.calls.load(Ordering::SeqCst), 1);
        provider.free(&input).unwrap();
    }
}

#[test]
fn exact_owner_is_used_instead_of_active_provider_and_discard_releases_result() {
    let _lock = test_support::accel_test_lock();
    let owner = fixture(Response::Valid);
    let unrelated = fixture(Response::Failure);
    let _active = runmat_accelerate_api::ThreadProviderGuard::set(Some(unrelated));
    let mut input = owner
        .upload(&HostTensorView {
            data: &[1., 0., 0., 1.],
            shape: &[2, 2],
        })
        .unwrap();
    runmat_accelerate_api::mark_handle_explicit(&mut input);
    for count in [0, 1] {
        let _outputs = crate::output_count::push_output_count(Some(count));
        let result = block_on(pagefun_builtin(
            Value::FunctionHandle("mtimes".into()),
            Value::GpuTensor(input.clone()),
            vec![Value::GpuTensor(input.clone())],
        ))
        .unwrap();
        if count == 0 {
            assert!(matches!(result,Value::OutputList(values) if values.is_empty()));
        } else {
            let Value::GpuTensor(output) = result else {
                panic!("resident output");
            };
            assert!(runmat_accelerate_api::handle_is_explicit(&output));
            assert_eq!(
                runmat_accelerate_api::handle_storage(&output),
                GpuTensorStorage::Real
            );
            owner.free(&output).unwrap();
        }
    }
    assert_eq!(owner.calls.load(Ordering::SeqCst), 2);
    assert_eq!(owner.frees.load(Ordering::SeqCst), 2);
    assert_eq!(unrelated.calls.load(Ordering::SeqCst), 0);
    owner.free(&input).unwrap();
}

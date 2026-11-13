use super::*;
use runmat_accelerate_api::allocation::WeakAllocationLease;
use runmat_accelerate_api::{AccelProvider, GpuTensorStorage};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Mutex;

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
    produced: Mutex<Vec<WeakAllocationLease>>,
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
        let output = match self.response {
            Response::WrongStorage => self
                .inner
                .upload_logical_raw(&runmat_accelerate_api::HostLogicalTensorView {
                    data: &[1, 0, 0, 1],
                    shape: &[2, 2],
                })
                .expect("fixture logical output allocation"),
            _ => self.inner.upload(&HostTensorView {
                data: &[1., 0., 0., 1.],
                shape: if matches!(self.response, Response::WrongShape) {
                    &[1, 4]
                } else {
                    &[2, 2]
                },
            })?,
        };
        self.produced
            .lock()
            .unwrap()
            .push(output.allocation_lease().downgrade());
        Ok(output)
    }
}
fn fixture(response: Response) -> &'static Provider {
    let provider = Box::leak(Box::new(Provider {
        inner: runmat_accelerate::simple_provider::InProcessProvider::new(),
        response,
        calls: AtomicUsize::new(0),
        produced: Mutex::new(Vec::new()),
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
        let before = input.wire_descriptor();
        let input_lease = input.allocation_lease().downgrade();
        let values = [
            Value::GpuTensor(input.clone()),
            Value::GpuTensor(input.clone()),
        ];
        let error = try_pagefun_gpu(&PageOperation::Mtimes, &values).unwrap_err();
        assert_eq!(error.identifier(), PAGEFUN_ERROR_INTERNAL.identifier);
        assert_eq!(provider.calls.load(Ordering::SeqCst), 1);
        drop(error);
        let produced = provider.produced.lock().unwrap();
        assert_eq!(
            produced.len(),
            usize::from(!matches!(response, Response::Alias))
        );
        assert!(produced.iter().all(|lease| lease.acquire().is_none()));
        drop(produced);
        assert_eq!(input.wire_descriptor(), before);
        assert!(block_on(provider.download(&input)).is_ok());
        drop(values);
        assert!(input_lease.acquire().is_some());
        drop(input);
        assert!(input_lease.acquire().is_none());
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
        let input_lease = input.allocation_lease().downgrade();
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
        drop(values);
        assert!(input_lease.acquire().is_some());
        drop(input);
        assert!(input_lease.acquire().is_none());
    }
}

#[test]
fn exact_owner_is_used_instead_of_active_provider_and_discard_releases_result() {
    let _lock = test_support::accel_test_lock();
    let owner = fixture(Response::Valid);
    let unrelated = fixture(Response::Failure);
    let _active = runmat_accelerate_api::ThreadProviderGuard::set(Some(unrelated));
    let input = owner
        .upload(&HostTensorView {
            data: &[1., 0., 0., 1.],
            shape: &[2, 2],
        })
        .unwrap()
        .with_provenance(runmat_accelerate_api::GpuHandleProvenance::Explicit)
        .expect("explicit immutable view");
    let input_lease = input.allocation_lease().downgrade();
    for count in [0, 1] {
        let _outputs =
            crate::output_count::push_output_count(Some(count)).expect("output-count scope");
        let result = block_on(pagefun_builtin(
            Value::FunctionHandle("mtimes".into()),
            Value::GpuTensor(input.clone()),
            crate::builtins::common::test_support::builtin_arguments(vec![Value::GpuTensor(
                input.clone(),
            )]),
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
            output.release().expect("release owned result");
        }
    }
    assert_eq!(owner.calls.load(Ordering::SeqCst), 2);
    let produced = owner.produced.lock().unwrap();
    assert_eq!(produced.len(), 2);
    assert!(produced.iter().all(|lease| lease.acquire().is_none()));
    assert_eq!(unrelated.calls.load(Ordering::SeqCst), 0);
    drop(input);
    assert!(input_lease.acquire().is_none());
}

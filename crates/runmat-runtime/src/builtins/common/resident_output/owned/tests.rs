use super::*;
use crate::builtins::common::test_support;
use futures::executor::block_on;
use runmat_accelerate_api::HostTensorView;
use std::future::Future;
use std::task::{Context, Poll};

#[test]
fn retained_owner_survives_thread_selection_and_metadata_loss() {
    use runmat_accelerate_api::{
        AccelDownloadFuture, AccelProvider, ProviderPrecision, ThreadProviderGuard,
    };
    use std::sync::atomic::{AtomicUsize, Ordering};
    struct Owner(AtomicUsize);
    static OWNER: Owner = Owner(AtomicUsize::new(0));
    impl AccelProvider for Owner {
        fn device_id(&self) -> u32 {
            u32::MAX - 237
        }
        fn precision(&self) -> ProviderPrecision {
            ProviderPrecision::F64
        }
        fn device_info(&self) -> String {
            "unregistered-result-owner".into()
        }
        fn upload(&self, _: &HostTensorView) -> anyhow::Result<GpuTensorHandle> {
            unreachable!()
        }
        fn download<'a>(&'a self, _: &'a GpuTensorHandle) -> AccelDownloadFuture<'a> {
            unreachable!()
        }
        fn free(&self, _: &GpuTensorHandle) -> anyhow::Result<()> {
            self.0.fetch_add(1, Ordering::Relaxed);
            Ok(())
        }
    }
    let _lock = test_support::accel_test_lock();
    OWNER.0.store(0, Ordering::Relaxed);
    let _empty = ThreadProviderGuard::set(None);
    let selected = ThreadProviderGuard::set(Some(&OWNER));
    let handle = GpuTensorHandle::new(vec![1, 1], OWNER.device_id(), 1);
    let guard = ResidentResultGuard::new(&Value::GpuTensor(handle.clone()), &[]).unwrap();
    drop(selected);
    runmat_accelerate_api::clear_handle_metadata(&handle);
    assert!(gpu_helpers::exact_provider_for_handle(&handle).is_none());
    drop(guard);
    assert_eq!(OWNER.0.load(Ordering::Relaxed), 1);
}

#[test]
fn pending_adaptation_drop_releases_fresh_result_but_not_borrowed_input() {
    test_support::with_test_provider(|provider| {
        let upload = || {
            provider
                .upload(&HostTensorView {
                    data: &[1.0],
                    shape: &[1, 1],
                })
                .unwrap()
        };
        let input = upload();
        let output = upload();
        let protected = [&input];
        let guard =
            ResidentResultGuard::new(&Value::GpuTensor(output.clone()), &protected).unwrap();
        let mut pending = Box::pin(async move {
            let _owned = guard;
            // Adaptation is allowed to change/drop supplementary metadata;
            // the already-owned allocation keeps its original authority.
            runmat_accelerate_api::clear_handle_metadata(&_owned.resident.as_ref().unwrap().handle);
            futures::future::pending::<()>().await;
        });
        let mut context = Context::from_waker(futures::task::noop_waker_ref());
        assert!(matches!(pending.as_mut().poll(&mut context), Poll::Pending));
        drop(pending);
        assert!(block_on(provider.download(&output)).is_err());
        assert!(block_on(provider.download(&input)).is_ok());
        provider.free(&input).unwrap();
    });
}

#[test]
fn publication_transfers_only_the_same_allocation() {
    test_support::with_test_provider(|provider| {
        let upload = || {
            provider
                .upload(&HostTensorView {
                    data: &[1.0],
                    shape: &[1, 1],
                })
                .unwrap()
        };
        let output = upload();
        let value = Value::GpuTensor(output.clone());
        ResidentResultGuard::new(&value, &[])
            .unwrap()
            .publish(&value);
        assert!(block_on(provider.download(&output)).is_ok());
        let replacement = upload();
        ResidentResultGuard::new(&value, &[])
            .unwrap()
            .publish(&Value::GpuTensor(replacement.clone()));
        assert!(block_on(provider.download(&output)).is_err());
        assert!(block_on(provider.download(&replacement)).is_ok());
        provider.free(&replacement).unwrap();
    });
}

#[test]
fn borrowed_result_drop_and_host_replacement_never_consume_the_caller() {
    test_support::with_test_provider(|provider| {
        let input = provider
            .upload(&HostTensorView {
                data: &[1.0],
                shape: &[1, 1],
            })
            .unwrap();
        let protected = [&input];
        let value = Value::GpuTensor(input.clone());
        drop(ResidentResultGuard::new(&value, &protected).unwrap());
        ResidentResultGuard::new(&value, &protected)
            .unwrap()
            .publish(&Value::Num(1.0));
        assert!(block_on(provider.download(&input)).is_ok());
        provider.free(&input).unwrap();
    });
}

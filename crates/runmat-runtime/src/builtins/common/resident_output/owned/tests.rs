use super::*;
use crate::builtins::common::test_support;
use futures::executor::block_on;
use runmat_accelerate_api::HostTensorView;
use std::future::Future;
use std::task::{Context, Poll};

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

use super::*;
use futures::{executor::block_on, task::noop_waker};
use runmat_accelerate_api::test_support::TestAllocationOwner;
use std::{
    future::Future,
    task::{Context, Poll},
};

struct Provider(TestAllocationOwner);
impl AccelProvider for Provider {
    fn device_id(&self) -> u32 {
        self.0.device_id()
    }
    fn device_info(&self) -> String {
        "coordinate ownership fixture".into()
    }
    fn upload(&self, _: &runmat_accelerate_api::HostTensorView) -> anyhow::Result<GpuTensorHandle> {
        panic!("ownership test does not upload")
    }
    fn download<'a>(
        &'a self,
        _: &'a GpuTensorHandle,
    ) -> runmat_accelerate_api::AccelDownloadFuture<'a> {
        Box::pin(async { panic!("ownership test does not download") })
    }
}
fn provider() -> &'static Provider {
    Box::leak(Box::new(Provider(TestAllocationOwner::new())))
}

#[test]
fn coordinate_publication_retains_output_but_retires_intermediates() {
    let _serial = crate::builtins::common::test_support::accel_test_lock();
    let provider = provider();
    let _selected = runmat_accelerate_api::ThreadProviderGuard::set(Some(provider));
    let allocate = || {
        provider
            .0
            .allocate_zeroed(
                vec![1, 2],
                runmat_accelerate_api::NumericElementType::F64,
                GpuTensorStorage::Real,
            )
            .unwrap()
    };
    let input = allocate();
    let intermediate = allocate();
    let result = allocate();
    let inputs = [Value::GpuTensor(input.handle.clone())];
    let mut outputs = Outputs::new(&inputs);
    drop(
        outputs
            .adopt(
                provider,
                intermediate.handle,
                &input.handle,
                &[1, 2],
                ProviderPrecision::F64,
            )
            .unwrap(),
    );
    let handle = outputs
        .adopt(
            provider,
            result.handle,
            &input.handle,
            &[1, 2],
            ProviderPrecision::F64,
        )
        .unwrap();
    let published = outputs.publish(handle).unwrap();
    drop(outputs);
    assert_eq!(intermediate.probe.retirements(), 1);
    assert_eq!(result.probe.retirements(), 0);
    assert_eq!(input.probe.retirements(), 0);
    drop(published);
    assert_eq!(result.probe.retirements(), 1);
    drop(inputs);
    drop(input.handle);
    assert_eq!(input.probe.retirements(), 1);
}

#[test]
fn pending_coordinate_operation_releases_owned_result_not_borrowed_input() {
    let _serial = crate::builtins::common::test_support::accel_test_lock();
    let provider = provider();
    let _selected = runmat_accelerate_api::ThreadProviderGuard::set(Some(provider));
    let input = provider
        .0
        .allocate_zeroed(
            vec![1, 2],
            runmat_accelerate_api::NumericElementType::F64,
            GpuTensorStorage::Real,
        )
        .unwrap();
    let temporary = provider
        .0
        .allocate_zeroed(
            vec![1, 2],
            runmat_accelerate_api::NumericElementType::F64,
            GpuTensorStorage::Real,
        )
        .unwrap();
    let borrowed = input.handle.clone();
    let mut future = Box::pin(async move {
        let mut outputs = Outputs::new(&[Value::GpuTensor(borrowed.clone())]);
        drop(
            outputs
                .adopt(
                    provider,
                    temporary.handle,
                    &borrowed,
                    &[1, 2],
                    ProviderPrecision::F64,
                )
                .unwrap(),
        );
        outputs
            .run_protected(std::future::pending::<anyhow::Result<()>>())
            .await
    });
    assert!(matches!(
        future
            .as_mut()
            .poll(&mut Context::from_waker(&noop_waker())),
        Poll::Pending
    ));
    drop(future);
    assert_eq!(temporary.probe.retirements(), 1);
    assert_eq!(input.probe.retirements(), 0);
    let mut outputs = Outputs::new(&[Value::GpuTensor(input.handle.clone())]);
    let alias = input
        .handle
        .with_provenance(runmat_accelerate_api::GpuHandleProvenance::Automatic)
        .unwrap();
    assert!(block_on(async {
        outputs.adopt(
            provider,
            alias,
            &input.handle,
            &[1, 2],
            ProviderPrecision::F64,
        )
    })
    .is_err());
    drop(outputs);
    drop(input.handle);
    assert_eq!(input.probe.retirements(), 1);
}

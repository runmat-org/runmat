use super::*;
use crate::mxarray::MxGpuLease;
use runmat_accelerate_api::NativeDeviceAccess;

#[test]
fn native_export_retains_allocation_through_sync_and_leave_even_after_array_drop() {
    for (sync, leave) in [
        (Fault::None, Fault::None),
        (Fault::Error, Fault::Error),
        (Fault::Panic, Fault::Panic),
    ] {
        let provider = Provider::new(Fault::None, sync, leave);
        let allocation = provider.empty_allocation();
        provider.expect_retained(&allocation.probe);
        let context = NativeInvocationContext::enter(provider).unwrap();
        let array = MxGpuLease::owned(allocation.handle);
        assert_eq!(
            array.device_address(NativeDeviceAccess::ReadWrite).unwrap(),
            0
        );
        drop(array);
        assert_eq!(allocation.probe.retirements(), 0);
        let result = context.leave();
        if matches!(sync, Fault::None) {
            assert!(result.is_ok());
        } else {
            assert!(result.is_err());
        }
        assert_eq!(
            provider.events(),
            ["enter", "export", "synchronize", "leave"]
        );
        assert_eq!(allocation.probe.retirements(), 1);
    }
}

#[test]
fn native_export_nested_contexts_do_not_reselect_or_cross_provider_ownership() {
    let outer = Provider::new(Fault::None, Fault::None, Fault::None);
    let inner = Provider::new(Fault::None, Fault::None, Fault::None);
    let allocation = outer.empty_allocation();
    let array = MxGpuLease::owned(allocation.handle);
    assert!(array.device_address(NativeDeviceAccess::ReadOnly).is_err());
    let outer_context = NativeInvocationContext::enter(outer).unwrap();
    let _selected = ThreadProviderGuard::set(Some(inner));
    assert_eq!(
        array.device_address(NativeDeviceAccess::ReadOnly).unwrap(),
        0
    );
    let inner_context = NativeInvocationContext::enter(inner).unwrap();
    assert!(array.device_address(NativeDeviceAccess::ReadOnly).is_err());
    inner_context.leave().unwrap();
    assert_eq!(
        array.device_address(NativeDeviceAccess::ReadOnly).unwrap(),
        0
    );
    drop(array);
    assert_eq!(allocation.probe.retirements(), 0);
    outer_context.leave().unwrap();
    assert_eq!(allocation.probe.retirements(), 1);
    assert_eq!(inner.events(), ["enter", "synchronize", "leave"]);
}

#[test]
fn native_export_pending_invocation_abandonment_drops_exports_after_context_cleanup() {
    let provider = Provider::new(Fault::None, Fault::None, Fault::None);
    let allocation = provider.empty_allocation();
    provider.expect_retained(&allocation.probe);
    let mut future = Box::pin(async move {
        let _context = NativeInvocationContext::enter(provider).unwrap();
        let array = MxGpuLease::owned(allocation.handle);
        array.device_address(NativeDeviceAccess::ReadWrite).unwrap();
        drop(array);
        std::future::pending::<()>().await;
    });
    let waker = waker();
    assert!(future
        .as_mut()
        .poll(&mut Context::from_waker(&waker))
        .is_pending());
    assert_eq!(allocation.probe.retirements(), 0);
    drop(future);
    assert_eq!(allocation.probe.retirements(), 1);
    assert_eq!(
        provider.events(),
        ["enter", "export", "synchronize", "leave"]
    );
}

#[cfg(panic = "unwind")]
#[test]
fn native_export_existing_unwind_releases_after_both_cleanup_attempts() {
    let provider = Provider::new(Fault::None, Fault::Panic, Fault::Panic);
    let allocation = provider.empty_allocation();
    provider.expect_retained(&allocation.probe);
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let _context = NativeInvocationContext::enter(provider).unwrap();
        let array = MxGpuLease::owned(allocation.handle);
        array.device_address(NativeDeviceAccess::ReadWrite).unwrap();
        drop(array);
        panic!("invocation abandoned");
    }));
    assert!(result.is_err());
    assert_eq!(allocation.probe.retirements(), 1);
    assert_eq!(
        provider.events(),
        ["enter", "export", "synchronize", "leave"]
    );
}

#[test]
fn native_export_cancelled_invocation_releases_after_sync_and_exit() {
    let provider = Provider::new(Fault::None, Fault::None, Fault::None);
    let allocation = provider.empty_allocation();
    provider.expect_retained(&allocation.probe);
    let source = runmat_execution::cancellation::CancellationSource::new();
    let token = source.token();
    let mut future = Box::pin(async move {
        let _context = NativeInvocationContext::enter(provider).unwrap();
        let array = MxGpuLease::owned(allocation.handle);
        array.device_address(NativeDeviceAccess::ReadOnly).unwrap();
        drop(array);
        token.cancelled().await
    });
    let waker = waker();
    let mut task = Context::from_waker(&waker);
    assert!(future.as_mut().poll(&mut task).is_pending());
    assert_eq!(allocation.probe.retirements(), 0);
    source.cancel(runmat_execution::state::CancellationReason::User);
    assert_eq!(
        future.as_mut().poll(&mut task),
        std::task::Poll::Ready(runmat_execution::state::CancellationReason::User)
    );
    assert_eq!(allocation.probe.retirements(), 1);
    assert_eq!(
        provider.events(),
        ["enter", "export", "synchronize", "leave"]
    );
}

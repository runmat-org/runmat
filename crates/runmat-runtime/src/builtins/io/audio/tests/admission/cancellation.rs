use super::*;
use crate::context::RuntimeContext;
use futures::task::{waker_ref, ArcWake};
use runmat_execution::cancellation::CancellationSource;
use runmat_execution::CancellationReason;
use runmat_execution::ResourceByteBound;
use std::future::Future;
use std::pin::Pin;
use std::rc::Rc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::task::{Context, Poll};

struct WakeCount(AtomicUsize);
impl ArcWake for WakeCount {
    fn wake_by_ref(waker: &Arc<Self>) {
        waker.0.fetch_add(1, Ordering::SeqCst);
    }
}

fn fixture() -> (Arc<CountedFs>, runmat_filesystem::ProviderGuard) {
    let fs = Arc::new(CountedFs::new("/audio"));
    fs.inner
        .write_project_path("/audio/tone.wav", &pcm16_wav(8_000, 1, &[0, 16_384]))
        .expect("fixture WAV");
    let guard = runmat_filesystem::replace_provider(fs.clone());
    (fs, guard)
}

fn known() -> Option<crate::context::host_effect::HostEffectResourceEvidence> {
    Some(evidence(
        ResourceByteBound::Known(4 * 1024),
        ResourceByteBound::Known(256 * 1024),
    ))
}

fn value() -> Value {
    Value::from("/audio/tone.wav")
}

fn cancel_pending_call(
    mut call: Pin<&mut impl Future<Output = BuiltinResult<Value>>>,
    source: &CancellationSource,
    placement: &Placement,
) {
    let wake_count = Arc::new(WakeCount(AtomicUsize::new(0)));
    let waker = waker_ref(&wake_count);
    assert!(matches!(
        call.as_mut().poll(&mut Context::from_waker(&waker)),
        Poll::Pending
    ));
    assert_eq!(placement.probes.load(Ordering::SeqCst), 1);
    source.cancel(CancellationReason::User);
    assert!(wake_count.0.load(Ordering::SeqCst) > 0);
    let result = call.as_mut().poll(&mut Context::from_waker(&waker));
    let Poll::Ready(Err(error)) = result else {
        panic!("cancelled audio call did not settle");
    };
    assert_eq!(error.identifier(), Some("RunMat:Cancelled"));
    assert_eq!(placement.probes.load(Ordering::SeqCst), 0);
    assert!(placement.observations.borrow().is_empty());
}

fn selected_context(fs: &Arc<CountedFs>) -> (RuntimeContext, Rc<Placement>) {
    let provider: Arc<dyn runmat_filesystem::FsProvider> = fs.clone();
    let (runtime, _, placement) = context(provider, known(), Some(512 * 1024));
    (runtime, placement)
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn pending_audio_open_is_woken_and_releases_selected_probe() {
    let _lock = runmat_filesystem::provider_override_lock();
    let (fs, _owner) = fixture();
    fs.pending_open.store(true, Ordering::SeqCst);
    let (runtime, placement) = selected_context(&fs);
    let source = CancellationSource::new();
    let runtime = runtime.with_cancellation_token(source.token());
    let mut call = Box::pin(audioread_builtin(
        Some(runtime),
        value(),
        crate::builtins::common::test_support::builtin_arguments(vec![]),
    ));
    cancel_pending_call(call.as_mut(), &source, &placement);
    assert_eq!(fs.effects().0, 1);
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn pending_provider_whole_read_is_woken_without_fallback_replay() {
    let _lock = runmat_filesystem::provider_override_lock();
    let (fs, _owner) = fixture();
    fs.unsupported_open.store(true, Ordering::SeqCst);
    fs.pending_read.store(true, Ordering::SeqCst);
    let (runtime, placement) = selected_context(&fs);
    let source = CancellationSource::new();
    let runtime = runtime.with_cancellation_token(source.token());
    let mut call = Box::pin(audioread_builtin(
        Some(runtime),
        value(),
        crate::builtins::common::test_support::builtin_arguments(vec![]),
    ));
    cancel_pending_call(call.as_mut(), &source, &placement);
    assert_eq!(fs.effects().0, 1);
    assert_eq!(fs.effects().1, 1);
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn pending_audio_metadata_is_woken_and_releases_selected_probe() {
    let _lock = runmat_filesystem::provider_override_lock();
    let (fs, _owner) = fixture();
    fs.pending_metadata.store(true, Ordering::SeqCst);
    let (runtime, placement) = selected_context(&fs);
    let source = CancellationSource::new();
    let runtime = runtime.with_cancellation_token(source.token());
    let mut call = Box::pin(audioinfo_builtin(Some(runtime), value()));
    cancel_pending_call(call.as_mut(), &source, &placement);
    assert_eq!(fs.effects().0, 1);
    assert_eq!(fs.effects().2, 0);
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn pending_provider_metadata_fallback_is_woken_without_reopening() {
    let _lock = runmat_filesystem::provider_override_lock();
    let (fs, _owner) = fixture();
    fs.pending_provider_metadata.store(true, Ordering::SeqCst);
    let (runtime, placement) = selected_context(&fs);
    let source = CancellationSource::new();
    let runtime = runtime.with_cancellation_token(source.token());
    let mut call = Box::pin(audioinfo_builtin(Some(runtime), value()));
    cancel_pending_call(call.as_mut(), &source, &placement);
    assert_eq!(fs.effects().0, 1);
    assert_eq!(fs.effects().2, 1);
}

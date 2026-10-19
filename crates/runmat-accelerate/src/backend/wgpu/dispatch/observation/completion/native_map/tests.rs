use super::{
    first_result, receive_after_progress, settle_after_progress, wait_pending, FirstResult,
};
use futures::channel::oneshot;
use std::{
    future::Future,
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc,
    },
    task::{Context, Poll},
};

struct Retired(Arc<AtomicBool>);

impl Drop for Retired {
    fn drop(&mut self) {
        self.0.store(true, Ordering::Release);
    }
}

fn poll_once<T>(future: std::pin::Pin<&mut impl Future<Output = T>>) -> Poll<T> {
    let waker = futures::task::noop_waker();
    future.poll(&mut Context::from_waker(&waker))
}

#[test]
fn successful_progress_waits_for_pending_callback_without_early_failure() {
    let (send, mut receive) = oneshot::channel::<Option<u8>>();
    let retired = Arc::new(AtomicBool::new(false));
    let marker = Retired(retired.clone());
    let mut settlement = Box::pin(async move {
        let _marker = marker;
        receive_after_progress(true, &mut receive).await
    });
    assert!(matches!(poll_once(settlement.as_mut()), Poll::Pending));
    assert!(!retired.load(Ordering::Acquire));
    send.send(Some(7)).expect("callback sender retained");
    assert_eq!(poll_once(settlement.as_mut()), Poll::Ready(Some(Some(7))));
    assert!(retired.load(Ordering::Acquire));
}

#[test]
fn delivered_failure_closed_and_optional_absence_are_distinct_inputs() {
    let (send, mut receive) = oneshot::channel::<Option<u8>>();
    send.send(None).expect("failure callback delivered");
    assert_eq!(settle_after_progress(true, Some(&mut receive)), Some(None));

    let (send, mut receive) = oneshot::channel::<bool>();
    send.send(false).expect("timing map failure delivered");
    assert_eq!(settle_after_progress(true, Some(&mut receive)), Some(false));

    let (send, mut receive) = oneshot::channel::<bool>();
    drop(send);
    assert_eq!(settle_after_progress(true, Some(&mut receive)), None);
    assert_eq!(settle_after_progress::<bool>(true, None), None);
}

#[test]
fn failed_progress_never_waits_for_live_callback_sender() {
    let (send, mut receive) = oneshot::channel::<bool>();
    let mut settlement = Box::pin(receive_after_progress(false, &mut receive));
    assert_eq!(poll_once(settlement.as_mut()), Poll::Ready(None));
    drop(settlement);
    send.send(true)
        .expect("failed path did not consume live callback");
    assert_eq!(receive.try_recv(), Ok(Some(true)));
}

#[test]
fn immediate_paths_do_not_enter_a_nested_executor() {
    futures::executor::block_on(async {
        let (send, mut receive) = oneshot::channel::<bool>();
        send.send(true).expect("ready callback retained");
        assert_eq!(settle_after_progress(true, Some(&mut receive)), Some(true));

        let (send, mut receive) = oneshot::channel::<bool>();
        assert_eq!(settle_after_progress(false, Some(&mut receive)), None);
        send.send(false)
            .expect("failed progress kept receiver live");
        assert_eq!(receive.try_recv(), Ok(Some(false)));

        let (send, mut receive) = oneshot::channel::<bool>();
        drop(send);
        assert_eq!(settle_after_progress(true, Some(&mut receive)), None);
        assert_eq!(settle_after_progress::<bool>(true, None), None);
    });
}

#[test]
fn pending_production_wait_works_inside_an_outer_executor() {
    let (send, mut receive) = oneshot::channel::<bool>();
    let (release, released) = std::sync::mpsc::channel::<()>();
    let callback = std::thread::spawn(move || {
        released.recv().expect("release pending callback");
        send.send(true).expect("receiver retained until callback");
    });
    assert!(matches!(
        first_result(true, &mut receive),
        FirstResult::Pending
    ));
    let mapped = futures::executor::block_on(async {
        release.send(()).expect("start callback delivery");
        wait_pending(&mut receive)
    });
    assert_eq!(mapped, Some(true));
    callback.join().expect("callback thread completed");
}

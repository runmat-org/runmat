//! Native map callback settlement after exact submission progress.

use futures::channel::oneshot;

enum FirstResult<T> {
    Settled(Option<T>),
    Pending,
}

fn first_result<T>(progressed: bool, receiver: &mut oneshot::Receiver<T>) -> FirstResult<T> {
    match receiver.try_recv() {
        Ok(Some(value)) => FirstResult::Settled(Some(value)),
        Ok(None) if progressed => FirstResult::Pending,
        Ok(None) | Err(_) => FirstResult::Settled(None),
    }
}

/// The progress worker has returned from `device.poll`, so it holds no WGPU
/// emission gate or core lock. That poll may have lost a race to another poll
/// thread which extracted the mapping callback but has not fired it yet.
/// Successful progress waits for that already-registered callback; failed
/// progress never waits on a live sender and retains the caller's failed path.
pub(super) fn settle_after_progress<T>(
    progressed: bool,
    receiver: Option<&mut oneshot::Receiver<T>>,
) -> Option<T> {
    let receiver = receiver?;
    match first_result(progressed, receiver) {
        FirstResult::Settled(value) => value,
        FirstResult::Pending => wait_pending(receiver),
    }
}

fn wait_pending<T>(receiver: &mut oneshot::Receiver<T>) -> Option<T> {
    pollster::block_on(receiver).ok()
}

#[cfg(test)]
async fn receive_after_progress<T>(
    progressed: bool,
    receiver: &mut oneshot::Receiver<T>,
) -> Option<T> {
    match first_result(progressed, receiver) {
        FirstResult::Settled(value) => value,
        FirstResult::Pending => receiver.await.ok(),
    }
}

#[cfg(test)]
mod tests;

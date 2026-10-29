//! Bounded pre-armed physical ownership for diagnostic scoped submissions.
//! Queue commands carry no physical owner; every exit/error retains slots until proof.
use super::*;
use std::sync::MutexGuard;

#[derive(Clone, Copy, PartialEq, Eq)]
enum Recovery {
    Polling,
    RetainedUncertain,
    ClosingUncertain,
}

pub(super) struct ScopedWork {
    pub(super) pending: Option<crate::backend::wgpu::dispatch::observation::PendingSubmissions>,
    #[cfg(test)]
    pub(super) probe: Option<Arc<()>>,
    pub(super) signal: crate::backend::wgpu::dispatch::observation::ScopedRetirement,
    pub(super) ready: bool,
    pub(super) _slot: Slot,
}

impl NativeObservationProgress {
    /// Held only across synchronous scoped emission and publication, never await.
    pub(in crate::backend::wgpu::provider::backend) fn scoped_emission_guard(
        &self,
        pending: crate::backend::wgpu::dispatch::observation::PendingSubmissions,
        signal: crate::backend::wgpu::dispatch::observation::ScopedRetirement,
    ) -> Result<ScopedEmissionGuard<'_>, WgpuObservationProgressError> {
        let gate = self
            .state
            .scoped_emission
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        if self.state.stopped.load(Ordering::Acquire)
            || self.state.worker_closed.load(Ordering::Acquire)
        {
            return Err(WgpuObservationProgressError::Stopped);
        }
        let reservation = self.reserve_slot()?;
        let mut work = self
            .state
            .scoped_work
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        let index = work
            .iter()
            .position(Option::is_none)
            .ok_or(WgpuObservationProgressError::Capacity)?;
        work[index] = Some(ScopedWork {
            pending: Some(pending),
            #[cfg(test)]
            probe: None,
            signal,
            ready: false,
            _slot: reservation.slot,
        });
        drop(work);
        Ok(ScopedEmissionGuard {
            control: &self.control,
            _gate: gate,
            index,
            attempted: false,
        })
    }
}

pub(in crate::backend::wgpu::provider::backend) struct ScopedEmissionGuard<'a> {
    control: &'a Arc<Control>,
    _gate: MutexGuard<'a, ()>,
    index: usize,
    attempted: bool,
}

impl ScopedEmissionGuard<'_> {
    pub(in crate::backend::wgpu::provider::backend) fn with_pending<T>(
        &mut self,
        operation: impl FnOnce(
            &mut crate::backend::wgpu::dispatch::observation::PendingSubmissions,
        ) -> T,
    ) -> T {
        self.attempted = true;
        let mut work = self
            .control
            .state
            .scoped_work
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        operation(
            work[self.index]
                .as_mut()
                .expect("pre-armed scoped owner")
                .pending
                .as_mut()
                .expect("pre-armed physical accumulator"),
        )
    }
}

impl Drop for ScopedEmissionGuard<'_> {
    fn drop(&mut self) {
        let mut work = self
            .control
            .state
            .scoped_work
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        if !self.attempted {
            // WgpuDeviceAccess rejected before invoking the emission closure:
            // this pre-armed slot contains only the initial, unsubmitted owner.
            let unused = work[self.index].take();
            drop(work);
            drop(unused);
            return;
        }
        work[self.index]
            .as_mut()
            .expect("pre-armed scoped owner")
            .ready = true;
        drop(work);
        // A failed wake never drops physical work: the existing worker scans
        // fixed slots before receiving queued work and before receiver exit.
        // Full means a queued wake/work item already exists. Receiver exit
        // cannot overtake the held scoped-emission gate.
        if self.control.sender.try_send(Command::Wake).is_err() {
            log::debug!("scoped retirement wake unavailable; owner remains in pre-armed slot");
        }
    }
}

impl State {
    fn recovery(&self) -> Recovery {
        if !self.retained_uncertain.load(Ordering::Acquire) {
            Recovery::Polling
        } else if self.shutdown_requested.load(Ordering::Acquire) {
            Recovery::ClosingUncertain
        } else {
            Recovery::RetainedUncertain
        }
    }

    pub(super) fn retain_uncertain(&self) {
        if !self.retained_uncertain.swap(true, Ordering::AcqRel) {
            // No emission permit is held by the progress worker. The canonical
            // fence waits out previously admitted synchronous emission, then
            // rejects all later emission without impersonating device loss.
            if let Some(access) = &self.access {
                access.fence_emission();
            }
            log::error!("WGPU progress failed: emission fenced; bounded physical owners retained until queue/device-loss proof or process exit");
        }
    }

    pub(super) fn next_scoped(self: &Arc<Self>) -> Option<Job> {
        let mut work = self.scoped_work.lock().unwrap_or_else(|e| e.into_inner());
        let closing = self.recovery() == Recovery::ClosingUncertain;
        let lost = self
            .health
            .as_ref()
            .is_some_and(|health| health.loss_reason().is_some());
        for selected in work.iter().flatten() {
            if lost {
                selected.signal.actual_device_lost();
            }
        }
        // Proven slots have priority: one uncertain tail must not prevent
        // another callback's factual completion being retired before shutdown.
        let index = work
            .iter()
            .position(|entry| {
                entry
                    .as_ref()
                    .is_some_and(|work| work.ready && work.signal.is_retired())
            })
            .or_else(|| {
                work.iter().position(|entry| {
                    entry
                        .as_ref()
                        .is_some_and(|work| work.ready && (!closing || work.signal.is_retired()))
                })
            })?;
        let selected = work[index].as_mut().unwrap();
        selected.ready = false;
        let signal = selected.signal.clone();
        drop(work);
        Some(Job {
            submission: WaitTarget::Scoped(signal),
            completion: Completion::scoped(self.clone(), index),
            _slot: None,
            retire_after_stop: true,
        })
    }

    pub(super) fn finish_scoped(&self, index: usize) {
        let mut work = self.scoped_work.lock().unwrap_or_else(|e| e.into_inner());
        let Some(selected) = work[index].as_mut() else {
            return;
        };
        let Some(proof) = selected.signal.proof() else {
            selected.ready = true;
            return;
        };
        let selected = work[index].take().unwrap();
        drop(work);
        let confirmed = selected.signal.is_confirmed()
            && !self.retained_uncertain.load(Ordering::Acquire)
            && self
                .health
                .as_ref()
                .is_none_or(|health| health.loss_reason().is_none());
        if let Some(pending) = selected.pending {
            pending.retire_scoped(proof, confirmed);
        }
        #[cfg(test)]
        drop(selected.probe);
        drop(selected._slot);
    }
}

pub(super) fn poll_scoped(
    access: &runmat_accelerate_api::WgpuDeviceAccess,
    state: &State,
    signal: &crate::backend::wgpu::dispatch::observation::ScopedRetirement,
) -> Result<(), WgpuObservationProgressError> {
    if signal.is_retired() {
        return Ok(());
    }
    if state
        .health
        .as_ref()
        .is_some_and(|health| health.loss_reason().is_some())
    {
        signal.actual_device_lost();
        return Ok(());
    }
    if state.recovery() == Recovery::Polling && signal.claim_callback_registration() {
        let callback = signal.clone();
        if access
            .on_submitted_work_done(move || callback.queue_retired())
            .is_err()
        {
            // Registration may have partially succeeded. Never register twice
            // or infer physical retirement from an access/driver error.
            return Err(WgpuObservationProgressError::Stopped);
        }
    }
    while !signal.is_retired() {
        if state
            .health
            .as_ref()
            .is_some_and(|health| health.loss_reason().is_some())
        {
            signal.actual_device_lost();
            break;
        }
        match state.recovery() {
            Recovery::Polling => {
                access.poll(wgpu::Maintain::Poll);
            }
            Recovery::RetainedUncertain => {
                // Yield to the existing slot scanner, including other proven
                // slots. No further driver calls after a fatal progress error.
                std::thread::park_timeout(std::time::Duration::from_millis(1));
                return Err(WgpuObservationProgressError::Stopped);
            }
            Recovery::ClosingUncertain => return Err(WgpuObservationProgressError::Stopped),
        }
        if !signal.is_retired() {
            std::thread::park_timeout(std::time::Duration::from_millis(1));
        }
    }

    Ok(())
}

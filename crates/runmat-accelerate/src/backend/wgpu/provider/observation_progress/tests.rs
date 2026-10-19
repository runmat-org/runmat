use super::*;

fn isolated(limit: u32) -> (NativeObservationProgress, std::sync::mpsc::Receiver<Job>) {
    let (sender, receiver) = sync_channel(limit as usize);
    (
        NativeObservationProgress {
            sender,
            state: Arc::new(State {
                used: AtomicU32::new(0),
                limit,
                stopped: Arc::new(AtomicBool::new(false)),
            }),
        },
        receiver,
    )
}

#[test]
fn reservation_capacity_is_nonblocking_and_uncommitted_drop_releases() {
    let (progress, _receiver) = isolated(1);
    let reservation = progress.reserve().unwrap();
    assert!(matches!(
        progress.reserve(),
        Err(WgpuObservationProgressError::Capacity)
    ));
    drop(reservation);
    assert_eq!(progress.state.used.load(Ordering::Acquire), 0);
    drop(progress.reserve().unwrap());
}

#[test]
fn stopped_progress_never_accepts_new_observed_submissions() {
    let (progress, receiver) = isolated(1);
    progress.state.stopped.store(true, Ordering::Release);
    drop(receiver);
    assert!(matches!(
        progress.reserve(),
        Err(WgpuObservationProgressError::Stopped)
    ));
    assert_eq!(progress.state.used.load(Ordering::Acquire), 0);
}

#[test]
fn failure_cleanup_runs_once_and_does_not_unwind_through_sender() {
    let count = Arc::new(AtomicU32::new(0));
    let observed = count.clone();
    drop(Completion(Some(Box::new(move |status| {
        assert_eq!(status, Err(WgpuObservationProgressError::Stopped));
        observed.fetch_add(1, Ordering::AcqRel);
        panic!("injected device-loss cleanup failure");
    }))));
    assert_eq!(count.load(Ordering::Acquire), 1);
    let observed = count.clone();
    let mut successful = Completion(Some(Box::new(move |status| {
        assert_eq!(status, Ok(()));
        observed.fetch_add(1, Ordering::AcqRel);
    })));
    successful.finish(Ok(()));
    drop(successful);
    assert_eq!(count.load(Ordering::Acquire), 2);
}

#[test]
fn rejected_commit_invokes_completion_once_and_releases_capacity() {
    for stopped in [false, true] {
        let state = Arc::new(State {
            used: AtomicU32::new(1),
            limit: 1,
            stopped: Arc::new(AtomicBool::new(stopped)),
        });
        let completed = Arc::new(AtomicU32::new(0));
        let observed = completed.clone();
        let (sender, receiver) = sync_channel(1);
        drop(receiver);
        let result = commit_job(
            &sender,
            Job {
                submission: 11_u32,
                completion: Completion(Some(Box::new(move |status| {
                    assert_eq!(status, Err(WgpuObservationProgressError::Stopped));
                    observed.fetch_add(1, Ordering::AcqRel);
                }))),
                _slot: Slot(state.clone()),
            },
        );
        assert_eq!(result, Err(WgpuObservationProgressError::Stopped));
        assert_eq!(completed.load(Ordering::Acquire), 1);
        assert_eq!(state.used.load(Ordering::Acquire), 0);
    }
}

#[test]
fn worker_poll_failure_or_device_loss_fails_and_drains_bounded_queue() {
    for panic_poll in [false, true] {
        let state = Arc::new(State {
            used: AtomicU32::new(2),
            limit: 2,
            stopped: Arc::new(AtomicBool::new(false)),
        });
        let failures = Arc::new(AtomicU32::new(0));
        let (sender, receiver) = sync_channel(2);
        for submission in [11_u32, 12] {
            let failed = failures.clone();
            sender
                .send(Job {
                    submission,
                    completion: Completion(Some(Box::new(move |status| {
                        assert!(status.is_err());
                        failed.fetch_add(1, Ordering::AcqRel);
                    }))),
                    _slot: Slot(state.clone()),
                })
                .unwrap_or_else(|_| panic!("test queue available"));
        }
        drop(sender);
        let mut polled = Vec::new();
        drive(receiver, &state, |submission| {
            polled.push(*submission);
            if panic_poll {
                panic!("injected device poll failure");
            }
            state.stopped.store(true, Ordering::Release); // device-lost callback
        });
        assert_eq!(polled, vec![11]);
        assert_eq!(failures.load(Ordering::Acquire), 2);
        assert_eq!(state.used.load(Ordering::Acquire), 0);
    }
}

#[test]
fn sender_shutdown_drains_without_failing_successful_jobs() {
    let state = Arc::new(State {
        used: AtomicU32::new(2),
        limit: 2,
        stopped: Arc::new(AtomicBool::new(false)),
    });
    let failures = Arc::new(AtomicU32::new(0));
    let (sender, receiver) = sync_channel(2);
    for submission in [1_u32, 2] {
        let failed = failures.clone();
        sender
            .send(Job {
                submission,
                completion: Completion(Some(Box::new(move |status| {
                    if status.is_err() {
                        failed.fetch_add(1, Ordering::AcqRel);
                    }
                }))),
                _slot: Slot(state.clone()),
            })
            .unwrap_or_else(|_| panic!("test queue available"));
    }
    drop(sender);
    let mut completed = Vec::new();
    drive(receiver, &state, |submission| completed.push(*submission));
    assert_eq!(completed, vec![1, 2]);
    assert_eq!(failures.load(Ordering::Acquire), 0);
    assert_eq!(state.used.load(Ordering::Acquire), 0);
    assert!(state.stopped.load(Ordering::Acquire));
}

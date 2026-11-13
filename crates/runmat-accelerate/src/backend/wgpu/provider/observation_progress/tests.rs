use super::*;

fn isolated(
    limit: u32,
) -> (
    NativeObservationProgress,
    std::sync::mpsc::Receiver<Command>,
) {
    let (sender, receiver) = sync_channel(limit as usize);
    let state = Arc::new(State {
        used: AtomicU32::new(0),
        limit,
        stopped: Arc::new(AtomicBool::new(false)),
        scoped_emission: Mutex::new(()),
        scoped_work: Mutex::new(Vec::new()),
        worker_closed: AtomicBool::new(false),
        retained_uncertain: AtomicBool::new(false),
        shutdown_requested: AtomicBool::new(false),
        access: None,
        health: None,
    });
    (
        NativeObservationProgress {
            control: Arc::new(Control {
                sender,
                state: state.clone(),
            }),
            state,
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
fn scoped_errors_and_lost_wakes_retain_owner_until_actual_proof() {
    let (progress, receiver) = isolated(2);
    let unused = progress.reserve_slot().unwrap();
    let reserved = progress.reserve_slot().unwrap();
    let signal = crate::backend::wgpu::dispatch::observation::ScopedRetirement::prepare().unwrap();
    let retained = Arc::new(());
    let weak = Arc::downgrade(&retained);
    progress
        .state
        .scoped_work
        .lock()
        .unwrap()
        .push(Some(ScopedWork {
            pending: None,
            probe: Some(retained),
            signal: signal.clone(),
            ready: true,
            _slot: reserved.slot,
        }));
    drop(receiver);
    assert!(progress.control.sender.try_send(Command::Wake).is_err());
    let mut job = progress.state.next_scoped().unwrap();
    job.completion
        .finish(Err(WgpuObservationProgressError::Stopped));
    drop(job);
    assert!(
        weak.upgrade().is_some(),
        "progress error is not retirement proof"
    );
    assert_eq!(progress.state.used.load(Ordering::Acquire), 2);
    assert!(signal.proof().is_none());
    let job = progress.state.next_scoped().unwrap();
    drop(job); // Worker/queued completion destruction is not physical proof.
    assert!(weak.upgrade().is_some());
    assert!(signal.proof().is_none());
    progress.state.retain_uncertain();
    assert!(progress.state.retained_uncertain.load(Ordering::Acquire));
    signal.queue_retired();
    let mut job = progress.state.next_scoped().unwrap();
    job.completion.finish(Ok(()));
    drop(job);
    assert!(weak.upgrade().is_none());
    assert_eq!(progress.state.used.load(Ordering::Acquire), 1);
    drop(unused);
    assert_eq!(progress.state.used.load(Ordering::Acquire), 0);
}

#[test]
fn fatal_progress_retains_charged_owner_and_shutdown_does_not_wait_for_proof() {
    let (progress, receiver) = isolated(2);
    let unused = progress.reserve_slot().unwrap();
    let reserved = progress.reserve_slot().unwrap();
    let signal = crate::backend::wgpu::dispatch::observation::ScopedRetirement::prepare().unwrap();
    let retained = Arc::new(());
    let weak = Arc::downgrade(&retained);
    progress
        .state
        .scoped_work
        .lock()
        .unwrap()
        .push(Some(ScopedWork {
            pending: None,
            probe: Some(retained),
            signal: signal.clone(),
            ready: true,
            _slot: reserved.slot,
        }));
    // Exercise the actual worker unwind boundary once, not a fabricated loss.
    drive(
        [progress.state.next_scoped().unwrap()],
        &progress.state,
        |_| panic!("fatal driver poll witness"),
    );
    assert!(progress.state.retained_uncertain.load(Ordering::Acquire));
    assert!(!progress.is_available());
    assert!(signal.proof().is_none());
    assert!(weak.upgrade().is_some());
    assert_eq!(progress.state.used.load(Ordering::Acquire), 2);
    progress.control.stop();
    let owner = progress.state.clone();
    drive(
        commands(receiver, &progress.state, move || owner.next_scoped()),
        &progress.state,
        |_| panic!("closing must not repoll unproved work"),
    );
    assert!(progress.state.worker_closed.load(Ordering::Acquire));
    assert!(
        weak.upgrade().is_some(),
        "unproved physical owner stays charged"
    );
    drop(unused);
    assert_eq!(progress.state.used.load(Ordering::Acquire), 1);
    // Explicit test cleanup uses actual callback proof, never shutdown itself.
    signal.queue_retired();
    let mut job = progress.state.next_scoped().unwrap();
    job.completion.finish(Ok(()));
    drop(job);
    assert!(weak.upgrade().is_none());
    assert_eq!(progress.state.used.load(Ordering::Acquire), 0);
}

#[test]
fn shutdown_drains_only_prearmed_work_not_unused_reservations() {
    let (progress, receiver) = isolated(2);
    let unused = progress.reserve_slot().unwrap();
    let reserved = progress.reserve_slot().unwrap();
    let signal = crate::backend::wgpu::dispatch::observation::ScopedRetirement::prepare().unwrap();
    let retained = Arc::new(());
    let weak = Arc::downgrade(&retained);
    progress
        .state
        .scoped_work
        .lock()
        .unwrap()
        .push(Some(ScopedWork {
            pending: None,
            probe: Some(retained),
            signal,
            ready: true,
            _slot: reserved.slot,
        }));
    progress.control.stop();
    let owner = progress.state.clone();
    drive(
        commands(receiver, &progress.state, move || owner.next_scoped()),
        &progress.state,
        |target| {
            let WaitTarget::Scoped(signal) = target else {
                panic!("only prearmed scoped work")
            };
            assert!(weak.upgrade().is_some());
            signal.queue_retired();
            Ok(())
        },
    );
    assert!(weak.upgrade().is_none());
    assert_eq!(progress.state.used.load(Ordering::Acquire), 1);
    drop(unused);
    assert_eq!(progress.state.used.load(Ordering::Acquire), 0);
}

#[test]
fn readback_reserves_before_registration_and_commit_races_retire_once() {
    for stopped_after_reservation in [false, true] {
        let (progress, receiver) = isolated(1);
        let reservation = progress.reserve_slot().unwrap();
        assert!(matches!(
            progress.reserve_slot(),
            Err(WgpuObservationProgressError::Capacity)
        ));
        let retained = Arc::new(());
        let weak = Arc::downgrade(&retained);
        let statuses = Arc::new(std::sync::Mutex::new(Vec::new()));
        let observed = statuses.clone();
        if stopped_after_reservation {
            progress.state.stopped.store(true, Ordering::Release);
        }
        let result = reservation.complete_queue_tail(Box::new(move |status| {
            observed.lock().unwrap().push(status);
            drop(retained);
        }));
        if stopped_after_reservation {
            assert_eq!(result, Err(WgpuObservationProgressError::Stopped));
            assert!(weak.upgrade().is_none());
            assert_eq!(
                *statuses.lock().unwrap(),
                vec![Err(WgpuObservationProgressError::Stopped)]
            );
        } else {
            result.unwrap();
            // Logical cancellation does not remove the queued physical owner.
            assert!(weak.upgrade().is_some());
            assert!(statuses.lock().unwrap().is_empty());
            let state = progress.state.clone();
            drop(progress);
            drive(commands(receiver, &state, || None), &state, |submission| {
                assert!(matches!(submission, WaitTarget::Existing(None)));
                Ok(())
            });
            assert!(weak.upgrade().is_none());
            assert_eq!(*statuses.lock().unwrap(), vec![Ok(())]);
            assert_eq!(state.used.load(Ordering::Acquire), 0);
            continue;
        }
        assert_eq!(progress.state.used.load(Ordering::Acquire), 0);
    }
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
    use std::sync::{atomic::AtomicBool, Weak};
    #[derive(Debug)]
    struct Claim(Arc<AtomicBool>);
    impl Drop for Claim {
        fn drop(&mut self) {
            assert!(
                self.0.load(Ordering::Acquire),
                "callback payload must retire before funding"
            );
        }
    }
    struct CallbackPayload {
        claim: Weak<Claim>,
        retired: Arc<AtomicBool>,
    }
    impl Drop for CallbackPayload {
        fn drop(&mut self) {
            assert!(
                self.claim.upgrade().is_some(),
                "callback drop must remain funded"
            );
            self.retired.store(true, Ordering::Release);
        }
    }
    let count = Arc::new(AtomicU32::new(0));
    for unwind in [true, false] {
        let retired = Arc::new(AtomicBool::new(false));
        let claim = Arc::new(Claim(retired.clone()));
        let weak = Arc::downgrade(&claim);
        let payload = CallbackPayload {
            claim: weak.clone(),
            retired: retired.clone(),
        };
        let observed = count.clone();
        let mut completion = Completion::funded(
            Box::new(move |status| {
                let _payload = payload;
                observed.fetch_add(1, Ordering::AcqRel);
                if unwind {
                    assert_eq!(status, Err(WgpuObservationProgressError::Stopped));
                    panic!("injected device-loss cleanup failure");
                }
                assert_eq!(status, Ok(()));
            }),
            Some(claim),
        );
        assert!(weak.upgrade().is_some());
        if unwind {
            drop(completion); // Abandonment callback panics; funding still retires once.
        } else {
            completion.finish(Ok(()));
            completion.finish(Ok(())); // A second finish must not replay.
            drop(completion);
        }
        assert!(retired.load(Ordering::Acquire));
        assert!(weak.upgrade().is_none());
        assert_eq!(count.load(Ordering::Acquire), if unwind { 1 } else { 2 });
    }
}

#[test]
fn rejected_commit_invokes_completion_once_and_releases_capacity() {
    for stopped in [false, true] {
        let state = Arc::new(State {
            used: AtomicU32::new(1),
            limit: 1,
            stopped: Arc::new(AtomicBool::new(stopped)),
            scoped_emission: Mutex::new(()),
            scoped_work: Mutex::new(Vec::new()),
            worker_closed: AtomicBool::new(false),
            retained_uncertain: AtomicBool::new(false),
            shutdown_requested: AtomicBool::new(false),
            access: None,
            health: None,
        });
        let completed = Arc::new(AtomicU32::new(0));
        let observed = completed.clone();
        let (sender, receiver) = sync_channel(1);
        drop(receiver);
        let result = commit_job(
            &sender,
            Job {
                submission: 11_u32,
                completion: Completion::ordinary(Box::new(move |status| {
                    assert_eq!(status, Err(WgpuObservationProgressError::Stopped));
                    observed.fetch_add(1, Ordering::AcqRel);
                })),
                _slot: Some(Slot(state.clone())),
                retire_after_stop: false,
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
            scoped_emission: Mutex::new(()),
            scoped_work: Mutex::new(Vec::new()),
            worker_closed: AtomicBool::new(false),
            retained_uncertain: AtomicBool::new(false),
            shutdown_requested: AtomicBool::new(false),
            access: None,
            health: None,
        });
        let failures = Arc::new(AtomicU32::new(0));
        let (sender, receiver) = sync_channel(2);
        for submission in [11_u32, 12] {
            let failed = failures.clone();
            sender
                .send(Job {
                    submission,
                    completion: Completion::ordinary(Box::new(move |status| {
                        assert!(status.is_err());
                        failed.fetch_add(1, Ordering::AcqRel);
                    })),
                    _slot: Some(Slot(state.clone())),
                    retire_after_stop: false,
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
            Ok(())
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
        scoped_emission: Mutex::new(()),
        scoped_work: Mutex::new(Vec::new()),
        worker_closed: AtomicBool::new(false),
        retained_uncertain: AtomicBool::new(false),
        shutdown_requested: AtomicBool::new(false),
        access: None,
        health: None,
    });
    let failures = Arc::new(AtomicU32::new(0));
    let (sender, receiver) = sync_channel(2);
    for submission in [1_u32, 2] {
        let failed = failures.clone();
        sender
            .send(Job {
                submission,
                completion: Completion::ordinary(Box::new(move |status| {
                    if status.is_err() {
                        failed.fetch_add(1, Ordering::AcqRel);
                    }
                })),
                _slot: Some(Slot(state.clone())),
                retire_after_stop: false,
            })
            .unwrap_or_else(|_| panic!("test queue available"));
    }
    drop(sender);
    let mut completed = Vec::new();
    drive(receiver, &state, |submission| {
        completed.push(*submission);
        Ok(())
    });
    assert_eq!(completed, vec![1, 2]);
    assert_eq!(failures.load(Ordering::Acquire), 0);
    assert_eq!(state.used.load(Ordering::Acquire), 0);
    assert!(state.stopped.load(Ordering::Acquire));
}

#[test]
fn readiness_does_not_retire_the_physically_running_callback_tail() {
    // Controlled channel handshakes, not wall-clock timing or a physical GPU.
    let (progress, receiver) = isolated(1);
    let retained = Arc::new(());
    let weak = Arc::downgrade(&retained);
    let (ready, ready_rx) = std::sync::mpsc::channel();
    let (release, release_rx) = std::sync::mpsc::channel();
    progress
        .complete_queue_tail(Box::new(move |status| {
            assert_eq!(status, Ok(()));
            ready.send(()).unwrap();
            release_rx.recv().unwrap();
            drop(retained);
        }))
        .unwrap();
    let state = progress.state.clone();
    drop(progress);
    let worker_state = state.clone();
    let worker = std::thread::spawn(move || {
        drive(
            commands(receiver, &worker_state, || None),
            &worker_state,
            |_| Ok(()),
        )
    });
    ready_rx.recv().unwrap();
    assert!(
        weak.upgrade().is_some(),
        "readiness cannot release the callback's retained backing"
    );
    assert_eq!(state.used.load(Ordering::Acquire), 1);
    release.send(()).unwrap();
    worker.join().unwrap();
    assert!(weak.upgrade().is_none());
    assert_eq!(state.used.load(Ordering::Acquire), 0);
}

#[test]
fn explicit_shutdown_retires_queued_owners_with_a_live_sender() {
    for full in [false, true] {
        let (progress, receiver) = isolated(1);
        let completed = Arc::new(AtomicU32::new(0));
        if full {
            let observed = completed.clone();
            progress
                .complete_queue_tail(Box::new(move |status| {
                    assert_eq!(status, Err(WgpuObservationProgressError::Stopped));
                    observed.fetch_add(1, Ordering::AcqRel);
                }))
                .unwrap();
        }
        progress.control.stop();
        // Keep progress (and its sender) live. A full queue cannot accept the
        // wake command, and an empty queue can: both must terminate normally.
        drive(
            commands(receiver, &progress.state, || None),
            &progress.state,
            |_| {
                panic!("shutdown must not start another device poll");
            },
        );
        assert_eq!(completed.load(Ordering::Acquire), u32::from(full));
        assert_eq!(progress.state.used.load(Ordering::Acquire), 0);
        assert!(progress.reserve_slot().is_err());
    }
}

use super::*;
#[cfg(not(target_arch = "wasm32"))]
use std::sync::Arc;

#[test]
fn provider_and_command_scope_swaps_fail_all_selected_owners() {
    use runmat_accelerate_api::observation::{
        ProviderObservationFailure, ProviderOperationReceipt,
    };
    use runmat_execution::profiling::CommandProfilingRoot;

    let provider_a = ProviderOperationReceipt::new();
    let provider_b = ProviderOperationReceipt::new();
    let recorder_a = provider_a.recorder();
    let recorder_b = provider_b.recorder();
    assert!(recorder_a.same_observation(&provider_a.recorder()));
    assert!(!recorder_a.same_observation(&recorder_b));

    let command_a = CommandProfilingRoot::new();
    let command_b = CommandProfilingRoot::new();
    let owner_a = command_a.receipt();
    let owner_b = command_b.receipt();
    assert!(!fail_mismatched_admission(
        &ProviderAdmission::Failed(recorder_a.clone()),
        Some(&recorder_b),
        Some(&owner_a),
        Some(&owner_b),
    ));
    assert_eq!(
        provider_a.status(),
        runmat_accelerate_api::observation::ProviderObservationStatus::Failed(
            ProviderObservationFailure::ExecutionUnproven
        )
    );
    assert_eq!(provider_a.status(), provider_b.status());
    assert!(command_a.seal().unproven_device_route);
    assert!(command_b.seal().unproven_device_route);

    let provider = ProviderOperationReceipt::new();
    let recorder = provider.recorder();
    let command = CommandProfilingRoot::new();
    let owner = command.receipt();
    assert!(!fail_mismatched_admission(
        &ProviderAdmission::Failed(recorder),
        None,
        Some(&owner),
        Some(&owner),
    ));
    assert!(command.seal().unproven_device_route);

    let command_a = CommandProfilingRoot::new();
    let command_b = CommandProfilingRoot::new();
    let owner_a = command_a.receipt();
    let owner_b = command_b.receipt();
    assert!(!fail_mismatched_admission(
        &ProviderAdmission::Absent,
        None,
        Some(&owner_a),
        Some(&owner_b),
    ));
    assert!(command_a.seal().unproven_device_route);
    assert!(command_b.seal().unproven_device_route);
}

#[cfg(not(target_arch = "wasm32"))]
#[test]
fn missing_native_progress_refuses_provider_before_any_query_admission() {
    use runmat_accelerate_api::observation::{
        ProviderObservationFailure, ProviderObservationStatus, ProviderOperationReceipt,
    };

    let refused = ProviderOperationReceipt::new();
    let admission = prepare(refused.recorder());
    assert!(matches!(admission, ProviderAdmission::Failed(_)));
    assert_eq!(
        refused.status(),
        ProviderObservationStatus::Failed(ProviderObservationFailure::ProviderFailed)
    );
}

#[cfg(not(target_arch = "wasm32"))]
#[test]
fn canonical_wgpu_source_rejects_new_raw_queue_or_mapped_initializer() {
    use std::path::Path;

    fn inspect(path: &Path) {
        for entry in std::fs::read_dir(path).expect("WGPU source subtree") {
            let entry = entry.expect("WGPU source entry");
            let path = entry.path();
            if path.is_dir() {
                inspect(&path);
                continue;
            }
            if path.extension().is_none_or(|extension| extension != "rs")
                || path
                    .file_name()
                    .is_some_and(|name| name.to_string_lossy().contains("test"))
            {
                continue;
            }
            let source = std::fs::read_to_string(&path).expect("WGPU production source");
            let code: String = source
                .lines()
                .filter(|line| {
                    let line = line.trim_start();
                    !line.starts_with("//") && !line.starts_with('*')
                })
                .map(|line| line.split("//").next().unwrap_or_default())
                .flat_map(|line| line.chars().filter(|character| !character.is_whitespace()))
                .collect();
            for forbidden in [
                ".create_buffer_init(",
                ".queue()",
                "=emission.device();",
                ".device().clone()",
                ".device()).clone()",
                "Clone::clone(emission.device()",
                "returnemission.device();",
            ] {
                assert!(
                    !code.contains(forbidden),
                    "new WGPU driver escape in {}: {forbidden}",
                    path.display()
                );
            }
            if !path.ends_with("dispatch/observation.rs") {
                assert!(
                    !code.contains(".submit_profiled("),
                    "physical WGPU submit bypasses PendingSubmissions in {}",
                    path.display()
                );
            } else {
                assert_eq!(
                    code.matches(".submit_profiled(").count(),
                    1,
                    "one canonical physical submit ingress in {}",
                    path.display()
                );
            }
        }
    }

    inspect(&Path::new(env!("CARGO_MANIFEST_DIR")).join("src/backend/wgpu"));
}

#[test]
fn fence_size_is_mapping_aligned_and_small_not_payload_dependent() {
    assert_eq!(FENCE_REQUESTED_BYTES, 8);
    assert_eq!(FENCE_REQUESTED_BYTES % wgpu::MAP_ALIGNMENT, 0);
}

#[test]
fn prospective_query_bounds_cumulative_sets_separately_from_byte_peak() {
    let query = prospective_encoder_query(2, 3).expect("known physical encoder route");
    assert_eq!(query.max_compute_encoders, 2);
    assert_eq!(query.max_total_encoders, 3);
    assert_eq!(query.max_query_sets, 3);
    assert_eq!(query.max_timestamp_slots, 54);
    assert_eq!(query.requested_peak_buffer_bytes, 864);
    assert_eq!(query.max_map_callbacks, 3);
    assert!(prospective_encoder_query(0, 0).is_none());
    assert!(prospective_encoder_query(2, 1).is_none());
    assert!(prospective_encoder_query(u32::MAX, u32::MAX).is_some());
}

#[test]
fn capability_masks_optional_queries_without_device_support_and_caps_only_peak() {
    use runmat_accelerate_api::observation::{
        ProviderObservationCapability, ProviderObservationOverhead,
    };
    let budget = ProviderObservationBudget::with_wgpu_query_bytes(
        std::num::NonZeroU32::new(1).unwrap(),
        2 * MAX_QUERY_REQUESTED_BYTES,
    );
    let candidate = ProviderObservationCapability::Supported(ProviderObservationOverhead {
        submissions: 4,
        requested_peak_buffer_bytes: 32,
        clear_commands: 4,
        map_callbacks: 4,
        wgpu_compute_query: prospective_encoder_query(3, 4),
    });
    let unsupported = gate_prospective_encoder_query(candidate, None, &budget);
    let ProviderObservationCapability::Supported(unsupported) = unsupported else {
        panic!("mandatory capability must remain supported");
    };
    assert_eq!(unsupported.requested_peak_buffer_bytes, 32);
    assert!(unsupported.wgpu_compute_query.is_none());
    let admitted =
        gate_prospective_encoder_query(candidate, TimestampCapability::new(2.0, false), &budget);
    let ProviderObservationCapability::Supported(admitted) = admitted else {
        panic!("compute capability remains supported");
    };
    let query = admitted.wgpu_compute_query.unwrap();
    assert_eq!(query.max_query_sets, 3);
    assert_eq!(query.max_timestamp_slots, 48);
    assert_eq!(query.max_map_callbacks, 3);
    assert_eq!(
        query.requested_peak_buffer_bytes,
        2 * compute_pass::MAX_COMPUTE_QUERY_REQUESTED_BYTES
    );
    assert_eq!(admitted.requested_peak_buffer_bytes, 32);
}

#[test]
fn command_encoder_feature_accounts_copy_only_encoder_without_double_query_set() {
    use runmat_accelerate_api::observation::{
        ProviderObservationCapability, ProviderObservationOverhead,
    };
    use runmat_execution::profiling::{with_command_profiling, CommandProfilingRoot};
    let owner = CommandProfilingRoot::new();
    let budget = ProviderObservationBudget::with_wgpu_query_bytes(
        std::num::NonZeroU32::new(1).unwrap(),
        MAX_QUERY_REQUESTED_BYTES,
    );
    let candidate = ProviderObservationCapability::Supported(ProviderObservationOverhead {
        submissions: 2,
        requested_peak_buffer_bytes: 16,
        clear_commands: 2,
        map_callbacks: 2,
        wgpu_compute_query: prospective_encoder_query(1, 2),
    });
    let admitted = with_command_profiling(Some(owner.receipt()), || {
        gate_prospective_encoder_query(candidate, TimestampCapability::new(1.0, true), &budget)
    });
    let ProviderObservationCapability::Supported(admitted) = admitted else {
        panic!("mandatory submission plan remains admitted")
    };
    let query = admitted.wgpu_compute_query.expect("one shared query owner");
    assert_eq!(query.max_compute_encoders, 1);
    assert_eq!(query.max_total_encoders, 2);
    assert_eq!(query.max_query_sets, 2);
    assert_eq!(query.max_timestamp_slots, 36);
    assert_eq!(query.max_map_callbacks, 2);
    assert_eq!(query.requested_peak_buffer_bytes, MAX_QUERY_REQUESTED_BYTES);
    assert_eq!(admitted.requested_peak_buffer_bytes, 16);
}

#[cfg(not(target_arch = "wasm32"))]
#[test]
#[ignore = "requires actual WGPU device; run explicitly, never an unavailable-device pass"]
fn actual_private_fences_survive_cancel_and_unrelated_queue_submission() {
    use crate::backend::wgpu::provider::{WgpuProvider, WgpuProviderOptions};
    use runmat_accelerate_api::observation::{
        with_provider_observation, ProviderObservationFailure, ProviderObservationStatus,
        ProviderObservationSupport,
    };
    use runmat_accelerate_api::{
        AccelContextHandle, AccelContextKind, AccelProvider, ProviderElementType,
        ProviderFeasibilityQuery, ProviderLayout, ProviderOperationFamily,
        ProviderOperationIdentity, ProviderRepresentation, ProviderResidency, ProviderStorage,
    };
    use runmat_execution::profiling::{
        with_command_profiling, CommandProfilingRoot, PhysicalMetricCoverage,
        PhysicalSubmissionCounts,
    };
    let provider = WgpuProvider::new(WgpuProviderOptions::default()).expect("required WGPU device");
    let AccelContextHandle::Wgpu(context) =
        provider.export_context(AccelContextKind::Plotting).unwrap();
    let emit_empty = || {
        let mut pending = PendingSubmissions::new(context.access.clone());
        let ticket = pending.arm_probe_completion();
        context
            .access
            .with_shared(|emission| {
                let encoder = pending.profiled_encoder(emission, &Default::default());
                pending.submit(emission, encoder)
            })
            .expect("emission available")
            .expect("same device");
        pending.commit();
        ticket
    };
    let value = ProviderRepresentation {
        element_type: ProviderElementType::F32,
        storage: ProviderStorage::DenseReal,
        layout: ProviderLayout::ColumnMajorContiguous,
        residency: ProviderResidency::Device,
        shape: vec![1, 1],
    };
    let query = ProviderFeasibilityQuery {
        operation: ProviderOperationIdentity::new("fusion.elementwise"),
        family: ProviderOperationFamily::Fusion,
        inputs: vec![value.clone()],
        outputs: vec![value],
        workload: Default::default(),
    };
    let begin = || match provider.begin_operation_observation(&query) {
        ProviderObservationSupport::Observed(receipt) => receipt,
        ProviderObservationSupport::Unsupported => panic!("required native completion driver"),
    };
    let mut first = begin();
    let first_ticket = with_provider_observation(Some(first.recorder()), emit_empty);
    assert!(matches!(
        first_ticket.state(),
        ProbeFenceState::Pending | ProbeFenceState::Completed
    ));
    first.fail(ProviderObservationFailure::Cancelled);
    // Unrelated work uses the same access authority but no observation helper;
    // measurement cannot depend on the newest global queue prefix.
    context
        .access
        .with_shared(|emission| {
            emission
                .queue()
                .submit(Some(emission.encode("unrelated", |_| {})));
        })
        .unwrap();
    assert_eq!(
        first.take_completed().unwrap().status,
        ProviderObservationStatus::Failed(ProviderObservationFailure::Cancelled)
    );
    let mut next = begin();
    let command = CommandProfilingRoot::new();
    let next_ticket = with_command_profiling(Some(command.receipt()), || {
        with_provider_observation(Some(next.recorder()), emit_empty)
    });
    assert_eq!(next_ticket.fence_requested_bytes(), Some(8));
    next.publish().unwrap();
    // No second caller poll/submit is needed to drive native completion.
    struct Signal(std::sync::mpsc::SyncSender<()>);
    impl std::task::Wake for Signal {
        fn wake(self: Arc<Self>) {
            let _ = self.0.try_send(());
        }
    }
    let (sender, receiver) = std::sync::mpsc::sync_channel(1);
    let waker = std::task::Waker::from(Arc::new(Signal(sender)));
    let mut cx = std::task::Context::from_waker(&waker);
    let evidence = match next.poll_completed(&mut cx) {
        std::task::Poll::Ready(Some(evidence)) => evidence,
        std::task::Poll::Pending => {
            receiver
                .recv_timeout(std::time::Duration::from_secs(10))
                .expect("native progress must wake without caller poll");
            match next.poll_completed(&mut cx) {
                std::task::Poll::Ready(Some(evidence)) => evidence,
                _ => panic!("completion wake must carry evidence"),
            }
        }
        _ => panic!("undelivered receipt"),
    };
    assert!(matches!(
        evidence.status,
        ProviderObservationStatus::Completed { submissions: 1, .. }
    ));
    assert_eq!(next_ticket.state(), ProbeFenceState::Completed);
    let physical = command.seal();
    assert_eq!(
        physical.physical_submissions,
        Some(PhysicalSubmissionCounts {
            incomplete: 1,
            ..PhysicalSubmissionCounts::default()
        }),
        "the submitted empty encoder has no device timestamp"
    );
    assert_eq!(
        physical.physical_metric_coverage,
        Some(PhysicalMetricCoverage {
            gpu_execution: false,
            kernel_count: true,
        })
    );
    assert_eq!(
        physical.physical_attributed_metrics.unwrap().gpu_execution,
        None
    );
    assert_eq!(
        physical.physical_attributed_metrics.unwrap().kernel_count,
        Some(0),
        "the empty provider encoder is one physical submission, not one provider operation"
    );
    assert_eq!(physical.kernel_count, None, "route coverage is incomplete");
    let absent = emit_empty();
    assert_eq!(absent.state(), ProbeFenceState::Absent);
    assert_eq!(absent.fence_requested_bytes(), None);
    let no_progress = CommandProfilingRoot::new();
    let _ = with_command_profiling(Some(no_progress.receipt()), emit_empty);
    let missing = no_progress.seal();
    assert_eq!(missing.physical_submissions.unwrap().missing, 1);
    assert_eq!(
        missing.physical_attributed_metrics.unwrap().kernel_count,
        None
    );
    assert_eq!(missing.kernel_count, None);
}

#[cfg(not(target_arch = "wasm32"))]
#[test]
#[ignore = "requires timestamp-capable WGPU device and native progress driver"]
fn scope_swapped_direct_queries_retain_quota_until_device_retirement() {
    use crate::backend::wgpu::provider::{WgpuProvider, WgpuProviderOptions};
    use runmat_accelerate_api::observation::{
        with_provider_observation, ProviderObservationFailure, ProviderObservationStatus,
        ProviderObservationSupport, ProviderOperationReceipt, WgpuObservationProgress,
        WgpuObservationProgressError, WgpuObservationProgressReservation,
    };
    use runmat_accelerate_api::{
        AccelContextHandle, AccelContextKind, AccelProvider, ProviderElementType,
        ProviderFeasibilityQuery, ProviderLayout, ProviderOperationFamily,
        ProviderOperationIdentity, ProviderRepresentation, ProviderResidency, ProviderStorage,
    };
    use runmat_execution::profiling::{with_command_profiling, CommandProfilingRoot};

    type Callback = runmat_accelerate_api::observation::WgpuObservationCompletion;
    struct Held {
        reservation: Box<dyn WgpuObservationProgressReservation>,
        submission: wgpu::SubmissionIndex,
        callback: Callback,
    }
    struct Gate {
        port: Arc<dyn WgpuObservationProgress>,
        held: Arc<std::sync::Mutex<Vec<Held>>>,
        panic_on_submit: bool,
    }
    struct GatedReservation {
        reservation: Box<dyn WgpuObservationProgressReservation>,
        held: Arc<std::sync::Mutex<Vec<Held>>>,
        panic_on_submit: bool,
    }
    struct RetentionDrop(Arc<std::sync::atomic::AtomicUsize>);
    impl Drop for RetentionDrop {
        fn drop(&mut self) {
            self.0.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        }
    }
    impl WgpuObservationProgress for Gate {
        fn reserve(
            &self,
        ) -> Result<Box<dyn WgpuObservationProgressReservation>, WgpuObservationProgressError>
        {
            Ok(Box::new(GatedReservation {
                reservation: self.port.reserve()?,
                held: self.held.clone(),
                panic_on_submit: self.panic_on_submit,
            }))
        }
    }
    impl WgpuObservationProgressReservation for GatedReservation {
        fn submit(
            self: Box<Self>,
            submission: wgpu::SubmissionIndex,
            callback: Callback,
        ) -> Result<(), WgpuObservationProgressError> {
            let Self {
                reservation,
                held,
                panic_on_submit,
            } = *self;
            if panic_on_submit {
                // Unwinding discards the accepted callback without invoking
                // it. Completion::Drop must retire the exact submission before
                // its query permit can be refunded.
                panic!("simulated progress reservation failure");
            }
            held.lock().unwrap().push(Held {
                reservation,
                submission,
                callback,
            });
            Ok(())
        }
    }

    let provider = WgpuProvider::new(WgpuProviderOptions::default()).expect("WGPU device");
    let AccelContextHandle::Wgpu(context) =
        provider.export_context(AccelContextKind::Plotting).unwrap();
    let features = context
        .access
        .with_shared(|emission| emission.device().features())
        .unwrap();
    assert!(features.contains(wgpu::Features::TIMESTAMP_QUERY));
    let value = ProviderRepresentation {
        element_type: ProviderElementType::F32,
        storage: ProviderStorage::DenseReal,
        layout: ProviderLayout::ColumnMajorContiguous,
        residency: ProviderResidency::Device,
        shape: vec![1, 1],
    };
    let query = ProviderFeasibilityQuery {
        operation: ProviderOperationIdentity::new("fusion.elementwise"),
        family: ProviderOperationFamily::Fusion,
        inputs: vec![value.clone()],
        outputs: vec![value],
        workload: Default::default(),
    };
    let progress = match provider.begin_operation_observation(&query) {
        ProviderObservationSupport::Observed(receipt) => receipt
            .recorder()
            .wgpu_progress()
            .expect("native provider progress"),
        ProviderObservationSupport::Unsupported => panic!("native progress required"),
    };

    for variant in 0..12 {
        let scope_case = variant % 4;
        let stopped = variant >= 4;
        let panicking = variant >= 10;
        let matched_provider = variant == 8 || variant == 10;
        let matched_direct = variant == 9 || variant == 11;
        let budget = Arc::new(ProviderObservationBudget::with_wgpu_query_bytes(
            std::num::NonZeroU32::new(1).unwrap(),
            MAX_QUERY_REQUESTED_BYTES,
        ));
        let gate = Arc::new(Gate {
            port: progress.clone(),
            held: Arc::new(std::sync::Mutex::new(Vec::new())),
            panic_on_submit: panicking,
        });
        let mut pending = PendingSubmissions::new(context.access.clone()).with_profiling(
            TimestampCapability::new(1.0, false),
            budget.clone(),
            Some(gate.clone()),
        );
        let provider_a = ProviderOperationReceipt::with_wgpu_progress(budget.clone(), gate.clone());
        let retention_drops = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let retirement = if matched_direct {
            // A real direct encoder has no selected provider recorder. Keep
            // its exact notifier, but obtain progress through the canonical
            // recorder-independent resource admission rather than provider_a.
            let (retention, retirement) = super::submitted_retention::with_retirement_for_test(
                SubmittedAllocationRetention::new(RetentionDrop(retention_drops.clone())),
            )
            .unwrap();
            pending.retain_submission_resources(retention).unwrap();
            retirement
        } else {
            let (retention_completion, retirement) =
                with_provider_observation(Some(provider_a.recorder()), || {
                    PreparedSubmittedAllocationRetention::reserve().unwrap()
                });
            pending
                .retain_allocations_until_next_submit(
                    retention_completion,
                    SubmittedAllocationRetention::new(RetentionDrop(retention_drops.clone())),
                )
                .unwrap();
            retirement
        };
        let command_a = CommandProfilingRoot::new();
        let command_b = CommandProfilingRoot::new();
        let provider_b = match provider.begin_operation_observation(&query) {
            ProviderObservationSupport::Observed(receipt) => receipt,
            ProviderObservationSupport::Unsupported => panic!("native progress required"),
        };
        context
            .access
            .with_shared(|emission| {
                let mut encoder = with_command_profiling(Some(command_a.receipt()), || {
                    if (scope_case == 3 && !matched_direct) || matched_provider {
                        with_provider_observation(Some(provider_a.recorder()), || {
                            pending.profiled_encoder(emission, &Default::default())
                        })
                    } else {
                        pending.profiled_encoder(emission, &Default::default())
                    }
                });
                // Exercise a real query owner on every supported backend,
                // without requiring unreliable encoder-wide timestamps.
                // No numerical dispatch is needed for this retirement test.
                drop(encoder.begin_compute_pass(&Default::default()));
                assert!(
                    budget
                        .try_reserve_wgpu_query(MAX_QUERY_REQUESTED_BYTES)
                        .is_none(),
                    "variant {variant}: query admission must retain its exact quota before submit"
                );
                match variant {
                    8 | 10 => with_command_profiling(Some(command_a.receipt()), || {
                        with_provider_observation(Some(provider_a.recorder()), || {
                            pending.submit(emission, encoder)
                        })
                    }),
                    9 | 11 => with_command_profiling(Some(command_a.receipt()), || {
                        pending.submit(emission, encoder)
                    }),
                    _ => match scope_case {
                        0 => with_command_profiling(Some(command_b.receipt()), || {
                            pending.submit(emission, encoder)
                        }),
                        1 => pending.submit(emission, encoder),
                        _ => with_provider_observation(Some(provider_b.recorder()), || {
                            pending.submit(emission, encoder)
                        }),
                    },
                }
                .expect("semantic queue submission");
            })
            .unwrap();
        pending.commit();
        assert_eq!(
            retention_drops.load(std::sync::atomic::Ordering::SeqCst),
            usize::from(panicking),
            "an accepted submission retains allocations until its exact progress callback"
        );
        let held = if panicking {
            assert!(gate.held.lock().unwrap().is_empty());
            None
        } else {
            Some(gate.held.lock().unwrap().pop().expect("one terminal owner"))
        };
        if !panicking {
            assert!(
                budget
                    .try_reserve_wgpu_query(MAX_QUERY_REQUESTED_BYTES)
                    .is_none(),
                "scope mismatch cannot refund a query permit before retirement"
            );
        }
        let command_a = if !matched_provider && !matched_direct {
            assert!(command_a.seal().unproven_device_route);
            None
        } else {
            Some(command_a)
        };
        if scope_case == 0 && !matched_provider && !matched_direct {
            assert!(command_b.seal().unproven_device_route);
        }
        if scope_case == 3 && !matched_provider && !matched_direct {
            assert_eq!(
                provider_a.status(),
                ProviderObservationStatus::Failed(ProviderObservationFailure::ExecutionUnproven)
            );
            assert_eq!(provider_a.status(), provider_b.status());
        }
        if let Some(held) = held {
            if stopped {
                // The progress owner may stop after physical submission. Its
                // failure callback must wait for this exact submission before
                // optional query capacity is recycled, without positive evidence.
                drop(held.reservation);
                held.callback
                    .finish(Err(WgpuObservationProgressError::Stopped));
            } else {
                held.reservation
                    .submit(held.submission, held.callback)
                    .expect("native progress accepted");
            }
        }
        // Query permits are released before the allocation bundle within the
        // same physical-completion callback. Await its exact retirement signal,
        // not a concurrent quota observation, before asserting bundle cleanup.
        assert_eq!(
            futures::executor::block_on(retirement).is_ok(),
            !stopped && !panicking,
            "retirement proof follows exact queue progress, never semantic-map completion"
        );
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
        loop {
            if budget
                .try_reserve_wgpu_query(MAX_QUERY_REQUESTED_BYTES)
                .is_some()
            {
                break;
            }
            assert!(
                std::time::Instant::now() < deadline,
                "device retirement stalled"
            );
            std::thread::yield_now();
        }
        assert_eq!(
            retention_drops.load(std::sync::atomic::Ordering::SeqCst),
            1,
            "submitted allocation retention releases exactly once after settlement"
        );
        if matched_provider {
            assert!(matches!(
                provider_a.status(),
                ProviderObservationStatus::Failed(_)
            ));
        }
        if matched_provider || matched_direct {
            let physical = command_a
                .expect("matched command retained through retirement")
                .seal();
            // Dropping the accepted callback delivers a typed progress error,
            // including during unwind. That is known failed observation, not
            // a missing callback or successful physical measurement.
            let submissions = physical.physical_submissions.unwrap();
            assert_eq!(submissions.failed, 1);
            assert_eq!(submissions.missing, 0);
            assert_eq!(physical.kernel_count, None);
        }
    }
}

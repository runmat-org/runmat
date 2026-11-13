//! A real opened native WAVE source with synthetic, enforcing test quotas.
//! This is a protocol witness, not a production peak-memory certificate.

use super::*;
use crate::context::host_effect::{
    AudioDecodeOutput, HostEffectOutputHeadroom, HostEffectRequest, HostEffectResourceAdmission,
    HostEffectResourceEvidence, HostEffectResourceLivenessLease, HostEffectStageAdmission,
    HostEffectStageEvidence, HostEffectStageReservation, HostEffectWorkload,
    ManagedHostEffectLedger, ManagedHostEffectStageReservation,
};
use crate::context::{RuntimeContext, RuntimeHostEffectResourceService, RuntimeServicePorts};
use runmat_execution::{Digest, ResourceByteBound};
use runmat_value::{ExactHostAllocationReservation, ExactHostNumericAllocationError};
use std::cell::Cell;
use std::rc::Rc;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

#[derive(Debug)]
struct TestOutputReservation {
    _managed: Arc<dyn ExactHostAllocationReservation>,
    live_total: Arc<AtomicUsize>,
    live_output: Arc<AtomicUsize>,
    bytes: usize,
}

impl Drop for TestOutputReservation {
    fn drop(&mut self) {
        self.live_output.fetch_sub(self.bytes, Ordering::SeqCst);
        self.live_total.fetch_sub(self.bytes, Ordering::SeqCst);
    }
}
impl ExactHostAllocationReservation for TestOutputReservation {}

#[derive(Debug)]
struct TestOutputHeadroom {
    managed: Arc<dyn HostEffectOutputHeadroom>,
    live_total: Arc<AtomicUsize>,
    live_output: Arc<AtomicUsize>,
}

impl HostEffectOutputHeadroom for TestOutputHeadroom {
    fn reserve_exact(
        &self,
        bytes: usize,
    ) -> Result<Arc<dyn ExactHostAllocationReservation>, ExactHostNumericAllocationError> {
        let managed = self.managed.reserve_exact(bytes)?;
        self.live_total.fetch_add(bytes, Ordering::SeqCst);
        self.live_output.fetch_add(bytes, Ordering::SeqCst);
        Ok(Arc::new(TestOutputReservation {
            _managed: managed,
            live_total: self.live_total.clone(),
            live_output: self.live_output.clone(),
            bytes,
        }))
    }

    fn retire_invocation(&self) {
        self.managed.retire_invocation();
    }
}

struct TestStageReservation {
    managed: Arc<ManagedHostEffectStageReservation>,
    ledger: ManagedHostEffectLedger,
    current: Arc<AtomicBool>,
    live_stage: Arc<AtomicUsize>,
    live_output: Arc<AtomicUsize>,
    live_total: Arc<AtomicUsize>,
}
impl Drop for TestStageReservation {
    fn drop(&mut self) {
        self.live_stage.fetch_sub(128 * 1024, Ordering::SeqCst);
        self.live_total.fetch_sub(128 * 1024, Ordering::SeqCst);
    }
}
impl HostEffectStageReservation for TestStageReservation {
    fn managed_ledger(&self) -> Option<ManagedHostEffectLedger> {
        Some(self.ledger.clone())
    }
    fn reserved_bytes(&self) -> u64 {
        self.managed.reserved_bytes()
    }
    fn capacity_bytes(&self) -> u64 {
        self.managed.capacity_bytes()
    }
    fn promote_scan_envelope(
        &self,
        bytes: u64,
    ) -> Result<(), crate::context::host_effect::HostEffectAllocationError> {
        self.managed.promote_scan_envelope(bytes)
    }
    fn reserve_live_exact(
        &self,
        kind: crate::context::host_effect::HostEffectLiveAllocationKind,
        bytes: usize,
    ) -> Result<
        Arc<dyn crate::context::host_effect::HostEffectLiveAllocationReservation>,
        crate::context::host_effect::HostEffectAllocationError,
    > {
        self.managed.reserve_live_exact(kind, bytes)
    }
    fn reserve_output_headroom(
        &self,
        bytes: u64,
    ) -> Result<
        Arc<dyn HostEffectOutputHeadroom>,
        crate::context::host_effect::HostEffectAllocationError,
    > {
        let managed = self.managed.reserve_output_headroom(bytes)?;
        Ok(Arc::new(TestOutputHeadroom {
            managed,
            live_total: self.live_total.clone(),
            live_output: self.live_output.clone(),
        }))
    }
    fn is_current(&self) -> bool {
        self.current.load(Ordering::SeqCst) && self.managed.is_current()
    }
}
struct TestCurrent(Arc<AtomicBool>);
impl HostEffectResourceLivenessLease for TestCurrent {
    fn is_current(&self) -> bool {
        self.0.load(Ordering::SeqCst)
    }
}

struct StagedTestResources {
    filesystem: Arc<dyn runmat_filesystem::FsProvider>,
    source: std::path::PathBuf,
    stage_calls: Cell<usize>,
    final_calls: Cell<usize>,
    direct_calls: Cell<usize>,
    live_output: Arc<AtomicUsize>,
    current: Arc<AtomicBool>,
    live_stage: Arc<AtomicUsize>,
    live_total: Arc<AtomicUsize>,
    ledger: ManagedHostEffectLedger,
}

struct ManagedBudget(ManagedHostEffectLedger);

impl crate::context::RuntimeParallelService for ManagedBudget {
    fn supports(&self, _: crate::context::ParallelCapability) -> bool {
        false
    }

    fn placement_resources(&self) -> crate::context::RuntimeParallelResources {
        let snapshot = self.0.snapshot();
        crate::context::RuntimeParallelResources {
            memory_available_bytes: Some(snapshot.free_bytes()),
            epoch: snapshot.epoch,
            ..Default::default()
        }
    }

    fn managed_host_effect_ledger(&self) -> Option<ManagedHostEffectLedger> {
        Some(self.0.clone())
    }
}

impl RuntimeHostEffectResourceService for StagedTestResources {
    fn admission(
        &self,
        _: &HostEffectRequest<'_>,
        _: &RuntimeContext,
    ) -> Option<HostEffectResourceAdmission> {
        self.direct_calls.set(self.direct_calls.get() + 1);
        None
    }

    fn stage_admission(
        &self,
        request: &HostEffectRequest<'_>,
        _: &RuntimeContext,
    ) -> Option<HostEffectStageAdmission> {
        self.stage_calls.set(self.stage_calls.get() + 1);
        if !Arc::ptr_eq(request.filesystem, &self.filesystem)
            || !matches!(request.workload, HostEffectWorkload::AudioDecode { source, .. } if source == self.source.as_path())
        {
            return None;
        }
        let managed = self.ledger.reserve_stage(128 * 1024).ok()?;
        self.live_total.fetch_add(128 * 1024, Ordering::SeqCst);
        self.live_stage.fetch_add(128 * 1024, Ordering::SeqCst);
        Some(HostEffectStageAdmission {
            evidence: HostEffectStageEvidence {
                authority_revision: Digest::sha256(b"test-staged-native-wave"),
                scratch_bytes: ResourceByteBound::Known(128 * 1024),
            },
            reservation: Arc::new(TestStageReservation {
                managed,
                ledger: self.ledger.clone(),
                current: self.current.clone(),
                live_stage: self.live_stage.clone(),
                live_output: self.live_output.clone(),
                live_total: self.live_total.clone(),
            }),
        })
    }

    fn final_admission_after_stage(
        &self,
        request: &HostEffectRequest<'_>,
        facts: crate::context::host_effect::HostEffectFinalFacts<'_>,
        _: &RuntimeContext,
    ) -> Option<HostEffectResourceAdmission> {
        self.final_calls.set(self.final_calls.get() + 1);
        if !Arc::ptr_eq(request.filesystem, &self.filesystem)
            || !matches!(request.workload, HostEffectWorkload::AudioDecode { source, .. } if source == self.source.as_path())
        {
            return None;
        }
        let crate::context::host_effect::HostEffectFinalFacts::AudioDecode(geometry) = facts else {
            return None;
        };
        if geometry.output_payload_bytes != 16
            || geometry.sample_count != 2
            || geometry.channels != 1
            || geometry.opened_extent != 48
        {
            return None;
        }
        let effect_only = matches!(
            request.workload,
            HostEffectWorkload::AudioDecode {
                output: AudioDecodeOutput::EffectOnly,
                ..
            }
        );
        let exact_output_bytes = runmat_value::ExactHostNumericAllocation::requested_bytes(
            geometry.output_dtype,
            geometry.sample_count,
        )
        .expect("fixture output owner layout") as u64;
        Some(HostEffectResourceAdmission {
            evidence: HostEffectResourceEvidence {
                authority_revision: Digest::sha256(b"test-staged-native-wave"),
                scratch_bytes: ResourceByteBound::Known(128 * 1024),
                returned_value_bytes: ResourceByteBound::Known(if effect_only {
                    0
                } else {
                    exact_output_bytes
                }),
            },
            liveness_lease: Arc::new(TestCurrent(self.current.clone())),
            output_allocation_permit: None,
        })
    }
}

#[test]
fn native_source_scans_then_selects_once_and_retains_exact_output_reservation() {
    let _lock = runmat_filesystem::provider_override_lock();
    let dir = tempfile::tempdir().expect("isolated native WAV root");
    let path = dir.path().join("tone.wav");
    std::fs::write(&path, pcm16_wav(8_000, 1, &[0, 16_384])).expect("write real WAVE bytes");
    let provider: Arc<dyn runmat_filesystem::FsProvider> =
        Arc::new(runmat_filesystem::NativeFsProvider);
    let _guard = runmat_filesystem::replace_provider(provider.clone());
    let ledger = ManagedHostEffectLedger::new(256 * 1024);
    let resources = Rc::new(StagedTestResources {
        filesystem: provider,
        source: path.clone(),
        stage_calls: Cell::new(0),
        final_calls: Cell::new(0),
        direct_calls: Cell::new(0),
        live_output: Arc::new(AtomicUsize::new(0)),
        current: Arc::new(AtomicBool::new(true)),
        live_stage: Arc::new(AtomicUsize::new(0)),
        live_total: Arc::new(AtomicUsize::new(0)),
        ledger: ledger.clone(),
    });
    let placement = Rc::new(Placement::default());
    *placement.managed_ledger.borrow_mut() = Some(ledger.clone());
    let runtime = RuntimeContext::new(Rc::new(crate::execution::RuntimeExecutionService::new()))
        .with_service_ports(
            RuntimeServicePorts::default()
                .with_parallel(Rc::new(ManagedBudget(ledger)))
                .with_host_effect_resources(resources.clone())
                .with_placement(placement.clone()),
        );
    let input = FsInputLease::capture(path.clone(), &AUDIOREAD_ERROR_IO, AUDIOREAD_BUILTIN_NAME)
        .expect("same captured source");
    let options = AudioreadOptions {
        range: None,
        datatype: AudioReadDataType::Double,
    };
    let exact_output_bytes = runmat_value::ExactHostNumericAllocation::requested_bytes(
        runmat_value::NumericDType::F64,
        2,
    )
    .expect("fixture output owner layout");
    let stage = admit_decode_stage(&runtime, &input, options, None)
        .expect("stage admission")
        .expect("offered stage");
    let mut cooperation = AudioCooperation::new(Some(&runtime), AUDIOREAD_BUILTIN_NAME);
    let prepared = block_on(prepare_wave_stream(&input, options, &mut cooperation))
        .expect("scan real opened WAVE")
        .expect("seekable WAVE");
    let wrong_controls = AudioreadOptions {
        datatype: AudioReadDataType::Native,
        ..options
    };
    let error = select_decode_after_stage(
        &runtime,
        &input,
        wrong_controls,
        None,
        &prepared.geometry,
        stage,
    )
    .err()
    .expect("changed public controls must reject");
    assert_eq!(error.identifier(), Some("RunMat:Placement:InvalidStage"));
    assert_eq!(placement.selections.get(), 0);
    assert_eq!(resources.live_stage.load(Ordering::SeqCst), 0);

    let stage = admit_decode_stage(&runtime, &input, options, None)
        .expect("second stage admission")
        .expect("offered stage");
    let other = FsInputLease::capture(
        path.with_file_name("other.wav"),
        &AUDIOREAD_ERROR_IO,
        AUDIOREAD_BUILTIN_NAME,
    )
    .expect("different source identity");
    let error =
        select_decode_after_stage(&runtime, &other, options, None, &prepared.geometry, stage)
            .err()
            .expect("changed source must reject");
    assert_eq!(error.identifier(), Some("RunMat:Placement:InvalidStage"));
    assert_eq!(placement.selections.get(), 0);
    assert_eq!(resources.live_stage.load(Ordering::SeqCst), 0);

    let value = block_on(audioread_builtin(
        Some(runtime.clone()),
        Value::from(path.to_string_lossy().as_ref()),
        crate::builtins::common::test_support::builtin_arguments(vec![]),
    ))
    .expect("staged native decode");
    assert_eq!(resources.stage_calls.get(), 3);
    assert_eq!(resources.final_calls.get(), 1);
    assert_eq!(resources.direct_calls.get(), 0);
    assert_eq!(placement.selections.get(), 1);
    assert_eq!(placement.observations.borrow().len(), 1);
    assert_eq!(
        resources.live_output.load(Ordering::SeqCst),
        exact_output_bytes
    );
    assert_eq!(resources.live_stage.load(Ordering::SeqCst), 0);
    assert_eq!(
        resources.live_total.load(Ordering::SeqCst),
        exact_output_bytes
    );
    let Value::Tensor(samples) = &value else {
        panic!("single-output decode must return tensor");
    };
    assert_eq!(samples.materialize_f64(), vec![0.0, 0.5]);
    drop(value);
    assert_eq!(resources.live_output.load(Ordering::SeqCst), 0);
    assert_eq!(resources.live_stage.load(Ordering::SeqCst), 0);
    assert_eq!(resources.live_total.load(Ordering::SeqCst), 0);

    let discarded = {
        let _outputs = crate::output_count::push_output_count(Some(0)).expect("output-count scope");
        block_on(audioread_builtin(
            Some(runtime.clone()),
            Value::from(path.to_string_lossy().as_ref()),
            crate::builtins::common::test_support::builtin_arguments(vec![]),
        ))
        .expect("selected effect-only decode without output quota")
    };
    assert!(output_list(discarded).is_empty());
    assert_eq!(resources.live_output.load(Ordering::SeqCst), 0);
    assert_eq!(resources.live_total.load(Ordering::SeqCst), 0);
    assert_eq!(resources.stage_calls.get(), 4);
    assert_eq!(resources.final_calls.get(), 2);
    assert_eq!(placement.selections.get(), 2);
    assert_eq!(placement.observations.borrow().len(), 2);
}

#[test]
fn scan_and_exact_output_compete_in_one_managed_allocation_ledger() {
    let live_stage = Arc::new(AtomicUsize::new(128 * 1024));
    let live_output = Arc::new(AtomicUsize::new(0));
    let live_total = Arc::new(AtomicUsize::new(128 * 1024));
    let first_ledger = ManagedHostEffectLedger::new(128 * 1024);
    let reservation = TestStageReservation {
        managed: first_ledger.reserve_stage(128 * 1024).expect("full stage"),
        ledger: first_ledger,
        current: Arc::new(AtomicBool::new(true)),
        live_stage: live_stage.clone(),
        live_output: live_output.clone(),
        live_total: live_total.clone(),
    };
    assert!(matches!(
        reservation.reserve_output_headroom(16).err(),
        Some(crate::context::host_effect::HostEffectAllocationError::ExceedsCapacity)
    ));
    assert_eq!(live_total.load(Ordering::SeqCst), 128 * 1024);
    drop(reservation);
    assert_eq!(live_total.load(Ordering::SeqCst), 0);

    let second_ledger = ManagedHostEffectLedger::new(128 * 1024 + 16);
    let reservation = TestStageReservation {
        managed: second_ledger
            .reserve_stage(128 * 1024)
            .expect("second stage"),
        ledger: second_ledger,
        current: Arc::new(AtomicBool::new(true)),
        live_stage,
        live_output: live_output.clone(),
        live_total: live_total.clone(),
    };
    live_total.store(128 * 1024, Ordering::SeqCst);
    reservation.live_stage.store(128 * 1024, Ordering::SeqCst);
    let headroom = reservation
        .reserve_output_headroom(16)
        .expect("combined peak fits");
    let output = headroom.reserve_exact(16).expect("exact output");
    assert_eq!(live_total.load(Ordering::SeqCst), 128 * 1024 + 16);
    drop(reservation);
    assert_eq!(live_total.load(Ordering::SeqCst), 16);
    drop(headroom);
    drop(output);
    assert_eq!(live_total.load(Ordering::SeqCst), 0);
    assert_eq!(live_output.load(Ordering::SeqCst), 0);
}

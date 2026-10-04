//! A real opened native WAVE source with synthetic, enforcing test quotas.
//! This is a protocol witness, not a production peak-memory certificate.

use super::*;
use crate::context::host_effect::{
    AudioDecodeOutput, HostEffectRequest, HostEffectResourceAdmission, HostEffectResourceEvidence,
    HostEffectResourceLivenessLease, HostEffectStageAdmission, HostEffectStageEvidence,
    HostEffectStageReservation, HostEffectWorkload,
};
use crate::context::{RuntimeContext, RuntimeHostEffectResourceService, RuntimeServicePorts};
use runmat_execution::{Digest, ResourceByteBound};
use runmat_value::{
    ExactHostAllocationPermit, ExactHostAllocationReservation, ExactHostNumericAllocationError,
};
use std::cell::Cell;
use std::rc::Rc;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

#[derive(Debug)]
struct TestOutputQuota {
    live: Arc<AtomicUsize>,
    limit: usize,
}

#[derive(Debug)]
struct TestOutputReservation {
    live: Arc<AtomicUsize>,
    bytes: usize,
}

impl Drop for TestOutputReservation {
    fn drop(&mut self) {
        self.live.fetch_sub(self.bytes, Ordering::SeqCst);
    }
}
impl ExactHostAllocationReservation for TestOutputReservation {}

impl ExactHostAllocationPermit for TestOutputQuota {
    fn reserve_exact(
        &self,
        bytes: usize,
    ) -> Result<Arc<dyn ExactHostAllocationReservation>, ExactHostNumericAllocationError> {
        self.live
            .fetch_update(Ordering::SeqCst, Ordering::SeqCst, |live| {
                live.checked_add(bytes).filter(|total| *total <= self.limit)
            })
            .map_err(|_| ExactHostNumericAllocationError::ExceedsPermit)?;
        Ok(Arc::new(TestOutputReservation {
            live: self.live.clone(),
            bytes,
        }))
    }
}

struct TestStageReservation {
    current: Arc<AtomicBool>,
    live: Arc<AtomicUsize>,
}
impl Drop for TestStageReservation {
    fn drop(&mut self) {
        self.live.fetch_sub(128 * 1024, Ordering::SeqCst);
    }
}
impl HostEffectStageReservation for TestStageReservation {
    fn reserved_bytes(&self) -> u64 {
        128 * 1024
    }
    fn is_current(&self) -> bool {
        self.current.load(Ordering::SeqCst)
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
        self.live_stage
            .fetch_update(Ordering::SeqCst, Ordering::SeqCst, |live| {
                live.checked_add(128 * 1024)
                    .filter(|total| *total <= 256 * 1024)
            })
            .ok()?;
        Some(HostEffectStageAdmission {
            evidence: HostEffectStageEvidence {
                authority_revision: Digest::sha256(b"test-staged-native-wave"),
                scratch_bytes: ResourceByteBound::Known(128 * 1024),
            },
            reservation: Arc::new(TestStageReservation {
                current: self.current.clone(),
                live: self.live_stage.clone(),
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
        Some(HostEffectResourceAdmission {
            evidence: HostEffectResourceEvidence {
                authority_revision: Digest::sha256(b"test-staged-native-wave"),
                scratch_bytes: ResourceByteBound::Known(128 * 1024),
                returned_value_bytes: ResourceByteBound::Known(if effect_only { 0 } else { 16 }),
            },
            liveness_lease: Arc::new(TestCurrent(self.current.clone())),
            output_allocation_permit: (!effect_only).then(|| {
                Arc::new(TestOutputQuota {
                    live: self.live_output.clone(),
                    limit: 16,
                }) as Arc<dyn ExactHostAllocationPermit>
            }),
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
        Arc::new(runmat_filesystem::NativeFsProvider::default());
    let _guard = runmat_filesystem::replace_provider(provider.clone());
    let resources = Rc::new(StagedTestResources {
        filesystem: provider,
        source: path.clone(),
        stage_calls: Cell::new(0),
        final_calls: Cell::new(0),
        direct_calls: Cell::new(0),
        live_output: Arc::new(AtomicUsize::new(0)),
        current: Arc::new(AtomicBool::new(true)),
        live_stage: Arc::new(AtomicUsize::new(0)),
    });
    let placement = Rc::new(Placement::default());
    let runtime = RuntimeContext::new(Rc::new(crate::execution::RuntimeExecutionService::new()))
        .with_service_ports(
            RuntimeServicePorts::default()
                .with_parallel(Rc::new(Budget(Some(256 * 1024))))
                .with_host_effect_resources(resources.clone())
                .with_placement(placement.clone()),
        );
    let input = FsInputLease::capture(path.clone(), &AUDIOREAD_ERROR_IO, AUDIOREAD_BUILTIN_NAME)
        .expect("same captured source");
    let options = AudioreadOptions {
        range: None,
        datatype: AudioReadDataType::Double,
    };
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
        vec![],
    ))
    .expect("staged native decode");
    assert_eq!(resources.stage_calls.get(), 3);
    assert_eq!(resources.final_calls.get(), 1);
    assert_eq!(resources.direct_calls.get(), 0);
    assert_eq!(placement.selections.get(), 1);
    assert_eq!(placement.observations.borrow().len(), 1);
    assert_eq!(resources.live_output.load(Ordering::SeqCst), 16);
    assert_eq!(resources.live_stage.load(Ordering::SeqCst), 0);
    let Value::Tensor(samples) = &value else {
        panic!("single-output decode must return tensor");
    };
    assert_eq!(samples.materialize_f64(), vec![0.0, 0.5]);
    drop(value);
    assert_eq!(resources.live_output.load(Ordering::SeqCst), 0);
    assert_eq!(resources.live_stage.load(Ordering::SeqCst), 0);

    let discarded = {
        let _outputs = crate::output_count::push_output_count(Some(0));
        block_on(audioread_builtin(
            Some(runtime.clone()),
            Value::from(path.to_string_lossy().as_ref()),
            vec![],
        ))
        .expect("selected effect-only decode without output quota")
    };
    assert!(output_list(discarded).is_empty());
    assert_eq!(resources.live_output.load(Ordering::SeqCst), 0);
    assert_eq!(resources.stage_calls.get(), 4);
    assert_eq!(resources.final_calls.get(), 2);
    assert_eq!(placement.selections.get(), 2);
    assert_eq!(placement.observations.borrow().len(), 2);
}

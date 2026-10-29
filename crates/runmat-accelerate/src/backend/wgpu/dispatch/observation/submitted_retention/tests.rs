use super::super::*;
use crate::backend::wgpu::provider::{WgpuProvider, WgpuProviderOptions};
use futures::FutureExt;
use runmat_accelerate_api::observation::{
    with_provider_observation, ProviderObservationBudget, ProviderOperationReceipt,
    WgpuObservationProgress, WgpuObservationProgressError, WgpuObservationProgressReservation,
};
use runmat_accelerate_api::{AccelContextHandle, AccelContextKind, AccelProvider};
use std::sync::{
    atomic::{AtomicBool, AtomicUsize, Ordering},
    Arc, Mutex,
};

type Callback = runmat_accelerate_api::observation::WgpuObservationCompletion;

struct Held {
    reservation: Box<dyn WgpuObservationProgressReservation>,
    submission: wgpu::SubmissionIndex,
    callback: Callback,
}

struct Gate {
    progress: Arc<dyn WgpuObservationProgress>,
    held: Arc<Mutex<Option<Held>>>,
}

struct Reservation {
    reservation: Box<dyn WgpuObservationProgressReservation>,
    held: Arc<Mutex<Option<Held>>>,
}

impl WgpuObservationProgress for Gate {
    fn reserve(
        &self,
    ) -> Result<Box<dyn WgpuObservationProgressReservation>, WgpuObservationProgressError> {
        Ok(Box::new(Reservation {
            reservation: self.progress.reserve()?,
            held: self.held.clone(),
        }))
    }
}

impl WgpuObservationProgressReservation for Reservation {
    fn submit(
        self: Box<Self>,
        submission: wgpu::SubmissionIndex,
        callback: Callback,
    ) -> Result<(), WgpuObservationProgressError> {
        let Self { reservation, held } = *self;
        *held.lock().unwrap() = Some(Held {
            reservation,
            submission,
            callback,
        });
        Ok(())
    }
}

struct CountDrop(Arc<AtomicUsize>);
impl Drop for CountDrop {
    fn drop(&mut self) {
        self.0.fetch_add(1, Ordering::SeqCst);
    }
}

#[test]
#[ignore = "requires actual WGPU device and native progress driver"]
fn semantic_map_completion_does_not_release_submitted_allocations() {
    let provider = WgpuProvider::new(WgpuProviderOptions::default()).expect("WGPU device");
    let AccelContextHandle::Wgpu(context) = provider
        .export_context(AccelContextKind::Plotting)
        .expect("WGPU context");
    let value = runmat_accelerate_api::ProviderRepresentation {
        element_type: runmat_accelerate_api::ProviderElementType::F32,
        storage: runmat_accelerate_api::ProviderStorage::DenseReal,
        layout: runmat_accelerate_api::ProviderLayout::ColumnMajorContiguous,
        shape: vec![1, 1],
        residency: runmat_accelerate_api::ProviderResidency::Device,
    };
    let base =
        provider.begin_operation_observation(&runmat_accelerate_api::ProviderFeasibilityQuery {
            operation: runmat_accelerate_api::ProviderOperationIdentity::new("fusion.elementwise"),
            family: runmat_accelerate_api::ProviderOperationFamily::Fusion,
            inputs: vec![value.clone()],
            outputs: vec![value],
            workload: Default::default(),
        });
    let progress = match base {
        runmat_accelerate_api::observation::ProviderObservationSupport::Observed(receipt) => {
            receipt.recorder().wgpu_progress().expect("native progress")
        }
        _ => panic!("provider observation required"),
    };
    let gate = Arc::new(Gate {
        progress,
        held: Arc::new(Mutex::new(None)),
    });
    let budget = Arc::new(ProviderObservationBudget::with_wgpu_query_bytes(
        std::num::NonZeroU32::new(1).unwrap(),
        MAX_QUERY_REQUESTED_BYTES,
    ));
    let receipt = ProviderOperationReceipt::with_wgpu_progress(budget.clone(), gate.clone());
    let mut pending = PendingSubmissions::new(context.access.clone()).with_profiling(
        None,
        budget,
        Some(gate.clone()),
    );
    let (prepared, retirement) = with_provider_observation(Some(receipt.recorder()), || {
        PreparedSubmittedAllocationRetention::reserve().unwrap()
    });
    let drops = Arc::new(AtomicUsize::new(0));
    pending
        .retain_allocations_until_next_submit(
            prepared,
            SubmittedAllocationRetention::new(CountDrop(drops.clone())),
        )
        .unwrap();
    let semantic_fence = context
        .access
        .with_shared(|emission| {
            let fence = runmat_accelerate_api::WgpuBufferOwner::new(
                emission.device().create_buffer(&wgpu::BufferDescriptor {
                    label: Some("submitted-retention-semantic-fence"),
                    size: 8,
                    usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                }),
            );
            with_provider_observation(Some(receipt.recorder()), || {
                let mut encoder = pending.profiled_encoder(emission, &Default::default());
                encoder.clear_buffer(fence.as_ref(), 0, None);
                pending.submit(emission, encoder)
            })?;
            Ok::<_, anyhow::Error>(fence)
        })
        .unwrap()
        .unwrap();
    pending.commit();

    let mapped = Arc::new(AtomicBool::new(false));
    let mapped_callback = mapped.clone();
    semantic_fence
        .slice(..)
        .map_async(wgpu::MapMode::Read, move |status| {
            assert!(status.is_ok());
            mapped_callback.store(true, Ordering::SeqCst);
        });
    let _ = context.access.poll(wgpu::Maintain::Wait);
    assert!(mapped.load(Ordering::SeqCst));
    assert_eq!(drops.load(Ordering::SeqCst), 0);
    let mut retirement = retirement;
    assert!((&mut retirement).now_or_never().is_none());
    context
        .access
        .with_cleanup(|cleanup| cleanup.unmap_buffer(semantic_fence.as_ref()))
        .unwrap();

    let held = gate.held.lock().unwrap().take().expect("held completion");
    held.reservation
        .submit(held.submission, held.callback)
        .unwrap();
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
    while drops.load(Ordering::SeqCst) == 0 {
        assert!(std::time::Instant::now() < deadline);
        std::thread::yield_now();
    }
    assert_eq!(drops.load(Ordering::SeqCst), 1);
    assert!(futures::executor::block_on(retirement).is_ok());
}

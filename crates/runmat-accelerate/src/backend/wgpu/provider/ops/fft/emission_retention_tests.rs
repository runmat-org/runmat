//! Physical-owner witness using the real adapter and its real progress driver.
//! The gate delays callback delivery, never manufactures successful retirement.
use super::*;
use crate::backend::wgpu::dispatch::observation::PendingSubmissions;
use runmat_accelerate_api::observation::{
    with_provider_observation, ProviderOperationReceipt, WgpuObservationCompletion,
    WgpuObservationProgress, WgpuObservationProgressError, WgpuObservationProgressReservation,
};
use runmat_accelerate_api::AccelProvider;
use std::sync::{Arc, Mutex};

type Callback = WgpuObservationCompletion;
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
        let previous = held.lock().unwrap().replace(Held {
            reservation,
            submission,
            callback,
        });
        assert!(previous.is_none(), "one exact completion per FFT chunk");
        Ok(())
    }
}

#[tokio::test]
#[ignore = "requires actual WGPU adapter and native progress driver"]
async fn fft_final_handle_drop_retains_allocation_until_real_direct_or_observed_completion() {
    let provider = provider().await;
    let (layout, pipeline) = scope_test_pipeline(&provider);
    for observed in [false, true] {
        let baseline = provider.allocations.retained_bytes();
        let value = provider
            .upload(&HostTensorView {
                data: &[3.0],
                shape: &[1, 1],
            })
            .unwrap();
        let entry = provider.get_entry_raw(&value).unwrap();
        let weak = value.allocation_lease().downgrade();
        let charged = provider.allocations.retained_bytes();
        assert!(charged > baseline);
        let bounded = FftDispatchResources::new(
            &[&value, &value, &value],
            &[&entry.buffer, &entry.buffer, &entry.buffer, &entry.buffer],
        )
        .unwrap();
        assert!(FftDispatchResources::new(&[&value, &value, &value, &value], &[]).is_err());
        assert!(FftDispatchResources::new(
            &[],
            &[
                &entry.buffer,
                &entry.buffer,
                &entry.buffer,
                &entry.buffer,
                &entry.buffer
            ],
        )
        .is_err());
        assert_eq!(
            provider.allocations.retained_bytes(),
            charged,
            "retaining existing identities must not reserve another physical allocation"
        );
        drop(bounded);
        let resources = FftDispatchResources::new(&[&value], &[&entry.buffer]).unwrap();
        let gate = Arc::new(Gate {
            progress: provider
                .observation_progress
                .clone()
                .expect("native progress owner"),
            held: Arc::new(Mutex::new(None)),
        });
        let receipt = ProviderOperationReceipt::with_wgpu_progress(
            provider.observation_budget.clone(),
            gate.clone(),
        );
        let recorder = observed.then(|| receipt.recorder());
        let mut pending = PendingSubmissions::new(provider.access.clone()).with_profiling(
            None,
            provider.observation_budget.clone(),
            Some(gate.clone()),
        );
        with_provider_observation(recorder, || {
            provider
                .access
                .with_shared(|emission| -> anyhow::Result<()> {
                    pending.retain_submission_resources(resources.into_retention())?;
                    let bind = emission
                        .device()
                        .create_bind_group(&wgpu::BindGroupDescriptor {
                            label: Some("fft-retained-lease-witness"),
                            layout: &layout,
                            entries: &[],
                        });
                    crate::backend::wgpu::dispatch::fft::run(
                        emission,
                        &mut pending,
                        &pipeline,
                        &bind,
                        1,
                        "fft-retained-lease-witness",
                    )
                })
        })
        .unwrap()
        .unwrap();
        pending.commit();
        drop(entry);
        drop(value); // No result carrier or test-owned strong lease remains.
        assert!(
            weak.acquire().is_some(),
            "completion owns the final allocation lease"
        );
        assert_eq!(
            provider.allocations.retained_bytes(),
            charged,
            "publishing/dropping the final tensor must not refund its allocation"
        );
        let held = gate
            .held
            .lock()
            .unwrap()
            .take()
            .expect("real exact completion held");
        held.reservation
            .submit(held.submission, held.callback)
            .unwrap();
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
        // A weak upgrade can fail once the final strong count reaches zero,
        // before that owner's destructor finishes refunding its allocation.
        // Observe both retirement effects, not the intermediate Arc state.
        while weak.acquire().is_some() || provider.allocations.retained_bytes() != baseline {
            assert!(
                std::time::Instant::now() < deadline,
                "real progress must retire the lease"
            );
            std::thread::yield_now();
        }
        assert_eq!(provider.allocations.retained_bytes(), baseline);
    }

    // The diagnostic route already owns a fixed physical slot. Hold publication
    // of that REAL slot, rather than substituting an indexed wait or fake proof.
    use crate::backend::wgpu::dispatch::observation::{ScopeConfirmation, ScopedRetirement};
    let baseline = provider.allocations.retained_bytes();
    let value = provider
        .upload(&HostTensorView {
            data: &[7.0],
            shape: &[1, 1],
        })
        .unwrap();
    let entry = provider.get_entry_raw(&value).unwrap();
    let weak = value.allocation_lease().downgrade();
    let charged = provider.allocations.retained_bytes();
    let resources = FftDispatchResources::new(&[&value], &[&entry.buffer]).unwrap();
    let signal = ScopedRetirement::prepare().unwrap();
    let confirmation = ScopeConfirmation(signal.clone());
    let progress = provider.observation_progress.as_ref().unwrap();
    let mut pending = PendingSubmissions::new(provider.access.clone())
        .with_profiling(
            None,
            provider.observation_budget.clone(),
            Some(progress.clone()),
        )
        .with_scoped_retirement();
    pending
        .retain_submission_resources(resources.into_retention())
        .unwrap();
    let mut gate = progress.scoped_emission_guard(pending, signal).unwrap();
    let receipt = provider
        .access
        .with_profiled_scoped(&provider.emission_authority, |emission| {
            gate.with_pending(|pending| -> anyhow::Result<()> {
                let bind = emission
                    .device()
                    .create_bind_group(&wgpu::BindGroupDescriptor {
                        label: Some("fft-scoped-retained-lease-witness"),
                        layout: &layout,
                        entries: &[],
                    });
                crate::backend::wgpu::dispatch::fft::run(
                    emission,
                    pending,
                    &pipeline,
                    &bind,
                    1,
                    "fft-scoped-retained-lease-witness",
                )
            })
        })
        .unwrap();
    drop(entry);
    drop(value);
    assert!(
        weak.acquire().is_some(),
        "prearmed physical slot owns the final lease"
    );
    assert_eq!(provider.allocations.retained_bytes(), charged);
    drop(gate); // Publish the existing slot to the real queue-retirement worker.
    receipt.resolve().await.unwrap().unwrap();
    confirmation.0.confirm();
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
    while weak.acquire().is_some() || provider.allocations.retained_bytes() != baseline {
        assert!(
            std::time::Instant::now() < deadline,
            "scoped proof must retire the lease"
        );
        std::thread::yield_now();
    }
    assert_eq!(provider.allocations.retained_bytes(), baseline);
}

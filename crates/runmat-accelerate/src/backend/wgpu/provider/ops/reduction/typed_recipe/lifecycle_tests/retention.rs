//! Delay delivery of REAL bounded progress reservations, never their proof.
use super::*;
use runmat_accelerate_api::observation::{
    with_provider_observation, ProviderOperationReceipt, WgpuObservationCompletion,
    WgpuObservationProgressError, WgpuObservationProgressReservation,
};
use std::sync::{Arc, Mutex};

struct Held {
    reservation: Box<dyn WgpuObservationProgressReservation>,
    submission: wgpu::SubmissionIndex,
    completion: WgpuObservationCompletion,
}
struct Gate {
    real: Arc<dyn WgpuObservationProgress>,
    held: Arc<Mutex<[Option<Held>; 2]>>,
}
struct Reservation {
    real: Box<dyn WgpuObservationProgressReservation>,
    held: Arc<Mutex<[Option<Held>; 2]>>,
}
impl WgpuObservationProgress for Gate {
    fn reserve(
        &self,
    ) -> Result<Box<dyn WgpuObservationProgressReservation>, WgpuObservationProgressError> {
        Ok(Box::new(Reservation {
            real: self.real.reserve()?,
            held: self.held.clone(),
        }))
    }
}
impl WgpuObservationProgressReservation for Reservation {
    fn submit(
        self: Box<Self>,
        submission: wgpu::SubmissionIndex,
        completion: WgpuObservationCompletion,
    ) -> Result<(), WgpuObservationProgressError> {
        let Self { real, held } = *self;
        let mut slots = held.lock().unwrap();
        let slot = slots
            .iter_mut()
            .find(|slot| slot.is_none())
            .expect("at most two real typed stages");
        *slot = Some(Held {
            reservation: real,
            submission,
            completion,
        });
        Ok(())
    }
}
impl Gate {
    fn release(&self, expected: usize) {
        let slots = {
            let mut held = self.held.lock().unwrap();
            std::mem::replace(&mut *held, std::array::from_fn(|_| None))
        };
        assert_eq!(slots.iter().filter(|slot| slot.is_some()).count(), expected);
        for held in slots.into_iter().flatten() {
            held.reservation
                .submit(held.submission, held.completion)
                .unwrap();
        }
    }
}

pub(super) fn prove_exact_owners(provider: &'static WgpuProvider) {
    for observed in [false, true] {
        for (strategy, fail_after_first) in [
            (ReductionStrategy::SinglePass, false),
            (ReductionStrategy::TwoPass, false),
            (ReductionStrategy::TwoPass, true),
        ] {
            let baseline = provider.allocations.retained_bytes();
            let upload = |data: &[f32]| {
                provider
                    .upload_numeric_raw(&HostNumericTensorView {
                        data: HostNumericDataView::F32(data),
                        shape: &[2, 3],
                        storage: GpuTensorStorage::Real,
                    })
                    .unwrap()
            };
            let a = upload(&[1., 2., 3., 4., 5., 6.]);
            let b = upload(&[10., 20., 30., 40., 50., 60.]);
            let weak_a = a.allocation_lease().downgrade();
            let weak_b = b.allocation_lease().downgrade();
            let recipe = checked_recipe(&a, &b);
            let inputs = [a, b];
            let gate = Arc::new(Gate {
                real: provider
                    .observation_progress
                    .clone()
                    .expect("actual native progress"),
                held: Arc::new(Mutex::new(std::array::from_fn(|_| None))),
            });
            let receipt = ProviderOperationReceipt::with_wgpu_progress(
                provider.observation_budget.clone(),
                gate.clone(),
            );
            let mut trace = Trace {
                fail_after_first,
                progress: Some(gate.clone()),
                ..Default::default()
            };
            let result = with_provider_observation(observed.then(|| receipt.recorder()), || {
                let selected = ReductionSelection::new(recipe.physical(), strategy).unwrap();
                let submission =
                    ReductionSubmission::new(selected, &inputs, provider.device_id()).unwrap();
                provider.submit_reduction_recipe_observed(&submission, &mut trace)
            });
            let weak_output = if fail_after_first {
                let error = result.expect_err("original injected after-pass1 failure");
                assert!(error.is_after_effect() && !error.is_unsupported());
                assert_eq!(trace.stages.len(), 1);
                None
            } else {
                let output = result.expect("actual selected reduction");
                let weak = output.allocation_lease().downgrade();
                drop(output);
                Some(weak)
            };
            drop(inputs);
            // All caller leases are gone. Only the exact canonical physical
            // completion owner can keep their charges and identities alive.
            assert!(weak_a.acquire().is_some());
            assert!(weak_b.acquire().is_some());
            if let Some(output) = &weak_output {
                assert!(
                    output.acquire().is_some(),
                    "final allocation lease must survive publication/drop"
                );
            }
            assert!(provider.allocations.retained_bytes() > baseline);
            if strategy == ReductionStrategy::TwoPass {
                assert!(trace.partial.as_ref().unwrap().upgrade().is_some());
            }
            let stages = if strategy == ReductionStrategy::TwoPass && !fail_after_first {
                2
            } else {
                1
            };
            gate.release(stages);
            trace.wait_retirements();
            assert!(weak_a.acquire().is_none());
            assert!(weak_b.acquire().is_none());
            if let Some(output) = &weak_output {
                assert!(output.acquire().is_none());
            }
            if strategy == ReductionStrategy::TwoPass {
                assert!(trace.partial.as_ref().unwrap().upgrade().is_none());
            }
            assert_eq!(provider.allocations.retained_bytes(), baseline);
        }
    }
}

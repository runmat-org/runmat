use super::*;
use futures::task::noop_waker_ref;
use std::future::Future;
use std::task::{Context, Poll};

#[test]
fn dropping_pending_resample_provider_call_retires_owned_work_but_not_borrowed_inputs() {
    let _guard = test_support::accel_test_lock();
    for resident_filter in [false, true] {
        let provider = Box::leak(Box::new(HostileDownsampleProvider::new()));
        unsafe {
            runmat_accelerate_api::register_provider(provider);
        }
        provider.pending_filter.store(true, Ordering::Relaxed);
        let input = provider
            .upload(&HostTensorView {
                data: &[1.0, 2.0, 3.0],
                shape: &[1, 3],
            })
            .expect("borrowed source");
        let resident = resident_filter.then(|| {
            provider
                .upload(&HostTensorView {
                    data: &[0.0, 1.0, 0.0],
                    shape: &[3, 1],
                })
                .expect("borrowed resident filter")
        });
        let options = ResampleOptions {
            filter: match &resident {
                Some(handle) => ResampleFilter::Gpu {
                    handle: handle.clone(),
                    len: 3,
                },
                None => ResampleFilter::Host(vec![0.0, 1.0, 0.0]),
            },
            dim: None,
        };
        let borrowed_ids = [
            Some(input.buffer_id()),
            resident.as_ref().map(GpuTensorHandle::buffer_id),
        ];
        let mut call = Box::pin(resample_gpu(&input, 2, 1, &options));
        let mut context = Context::from_waker(noop_waker_ref());
        assert!(matches!(call.as_mut().poll(&mut context), Poll::Pending));
        assert_ne!(provider.filter_output.load(Ordering::Relaxed), 0);
        assert_eq!(provider.gathers.load(Ordering::Relaxed), 0);
        assert_eq!(provider.downloads.load(Ordering::Relaxed), 0);
        assert_eq!(
            provider.numeric_uploads.load(Ordering::Relaxed),
            usize::from(!resident_filter),
            "only a host filter creates an owned coefficient upload"
        );

        // The hostile fixture's scatter recorder owns extra clones. Remove
        // those before testing whether the suspended call owns its own padded
        // and coefficient allocations until the future is dropped.
        assert!(provider.staged_scatter.lock().unwrap().take().is_some());
        let produced = provider
            .probes
            .lock()
            .unwrap()
            .iter()
            .filter(|(id, _)| !borrowed_ids.contains(&Some(**id)))
            .map(|(id, probe)| (*id, probe.clone()))
            .collect::<Vec<_>>();
        assert_eq!(produced.len(), if resident_filter { 2 } else { 3 });
        assert!(produced.iter().all(|(_, probe)| probe.retirements() == 0));
        assert!(produced
            .iter()
            .any(|(id, _)| { *id == provider.filter_output.load(Ordering::Relaxed) }));
        assert!(input.allocation_lease().validate_live().is_ok());
        assert!(resident
            .as_ref()
            .is_none_or(|filter| { filter.allocation_lease().validate_live().is_ok() }));

        drop(call);
        assert!(produced.iter().all(|(_, probe)| probe.retirements() == 1));
        assert_eq!(provider.frees(), produced.len());
        assert_eq!(provider.gathers.load(Ordering::Relaxed), 0);
        assert!(input.allocation_lease().validate_live().is_ok());
        assert!(resident
            .as_ref()
            .is_none_or(|filter| { filter.allocation_lease().validate_live().is_ok() }));
        drop(options);
        if let Some(filter) = resident {
            filter
                .release()
                .expect("borrowed filter remains caller-owned");
        }
        input
            .release()
            .expect("borrowed source remains caller-owned");
    }
}

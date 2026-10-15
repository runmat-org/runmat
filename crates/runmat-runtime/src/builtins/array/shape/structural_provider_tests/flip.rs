//! Same producer fixture: flip absence versus terminal failure, and staged rotation.
use super::*;

#[test]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
fn flip_and_completed_rotation_do_not_replay_errors_or_completed_permutations() {
    for rotation in [false, true] {
        let modes = if rotation {
            vec![Mode::CompletedFlipNone, Mode::CompletedFlipFailure]
        } else {
            vec![
                Mode::NoAttempt,
                Mode::Failed,
                Mode::WrongClass,
                Mode::BorrowedAlias,
            ]
        };
        for mode in modes {
            let provider = Box::leak(Box::new(Provider {
                inner: InProcessProvider::new(),
                view_owner: runmat_accelerate_api::test_support::TestAllocationOwner::new(),
                mode,
                reads: AtomicUsize::new(0),
                permutations: AtomicUsize::new(0),
                unpublished: Mutex::new(Vec::new()),
            }));
            let _selected = provider.select();
            let input = provider
                .inner
                .upload(&HostTensorView {
                    data: &[1., 2., 3., 4., 5., 6.],
                    shape: &[2, 3],
                })
                .unwrap()
                .with_provenance(GpuHandleProvenance::Explicit)
                .unwrap();
            let result = if rotation {
                block_on(crate::call_builtin_async(
                    "rot90",
                    &[Value::GpuTensor(input.clone())],
                ))
            } else {
                block_on(flip::flip_gpu_with("flip", input.clone(), &[1])).and_then(|value| {
                    crate::execution::RootedValue::new(value)
                        .map_err(|cause| cause.into_runtime_error())
                })
            };
            if matches!(mode, Mode::NoAttempt | Mode::CompletedFlipNone) {
                let result = result.unwrap();
                let Value::GpuTensor(output) = result.value() else {
                    panic!("resident result");
                };
                assert!(runmat_accelerate_api::handle_is_explicit(output));
                let host = block_on(provider.inner.download(output)).unwrap();
                assert_eq!(
                    host.data,
                    if rotation {
                        vec![5., 3., 1., 6., 4., 2.]
                    } else {
                        vec![2., 1., 4., 3., 6., 5.]
                    }
                );
                assert_eq!(
                    output.shape(),
                    if rotation { &[3, 2][..] } else { &[2, 3][..] }
                );
                assert!(provider.reads.load(Ordering::SeqCst) > 0);
            } else {
                let error = result.unwrap_err();
                assert_eq!(error.gpu_gather_retry, crate::GpuGatherRetry::Never);
                assert_eq!(provider.reads.load(Ordering::SeqCst), 0);
            }
            assert_eq!(
                provider.permutations.load(Ordering::SeqCst),
                usize::from(rotation)
            );
            assert!(provider
                .unpublished
                .lock()
                .unwrap()
                .iter()
                .all(|lease| lease.acquire().is_none()));
            input.allocation_lease().validate_live().unwrap();

            // A no-op axis may lawfully return the exact same owned view.
            if mode == Mode::BorrowedAlias {
                let Value::GpuTensor(output) =
                    block_on(flip::flip_gpu_with("flip", input.clone(), &[usize::MAX])).unwrap()
                else {
                    panic!("identity result");
                };
                assert!(output.shares_allocation_with(&input));
                assert_eq!(output.view(), input.view());
            }
        }
    }
}

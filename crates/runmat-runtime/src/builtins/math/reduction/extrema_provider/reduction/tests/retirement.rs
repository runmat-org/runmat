//! Extend the existing dimensional extrema fixture: no new provider/mock owner.
use super::*;

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[cfg_attr(not(target_arch = "wasm32"), test)]
fn discarded_index_retirement_failure_is_terminal_and_keeps_original_cause() {
    for kind in [ExtremaKind::Min, ExtremaKind::Max] {
        for (element, demand, values_alias_input) in [
            (NumericElementType::F64, ExtremaDemand::ValuesOnly, false),
            (NumericElementType::F64, ExtremaDemand::ValuesOnly, true),
            (
                NumericElementType::F32,
                ExtremaDemand::ValuesAndIndices,
                false,
            ),
        ] {
            let provider = provider(false);
            let _selected = ThreadProviderGuard::set(Some(provider));
            let input = provider.allocate(vec![1, 3], element, 1.0);
            let values = provider.allocate(vec![1, 3], element, 1.0);
            let indices = provider.allocate(vec![1, 3], element, 1.0);
            indices
                .probe
                .fail_retirement("original discarded index retirement");
            let retained_values = if values_alias_input {
                input.handle.with_view(input.handle.view().clone()).unwrap()
            } else {
                values
                    .probe
                    .fail_retirement("secondary unpublished values retirement");
                values.handle
            };
            let error = futures::executor::block_on(kind.adopt_reduction_result(
                provider,
                &input.handle,
                0,
                ReduceDimResult {
                    values: retained_values,
                    indices: indices.handle,
                },
                demand,
            ))
            .expect_err("discard must not publish success");
            assert_eq!(error.gpu_gather_retry, crate::GpuGatherRetry::Never);
            assert!(error
                .message()
                .contains("discarded dimensional extrema indices could not retire"));
            let mut cause = std::error::Error::source(&error);
            let mut original = false;
            while let Some(source) = cause {
                original |= source
                    .downcast_ref::<runmat_accelerate_api::allocation::GpuHandleReleaseError>()
                    .is_some_and(|source| {
                        source
                            .to_string()
                            .contains("original discarded index retirement")
                    });
                cause = source.source();
            }
            assert!(
                original,
                "original retirement error survives secondary cleanup"
            );
            assert_eq!(indices.probe.retirements(), 1);
            assert_eq!(values.probe.retirements(), usize::from(!values_alias_input));
            assert_eq!(input.probe.retirements(), 0);
            input.handle.allocation_lease().validate_live().unwrap();
            assert_eq!(
                provider.downloads.load(Ordering::SeqCst),
                usize::from(element == NumericElementType::F32)
            );
            drop(input.handle);
            assert_eq!(input.probe.retirements(), 1);
        }
    }
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[cfg_attr(not(target_arch = "wasm32"), test)]
fn singleton_input_alias_is_not_invalidated_by_discarded_sibling_release() {
    for indices_alias_input in [false, true] {
        let provider = provider(false);
        let _selected = ThreadProviderGuard::set(Some(provider));
        let input = provider.allocate(vec![1, 3], NumericElementType::F64, 1.0);
        let other = provider.allocate(vec![1, 3], NumericElementType::F64, 1.0);
        let alias = input.handle.with_view(input.handle.view().clone()).unwrap();
        assert_ne!(alias.view_identity(), input.handle.view_identity());
        let result = if indices_alias_input {
            ReduceDimResult {
                values: other.handle,
                indices: alias,
            }
        } else {
            ReduceDimResult {
                values: alias,
                indices: other.handle,
            }
        };
        let output = futures::executor::block_on(ExtremaKind::Min.adopt_reduction_result(
            provider,
            &input.handle,
            0,
            result,
            ExtremaDemand::ValuesOnly,
        ))
        .unwrap();
        input.handle.allocation_lease().validate_live().unwrap();
        assert_eq!(input.probe.retirements(), 0);
        assert_eq!(other.probe.retirements(), usize::from(!indices_alias_input));
        assert_eq!(provider.downloads.load(Ordering::SeqCst), 0);
        drop(output);
        assert_eq!(other.probe.retirements(), 1);
        assert_eq!(input.probe.retirements(), 0);
        drop(input.handle);
        assert_eq!(input.probe.retirements(), 1);
    }
}

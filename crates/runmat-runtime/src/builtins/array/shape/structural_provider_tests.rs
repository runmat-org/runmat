//! Applicability: actual permute/ipermute producer and repmat producer boundaries.
use super::*;
use crate::builtins::common::gpu_helpers;
use futures::executor::block_on;
use runmat_accelerate::simple_provider::InProcessProvider;
use runmat_accelerate_api::{
    allocation::WeakAllocationLease, AccelDownloadFuture, AccelNumericDownloadFuture,
    AccelProvider, GpuHandleProvenance, GpuTensorHandle, GpuTensorLayout, GpuTensorStorage,
    HostNumericDataView, HostNumericTensorView, HostTensorView, ProviderExecutionError,
    ProviderExecutionResult, ThreadProviderGuard, TransposeInfo,
};
use runmat_value::{Tensor, Value};
use std::sync::{
    atomic::{AtomicUsize, Ordering},
    Mutex,
};

#[derive(Clone, Copy, PartialEq)]
enum Mode {
    NoAttempt,
    Failed,
    BorrowedAlias,
    WrongClass,
    LazyAlias,
    CompletedFlipNone,
    CompletedFlipFailure,
}
struct Provider {
    inner: InProcessProvider,
    view_owner: runmat_accelerate_api::test_support::TestAllocationOwner,
    mode: Mode,
    reads: AtomicUsize,
    permutations: AtomicUsize,
    unpublished: Mutex<Vec<WeakAllocationLease>>,
}
impl Provider {
    fn select(&'static self) -> ThreadProviderGuard {
        // SAFETY: fixtures leak this facade and allocate a unique namespace;
        // owner addressability must not depend on the ambient default provider.
        unsafe { runmat_accelerate_api::register_device_provider(self) };
        ThreadProviderGuard::set(Some(self))
    }

    fn result(
        &self,
        input: &GpuTensorHandle,
        shape: Vec<usize>,
    ) -> ProviderExecutionResult<Option<GpuTensorHandle>> {
        match self.mode {
            Mode::NoAttempt | Mode::CompletedFlipNone => Ok(None),
            Mode::Failed | Mode::CompletedFlipFailure => Err(ProviderExecutionError::from(
                runmat_accelerate_api::unsupported_provider_operation(
                    "structural hook failed after unreported staging",
                ),
            )
            .with_unproven_effects()),
            Mode::BorrowedAlias => Ok(Some(input.clone())),
            Mode::WrongClass => {
                let values = vec![0.0f32; shape.iter().product()];
                let output = self.inner.upload_numeric_raw(&HostNumericTensorView {
                    data: HostNumericDataView::F32(&values),
                    shape: &shape,
                    storage: GpuTensorStorage::Real,
                })?;
                self.unpublished
                    .lock()
                    .unwrap()
                    .push(output.allocation_lease().downgrade());
                Ok(Some(output))
            }
            Mode::LazyAlias => Ok(Some(
                input.with_view(input.view().clone().with_shape(shape).with_layout(
                    GpuTensorLayout::Transpose(TransposeInfo {
                        base_rows: input.shape()[0],
                        base_cols: input.shape()[1],
                    }),
                ))?,
            )),
        }
    }
}
impl AccelProvider for Provider {
    fn device_id(&self) -> u32 {
        if self.mode == Mode::LazyAlias {
            self.view_owner.device_id()
        } else {
            self.inner.device_id()
        }
    }
    fn device_info(&self) -> String {
        "structural boundary fixture".into()
    }
    fn upload(&self, host: &HostTensorView) -> anyhow::Result<GpuTensorHandle> {
        self.inner.upload(host)
    }
    fn upload_numeric_raw(
        &self,
        host: &HostNumericTensorView<'_>,
    ) -> ProviderExecutionResult<GpuTensorHandle> {
        self.inner.upload_numeric_raw(host)
    }
    fn upload_logical_raw(
        &self,
        host: &runmat_accelerate_api::HostLogicalTensorView<'_>,
    ) -> ProviderExecutionResult<GpuTensorHandle> {
        self.inner.upload_logical_raw(host)
    }
    fn download<'a>(&'a self, input: &'a GpuTensorHandle) -> AccelDownloadFuture<'a> {
        self.reads.fetch_add(1, Ordering::SeqCst);
        self.inner.download(input)
    }
    fn download_numeric<'a>(
        &'a self,
        input: &'a GpuTensorHandle,
    ) -> AccelNumericDownloadFuture<'a> {
        self.reads.fetch_add(1, Ordering::SeqCst);
        self.inner.download_numeric(input)
    }
    fn permute(
        &self,
        input: &GpuTensorHandle,
        order: &[usize],
    ) -> ProviderExecutionResult<Option<GpuTensorHandle>> {
        self.permutations.fetch_add(1, Ordering::SeqCst);
        if matches!(
            self.mode,
            Mode::CompletedFlipNone | Mode::CompletedFlipFailure
        ) {
            return self.inner.permute(input, order);
        }
        self.result(input, vec![3, 2])
    }
    fn flip(
        &self,
        input: &GpuTensorHandle,
        _: &[usize],
    ) -> ProviderExecutionResult<Option<GpuTensorHandle>> {
        self.result(input, input.shape().to_vec())
    }
    fn repmat(
        &self,
        input: &GpuTensorHandle,
        _: &[usize],
    ) -> ProviderExecutionResult<Option<GpuTensorHandle>> {
        self.result(input, vec![4, 3])
    }
}
fn call_transform(permutation: bool, source: &GpuTensorHandle) -> crate::BuiltinResult<Value> {
    let control = Value::Tensor(Tensor::new(vec![2.0, 1.0], vec![1, 2]).unwrap());
    if permutation {
        block_on(permute::permute_builtin(
            Value::GpuTensor(source.clone()),
            control,
        ))
    } else {
        block_on(repmat::repmat_builtin(
            Value::GpuTensor(source.clone()),
            crate::builtins::common::test_support::builtin_arguments(vec![control]),
        ))
    }
}

#[test]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
fn structural_no_attempt_fallback_and_terminal_failures_preserve_borrowed_explicit_inputs() {
    for permutation in [true, false] {
        for mode in [
            Mode::NoAttempt,
            Mode::Failed,
            Mode::BorrowedAlias,
            Mode::WrongClass,
            Mode::LazyAlias,
        ] {
            if !permutation && mode == Mode::LazyAlias {
                continue;
            }
            let provider = Box::leak(Box::new(Provider {
                inner: InProcessProvider::new(),
                view_owner: runmat_accelerate_api::test_support::TestAllocationOwner::new(),
                mode,
                reads: AtomicUsize::new(0),
                permutations: AtomicUsize::new(0),
                unpublished: Mutex::new(Vec::new()),
            }));
            let _selected = provider.select();
            let input = if mode == Mode::LazyAlias {
                provider
                    .view_owner
                    .allocate_zeroed(
                        vec![2, 3],
                        runmat_accelerate_api::NumericElementType::F64,
                        GpuTensorStorage::Real,
                    )
                    .unwrap()
                    .handle
            } else {
                provider
                    .inner
                    .upload(&HostTensorView {
                        data: &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
                        shape: &[2, 3],
                    })
                    .unwrap()
            }
            .with_provenance(GpuHandleProvenance::Explicit)
            .unwrap();
            let view = input.view().clone();
            let result = call_transform(permutation, &input);
            if matches!(mode, Mode::NoAttempt | Mode::LazyAlias) {
                let Value::GpuTensor(output) = result.unwrap() else {
                    panic!("explicit result must remain resident");
                };
                assert!(runmat_accelerate_api::handle_is_explicit(&output));
                assert_eq!(
                    output.physical().element_type(),
                    input.physical().element_type()
                );
                assert_eq!(
                    output.shape(),
                    if permutation {
                        &[3, 2][..]
                    } else {
                        &[4, 3][..]
                    }
                );
                if mode == Mode::LazyAlias {
                    assert!(output.shares_allocation_with(&input));
                } else {
                    assert!(provider.reads.load(Ordering::SeqCst) > 0);
                }
            } else {
                let error = result.unwrap_err();
                assert_eq!(error.gpu_gather_retry, crate::GpuGatherRetry::Never);
                assert_eq!(provider.reads.load(Ordering::SeqCst), 0);
                if mode == Mode::Failed {
                    let source = error.source.as_ref().unwrap();
                    assert!(!runmat_accelerate_api::is_unsupported_provider_error(
                        source.as_ref()
                    ));
                    let mut cause: &(dyn std::error::Error + 'static) = source.as_ref();
                    while !cause.is::<runmat_accelerate_api::UnsupportedProviderOperation>() {
                        cause = cause.source().expect(
                            "original unsupported cause retained without replay permission",
                        );
                    }
                }
            }
            assert_eq!(input.view(), &view);
            input.allocation_lease().validate_live().unwrap();
            assert!(provider
                .unpublished
                .lock()
                .unwrap()
                .iter()
                .all(|lease| lease.acquire().is_none()));
            assert!(gpu_helpers::exact_provider_for_handle(&input).is_some());
            if mode == Mode::NoAttempt {
                let complex_data = [
                    1.0f32, 11.0, 2.0, 12.0, 3.0, 13.0, 4.0, 14.0, 5.0, 15.0, 6.0, 16.0,
                ];
                let logical_data = [1, 0, 1, 1, 0, 0];
                let indices = if permutation {
                    vec![0, 2, 4, 1, 3, 5]
                } else {
                    vec![0, 1, 0, 1, 2, 3, 2, 3, 4, 5, 4, 5]
                };
                let complex = provider
                    .inner
                    .upload_numeric_raw(&HostNumericTensorView {
                        data: HostNumericDataView::F32(&complex_data),
                        shape: &[2, 3],
                        storage: GpuTensorStorage::ComplexInterleaved,
                    })
                    .unwrap();
                let logical = provider
                    .inner
                    .upload_logical_raw(&runmat_accelerate_api::HostLogicalTensorView {
                        data: &logical_data,
                        shape: &[2, 3],
                    })
                    .unwrap();
                for (source, logical) in [(complex, false), (logical, true)] {
                    let source = source
                        .with_provenance(GpuHandleProvenance::Explicit)
                        .unwrap();
                    let result = call_transform(permutation, &source).unwrap();
                    let Value::GpuTensor(output) = result else {
                        panic!("explicit typed fallback must remain resident");
                    };
                    assert!(runmat_accelerate_api::handle_is_explicit(&output));
                    assert_eq!(
                        output.physical().element_type(),
                        source.physical().element_type()
                    );
                    assert_eq!(output.physical().storage(), source.physical().storage());
                    assert_eq!(output.view().is_logical(), logical);
                    let downloaded = block_on(
                        gpu_helpers::download_value_preserving_residency_async(provider, &output),
                    )
                    .unwrap();
                    if logical {
                        let Value::LogicalArray(actual) = downloaded else {
                            panic!("logical carrier");
                        };
                        assert_eq!(
                            actual.data,
                            indices
                                .iter()
                                .map(|&index| logical_data[index])
                                .collect::<Vec<_>>()
                        );
                    } else {
                        let Value::ComplexTensor(actual) = downloaded else {
                            panic!("complex carrier");
                        };
                        assert_eq!(
                            actual.as_f32_slice(),
                            Some(
                                indices
                                    .iter()
                                    .map(|&index| runmat_value::ComplexElement(
                                        complex_data[2 * index],
                                        complex_data[2 * index + 1]
                                    ))
                                    .collect::<Vec<_>>()
                                    .as_slice()
                            )
                        );
                    }
                    source.allocation_lease().validate_live().unwrap();
                }
            }
        }
    }
}

#[path = "structural_provider_tests/flip.rs"]
mod flip_tests;

use super::*;
mod retirement;
use runmat_accelerate_api::{
    test_support::{RetirementProbe, TestAllocation, TestAllocationOwner},
    AccelDownloadFuture, AccelNumericDownloadFuture, GpuHandleProvenance, HostNumericDataOwned,
    HostNumericDataView, HostNumericTensorOwned, HostTensorView, ThreadProviderGuard,
};
use std::{
    future::Future,
    sync::{
        atomic::{AtomicUsize, Ordering},
        Mutex,
    },
    task::{Context, Poll},
};

struct Provider {
    owner: TestAllocationOwner,
    records: Mutex<Vec<(u64, RetirementProbe)>>,
    downloads: AtomicUsize,
    reductions: AtomicUsize,
    pending: bool,
}
impl Provider {
    fn reduce(&self, input: &GpuTensorHandle, dimension: usize) -> ReduceDimResult {
        self.reductions.fetch_add(1, Ordering::SeqCst);
        let mut shape = input.shape().to_vec();
        if !shape.contains(&0) {
            shape[dimension] = 1;
        }
        let values = self.allocate(shape.clone(), input.physical().element_type(), 0.0);
        let indices = self.allocate(shape, input.physical().element_type(), 1.0);
        ReduceDimResult {
            values: values.handle,
            indices: indices.handle,
        }
    }
    fn allocate(
        &self,
        shape: Vec<usize>,
        element: NumericElementType,
        fill: f64,
    ) -> TestAllocation {
        let count = shape.iter().product();
        let data = match element {
            NumericElementType::F32 => HostNumericDataOwned::F32(vec![fill as f32; count]),
            NumericElementType::F64 => HostNumericDataOwned::F64(vec![fill; count]),
            _ => panic!("fixture floating storage"),
        };
        let allocation = self
            .owner
            .allocate_numeric(HostNumericTensorOwned {
                data,
                shape,
                storage: GpuTensorStorage::Real,
            })
            .unwrap();
        self.records
            .lock()
            .unwrap()
            .push((allocation.handle.buffer_id(), allocation.probe.clone()));
        allocation
    }
}
impl AccelProvider for Provider {
    fn reduce_min_dim<'a>(
        &'a self,
        input: &'a GpuTensorHandle,
        dimension: usize,
    ) -> runmat_accelerate_api::ProviderExecutionFuture<'a, ReduceDimResult> {
        Box::pin(async move { Ok(self.reduce(input, dimension)) })
    }
    fn reduce_max_dim<'a>(
        &'a self,
        input: &'a GpuTensorHandle,
        dimension: usize,
    ) -> runmat_accelerate_api::ProviderExecutionFuture<'a, ReduceDimResult> {
        Box::pin(async move { Ok(self.reduce(input, dimension)) })
    }
    fn device_id(&self) -> u32 {
        self.owner.device_id()
    }
    fn device_info(&self) -> String {
        "dimensional extrema publication owner".into()
    }
    fn upload(&self, _: &HostTensorView) -> anyhow::Result<GpuTensorHandle> {
        panic!("no index reupload")
    }
    fn download<'a>(&'a self, _: &'a GpuTensorHandle) -> AccelDownloadFuture<'a> {
        Box::pin(async { panic!("typed index download") })
    }
    fn download_numeric<'a>(
        &'a self,
        input: &'a GpuTensorHandle,
    ) -> AccelNumericDownloadFuture<'a> {
        Box::pin(async move {
            self.downloads.fetch_add(1, Ordering::SeqCst);
            if self.pending {
                std::future::pending::<()>().await;
            }
            let probe = self
                .records
                .lock()
                .unwrap()
                .iter()
                .find(|(id, _)| *id == input.buffer_id())
                .unwrap()
                .1
                .clone();
            let data = probe.with_data(input, |data| match data {
                HostNumericDataView::F32(data) => HostNumericDataOwned::F32(data.to_vec()),
                _ => panic!("native Double indices must stay resident"),
            })?;
            Ok(HostNumericTensorOwned {
                data,
                shape: input.shape().to_vec(),
                storage: GpuTensorStorage::Real,
            })
        })
    }
}
fn provider(pending: bool) -> &'static Provider {
    Box::leak(Box::new(Provider {
        owner: TestAllocationOwner::new(),
        records: Mutex::new(Vec::new()),
        downloads: AtomicUsize::new(0),
        reductions: AtomicUsize::new(0),
        pending,
    }))
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[cfg_attr(not(target_arch = "wasm32"), test)]
fn dimensional_values_and_hidden_index_cannot_share_one_allocation() {
    let provider = provider(false);
    let _selected = ThreadProviderGuard::set(Some(provider));
    let input = provider.allocate(vec![2, 3], NumericElementType::F64, 0.0);
    let value_and_index = provider.allocate(vec![1, 3], NumericElementType::F64, 1.0);
    let hidden_index = value_and_index
        .handle
        .with_view(value_and_index.handle.view().clone())
        .unwrap();
    assert_ne!(
        hidden_index.view_identity(),
        value_and_index.handle.view_identity()
    );
    assert!(hidden_index.shares_allocation_with(&value_and_index.handle));
    let error = futures::executor::block_on(ExtremaKind::Min.adopt_reduction_result(
        provider,
        &input.handle,
        0,
        ReduceDimResult {
            values: value_and_index.handle,
            indices: hidden_index,
        },
        ExtremaDemand::ValuesOnly,
    ))
    .expect_err("one allocation cannot represent values and one-based indices");
    assert_eq!(error.gpu_gather_retry, crate::GpuGatherRetry::Never);
    assert_eq!(provider.downloads.load(Ordering::SeqCst), 0);
    assert_eq!(value_and_index.probe.retirements(), 1);
    assert_eq!(input.probe.retirements(), 0);
    input.handle.allocation_lease().validate_live().unwrap();
    drop(input.handle);
    assert_eq!(input.probe.retirements(), 1);
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[cfg_attr(not(target_arch = "wasm32"), test)]
fn dimensional_publication_handles_rank_one_and_rejects_rank_zero_before_download() {
    let provider = provider(false);
    let _selected = ThreadProviderGuard::set(Some(provider));

    let input = provider.allocate(vec![2], NumericElementType::F32, 0.0);
    let values = provider.allocate(vec![1], NumericElementType::F32, 0.0);
    let indices = provider.allocate(vec![1], NumericElementType::F32, 1.0);
    let output = futures::executor::block_on(ExtremaKind::Min.adopt_reduction_result(
        provider,
        &input.handle,
        0,
        ReduceDimResult {
            values: values.handle,
            indices: indices.handle,
        },
        ExtremaDemand::ValuesAndIndices,
    ))
    .expect("rank-one reduction publication");
    let (output, index) = output.into_pair(ExtremaKind::Min).expect("paired output");
    let Value::GpuTensor(output) = output else {
        panic!("values must remain resident")
    };
    assert_eq!(output.shape(), &[1]);
    assert!(!output.shares_allocation_with(&input.handle));
    assert!(runmat_accelerate_api::same_provider_instance(
        gpu_helpers::exact_provider_for_handle(&output).expect("output owner"),
        gpu_helpers::exact_provider_for_handle(&input.handle).expect("input owner"),
    ));
    assert_eq!(index, Value::Num(1.0));
    assert_eq!(provider.downloads.load(Ordering::SeqCst), 1);
    drop((output, index, input.handle));
    assert_eq!(input.probe.retirements(), 1);
    assert_eq!(values.probe.retirements(), 1);
    assert_eq!(indices.probe.retirements(), 1);

    let input = provider.allocate(Vec::new(), NumericElementType::F32, 0.0);
    let values = provider.allocate(Vec::new(), NumericElementType::F32, 0.0);
    let indices = provider.allocate(Vec::new(), NumericElementType::F32, 1.0);
    let before = provider.downloads.load(Ordering::SeqCst);
    let error = futures::executor::block_on(ExtremaKind::Max.adopt_reduction_result(
        provider,
        &input.handle,
        0,
        ReduceDimResult {
            values: values.handle,
            indices: indices.handle,
        },
        ExtremaDemand::ValuesAndIndices,
    ))
    .expect_err("rank-zero input has no reducible axis");
    assert!(error.message().contains("outside the admitted input rank"));
    assert_eq!(provider.downloads.load(Ordering::SeqCst), before);
    assert_eq!(values.probe.retirements(), 1);
    assert_eq!(indices.probe.retirements(), 1);
    assert_eq!(input.probe.retirements(), 0);
    drop(input.handle);
    assert_eq!(input.probe.retirements(), 1);
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[cfg_attr(not(target_arch = "wasm32"), test)]
fn dimensional_publication_preserves_native_values_and_double_index_contract() {
    let provider = provider(false);
    let _selected = ThreadProviderGuard::set(Some(provider));
    for kind in [ExtremaKind::Min, ExtremaKind::Max] {
        for (element, index_element) in [
            (NumericElementType::F32, NumericElementType::F32),
            (NumericElementType::F32, NumericElementType::F64),
            (NumericElementType::F64, NumericElementType::F64),
        ] {
            for shape in [vec![2, 3], vec![0, 3], vec![3, 0]] {
                for dimension in [0, 1] {
                    let input = provider.allocate(shape.clone(), element, 0.0);
                    let source = input
                        .handle
                        .with_provenance(GpuHandleProvenance::Explicit)
                        .unwrap();
                    let mut output_shape = shape.clone();
                    if !shape.contains(&0) {
                        output_shape[dimension] = 1;
                    }
                    let values = provider.allocate(output_shape.clone(), element, 0.0);
                    let indices = provider.allocate(output_shape.clone(), index_element, 1.0);
                    let before = provider.downloads.load(Ordering::SeqCst);
                    let (output, index_output) =
                        futures::executor::block_on(kind.adopt_reduction_result(
                            provider,
                            &source,
                            dimension,
                            ReduceDimResult {
                                values: values.handle,
                                indices: indices.handle,
                            },
                            ExtremaDemand::ValuesAndIndices,
                        ))
                        .unwrap()
                        .into_pair(kind)
                        .unwrap();
                    let Value::GpuTensor(ref output_handle) = output else {
                        panic!("values must stay resident")
                    };
                    assert_eq!(output_handle.physical().element_type(), element);
                    assert_eq!(output_handle.shape(), output_shape);
                    assert!(runmat_accelerate_api::handle_is_explicit(output_handle));
                    if index_element == NumericElementType::F32 {
                        assert_eq!(
                            index_output,
                            crate::builtins::common::tensor::tensor_into_value(
                                Tensor::new(vec![1.0; output_shape.iter().product()], output_shape)
                                    .unwrap()
                            )
                        );
                        assert_eq!(indices.probe.retirements(), 1);
                        assert_eq!(provider.downloads.load(Ordering::SeqCst), before + 1);
                    } else {
                        let Value::GpuTensor(ref index_handle) = index_output else {
                            panic!("native Double indices stay resident")
                        };
                        assert_eq!(
                            index_handle.physical().element_type(),
                            NumericElementType::F64
                        );
                        assert!(runmat_accelerate_api::handle_is_explicit(index_handle));
                        assert_eq!(provider.downloads.load(Ordering::SeqCst), before);
                    }
                    drop((output, index_output));
                    assert_eq!(values.probe.retirements(), 1);
                    assert_eq!(indices.probe.retirements(), 1);
                    drop(source);
                    input.handle.allocation_lease().validate_live().unwrap();
                    assert_eq!(input.probe.retirements(), 0);
                    drop(input.handle);
                    assert_eq!(input.probe.retirements(), 1);
                }
            }
        }
    }
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[cfg_attr(not(target_arch = "wasm32"), test)]
fn dimensional_pending_index_download_owns_both_outputs_until_abandonment() {
    let provider = provider(true);
    let _selected = ThreadProviderGuard::set(Some(provider));
    let input = provider.allocate(vec![2, 3], NumericElementType::F32, 0.0);
    let values = provider.allocate(vec![1, 3], NumericElementType::F32, 0.0);
    let indices = provider.allocate(vec![1, 3], NumericElementType::F32, 1.0);
    let mut future = Box::pin(ExtremaKind::Min.adopt_reduction_result(
        provider,
        &input.handle,
        0,
        ReduceDimResult {
            values: values.handle,
            indices: indices.handle,
        },
        ExtremaDemand::ValuesAndIndices,
    ));
    let mut context = Context::from_waker(futures::task::noop_waker_ref());
    assert!(matches!(future.as_mut().poll(&mut context), Poll::Pending));
    assert_eq!(values.probe.retirements(), 0);
    assert_eq!(indices.probe.retirements(), 0);
    drop(future);
    assert_eq!(values.probe.retirements(), 1);
    assert_eq!(indices.probe.retirements(), 1);
    assert_eq!(input.probe.retirements(), 0);
    drop(input.handle);
    assert_eq!(input.probe.retirements(), 1);
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[cfg_attr(not(target_arch = "wasm32"), test)]
fn dimensional_malformed_borrowed_result_rejects_without_retiring_caller() {
    let provider = provider(false);
    let _selected = ThreadProviderGuard::set(Some(provider));
    let input = provider.allocate(vec![2, 3], NumericElementType::F32, 0.0);
    let indices = provider.allocate(vec![1, 3], NumericElementType::F64, 1.0);
    let error = futures::executor::block_on(ExtremaKind::Max.adopt_reduction_result(
        provider,
        &input.handle,
        0,
        ReduceDimResult {
            values: input.handle.clone(),
            indices: indices.handle,
        },
        ExtremaDemand::ValuesAndIndices,
    ))
    .unwrap_err();
    assert_eq!(error.gpu_gather_retry, crate::GpuGatherRetry::Never);
    assert_eq!(provider.downloads.load(Ordering::SeqCst), 0);
    assert_eq!(indices.probe.retirements(), 1);
    assert_eq!(input.probe.retirements(), 0);
    input.handle.allocation_lease().validate_live().unwrap();
    drop(input.handle);
    assert_eq!(input.probe.retirements(), 1);
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[cfg_attr(not(target_arch = "wasm32"), test)]
fn dimensional_owned_output_contract_rejects_wrong_class_owner_and_indices() {
    let provider = provider(false);
    let _selected = ThreadProviderGuard::set(Some(provider));
    let foreign = TestAllocationOwner::new();
    for fault in [
        "value-class",
        "value-owner",
        "index-shape",
        "index-logical",
        "index-value",
    ] {
        let input = provider.allocate(vec![2, 3], NumericElementType::F32, 0.0);
        let values = match fault {
            "value-class" => provider.allocate(vec![1, 3], NumericElementType::F64, 0.0),
            "value-owner" => foreign
                .allocate_zeroed(vec![1, 3], NumericElementType::F32, GpuTensorStorage::Real)
                .unwrap(),
            _ => provider.allocate(vec![1, 3], NumericElementType::F32, 0.0),
        };
        let indices = match fault {
            "index-shape" => provider.allocate(vec![2, 3], NumericElementType::F64, 1.0),
            "index-logical" => provider
                .owner
                .allocate_logical(vec![1, 1, 1], vec![1, 3])
                .unwrap(),
            "index-value" => provider.allocate(vec![1, 3], NumericElementType::F32, 1.5),
            _ => provider.allocate(vec![1, 3], NumericElementType::F64, 1.0),
        };
        let before = provider.downloads.load(Ordering::SeqCst);
        let error = futures::executor::block_on(ExtremaKind::Min.adopt_reduction_result(
            provider,
            &input.handle,
            0,
            ReduceDimResult {
                values: values.handle,
                indices: indices.handle,
            },
            ExtremaDemand::ValuesAndIndices,
        ))
        .unwrap_err();
        assert_eq!(error.gpu_gather_retry, crate::GpuGatherRetry::Never);
        assert_eq!(
            provider.downloads.load(Ordering::SeqCst),
            before + usize::from(fault == "index-value")
        );
        assert_eq!(values.probe.retirements(), 1, "{fault}");
        assert_eq!(indices.probe.retirements(), 1, "{fault}");
        input.handle.allocation_lease().validate_live().unwrap();
        assert_eq!(input.probe.retirements(), 0);
        drop(input.handle);
        assert_eq!(input.probe.retirements(), 1);
    }
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[cfg_attr(not(target_arch = "wasm32"), test)]
fn dimensional_publication_checks_axis_before_shape_access() {
    let provider = provider(false);
    let _selected = ThreadProviderGuard::set(Some(provider));
    for shape in [vec![], vec![1, 2, 3]] {
        let dimension = shape.len();
        let input = provider.allocate(shape, NumericElementType::F32, 0.0);
        let values = provider.allocate(vec![1, 1], NumericElementType::F32, 0.0);
        let indices = provider.allocate(vec![1, 1], NumericElementType::F32, 1.0);
        let error = futures::executor::block_on(ExtremaKind::Min.adopt_reduction_result(
            provider,
            &input.handle,
            dimension,
            ReduceDimResult {
                values: values.handle,
                indices: indices.handle,
            },
            ExtremaDemand::ValuesAndIndices,
        ))
        .unwrap_err();
        assert_eq!(error.gpu_gather_retry, crate::GpuGatherRetry::Never);
        assert_eq!(provider.downloads.load(Ordering::SeqCst), 0);
        assert_eq!(values.probe.retirements(), 1);
        assert_eq!(indices.probe.retirements(), 1);
        assert_eq!(input.probe.retirements(), 0);
        drop(input.handle);
        assert_eq!(input.probe.retirements(), 1);
    }
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[cfg_attr(not(target_arch = "wasm32"), test)]
fn builtin_dimensional_demand_skips_only_unrequested_index_adaptation() {
    for name in ["min", "max"] {
        let provider = provider(false);
        let _selected = ThreadProviderGuard::set(Some(provider));
        let input = provider.allocate(vec![2, 3], NumericElementType::F32, 0.0);
        let source = input
            .handle
            .with_provenance(GpuHandleProvenance::Explicit)
            .unwrap();
        for count in [0, 1, 2] {
            let output = futures::executor::block_on(crate::call_builtin_async_with_outputs(
                name,
                &[Value::GpuTensor(source.clone())],
                count,
            ))
            .unwrap();
            let Value::OutputList(outputs) = output.value() else {
                panic!("explicit output-count result")
            };
            assert_eq!(outputs.len(), count);
            assert_eq!(provider.reductions.load(Ordering::SeqCst), count + 1);
            assert_eq!(
                provider.downloads.load(Ordering::SeqCst),
                usize::from(count == 2)
            );
            if count > 0 {
                let Value::GpuTensor(ref values) = outputs[0] else {
                    panic!("resident Single values")
                };
                assert_eq!(values.physical().element_type(), NumericElementType::F32);
                assert!(runmat_accelerate_api::handle_is_explicit(values));
            }
            if count == 2 {
                assert_eq!(
                    outputs[1],
                    Value::Tensor(Tensor::new(vec![1.0; 3], vec![1, 3]).unwrap())
                );
            }
            drop(output);
            for (id, probe) in provider.records.lock().unwrap().iter() {
                assert_eq!(
                    probe.retirements(),
                    usize::from(*id != input.handle.buffer_id())
                );
            }
        }
        drop(source);
        drop(input.handle);
        assert_eq!(input.probe.retirements(), 1);
    }
}

use super::*;
use runmat_accelerate_api::{AccelProvider, HostLogicalTensorView, HostTensorView};

fn values(provider: &WgpuProvider, input: &GpuTensorHandle) -> Vec<f64> {
    futures::executor::block_on(provider.download(input))
        .unwrap()
        .data
}

fn managed_index_context() -> (
    runmat_runtime::context::RuntimeContext,
    runmat_runtime::context::host_effect::ManagedHostEffectLedger,
) {
    use runmat_runtime::context::{
        host_effect::ManagedHostEffectLedger, ParallelCapability, RuntimeContext,
        RuntimeParallelResources, RuntimeParallelService, RuntimeServicePorts,
    };
    use std::rc::Rc;
    struct Managed(ManagedHostEffectLedger);
    impl RuntimeParallelService for Managed {
        fn supports(&self, _: ParallelCapability) -> bool {
            false
        }
        fn placement_resources(&self) -> RuntimeParallelResources {
            let state = self.0.snapshot();
            RuntimeParallelResources {
                memory_available_bytes: Some(state.free_bytes()),
                epoch: state.epoch,
                ..Default::default()
            }
        }
        fn managed_host_effect_ledger(&self) -> Option<ManagedHostEffectLedger> {
            Some(self.0.clone())
        }
    }
    let ledger = ManagedHostEffectLedger::new(16 * 1024 * 1024);
    let runtime = RuntimeContext::new(Rc::new(
        runmat_runtime::execution::RuntimeExecutionService::new(),
    ))
    .with_service_ports(
        RuntimeServicePorts::default().with_parallel(Rc::new(Managed(ledger.clone()))),
    );
    (runtime, ledger)
}

fn index_admission(
    provider: &WgpuProvider,
    runtime: &runmat_runtime::context::RuntimeContext,
    source: &GpuTensorHandle,
    rhs: Option<&GpuTensorHandle>,
    rank: usize,
) -> (
    Box<dyn runmat_execution::resource::ProviderHostMetadataAdmission>,
    runmat_runtime::context::host_allocation::HostAllocationReceipt,
) {
    use runmat_runtime::context::host_allocation::{
        provider_registration_host_quote, HostAllocationPlan, HostAllocationReceipt,
    };
    let quote = provider_registration_host_quote(provider, rank, 1, 1)
        .unwrap()
        .unwrap();
    let runmat_execution::ResourceByteBound::Known(scratch) =
        provider.indexing_host_workspace_bound(source, rhs)
    else {
        panic!("exact native indexing bound")
    };
    let receipt = HostAllocationReceipt::admit(
        Some(runtime),
        HostAllocationPlan::new(
            quote.preparation_bytes.checked_add(scratch).unwrap(),
            quote.retained_bytes,
        )
        .unwrap(),
    )
    .unwrap()
    .unwrap();
    let permit = provider
        .prepare_output_host_metadata(Box::new(receipt.retained_view_permit(runtime).unwrap()), 1)
        .unwrap();
    (permit, receipt)
}

#[test]
#[ignore = "requires actual WGPU device; run explicitly for F32 and F64"]
fn transposed_diag_gather_and_in_place_scatter_use_logical_order() {
    let provider = WgpuProvider::new(Default::default()).expect("actual WGPU device required");
    let source = provider
        .upload(&HostTensorView {
            data: &[1., 2., 3., 4., 5., 6.],
            shape: &[2, 3],
        })
        .unwrap();
    let transposed = provider.transpose_exec(&source).unwrap();
    assert!(transposed.shares_allocation_with(&source));
    let diag = provider.diag_extract_exec(&transposed, 0).unwrap();
    assert_eq!(values(&provider, &diag), vec![1., 4.]);
    let gathered = provider
        .gather_linear_exec(&transposed, &[1, 4], &[2, 1])
        .unwrap();
    assert_eq!(values(&provider, &gathered), vec![3., 4.]);
    // Finite registration uses the same transposed bit-copy kernel without
    // materializing a reshape/host index mirror. Claims outlive preparation.
    let (runtime, ledger) = managed_index_context();
    let (permit, receipt) = index_admission(&provider, &runtime, &transposed, None, 2);
    let admitted = provider
        .gather_linear_admitted(&transposed, &[1, 4], &[2, 1], permit.as_ref())
        .unwrap();
    assert_eq!(values(&provider, &admitted), vec![3., 4.]);
    drop(permit);
    drop(receipt);
    assert!(ledger.snapshot().live_claimed_bytes > 0);
    drop(admitted);
    assert_eq!(ledger.snapshot().live_claimed_bytes, 0);
    let paired = runmat_accelerate_api::upload_numeric_protecting(
        &provider,
        &runmat_accelerate_api::HostNumericTensorView {
            data: runmat_accelerate_api::HostNumericDataView::U64(&[
                u64::MAX,
                9_007_199_254_740_993,
                2,
                3,
                4,
                5,
            ]),
            shape: &[1, 3],
            storage: GpuTensorStorage::ComplexInterleaved,
        },
        &[],
    )
    .unwrap();
    let scalar = runmat_accelerate_api::upload_numeric_protecting(
        &provider,
        &runmat_accelerate_api::HostNumericTensorView {
            data: runmat_accelerate_api::HostNumericDataView::U64(&[7, 9]),
            shape: &[1, 1],
            storage: GpuTensorStorage::ComplexInterleaved,
        },
        &[],
    )
    .unwrap();
    let (permit, receipt) = index_admission(&provider, &runtime, &paired, Some(&scalar), 2);
    let broadcast = provider
        .scatter_linear_functional_admitted(
            &paired,
            &[0, 2],
            &scalar,
            runmat_accelerate_api::ProviderScatterValueMapping::Scalar,
            permit.as_ref(),
        )
        .unwrap();
    assert_eq!(
        futures::executor::block_on(provider.download_numeric(&broadcast))
            .unwrap()
            .data,
        runmat_accelerate_api::HostNumericDataOwned::U64(vec![7, 9, 2, 3, 7, 9])
    );
    assert_eq!(
        futures::executor::block_on(provider.download_numeric(&paired))
            .unwrap()
            .data,
        runmat_accelerate_api::HostNumericDataOwned::U64(vec![
            u64::MAX,
            9_007_199_254_740_993,
            2,
            3,
            4,
            5
        ])
    );
    drop(permit);
    drop(receipt);
    drop(broadcast);
    assert_eq!(ledger.snapshot().live_claimed_bytes, 0);
    assert_eq!(ledger.snapshot().live_reserved_bytes, 0);
    // The same typed upload validator consumes exact caller-funded native
    // integer encoding, with both borrowed allocations protected.
    {
        use runmat_runtime::context::host_allocation::{
            provider_registration_host_quote, HostAllocationPlan, HostAllocationReceipt,
            HostTypedWorkspace,
        };
        let lanes = [u64::MAX, 9_007_199_254_740_993];
        let view = runmat_accelerate_api::HostNumericTensorView {
            data: runmat_accelerate_api::HostNumericDataView::U64(&lanes),
            shape: &[1, 1],
            storage: GpuTensorStorage::ComplexInterleaved,
        };
        let words = provider
            .numeric_upload_encoding_words(view.data.element_type(), view.data.len())
            .unwrap();
        assert_eq!(words, 4);
        let quote = provider_registration_host_quote(&provider, 2, 1, 1)
            .unwrap()
            .unwrap();
        let temporary = HostTypedWorkspace::<u32>::quoted_bytes(words).unwrap();
        let receipt = HostAllocationReceipt::admit(
            Some(&runtime),
            HostAllocationPlan::new(
                quote.preparation_bytes.checked_add(temporary).unwrap(),
                quote.retained_bytes,
            )
            .unwrap(),
        )
        .unwrap()
        .unwrap();
        let permit = provider
            .prepare_output_host_metadata(
                Box::new(receipt.retained_view_permit(&runtime).unwrap()),
                1,
            )
            .unwrap();
        let mut scratch = receipt.temporary_typed::<u32>(&runtime, words).unwrap();
        let uploaded = runmat_accelerate_api::upload_numeric_protecting_admitted_with_scratch(
            &provider,
            &view,
            &[&source, &transposed],
            Some(permit.as_ref()),
            Some(runmat_accelerate_api::HostNumericUploadScratch {
                words: scratch.as_mut_slice(),
            }),
        )
        .unwrap();
        assert_eq!(
            futures::executor::block_on(provider.download_numeric(&uploaded))
                .unwrap()
                .data,
            runmat_accelerate_api::HostNumericDataOwned::U64(lanes.to_vec())
        );
        drop(scratch);
        drop(permit);
        drop(receipt);
        assert!(ledger.snapshot().live_claimed_bytes > 0);
        drop(uploaded);
        assert_eq!(ledger.snapshot().live_claimed_bytes, 0);
        assert_eq!(ledger.snapshot().live_reserved_bytes, 0);
    }
    let replacement = provider
        .upload(&HostTensorView {
            data: &[20., 30.],
            shape: &[2, 1],
        })
        .unwrap();
    let (permit, receipt) = index_admission(&provider, &runtime, &source, Some(&replacement), 2);
    let admitted = provider
        .scatter_linear_functional_admitted(
            &source,
            &[1, 4],
            &replacement,
            runmat_accelerate_api::ProviderScatterValueMapping::Linear,
            permit.as_ref(),
        )
        .unwrap();
    assert_eq!(values(&provider, &admitted), vec![1., 20., 3., 4., 30., 6.]);
    assert_eq!(values(&provider, &source), vec![1., 2., 3., 4., 5., 6.]);
    drop(permit);
    drop(receipt);
    assert!(ledger.snapshot().live_claimed_bytes > 0);
    drop(admitted);
    assert_eq!(ledger.snapshot().live_claimed_bytes, 0);
    assert_eq!(ledger.snapshot().live_reserved_bytes, 0);
    let (permit, receipt) =
        index_admission(&provider, &runtime, &transposed, Some(&replacement), 2);
    assert!(provider
        .scatter_linear_functional_admitted(
            &transposed,
            &[1, 4],
            &replacement,
            runmat_accelerate_api::ProviderScatterValueMapping::Linear,
            permit.as_ref()
        )
        .unwrap_err()
        .is_unsupported());
    drop(permit);
    drop(receipt);
    assert_eq!(ledger.snapshot().live_claimed_bytes, 0);
    let functional = provider
        .scatter_linear_functional_exec(&transposed, &[1, 4], &replacement)
        .unwrap();
    assert!(!functional.shares_allocation_with(&source));
    assert!(!functional.shares_allocation_with(&replacement));
    assert_eq!(
        values(&provider, &functional),
        vec![1., 20., 5., 2., 30., 6.]
    );
    assert_eq!(values(&provider, &source), vec![1., 2., 3., 4., 5., 6.]);
    provider
        .scatter_linear_exec(&transposed, &[1, 4], &replacement)
        .unwrap();
    assert_eq!(values(&provider, &source), vec![1., 2., 20., 30., 5., 6.]);
    assert_eq!(
        values(&provider, &transposed),
        vec![1., 20., 5., 2., 30., 6.]
    );
    drop(diag);
    drop(gathered);
    assert!(source.allocation_lease().validate_live().is_ok());
    // Exact physical aliases retain snapshot assignment semantics, including
    // when the read and write logical orders differ.
    provider
        .scatter_linear_exec(&transposed, &[0, 1, 2, 3, 4, 5], &source)
        .unwrap();
    assert_eq!(
        values(&provider, &transposed),
        vec![1., 2., 20., 30., 5., 6.]
    );
}

#[test]
#[ignore = "requires actual WGPU F32 device; RUNMAT_WGPU_FORCE_PRECISION=f32"]
fn logical_transposed_diag_and_functional_scatter_preserve_producer_and_source() {
    let provider = WgpuProvider::new(Default::default()).expect("actual WGPU device required");
    assert_eq!(
        provider.precision(),
        runmat_accelerate_api::ProviderPrecision::F32
    );
    let source = provider
        .upload_logical(&HostLogicalTensorView {
            data: &[0, 1, 0, 1, 0, 0],
            shape: &[2, 3],
        })
        .unwrap();
    let transposed = provider.transpose_exec(&source).unwrap();
    let diag = provider.diag_extract_exec(&transposed, 0).unwrap();
    assert!(diag.view().is_logical());
    assert_eq!(values(&provider, &diag), vec![0., 1.]);
    let column = provider
        .upload_logical(&HostLogicalTensorView {
            data: &[1, 1, 1],
            shape: &[3, 1],
        })
        .unwrap();
    let replaced =
        futures::executor::block_on(provider.scatter_column_exec(&transposed, 0, &column)).unwrap();
    assert!(replaced.view().is_logical());
    assert!(!replaced.shares_allocation_with(&source));
    assert_eq!(values(&provider, &replaced), vec![1., 1., 1., 1., 1., 0.]);
    let row = provider
        .upload_logical(&HostLogicalTensorView {
            data: &[1, 0],
            shape: &[1, 2],
        })
        .unwrap();
    let replaced =
        futures::executor::block_on(provider.scatter_row_exec(&transposed, 1, &row)).unwrap();
    assert!(replaced.view().is_logical());
    assert_eq!(values(&provider, &replaced), vec![0., 1., 0., 1., 0., 0.]);
    assert_eq!(values(&provider, &source), vec![0., 1., 0., 1., 0., 0.]);
}

#[test]
#[ignore = "requires actual WGPU device and RUNMAT_WG=64"]
fn diagonal_constructor_covers_nondefault_workgroup_boundary() {
    let provider = WgpuProvider::new(Default::default()).expect("actual WGPU device required");
    assert_eq!(
        provider.workgroup_config.scalar, 64,
        "run with RUNMAT_WG=64"
    );
    let data: Vec<f64> = (0..65).map(|value| value as f64 + 1.).collect();
    let source = provider
        .upload(&HostTensorView {
            data: &data,
            shape: &[65, 1],
        })
        .unwrap();
    let output = provider.diag_from_vector_exec(&source, 0).unwrap();
    let actual = values(&provider, &output);
    for (index, expected) in data.iter().enumerate() {
        assert_eq!(actual[index * 66], *expected);
    }
    assert_eq!(output.shape(), &[65, 65]);
}

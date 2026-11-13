use super::*;
use runmat_accelerate_api::{
    AccelProvider, GpuTensorStorage, HostNumericDataView, HostNumericTensorView,
};

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen_test::wasm_bindgen_test(async)]
async fn browser_logical_export_materializes_its_own_buffer_without_host_transfer() {
    let provider = WgpuProvider::new_async(Default::default())
        .await
        .expect("actual browser WebGPU device required");
    let input = provider
        .upload_numeric(&HostNumericTensorView {
            data: HostNumericDataView::F32(&[1.25, -0.0, 3.5, 4.25, 5.75, 6.125]),
            shape: &[2, 3],
            storage: GpuTensorStorage::Real,
        })
        .expect("selected provider upload");
    let transposed = provider.transpose_exec(&input).expect("lazy transpose");
    let before = provider.telemetry_snapshot();
    let source = provider
        .export_wgpu_buffer_exec(&input)
        .unwrap()
        .expect("source export");
    let materialized = provider
        .export_wgpu_buffer_exec(&transposed)
        .unwrap()
        .expect("materialized export");
    assert_eq!(materialized.shape, vec![3, 2]);
    assert_eq!(materialized.len, 6);
    assert!(!same_buffer(&source, &materialized));
    assert!(provider
        .access
        .with_shared(|permit| std::ptr::eq(
            source.buffer(permit).unwrap(),
            provider.get_entry_raw(&input).unwrap().buffer.as_ref()
        ))
        .unwrap());
    let after = provider.telemetry_snapshot();
    assert_eq!(after.upload_bytes, before.upload_bytes);
    assert_eq!(after.download_bytes, before.download_bytes);
    assert!(!materialized
        .owning_handle()
        .shares_allocation_with(&transposed));
}

fn buffer_size(exported: &WgpuBufferRef) -> u64 {
    exported
        .access()
        .with_shared(|permit| exported.buffer(permit).unwrap().size())
        .unwrap()
}

fn same_buffer(left: &WgpuBufferRef, right: &WgpuBufferRef) -> bool {
    left.access()
        .with_shared(|permit| {
            std::ptr::eq(left.buffer(permit).unwrap(), right.buffer(permit).unwrap())
        })
        .unwrap()
}

fn read_f32(provider: &WgpuProvider, exported: &WgpuBufferRef) -> Vec<u32> {
    let bytes = (exported.len * 4) as u64;
    let staging = provider
        .with_emission(|emission, pending| -> anyhow::Result<_> {
            let staging = runmat_accelerate_api::WgpuBufferOwner::new(
                emission.device().create_buffer(&wgpu::BufferDescriptor {
                    label: Some("logical-export-test-readback"),
                    size: bytes,
                    usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                }),
            );
            let mut encoder = pending.profiled_encoder(emission, &Default::default());
            encoder.copy_buffer_to_buffer(exported.buffer(emission)?, 0, &staging, 0, bytes);
            provider.submit(emission, pending, encoder)?;
            Ok(staging)
        })
        .unwrap()
        .unwrap();
    let data = futures::executor::block_on(provider.map_readback_bytes(
        staging,
        bytes,
        "logical export test",
    ))
    .unwrap();
    assert_eq!(data.len(), bytes as usize);
    data.chunks_exact(4)
        .map(|bytes| u32::from_le_bytes(bytes.try_into().unwrap()))
        .collect()
}

#[test]
#[ignore = "requires actual WGPU device; explicitly run, no unavailable-device skip"]
fn logical_wgpu_export_materializes_asymmetric_transpose_without_host_transfer_or_registration_leak(
) {
    let provider = WgpuProvider::new(Default::default()).expect("actual WGPU device required");
    let baseline = provider.allocations.retained_bytes();
    let values = [1.25_f32, -0.0, 3.5, 4.25, 5.75, 6.125];
    let input = provider
        .upload_numeric(&HostNumericTensorView {
            data: HostNumericDataView::F32(&values),
            shape: &[2, 3],
            storage: GpuTensorStorage::Real,
        })
        .unwrap();
    let transposed = provider.transpose_exec(&input).unwrap();
    let original = transposed.clone();
    let layout = runmat_accelerate_api::handle_transpose_info(&transposed);
    assert!(layout.is_some());
    let logical = runmat_accelerate_api::handle_is_logical(&transposed);
    let class = runmat_accelerate_api::handle_class_identity(&transposed);
    let provenance = runmat_accelerate_api::handle_provenance(&transposed);
    // Unsupported physical rendering forms are pure absence, not an
    // operational failure or an implicit integer-to-float export.
    let integer = provider
        .upload_numeric(&HostNumericTensorView {
            data: HostNumericDataView::U64(&[9_007_199_254_740_993]),
            shape: &[1, 1],
            storage: GpuTensorStorage::Real,
        })
        .unwrap();
    let retained = provider.allocations.retained_bytes();
    let integer_before = provider.telemetry_snapshot();
    assert!(provider
        .export_wgpu_buffer_exec(&integer)
        .unwrap()
        .is_none());
    let integer_after = provider.telemetry_snapshot();
    assert_eq!(integer_before.upload_bytes, integer_after.upload_bytes);
    assert_eq!(integer_before.download_bytes, integer_after.download_bytes);
    assert_eq!(provider.allocations.retained_bytes(), retained);
    integer.allocation_lease().validate_live().unwrap();
    drop(integer);
    let before = provider.telemetry_snapshot();
    let registered = provider.allocations.retained_bytes();
    let contiguous = provider.export_wgpu_buffer_exec(&input).unwrap().unwrap();
    assert!(provider
        .access
        .with_shared(|permit| std::ptr::eq(
            contiguous.buffer(permit).unwrap(),
            provider.get_entry_raw(&input).unwrap().buffer.as_ref()
        ))
        .unwrap());
    let exported = provider
        .export_wgpu_buffer_exec(&transposed)
        .unwrap()
        .unwrap();
    assert_eq!(exported.shape, vec![3, 2]);
    assert_eq!(exported.len, 6);
    assert_eq!(exported.element_size, 4);
    assert_eq!(exported.precision, ProviderPrecision::F32);
    assert_eq!(buffer_size(&exported), 24);
    assert!(!same_buffer(&exported, &contiguous));
    assert_eq!(transposed, original);
    assert_eq!(
        runmat_accelerate_api::handle_transpose_info(&transposed),
        layout
    );
    assert_eq!(
        runmat_accelerate_api::handle_is_logical(&transposed),
        logical
    );
    assert_eq!(
        runmat_accelerate_api::handle_class_identity(&transposed),
        class
    );
    assert_eq!(
        runmat_accelerate_api::handle_provenance(&transposed),
        provenance
    );
    assert_eq!(
        provider.get_entry_raw(&transposed).unwrap().shape,
        vec![3, 2]
    );
    let after = provider.telemetry_snapshot();
    assert_eq!(after.upload_bytes, before.upload_bytes);
    assert_eq!(after.download_bytes, before.download_bytes);
    assert_eq!(
        provider.allocations.retained_bytes(),
        registered + buffer_size(&exported),
        "materialized export retains its own physical allocation accounting"
    );
    assert!(!exported
        .owning_handle()
        .shares_allocation_with(contiguous.owning_handle()));
    drop((transposed, input, original));
    assert_eq!(
        provider.allocations.retained_bytes(),
        registered + buffer_size(&exported)
    );
    // Both Arc exports, including the materialized independent allocation,
    // remain valid after source registration retirement and later allocation.
    let replacement = provider
        .upload_numeric(&HostNumericTensorView {
            data: HostNumericDataView::F32(&[99.0; 6]),
            shape: &[2, 3],
            storage: GpuTensorStorage::Real,
        })
        .unwrap();
    assert!(!provider
        .access
        .with_shared(|permit| std::ptr::eq(
            provider
                .get_entry_raw(&replacement)
                .unwrap()
                .buffer
                .as_ref(),
            contiguous.buffer(permit).unwrap()
        ))
        .unwrap());
    assert_eq!(read_f32(&provider, &contiguous), values.map(f32::to_bits));
    assert_eq!(
        read_f32(&provider, &exported),
        [values[0], values[2], values[4], values[1], values[3], values[5]].map(f32::to_bits)
    );
    drop(replacement);
    let export_clone = exported.clone();
    drop(exported);
    assert_eq!(
        provider.allocations.retained_bytes(),
        registered + buffer_size(&export_clone)
    );
    drop(export_clone);
    assert_eq!(provider.allocations.retained_bytes(), registered);
    drop(contiguous);
    assert_eq!(provider.allocations.retained_bytes(), baseline);
}

#[test]
#[ignore = "requires actual WGPU device; explicitly run, no unavailable-device skip"]
fn malformed_export_layout_is_rejected_before_dispatch_or_registration() {
    use runmat_accelerate_api::observation::{
        with_provider_observation, ProviderObservationStatus, ProviderOperationReceipt,
    };
    let provider = WgpuProvider::new(Default::default()).expect("actual WGPU device required");
    let input = provider
        .upload_numeric(&HostNumericTensorView {
            data: HostNumericDataView::F32(&[1.0; 6]),
            shape: &[2, 3],
            storage: GpuTensorStorage::Real,
        })
        .unwrap();
    let transposed = provider.transpose_exec(&input).unwrap();
    let before = provider.placement_resources().live_bytes;
    let mut wire = transposed.wire_descriptor();
    wire.view = wire
        .view
        .with_layout(runmat_accelerate_api::GpuTensorLayout::Transpose(
            runmat_accelerate_api::TransposeInfo {
                base_rows: 1,
                base_cols: 6,
            },
        ));
    let mut receipt = ProviderOperationReceipt::new();
    assert!(
        with_provider_observation(Some(receipt.recorder()), || provider.acquire_live(&wire))
            .is_err()
    );
    receipt.publish().unwrap();
    assert!(matches!(
        receipt.take_completed().unwrap().status,
        ProviderObservationStatus::Completed { submissions: 0, .. }
    ));
    assert_eq!(provider.placement_resources().live_bytes, before);
    drop((transposed, input));
}

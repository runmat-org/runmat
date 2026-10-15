use super::*;
use pollster::FutureExt;
use runmat_accelerate_api::{
    test_support::TestAllocationOwner, GpuTensorStorage, NumericElementType, WgpuAccessError,
    WgpuBufferRef,
};
use std::error::Error;
use wgpu::util::DeviceExt;

#[test]
fn cleanup_failure_cannot_report_success_or_erase_the_original_error() {
    let cleanup = PlotReadbackError::UnmapPanicked;
    let primary = PlotReadbackError::Decode("decode");
    assert_eq!(
        combine_cleanup(Ok(42), Err(cleanup.clone())),
        Err(cleanup.clone())
    );
    assert_eq!(
        combine_cleanup::<()>(Err(primary.clone()), Err(cleanup.clone())),
        Err(PlotReadbackError::WithCleanup {
            primary: Box::new(primary),
            cleanup: Box::new(cleanup)
        })
    );
    assert_eq!(combine_cleanup(Ok(42), Ok(())), Ok(42));

    let error = combine_cleanup::<()>(
        Err(PlotReadbackError::Access(
            runmat_accelerate_api::WgpuAccessError::WrongDevice,
        )),
        Err(PlotReadbackError::CleanupAccess(
            runmat_accelerate_api::WgpuAccessError::Poisoned,
        )),
    )
    .unwrap_err();
    assert_eq!(error.to_string(),
        "WGPU permit belongs to another device authority; additionally, GPU readback cleanup could not acquire its device: WGPU emission gate is poisoned");
    assert_eq!(
        error.cleanup_cause(),
        Some(&PlotReadbackError::CleanupAccess(
            runmat_accelerate_api::WgpuAccessError::Poisoned,
        ))
    );
    let primary = error.source().expect("primary source remains inspectable");
    assert!(primary.downcast_ref::<PlotReadbackError>().is_some());
    assert!(primary
        .source()
        .and_then(|source| source.downcast_ref::<runmat_accelerate_api::WgpuAccessError>())
        .is_some());
}

fn device_access() -> WgpuDeviceAccess {
    let instance = wgpu::Instance::default();
    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions::default())
        .block_on()
        .expect("this hardware gate requires an actual WGPU adapter");
    let (device, queue) = adapter
        .request_device(&crate::wgpu_compat::default_device_descriptor(), None)
        .block_on()
        .expect("this hardware gate requires an actual WGPU device");
    WgpuDeviceAccess::new(device, queue)
}

#[test]
#[ignore = "requires an actual native WGPU adapter"]
fn empty_export_authenticates_device_and_live_allocation_before_success() {
    let access = device_access();
    let other_access = device_access();
    let allocation = TestAllocationOwner::default()
        .allocate_zeroed(vec![0, 0], NumericElementType::F64, GpuTensorStorage::Real)
        .expect("empty contiguous allocation");
    let buffer = access
        .with_shared(|emission| {
            runmat_accelerate_api::WgpuBufferOwner::new(emission.device().create_buffer(
                &wgpu::BufferDescriptor {
                    label: Some("empty-plot-export-authentication"),
                    size: 4,
                    usage: wgpu::BufferUsages::COPY_SRC,
                    mapped_at_creation: false,
                },
            ))
        })
        .expect("create source on exact device");
    let export =
        WgpuBufferRef::from_owned_buffer(buffer, allocation.handle.clone(), access.clone())
            .expect("authenticate empty exported allocation");
    let input = GpuInputBuffer::Exported(export);

    assert!(
        copy_readback_mapped(&access, &input, 0, |bytes| Ok(bytes.to_vec()))
            .block_on()
            .unwrap()
            .is_empty()
    );
    assert_eq!(
        copy_readback_mapped(&other_access, &input, 0, |bytes| Ok(bytes.to_vec()))
            .block_on()
            .unwrap_err(),
        PlotReadbackError::Access(WgpuAccessError::WrongDevice)
    );
    allocation.probe.mark_unavailable();
    assert!(matches!(
        copy_readback_mapped(&access, &input, 0, |bytes| Ok(bytes.to_vec())).block_on(),
        Err(PlotReadbackError::Access(WgpuAccessError::Allocation(_)))
    ));
    assert_eq!(allocation.probe.retirements(), 0);
}

#[test]
#[ignore = "requires an actual native WGPU adapter"]
fn readback_releases_access_and_mapping_for_reuse() {
    let access = device_access();
    let expected = [1.25f32, -3.5, 0.0, 9.0];
    let buffer = access
        .with_shared(|emission| {
            let buffer = emission
                .device()
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("plot-readback-owner-test"),
                    contents: bytemuck::cast_slice(&expected),
                    usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                });
            // Initial mapped writes may be staged until the next submission.
            // Readback consumes completed producer work; polling alone does
            // not submit the producer's pending upload.
            emission.queue().submit(std::iter::empty());
            buffer
        })
        .expect("create test buffer");

    // An unpolled future does not register a mapping or hold a device permit.
    drop(readback_f32_buffer(&access, &buffer, expected.len()));
    for _ in 0..2 {
        let actual = readback_f32_buffer(&access, &buffer, expected.len())
            .block_on()
            .expect("map, progress outside the permit, and unmap");
        assert_eq!(actual, expected);
    }
    assert_eq!(
        readback_f32(&access, &buffer).block_on().unwrap(),
        expected[0]
    );
    let decode_error = readback_mapped::<()>(&access, &buffer, 16, |_| {
        Err(PlotReadbackError::Decode("decoder rejected payload"))
    })
    .block_on()
    .unwrap_err();
    assert_eq!(
        decode_error,
        PlotReadbackError::Decode("decoder rejected payload")
    );
    #[cfg(panic = "unwind")]
    {
        // Exercise the actual async readback owner, not a standalone Mapping.
        // The BufferView and read permit must unwind before its cleanup runs.
        let panic = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            readback_mapped::<()>(&access, &buffer, 16, |_| panic!("decoder panic witness"))
                .block_on()
        }));
        assert!(panic.is_err(), "decoder panic must remain observable");
    }
    // Numerical histogram evidence uses this same actual mapping/lease owner.
    // Binary64 words require no native shader-f64 feature on this device.
    let cases = [
        (
            vec![1.0f64, 1.0 + 2.0f64.powi(-30), 1.0 + 2.0f64.powi(-29)],
            vec![1.0, 1.0 + 2.0f64.powi(-30), 1.0 + 2.0f64.powi(-29)],
            vec![1.0, 2.0],
        ),
        (
            vec![2.5e-6, -1.0, 8.0, f64::NAN],
            vec![0.0, 1.0e-6, 3.0e-6, 4.0e-6],
            vec![1.0, 1.0, 2.0],
        ),
    ];
    for (samples, edges, expected) in cases {
        let allocation = TestAllocationOwner::default()
            .allocate_zeroed(
                vec![samples.len()],
                NumericElementType::F64,
                GpuTensorStorage::Real,
            )
            .unwrap();
        let bytes = samples
            .iter()
            .flat_map(|value| value.to_bits().to_le_bytes())
            .collect::<Vec<_>>();
        let source = access
            .with_shared(|emission| {
                runmat_accelerate_api::WgpuBufferOwner::new(emission.device().create_buffer_init(
                    &wgpu::util::BufferInitDescriptor {
                        label: Some("histogram-native-word-numerical-fixture"),
                        contents: &bytes,
                        usage: wgpu::BufferUsages::STORAGE,
                    },
                ))
            })
            .unwrap();
        let exported =
            WgpuBufferRef::from_owned_buffer(source, allocation.handle.clone(), access.clone())
                .unwrap();
        let exact_edges = crate::gpu::histogram::ExactHistogramEdges::new(&edges).unwrap();
        let actual = crate::gpu::histogram::histogram_counts(&access, exported, &exact_edges)
            .block_on()
            .unwrap()
            .into_double_counts();
        assert_eq!(actual, expected);
        allocation
            .handle
            .allocation_lease()
            .validate_live()
            .unwrap();
        assert_eq!(allocation.probe.retirements(), 0);
    }
    // The same actual-device fixture also covers an authenticated empty
    // export: zero numerical evidence requires no zero-sized shader binding.
    let empty_allocation = TestAllocationOwner::default()
        .allocate_zeroed(vec![0], NumericElementType::F64, GpuTensorStorage::Real)
        .unwrap();
    let empty_source = access
        .with_shared(|emission| {
            runmat_accelerate_api::WgpuBufferOwner::new(emission.device().create_buffer(
                &wgpu::BufferDescriptor {
                    label: Some("histogram-empty-word-source"),
                    size: 4,
                    usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
                    mapped_at_creation: false,
                },
            ))
        })
        .unwrap();
    let empty_export = WgpuBufferRef::from_owned_buffer(
        empty_source,
        empty_allocation.handle.clone(),
        access.clone(),
    )
    .unwrap();
    let empty_edges = crate::gpu::histogram::ExactHistogramEdges::new(&[-1.0, 0.0, 1.0]).unwrap();
    let empty_counts = crate::gpu::histogram::histogram_counts(&access, empty_export, &empty_edges)
        .block_on()
        .unwrap()
        .into_double_counts();
    assert_eq!(empty_counts, vec![0.0, 0.0]);
    empty_allocation
        .handle
        .allocation_lease()
        .validate_live()
        .unwrap();
    assert_eq!(empty_allocation.probe.retirements(), 0);
    let samples = [
        f32::NEG_INFINITY,
        -2.0,
        -1.0,
        -f32::from_bits(1),
        -0.0,
        0.0,
        f32::from_bits(1),
        f32::INFINITY,
        f32::NAN,
    ];
    let allocation = TestAllocationOwner::default()
        .allocate_zeroed(
            vec![samples.len()],
            NumericElementType::F32,
            GpuTensorStorage::Real,
        )
        .unwrap();
    let bytes = samples
        .iter()
        .flat_map(|value| value.to_bits().to_le_bytes())
        .collect::<Vec<_>>();
    let source = access
        .with_shared(|emission| {
            runmat_accelerate_api::WgpuBufferOwner::new(emission.device().create_buffer_init(
                &wgpu::util::BufferInitDescriptor {
                    label: Some("histogram-binary32-word-boundaries"),
                    contents: &bytes,
                    usage: wgpu::BufferUsages::STORAGE,
                },
            ))
        })
        .unwrap();
    let exported =
        WgpuBufferRef::from_owned_buffer(source, allocation.handle.clone(), access.clone())
            .unwrap();
    let edges = crate::gpu::histogram::ExactHistogramEdges::new(&[
        f64::NEG_INFINITY,
        -1.0,
        0.0,
        f64::from(f32::from_bits(1)),
        1.0,
        f64::INFINITY,
    ])
    .unwrap();
    assert_eq!(
        crate::gpu::histogram::histogram_counts(&access, exported, &edges)
            .block_on()
            .unwrap()
            .into_double_counts(),
        vec![2.0, 2.0, 2.0, 1.0, 2.0]
    );
    allocation
        .handle
        .allocation_lease()
        .validate_live()
        .unwrap();

    // One real bin above the binary32 consecutive-integer limit. A fresh WGPU
    // storage buffer is zero-initialized; no fabricated count receipt is used.
    let sample_count = (1usize << 24) + 1;
    let allocation = TestAllocationOwner::default()
        .allocate_zeroed(
            vec![sample_count],
            NumericElementType::F32,
            GpuTensorStorage::Real,
        )
        .unwrap();
    let source = access
        .with_shared(|emission| {
            runmat_accelerate_api::WgpuBufferOwner::new(emission.device().create_buffer(
                &wgpu::BufferDescriptor {
                    label: Some("histogram-u32-count-above-f32-integer-limit"),
                    size: (sample_count as u64) * 4,
                    usage: wgpu::BufferUsages::STORAGE,
                    mapped_at_creation: false,
                },
            ))
        })
        .unwrap();
    let exported =
        WgpuBufferRef::from_owned_buffer(source, allocation.handle.clone(), access.clone())
            .unwrap();
    let edges = crate::gpu::histogram::ExactHistogramEdges::new(&[-1.0, 1.0]).unwrap();
    assert_eq!(
        crate::gpu::histogram::histogram_counts(&access, exported, &edges)
            .block_on()
            .unwrap()
            .into_double_counts(),
        vec![sample_count as f64]
    );
    allocation
        .handle
        .allocation_lease()
        .validate_live()
        .unwrap();

    // Decoder error/panic must end the mapped view and unmap before remapping.
    assert_eq!(
        readback_f32_buffer(&access, &buffer, expected.len())
            .block_on()
            .unwrap(),
        expected
    );
    assert!(readback_f32_buffer(&access, &buffer, expected.len() + 1)
        .block_on()
        .is_err());
    assert_eq!(
        readback_f32_buffer(&access, &buffer, expected.len())
            .block_on()
            .expect("extent rejection leaves the mapping reusable"),
        expected
    );
}

#[test]
#[ignore = "requires an actual native WGPU adapter"]
fn pending_mapping_owner_cancellation_unmaps_outside_emission() {
    let access = device_access();
    let buffer = access
        .with_shared(|emission| {
            emission.device().create_buffer(&wgpu::BufferDescriptor {
                label: Some("plot-cancelled-mapping-owner-test"),
                size: 16,
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            })
        })
        .expect("create test buffer");
    let (tx, rx) = oneshot::channel();
    let mut mapping = Mapping {
        access: &access,
        buffer: &buffer,
        registered: false,
    };
    access
        .with_shared(|_| {
            mapping.registered = true;
            buffer
                .slice(..)
                .map_async(wgpu::MapMode::Read, move |result| {
                    let _ = tx.send(result);
                });
        })
        .expect("register pending mapping");
    // This is the same owner used by readback_bytes on cancellation. No
    // emission is active when unmap may synchronously invoke the callback.
    drop(mapping);
    access.poll(wgpu::Maintain::Wait);
    let _completed_mapping_result = rx.block_on().expect("map callback delivered once");
    assert_eq!(
        readback_f32_buffer(&access, &buffer, 4)
            .block_on()
            .expect("cancelled mapping can be remapped"),
        [0.0; 4]
    );
}

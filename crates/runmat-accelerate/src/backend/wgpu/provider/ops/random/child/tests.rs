//! Required-device numerical checks; shader validation and in-process tests
//! cannot establish that the selected physical pipelines execute correctly.
use super::*;
use futures::executor::block_on;
use runmat_accelerate_api::random::GpuRandomReservation;
use runmat_accelerate_api::{AccelProvider, HostNumericDataOwned};

fn check_child_pipelines(precision: ProviderPrecision) {
    let provider = WgpuProvider::new(Default::default()).expect("actual WGPU adapter required");
    assert_eq!(
        provider.precision(),
        precision,
        "set RUNMAT_WGPU_FORCE_PRECISION"
    );
    for distribution in [
        GpuRandomDistribution::Uniform,
        GpuRandomDistribution::StandardNormal,
    ] {
        for storage in [GpuTensorStorage::Real, GpuTensorStorage::ComplexInterleaved] {
            for empty in [false, true] {
                let shape = match (storage, empty) {
                    (_, true) => [0, 3],
                    (GpuTensorStorage::Real, false) => [3, 44],
                    (GpuTensorStorage::ComplexInterleaved, false) => [3, 22],
                };
                let request = GpuRandomRequest {
                    shape: &shape,
                    precision,
                    storage,
                    distribution,
                    reservation: GpuRandomReservation {
                        algorithm: GpuRandomAlgorithm::Threefry4x64_20,
                        seed: 17,
                        // Exercise carry into the high half of the counter.
                        first: (1_u128 << 64) - 6,
                        count: if empty { 0 } else { 132 },
                        normal_transform: GpuNormalTransform::BoxMuller,
                    },
                };
                let mut previous = None;
                for _ in 0..2 {
                    let handle = provider
                        .execute_gpu_random(&request, None)
                        .expect("device generation");
                    assert_eq!(handle.shape(), shape);
                    assert_eq!(handle.physical().storage(), storage);
                    let output =
                        block_on(provider.download_numeric(&handle)).expect("device readback");
                    assert_eq!(output.shape, shape);
                    assert_eq!(output.storage, storage);
                    let actual: Vec<f64> = match (precision, output.data) {
                        (ProviderPrecision::F32, HostNumericDataOwned::F32(values)) => {
                            values.into_iter().map(f64::from).collect()
                        }
                        (ProviderPrecision::F64, HostNumericDataOwned::F64(values)) => values,
                        (_, other) => panic!("incorrect native output class: {other:?}"),
                    };
                    assert_eq!(actual.len(), request.reservation.count as usize);
                    for (index, &value) in actual.iter().enumerate() {
                        let expected = match (precision, distribution) {
                            (ProviderPrecision::F32, GpuRandomDistribution::Uniform) => {
                                f64::from(request.reservation.uniform_f32(index as u64).unwrap())
                            }
                            (ProviderPrecision::F64, GpuRandomDistribution::Uniform) => {
                                request.reservation.uniform_f64(index as u64).unwrap()
                            }
                            (ProviderPrecision::F32, GpuRandomDistribution::StandardNormal) => {
                                f64::from(request.normal_component_f32(index as u64).unwrap())
                            }
                            (ProviderPrecision::F64, GpuRandomDistribution::StandardNormal) => {
                                request.normal_component_f64(index as u64).unwrap()
                            }
                        };
                        assert!(value.is_finite());
                        if distribution == GpuRandomDistribution::Uniform {
                            assert_eq!(value, expected, "exact Threefry word mapping at {index}");
                        } else {
                            let tolerance = match precision {
                                ProviderPrecision::F32 => 2e-5,
                                ProviderPrecision::F64 => 2e-12,
                            };
                            assert!(
                                (value - expected).abs() <= tolerance * (1.0 + expected.abs()),
                                "normal component {index}: {value} != {expected}"
                            );
                        }
                    }
                    if let Some(previous) = previous.as_ref() {
                        assert_eq!(
                            &actual, previous,
                            "same child range must replay exactly on the same device"
                        );
                    }
                    previous = Some(actual);
                }
            }
        }
    }
    let malformed = GpuRandomRequest {
        shape: &[3, 4],
        precision,
        storage: GpuTensorStorage::Real,
        distribution: GpuRandomDistribution::Uniform,
        reservation: GpuRandomReservation {
            algorithm: GpuRandomAlgorithm::Threefry4x64_20,
            seed: 17,
            first: 0,
            count: 11,
            normal_transform: GpuNormalTransform::BoxMuller,
        },
    };
    let retained = provider.allocations.retained_bytes();
    assert!(provider.propose_gpu_random(&malformed).unwrap().is_none());
    assert!(provider.execute_gpu_random(&malformed, None).is_err());
    assert_eq!(provider.allocations.retained_bytes(), retained);
}

#[test]
#[ignore = "requires actual WGPU adapter and forced F32 precision"]
fn actual_child_random_f32_executes_and_replays() {
    check_child_pipelines(ProviderPrecision::F32);
}

#[test]
#[ignore = "requires actual WGPU adapter and forced F64 precision"]
fn actual_child_random_f64_executes_and_replays() {
    check_child_pipelines(ProviderPrecision::F64);
}

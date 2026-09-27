use super::*;
use crate::test_support::{RetirementProbe, TestAllocationOwner};
use std::sync::Mutex;

#[derive(Clone, Copy)]
enum FillMode {
    Unsupported,
    ZeroFailure,
    AddFailure,
}

struct Provider {
    owner: TestAllocationOwner,
    mode: FillMode,
    uploads: Mutex<usize>,
    temporaries: Mutex<Vec<RetirementProbe>>,
}

struct Defaults;

fn ready_default<F: std::future::Future>(future: F) -> F::Output {
    let mut future = std::pin::pin!(future);
    let mut context = std::task::Context::from_waker(std::task::Waker::noop());
    match future.as_mut().poll(&mut context) {
        std::task::Poll::Ready(output) => output,
        std::task::Poll::Pending => {
            panic!("an unsupported default must resolve without device work")
        }
    }
}

#[test]
fn polynomial_defaults_are_typed_unsupported() {
    let input = Provider::new(FillMode::Unsupported).allocation(&[1, 2]);
    let options = ProviderPolyvalOptions { mu: None };
    assert!(is_unsupported_provider_operation(
        &Defaults.polyval(&input, &input, &options).unwrap_err()
    ));
    assert!(is_unsupported_provider_operation(
        &ready_default(Defaults.polyfit(&input, &input, 1, None)).unwrap_err()
    ));
    assert!(is_unsupported_provider_operation(
        &ready_default(Defaults.polyder_single(&input)).unwrap_err()
    ));
    assert!(is_unsupported_provider_operation(
        &ready_default(Defaults.polyder_product(&input, &input)).unwrap_err()
    ));
    assert!(is_unsupported_provider_operation(
        &ready_default(Defaults.polyder_quotient(&input, &input)).unwrap_err()
    ));
    assert!(is_unsupported_provider_operation(
        &Defaults.polyint(&input, 0.0).unwrap_err()
    ));
}

#[test]
fn structure_defaults_are_typed_unsupported() {
    let input = Provider::new(FillMode::Unsupported).allocation(&[2, 2]);
    assert!(is_unsupported_provider_operation(
        &Defaults
            .issymmetric(&input, ProviderSymmetryKind::Symmetric, 0.0)
            .unwrap_err()
    ));
    assert!(is_unsupported_provider_operation(
        &ready_default(Defaults.ishermitian(&input, ProviderHermitianKind::Hermitian, 0.0))
            .unwrap_err()
    ));
    assert!(is_unsupported_provider_operation(
        &Defaults.bandwidth(&input).unwrap_err()
    ));
    assert!(is_unsupported_provider_operation(
        &ready_default(Defaults.sym_rcm(&input)).unwrap_err()
    ));
}
impl AccelProvider for Defaults {
    fn device_info(&self) -> String {
        "defaults only".into()
    }
    fn upload(&self, _: &HostTensorView) -> anyhow::Result<GpuTensorHandle> {
        Err(unsupported_provider_operation("upload unavailable"))
    }
    fn download<'a>(&'a self, _: &'a GpuTensorHandle) -> AccelDownloadFuture<'a> {
        Box::pin(async { Err(unsupported_provider_operation("download unavailable")) })
    }
}

impl Provider {
    fn new(mode: FillMode) -> Self {
        Self {
            owner: TestAllocationOwner::new(),
            mode,
            uploads: Mutex::new(0),
            temporaries: Mutex::new(Vec::new()),
        }
    }
    fn allocation(&self, shape: &[usize]) -> GpuTensorHandle {
        self.owner
            .allocate_zeroed(
                shape.to_vec(),
                NumericElementType::F64,
                GpuTensorStorage::Real,
            )
            .unwrap()
            .handle
    }
}

impl AccelProvider for Provider {
    fn device_id(&self) -> u32 {
        self.owner.device_id()
    }
    fn device_info(&self) -> String {
        "optional default witness".into()
    }
    fn upload(&self, host: &HostTensorView) -> anyhow::Result<GpuTensorHandle> {
        *self.uploads.lock().unwrap() += 1;
        Ok(self.allocation(host.shape))
    }
    fn download<'a>(&'a self, _: &'a GpuTensorHandle) -> AccelDownloadFuture<'a> {
        Box::pin(async { Err(unsupported_provider_operation("unused download")) })
    }
    fn zeros(&self, shape: &[usize]) -> anyhow::Result<GpuTensorHandle> {
        match self.mode {
            FillMode::Unsupported => Err(unsupported_provider_operation("zeros unavailable")),
            FillMode::ZeroFailure => {
                Err(std::io::Error::new(std::io::ErrorKind::BrokenPipe, "zeros failed").into())
            }
            FillMode::AddFailure => {
                let allocation = self.owner.allocate_zeroed(
                    shape.to_vec(),
                    NumericElementType::F64,
                    GpuTensorStorage::Real,
                )?;
                self.temporaries.lock().unwrap().push(allocation.probe);
                Ok(allocation.handle)
            }
        }
    }
    fn scalar_add(&self, _: &GpuTensorHandle, _: f64) -> anyhow::Result<GpuTensorHandle> {
        Err(std::io::Error::new(std::io::ErrorKind::BrokenPipe, "scalar add failed").into())
    }
}

#[test]
fn optional_reduction_and_calculus_defaults_are_typed_unsupported() {
    let provider = Provider::new(FillMode::Unsupported);
    let input = provider.allocation(&[2, 3]);
    let results = [
        Defaults.zeros(&[2, 3]),
        Defaults.scalar_add(&input, 1.0),
        Defaults.fill(&[2, 3], 0.0),
        Defaults.fill(&[2, 3], 1.0),
        Defaults.fill_like(&input, 0.0),
        provider.gradient_dim(&input, 0, 1.0),
        provider.gradient_dim_with_coordinates(&input, 0, &input),
        provider.cumtrapz_dim(&input, 0, ProviderTrapezoidSpacing::Scalar(1.0)),
        provider.map_nan_to_zero(&input),
        provider.not_nan_mask(&input),
        provider.permute(&input, &[1, 0]),
        Defaults.gather_linear(&input, &[0], &[1, 1]),
    ];
    assert!(is_unsupported_provider_operation(
        &Defaults.scatter_linear(&input, &[0], &input).unwrap_err()
    ));
    for result in results {
        assert!(is_unsupported_provider_operation(&result.unwrap_err()));
    }
    assert_eq!(*provider.uploads.lock().unwrap(), 0);
    input.allocation_lease().validate_live().unwrap();
}

#[test]
fn reshape_preserves_alias_and_does_not_misclassify_invalid_shape() {
    let provider = Provider::new(FillMode::Unsupported);
    let input = provider.allocation(&[2, 3]);
    let reshaped = provider.reshape(&input, &[3, 2]).unwrap();
    assert_eq!(reshaped.shape(), &[3, 2]);
    assert!(reshaped.shares_allocation_with(&input));
    assert!(!is_unsupported_provider_operation(
        &provider.reshape(&input, &[7, 1]).unwrap_err()
    ));
    assert_eq!(input.shape(), &[2, 3]);
}

#[test]
fn fill_defaults_fallback_only_for_typed_unsupported() {
    for like in [false, true] {
        for mode in [
            FillMode::Unsupported,
            FillMode::ZeroFailure,
            FillMode::AddFailure,
        ] {
            let provider = Provider::new(mode);
            let input = provider.allocation(&[2, 3]);
            if like {
                // Generic arithmetic cannot establish prototype-class preservation.
                assert!(is_unsupported_provider_operation(
                    &provider.fill_like(&input, 2.0).unwrap_err()
                ));
                assert_eq!(*provider.uploads.lock().unwrap(), 0);
                assert!(provider.temporaries.lock().unwrap().is_empty());
                input.allocation_lease().validate_live().unwrap();
                continue;
            }
            let result = provider.fill(&[2, 3], 2.0);
            if matches!(mode, FillMode::Unsupported) {
                assert_eq!(result.unwrap().shape(), &[2, 3]);
                assert_eq!(*provider.uploads.lock().unwrap(), 1);
            } else {
                let error = result.unwrap_err();
                assert!(!is_unsupported_provider_operation(&error));
                assert_eq!(
                    error.downcast_ref::<std::io::Error>().unwrap().kind(),
                    std::io::ErrorKind::BrokenPipe
                );
                assert_eq!(*provider.uploads.lock().unwrap(), 0);
            }
            for temporary in provider.temporaries.lock().unwrap().iter() {
                assert_eq!(temporary.retirements(), 1);
            }
            input.allocation_lease().validate_live().unwrap();
        }
    }
}

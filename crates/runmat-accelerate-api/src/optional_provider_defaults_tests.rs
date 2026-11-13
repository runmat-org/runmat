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

#[test]
fn transpose_default_is_typed_unsupported_and_preserves_input_ownership() {
    let owner = TestAllocationOwner::new();
    let input = owner
        .allocate_zeroed(vec![2, 3], NumericElementType::F32, GpuTensorStorage::Real)
        .unwrap();
    let error = Defaults.transpose(&input.handle).unwrap_err();
    assert!(ready_default(Defaults.unary_conj(&input.handle))
        .unwrap()
        .is_none());
    #[cfg(feature = "wgpu")]
    assert!(Defaults
        .export_wgpu_buffer(&input.handle)
        .unwrap()
        .is_none());
    assert!(error
        .downcast_ref::<UnsupportedProviderOperation>()
        .is_some());
    assert!(is_unsupported_provider_operation(&error));
    assert!(!is_unsupported_provider_operation(&anyhow::anyhow!(
        "transpose not supported by provider"
    )));
    input.handle.allocation_lease().validate_live().unwrap();
    assert_eq!(input.probe.retirements(), 0);
    drop(input.handle);
    assert_eq!(input.probe.retirements(), 1);
}

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
fn extrema_composite_defaults_are_typed_unsupported_without_device_work() {
    let owner = TestAllocationOwner::new();
    let input = owner
        .allocate_zeroed(vec![1, 2], NumericElementType::F64, GpuTensorStorage::Real)
        .unwrap();
    for result in [
        Defaults.scalar_min(&input.handle, 1.0),
        Defaults.scalar_max(&input.handle, 1.0),
        Defaults.scalar_mul(&input.handle, -1.0),
        Defaults.scalar_add(&input.handle, 2.0),
    ] {
        assert!(is_unsupported_provider_operation(&result.unwrap_err()));
        input.handle.allocation_lease().validate_live().unwrap();
        assert_eq!(input.probe.retirements(), 0);
    }
    assert!(Defaults.repmat(&input.handle, &[2, 1]).unwrap().is_none());
    for result in [
        ready_default(Defaults.elem_add(&input.handle, &input.handle)),
        ready_default(Defaults.elem_sub(&input.handle, &input.handle)),
        ready_default(Defaults.elem_mul(&input.handle, &input.handle)),
        ready_default(Defaults.elem_div(&input.handle, &input.handle)),
        ready_default(Defaults.elem_pow(&input.handle, &input.handle)),
        ready_default(Defaults.elem_rem(&input.handle, &input.handle)),
        ready_default(Defaults.elem_mod(&input.handle, &input.handle)),
        ready_default(Defaults.elem_min(&input.handle, &input.handle)),
        ready_default(Defaults.elem_max(&input.handle, &input.handle)),
        ready_default(Defaults.elem_le(&input.handle, &input.handle)),
        ready_default(Defaults.elem_ge(&input.handle, &input.handle)),
    ] {
        assert!(result.unwrap_err().is_unsupported());
        input.handle.allocation_lease().validate_live().unwrap();
        assert_eq!(input.probe.retirements(), 0);
    }
    assert!(!is_unsupported_provider_operation(&anyhow::anyhow!(
        "repmat allocation failed"
    )));
    drop(input.handle);
    assert_eq!(input.probe.retirements(), 1);
}

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

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
fn four_extrema_defaults_are_pre_effect_unsupported() {
    let owner = TestAllocationOwner::new();
    let input = owner
        .allocate_zeroed(vec![2, 1], NumericElementType::F64, GpuTensorStorage::Real)
        .unwrap();
    let provider = Defaults;
    let failures = [
        ready_default(provider.reduce_min(&input.handle)).unwrap_err(),
        ready_default(provider.reduce_max(&input.handle)).unwrap_err(),
        ready_default(provider.reduce_min_dim(&input.handle, 0)).unwrap_err(),
        ready_default(provider.reduce_max_dim(&input.handle, 0)).unwrap_err(),
    ];
    for failure in failures {
        assert!(failure.is_unsupported());
        assert!(!failure.is_after_effect());
        input.handle.allocation_lease().validate_live().unwrap();
        assert_eq!(input.probe.retirements(), 0);
    }
}

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
fn structural_defaults_are_pre_effect_unsupported_and_preserve_the_input() {
    let owner = TestAllocationOwner::new();
    let input = owner
        .allocate_zeroed(vec![2, 2], NumericElementType::F64, GpuTensorStorage::Real)
        .unwrap();
    let provider = Defaults;
    for failure in [
        ready_default(provider.tril(&input.handle, -1)).unwrap_err(),
        ready_default(provider.triu(&input.handle, 1)).unwrap_err(),
        ready_default(provider.kron(&input.handle, &input.handle)).unwrap_err(),
        ready_default(provider.cat_optional(1, &[input.handle.clone(), input.handle.clone()]))
            .unwrap_err(),
    ] {
        assert!(failure.is_unsupported());
        assert!(!failure.is_after_effect());
        input.handle.allocation_lease().validate_live().unwrap();
        assert_eq!(input.probe.retirements(), 0);
    }
}

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
fn std_defaults_are_pre_effect_unsupported_without_device_work() {
    let owner = TestAllocationOwner::new();
    let input = owner
        .allocate_zeroed(vec![2, 1], NumericElementType::F64, GpuTensorStorage::Real)
        .unwrap();
    let provider = Defaults;
    for failure in [
        ready_default(provider.reduce_std(
            &input.handle,
            ProviderStdNormalization::Sample,
            ProviderNanMode::Include,
        ))
        .unwrap_err(),
        ready_default(provider.reduce_std_dim(
            &input.handle,
            0,
            ProviderStdNormalization::Population,
            ProviderNanMode::Omit,
        ))
        .unwrap_err(),
    ] {
        assert!(failure.is_unsupported());
        assert!(!failure.is_after_effect());
        input.handle.allocation_lease().validate_live().unwrap();
        assert_eq!(input.probe.retirements(), 0);
    }
}

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
fn fused_defaults_are_lazy_typed_unsupported_and_preserve_borrowed_input() {
    let owner = TestAllocationOwner::new();
    let allocation = owner
        .allocate_zeroed(vec![1, 1], NumericElementType::F64, GpuTensorStorage::Real)
        .unwrap();
    let provider = Defaults;
    let inputs = [allocation.handle.clone()];
    let shape = [1, 1];
    let classes = [ProviderElementType::F64];
    let single = provider.fused_elementwise(
        "",
        ProviderShaderRequirements::Basic,
        &inputs,
        &shape,
        1,
        ProviderElementType::F64,
    );
    let multi = provider.fused_elementwise_multi(
        "",
        ProviderShaderRequirements::Basic,
        &inputs,
        &shape,
        1,
        &classes,
    );
    allocation
        .handle
        .allocation_lease()
        .validate_live()
        .unwrap();
    assert_eq!(allocation.probe.retirements(), 0);
    drop(single);
    assert!(ready_default(multi).unwrap_err().is_unsupported());
    allocation
        .handle
        .allocation_lease()
        .validate_live()
        .unwrap();
    assert_eq!(allocation.probe.retirements(), 0);
    assert!(ready_default(provider.fused_elementwise(
        "",
        ProviderShaderRequirements::Basic,
        &inputs,
        &shape,
        1,
        ProviderElementType::F64,
    ))
    .unwrap_err()
    .is_unsupported());
}

#[test]
fn explicit_fused_data_abi_defaults_are_lazy_unsupported() {
    let owner = TestAllocationOwner::new();
    let allocated = owner
        .allocate_zeroed(vec![1, 2], NumericElementType::F32, GpuTensorStorage::Real)
        .unwrap();
    let inputs = [allocated.handle.clone()];
    let abi = ProviderFusedElementwiseAbi::BroadcastReal {
        precision: ProviderPrecision::F32,
    };
    let request = || ProviderFusedElementwiseRequest {
        shader: "",
        requirements: ProviderShaderRequirements::Basic,
        inputs: &inputs,
        output_shape: &[1, 2],
        len: 2,
        output_classes: &[ProviderElementType::F32],
        abi,
    };
    let provider = Defaults;
    assert!(!provider.supports_fused_elementwise_abi(abi));
    drop(provider.fused_elementwise_typed(request()));
    assert!(ready_default(provider.fused_elementwise_typed(request()))
        .unwrap_err()
        .is_unsupported());
    assert!(
        ready_default(provider.fused_elementwise_multi_typed(request()))
            .unwrap_err()
            .is_unsupported()
    );
    allocated.handle.allocation_lease().validate_live().unwrap();
    assert_eq!(allocated.probe.retirements(), 0);
}

#[test]
fn polynomial_defaults_are_typed_unsupported() {
    let input = Provider::new(FillMode::Unsupported).allocation(&[1, 2]);
    let options = ProviderPolyvalOptions { mu: None };
    assert!(is_unsupported_provider_operation(
        &Defaults.polyval(&input, &input, &options).unwrap_err()
    ));
    assert!(is_unsupported_provider_operation(
        &ready_default(Defaults.polyfit(
            &input,
            &input,
            1,
            None,
            ProviderPolyfitOutputs::Statistics
        ))
        .unwrap_err()
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
        provider.gradient_dim_with_coordinates(&input, 0, &input),
        provider.not_nan_mask(&input),
        Defaults.gather_linear(&input, &[0], &[1, 1]),
    ];
    for result in [
        futures::executor::block_on(provider.reduce_nnz(&input)),
        futures::executor::block_on(provider.reduce_nnz_dim(&input, 0)),
        futures::executor::block_on(provider.reduce_prod(&input)),
        futures::executor::block_on(provider.reduce_prod_dim(&input, 0)),
        futures::executor::block_on(provider.reduce_integer_prod_native(&input)),
        futures::executor::block_on(provider.reduce_integer_prod_native_dim(&input, 0)),
    ] {
        assert!(result.unwrap().is_none());
    }
    for result in [
        Defaults.fill(&[2, 3], 0.0),
        Defaults.fill(&[2, 3], 1.0),
        Defaults.fill_like(&input, 0.0),
        Defaults.ones_like(&input),
        Defaults.zeros_like(&input, &[2, 3]),
        Defaults.logical_fill(&[2, 3], true),
        Defaults.ones(&[2, 3]),
    ] {
        assert!(result.unwrap().is_none());
    }
    for result in [
        provider.gradient_dim(&input, 0, 1.0),
        provider.diff_dim(&input, 1, 0),
        provider.trapz_dim(&input, 0, ProviderTrapezoidSpacing::Unit),
        provider.cumtrapz_dim(&input, 0, ProviderTrapezoidSpacing::Scalar(1.0)),
    ] {
        assert!(result.unwrap().is_none());
    }
    assert!(provider.permute(&input, &[1, 0]).unwrap().is_none());
    assert!(provider
        .map_nan_to_zero(&input)
        .unwrap_err()
        .is_unsupported());
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
fn fill_defaults_never_probe_raw_primitives_or_upload() {
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
                assert!(provider.fill_like(&input, 2.0).unwrap().is_none());
                assert_eq!(*provider.uploads.lock().unwrap(), 0);
                assert!(provider.temporaries.lock().unwrap().is_empty());
                input.allocation_lease().validate_live().unwrap();
                continue;
            }
            let result = provider.fill(&[2, 3], 2.0);
            assert!(result.unwrap().is_none());
            assert_eq!(*provider.uploads.lock().unwrap(), 0);
            assert!(provider.temporaries.lock().unwrap().is_empty());
            for temporary in provider.temporaries.lock().unwrap().iter() {
                assert_eq!(temporary.retirements(), 1);
            }
            input.allocation_lease().validate_live().unwrap();
        }
    }
}

use super::*;

struct DownloadFailProvider {
    owner: runmat_accelerate_api::test_support::TestAllocationOwner,
    probes: std::sync::Mutex<Vec<runmat_accelerate_api::test_support::RetirementProbe>>,
}
impl DownloadFailProvider {
    fn new() -> Self {
        Self {
            owner: Default::default(),
            probes: Default::default(),
        }
    }
    fn free_count(&self) -> usize {
        self.probes
            .lock()
            .unwrap()
            .iter()
            .map(|probe| probe.retirements())
            .sum()
    }
}
impl AccelProvider for DownloadFailProvider {
    fn upload(&self, host: &HostTensorView) -> anyhow::Result<GpuTensorHandle> {
        let allocation =
            self.owner
                .allocate_numeric(runmat_accelerate_api::HostNumericTensorOwned {
                    data: runmat_accelerate_api::HostNumericDataOwned::F64(host.data.to_vec()),
                    shape: host.shape.to_vec(),
                    storage: runmat_accelerate_api::GpuTensorStorage::Real,
                })?;
        self.probes.lock().unwrap().push(allocation.probe);
        Ok(allocation.handle)
    }
    fn download<'a>(&'a self, _h: &'a GpuTensorHandle) -> AccelDownloadFuture<'a> {
        Box::pin(async { Err(anyhow::anyhow!("forced download failure")) })
    }
    fn device_info(&self) -> String {
        "download-fail-provider".to_string()
    }
    fn device_id(&self) -> u32 {
        self.owner.device_id()
    }
}

#[test]
fn final_provider_output_rejects_borrowed_alias_and_wrong_shape() {
    let _lock = test_support::accel_test_lock();
    let provider: &'static runmat_accelerate::simple_provider::InProcessProvider = Box::leak(
        Box::new(runmat_accelerate::simple_provider::InProcessProvider::new()),
    );
    let _scope = runmat_accelerate_api::ThreadProviderGuard::set(Some(provider));
    let signal = provider
        .upload(&HostTensorView {
            data: &[1.0, 2.0],
            shape: &[1, 2],
        })
        .unwrap();
    let borrowed = provider
        .upload(&HostTensorView {
            data: &[3.0, 4.0],
            shape: &[1, 2],
        })
        .unwrap();
    let protected = [&borrowed];
    let context = OutputContext {
        input: &signal,
        provider,
        caller_inputs: &protected,
    };
    let contract = resident_output::floating_contract(
        &signal,
        vec![1, 2],
        runmat_accelerate_api::GpuTensorStorage::Real,
    );
    let alias_error = context
        .admit(borrowed.clone(), &contract, &[], || {
            filtfilt_error(&FILTFILT_ERROR_PROVIDER_OUTPUT)
        })
        .expect_err("a borrowed coefficient may not be published as the result");
    assert_eq!(
        alias_error.identifier(),
        FILTFILT_ERROR_PROVIDER_OUTPUT.identifier
    );
    assert!(borrowed.allocation_lease().validate_live().is_ok());
    let wrong_shape = provider
        .upload(&HostTensorView {
            data: &[5.0, 6.0],
            shape: &[2, 1],
        })
        .unwrap();
    let shape_error = context
        .admit(wrong_shape, &contract, &[], || {
            filtfilt_error(&FILTFILT_ERROR_PROVIDER_OUTPUT)
        })
        .expect_err("same-count transposed result must be rejected");
    assert_eq!(
        shape_error.identifier(),
        FILTFILT_ERROR_PROVIDER_OUTPUT.identifier
    );
    assert!(signal.allocation_lease().validate_live().is_ok());
}

#[test]
fn gpu_filter_pass_retires_all_temporaries_when_hook_is_unsupported() {
    let _guard = test_support::accel_test_lock();
    let provider = Box::leak(Box::new(DownloadFailProvider::new()));
    let _provider_guard = runmat_accelerate_api::ThreadProviderGuard::set(Some(provider));

    let b = vec![Complex::new(0.5, 0.0), Complex::new(0.5, 0.0)];
    let a = vec![Complex::new(1.0, 0.0)];
    let coeffs = NormalizedCoefficients::new(&b, &a).expect("coefficients");
    let signal = (1..=8)
        .map(|value| Complex::new(value as f64, 0.0))
        .collect::<Vec<_>>();
    let zi = vec![Complex::new(0.0, 0.0)];
    let prepared = ProviderPreparedIirCoefficients::upload(
        provider,
        &[0.5, 0.5],
        &[1.0, 0.0],
        provider.precision(),
        &[],
    )
    .expect("prepared coefficients");
    let result = block_on(gpu_filter_pass(
        provider,
        &prepared,
        &coeffs,
        &signal,
        &[1, 8],
        &zi,
        &[],
    ));

    assert!(result.is_err());
    drop(prepared);
    assert_eq!(provider.free_count(), 4);
}

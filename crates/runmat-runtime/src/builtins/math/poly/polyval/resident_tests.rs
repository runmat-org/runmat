use super::*;
use runmat_accelerate_api::{
    test_support::{RetirementProbe, TestAllocationOwner},
    AccelDownloadFuture, HostNumericDataOwned, HostNumericDataView, HostNumericTensorOwned,
    HostNumericTensorView, HostTensorView, NumericElementType,
};
use std::sync::Mutex;

#[derive(Clone, Copy)]
enum Mode {
    Success,
    Failure,
    Borrowed,
    Unsupported,
}
struct Provider {
    owner: TestAllocationOwner,
    mode: Mode,
    probes: Mutex<Vec<RetirementProbe>>,
}
impl AccelProvider for Provider {
    fn device_id(&self) -> u32 {
        self.owner.device_id()
    }
    fn device_info(&self) -> String {
        "owned polynomial fixture".into()
    }
    fn upload(&self, _: &HostTensorView) -> anyhow::Result<GpuTensorHandle> {
        panic!("typed upload required")
    }
    fn upload_numeric_raw(
        &self,
        host: &HostNumericTensorView,
    ) -> runmat_accelerate_api::ProviderExecutionResult<GpuTensorHandle> {
        let HostNumericDataView::F64(data) = host.data else {
            return Err(anyhow::anyhow!("F64 expected").into());
        };
        let allocation = self.owner.allocate_numeric(HostNumericTensorOwned {
            data: HostNumericDataOwned::F64(data.to_vec()),
            shape: host.shape.to_vec(),
            storage: host.storage,
        })?;
        self.probes.lock().unwrap().push(allocation.probe);
        Ok(allocation.handle)
    }
    fn download<'a>(&'a self, _: &'a GpuTensorHandle) -> AccelDownloadFuture<'a> {
        panic!("prepared polynomial route must not download")
    }
    fn polyval(
        &self,
        _: &GpuTensorHandle,
        points: &GpuTensorHandle,
        _: &ProviderPolyvalOptions,
    ) -> anyhow::Result<GpuTensorHandle> {
        match self.mode {
            Mode::Failure => anyhow::bail!("injected polynomial failure"),
            Mode::Borrowed => Ok(points.clone()),
            Mode::Unsupported => Err(runmat_accelerate_api::unsupported_provider_operation(
                "fixture",
            )),
            Mode::Success => {
                let allocation = self.owner.allocate_zeroed(
                    points.shape().to_vec(),
                    NumericElementType::F64,
                    GpuTensorStorage::Real,
                )?;
                self.probes.lock().unwrap().push(allocation.probe);
                Ok(allocation.handle)
            }
        }
    }
}

#[test]
fn missing_statistics_rejects_before_resident_readback() {
    let _lock = crate::builtins::common::test_support::accel_test_lock();
    let provider = Box::leak(Box::new(Provider {
        owner: TestAllocationOwner::new(),
        mode: Mode::Success,
        probes: Mutex::new(Vec::new()),
    }));
    let _scope = runmat_accelerate_api::ThreadProviderGuard::set(Some(provider));
    let source = provider
        .owner
        .allocate_zeroed(vec![1, 2], NumericElementType::F64, GpuTensorStorage::Real)
        .expect("resident points");
    let error = futures::executor::block_on(super::super::evaluate(
        Value::Num(1.0),
        Value::GpuTensor(source.handle.clone()),
        &[],
        true,
    ))
    .expect_err("missing S must fail before download");
    assert!(error.message().contains("S input"));
    assert!(provider.probes.lock().unwrap().is_empty());
    assert_eq!(source.probe.retirements(), 0);
}

#[test]
fn malformed_host_controls_reject_before_resident_readback() {
    let _lock = crate::builtins::common::test_support::accel_test_lock();
    let provider = Box::leak(Box::new(Provider {
        owner: TestAllocationOwner::new(),
        mode: Mode::Success,
        probes: Mutex::new(Vec::new()),
    }));
    let _scope = runmat_accelerate_api::ThreadProviderGuard::set(Some(provider));
    let source = provider
        .owner
        .allocate_zeroed(vec![1, 2], NumericElementType::F64, GpuTensorStorage::Real)
        .expect("resident points");
    for (options, wants_delta, expected) in [
        (vec![Value::Num(1.0)], true, "S input must be"),
        (
            vec![Value::GpuTensor(source.handle.clone())],
            true,
            "S input must be",
        ),
        (
            vec![
                Value::Tensor(Tensor::new(Vec::new(), vec![0, 0]).unwrap()),
                Value::Num(1.0),
            ],
            false,
            "mu must be a numeric vector",
        ),
    ] {
        let error = futures::executor::block_on(super::super::evaluate(
            Value::Num(1.0),
            Value::GpuTensor(source.handle.clone()),
            &options,
            wants_delta,
        ))
        .expect_err("known-invalid host control");
        assert!(error.message().contains(expected), "{error:?}");
    }
    assert!(provider.probes.lock().unwrap().is_empty());
    assert_eq!(source.probe.retirements(), 0);
}

#[test]
fn preparation_failure_and_alias_rejection_retire_only_owned_temporaries() {
    let _lock = crate::builtins::common::test_support::accel_test_lock();
    for mode in [
        Mode::Failure,
        Mode::Borrowed,
        Mode::Unsupported,
        Mode::Success,
    ] {
        let provider = Box::leak(Box::new(Provider {
            owner: TestAllocationOwner::new(),
            mode,
            probes: Mutex::new(Vec::new()),
        }));
        let _scope = runmat_accelerate_api::ThreadProviderGuard::set(Some(provider));
        let original = provider
            .owner
            .allocate_zeroed(vec![1, 2], NumericElementType::F64, GpuTensorStorage::Real)
            .unwrap();
        let request = Request::new(
            &Value::GpuTensor(original.handle.clone()),
            &Value::Num(1.0),
            crate::call::FunctionArgumentView::from_slice(&[]),
        )
        .unwrap();
        let output = request.try_evaluate(
            &[Complex64::new(1.0, 0.0)],
            true,
            &NumericArray {
                data: vec![Complex64::new(2.0, 0.0), Complex64::new(3.0, 0.0)],
                shape: vec![1, 2],
                all_real: true,
            },
            None,
        );
        match mode {
            Mode::Failure => assert!(output
                .unwrap_err()
                .message()
                .contains("injected polynomial failure")),
            Mode::Borrowed => assert!(output
                .unwrap_err()
                .message()
                .contains("invalid provider output")),
            Mode::Unsupported => assert!(output.unwrap().is_none()),
            Mode::Success => {
                let output = output.unwrap().unwrap();
                assert_eq!(provider.probes.lock().unwrap()[2].retirements(), 0);
                drop(output);
                assert_eq!(provider.probes.lock().unwrap()[2].retirements(), 1);
            }
        }
        assert_eq!(provider.probes.lock().unwrap()[0].retirements(), 1);
        assert_eq!(provider.probes.lock().unwrap()[1].retirements(), 1);
        assert_eq!(original.probe.retirements(), 0);
        drop(request);
        drop(original.handle);
        assert_eq!(original.probe.retirements(), 1);
    }
}

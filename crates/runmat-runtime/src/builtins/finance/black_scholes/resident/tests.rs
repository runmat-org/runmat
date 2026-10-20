use super::*;
use runmat_accelerate_api::{
    test_support::{RetirementProbe, TestAllocationOwner},
    AccelDownloadFuture, AccelProvider, HostNumericDataOwned, HostNumericDataView,
    HostNumericTensorOwned, HostNumericTensorView, HostTensorView, NumericElementType,
    ProviderBlackScholesPriceResult,
};
use std::sync::Mutex;

#[derive(Clone, Copy)]
enum ResultMode {
    Valid,
    Borrowed,
    SameOutput,
    Failure,
    Unsupported,
    WrongShape,
    WrongPrecision,
    Logical,
    ForeignOwner,
}

struct Provider {
    owner: TestAllocationOwner,
    foreign_owner: TestAllocationOwner,
    precision: ProviderPrecision,
    mode: ResultMode,
    probes: Mutex<Vec<RetirementProbe>>,
    requests: Mutex<Vec<Vec<usize>>>,
}

impl Provider {
    fn allocate(&self, shape: Vec<usize>) -> anyhow::Result<GpuTensorHandle> {
        let element = match self.precision {
            ProviderPrecision::F32 => NumericElementType::F32,
            ProviderPrecision::F64 => NumericElementType::F64,
        };
        let allocation = self
            .owner
            .allocate_zeroed(shape, element, GpuTensorStorage::Real)?;
        self.probes.lock().unwrap().push(allocation.probe);
        Ok(allocation.handle)
    }
}
impl AccelProvider for Provider {
    fn device_id(&self) -> u32 {
        self.owner.device_id()
    }
    fn device_info(&self) -> String {
        "owned Black-Scholes fixture".into()
    }
    fn precision(&self) -> ProviderPrecision {
        self.precision
    }
    fn upload(&self, _: &HostTensorView) -> anyhow::Result<GpuTensorHandle> {
        panic!("typed upload required")
    }
    fn upload_numeric_raw(
        &self,
        view: &HostNumericTensorView,
    ) -> runmat_accelerate_api::ProviderExecutionResult<GpuTensorHandle> {
        let data = match view.data {
            HostNumericDataView::F64(data) => HostNumericDataOwned::F64(data.to_vec()),
            HostNumericDataView::F32(data) => HostNumericDataOwned::F32(data.to_vec()),
            _ => return Err(anyhow::anyhow!("floating fixture expected").into()),
        };
        let allocation = self.owner.allocate_numeric(HostNumericTensorOwned {
            data,
            shape: view.shape.to_vec(),
            storage: view.storage,
        })?;
        self.probes.lock().unwrap().push(allocation.probe);
        Ok(allocation.handle)
    }
    fn download<'a>(&'a self, _: &'a GpuTensorHandle) -> AccelDownloadFuture<'a> {
        panic!("direct route must not gather")
    }
    fn black_scholes_price(
        &self,
        request: &ProviderBlackScholesPriceRequest<'_>,
    ) -> anyhow::Result<ProviderBlackScholesPriceResult> {
        self.requests
            .lock()
            .unwrap()
            .push(request.inputs[0].strides.to_vec());
        let call = match self.mode {
            ResultMode::Failure => anyhow::bail!("injected execution failure"),
            ResultMode::Unsupported => {
                return Err(runmat_accelerate_api::unsupported_provider_operation(
                    "fixture unsupported",
                ))
            }
            ResultMode::Borrowed => request.inputs[0].handle.clone(),
            ResultMode::WrongShape => self.allocate(vec![1, 1])?,
            ResultMode::WrongPrecision => {
                let wrong = if self.precision == ProviderPrecision::F32 {
                    NumericElementType::F64
                } else {
                    NumericElementType::F32
                };
                let allocation = self.owner.allocate_zeroed(
                    request.output_shape.to_vec(),
                    wrong,
                    GpuTensorStorage::Real,
                )?;
                self.probes.lock().unwrap().push(allocation.probe);
                allocation.handle
            }
            ResultMode::Logical => {
                let allocation = self
                    .owner
                    .allocate_logical_zeroed(request.output_shape.to_vec(), self.precision)?;
                self.probes.lock().unwrap().push(allocation.probe);
                allocation.handle
            }
            ResultMode::ForeignOwner => {
                let element = if self.precision == ProviderPrecision::F32 {
                    NumericElementType::F32
                } else {
                    NumericElementType::F64
                };
                let allocation = self.foreign_owner.allocate_zeroed(
                    request.output_shape.to_vec(),
                    element,
                    GpuTensorStorage::Real,
                )?;
                self.probes.lock().unwrap().push(allocation.probe);
                allocation.handle
            }
            _ => self.allocate(request.output_shape.to_vec())?,
        };
        let put = if matches!(self.mode, ResultMode::SameOutput) {
            call.clone()
        } else {
            self.allocate(request.output_shape.to_vec())?
        };
        Ok(ProviderBlackScholesPriceResult { call, put })
    }
}

#[test]
fn prepared_inputs_and_unrequested_outputs_retire_without_retiring_borrowed_input() {
    for mode in [
        ResultMode::Valid,
        ResultMode::Borrowed,
        ResultMode::SameOutput,
        ResultMode::Failure,
        ResultMode::Unsupported,
        ResultMode::WrongShape,
        ResultMode::WrongPrecision,
        ResultMode::Logical,
        ResultMode::ForeignOwner,
    ] {
        let provider = Box::leak(Box::new(Provider {
            owner: TestAllocationOwner::new(),
            foreign_owner: TestAllocationOwner::new(),
            precision: ProviderPrecision::F64,
            mode,
            probes: Mutex::new(Vec::new()),
            requests: Mutex::new(Vec::new()),
        }));
        let input = provider
            .owner
            .allocate_zeroed(vec![2, 1], NumericElementType::F64, GpuTensorStorage::Real)
            .unwrap();
        let _scope = runmat_accelerate_api::ThreadProviderGuard::set(Some(provider));
        let values = vec![
            Value::GpuTensor(input.handle.clone()),
            Value::Num(1.0),
            Value::Num(0.1),
            Value::Num(1.0),
            Value::Num(0.2),
            Value::Num(0.0),
        ];
        let result =
            futures::executor::block_on(try_blsprice_gpu(&values, precision::PricingClass::Double));
        match mode {
            ResultMode::Valid => {
                let output = result.unwrap().unwrap().output(Some(1)).unwrap();
                let probes = provider.probes.lock().unwrap();
                assert_eq!(probes.len(), 7);
                assert!(probes[..5].iter().all(|probe| probe.retirements() == 1));
                assert_eq!(probes[5].retirements(), 0);
                assert_eq!(probes[6].retirements(), 1);
                drop(output);
                assert_eq!(probes[5].retirements(), 1);
            }
            ResultMode::Unsupported => assert!(result.unwrap().is_none()),
            _ => assert_eq!(
                result.err().expect("provider rejection").identifier(),
                ERROR_INTERNAL.identifier
            ),
        }
        assert!(provider
            .probes
            .lock()
            .unwrap()
            .iter()
            .all(|probe| probe.retirements() == 1));
        assert_eq!(input.probe.retirements(), 0);
        drop(values);
        drop(input.handle);
        assert_eq!(input.probe.retirements(), 1);
    }
}

#[test]
fn invalid_later_shape_performs_no_upload_or_kernel_call() {
    let provider = Box::leak(Box::new(Provider {
        owner: TestAllocationOwner::new(),
        foreign_owner: TestAllocationOwner::new(),
        precision: ProviderPrecision::F64,
        mode: ResultMode::Valid,
        probes: Mutex::new(Vec::new()),
        requests: Mutex::new(Vec::new()),
    }));
    let input = provider
        .owner
        .allocate_zeroed(vec![2, 1], NumericElementType::F64, GpuTensorStorage::Real)
        .unwrap();
    let _scope = runmat_accelerate_api::ThreadProviderGuard::set(Some(provider));
    let values = vec![
        Value::GpuTensor(input.handle),
        Value::Num(1.0),
        Value::Tensor(Tensor::new(vec![0.0; 3], vec![3, 1]).unwrap()),
    ];
    assert!(futures::executor::block_on(try_blsprice_gpu(
        &values,
        precision::PricingClass::Double
    ))
    .is_err());
    assert!(provider.probes.lock().unwrap().is_empty());
    assert!(provider.requests.lock().unwrap().is_empty());
}

#[test]
fn transposed_input_uses_physical_strides_without_download() {
    let provider = Box::leak(Box::new(Provider {
        owner: TestAllocationOwner::new(),
        foreign_owner: TestAllocationOwner::new(),
        precision: ProviderPrecision::F64,
        mode: ResultMode::Valid,
        probes: Mutex::new(Vec::new()),
        requests: Mutex::new(Vec::new()),
    }));
    let input = provider
        .owner
        .allocate_zeroed(vec![2, 3], NumericElementType::F64, GpuTensorStorage::Real)
        .unwrap();
    let transposed = input
        .handle
        .with_view(
            runmat_accelerate_api::GpuTensorView::contiguous(vec![3, 2]).with_layout(
                runmat_accelerate_api::GpuTensorLayout::Transpose(
                    runmat_accelerate_api::TransposeInfo {
                        base_rows: 2,
                        base_cols: 3,
                    },
                ),
            ),
        )
        .unwrap();
    let _scope = runmat_accelerate_api::ThreadProviderGuard::set(Some(provider));
    let mut values = vec![Value::GpuTensor(transposed)];
    values.extend((0..5).map(|_| Value::Num(1.0)));
    let result =
        futures::executor::block_on(try_blsprice_gpu(&values, precision::PricingClass::Double))
            .unwrap()
            .unwrap();
    assert_eq!(*provider.requests.lock().unwrap(), vec![vec![2, 1]]);
    assert_eq!(result.call.shape(), &[3, 2]);
    assert_eq!(result.put.shape(), &[3, 2]);
    drop(result.output(Some(2)).unwrap());
    assert!(provider
        .probes
        .lock()
        .unwrap()
        .iter()
        .all(|probe| probe.retirements() == 1));
    assert_eq!(input.probe.retirements(), 0);
}

#[test]
fn single_provider_returns_owned_single_without_narrowing_double_arrays() {
    let provider = Box::leak(Box::new(Provider {
        owner: TestAllocationOwner::new(),
        foreign_owner: TestAllocationOwner::new(),
        precision: ProviderPrecision::F32,
        mode: ResultMode::Valid,
        probes: Mutex::new(Vec::new()),
        requests: Mutex::new(Vec::new()),
    }));
    let input = provider
        .owner
        .allocate_zeroed(vec![2, 1], NumericElementType::F32, GpuTensorStorage::Real)
        .expect("single input");
    let _scope = runmat_accelerate_api::ThreadProviderGuard::set(Some(provider));
    let values = vec![
        Value::GpuTensor(input.handle.clone()),
        Value::Num(95.0),
        Value::Num(0.1),
        Value::Num(0.25),
        Value::Num(0.5),
        Value::Num(0.0),
    ];
    let selected = precision::select(&values);
    assert_eq!(selected, precision::PricingClass::Single);
    let result = futures::executor::block_on(try_blsprice_gpu(&values, selected))
        .expect("single provider")
        .expect("single resident result");
    assert_eq!(
        runmat_accelerate_api::handle_precision(&result.call),
        Some(ProviderPrecision::F32)
    );
    assert_eq!(
        runmat_accelerate_api::handle_precision(&result.put),
        Some(ProviderPrecision::F32)
    );
    drop(result);

    let mut mixed = values.clone();
    mixed[1] = Value::Tensor(Tensor::new(vec![95.0, 95.0], vec![2, 1]).unwrap());
    let selected = precision::select(&mixed);
    assert_eq!(selected, precision::PricingClass::Double);
    let prior_calls = provider.requests.lock().unwrap().len();
    assert!(
        futures::executor::block_on(try_blsprice_gpu(&mixed, selected))
            .unwrap()
            .is_none()
    );
    assert_eq!(provider.requests.lock().unwrap().len(), prior_calls);
    assert_eq!(input.probe.retirements(), 0);
    drop(mixed);
    drop(values);
    drop(input.handle);
    assert_eq!(input.probe.retirements(), 1);
    assert!(provider
        .probes
        .lock()
        .unwrap()
        .iter()
        .all(|probe| probe.retirements() == 1));
}

#[test]
fn unrepresentable_host_scalar_precedes_single_provider_effects() {
    let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
    let provider = Box::leak(Box::new(Provider {
        owner: TestAllocationOwner::new(),
        foreign_owner: TestAllocationOwner::new(),
        precision: ProviderPrecision::F32,
        mode: ResultMode::Valid,
        probes: Mutex::new(Vec::new()),
        requests: Mutex::new(Vec::new()),
    }));
    let input = provider
        .owner
        .allocate_zeroed(vec![2, 1], NumericElementType::F32, GpuTensorStorage::Real)
        .expect("single input");
    let _scope = runmat_accelerate_api::ThreadProviderGuard::set(Some(provider));
    for scalar in [
        Value::Num(1.0e100),
        Value::Tensor(Tensor::new(vec![1.0e-100], vec![1, 1]).unwrap()),
    ] {
        let failure = futures::executor::block_on(blsprice_builtin(
            Value::GpuTensor(input.handle.clone()),
            scalar,
            Value::Num(0.1),
            Value::Num(0.25),
            Value::Num(0.5),
            crate::builtins::common::test_support::builtin_arguments(Vec::new()),
        ))
        .expect_err("finite scalar cannot enter single calculation");
        assert_eq!(failure.identifier(), ERROR_INVALID_INPUT.identifier);
    }
    assert!(provider.requests.lock().unwrap().is_empty());
    assert!(provider.probes.lock().unwrap().is_empty());
    assert_eq!(input.probe.retirements(), 0);
}

#[test]
fn single_provider_rejects_double_output_and_retires_allocations() {
    let provider = Box::leak(Box::new(Provider {
        owner: TestAllocationOwner::new(),
        foreign_owner: TestAllocationOwner::new(),
        precision: ProviderPrecision::F32,
        mode: ResultMode::WrongPrecision,
        probes: Mutex::new(Vec::new()),
        requests: Mutex::new(Vec::new()),
    }));
    let input = provider
        .owner
        .allocate_zeroed(vec![2, 1], NumericElementType::F32, GpuTensorStorage::Real)
        .expect("single input");
    let _scope = runmat_accelerate_api::ThreadProviderGuard::set(Some(provider));
    let values = vec![
        Value::GpuTensor(input.handle.clone()),
        Value::Num(95.0),
        Value::Num(0.1),
        Value::Num(0.25),
        Value::Num(0.5),
        Value::Num(0.0),
    ];
    let failure =
        futures::executor::block_on(try_blsprice_gpu(&values, precision::PricingClass::Single))
            .err()
            .expect("double provider output must be rejected");
    assert_eq!(failure.identifier(), ERROR_INTERNAL.identifier);
    assert!(provider
        .probes
        .lock()
        .unwrap()
        .iter()
        .all(|probe| probe.retirements() == 1));
    assert_eq!(input.probe.retirements(), 0);
}

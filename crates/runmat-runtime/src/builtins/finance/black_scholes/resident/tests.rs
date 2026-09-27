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
}

struct Provider {
    owner: TestAllocationOwner,
    mode: ResultMode,
    probes: Mutex<Vec<RetirementProbe>>,
    requests: Mutex<Vec<Vec<usize>>>,
}

impl Provider {
    fn allocate(&self, shape: Vec<usize>) -> anyhow::Result<GpuTensorHandle> {
        let allocation =
            self.owner
                .allocate_zeroed(shape, NumericElementType::F64, GpuTensorStorage::Real)?;
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
        ProviderPrecision::F64
    }
    fn upload(&self, _: &HostTensorView) -> anyhow::Result<GpuTensorHandle> {
        panic!("typed upload required")
    }
    fn upload_numeric_raw(&self, view: &HostNumericTensorView) -> anyhow::Result<GpuTensorHandle> {
        let HostNumericDataView::F64(data) = view.data else {
            anyhow::bail!("F64 fixture");
        };
        let allocation = self.owner.allocate_numeric(HostNumericTensorOwned {
            data: HostNumericDataOwned::F64(data.to_vec()),
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
    ] {
        let provider = Box::leak(Box::new(Provider {
            owner: TestAllocationOwner::new(),
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
        let result = futures::executor::block_on(try_blsprice_gpu(&values));
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
            _ => assert!(result.is_err()),
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
    assert!(futures::executor::block_on(try_blsprice_gpu(&values)).is_err());
    assert!(provider.probes.lock().unwrap().is_empty());
    assert!(provider.requests.lock().unwrap().is_empty());
}

#[test]
fn transposed_input_uses_physical_strides_without_download() {
    let provider = Box::leak(Box::new(Provider {
        owner: TestAllocationOwner::new(),
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
    let result = futures::executor::block_on(try_blsprice_gpu(&values))
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

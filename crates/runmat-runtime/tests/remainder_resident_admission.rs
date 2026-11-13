//! Real owning carriers exercise semantic admission before provider availability.
use futures::executor::block_on;
use runmat_accelerate_api::test_support::{RetirementProbe, TestAllocationOwner};
use runmat_accelerate_api::*;
use runmat_value::{IntValue, LogicalArray, Value};
use std::{collections::HashMap, sync::Mutex};

#[test]
fn remainder_pair_errors_precede_provider_lookup_but_not_invalid_storage() {
    let owner = TestAllocationOwner::new();
    let integer = owner
        .allocate_zeroed(vec![1, 1], NumericElementType::I64, GpuTensorStorage::Real)
        .unwrap();
    let complex = owner
        .allocate_zeroed(
            vec![1, 1],
            NumericElementType::I64,
            GpuTensorStorage::ComplexInterleaved,
        )
        .unwrap();
    assert!(provider_for_handle(&integer.handle).is_none());
    for name in ["rem", "mod"] {
        for args in [
            [Value::GpuTensor(integer.handle.clone()), Value::Bool(true)],
            [Value::Bool(true), Value::GpuTensor(integer.handle.clone())],
        ] {
            let error = runmat_runtime::call_builtin(name, &args).unwrap_err();
            assert!(error
                .message()
                .contains("integer arrays can only be combined with scalar double values"));
        }
        for args in [
            [Value::GpuTensor(complex.handle.clone()), Value::Bool(true)],
            [Value::Bool(true), Value::GpuTensor(complex.handle.clone())],
        ] {
            let error = runmat_runtime::call_builtin(name, &args).unwrap_err();
            assert!(error
                .message()
                .contains("expected an owned real numeric resident input"));
        }
        let error = runmat_runtime::call_builtin(
            name,
            &[
                Value::GpuTensor(integer.handle.clone()),
                Value::Int(IntValue::I64(2)),
            ],
        )
        .unwrap_err();
        assert!(
            error
                .message()
                .contains("expected an owned real numeric resident input"),
            "execution still requires the real provider"
        );
        let error =
            runmat_runtime::call_builtin(name, &[Value::Complex(1.0, 1.0), Value::Bool(true)])
                .unwrap_err();
        assert!(error.message().contains("inputs must be real"));
    }
    for allocation in [integer, complex] {
        assert_eq!(allocation.probe.retirements(), 0);
        allocation.handle.release().unwrap();
        assert_eq!(allocation.probe.retirements(), 1);
    }
}

struct LogicalProvider {
    owner: TestAllocationOwner,
    probes: Mutex<HashMap<u64, RetirementProbe>>,
}
impl AccelProvider for LogicalProvider {
    fn elem_rem<'a>(
        &'a self,
        _: &'a GpuTensorHandle,
        _: &'a GpuTensorHandle,
    ) -> runmat_accelerate_api::ProviderExecutionFuture<'a, GpuTensorHandle> {
        panic!("logical U8 must not enter the integer remainder hook")
    }
    fn elem_mod<'a>(
        &'a self,
        _: &'a GpuTensorHandle,
        _: &'a GpuTensorHandle,
    ) -> runmat_accelerate_api::ProviderExecutionFuture<'a, GpuTensorHandle> {
        panic!("logical U8 must not enter the integer modulus hook")
    }
    fn device_id(&self) -> u32 {
        self.owner.device_id()
    }
    fn device_info(&self) -> String {
        "logical U8 admission fixture".into()
    }
    fn upload(&self, _: &HostTensorView) -> anyhow::Result<GpuTensorHandle> {
        panic!("restoration must preserve its exact typed numeric carrier")
    }
    fn upload_numeric_raw(
        &self,
        host: &HostNumericTensorView,
    ) -> ProviderExecutionResult<GpuTensorHandle> {
        // Explicit resident intent restores the actual host result on this
        // same owner. This is a transfer, never an integer remainder hook.
        let data = match host.data {
            HostNumericDataView::F64(data) => HostNumericDataOwned::F64(data.to_vec()),
            HostNumericDataView::U8(data) => HostNumericDataOwned::U8(data.to_vec()),
            _ => panic!("unexpected logical remainder publication class"),
        };
        let allocation = self.owner.allocate_numeric(HostNumericTensorOwned {
            data,
            shape: host.shape.to_vec(),
            storage: host.storage,
        })?;
        self.probes
            .lock()
            .unwrap()
            .insert(allocation.handle.buffer_id(), allocation.probe);
        Ok(allocation.handle)
    }
    fn download<'a>(&'a self, input: &'a GpuTensorHandle) -> AccelDownloadFuture<'a> {
        Box::pin(async move {
            let host = self.download_numeric(input).await?;
            let data = match host.data {
                HostNumericDataOwned::U8(data) => data.into_iter().map(f64::from).collect(),
                HostNumericDataOwned::F64(data) => data,
                _ => panic!("logical input or exact restored host output"),
            };
            Ok(HostTensorOwned {
                data,
                shape: host.shape,
                storage: host.storage,
            })
        })
    }
    fn download_numeric<'a>(
        &'a self,
        input: &'a GpuTensorHandle,
    ) -> AccelNumericDownloadFuture<'a> {
        Box::pin(async move {
            let probe = self
                .probes
                .lock()
                .unwrap()
                .get(&input.buffer_id())
                .unwrap()
                .clone();
            probe.with_data(input, |data| {
                let data = match data {
                    HostNumericDataView::U8(data) => HostNumericDataOwned::U8(data.to_vec()),
                    HostNumericDataView::F64(data) => HostNumericDataOwned::F64(data.to_vec()),
                    _ => panic!("logical input or exact restored host output"),
                };
                HostNumericTensorOwned {
                    data,
                    shape: input.shape().to_vec(),
                    storage: GpuTensorStorage::Real,
                }
            })
        })
    }
}

#[test]
fn public_logical_u8_remainder_with_logical_or_bool_matches_host_domain() {
    let provider = Box::leak(Box::new(LogicalProvider {
        owner: TestAllocationOwner::new(),
        probes: Mutex::new(HashMap::new()),
    }));
    let _selected = ThreadProviderGuard::set(Some(provider));
    let input = provider
        .owner
        .allocate_logical(vec![1, 1], vec![2, 1])
        .unwrap();
    provider
        .probes
        .lock()
        .unwrap()
        .insert(input.handle.buffer_id(), input.probe.clone());
    let host = Value::LogicalArray(LogicalArray::new(vec![1, 1], vec![2, 1]).unwrap());
    let resident = Value::GpuTensor(input.handle.clone());
    for name in ["rem", "mod"] {
        for (args, host_args) in [
            (
                [resident.clone(), Value::Bool(true)],
                [host.clone(), Value::Bool(true)],
            ),
            (
                [Value::Bool(true), resident.clone()],
                [Value::Bool(true), host.clone()],
            ),
            (
                [resident.clone(), resident.clone()],
                [host.clone(), host.clone()],
            ),
        ] {
            let expected = runmat_runtime::call_builtin(name, &host_args).unwrap();
            let actual = runmat_runtime::call_builtin(name, &args).unwrap();
            let Value::GpuTensor(output) = actual.value() else {
                panic!("explicit resident input must restore on its exact owner");
            };
            assert_eq!(output.device_id(), input.handle.device_id());
            let actual = block_on(runmat_runtime::gather_if_needed_async(&actual)).unwrap();
            assert_eq!(
                &actual,
                expected.value(),
                "{name}: logical physical encoding must not select integer arithmetic"
            );
        }
    }
    assert!(handle_is_logical(&input.handle));
    assert_eq!(input.probe.retirements(), 0);
    drop(resident);
    input.handle.release().unwrap();
    assert_eq!(input.probe.retirements(), 1);
}

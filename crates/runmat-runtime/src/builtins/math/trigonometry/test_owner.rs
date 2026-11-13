//! Shared test-only payload ownership. Probes observe the same allocation that
//! supplies downloads; they do not retain a tensor lease or duplicate payloads.
use runmat_accelerate_api::{
    test_support::{RetirementProbe, TestAllocationOwner},
    *,
};
use std::sync::Mutex;

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn real_integer_boundary_rejects_resident_complex_before_transfer() {
    let _lock = crate::builtins::common::test_support::accel_test_lock();
    let _provider = ThreadProviderGuard::set(None);
    for (builtin, invalid, extension) in [
        (
            runmat_builtins::COS_CATALOG_ENTRY.identity,
            &runmat_builtins::COS_ERROR_INVALID_INPUT,
            &runmat_builtins::COS_INTEGER_INPUT_EXTENSION,
        ),
        (
            runmat_builtins::SIN_CATALOG_ENTRY.identity,
            &runmat_builtins::SIN_ERROR_INVALID_INPUT,
            &runmat_builtins::SIN_INTEGER_INPUT_EXTENSION,
        ),
        (
            runmat_builtins::TAN_CATALOG_ENTRY.identity,
            &runmat_builtins::TAN_ERROR_INVALID_INPUT,
            &runmat_builtins::TAN_INTEGER_INPUT_EXTENSION,
        ),
        (
            runmat_builtins::COSH_CATALOG_ENTRY.identity,
            &runmat_builtins::COSH_ERROR_INVALID_INPUT,
            &runmat_builtins::COSH_INTEGER_INPUT_EXTENSION,
        ),
        (
            runmat_builtins::SINH_CATALOG_ENTRY.identity,
            &runmat_builtins::SINH_ERROR_INVALID_INPUT,
            &runmat_builtins::SINH_INTEGER_INPUT_EXTENSION,
        ),
        (
            runmat_builtins::TANH_CATALOG_ENTRY.identity,
            &runmat_builtins::TANH_ERROR_INVALID_INPUT,
            &runmat_builtins::TANH_INTEGER_INPUT_EXTENSION,
        ),
        (
            runmat_builtins::SIND_CATALOG_ENTRY.identity,
            &runmat_builtins::SIND_ERROR_INVALID_INPUT,
            &runmat_builtins::SIND_INTEGER_INPUT_EXTENSION,
        ),
    ] {
        for enabled in [false, true] {
            let _compat = crate::compatibility::push_runmat_extensions_enabled(enabled);
            let allocation = TestAllocationOwner::new()
                .allocate_zeroed(
                    vec![1, 2],
                    NumericElementType::U64,
                    GpuTensorStorage::ComplexInterleaved,
                )
                .unwrap();
            let error = futures::executor::block_on(crate::call_builtin_async(
                builtin.name,
                &[runmat_value::Value::GpuTensor(allocation.handle.clone())],
            ))
            .expect_err("reject before exactness download or provider access");
            assert_eq!(
                error.identifier(),
                if enabled {
                    invalid.identifier
                } else {
                    extension.error_identifier
                },
                "{}",
                builtin.name
            );
            assert_eq!(allocation.probe.retirements(), 0);
            allocation
                .handle
                .allocation_lease()
                .validate_live()
                .unwrap();
            drop(allocation.handle);
            assert_eq!(allocation.probe.retirements(), 1);
        }
    }
}

pub(super) struct Owner {
    owner: TestAllocationOwner,
    probes: Mutex<Vec<RetirementProbe>>,
}
impl Owner {
    pub(super) fn new() -> Self {
        Self {
            owner: TestAllocationOwner::new(),
            probes: Mutex::new(Vec::new()),
        }
    }
    pub(super) fn device_id(&self) -> u32 {
        self.owner.device_id()
    }
    pub(super) fn upload(&self, host: &HostTensorView) -> anyhow::Result<GpuTensorHandle> {
        self.upload_numeric(&HostNumericTensorView {
            data: HostNumericDataView::F64(host.data),
            shape: host.shape,
            storage: GpuTensorStorage::Real,
        })
    }
    pub(super) fn upload_numeric(
        &self,
        host: &HostNumericTensorView,
    ) -> anyhow::Result<GpuTensorHandle> {
        host.validate()?;
        macro_rules! owned { ($($variant:ident),*) => {
            match host.data { $(HostNumericDataView::$variant(values) => HostNumericDataOwned::$variant(values.to_vec()),)* }
        }; }
        let allocation = self.owner.allocate_numeric(HostNumericTensorOwned {
            data: owned!(F32, F64, I8, I16, I32, I64, U8, U16, U32, U64),
            shape: host.shape.to_vec(),
            storage: host.storage,
        })?;
        self.probes.lock().unwrap().push(allocation.probe);
        Ok(allocation.handle)
    }
    pub(super) fn probe(&self, handle: &GpuTensorHandle) -> RetirementProbe {
        self.probes
            .lock()
            .unwrap()
            .iter()
            .find(|probe| probe.with_data(handle, |_| ()).is_ok())
            .expect("live fixture owner")
            .clone()
    }
    pub(super) fn download(&self, handle: &GpuTensorHandle) -> anyhow::Result<HostTensorOwned> {
        self.probe(handle).with_data(handle, |data| {
            let data = match data {
                HostNumericDataView::F64(values) => values.to_vec(),
                HostNumericDataView::F32(values) => values.iter().map(|&v| f64::from(v)).collect(),
                _ => panic!("floating numerical fixture only"),
            };
            HostTensorOwned {
                data,
                shape: handle.shape().to_vec(),
                storage: handle.physical().storage(),
            }
        })
    }
}

use super::super::{cdf::finish_cdf, pdf::finish_pdf};
use super::*;
use runmat_accelerate_api::{
    test_support::TestAllocationOwner, AccelDownloadFuture, AccelProvider, GpuTensorStorage,
    HostNumericTensorView, HostTensorView, NumericElementType, ThreadProviderGuard,
};
use std::sync::Mutex;

#[derive(Clone, Copy)]
enum UploadBehavior {
    AliasParameter,
    Fail,
    Fresh,
}

struct RestoreProvider {
    owner: TestAllocationOwner,
    behavior: UploadBehavior,
    alias: Mutex<Option<GpuTensorHandle>>,
}

impl RestoreProvider {
    fn new(behavior: UploadBehavior) -> &'static Self {
        let provider = Box::leak(Box::new(Self {
            owner: TestAllocationOwner::new(),
            behavior,
            alias: Mutex::new(None),
        }));
        // SAFETY: the fixture is leaked for the process lifetime, and each
        // TestAllocationOwner mints a distinct device identity.
        unsafe { runmat_accelerate_api::register_device_provider(provider) };
        provider
    }
}

impl AccelProvider for RestoreProvider {
    fn device_id(&self) -> u32 {
        self.owner.device_id()
    }

    fn device_info(&self) -> String {
        "fitted result restoration fixture".into()
    }

    fn upload(&self, _: &HostTensorView) -> anyhow::Result<GpuTensorHandle> {
        panic!("typed restoration must use numeric upload")
    }

    fn download<'a>(&'a self, _: &'a GpuTensorHandle) -> AccelDownloadFuture<'a> {
        Box::pin(async { panic!("result publication must not gather") })
    }

    fn upload_numeric_raw(
        &self,
        host: &HostNumericTensorView,
    ) -> runmat_accelerate_api::ProviderExecutionResult<GpuTensorHandle> {
        match self.behavior {
            UploadBehavior::AliasParameter => Ok(self
                .alias
                .lock()
                .unwrap()
                .take()
                .expect("one borrowed-alias result")),
            UploadBehavior::Fail => Err(anyhow::anyhow!("provider restoration failed").into()),
            UploadBehavior::Fresh => Ok(self
                .owner
                .allocate_zeroed(host.shape.to_vec(), host.data.element_type(), host.storage)?
                .handle),
        }
    }
}

fn finish(operation: FittedOperation, residency: EvaluationResidency) -> BuiltinResult<Value> {
    match operation {
        FittedOperation::Density => finish_pdf(
            vec![1, 2],
            vec![0.25, 0.75],
            CdfOutputPrecision::Double,
            residency,
        ),
        FittedOperation::Cumulative => finish_cdf(
            vec![1, 2],
            vec![0.25, 0.75],
            CdfOutputPrecision::Double,
            residency,
        ),
        FittedOperation::Inverse | FittedOperation::Sampling => {
            panic!("test fixture only exercises density and cumulative evaluation")
        }
    }
}

#[test]
fn fitted_result_rejects_alias_of_non_anchor_parameter_without_retiring_it() {
    let _lock = crate::builtins::common::test_support::accel_test_lock();
    for operation in [FittedOperation::Density, FittedOperation::Cumulative] {
        let builtin = operation.name();
        let provider = RestoreProvider::new(UploadBehavior::AliasParameter);
        let _selected = ThreadProviderGuard::set(Some(provider));
        let source = provider
            .owner
            .allocate_zeroed(vec![1, 2], NumericElementType::F64, GpuTensorStorage::Real)
            .unwrap();
        let parameter = provider
            .owner
            .allocate_zeroed(vec![1, 2], NumericElementType::F64, GpuTensorStorage::Real)
            .unwrap();
        let original = parameter.handle.wire_descriptor();
        *provider.alias.lock().unwrap() = Some(
            parameter
                .handle
                .with_view(parameter.handle.view().clone())
                .unwrap(),
        );
        let values = [
            Value::GpuTensor(source.handle.clone()),
            Value::GpuTensor(parameter.handle.clone()),
        ];
        let residency = EvaluationResidency::capture(&[&values[0], &values[1]]);
        let error =
            finish(operation, residency).expect_err("borrowed parameter must not be published");
        assert_eq!(error.context.builtin.as_deref(), Some(builtin));
        assert_eq!(parameter.handle.wire_descriptor(), original);
        assert_eq!(parameter.probe.retirements(), 0);
        assert!(parameter.handle.allocation_lease().validate_live().is_ok());
        assert!(source.handle.allocation_lease().validate_live().is_ok());
    }
}

#[test]
fn fitted_result_preserves_owner_cause_and_non_anchor_explicit_intent() {
    let _lock = crate::builtins::common::test_support::accel_test_lock();
    for operation in [FittedOperation::Density, FittedOperation::Cumulative] {
        let builtin = operation.name();
        for behavior in [UploadBehavior::Fail, UploadBehavior::Fresh] {
            let provider = RestoreProvider::new(behavior);
            let _selected = ThreadProviderGuard::set(Some(provider));
            let source = provider
                .owner
                .allocate_zeroed(vec![1, 2], NumericElementType::F64, GpuTensorStorage::Real)
                .unwrap();
            let parameter = provider
                .owner
                .allocate_zeroed(vec![1, 2], NumericElementType::F64, GpuTensorStorage::Real)
                .unwrap();
            let parameter_handle = parameter
                .handle
                .with_provenance(runmat_accelerate_api::GpuHandleProvenance::Explicit)
                .unwrap();
            let values = [
                Value::GpuTensor(source.handle.clone()),
                Value::GpuTensor(parameter_handle),
            ];
            let residency = EvaluationResidency::capture(&[&values[0], &values[1]]);
            match behavior {
                UploadBehavior::Fail => {
                    let error =
                        finish(operation, residency).expect_err("owner failure must surface");
                    assert_eq!(error.context.builtin.as_deref(), Some(builtin));
                    assert!(error.message().contains("provider restoration failed"));
                    assert!(std::error::Error::source(&error).is_some());
                }
                UploadBehavior::Fresh => {
                    let Value::GpuTensor(output) = finish(operation, residency).unwrap() else {
                        panic!("expected resident result");
                    };
                    assert!(runmat_accelerate_api::handle_is_explicit(&output));
                    assert_eq!(output.shape(), &[1, 2]);
                    assert_eq!(output.device_id(), provider.device_id());
                    assert!(!output.shares_allocation_with(&source.handle));
                    assert!(!output.shares_allocation_with(&parameter.handle));
                }
                UploadBehavior::AliasParameter => unreachable!("covered by alias witness"),
            }
            assert_eq!(source.probe.retirements(), 0);
            assert_eq!(parameter.probe.retirements(), 0);
        }
    }
}

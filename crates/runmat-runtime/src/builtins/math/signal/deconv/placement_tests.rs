//! Actual bounded host-route invocation and publication witnesses for deconv.

use futures::executor::block_on;
use runmat_accelerate::simple_provider::InProcessProvider;
use runmat_accelerate_api::{
    allocation::WeakGpuTensorHandle,
    observation::{ProviderObservationCapability, ProviderObservationSupport},
    AccelDownloadFuture, AccelProvider, GpuTensorHandle, HostNumericTensorView, HostTensorView,
    ProviderCapabilitySnapshot, ProviderFeasibility, ProviderFeasibilityQuery,
    ProviderOperationFamily, ProviderRejectionCode, ThreadProviderGuard,
};
use runmat_execution::{CandidateOutputPresentation, ExecutionCandidateKind, ResourceByteBound};
use runmat_value::Value;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Mutex;

use crate::builtins::common::test_support;
use crate::builtins::math::signal::common::convolution::placement::{
    tests::PlacementWitnessService, LateUploadAdmissionError,
};

use super::deconv_builtin;

struct ControlledUploadProvider {
    inner: InProcessProvider,
    uploads: AtomicUsize,
    fail_second: bool,
    reject_exact: bool,
    extra_physical_bytes: u64,
    first_result: Mutex<Option<WeakGpuTensorHandle>>,
}

impl ControlledUploadProvider {
    fn new(fail_second: bool, reject_exact: bool) -> &'static Self {
        Self::new_with_extra(fail_second, reject_exact, 0)
    }

    fn new_with_extra(
        fail_second: bool,
        reject_exact: bool,
        extra_physical_bytes: u64,
    ) -> &'static Self {
        Box::leak(Box::new(Self {
            inner: InProcessProvider::new(),
            uploads: AtomicUsize::new(0),
            fail_second,
            reject_exact,
            extra_physical_bytes,
            first_result: Mutex::new(None),
        }))
    }
}

impl AccelProvider for ControlledUploadProvider {
    fn device_id(&self) -> u32 {
        self.inner.device_id()
    }
    fn device_info(&self) -> String {
        "deconv-controlled-upload".into()
    }
    fn upload(&self, host: &HostTensorView) -> anyhow::Result<GpuTensorHandle> {
        self.inner.upload(host)
    }
    fn download<'a>(&'a self, handle: &'a GpuTensorHandle) -> AccelDownloadFuture<'a> {
        self.inner.download(handle)
    }
    fn download_numeric<'a>(
        &'a self,
        handle: &'a GpuTensorHandle,
    ) -> runmat_accelerate_api::AccelNumericDownloadFuture<'a> {
        self.inner.download_numeric(handle)
    }
    fn upload_numeric_raw(
        &self,
        host: &HostNumericTensorView,
    ) -> runmat_accelerate_api::ProviderExecutionResult<GpuTensorHandle> {
        let index = self.uploads.fetch_add(1, Ordering::SeqCst);
        if self.fail_second && index == 1 {
            return Err(anyhow::anyhow!(
                "second result upload failed after first result publication"
            )
            .into());
        }
        let output = self.inner.upload_numeric_raw(host)?;
        if index == 0 {
            *self.first_result.lock().unwrap() = Some(output.downgrade());
        }
        Ok(output)
    }
    fn capability_snapshot(&self) -> ProviderCapabilitySnapshot {
        self.inner.capability_snapshot()
    }
    fn query_feasibility(&self, query: &ProviderFeasibilityQuery) -> ProviderFeasibility {
        if self.reject_exact
            && query.family == ProviderOperationFamily::Upload
            && query.workload.transfer_output_bounds.is_none()
            && !query.outputs.is_empty()
        {
            return ProviderFeasibility::rejected(
                ProviderRejectionCode::UnsupportedOperation,
                "deconv.fixture.exact_upload_rejected",
            );
        }
        let feasibility = self.inner.query_feasibility(query);
        if query.workload.transfer_output_bounds.is_some() {
            if let ProviderFeasibility::Supported { mut estimate } = feasibility {
                estimate.output_bytes = estimate
                    .output_bytes
                    .and_then(|bytes| bytes.checked_add(self.extra_physical_bytes));
                return ProviderFeasibility::Supported { estimate };
            }
        }
        feasibility
    }
    fn operation_observation_capability(
        &self,
        query: &ProviderFeasibilityQuery,
    ) -> ProviderObservationCapability {
        self.inner.operation_observation_capability(query)
    }
    fn begin_operation_observation(
        &self,
        query: &ProviderFeasibilityQuery,
    ) -> ProviderObservationSupport {
        self.inner.begin_operation_observation(query)
    }
}

fn inputs(provider: &'static ControlledUploadProvider) -> (GpuTensorHandle, GpuTensorHandle) {
    let numerator = provider
        .upload(&HostTensorView {
            data: &[1.0, 3.0, 3.0, 1.0],
            shape: &[1, 4],
        })
        .unwrap();
    let denominator = provider
        .upload(&HostTensorView {
            data: &[1.0, 1.0],
            shape: &[1, 2],
        })
        .unwrap();
    (numerator, denominator)
}

#[test]
fn requested_zero_one_two_outputs_select_only_real_published_work() {
    let _serial = test_support::accel_test_lock();
    let provider = ControlledUploadProvider::new(false, false);
    let _selected = ThreadProviderGuard::set(Some(provider));
    for requested in [0, 1, 2] {
        let service = PlacementWitnessService::new(false);
        let (numerator, denominator) = inputs(provider);
        let uploads_before = provider.uploads.load(Ordering::SeqCst);
        let _outputs =
            crate::output_count::push_output_count(Some(requested)).expect("output-count scope");
        let result = block_on(deconv_builtin(
            Some(service.context()),
            Value::GpuTensor(numerator.clone()),
            Value::GpuTensor(denominator.clone()),
        ))
        .expect("bounded deconv invocation");
        let Value::OutputList(values) = result else {
            panic!("explicit output count returns list")
        };
        assert_eq!(values.len(), requested);
        assert!(values
            .iter()
            .all(|value| matches!(value, Value::GpuTensor(_))));
        assert_eq!(
            provider.uploads.load(Ordering::SeqCst) - uploads_before,
            requested
        );
        if let Some(quotient) = values.first() {
            assert_eq!(
                test_support::gather(quotient.clone())
                    .unwrap()
                    .materialize_f64(),
                vec![1.0, 2.0, 1.0]
            );
        }
        if let Some(remainder) = values.get(1) {
            assert_eq!(
                test_support::gather(remainder.clone())
                    .unwrap()
                    .materialize_f64(),
                vec![0.0; 4]
            );
        }
        assert_eq!(service.selections(), usize::from(requested != 0));
        let feedback = service.feedback();
        assert_eq!(feedback.len(), usize::from(requested != 0));
        if let Some(feedback) = feedback.first() {
            assert!(feedback.succeeded);
            assert_eq!(feedback.kind, ExecutionCandidateKind::SharedRuntime);
        }
        assert!(numerator.allocation_lease().validate_live().is_ok());
        assert!(denominator.allocation_lease().validate_live().is_ok());
    }
}

#[test]
fn host_route_retained_budget_uses_provider_physical_estimate_not_copy_volume() {
    let _serial = test_support::accel_test_lock();
    let provider = ControlledUploadProvider::new_with_extra(false, false, 11);
    let _selected = ThreadProviderGuard::set(Some(provider));
    let (numerator, denominator) = inputs(provider);
    let service = PlacementWitnessService::new(false);
    let _outputs = crate::output_count::push_output_count(Some(2)).expect("output-count scope");
    let _result = block_on(deconv_builtin(
        Some(service.context()),
        Value::GpuTensor(numerator.clone()),
        Value::GpuTensor(denominator.clone()),
    ))
    .expect("admitted bounded host route");
    let candidate = service
        .selected_candidate()
        .expect("host candidate selected");
    let CandidateOutputPresentation::ProviderUpload { transfer_bytes, .. } = candidate.presentation
    else {
        panic!("host route presents provider uploads")
    };
    assert_eq!(transfer_bytes, Some(64));
    assert_eq!(
        candidate.resources.retained_bytes,
        ResourceByteBound::Known(75)
    );
    assert_eq!(provider.uploads.load(Ordering::SeqCst), 2);
    assert!(numerator.allocation_lease().validate_live().is_ok());
    assert!(denominator.allocation_lease().validate_live().is_ok());
}

#[test]
fn second_restore_failure_drops_first_result_without_retiring_callers() {
    let _serial = test_support::accel_test_lock();
    let provider = ControlledUploadProvider::new(true, false);
    let _selected = ThreadProviderGuard::set(Some(provider));
    let (numerator, denominator) = inputs(provider);
    let service = PlacementWitnessService::new(false);
    let _outputs = crate::output_count::push_output_count(Some(2)).expect("output-count scope");
    let error = block_on(deconv_builtin(
        Some(service.context()),
        Value::GpuTensor(numerator.clone()),
        Value::GpuTensor(denominator.clone()),
    ))
    .expect_err("second upload failure must terminate");
    assert_eq!(error.gpu_gather_retry, crate::GpuGatherRetry::Never);
    assert_eq!(service.selections(), 1);
    assert_eq!(provider.uploads.load(Ordering::SeqCst), 2);
    assert!(
        service.feedback().is_empty(),
        "partial work cannot train a complete-call profile"
    );
    assert!(provider
        .first_result
        .lock()
        .unwrap()
        .as_ref()
        .unwrap()
        .acquire()
        .is_err());
    assert!(numerator.allocation_lease().validate_live().is_ok());
    assert!(denominator.allocation_lease().validate_live().is_ok());
}

#[test]
fn exact_late_upload_rejection_precedes_any_result_upload() {
    let _serial = test_support::accel_test_lock();
    let provider = ControlledUploadProvider::new(false, true);
    let _selected = ThreadProviderGuard::set(Some(provider));
    let (numerator, denominator) = inputs(provider);
    let service = PlacementWitnessService::new(false);
    let _outputs = crate::output_count::push_output_count(Some(2)).expect("output-count scope");
    let error = block_on(deconv_builtin(
        Some(service.context()),
        Value::GpuTensor(numerator.clone()),
        Value::GpuTensor(denominator.clone()),
    ))
    .expect_err("late typed rejection must stop before publication");
    assert_eq!(
        error.identifier(),
        Some("RunMat:Placement:LateUploadAdmissionFailed")
    );
    let source = error
        .source
        .as_deref()
        .and_then(|cause| cause.downcast_ref::<LateUploadAdmissionError>());
    assert!(matches!(
        source,
        Some(LateUploadAdmissionError::ProviderRejected { rejection })
            if rejection.code == ProviderRejectionCode::UnsupportedOperation
                && rejection.detail.as_deref()
                    == Some("deconv.fixture.exact_upload_rejected")
    ));
    assert_eq!(provider.uploads.load(Ordering::SeqCst), 0);
    assert_eq!(service.selections(), 1);
    assert!(service.feedback().is_empty());
    assert!(numerator.allocation_lease().validate_live().is_ok());
    assert!(denominator.allocation_lease().validate_live().is_ok());
}

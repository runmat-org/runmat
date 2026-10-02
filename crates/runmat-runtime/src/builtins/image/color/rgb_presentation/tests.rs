use super::*;
use crate::builtins::common::{gpu_helpers, test_support};
use crate::context::host_presentation::tests::Witness;
use crate::context::{
    preparation::{AdmittedBuiltinPreparation, PreparedBuiltinOperands},
    RuntimeAccelerationService, RuntimeContextGuard, RuntimeParallelResources,
    RuntimeParallelService, RuntimeServiceFuture, RuntimeServicePorts,
};
use futures::executor::block_on;
use runmat_value::{IntegerStorage, Tensor};
use std::rc::Rc;

struct HostBudget(u64);
impl RuntimeParallelService for HostBudget {
    fn supports(&self, _: crate::context::ParallelCapability) -> bool {
        false
    }
    fn placement_resources(&self) -> RuntimeParallelResources {
        RuntimeParallelResources {
            cpu_millicores_available: 1_000,
            memory_available_bytes: Some(self.0),
            epoch: 0,
        }
    }
}

struct RejectPreparation;
impl RuntimeAccelerationService for RejectPreparation {
    fn supports_operation(&self, operation: &str) -> bool {
        matches!(operation, "rgb2gray" | "rgb2hsv")
    }
    fn prepare_admitted(
        &self,
        _: RuntimeContext,
        _: AdmittedBuiltinPreparation,
    ) -> RuntimeServiceFuture<Result<PreparedBuiltinOperands, crate::RuntimeError>> {
        Box::pin(async {
            Err(crate::build_runtime_error("RGB preparation failed")
                .with_identifier("RunMat:Preparation:FixtureFailure")
                .build())
        })
    }
}

fn budgeted_context(bytes: u64) -> RuntimeContext {
    RuntimeContext::new(Rc::new(crate::execution::RuntimeExecutionService::new()))
        .with_service_ports(
            RuntimeServicePorts::default()
                .with_placement(Rc::new(
                    runmat_accelerate::placement::PlacementSession::new(
                        runmat_accelerate::placement::PlacementSessionConfig::default(),
                    ),
                ))
                .with_parallel(Rc::new(HostBudget(bytes))),
        )
}

#[test]
fn typed_rgb_output_and_operation_scratch_bounds_cover_scalar_and_classes() {
    let one = ColorLayout::Truecolor { rows: 1, cols: 1 };
    let map = ColorLayout::Colormap { rows: 2 };
    assert_eq!(
        output_dtype(RgbConversion::Gray, one, Some(1), NumericDType::F32),
        NumericDType::F64
    );
    assert_eq!(
        output_dtype(RgbConversion::Gray, one, Some(1), NumericDType::U8),
        NumericDType::U8
    );
    assert_eq!(
        output_dtype(RgbConversion::Gray, map, Some(2), NumericDType::F64),
        NumericDType::F64
    );
    assert_eq!(
        output_dtype(RgbConversion::Hsv, one, Some(1), NumericDType::U16),
        NumericDType::F64
    );
    assert_eq!(
        host_conversion_scratch(RgbConversion::Gray, NumericDType::U8, Some(4)),
        ResourceByteBound::Known(32)
    );
    assert_eq!(
        host_conversion_scratch(RgbConversion::Hsv, NumericDType::F64, Some(4)),
        ResourceByteBound::Known(96)
    );
    assert_eq!(
        host_conversion_scratch(RgbConversion::Hsv, NumericDType::F32, Some(4)),
        ResourceByteBound::Known(192)
    );
    assert_eq!(
        host_conversion_scratch(RgbConversion::Hsv, NumericDType::F32, Some(u64::MAX)),
        ResourceByteBound::Unavailable(ResourceBoundUnavailable::NoScratchBound)
    );
}

#[test]
fn both_rgb_calls_select_one_owner_upload_and_observe_complete_publication() {
    test_support::with_test_provider(|provider| {
        let rgb = Tensor::new_integer(IntegerStorage::U8(vec![255, 0, 0]), vec![1, 1, 3]).unwrap();
        let source = gpu_helpers::upload_tensor(provider, &rgb)
            .unwrap()
            .with_provenance(runmat_accelerate_api::GpuHandleProvenance::Explicit)
            .unwrap();
        for name in ["rgb2gray", "rgb2hsv"] {
            let witness = Rc::new(Witness::default());
            let _runtime = RuntimeContextGuard::enter(
                runmat_accelerate::native_auto::install_preparation_service(witness.context()),
            );
            let result = block_on(crate::dispatcher::call_builtin_async(
                name,
                &[Value::GpuTensor(source.clone())],
            ))
            .unwrap();
            let Value::GpuTensor(output) = result else {
                panic!("{name} must restore to source owner");
            };
            assert!(!output.shares_allocation_with(&source));
            assert!(provider_for_handle(&output)
                .is_some_and(|owner| same_provider_instance(owner, provider)));
            assert_eq!(
                witness.selections(),
                1,
                "{name} selected one whole work route"
            );
            witness.drain();
            assert_eq!(
                witness.feedback().len(),
                1,
                "{name} has one completed sample"
            );
        }
        assert!(source.allocation_lease().validate_live().is_ok());
    });
}

#[test]
fn finite_host_quota_rejects_rgb_route_before_readback_or_upload() {
    test_support::with_test_provider(|provider| {
        let rgb = Tensor::new_integer(IntegerStorage::U8(vec![255, 0, 0]), vec![1, 1, 3]).unwrap();
        let source = gpu_helpers::upload_tensor(provider, &rgb).unwrap();
        provider.reset_telemetry();
        let _runtime = RuntimeContextGuard::enter(
            runmat_accelerate::native_auto::install_preparation_service(budgeted_context(1)),
        );
        let error = block_on(crate::dispatcher::call_builtin_async(
            "rgb2gray",
            &[Value::GpuTensor(source.clone())],
        ))
        .expect_err("finite quota must reject before Sink(Gather)");
        assert!(error
            .identifier()
            .is_some_and(|id| id.starts_with("RunMat:Placement:")));
        assert_eq!(provider.telemetry_snapshot().download_bytes, 0);
        assert_eq!(provider.telemetry_snapshot().upload_bytes, 0);
        assert!(source.allocation_lease().validate_live().is_ok());
    });
}

#[test]
fn one_pixel_single_gray_upload_is_typed_double_scalar_hsv_stays_single() {
    test_support::with_test_provider(|provider| {
        let rgb = Tensor::from_f32(vec![1.0, 0.0, 0.0], vec![1, 1, 3]).unwrap();
        let source = gpu_helpers::upload_tensor(provider, &rgb).unwrap();
        for (name, expected) in [
            ("rgb2gray", ProviderPrecision::F64),
            ("rgb2hsv", ProviderPrecision::F32),
        ] {
            let witness = Rc::new(Witness::default());
            let _runtime = RuntimeContextGuard::enter(
                runmat_accelerate::native_auto::install_preparation_service(witness.context()),
            );
            let Value::GpuTensor(output) = block_on(crate::dispatcher::call_builtin_async(
                name,
                &[Value::GpuTensor(source.clone())],
            ))
            .unwrap() else {
                panic!("{name} output must be resident");
            };
            assert_eq!(handle_precision(&output), Some(expected));
            assert_eq!(witness.selections(), 1);
        }
    });
}

#[test]
fn unpolled_rgb_call_has_no_selection_or_provider_effects() {
    test_support::with_test_provider(|provider| {
        let rgb = Tensor::new_integer(IntegerStorage::U8(vec![255, 0, 0]), vec![1, 1, 3]).unwrap();
        let source = gpu_helpers::upload_tensor(provider, &rgb).unwrap();
        let witness = Rc::new(Witness::default());
        let _runtime = RuntimeContextGuard::enter(
            runmat_accelerate::native_auto::install_preparation_service(witness.context()),
        );
        provider.reset_telemetry();
        let operands = [Value::GpuTensor(source.clone())];
        let future = crate::dispatcher::call_builtin_async("rgb2hsv", &operands);
        drop(future);
        assert_eq!(witness.selections(), 0);
        assert!(witness.feedback().is_empty());
        assert_eq!(provider.telemetry_snapshot().download_bytes, 0);
        assert_eq!(provider.telemetry_snapshot().upload_bytes, 0);
        assert!(source.allocation_lease().validate_live().is_ok());
    });
}

#[test]
fn failed_preparation_abandons_selected_rgb_receipt_without_upload() {
    test_support::with_test_provider(|provider| {
        let rgb = Tensor::new_integer(IntegerStorage::U8(vec![255, 0, 0]), vec![1, 1, 3]).unwrap();
        let source = gpu_helpers::upload_tensor(provider, &rgb).unwrap();
        let witness = Rc::new(Witness::default());
        let context = witness.context();
        let ports = context
            .service_ports()
            .clone()
            .with_acceleration(Rc::new(RejectPreparation));
        let _runtime = RuntimeContextGuard::enter(context.with_service_ports(ports));
        provider.reset_telemetry();
        let error = block_on(crate::dispatcher::call_builtin_async(
            "rgb2gray",
            &[Value::GpuTensor(source.clone())],
        ))
        .expect_err("preparation failure must not replay or publish");
        assert_eq!(
            error.identifier(),
            Some("RunMat:Preparation:FixtureFailure")
        );
        assert_eq!(witness.selections(), 1);
        witness.drain();
        assert!(witness.feedback().is_empty());
        assert_eq!(provider.telemetry_snapshot().download_bytes, 0);
        assert_eq!(provider.telemetry_snapshot().upload_bytes, 0);
        assert!(source.allocation_lease().validate_live().is_ok());
    });
}

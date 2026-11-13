#![cfg(feature = "wgpu")]
//! These witnesses require an actual adapter and execute the admitted shader.
#[path = "support/typed_fusion_plan.rs"]
mod fixture;
#[path = "support/retirement.rs"]
mod retirement;

use futures::executor::block_on;
use runmat_accelerate::backend::wgpu::provider::{register_wgpu_provider, WgpuProviderOptions};
use runmat_accelerate::fusion_exec::{
    execute_elementwise, execute_elementwise_placed, FusionExecutionOutcome, FusionExecutionRequest,
};
use runmat_accelerate_api::{
    AccelProvider, GpuHandleProvenance, GpuTensorStorage, HostNumericDataOwned,
    HostNumericDataView, HostNumericTensorView, NumericElementType, ProviderPrecision,
};
use runmat_value::Value;

fn guarded_plan(
    entry: &'static runmat_builtins::BuiltinCatalogEntry,
    shape: &[usize],
) -> runmat_accelerate::fusion::FusionGroupPlan {
    let [binding] = entry.bindings else {
        panic!("guarded unary needs one canonical binding");
    };
    let identity = entry.binding_identity(binding);
    let declaration =
        runmat_runtime::builtins::common::spec::canonical_fusion_declaration(identity)
            .expect("compiled guarded unary declaration");
    assert_eq!(declaration.binding, identity);
    assert!(declaration
        .semantics
        .expect("real-domain semantic declaration")
        .required_real_unary
        .is_some());
    let mut plan = fixture::plan(vec![fixture::cast(entry.identity.name, 0, 1)], 1);
    plan.group.shape =
        runmat_accelerate::graph::ShapeInfo::Tensor(shape.iter().copied().map(Some).collect());
    plan
}

#[test]
fn actual_single_adapter_executes_guarded_inverse_domains_and_preserves_owner() {
    let provider = register_wgpu_provider(WgpuProviderOptions::default())
        .expect("actual WGPU adapter required");
    assert_eq!(provider.precision(), ProviderPrecision::F32);
    for (entry, expected_value, values) in [
        (
            &runmat_builtins::ACOSH_CATALOG_ENTRY,
            f32::acosh as fn(f32) -> f32,
            vec![1.0f32, 1.25, 2.0, 4.0],
        ),
        (
            &runmat_builtins::ATANH_CATALOG_ENTRY,
            f32::atanh as fn(f32) -> f32,
            vec![-1.0f32, -0.25, 0.0, 0.5, 1.0],
        ),
    ] {
        let shape = [1, values.len()];
        let input = provider
            .upload_numeric_raw(&HostNumericTensorView {
                data: HostNumericDataView::F32(&values),
                shape: &shape,
                storage: GpuTensorStorage::Real,
            })
            .expect("native single upload")
            .with_provenance(GpuHandleProvenance::Explicit)
            .expect("explicit residency");
        let plan = guarded_plan(entry, &shape);
        let result = block_on(execute_elementwise(FusionExecutionRequest {
            plan: &plan,
            inputs: vec![Value::GpuTensor(input.clone())],
            placement: None,
            runtime: None,
        }))
        .expect("admitted real-domain shader execution");
        assert!(result.materialized_stores.is_empty());
        let Value::GpuTensor(output) = result.final_value else {
            panic!("explicitly resident guarded output required");
        };
        assert_eq!(output.shape(), shape);
        assert_eq!(output.physical().element_type(), NumericElementType::F32);
        assert_eq!(output.physical().storage(), GpuTensorStorage::Real);
        assert!(runmat_accelerate_api::handle_is_explicit(&output));
        assert!(!input.shares_allocation_with(&output));
        let owner =
            runmat_runtime::builtins::common::gpu_helpers::exact_provider_for_handle(&output)
                .expect("output concrete owner");
        assert!(runmat_accelerate_api::same_provider_instance(
            owner, provider
        ));
        let downloaded = block_on(provider.download_numeric(&output)).expect("native readback");
        assert_eq!(downloaded.shape, shape);
        let HostNumericDataOwned::F32(actual) = downloaded.data else {
            panic!("physical single output required");
        };
        assert_eq!(actual.len(), values.len());
        for (actual, value) in actual.iter().zip(values) {
            let expected = expected_value(value);
            if expected.is_infinite() {
                assert_eq!(*actual, expected);
            } else {
                assert!(
                    (*actual - expected).abs() <= 3e-6 * expected.abs().max(1.0),
                    "{}: {value} produced {actual}, expected {expected}",
                    entry.identity.name,
                );
            }
        }
        input.allocation_lease().validate_live().unwrap();
        retirement::release_all([input, output]);
    }
}

#[test]
fn actual_single_adapter_defers_unproved_domains_without_retiring_inputs() {
    let provider = register_wgpu_provider(WgpuProviderOptions::default())
        .expect("actual WGPU adapter required");
    assert_eq!(provider.precision(), ProviderPrecision::F32);
    for (entry, values) in [
        (&runmat_builtins::ACOSH_CATALOG_ENTRY, vec![0.5f32, 2.0]),
        (&runmat_builtins::ATANH_CATALOG_ENTRY, vec![1.25f32, 0.5]),
        (&runmat_builtins::ATANH_CATALOG_ENTRY, vec![f32::NAN, 0.5]),
        (&runmat_builtins::ACOSH_CATALOG_ENTRY, Vec::new()),
    ] {
        let shape = [1, values.len()];
        let input = provider
            .upload_numeric_raw(&HostNumericTensorView {
                data: HostNumericDataView::F32(&values),
                shape: &shape,
                storage: GpuTensorStorage::Real,
            })
            .expect("native single upload")
            .with_provenance(GpuHandleProvenance::Explicit)
            .expect("explicit residency");
        let before = input.wire_descriptor();
        let plan = guarded_plan(entry, &shape);
        let result = block_on(execute_elementwise_placed(FusionExecutionRequest {
            plan: &plan,
            inputs: vec![Value::GpuTensor(input.clone())],
            placement: None,
            runtime: None,
        }))
        .expect("unknown real domain retains shared runtime");
        assert!(matches!(result, FusionExecutionOutcome::SharedRuntime(_)));
        assert_eq!(input.wire_descriptor(), before);
        input.allocation_lease().validate_live().unwrap();
        retirement::release_all([input]);
    }
}

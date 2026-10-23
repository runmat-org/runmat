//! Actual typed caller witnesses. The provider hooks, allocation owner and
//! selected recipe path are real; numerical GPU qualification remains in the
//! WGPU product tests.
use super::*;
use crate::fusion::{FusionGroup, FusionGroupPlan, FusionKernelSpec, FusionKind, FusionOp};
use crate::graph::{FusionCallOrigin, InstrSpan, ShapeInfo};
use runmat_accelerate_api::reduction::ReductionStrategy;
use runmat_accelerate_api::test_support::{RetirementProbe, TestAllocationOwner};
use runmat_accelerate_api::{
    AccelProvider, GpuTensorHandle, GpuTensorStorage, HostNumericDataOwned, NumericElementType,
    ProviderExecutionResult, ThreadProviderGuard,
};
use runmat_value::{Tensor, Value};
use std::collections::HashMap;
use std::sync::atomic::Ordering;

mod provider;
use provider::*;
mod routes;
mod projection;

fn initialize_registry() {
    #[cfg(target_arch = "wasm32")]
    runmat_runtime::builtins::wasm_registry::register_all();
}

fn sum_plan() -> FusionGroupPlan {
    initialize_registry();
    let mut const_values = HashMap::new();
    const_values.insert(1, Value::Num(1.0));
    FusionGroupPlan {
        index: 0,
        group: FusionGroup {
            id: 0,
            kind: FusionKind::Reduction,
            nodes: vec![],
            shape: ShapeInfo::Tensor(vec![Some(1), Some(1)]),
            span: InstrSpan { start: 0, end: 1 },
            pattern: None,
            stack_layout: None,
        },
        operations: vec![FusionOp::Builtin {
            name: "sum".into(),
            origin: FusionCallOrigin::named("sum"),
            inputs: vec![0, 1],
            output: Some(2),
        }],
        inputs: vec![0],
        stack_pattern: vec![0],
        constants: HashMap::new(),
        const_values,
        materialized_stores: vec![],
        output: Some(2),
        kernel: FusionKernelSpec {
            kind: FusionKind::Reduction,
            supported: true,
        },
        reduction_data: Some(0),
        reduction_dim: None,
        reduction_flavor: Some(runmat_accelerate_api::ReductionFlavor::Sum),
        reduction_axes: None,
        pattern: None,
    }
}

fn sum_plan_double() -> FusionGroupPlan {
    let mut plan = sum_plan();
    let output_class = 3;
    plan.const_values
        .insert(output_class, Value::String("double".into()));
    let FusionOp::Builtin { inputs, .. } = &mut plan.operations[0] else {
        unreachable!("sum fixture must retain its builtin operation")
    };
    inputs.push(output_class);
    plan
}

#[cfg(not(target_arch = "wasm32"))]
fn min_plan() -> FusionGroupPlan {
    let mut plan = sum_plan();
    // The ordinary reduction form is min(A, [], dim), not pairwise min(A,B).
    plan.const_values.insert(
        1,
        Value::Tensor(Tensor::new(Vec::new(), vec![0, 0]).unwrap()),
    );
    plan.const_values.insert(3, Value::Num(1.0));
    let FusionOp::Builtin {
        name,
        origin,
        inputs,
        ..
    } = &mut plan.operations[0]
    else {
        unreachable!("minimum fixture must retain its builtin operation")
    };
    *name = "min".into();
    *origin = FusionCallOrigin::named("min");
    *inputs = vec![0, 1, 3];
    plan.reduction_flavor = None;
    plan
}

#[cfg(not(target_arch = "wasm32"))]
#[test]
fn legacy_flavor_uses_selected_operation_not_planner_sum_default() {
    runmat_gc::gc_test_context(|| {
        for (name, expected) in [
            ("sum", Some(runmat_accelerate_api::ReductionFlavor::Sum)),
            ("mean", Some(runmat_accelerate_api::ReductionFlavor::Mean)),
            ("min", None),
            ("max", None),
            ("prod", None),
            ("nnz", None),
        ] {
            let mut plan = sum_plan();
            let FusionOp::Builtin {
                name: selected_name,
                origin,
                ..
            } = &mut plan.operations[0]
            else {
                unreachable!("fixture has a terminal selected builtin")
            };
            *selected_name = name.into();
            *origin = FusionCallOrigin::named(name);
            // The old graph heuristic claims Sum even for extrema. The
            // selected operation, not this metadata, governs legacy admission.
            plan.reduction_flavor = Some(runmat_accelerate_api::ReductionFlavor::Sum);
            let request = host_request(&plan, vec![1.0, 2.0]);
            let bindings = super::AdmittedBindings::new(&request).expect("selected binding");
            assert_eq!(
                super::legacy::selected_legacy_flavor(&request, &bindings),
                expected
            );
        }
    });
}

fn host_request<'a>(plan: &'a FusionGroupPlan, values: Vec<f64>) -> FusionExecutionRequest<'a> {
    let rows = values.len();
    FusionExecutionRequest {
        plan,
        inputs: vec![Value::Tensor(Tensor::new(values, vec![rows, 1]).unwrap())],
        placement: None,
        runtime: None,
    }
}

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
fn actual_host_caller_reaches_typed_config_placement_preparation_and_submission() {
    runmat_gc::gc_test_context(|| {
        let provider = TypedCallerProvider::registered(1 << 20);
        let _selected = ThreadProviderGuard::set(Some(provider));
        let plan = sum_plan();
        let outcome = futures::executor::block_on(execute_reduction_with_shape_placed_async(
            host_request(&plan, vec![1.0, 2.0]),
            2,
            1,
            &[1, 1],
            8,
        ))
        .expect("typed reduction caller");
        let FusionExecutionOutcome::Executed(Value::GpuTensor(output)) = outcome else {
            panic!("typed provider route must execute")
        };
        assert_eq!(output.shape(), [1, 1]);
        assert_eq!(output.physical().element_type(), NumericElementType::F64);
        output.allocation_lease().validate_live().unwrap();
        let downloaded = futures::executor::block_on(provider.download_numeric(&output)).unwrap();
        assert_eq!(downloaded.shape, vec![1, 1]);
        assert_eq!(downloaded.data, HostNumericDataOwned::F64(vec![3.0]));
        assert_eq!(provider.configs.load(Ordering::SeqCst), 1);
        assert_eq!(provider.candidates.load(Ordering::SeqCst), 1);
        assert_eq!(provider.feasibility.load(Ordering::SeqCst), 1);
        assert_eq!(
            provider.evidence.load(Ordering::SeqCst),
            2,
            "planning evidence must be revalidated immediately before upload"
        );
        assert_eq!(provider.uploads.load(Ordering::SeqCst), 1);
        assert_eq!(provider.submissions.load(Ordering::SeqCst), 1);
        let probes = provider.probes.lock().unwrap();
        assert_eq!(probes.len(), 2);
        assert_eq!(
            probes[0].retirements(),
            1,
            "prepared upload retires after submit"
        );
        assert_eq!(probes[1].retirements(), 0, "published output remains owned");
        drop(probes);
        drop(output);
        assert_eq!(provider.probes.lock().unwrap()[1].retirements(), 1);
    });
}

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
fn selected_binding_cap_stops_actual_caller_before_candidates_proposals_or_effects() {
    runmat_gc::gc_test_context(|| {
        let provider = TypedCallerProvider::registered(8);
        let _selected = ThreadProviderGuard::set(Some(provider));
        let plan = sum_plan();
        let error = futures::executor::block_on(execute_reduction_with_shape_placed_async(
            host_request(&plan, vec![1.0, 2.0]),
            2,
            1,
            &[1, 1],
            8,
        ))
        .err()
        .expect("16-byte host binding exceeds selected 8-byte cap");
        assert!(!error.is_after_effect());
        assert_eq!(provider.configs.load(Ordering::SeqCst), 1);
        assert_eq!(provider.candidates.load(Ordering::SeqCst), 0);
        assert_eq!(provider.feasibility.load(Ordering::SeqCst), 0);
        assert_eq!(provider.evidence.load(Ordering::SeqCst), 0);
        assert_eq!(provider.uploads.load(Ordering::SeqCst), 0);
        assert_eq!(provider.submissions.load(Ordering::SeqCst), 0);
    });
}

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
fn foreign_resident_actual_caller_stops_before_config_or_any_provider_query() {
    runmat_gc::gc_test_context(|| {
        let provider = TypedCallerProvider::registered(1 << 20);
        let _selected = ThreadProviderGuard::set(Some(provider));
        let foreign = TestAllocationOwner::new()
            .allocate_zeroed(vec![2, 1], NumericElementType::F64, GpuTensorStorage::Real)
            .unwrap();
        let plan = sum_plan();
        let error = futures::executor::block_on(execute_reduction_with_shape_placed_async(
            FusionExecutionRequest {
                plan: &plan,
                inputs: vec![Value::GpuTensor(foreign.handle.clone())],
                placement: None,
                runtime: None,
            },
            2,
            1,
            &[1, 1],
            8,
        ))
        .err()
        .expect("foreign resident source must fail query-free preflight");
        assert!(!error.is_after_effect());
        assert_eq!(provider.configs.load(Ordering::SeqCst), 0);
        assert_eq!(provider.candidates.load(Ordering::SeqCst), 0);
        assert_eq!(provider.feasibility.load(Ordering::SeqCst), 0);
        assert_eq!(provider.evidence.load(Ordering::SeqCst), 0);
        assert_eq!(provider.uploads.load(Ordering::SeqCst), 0);
        assert_eq!(provider.submissions.load(Ordering::SeqCst), 0);
        assert_eq!(foreign.probe.retirements(), 0);
    });
}

use super::*;
#[path = "tests/projection.rs"]
mod projection;
#[path = "tests/raw_constants.rs"]
mod raw_constants;
use crate::fusion::{FusionGroup, FusionGroupPlan, FusionKernelSpec, FusionKind, FusionOp};
use crate::fusion_exec::{
    execute_reduction_with_shape_placed, execute_reduction_with_shape_placed_async,
    FusionExecutionOutcome,
};
use crate::graph::{FusionCallOrigin, InstrSpan, PrimitiveOp, ShapeInfo};
use runmat_accelerate_api::{
    test_support::TestAllocationOwner, AccelProvider, GpuTensorStorage, NumericElementType,
    ProviderFeasibility, ProviderFeasibilityQuery, ProviderRejectionCode, ThreadProviderGuard,
};
use runmat_value::{Tensor, Value};
use std::{
    collections::HashMap,
    sync::atomic::{AtomicUsize, Ordering},
};

struct NoUploadProvider {
    queries: AtomicUsize,
}
impl AccelProvider for NoUploadProvider {
    fn upload(
        &self,
        _: &runmat_accelerate_api::HostTensorView<'_>,
    ) -> anyhow::Result<runmat_accelerate_api::GpuTensorHandle> {
        panic!("arithmetic gate must precede upload")
    }
    fn download<'a>(
        &'a self,
        _: &'a runmat_accelerate_api::GpuTensorHandle,
    ) -> runmat_accelerate_api::AccelDownloadFuture<'a> {
        panic!("arithmetic gate must not download")
    }
    fn precision(&self) -> ProviderPrecision {
        ProviderPrecision::F32
    }
    fn device_info(&self) -> String {
        "reduction-arithmetic-no-upload".into()
    }
    fn query_feasibility(&self, _: &ProviderFeasibilityQuery) -> ProviderFeasibility {
        self.queries.fetch_add(1, Ordering::SeqCst);
        ProviderFeasibility::rejected(
            ProviderRejectionCode::UnsupportedOperation,
            "test-no-upload",
        )
    }
}
static PROVIDER: NoUploadProvider = NoUploadProvider {
    queries: AtomicUsize::new(0),
};

fn plan() -> FusionGroupPlan {
    #[cfg(target_arch = "wasm32")]
    runmat_runtime::builtins::wasm_registry::register_all();
    let mut const_values = HashMap::new();
    // The selected Sum reduces the producer on the first matrix axis.
    const_values.insert(4, Value::Num(1.0));
    FusionGroupPlan {
        index: 0,
        group: FusionGroup {
            id: 0,
            kind: FusionKind::Reduction,
            nodes: vec![],
            shape: ShapeInfo::Scalar,
            span: InstrSpan { start: 0, end: 1 },
            pattern: None,
            stack_layout: None,
        },
        operations: vec![
            FusionOp::Primitive {
                op: PrimitiveOp::ElemPow,
                inputs: vec![0, 1],
                output: Some(2),
            },
            FusionOp::Builtin {
                name: "sum".into(),
                origin: FusionCallOrigin::named("sum"),
                inputs: vec![2, 4],
                output: Some(3),
            },
        ],
        inputs: vec![0, 1],
        stack_pattern: vec![0, 1],
        constants: HashMap::new(),
        const_values,
        materialized_stores: vec![],
        output: Some(3),
        kernel: FusionKernelSpec {
            kind: FusionKind::Reduction,
            supported: true,
        },
        reduction_data: Some(2),
        reduction_dim: Some(4),
        reduction_flavor: Some(runmat_accelerate_api::ReductionFlavor::Sum),
        reduction_axes: None,
        pattern: None,
    }
}

fn request(plan: &FusionGroupPlan, exponent: Value) -> FusionExecutionRequest<'_> {
    FusionExecutionRequest {
        plan,
        inputs: vec![
            Value::Tensor(Tensor::from_f32(vec![1.000_000_1, 0.999_999_94], vec![2, 1]).unwrap()),
            exponent,
        ],
        placement: None,
        runtime: None,
    }
}

#[test]
fn mixed_and_native_single_power_preserve_sum_accumulator_before_provider_placement() {
    runmat_gc::gc_test_context(|| {
        let _selected = ThreadProviderGuard::set(Some(&PROVIDER));
        PROVIDER.queries.store(0, Ordering::SeqCst);
        let source = plan();
        let sync = execute_reduction_with_shape_placed(
            request(&source, Value::Num(2.0)),
            2,
            1,
            &[1, 1],
            64,
        )
        .unwrap();
        assert!(matches!(sync, FusionExecutionOutcome::SharedRuntime(_)));
        assert_eq!(PROVIDER.queries.load(Ordering::SeqCst), 0);
        let mixed = futures::executor::block_on(execute_reduction_with_shape_placed_async(
            request(&source, Value::Num(2.0)),
            2,
            1,
            &[1, 1],
            64,
        ))
        .unwrap();
        assert!(matches!(mixed, FusionExecutionOutcome::SharedRuntime(_)));
        assert_eq!(PROVIDER.queries.load(Ordering::SeqCst), 0);
        let native_exponent = Value::Tensor(Tensor::from_f32(vec![2.0], vec![1, 1]).unwrap());
        let native = request(&source, native_exponent.clone());
        let bindings = AdmittedBindings::new(&native).unwrap();
        assert!(admitted_static(&native, &bindings, ProviderPrecision::F32)
            .unwrap()
            .is_some());
        let result = execute_reduction_with_shape_placed(
            request(&source, native_exponent.clone()),
            2,
            1,
            &[1, 1],
            64,
        )
        .unwrap();
        assert!(matches!(result, FusionExecutionOutcome::SharedRuntime(_)));
        assert_eq!(PROVIDER.queries.load(Ordering::SeqCst), 0);
        let result = futures::executor::block_on(execute_reduction_with_shape_placed_async(
            request(&source, native_exponent),
            2,
            1,
            &[1, 1],
            64,
        ))
        .unwrap();
        assert!(matches!(result, FusionExecutionOutcome::SharedRuntime(_)));
        assert_eq!(PROVIDER.queries.load(Ordering::SeqCst), 0);
    });
}

#[test]
fn synchronous_probe_required_power_stops_before_placement_or_provider_effect() {
    runmat_gc::gc_test_context(|| {
        let _selected = ThreadProviderGuard::set(Some(&PROVIDER));
        PROVIDER.queries.store(0, Ordering::SeqCst);
        let source = plan();
        let allocation = TestAllocationOwner::new()
            .allocate_zeroed(vec![2, 1], NumericElementType::F32, GpuTensorStorage::Real)
            .unwrap();
        let request = FusionExecutionRequest {
            plan: &source,
            inputs: vec![
                Value::GpuTensor(allocation.handle),
                Value::Tensor(Tensor::from_f32(vec![0.5], vec![1, 1]).unwrap()),
            ],
            placement: None,
            runtime: None,
        };
        assert!(matches!(
            execute_reduction_with_shape_placed(request, 2, 1, &[1, 1], 64).unwrap(),
            FusionExecutionOutcome::SharedRuntime(_)
        ));
        assert_eq!(PROVIDER.queries.load(Ordering::SeqCst), 0);
    });
}

#[test]
fn single_multiply_sum_never_uses_a_legacy_single_accumulator() {
    runmat_gc::gc_test_context(|| {
        let _selected = ThreadProviderGuard::set(Some(&PROVIDER));
        PROVIDER.queries.store(0, Ordering::SeqCst);
        let mut source = plan();
        source.operations[0] = FusionOp::Primitive {
            op: PrimitiveOp::ElemMul,
            inputs: vec![0, 1],
            output: Some(2),
        };
        let tensor = || Value::Tensor(Tensor::from_f32(vec![1.0, 2.0], vec![2, 1]).unwrap());
        let request = FusionExecutionRequest {
            plan: &source,
            inputs: vec![tensor(), tensor()],
            placement: None,
            runtime: None,
        };
        assert!(
            futures::executor::block_on(super::super::resolve_reduction_program(
                &request,
                ProviderPrecision::F32,
            ))
            .unwrap()
            .is_none(),
            "F32 ownership cannot supply the canonical Double accumulator"
        );
        let program = futures::executor::block_on(super::super::resolve_reduction_program(
            &request,
            ProviderPrecision::F64,
        ))
        .unwrap()
        .expect("ordinary multiplication has an admitted mathematical contract");
        assert_eq!(
            program.contract().input_class,
            runmat_types::NumericClass::Single
        );
        assert_eq!(
            program.contract().accumulator_class,
            runmat_types::NumericClass::Double
        );
        assert_eq!(
            program.contract().output_class,
            runmat_types::NumericClass::Single
        );
        drop(program);
        let result = futures::executor::block_on(execute_reduction_with_shape_placed_async(
            request,
            2,
            1,
            &[1, 1],
            64,
        ))
        .unwrap();
        assert!(matches!(result, FusionExecutionOutcome::SharedRuntime(_)));
        assert_eq!(PROVIDER.queries.load(Ordering::SeqCst), 0);
    });
}

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
fn selected_native_power_reduction_uses_the_ordinary_helper_for_edge_controls() {
    // This direct fixture bypasses the browser runtime entrypoint; native
    // inventory installs declarations automatically.
    #[cfg(target_arch = "wasm32")]
    runmat_runtime::builtins::wasm_registry::register_all();
    runmat_gc::gc_test_context(|| {
        for exponent in [3.0_f32, -3.0, 0.0] {
            let source = plan();
            let request = FusionExecutionRequest {
                plan: &source,
                inputs: vec![
                    Value::Tensor(
                        Tensor::from_f32(vec![1.000_000_1, -0.0, -2.0, 0.999_999_94], vec![4, 1])
                            .unwrap(),
                    ),
                    Value::Tensor(Tensor::from_f32(vec![exponent], vec![1, 1]).unwrap()),
                ],
                placement: None,
                runtime: None,
            };
            let bindings = AdmittedBindings::new(&request).unwrap();
            let prepared = admitted_static(&request, &bindings, ProviderPrecision::F32)
                .unwrap()
                .expect("integral and zero Single controls are statically real");
            let emission = prepared.emission().expect("selected typed producer");
            let generated = source
                .prepare_reduction_shader(
                    ProviderPrecision::F32,
                    runmat_accelerate_api::ReductionFlavor::Sum,
                    Some(emission),
                )
                .unwrap();
            let reduction = generated.source();
            let PreparedReductionArithmetic::Selected(selected) = &prepared else {
                unreachable!()
            };
            let SelectedReductionArithmetic {
                source: producer,
                program,
                ..
            } = selected.as_ref();
            let ordinary = producer
                .generate_typed_wgsl(
                    &[2],
                    program,
                    runmat_runtime::builtins::common::spec::ScalarType::F32,
                )
                .unwrap();
            assert!(reduction.contains("fn rm_power_f32("));
            assert!(reduction.contains("let typed0: f32 = rm_power_f32("));
            assert!(ordinary.contains("fn rm_power_f32("));
            assert!(ordinary.contains("let typed0: f32 = rm_power_f32("));
            assert!(!reduction.contains("let val: f32 = pow("));
            assert!(reduction.contains("let val: f32 = typed0;"));
            assert!(reduction.contains("let v1 = input1.data[(r * 0u + col * 0u)]"));
            #[cfg(feature = "wgpu")]
            {
                use crate::backend::wgpu::shaders::validation_tests::{
                    validate_generated, RequiredCaps,
                };
                validate_generated(
                    "selected-single-power-reduction",
                    reduction,
                    64,
                    8,
                    RequiredCaps::None,
                );
            }
        }
    });
}

#[test]
fn sealed_reduction_power_rejects_changed_constant_control_before_emission() {
    runmat_gc::gc_test_context(|| {
        let mut source = plan();
        source.inputs = vec![0];
        source.stack_pattern = vec![0];
        source.const_values.insert(
            1,
            Value::Tensor(Tensor::from_f32(vec![0.0], vec![1, 1]).unwrap()),
        );
        let request = FusionExecutionRequest {
            plan: &source,
            inputs: vec![Value::Tensor(
                Tensor::from_f32(vec![-0.0, -2.0], vec![2, 1]).unwrap(),
            )],
            placement: None,
            runtime: None,
        };
        let bindings = AdmittedBindings::new(&request).unwrap();
        let prepared = admitted_static(&request, &bindings, ProviderPrecision::F32)
            .unwrap()
            .expect("selected zero control");
        let generated = source
            .prepare_reduction_shader(
                ProviderPrecision::F32,
                runmat_accelerate_api::ReductionFlavor::Sum,
                prepared.emission(),
            )
            .unwrap();
        let shader = generated.source();
        assert!(shader.contains("let control1 = bitcast<f32>(0x00000000u);"));
        let mut changed = source.clone();
        changed.const_values.insert(
            1,
            Value::Tensor(Tensor::from_f32(vec![-0.0], vec![1, 1]).unwrap()),
        );
        assert!(prepared
            .validate(&changed, &request.inputs, ProviderPrecision::F32, 2, 1)
            .is_err());
        let mut changed_class = source.clone();
        changed_class.const_values.insert(1, Value::Num(0.0));
        assert!(prepared
            .validate(
                &changed_class,
                &request.inputs,
                ProviderPrecision::F32,
                2,
                1
            )
            .is_err());
    });
}

#[test]
fn selected_reduction_projects_general_two_axis_broadcast_from_canonical_layout() {
    runmat_gc::gc_test_context(|| {
        let source = plan();
        let request = FusionExecutionRequest {
            plan: &source,
            inputs: vec![
                Value::Tensor(Tensor::from_f32(vec![1.0, 2.0, 3.0, 4.0], vec![4, 1]).unwrap()),
                Value::Tensor(Tensor::from_f32(vec![2.0, 3.0, 4.0], vec![1, 3]).unwrap()),
            ],
            placement: None,
            runtime: None,
        };
        let bindings = AdmittedBindings::new(&request).unwrap();
        let prepared = admitted_static(&request, &bindings, ProviderPrecision::F32)
            .unwrap()
            .expect("positive broadcast POWER is real");
        prepared
            .validate(&source, &request.inputs, ProviderPrecision::F32, 4, 3)
            .unwrap();
        let generated = source
            .prepare_reduction_shader(
                ProviderPrecision::F32,
                runmat_accelerate_api::ReductionFlavor::Sum,
                prepared.emission(),
            )
            .unwrap();
        let shader = generated.source();
        assert!(shader.contains("let v = input0.data[(r * 1u + col * 0u)]"));
        assert!(shader.contains("let v1 = input1.data[(r * 0u + col * 1u)]"));
        assert!(prepared
            .validate(&source, &request.inputs, ProviderPrecision::F32, 3, 4)
            .is_err());
    });
}

#[test]
fn selected_vector_root_uses_the_existing_row_vector_normalization() {
    runmat_gc::gc_test_context(|| {
        let source = plan();
        let request = FusionExecutionRequest {
            plan: &source,
            inputs: vec![
                Value::Tensor(Tensor::from_f32(vec![1.0, 2.0, 3.0, 4.0], vec![4]).unwrap()),
                Value::Tensor(Tensor::from_f32(vec![2.0], vec![1, 1]).unwrap()),
            ],
            placement: None,
            runtime: None,
        };
        let bindings = AdmittedBindings::new(&request).unwrap();
        let prepared = admitted_static(&request, &bindings, ProviderPrecision::F32)
            .unwrap()
            .unwrap();
        prepared
            .validate(&source, &request.inputs, ProviderPrecision::F32, 1, 4)
            .unwrap();
        let generated = source
            .prepare_reduction_shader(
                ProviderPrecision::F32,
                runmat_accelerate_api::ReductionFlavor::Sum,
                prepared.emission(),
            )
            .unwrap();
        let shader = generated.source();
        assert!(shader.contains("let v = input0.data[(r * 0u + col * 1u)]"));
    });
}

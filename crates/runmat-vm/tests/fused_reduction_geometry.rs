#![cfg(feature = "native-accel")]

#[path = "support/mod.rs"]
mod test_helpers;

use futures::executor::block_on;
use runmat_accelerate::{
    fusion::{prepare_fusion_plan, FusionGroupPlan, FusionOp},
    fusion_exec::{FusionExecutionOutcome, FusionExecutionRequest},
    FusionKind, ReductionAxes, ShapeInfo,
};
use runmat_accelerate_api::{
    AccelDownloadFuture, AccelProvider, GpuTensorHandle, HostTensorOwned, HostTensorView,
    ProviderFeasibility, ProviderFeasibilityQuery, ProviderPrecision, ProviderResourceEstimate,
    ReductionFlavor, ThreadProviderGuard,
};
use runmat_value::{Tensor, Value};
use runmat_vm::accel::fusion::{execute_fusion_reduction, StackSliceGuard};
use std::sync::atomic::{AtomicUsize, Ordering};

// This deterministic provider exercises the actual VM preparation and provider
// boundary. It is not shader/hardware qualification: the hook computes a small
// independent CPU oracle after asserting the requested shader orientation.
struct Provider {
    inner: runmat_accelerate::simple_provider::InProcessProvider,
    uploads: AtomicUsize,
    queries: AtomicUsize,
    reductions: AtomicUsize,
}
impl Provider {
    fn fixture() -> &'static Self {
        Box::leak(Box::new(Self {
            inner: runmat_accelerate::simple_provider::InProcessProvider::new(),
            uploads: AtomicUsize::new(0),
            queries: AtomicUsize::new(0),
            reductions: AtomicUsize::new(0),
        }))
    }
}
impl AccelProvider for Provider {
    fn device_id(&self) -> u32 {
        self.inner.device_id()
    }
    fn precision(&self) -> ProviderPrecision {
        ProviderPrecision::F64
    }
    fn device_info(&self) -> String {
        "deterministic reduction geometry fixture".into()
    }
    fn upload(&self, value: &HostTensorView<'_>) -> anyhow::Result<GpuTensorHandle> {
        self.uploads.fetch_add(1, Ordering::SeqCst);
        self.inner.upload(value)
    }
    fn download<'a>(&'a self, handle: &'a GpuTensorHandle) -> AccelDownloadFuture<'a> {
        self.inner.download(handle)
    }
    fn query_feasibility(&self, query: &ProviderFeasibilityQuery) -> ProviderFeasibility {
        self.queries.fetch_add(1, Ordering::SeqCst);
        assert_eq!(query.operation.0, "fusion.reduction");
        let bytes = query.outputs.iter().try_fold(0u64, |total, output| {
            total.checked_add(output.checked_element_count()?.checked_mul(8)?)
        });
        ProviderFeasibility::supported(ProviderResourceEstimate {
            output_bytes: bytes,
            ..Default::default()
        })
    }
    fn fused_reduction(
        &self,
        shader: &str,
        inputs: &[GpuTensorHandle],
        output_shape: &[usize],
        reduce_len: usize,
        num_slices: usize,
        _: u32,
        flavor: ReductionFlavor,
    ) -> anyhow::Result<GpuTensorHandle> {
        self.reductions.fetch_add(1, Ordering::SeqCst);
        let row_axis = shader.contains("let row = wid.x");
        let mut operands = Vec::new();
        for input in inputs {
            operands.push(block_on(self.inner.download(input))?);
        }
        assert!(!operands.is_empty());
        assert!(operands.len() <= 2);
        if operands.len() == 2 {
            assert!(
                shader.contains("v + v1"),
                "fixture only implements the requested addition expression"
            );
        }
        let mut data = Vec::with_capacity(num_slices);
        for slice in 0..num_slices {
            let mut sum = 0.;
            for element in 0..reduce_len {
                let index = if row_axis {
                    slice + element * num_slices
                } else {
                    element + slice * reduce_len
                };
                for operand in &operands {
                    sum += operand.data[index];
                }
            }
            data.push(if matches!(flavor, ReductionFlavor::Mean) {
                sum / reduce_len as f64
            } else {
                sum
            });
        }
        self.inner.upload(&HostTensorView {
            data: &data,
            shape: output_shape,
        })
    }
}

fn compiled_plan() -> FusionGroupPlan {
    // The compiler requires a genuine multi-operation fusion candidate; rand
    // is an effectful boundary, not a second numerical signal for bare sum.
    let bytecode = test_helpers::compile_source("A=rand(2,3); y=sum(A+1,1);").unwrap();
    let groups = bytecode.runtime_fusion_groups();
    let graph = bytecode
        .runtime_accel_graph_for_fusion(&groups)
        .expect("compiled reduction graph");
    let groups = bytecode.runtime_fusion_groups_for_graph(&graph);
    let plan = prepare_fusion_plan(
        Some(&graph),
        &groups,
        bytecode.fusion_metadata.mir_fusion_candidate_group_count,
    )
    .expect("compiled reduction plan");
    let plan = plan
        .groups
        .iter()
        .find(|group| group.group.kind == FusionKind::Reduction)
        .expect("mandatory reduction group")
        .clone();
    assert!(plan.kernel.supported);
    plan
}

fn request_plan(shape: &[usize], axis: ReductionAxes, add: bool) -> FusionGroupPlan {
    let mut plan = compiled_plan();
    let data = plan.reduction_data.unwrap();
    // Request a reduction over an already materialized data value. Discard its
    // former producer; no unreachable operation may retain unbound operands.
    plan.operations.retain(
        |operation| matches!(operation, FusionOp::Builtin { output, .. } if *output == plan.output),
    );
    assert_eq!(plan.operations.len(), 1, "one compiled reduction binding");
    let FusionOp::Builtin {
        inputs: reduction_inputs,
        ..
    } = &mut plan.operations[0]
    else {
        unreachable!()
    };
    assert_eq!(reduction_inputs.first(), Some(&data));
    reduction_inputs.clear();
    reduction_inputs.push(data);
    plan.inputs = vec![data];
    plan.const_values.clear();
    plan.constants.clear();
    plan.reduction_dim = None;
    let dimension = data.max(plan.output.unwrap()) + 100;
    match &axis {
        ReductionAxes::Explicit(dims) => {
            assert_eq!(dims.len(), 1);
            plan.reduction_dim = Some(dimension);
            plan.const_values
                .insert(dimension, Value::Num(dims[0] as f64));
            reduction_inputs.push(dimension);
        }
        ReductionAxes::All => {
            plan.const_values
                .insert(dimension, Value::String("all".into()));
            reduction_inputs.push(dimension);
        }
        ReductionAxes::Unspecified => {}
    }
    plan.reduction_axes = Some(axis);
    // Deliberately stale cached shape must not decide actual runtime geometry.
    plan.group.shape = ShapeInfo::Tensor(vec![Some(99), Some(99)]);
    if add {
        let right = plan
            .operations
            .iter()
            .flat_map(|op| match op {
                FusionOp::Primitive { inputs, .. } | FusionOp::Builtin { inputs, .. } => inputs,
            })
            .copied()
            .max()
            .unwrap_or(data)
            + 100;
        let output = right + 1;
        let FusionOp::Builtin { inputs, .. } = &mut plan.operations[0] else {
            unreachable!()
        };
        inputs[0] = output;
        plan.inputs.push(right);
        plan.operations.insert(
            0,
            FusionOp::Primitive {
                op: runmat_accelerate::graph::PrimitiveOp::Add,
                inputs: vec![data, right],
                output: Some(output),
            },
        );
        plan.reduction_data = Some(output);
    }
    assert!(!shape.is_empty());
    plan
}

fn tensor(shape: &[usize]) -> Value {
    let count = if shape.contains(&0) {
        0
    } else {
        shape.iter().product()
    };
    Value::Tensor(Tensor::new((1..=count).map(|n| n as f64).collect(), shape.to_vec()).unwrap())
}

#[test]
fn forced_reduction_executes_actual_geometry_and_keeps_cached_plan_unchanged() {
    runmat_gc::gc_test_context(|| {
        let provider = Provider::fixture();
        let _scope = ThreadProviderGuard::set(Some(provider));
        for (shape, axes, add, expected_shape, expected) in [
            (
                vec![2, 3],
                ReductionAxes::Explicit(vec![1]),
                false,
                vec![1, 3],
                vec![3., 7., 11.],
            ),
            (
                vec![2, 3],
                ReductionAxes::Explicit(vec![2]),
                false,
                vec![2, 1],
                vec![9., 12.],
            ),
            (
                vec![1, 3],
                ReductionAxes::Unspecified,
                false,
                vec![1, 1],
                vec![6.],
            ),
            (
                vec![1, 3],
                ReductionAxes::Explicit(vec![2]),
                false,
                vec![1, 1],
                vec![6.],
            ),
            (
                vec![2, 2, 2],
                ReductionAxes::All,
                false,
                vec![1, 1],
                vec![36.],
            ),
            (
                vec![2, 3],
                ReductionAxes::Explicit(vec![1]),
                true,
                vec![1, 3],
                vec![6., 14., 22.],
            ),
        ] {
            let plan = request_plan(&shape, axes, add);
            let original_dimension = plan.reduction_dim;
            let original_constants = plan.const_values.clone();
            let input = tensor(&shape);
            let inputs = if add {
                vec![input.clone(), input]
            } else {
                vec![input]
            };
            let mut stack = inputs.clone();
            let guard = StackSliceGuard::new(&mut stack, 0);
            let before = provider.reductions.load(Ordering::SeqCst);
            let output = execute_fusion_reduction(
                &plan,
                FusionExecutionRequest {
                    plan: &plan,
                    inputs,
                    placement: None,
                    runtime: None,
                },
                guard,
            )
            .unwrap();
            let FusionExecutionOutcome::Executed(Value::GpuTensor(output)) = output else {
                panic!("supported direct route must execute")
            };
            let HostTensorOwned { shape, data, .. } = block_on(provider.download(&output)).unwrap();
            assert_eq!(shape, expected_shape);
            assert_eq!(data, expected);
            assert_eq!(provider.reductions.load(Ordering::SeqCst), before + 1);
            assert!(stack.is_empty());
            assert_eq!(plan.reduction_dim, original_dimension);
            assert_eq!(plan.const_values, original_constants);
        }
    });
}

#[test]
fn forced_invalid_geometry_restores_stack_without_provider_queries_or_uploads() {
    runmat_gc::gc_test_context(|| {
        let provider = Provider::fixture();
        let _scope = ThreadProviderGuard::set(Some(provider));
        for (shape, axes) in [
            (vec![0, 3], ReductionAxes::All),
            (vec![3, 0], ReductionAxes::Explicit(vec![2])),
            (vec![2, 2, 2], ReductionAxes::Explicit(vec![1])),
            (vec![2, 3], ReductionAxes::Explicit(vec![0])),
            (vec![2, 3], ReductionAxes::Explicit(vec![3])),
        ] {
            let plan = request_plan(&shape, axes, false);
            let inputs = vec![tensor(&shape)];
            let mut stack = inputs.clone();
            let guard = StackSliceGuard::new(&mut stack, 0);
            assert!(execute_fusion_reduction(
                &plan,
                FusionExecutionRequest {
                    plan: &plan,
                    inputs: inputs.clone(),
                    placement: None,
                    runtime: None
                },
                guard
            )
            .is_err());
            assert_eq!(stack, inputs);
        }
        let plan = request_plan(&[2, 3], ReductionAxes::All, true);
        let inputs = vec![tensor(&[2, 1]), tensor(&[1, 3])];
        let mut stack = inputs.clone();
        let guard = StackSliceGuard::new(&mut stack, 0);
        assert!(execute_fusion_reduction(
            &plan,
            FusionExecutionRequest {
                plan: &plan,
                inputs: inputs.clone(),
                placement: None,
                runtime: None
            },
            guard
        )
        .is_err());
        assert_eq!(stack, inputs);
        for dimension in [1.5, f64::NAN, f64::INFINITY] {
            let mut plan = request_plan(&[2, 3], ReductionAxes::Unspecified, false);
            let id = plan.inputs.iter().copied().max().unwrap() + 100;
            plan.reduction_dim = Some(id);
            plan.const_values.insert(id, Value::Num(dimension));
            let inputs = vec![tensor(&[2, 3])];
            let mut stack = inputs.clone();
            let guard = StackSliceGuard::new(&mut stack, 0);
            assert!(execute_fusion_reduction(
                &plan,
                FusionExecutionRequest {
                    plan: &plan,
                    inputs: inputs.clone(),
                    placement: None,
                    runtime: None,
                },
                guard
            )
            .is_err());
            assert_eq!(stack, inputs);
        }
        assert_eq!(provider.queries.load(Ordering::SeqCst), 0);
        assert_eq!(provider.uploads.load(Ordering::SeqCst), 0);
        assert_eq!(provider.reductions.load(Ordering::SeqCst), 0);
    });
}

#[test]
fn interpreter_keeps_empty_nd_broadcast_and_dimension_error_semantics() {
    runmat_gc::gc_test_context(|| {
        let _scope = ThreadProviderGuard::set(None);
        for (source, expected_shape, expected) in [
            ("A=zeros(0,3); y=sum(A,1);", vec![1, 3], vec![0., 0., 0.]),
            ("A=zeros(3,0); y=sum(A,2);", vec![3, 1], vec![0., 0., 0.]),
            (
                "A=reshape(1:8,2,2,2); y=sum(A+0,'all');",
                vec![1, 1],
                vec![36.],
            ),
            (
                "A=[1;2]; B=[1,2,3]; y=sum(A+B,'all');",
                vec![1, 1],
                vec![21.],
            ),
        ] {
            let bytecode = test_helpers::compile_source(source).unwrap();
            let groups = bytecode.runtime_fusion_groups();
            assert!(
                groups
                    .iter()
                    .any(|group| group.kind == FusionKind::Reduction),
                "required reduction candidate: {source}"
            );
            let result = test_helpers::interpret(&bytecode).unwrap();
            let value = result.last().expect("last assignment");
            match value {
                Value::Num(number) => {
                    assert_eq!(expected_shape, [1, 1]);
                    assert_eq!(expected, [*number]);
                }
                Value::Tensor(tensor) => {
                    assert_eq!(tensor.shape, expected_shape);
                    assert_eq!(tensor.materialize_f64(), expected);
                }
                _ => panic!("numeric interpreter result: {value:?}"),
            }
        }
        let result = test_helpers::execute_source("A=zeros(0,3); y=mean(A,'all');").unwrap();
        assert!(
            matches!(result.last(), Some(Value::Num(number)) if number.is_nan())
                || matches!(result.last(), Some(Value::Tensor(tensor)) if tensor.len() == 1 && tensor.materialize_f64()[0].is_nan())
        );
        assert!(test_helpers::execute_source("A=ones(2,3); y=sum(A,1.5);").is_err());
    });
}

#![cfg(feature = "wgpu")]

use runmat_accelerate::backend::wgpu::provider_impl::WgpuProviderOptions;
use runmat_accelerate::fusion::{
    detect_fusion_groups, FusionGroupPlan, FusionKernelSpec, FusionKind, FusionOp,
};
use runmat_accelerate::fusion_exec::{execute_reduction, FusionExecutionRequest};
use runmat_accelerate::graph::{
    AccelGraph, AccelGraphTag, AccelNode, AccelNodeLabel, AccelOpCategory, InstrSpan, PrimitiveOp,
    ShapeInfo, ValueId, ValueInfo, ValueOrigin, VarBinding, VarKind,
};
use runmat_accelerate_api::{AccelProvider, GpuTensorHandle, HostTensorView};
use runmat_builtins::{Type, Value};
use std::collections::HashMap;

fn upload(
    provider: &runmat_accelerate::backend::wgpu::provider_impl::WgpuProvider,
    rows: usize,
    cols: usize,
    data: &[f64],
) -> GpuTensorHandle {
    provider
        .upload(&HostTensorView {
            data,
            shape: &[rows, cols],
        })
        .expect("upload")
}

#[test]
fn fused_mean_all_codegen_and_exec_on_vector() {
    // Ensure provider is registered
    let provider = runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
        WgpuProviderOptions::default(),
    )
    .expect("wgpu");

    // Create a simple 1D vector as 1 x N
    let n = 64usize;
    let rows = n;
    let cols = 1usize;
    let mut data = vec![0.0f64; n];
    for i in 0..n {
        data[i] = (i as f64 + 1.0) * 0.01;
    }
    let e = upload(&provider, rows, cols, &data);

    // Graph: sq = e .* e; mse = mean(sq, 'all')
    let v_e: ValueId = 0;
    let v_sq: ValueId = 1;
    let v_all: ValueId = 2;
    let v_mse: ValueId = 3;

    let values = vec![
        ValueInfo {
            id: v_e,
            origin: ValueOrigin::Variable {
                kind: VarKind::Global,
                index: 0,
            },
            ty: Type::tensor(),
            shape: ShapeInfo::Tensor(vec![Some(rows), Some(cols)]),
            constant: None,
        },
        ValueInfo {
            id: v_sq,
            origin: ValueOrigin::NodeOutput { node: 0, output: 0 },
            ty: Type::tensor(),
            shape: ShapeInfo::Tensor(vec![Some(rows), Some(cols)]),
            constant: None,
        },
        ValueInfo {
            id: v_all,
            origin: ValueOrigin::Constant,
            ty: Type::String,
            shape: ShapeInfo::Scalar,
            constant: Some(Value::String("all".to_string())),
        },
        ValueInfo {
            id: v_mse,
            origin: ValueOrigin::NodeOutput { node: 1, output: 0 },
            ty: Type::tensor(),
            shape: ShapeInfo::Tensor(vec![Some(1)]),
            constant: None,
        },
    ];
    let node_sq = AccelNode {
        id: 0,
        label: AccelNodeLabel::Primitive(PrimitiveOp::ElemMul),
        category: AccelOpCategory::Elementwise,
        inputs: vec![v_e, v_e],
        outputs: vec![v_sq],
        span: InstrSpan { start: 0, end: 0 },
        tags: vec![AccelGraphTag::Elementwise],
    };
    let node_mean = AccelNode {
        id: 1,
        label: AccelNodeLabel::Builtin {
            name: "mean".into(),
        },
        category: AccelOpCategory::Reduction,
        inputs: vec![v_sq, v_all],
        outputs: vec![v_mse],
        span: InstrSpan { start: 1, end: 1 },
        tags: vec![AccelGraphTag::Reduction],
    };

    let mut var_bindings = HashMap::new();
    var_bindings.insert(
        v_e,
        VarBinding {
            kind: VarKind::Global,
            index: 0,
        },
    );
    let graph = AccelGraph {
        nodes: vec![node_sq, node_mean],
        values,
        var_bindings,
    };

    // Detect and build plan
    let groups = detect_fusion_groups(&graph);
    let red_group = groups.iter().find(|g| g.kind.is_reduction()).unwrap().clone();
    let plan = FusionGroupPlan {
        index: red_group.id,
        group: red_group.clone(),
        operations: vec![
            FusionOp::Primitive {
                op: PrimitiveOp::ElemMul,
                inputs: vec![v_e, v_e],
                output: Some(v_sq),
            },
            FusionOp::Builtin {
                name: "mean".to_string(),
                inputs: vec![v_sq, v_all],
                output: Some(v_mse),
            },
        ],
        inputs: vec![v_e],
        stack_pattern: vec![],
        constants: HashMap::new(),
        const_values: {
            let mut cv = HashMap::new();
            cv.insert(v_all, Value::String("all".to_string()));
            cv
        },
        output: Some(v_mse),
        kernel: FusionKernelSpec {
            kind: FusionKind::Reduction,
            supported: true,
        },
        reduction_data: Some(v_sq),
        reduction_dim: None,
        reduction_mode: Some(runmat_accelerate::fusion::ReductionMode::Mean),
        pattern: None,
    };

    // Codegen should succeed
    assert!(
        plan.generate_reduction_wgsl("f32").is_some(),
        "mean(all) reduction should generate WGSL"
    );

    // Execute and ensure scalar-shaped GPU tensor is returned
    let req = FusionExecutionRequest {
        plan: &plan,
        inputs: vec![Value::GpuTensor(e)],
    };
    let out = execute_reduction(req, rows, 1, 0).expect("execute fused mean(all)");
    match out {
        Value::GpuTensor(h) => {
            assert_eq!(h.shape, vec![1], "mean(all) should be scalar [1]");
        }
        _ => panic!("expected GPU tensor result"),
    }
}



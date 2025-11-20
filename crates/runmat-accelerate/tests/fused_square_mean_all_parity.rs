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
use runmat_accelerate::ReductionAxes;
use runmat_accelerate_api::{AccelProvider, HostTensorView, ReductionFlavor};
use runmat_builtins::{Type, Value};
use std::collections::HashMap;

#[test]
fn fused_square_mean_all_matches_cpu() {
    let provider = runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
        WgpuProviderOptions::default(),
    )
    .expect("wgpu");

    // Build simple input
    let n = 1024usize;
    let mut data_h = vec![0.0f64; n];
    for (i, value) in data_h.iter_mut().enumerate().take(n) {
        *value = (i as f64 * 0.001).sin();
    }
    let x = provider
        .upload(&HostTensorView {
            data: &data_h,
            shape: &[n, 1],
        })
        .expect("upload");

    // Graph: sq = x .* x; mean = mean(sq, 'all')
    let v_x: ValueId = 0;
    let v_sq: ValueId = 1;
    let v_all: ValueId = 2;
    let v_mean: ValueId = 3;
    let values = vec![
        ValueInfo {
            id: v_x,
            origin: ValueOrigin::Variable {
                kind: VarKind::Global,
                index: 0,
            },
            ty: Type::tensor(),
            shape: ShapeInfo::Tensor(vec![Some(n), Some(1)]),
            constant: None,
        },
        ValueInfo {
            id: v_sq,
            origin: ValueOrigin::Unknown,
            ty: Type::tensor(),
            shape: ShapeInfo::Tensor(vec![Some(n), Some(1)]),
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
            id: v_mean,
            origin: ValueOrigin::Unknown,
            ty: Type::tensor(),
            shape: ShapeInfo::Scalar,
            constant: None,
        },
    ];
    let pow_node = AccelNode {
        id: 0,
        label: AccelNodeLabel::Primitive(PrimitiveOp::ElemMul),
        category: AccelOpCategory::Elementwise,
        inputs: vec![v_x, v_x],
        outputs: vec![v_sq],
        span: InstrSpan { start: 0, end: 0 },
        tags: vec![AccelGraphTag::Elementwise],
    };
    let mean_node = AccelNode {
        id: 1,
        label: AccelNodeLabel::Builtin {
            name: "mean".into(),
        },
        category: AccelOpCategory::Reduction,
        inputs: vec![v_sq, v_all],
        outputs: vec![v_mean],
        span: InstrSpan { start: 1, end: 1 },
        tags: vec![AccelGraphTag::Reduction],
    };
    let mut var_bindings = HashMap::new();
    var_bindings.insert(
        v_x,
        VarBinding {
            kind: VarKind::Global,
            index: 0,
        },
    );
    let graph = AccelGraph {
        nodes: vec![pow_node, mean_node],
        values,
        var_bindings,
        node_bindings: HashMap::new(),
    };
    let groups = detect_fusion_groups(&graph);
    let red_group = groups
        .iter()
        .find(|g| g.kind.is_reduction())
        .unwrap()
        .clone();

    let plan = FusionGroupPlan {
        index: red_group.id,
        group: red_group.clone(),
        operations: vec![
            FusionOp::Primitive {
                op: PrimitiveOp::ElemMul,
                inputs: vec![v_x, v_x],
                output: Some(v_sq),
            },
            FusionOp::Builtin {
                name: "mean".to_string(),
                inputs: vec![v_sq, v_all],
                output: Some(v_mean),
            },
        ],
        inputs: vec![v_x],
        stack_pattern: vec![],
        constants: HashMap::new(),
        const_values: {
            let mut cv = HashMap::new();
            cv.insert(v_all, Value::String("all".to_string()));
            cv
        },
        output: Some(v_mean),
        kernel: FusionKernelSpec {
            kind: FusionKind::Reduction,
            supported: true,
        },
        reduction_data: Some(v_sq),
        reduction_dim: None,
        reduction_flavor: Some(ReductionFlavor::Mean),
        reduction_axes: Some(ReductionAxes::All),
        pattern: None,
    };

    let req = FusionExecutionRequest {
        plan: &plan,
        inputs: vec![runmat_builtins::Value::GpuTensor(x.clone())],
    };
    let result = execute_reduction(req, n, 1, 0).expect("fused mean(all) of square");
    let gpu_scalar = match result {
        runmat_builtins::Value::GpuTensor(h) => {
            let host = provider.download(&h).expect("download");
            host.data[0]
        }
        _ => panic!("expected GpuTensor result"),
    };

    let expected = data_h.iter().map(|v| v * v).sum::<f64>() / (n as f64);
    assert!(
        (gpu_scalar - expected).abs() < 1e-6,
        "gpu={} expected={}",
        gpu_scalar,
        expected
    );
}

#![cfg(feature = "wgpu")]
#[path = "support/direct_reduction_runtime.rs"]
mod invocation;
#[path = "support/selected_reduction.rs"]
mod selected_reduction;

use runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions;
use runmat_accelerate::fusion::{
    detect_fusion_groups, FusionGroupPlan, FusionKernelSpec, FusionKind, FusionOp,
};
use runmat_accelerate::fusion_exec::FusionExecutionRequest;
use runmat_accelerate::graph::{
    AccelGraph, AccelGraphTag, AccelNode, AccelNodeLabel, AccelOpCategory, InstrSpan, PrimitiveOp,
    ShapeInfo, ValueId, ValueInfo, ValueOrigin, VarBinding, VarKind,
};
use runmat_accelerate::ReductionAxes;
use runmat_accelerate_api::{AccelProvider, HostTensorView, ReductionFlavor};
use runmat_builtins::Type;
use runmat_value::Value;
use std::collections::HashMap;

#[tokio::test]
async fn fused_square_mean_all_matches_cpu() {
    let runtime = invocation::runtime();
    let _invocation = runtime.enter();
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
            constant: Some(Value::String("all".into())),
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
            origin: Default::default(),
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
                origin: Default::default(),
                name: "mean".into(),
                inputs: vec![v_sq, v_all],
                output: Some(v_mean),
            },
        ],
        inputs: vec![v_x],
        stack_pattern: vec![],
        constants: HashMap::new(),
        const_values: {
            let mut cv = HashMap::new();
            cv.insert(v_all, Value::String("all".into()));
            cv
        },
        materialized_stores: Vec::new(),
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
        inputs: vec![runmat_value::Value::GpuTensor(x.clone())],
        placement: None,
        runtime: Some(runtime.clone()),
    };
    let Some(result) = selected_reduction::execute(req, n, 1, &[1, 1], 0)
        .await
        .expect("canonical fused mean(all) admission")
    else {
        return;
    };
    let gpu_scalar = match result {
        runmat_value::Value::GpuTensor(h) => {
            let host = AccelProvider::download(provider, &h)
                .await
                .expect("download");
            assert_eq!(host.shape, vec![1, 1]);
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

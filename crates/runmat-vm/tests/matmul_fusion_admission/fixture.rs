//! Start from a real compiled matrix group, then exercise an explicitly requested
//! whole epilogue. This does not assume the compiler groups every outer function.
use super::*;
use runmat_accelerate::{
    fusion::{FusionGroupPlan, FusionOp},
    graph::{PrimitiveOp, ValueId},
};

pub(super) fn requested(expression: &str, left: Option<Value>) -> (FusionGroupPlan, Vec<Value>) {
    let bytecode = test_helpers::compile_source("A=rand(2,2); B=rand(2,2); y=(A*B)+2;").unwrap();
    let windows = bytecode.runtime_fusion_groups();
    let graph = bytecode
        .runtime_accel_graph_for_fusion(&windows)
        .expect("matrix plus scalar graph");
    let groups = bytecode.runtime_fusion_groups_for_graph(&graph);
    let plan = prepare_fusion_plan(
        Some(&graph),
        &groups,
        bytecode.fusion_metadata.mir_fusion_candidate_group_count,
    )
    .unwrap();
    let mut plan = plan
        .groups
        .iter()
        .find(|g| g.group.kind == FusionKind::MatmulEpilogue)
        .expect("matrix plus scalar group")
        .clone();
    let root = plan.operations[0].clone();
    let FusionOp::Builtin {
        inputs,
        output: Some(product),
        ..
    } = &root
    else {
        panic!("matrix root")
    };
    assert_eq!(inputs.len(), 2);
    let mut inputs = vec![
        (
            inputs[0],
            left.unwrap_or_else(|| tensor(&[1., 3., 2., 4.], &[2, 2])),
        ),
        (inputs[1], tensor(&[2., 0., 0., 2.], &[2, 2])),
    ];
    let mut current = *product;
    let mut operations = vec![root];
    let mut next: ValueId = 10000;
    let mut append =
        |primitive: Option<PrimitiveOp>, name: &str, other: Option<Value>, reverse: bool| {
            let mut operands = vec![current];
            if let Some(value) = other {
                inputs.push((next, value));
                operands.push(next);
                next += 1;
                if reverse {
                    operands.reverse();
                }
            }
            let output = next;
            next += 1;
            operations.push(match primitive {
                Some(op) => FusionOp::Primitive {
                    op,
                    inputs: operands,
                    output: Some(output),
                },
                None => FusionOp::Builtin {
                    origin: Default::default(),
                    name: name.into(),
                    inputs: operands,
                    output: Some(output),
                },
            });
            current = output;
        };
    use PrimitiveOp::*;
    match expression {
        "A*B" => {}
        "2-(A*B)" => append(Some(Sub), "", Some(Value::Num(2.)), true),
        "(A*B)-2" => append(Some(Sub), "", Some(Value::Num(2.)), false),
        "(A*B)*3+2" => {
            append(Some(ElemMul), "", Some(Value::Num(3.)), false);
            append(Some(Add), "", Some(Value::Num(2.)), false);
        }
        "((A*B).*r)./c" => {
            append(Some(ElemMul), "", Some(tensor(&[2., 4.], &[2, 1])), false);
            append(Some(ElemDiv), "", Some(tensor(&[2., 4.], &[1, 2])), false);
        }
        "sin(A*B)" => append(None, "sin", None, false),
        "(A*B)+C" => append(Some(Add), "", Some(tensor(&[1.; 4], &[2, 2])), false),
        "((A*B)+2)*3" => {
            append(Some(Add), "", Some(Value::Num(2.)), false);
            append(Some(ElemMul), "", Some(Value::Num(3.)), false);
        }
        "2./(A*B)" => append(Some(ElemDiv), "", Some(Value::Num(2.)), true),
        "((A*B).*c).*r" => {
            append(Some(ElemMul), "", Some(tensor(&[2., 4.], &[1, 2])), false);
            append(Some(ElemMul), "", Some(tensor(&[2., 4.], &[2, 1])), false);
        }
        "(A*B).^0.5" => append(Some(ElemPow), "", Some(Value::Num(0.5)), false),
        "min(max(A*B,3),7)" => {
            append(None, "max", Some(Value::Num(3.)), false);
            append(None, "min", Some(Value::Num(7.)), false);
        }
        "max(min(A*B,7),3)" => {
            append(None, "min", Some(Value::Num(7.)), false);
            append(None, "max", Some(Value::Num(3.)), false);
        }
        "(A*B).^2" => append(Some(ElemPow), "", Some(Value::Num(2.)), false),
        "(A*B).^0" => append(Some(ElemPow), "", Some(Value::Num(0.)), false),
        "-0-(A*B)" => append(Some(Sub), "", Some(Value::Num(-0.)), true),
        "max(A*B,0).^0.5" => {
            append(None, "max", Some(Value::Num(0.)), false);
            append(Some(ElemPow), "", Some(Value::Num(0.5)), false);
        }
        "diag(A*B)" => append(None, "diag", None, false),
        _ => panic!("unknown fixture {expression}"),
    }
    plan.operations = operations;
    plan.inputs = inputs.iter().map(|(id, _)| *id).collect();
    plan.output = Some(current);
    plan.constants.clear();
    plan.const_values.clear();
    (plan, inputs.into_iter().map(|(_, value)| value).collect())
}

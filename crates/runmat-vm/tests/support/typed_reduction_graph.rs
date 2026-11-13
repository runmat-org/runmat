//! Compiler-owned folded-group checks shared by native and browser VM tests.
use runmat_accelerate::graph::{AccelGraphTag, AccelNodeLabel, PrimitiveOp};
use runmat_accelerate::FusionKind;
use runmat_value::Value;

pub(crate) fn assert_compiled_folded_graph(bytecode: &runmat_vm::Bytecode, expected_adds: usize) {
    let (graph, groups) = bytecode
        .reconciled_runtime_fusion_graph()
        .expect("compiled source must produce a VM-reconciled graph");
    assert!(
        groups.iter().any(|group| {
            if group.kind != FusionKind::Reduction || group.nodes.len() != expected_adds + 1 {
                return false;
            }
            let Some(nodes) = group
                .nodes
                .iter()
                .map(|id| graph.node(*id))
                .collect::<Option<Vec<_>>>()
            else {
                return false;
            };
            let contained = |node: &runmat_accelerate::graph::AccelNode| {
                node.span.start >= group.span.start && node.span.end <= group.span.end
            };
            if !nodes.iter().all(|node| contained(node))
                || nodes
                    .iter()
                    .filter(|node| {
                        matches!(&node.label, AccelNodeLabel::Primitive(PrimitiveOp::Add))
                    })
                    .count()
                    != expected_adds
                || nodes.iter().filter(|node| node.is_reduction()).count() != 1
            {
                return false;
            }
            let root = nodes
                .iter()
                .find(|node| node.is_reduction())
                .expect("one selected reduction");
            let omitted = graph
                .nodes
                .iter()
                .filter(|node| contained(node) && !group.nodes.contains(&node.id))
                .collect::<Vec<_>>();
            let Some(span_instructions) = bytecode.instructions.get(group.span.start..=group.span.end)
            else {
                return false;
            };
            let empty_control_instructions = span_instructions
                .iter()
                .filter(|instruction| {
                    matches!(instruction, &&runmat_vm::Instr::CreateMatrix(0, 0))
                })
                .count();
            omitted.len() == empty_control_instructions
                && omitted.iter().all(|control| {
                    let [output] = control.outputs.as_slice() else {
                        return false;
                    };
                    control.tags.as_slice() == [AccelGraphTag::FoldedLiteralArray]
                        && control.inputs.is_empty()
                        && root.inputs.contains(output)
                        && graph.var_binding(*output).is_none()
                        && graph.node_binding(control.id).is_none()
                        && graph.value(*output).is_some_and(|value| {
                            matches!(&value.constant, Some(Value::Tensor(tensor))
                                if tensor.shape() == [0, 0] && tensor.is_empty())
                        })
                })
        }),
        "compiled source must reconcile {expected_adds} Adds and one reduction (with only certified empty controls outside shader nodes); actual groups: {:?}; instructions: {:?}; graph nodes: {:?}",
        groups,
        bytecode.instructions,
        graph.nodes
    );
}

pub(crate) fn assert_compiled_function_graph(
    bytecode: &runmat_vm::Bytecode,
    function_name: &str,
    expected_adds: usize,
) {
    let registry = bytecode.function_registry();
    let function_id = registry
        .resolve_name(function_name)
        .expect("compiled function must be retained in its registry");
    let function = registry
        .get(function_id)
        .expect("retained compiled function body");
    let mut function_view = runmat_vm::Bytecode::for_function(
        function,
        registry.clone(),
        bytecode.layout.clone().expect("compiler-owned VM layout"),
    );
    // The ordinary runner consumes execution_bytecode without this optional
    // entrypoint layout; the function graph cannot borrow its slot authority.
    function_view.layout = None;
    assert_compiled_folded_graph(&function_view, expected_adds);
}

pub(crate) fn expected_column_min(scale: f64) -> Vec<f64> {
    (0..128)
        .map(|column| scale * f64::from(1 + 128 * column))
        .collect()
}

pub(crate) fn expected_row_min(scale: f64) -> Vec<f64> {
    (0..128).map(|row| scale * f64::from(row + 1)).collect()
}

pub(crate) fn expected_third_axis_min(scale: f64) -> Vec<f64> {
    (0..256).map(|index| scale * f64::from(index + 1)).collect()
}

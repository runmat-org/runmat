use std::collections::{BTreeMap, BTreeSet};

use runmat_hir::FunctionId;

use crate::{MirAssembly, MirCallee, MirRvalue, MirStmtKind};

pub(crate) fn externally_reachable_functions(assembly: &MirAssembly) -> BTreeSet<FunctionId> {
    let mut reachable = assembly
        .entrypoints
        .iter()
        .copied()
        .collect::<BTreeSet<_>>();
    reachable.extend(
        assembly
            .functions
            .iter()
            .filter_map(|(function, metadata)| {
                (metadata.parent.is_none()
                    || matches!(metadata.kind, runmat_hir::FunctionKind::ClassMethod { .. }))
                .then_some(*function)
            }),
    );
    for body in assembly.bodies.values() {
        for statement in body.blocks.iter().flat_map(|block| block.statements.iter()) {
            collect_statement_handles(&statement.kind, &mut reachable);
        }
    }
    reachable
}

fn collect_escaped_handles(value: &MirRvalue, reachable: &mut BTreeSet<FunctionId>) {
    if !matches!(value, MirRvalue::ShortCircuit { .. }) {
        value.visit_direct_expression_regions_dyn(&mut |region| {
            region.visit_operands(|value| operand_handle(value, reachable));
        });
    }
    match value {
        MirRvalue::Use(value)
        | MirRvalue::Unary(_, value)
        | MirRvalue::Spawn(value)
        | MirRvalue::Member { base: value, .. } => operand_handle(value, reachable),
        MirRvalue::Binary(left, _, right) => {
            operand_handle(left, reachable);
            operand_handle(right, reachable);
        }
        MirRvalue::Call(call) => {
            for argument in &call.args {
                argument.visit_operands(|value| operand_handle(value, reachable));
            }
        }
        MirRvalue::Future { args, .. } => {
            for argument in args {
                argument.visit_operands(|value| operand_handle(value, reachable));
            }
        }
        MirRvalue::ShortCircuit {
            left,
            right_temps,
            right,
            ..
        } => {
            operand_handle(left, reachable);
            operand_handle(right, reachable);
            for statement in right_temps {
                collect_statement_handles(&statement.kind, reachable);
            }
        }
        _ => {}
    }
}

fn collect_statement_handles(statement: &MirStmtKind, reachable: &mut BTreeSet<FunctionId>) {
    match statement {
        MirStmtKind::Assign { value, .. }
        | MirStmtKind::MultiAssign { value, .. }
        | MirStmtKind::SequenceAssign { value, .. }
        | MirStmtKind::Expr(value) => collect_escaped_handles(value, reachable),
        MirStmtKind::CaptureSequence { source, .. } => {
            source.visit_operands(|value| operand_handle(value, reachable));
        }
        _ => {}
    }
}

fn operand_handle(operand: &crate::MirOperand, reachable: &mut BTreeSet<FunctionId>) {
    if let crate::MirOperand::FunctionHandle(
        runmat_hir::CallableIdentity::BoundFunction(function)
        | runmat_hir::CallableIdentity::AnonymousFunction(function)
        | runmat_hir::CallableIdentity::ExternalFunction { function, .. },
    ) = operand
    {
        reachable.insert(*function);
    }
}

pub(crate) fn call_graph(assembly: &MirAssembly) -> BTreeMap<FunctionId, BTreeSet<FunctionId>> {
    assembly
        .bodies
        .iter()
        .map(|(function, body)| {
            let mut callees = BTreeSet::new();
            for statement in body.blocks.iter().flat_map(|block| block.statements.iter()) {
                collect_statement_calls(&statement.kind, &mut callees);
            }
            (*function, callees)
        })
        .collect()
}

fn collect_expansion_calls(source: &crate::MirExpansionSource, callees: &mut BTreeSet<FunctionId>) {
    source.visit_direct_expression_regions_dyn(&mut |region| {
        for step in region.steps() {
            match step {
                crate::MirExpressionStep::Let { value, .. } => collect_calls(value, callees),
                crate::MirExpressionStep::CaptureSequence { source, .. } => {
                    collect_expansion_calls(source, callees)
                }
            }
        }
    });
}

fn collect_calls(value: &MirRvalue, callees: &mut BTreeSet<FunctionId>) {
    if !matches!(value, MirRvalue::ShortCircuit { .. }) {
        value.visit_direct_expression_regions_dyn(&mut |region| {
            for step in region.steps() {
                match step {
                    crate::MirExpressionStep::Let { value, .. } => collect_calls(value, callees),
                    crate::MirExpressionStep::CaptureSequence { source, .. } => {
                        collect_expansion_calls(source, callees)
                    }
                }
            }
        });
    }
    match value {
        MirRvalue::Call(call) => {
            if let MirCallee::Static(
                runmat_hir::CallableIdentity::BoundFunction(function)
                | runmat_hir::CallableIdentity::AnonymousFunction(function)
                | runmat_hir::CallableIdentity::ExternalFunction { function, .. },
            ) = &call.callee
            {
                callees.insert(*function);
            }
        }
        MirRvalue::ShortCircuit { right_temps, .. } => {
            for statement in right_temps {
                collect_statement_calls(&statement.kind, callees);
            }
        }
        MirRvalue::Future { function, .. } => {
            callees.insert(*function);
        }
        _ => {}
    }
}

fn collect_statement_calls(statement: &MirStmtKind, callees: &mut BTreeSet<FunctionId>) {
    match statement {
        MirStmtKind::Assign { value, .. }
        | MirStmtKind::MultiAssign { value, .. }
        | MirStmtKind::SequenceAssign { value, .. }
        | MirStmtKind::Expr(value) => collect_calls(value, callees),
        MirStmtKind::CaptureSequence { source, .. } => collect_expansion_calls(source, callees),
        _ => {}
    }
}

pub(crate) fn strongly_connected_components(
    graph: &BTreeMap<FunctionId, BTreeSet<FunctionId>>,
) -> Vec<Vec<FunctionId>> {
    struct Tarjan<'a> {
        graph: &'a BTreeMap<FunctionId, BTreeSet<FunctionId>>,
        next: usize,
        indices: BTreeMap<FunctionId, usize>,
        low: BTreeMap<FunctionId, usize>,
        stack: Vec<FunctionId>,
        on_stack: BTreeSet<FunctionId>,
        components: Vec<Vec<FunctionId>>,
    }
    fn visit(node: FunctionId, state: &mut Tarjan<'_>) {
        let index = state.next;
        state.next += 1;
        state.indices.insert(node, index);
        state.low.insert(node, index);
        state.stack.push(node);
        state.on_stack.insert(node);
        for successor in state.graph.get(&node).into_iter().flatten() {
            if !state.graph.contains_key(successor) {
                continue;
            }
            if !state.indices.contains_key(successor) {
                visit(*successor, state);
                let next_low = state.low[successor];
                if let Some(low) = state.low.get_mut(&node) {
                    *low = (*low).min(next_low);
                }
            } else if state.on_stack.contains(successor) {
                let successor_index = state.indices[successor];
                if let Some(low) = state.low.get_mut(&node) {
                    *low = (*low).min(successor_index);
                }
            }
        }
        if state.low[&node] == state.indices[&node] {
            let mut component = Vec::new();
            while let Some(member) = state.stack.pop() {
                state.on_stack.remove(&member);
                component.push(member);
                if member == node {
                    break;
                }
            }
            component.sort();
            state.components.push(component);
        }
    }
    let mut state = Tarjan {
        graph,
        next: 0,
        indices: BTreeMap::new(),
        low: BTreeMap::new(),
        stack: Vec::new(),
        on_stack: BTreeSet::new(),
        components: Vec::new(),
    };
    for node in graph.keys() {
        if !state.indices.contains_key(node) {
            visit(*node, &mut state);
        }
    }
    state.components
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn calls_inside_top_level_sequence_capture_regions_enter_the_graph() {
        let ast = runmat_parser::parse(
            r#"
function out = main(values)
  out = consume(values{helper(end)});
end
function out = helper(index)
  out = index;
end
function out = consume(varargin)
  out = varargin;
end
"#,
        )
        .expect("parse fixture");
        let hir = runmat_hir::lower(&ast, &runmat_hir::LoweringContext::empty())
            .expect("lower HIR fixture");
        let mir = crate::lowering::lower_assembly(&hir.assembly).expect("lower MIR fixture");
        assert!(mir.bodies.values().any(|body| {
            body.blocks.iter().any(|block| {
                block
                    .statements
                    .iter()
                    .any(|statement| matches!(statement.kind, MirStmtKind::CaptureSequence { .. }))
            })
        }));

        let by_name = mir
            .functions
            .iter()
            .map(|(id, metadata)| (metadata.name.0.as_str(), *id))
            .collect::<BTreeMap<_, _>>();
        let graph = call_graph(&mir);
        let main = by_name["main"];
        assert!(graph[&main].contains(&by_name["helper"]));
        assert!(graph[&main].contains(&by_name["consume"]));
    }
}

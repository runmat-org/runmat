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
        for value in body.blocks.iter().flat_map(|block| {
            block
                .statements
                .iter()
                .filter_map(|statement| match &statement.kind {
                    MirStmtKind::Assign { value, .. }
                    | MirStmtKind::MultiAssign { value, .. }
                    | MirStmtKind::Expr(value) => Some(value),
                    _ => None,
                })
        }) {
            collect_escaped_handles(value, &mut reachable);
        }
    }
    reachable
}

fn collect_escaped_handles(value: &MirRvalue, reachable: &mut BTreeSet<FunctionId>) {
    fn operand(operand: &crate::MirOperand, reachable: &mut BTreeSet<FunctionId>) {
        if let crate::MirOperand::FunctionHandle(
            runmat_hir::CallableIdentity::BoundFunction(function)
            | runmat_hir::CallableIdentity::AnonymousFunction(function)
            | runmat_hir::CallableIdentity::ExternalFunction { function, .. },
        ) = operand
        {
            reachable.insert(*function);
        }
    }
    match value {
        MirRvalue::Use(value)
        | MirRvalue::Unary(_, value)
        | MirRvalue::Spawn(value)
        | MirRvalue::Member { base: value, .. } => operand(value, reachable),
        MirRvalue::Binary(left, _, right) => {
            operand(left, reachable);
            operand(right, reachable);
        }
        MirRvalue::Call(call) => {
            for argument in &call.args {
                operand(argument.operand(), reachable);
            }
        }
        MirRvalue::Future { args, .. } => {
            for argument in args {
                operand(argument.operand(), reachable);
            }
        }
        MirRvalue::ShortCircuit {
            left,
            right_temps,
            right,
            ..
        } => {
            operand(left, reachable);
            operand(right, reachable);
            for statement in right_temps {
                match &statement.kind {
                    MirStmtKind::Assign { value, .. }
                    | MirStmtKind::MultiAssign { value, .. }
                    | MirStmtKind::Expr(value) => collect_escaped_handles(value, reachable),
                    _ => {}
                }
            }
        }
        _ => {}
    }
}

pub(crate) fn call_graph(assembly: &MirAssembly) -> BTreeMap<FunctionId, BTreeSet<FunctionId>> {
    assembly
        .bodies
        .iter()
        .map(|(function, body)| {
            let mut callees = BTreeSet::new();
            for value in body.blocks.iter().flat_map(|block| {
                block
                    .statements
                    .iter()
                    .filter_map(|statement| match &statement.kind {
                        MirStmtKind::Assign { value, .. }
                        | MirStmtKind::MultiAssign { value, .. }
                        | MirStmtKind::Expr(value) => Some(value),
                        _ => None,
                    })
            }) {
                collect_calls(value, &mut callees);
            }
            (*function, callees)
        })
        .collect()
}

fn collect_calls(value: &MirRvalue, callees: &mut BTreeSet<FunctionId>) {
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
                match &statement.kind {
                    MirStmtKind::Assign { value, .. }
                    | MirStmtKind::MultiAssign { value, .. }
                    | MirStmtKind::Expr(value) => collect_calls(value, callees),
                    _ => {}
                }
            }
        }
        MirRvalue::Future { function, .. } => {
            callees.insert(*function);
        }
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

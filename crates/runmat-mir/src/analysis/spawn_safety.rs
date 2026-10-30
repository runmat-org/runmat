use crate::{
    BasicBlockId, MirAssembly, MirBody, MirCallee, MirDiagnostic, MirDiagnosticSeverity,
    MirIndexComponent, MirIndexing, MirLocal, MirLocalId, MirLocalKind, MirOperand, MirPlace,
    MirRvalue, MirStmtKind, MirTerminatorKind, SpawnBoundary,
};
use runmat_hir::{BindingId, FunctionId, Span};
use runmat_types::{SpawnSafetyFact, SpawnSafetyReason};
use std::collections::{BTreeSet, HashMap};

mod definition;
mod targets;
pub use definition::function_spawn_safety;
use targets::{entry_targets, FutureTargets};

#[derive(Debug, Clone, PartialEq)]
struct CaptureFacts {
    reads_captures: BTreeSet<BindingId>,
    writes_captures: BTreeSet<BindingId>,
    requires_isolation: bool,
}

fn analyze_capture_facts(body: &MirBody) -> CaptureFacts {
    let mut reads_captures = BTreeSet::new();
    let mut writes_captures = BTreeSet::new();

    for block in &body.blocks {
        for stmt in &block.statements {
            match &stmt.kind {
                MirStmtKind::PrepareFixedAssignment(owner) => {
                    owner.visit_external_operands(|operand| {
                        scan_operand(body, operand, &mut reads_captures)
                    });
                    owner.visit_preparation_steps_dyn(&mut |steps| {
                        scan_preparation_steps(body, steps, &mut reads_captures)
                    });
                }
                MirStmtKind::PreparedAssign { preparation, value } => {
                    scan_rvalue(body, value, &mut reads_captures);
                    scan_place_write(body, &preparation.mutation.place, &mut writes_captures);
                }
                MirStmtKind::Assign { place, value } => {
                    scan_rvalue(body, value, &mut reads_captures);
                    scan_place_write(body, place, &mut writes_captures);
                }
                MirStmtKind::MultiAssign { targets, value } => {
                    scan_rvalue(body, value, &mut reads_captures);
                    for target in &targets.targets {
                        match target {
                            crate::MirOutputTarget::Place(place) => {
                                scan_place_write(body, place, &mut writes_captures)
                            }
                            crate::MirOutputTarget::Sequence(target) => scan_sequence_target(
                                body,
                                target,
                                &mut reads_captures,
                                &mut writes_captures,
                            ),
                            crate::MirOutputTarget::Discard => {}
                        }
                    }
                }
                MirStmtKind::SequenceAssign { target, value } => {
                    scan_rvalue(body, value, &mut reads_captures);
                    match target {
                        crate::MirSequenceTarget::Member { base, .. } => {
                            scan_place_write(body, base, &mut writes_captures)
                        }
                        crate::MirSequenceTarget::DynamicMember { base, member } => {
                            scan_place_write(body, base, &mut writes_captures);
                            scan_operand(body, member, &mut reads_captures);
                        }
                        crate::MirSequenceTarget::CellContents { base, indexing } => {
                            scan_place_write(body, base, &mut writes_captures);
                            scan_indexing(body, indexing, &mut reads_captures);
                        }
                    }
                }
                MirStmtKind::CaptureSequence { source, .. } => {
                    source
                        .visit_operands(|operand| scan_operand(body, operand, &mut reads_captures));
                }
                MirStmtKind::Expr(value) => {
                    scan_rvalue(body, value, &mut reads_captures);
                }
                MirStmtKind::StatementOutput(output) => {
                    if let crate::MirStatementOutputSource::Local(local) = output.source {
                        scan_operand(body, &crate::MirOperand::Local(local), &mut reads_captures);
                    }
                }
                MirStmtKind::PlaceMutation(_) => {}
                MirStmtKind::WorkspaceEffect { .. } | MirStmtKind::EnvironmentEffect(_) => {}
            }
        }

        match &block.terminator.kind {
            MirTerminatorKind::Branch { cond, .. } => {
                scan_operand(body, cond, &mut reads_captures);
            }
            MirTerminatorKind::Switch { discr, cases, .. } => {
                scan_operand(body, discr, &mut reads_captures);
                for (case, _) in cases {
                    scan_operand(body, case, &mut reads_captures);
                }
            }
            MirTerminatorKind::For {
                binding, iterable, ..
            } => {
                scan_rvalue(body, iterable, &mut reads_captures);
                scan_local_write(body, *binding, &mut writes_captures);
            }
            MirTerminatorKind::ParFor {
                binding,
                iterable,
                maximum_workers,
                ..
            } => {
                scan_rvalue(body, iterable, &mut reads_captures);
                if let Some(workers) = maximum_workers.as_deref() {
                    scan_rvalue(body, workers, &mut reads_captures);
                }
                scan_local_write(body, *binding, &mut writes_captures);
            }
            MirTerminatorKind::Spmd { header, .. } => match header.as_ref() {
                crate::parallel::MirSpmdHeader::Default => {}
                crate::parallel::MirSpmdHeader::One(value) => {
                    scan_rvalue(body, value, &mut reads_captures);
                }
                crate::parallel::MirSpmdHeader::Two(first, second) => {
                    scan_rvalue(body, first, &mut reads_captures);
                    scan_rvalue(body, second, &mut reads_captures);
                }
                crate::parallel::MirSpmdHeader::Three(first, second, third) => {
                    scan_rvalue(body, first, &mut reads_captures);
                    scan_rvalue(body, second, &mut reads_captures);
                    scan_rvalue(body, third, &mut reads_captures);
                }
            },
            MirTerminatorKind::Return(return_outputs) => {
                for output in return_outputs {
                    scan_operand(body, output, &mut reads_captures);
                }
            }
            MirTerminatorKind::Await { future, .. } => {
                scan_operand(body, future, &mut reads_captures);
            }
            MirTerminatorKind::Goto(_)
            | MirTerminatorKind::TryCatch { .. }
            | MirTerminatorKind::Unreachable => {}
        }
    }

    CaptureFacts {
        reads_captures,
        writes_captures,
        requires_isolation: false,
    }
}

fn scan_sequence_target(
    body: &MirBody,
    target: &crate::MirSequenceTarget,
    reads_captures: &mut BTreeSet<BindingId>,
    writes_captures: &mut BTreeSet<BindingId>,
) {
    scan_place_write(body, target.base(), writes_captures);
    match target {
        crate::MirSequenceTarget::Member { .. } => {}
        crate::MirSequenceTarget::DynamicMember { member, .. } => {
            scan_operand(body, member, reads_captures)
        }
        crate::MirSequenceTarget::CellContents { indexing, .. } => {
            scan_indexing(body, indexing, reads_captures)
        }
    }
}

fn scan_rvalue(body: &MirBody, value: &MirRvalue, reads_captures: &mut BTreeSet<BindingId>) {
    match value {
        MirRvalue::Use(operand) | MirRvalue::Unary(_, operand) => {
            scan_operand(body, operand, reads_captures);
        }
        MirRvalue::Binary(left, _, right) => {
            scan_operand(body, left, reads_captures);
            scan_operand(body, right, reads_captures);
        }
        MirRvalue::ShortCircuit {
            left,
            right_temps,
            right,
            ..
        } => {
            scan_operand(body, left, reads_captures);
            for stmt in right_temps {
                match &stmt.kind {
                    crate::MirStmtKind::PreparedAssign { value, .. }
                    | crate::MirStmtKind::Assign { value, .. }
                    | crate::MirStmtKind::Expr(value)
                    | crate::MirStmtKind::MultiAssign { value, .. }
                    | crate::MirStmtKind::SequenceAssign { value, .. } => {
                        scan_rvalue(body, value, reads_captures);
                    }
                    crate::MirStmtKind::PrepareFixedAssignment(owner) => {
                        owner.visit_external_operands(|operand| {
                            scan_operand(body, operand, reads_captures)
                        });
                        owner.visit_preparation_steps_dyn(&mut |steps| {
                            scan_preparation_steps(body, steps, reads_captures)
                        });
                    }
                    crate::MirStmtKind::CaptureSequence { source, .. } => {
                        source
                            .visit_operands(|operand| scan_operand(body, operand, reads_captures));
                        source.visit_direct_expression_steps_dyn(&mut |steps| {
                            scan_preparation_steps(body, steps, reads_captures)
                        });
                    }
                    crate::MirStmtKind::PlaceMutation(_)
                    | crate::MirStmtKind::StatementOutput(_)
                    | crate::MirStmtKind::WorkspaceEffect { .. }
                    | crate::MirStmtKind::EnvironmentEffect(_) => {}
                }
            }
            scan_operand(body, right, reads_captures);
        }
        MirRvalue::Range { start, step, end } => {
            scan_operand(body, start, reads_captures);
            if let Some(step) = step {
                scan_operand(body, step, reads_captures);
            }
            scan_operand(body, end, reads_captures);
        }
        MirRvalue::Call(call) => {
            if let MirCallee::Dynamic(callee) = &call.callee {
                scan_operand(body, callee, reads_captures);
            }
            for arg in &call.args {
                arg.visit_operands(|operand| scan_operand(body, operand, reads_captures));
            }
        }
        MirRvalue::Aggregate { elements, .. } => {
            for element in elements {
                if let Some(operand) = element.operand() {
                    scan_operand(body, operand, reads_captures);
                }
            }
        }
        MirRvalue::StructLiteral { fields } | MirRvalue::ObjectLiteral { fields, .. } => {
            for (_, value) in fields {
                scan_operand(body, value, reads_captures);
            }
        }
        MirRvalue::Index { base, indexing } => {
            scan_operand(body, base, reads_captures);
            scan_indexing(body, indexing, reads_captures);
        }
        MirRvalue::SubscriptChain(chain) => {
            scan_operand(body, &chain.root, reads_captures);
            for step in &chain.steps {
                match step {
                    crate::MirSubscriptStep::Index(indexing) => {
                        scan_indexing(body, indexing, reads_captures)
                    }
                    crate::MirSubscriptStep::DottedInvoke { arguments, .. } => {
                        for argument in arguments {
                            match argument {
                                crate::MirDottedInvokeArgument::Colon => {}
                                crate::MirDottedInvokeArgument::Contextual(region) => {
                                    scan_expression_region(
                                        body,
                                        crate::MirExpressionRegionRef::Contextual(region),
                                        reads_captures,
                                    )
                                }
                                crate::MirDottedInvokeArgument::Call(region) => {
                                    for step in &region.steps {
                                        match step {
                                            crate::MirExpressionStep::Let { value, .. } => {
                                                scan_rvalue(body, value, reads_captures)
                                            }
                                            crate::MirExpressionStep::CaptureSequence {
                                                source,
                                                ..
                                            } => source.visit_operands(|operand| {
                                                scan_operand(body, operand, reads_captures)
                                            }),
                                        }
                                    }
                                    region.argument.visit_operands(|operand| {
                                        scan_operand(body, operand, reads_captures)
                                    });
                                }
                            }
                        }
                    }
                    crate::MirSubscriptStep::DynamicMember(member) => {
                        scan_operand(body, member, reads_captures)
                    }
                    crate::MirSubscriptStep::PreparedDynamicMember(region) => {
                        scan_expression_region(
                            body,
                            crate::MirExpressionRegionRef::Prepared(region),
                            reads_captures,
                        )
                    }
                    crate::MirSubscriptStep::Member(_) => {}
                }
            }
        }
        MirRvalue::Future { args, .. } | MirRvalue::ExternalFuture { args, .. } => {
            for arg in args {
                arg.visit_operands(|operand| scan_operand(body, operand, reads_captures));
            }
        }
        MirRvalue::Member { base, .. } => {
            scan_operand(body, base, reads_captures);
        }
        MirRvalue::DynamicMember { base, member, .. } => {
            scan_operand(body, base, reads_captures);
            scan_operand(body, member, reads_captures);
        }
        MirRvalue::WorkspaceFirstStaticProperty { .. } => {}
        MirRvalue::MetaClass(_) | MirRvalue::Colon | MirRvalue::End => {}
        MirRvalue::Spawn(future) => {
            scan_operand(body, future, reads_captures);
        }
        MirRvalue::Distributed(operation) => {
            for operand in operation.operands() {
                scan_operand(body, operand, reads_captures);
            }
        }
        MirRvalue::Collective(operation) => {
            for operand in operation.operands() {
                scan_operand(body, operand, reads_captures);
            }
        }
    }
}

fn scan_place_write(body: &MirBody, place: &MirPlace, writes_captures: &mut BTreeSet<BindingId>) {
    if let Some(local) = place_root(place) {
        scan_local_write(body, local, writes_captures);
    }
}

fn scan_indexing(body: &MirBody, indexing: &MirIndexing, reads_captures: &mut BTreeSet<BindingId>) {
    for component in &indexing.components {
        match component {
            MirIndexComponent::Expr(operand) => {
                scan_operand(body, operand, reads_captures);
            }
            MirIndexComponent::ContextualExpr(region) => scan_expression_region(
                body,
                crate::MirExpressionRegionRef::Contextual(region),
                reads_captures,
            ),
            MirIndexComponent::PreparedExpr(region) => scan_expression_region(
                body,
                crate::MirExpressionRegionRef::Prepared(region),
                reads_captures,
            ),
            MirIndexComponent::Expansion(region) => {
                for step in &region.steps {
                    match step {
                        crate::MirExpressionStep::Let { value, .. } => {
                            scan_rvalue(body, value, reads_captures)
                        }
                        crate::MirExpressionStep::CaptureSequence { source, .. } => {
                            source.visit_operands(|operand| {
                                scan_operand(body, operand, reads_captures)
                            });
                        }
                    }
                }
                region
                    .argument
                    .visit_operands(|operand| scan_operand(body, operand, reads_captures));
            }
            MirIndexComponent::Colon => {}
        }
    }
}

fn scan_expression_region(
    body: &MirBody,
    region: crate::MirExpressionRegionRef<'_>,
    reads_captures: &mut BTreeSet<BindingId>,
) {
    for step in region.steps() {
        match step {
            crate::MirExpressionStep::Let { value, .. } => scan_rvalue(body, value, reads_captures),
            crate::MirExpressionStep::CaptureSequence { source, .. } => {
                source.visit_operands(|operand| scan_operand(body, operand, reads_captures));
                source.visit_expression_regions_dyn(&mut |nested| {
                    scan_expression_region(body, nested, reads_captures)
                });
            }
        }
    }
    scan_operand(body, region.result(), reads_captures);
}

fn scan_preparation_steps(
    body: &MirBody,
    steps: &[crate::MirExpressionStep],
    reads: &mut BTreeSet<BindingId>,
) {
    for step in steps {
        match step {
            crate::MirExpressionStep::Let { value, .. } => scan_rvalue(body, value, reads),
            crate::MirExpressionStep::CaptureSequence { source, .. } => {
                source.visit_operands(|operand| scan_operand(body, operand, reads));
                source.visit_direct_expression_steps_dyn(&mut |steps| {
                    scan_preparation_steps(body, steps, reads)
                });
            }
        }
    }
}

fn place_root(place: &MirPlace) -> Option<MirLocalId> {
    match place {
        MirPlace::Local(local) => Some(*local),
        MirPlace::Member(base, _) | MirPlace::DynamicMember(base, _) | MirPlace::Index(base, _) => {
            place_root(base)
        }
        MirPlace::Binding(_) => None,
    }
}

fn scan_operand(body: &MirBody, operand: &MirOperand, reads_captures: &mut BTreeSet<BindingId>) {
    match operand {
        MirOperand::Local(local) => {
            if let Some(binding) = capture_binding(body, *local) {
                reads_captures.insert(binding);
            }
        }
        MirOperand::FunctionHandle(_) => {}
        MirOperand::Constant(_) => {}
    }
}

fn scan_local_write(body: &MirBody, local: MirLocalId, writes_captures: &mut BTreeSet<BindingId>) {
    if let Some(binding) = capture_binding(body, local) {
        writes_captures.insert(binding);
    }
}

fn capture_binding(body: &MirBody, local: MirLocalId) -> Option<BindingId> {
    local_for_id(body, local).and_then(|local| {
        if matches!(local.kind, MirLocalKind::Capture) {
            local.binding
        } else {
            None
        }
    })
}

fn local_for_id(body: &MirBody, local: MirLocalId) -> Option<&MirLocal> {
    body.locals.iter().find(|candidate| candidate.id == local)
}

pub(super) fn analyze_assembly_spawn_boundaries(
    assembly: &MirAssembly,
) -> HashMap<FunctionId, Vec<SpawnBoundary>> {
    let capture_facts: HashMap<_, _> = assembly
        .bodies
        .values()
        .map(|body| (body.function, definition::capture_facts(assembly, body)))
        .collect();
    assembly
        .bodies
        .iter()
        .map(|(function, body)| {
            (
                *function,
                analyze_spawn_boundaries_with_capture_facts(body, &capture_facts),
            )
        })
        .collect()
}

fn analyze_spawn_boundaries_with_capture_facts(
    body: &MirBody,
    capture_facts: &HashMap<FunctionId, CaptureFacts>,
) -> Vec<SpawnBoundary> {
    let in_states = future_target_in_states(body);
    let block_index: HashMap<_, _> = body
        .blocks
        .iter()
        .enumerate()
        .map(|(idx, block)| (block.id, idx))
        .collect();
    let mut boundaries = Vec::new();
    for block in &body.blocks {
        let mut state = block_index
            .get(&block.id)
            .and_then(|idx| in_states.get(*idx))
            .cloned()
            .unwrap_or_else(|| vec![FutureTargets::new(); body.locals.len()]);
        for stmt in &block.statements {
            walk_spawn_statement(body, stmt, &mut state, &mut |value, span, state| {
                collect_classified_spawn_rvalue(value, span, capture_facts, state, &mut boundaries);
            });
        }
    }
    boundaries
}

fn future_target_in_states(body: &MirBody) -> Vec<Vec<FutureTargets>> {
    let local_count = body.locals.len();
    let block_index: HashMap<BasicBlockId, usize> = body
        .blocks
        .iter()
        .enumerate()
        .map(|(idx, block)| (block.id, idx))
        .collect();
    // Unreachable is distinct from a reachable unknown target. Unknown is an
    // absorbing join member, not an empty list of known incoming possibilities.
    let mut predecessors = vec![Vec::new(); body.blocks.len()];
    for (index, block) in body.blocks.iter().enumerate() {
        for successor in successors(&block.terminator.kind) {
            if let Some(&target) = block_index.get(&successor) {
                predecessors[target].push(index);
            }
        }
    }
    let unknown = || vec![FutureTargets::new(); local_count];
    let mut in_states: Vec<Option<Vec<FutureTargets>>> = vec![None; body.blocks.len()];
    let mut out_states = in_states.clone();
    let mut changed = true;
    while changed {
        changed = false;
        for (index, block) in body.blocks.iter().enumerate() {
            let mut incoming = (index == 0).then(|| entry_targets(body));
            for &predecessor in &predecessors[index] {
                if let Some(state) = &out_states[predecessor] {
                    match &mut incoming {
                        Some(joined) => join_future_targets(joined, state),
                        None => incoming = Some(state.clone()),
                    }
                }
            }
            in_states[index] = incoming.clone();
            let output = incoming.map(|mut state| {
                transfer_future_targets(body, block, &mut state);
                state
            });
            if output != out_states[index] {
                out_states[index] = output;
                changed = true;
            }
        }
    }
    in_states
        .into_iter()
        .map(|state| state.unwrap_or_else(unknown))
        .collect()
}

fn join_future_targets(joined: &mut [FutureTargets], incoming: &[FutureTargets]) {
    for (target, source) in joined.iter_mut().zip(incoming) {
        target.join(source);
    }
}

fn transfer_future_targets(body: &MirBody, block: &crate::BasicBlock, state: &mut [FutureTargets]) {
    for stmt in &block.statements {
        walk_spawn_statement(body, stmt, state, &mut |_, _, _| {});
    }
    if let MirTerminatorKind::Await {
        result: Some(place),
        ..
    } = &block.terminator.kind
    {
        invalidate_place(body, place, state);
    }
}

type SpawnClassifier<'a> = dyn FnMut(&MirRvalue, Span, &[FutureTargets]) + 'a;

/// Execute the same ordered static transfer for classification and the
/// predecessor fixed point. Descriptor-only phases do not replay selectors.
fn walk_spawn_statement(
    body: &MirBody,
    statement: &crate::MirStmt,
    state: &mut [FutureTargets],
    classify: &mut SpawnClassifier<'_>,
) {
    match &statement.kind {
        MirStmtKind::PrepareFixedAssignment(owner) => {
            walk_statement_preparation(body, statement, state, classify);
            if let crate::MirFixedPreparationPhase::EvaluateSelector {
                region,
                destination: crate::MirFixedSelectorDestination::Scalar(local),
                ..
            } = &owner.phase
            {
                state[local.0] = match &region.argument {
                    crate::MirCallArg::Single(operand) => operand_future_targets(operand, state),
                    _ => FutureTargets::new(),
                };
            }
        }
        MirStmtKind::PreparedAssign { preparation, value } => {
            walk_spawn_rvalue(body, value, statement.span, state, classify);
            assign_future_targets(body, &preparation.mutation.place, value, state);
        }
        MirStmtKind::Assign { place, value } => {
            walk_spawn_rvalue(body, value, statement.span, state, classify);
            walk_statement_preparation(body, statement, state, classify);
            assign_future_targets(body, place, value, state);
        }
        MirStmtKind::MultiAssign { targets, value } => {
            walk_spawn_rvalue(body, value, statement.span, state, classify);
            walk_statement_preparation(body, statement, state, classify);
            for target in &targets.targets {
                match target {
                    crate::MirOutputTarget::Place(place) => invalidate_place(body, place, state),
                    crate::MirOutputTarget::Sequence(target) => {
                        invalidate_place(body, target.base(), state)
                    }
                    crate::MirOutputTarget::Discard => {}
                }
            }
        }
        MirStmtKind::SequenceAssign { target, value } => {
            walk_spawn_rvalue(body, value, statement.span, state, classify);
            walk_statement_preparation(body, statement, state, classify);
            invalidate_place(body, target.base(), state);
        }
        MirStmtKind::Expr(value) => walk_spawn_rvalue(body, value, statement.span, state, classify),
        // CaptureSequence's directly executed source preparation was visited
        // above. It defines a sequence receipt, never a scalar future local.
        MirStmtKind::CaptureSequence { .. } => {
            walk_statement_preparation(body, statement, state, classify)
        }
        MirStmtKind::WorkspaceEffect { effect, bindings } => {
            use runmat_types::WorkspaceEffect;
            match effect {
                WorkspaceEffect::CreatesBinding
                | WorkspaceEffect::ClearsBinding
                | WorkspaceEffect::MutatesGlobal
                | WorkspaceEffect::MutatesPersistent
                | WorkspaceEffect::LoadsExternalBindings => {
                    // Empty builtin annotations precede the executing call.
                    // Its typed effect is applied after its arguments below;
                    // declarations instead identify their exact binding slots.
                    for local in bindings {
                        state[local.0].clear();
                    }
                }
                // DynamicEval annotations also represent foreign-workspace eval.
                // The executing call's typed effect below identifies current eval.
                WorkspaceEffect::None
                | WorkspaceEffect::DynamicEval
                | WorkspaceEffect::ReadsWorkspace
                | WorkspaceEffect::ClearsFunctionCache => {}
            }
        }
        // PlaceMutation describes the following store, not an earlier write:
        // its RHS must still see the original future target.
        MirStmtKind::PlaceMutation(_)
        | MirStmtKind::StatementOutput(_)
        | MirStmtKind::EnvironmentEffect(_) => {}
    }
}

fn walk_statement_preparation(
    body: &MirBody,
    statement: &crate::MirStmt,
    state: &mut [FutureTargets],
    classify: &mut SpawnClassifier<'_>,
) {
    statement
        .kind
        .visit_statement_expression_steps_dyn(&mut |steps| {
            walk_spawn_steps(body, steps, state, classify)
        });
}

fn walk_spawn_steps(
    body: &MirBody,
    steps: &[crate::MirExpressionStep],
    state: &mut [FutureTargets],
    classify: &mut SpawnClassifier<'_>,
) {
    for step in steps {
        match step {
            crate::MirExpressionStep::Let { local, value, span } => {
                walk_spawn_rvalue(body, value, *span, state, classify);
                state[local.0] = rvalue_future_targets(value, state);
            }
            crate::MirExpressionStep::CaptureSequence { source, .. } => {
                source.visit_direct_expression_steps_dyn(&mut |steps| {
                    walk_spawn_steps(body, steps, state, classify)
                });
            }
        }
    }
}

fn walk_spawn_rvalue(
    body: &MirBody,
    value: &MirRvalue,
    span: Span,
    state: &mut [FutureTargets],
    classify: &mut SpawnClassifier<'_>,
) {
    if let MirRvalue::ShortCircuit { right_temps, .. } = value {
        let mut executed = state.to_vec();
        for statement in right_temps {
            walk_spawn_statement(body, statement, &mut executed, classify);
        }
        // Either the complete RHS ran or it was skipped. Empty means unknown,
        // not evidence that a known safe target survived the other branch.
        join_future_targets(state, &executed);
    } else {
        value.visit_direct_expression_steps_dyn(&mut |steps| {
            walk_spawn_steps(body, steps, state, classify)
        });
    }
    classify(value, span, state);
    if matches!(value, MirRvalue::Call(call) if matches!(call.workspace_effect,
        Some(runmat_builtins::BuiltinWorkspaceEffect::CreatesBinding
            | runmat_builtins::BuiltinWorkspaceEffect::ClearsBinding
            | runmat_builtins::BuiltinWorkspaceEffect::LoadsExternalBindings
            | runmat_builtins::BuiltinWorkspaceEffect::DynamicEvalCurrentWorkspace)))
    {
        invalidate_workspace(body, state);
    }
}

fn operand_future_targets(operand: &MirOperand, state: &[FutureTargets]) -> FutureTargets {
    match operand {
        MirOperand::Local(local) => state[local.0].clone(),
        MirOperand::Constant(_) | MirOperand::FunctionHandle(_) => FutureTargets::new(),
    }
}
fn rvalue_future_targets(value: &MirRvalue, state: &[FutureTargets]) -> FutureTargets {
    match value {
        MirRvalue::Future { function, .. } => FutureTargets::from([*function]),
        MirRvalue::ExternalFuture { .. } => FutureTargets::runtime_checked(),
        MirRvalue::Use(operand) => operand_future_targets(operand, state),
        _ => FutureTargets::new(),
    }
}
fn assign_future_targets(
    body: &MirBody,
    place: &MirPlace,
    value: &MirRvalue,
    state: &mut [FutureTargets],
) {
    if let MirPlace::Local(local) = place {
        state[local.0] = rvalue_future_targets(value, state);
    } else {
        invalidate_place(body, place, state);
    }
}

fn invalidate_place(body: &MirBody, place: &MirPlace, state: &mut [FutureTargets]) {
    if let Some(local) = place.root_local() {
        state[local.0].clear();
        return;
    }
    match place {
        MirPlace::Binding(binding) => {
            for local in &body.locals {
                if local.binding == Some(*binding) {
                    state[local.id.0].clear();
                }
            }
        }
        MirPlace::Member(base, _) | MirPlace::DynamicMember(base, _) | MirPlace::Index(base, _) => {
            invalidate_place(body, base, state)
        }
        MirPlace::Local(_) => unreachable!("local root handled above"),
    }
}

fn invalidate_workspace(body: &MirBody, state: &mut [FutureTargets]) {
    for local in super::workspace_effect::current_scope_bindings(body) {
        state[local.0].clear();
    }
}

pub(super) fn diagnose_spawn_safety(boundaries: &[SpawnBoundary]) -> Vec<MirDiagnostic> {
    boundaries
        .iter()
        .filter_map(|boundary| match &boundary.safety {
            SpawnSafetyFact::NotSpawnSafe { reason } => {
                Some(spawn_safety_diagnostic(*reason, boundary.span))
            }
            SpawnSafetyFact::SpawnSafe | SpawnSafetyFact::RequiresIsolation => None,
        })
        .collect()
}

fn collect_classified_spawn_rvalue(
    value: &MirRvalue,
    span: Span,
    capture_facts: &HashMap<FunctionId, CaptureFacts>,
    future_targets: &[FutureTargets],
    boundaries: &mut Vec<SpawnBoundary>,
) {
    let MirRvalue::Spawn(future) = value else {
        return;
    };
    let mut boundary = SpawnBoundary {
        future: future.clone(),
        safety: SpawnSafetyFact::RequiresIsolation,
        span,
    };
    boundary.safety = classify_spawn_boundary(&boundary, capture_facts, future_targets);
    boundaries.push(boundary);
}

fn classify_spawn_boundary(
    boundary: &SpawnBoundary,
    capture_facts: &HashMap<FunctionId, CaptureFacts>,
    future_targets: &[FutureTargets],
) -> SpawnSafetyFact {
    let Some(targets) = spawn_targets(&boundary.future, future_targets) else {
        return SpawnSafetyFact::NotSpawnSafe {
            reason: SpawnSafetyReason::UnknownDynamicCapture,
        };
    };
    let mut requires_isolation = targets.requires_runtime_owner();
    for target in targets.known() {
        let Some(summary) = capture_facts.get(&target) else {
            requires_isolation = true;
            continue;
        };
        match definition::capture_safety(summary) {
            denied @ SpawnSafetyFact::NotSpawnSafe { .. } => return denied,
            SpawnSafetyFact::RequiresIsolation => requires_isolation = true,
            SpawnSafetyFact::SpawnSafe => {}
        }
    }
    if requires_isolation {
        SpawnSafetyFact::RequiresIsolation
    } else {
        SpawnSafetyFact::SpawnSafe
    }
}

fn spawn_targets(future: &MirOperand, future_targets: &[FutureTargets]) -> Option<FutureTargets> {
    match future {
        MirOperand::Local(local) => future_targets
            .get(local.0)
            .filter(|targets| !targets.is_empty())
            .cloned(),
        MirOperand::FunctionHandle(_) | MirOperand::Constant(_) => None,
    }
}

fn successors(kind: &MirTerminatorKind) -> Vec<BasicBlockId> {
    match kind {
        MirTerminatorKind::Goto(target) => vec![*target],
        MirTerminatorKind::Branch {
            then_block,
            else_block,
            ..
        } => vec![*then_block, *else_block],
        MirTerminatorKind::Switch {
            cases, otherwise, ..
        } => cases
            .iter()
            .map(|(_, block)| *block)
            .chain(std::iter::once(*otherwise))
            .collect(),
        MirTerminatorKind::For {
            body_block,
            exit_block,
            ..
        } => vec![*body_block, *exit_block],
        MirTerminatorKind::ParFor {
            body_block,
            exit_block,
            ..
        }
        | MirTerminatorKind::Spmd {
            body_block,
            exit_block,
            ..
        } => vec![*body_block, *exit_block],
        MirTerminatorKind::TryCatch {
            try_block,
            catch_block,
            ..
        } => vec![*try_block, *catch_block],
        MirTerminatorKind::Await { resume, .. } => vec![*resume],
        MirTerminatorKind::Return(_) | MirTerminatorKind::Unreachable => Vec::new(),
    }
}

fn spawn_safety_diagnostic(reason: SpawnSafetyReason, span: Span) -> MirDiagnostic {
    let message = match reason {
        SpawnSafetyReason::MutableLexicalCapture => {
            "spawned future mutates a lexical capture from its parent frame"
        }
        SpawnSafetyReason::NonSendableRuntimeHandle => {
            "spawned future captures a non-sendable runtime handle"
        }
        SpawnSafetyReason::UnsynchronizedSharedMutation => {
            "spawned future performs unsynchronized shared mutation"
        }
        SpawnSafetyReason::UnknownDynamicCapture => {
            "spawned future has dynamically captured state with unknown spawn safety"
        }
    };
    MirDiagnostic::new("RM-MIR0003", MirDiagnosticSeverity::Error, message, span)
        .with_primary_label("this spawn crosses an unsafe task boundary")
        .with_help("avoid mutating parent-frame lexical captures from spawned futures")
        .with_category("spawn-safety")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn lower(source: &str) -> MirAssembly {
        let parsed = runmat_parser::parse(source).unwrap();
        let hir = runmat_hir::lower(&parsed, &runmat_hir::LoweringContext::empty()).unwrap();
        crate::lowering::lower_assembly(&hir.assembly).unwrap()
    }

    #[test]
    fn value_snapshot_callbacks_and_live_lexical_captures_have_distinct_admission() {
        for (source, role, expected) in [
            (
                "function h=outer(); cap=1; h=@(x) x+cap; end",
                runmat_types::FunctionSourceRole::Anonymous,
                SpawnSafetyFact::RequiresIsolation,
            ),
            (
                "function h=outer(); cap=1; function y=read(); y=cap; end; h=@read; end",
                runmat_types::FunctionSourceRole::Nested,
                SpawnSafetyFact::NotSpawnSafe {
                    reason: SpawnSafetyReason::MutableLexicalCapture,
                },
            ),
        ] {
            let assembly = lower(source);
            let (function, _) = assembly
                .functions
                .iter()
                .find(|(_, metadata)| metadata.source_role == role)
                .expect("captured definition");
            assert_eq!(function_spawn_safety(&assembly, *function), Some(expected));
        }
    }

    #[test]
    fn phased_selector_and_prepared_rhs_classify_original_future_targets() {
        for tail in [
            "a(c{:}, spawn(work())) = 1;",
            "fut = work(); a(c{:}) = spawn(fut);",
            "fut = work(); a(c{:}, flag && spawn(fut)) = 1;",
        ] {
            let source = format!("function y = outer(c,flag); cap=1; async function z=work(); z=cap; end; {tail} y=cap; end");
            let assembly = lower(&source);
            assert!(assembly
                .bodies
                .values()
                .flat_map(|body| &body.blocks)
                .flat_map(|block| &block.statements)
                .any(|statement| matches!(statement.kind, MirStmtKind::PrepareFixedAssignment(_))));
            let boundaries = analyze_assembly_spawn_boundaries(&assembly);
            let boundaries: Vec<_> = boundaries.values().flatten().collect();
            assert_eq!(
                boundaries.len(),
                1,
                "one actual spawn, no descriptor replay: {tail}"
            );
            assert_eq!(
                boundaries[0].safety,
                SpawnSafetyFact::NotSpawnSafe {
                    reason: SpawnSafetyReason::MutableLexicalCapture,
                },
                "original target must not degrade to unknown or disappear: {tail}"
            );
        }
    }

    #[test]
    fn phased_region_transfer_and_scalar_publication_share_one_ordered_walker() {
        let assembly = lower(
            "function y=outer(c); async function z=work(); z=1; end; a(c{:},work())=1; y=1; end",
        );
        let body = assembly
            .bodies
            .values()
            .find(|body| {
                body.blocks.iter().any(|block| {
                    block.statements.iter().any(|statement| {
                        matches!(statement.kind, MirStmtKind::PrepareFixedAssignment(_))
                    })
                })
            })
            .unwrap();
        let mut state = vec![FutureTargets::new(); body.locals.len()];
        let mut observed = false;
        for block in &body.blocks {
            let before = state.clone();
            for statement in &block.statements {
                walk_spawn_statement(body, statement, &mut state, &mut |_, _, _| {});
                if let MirStmtKind::PrepareFixedAssignment(owner) = &statement.kind {
                    if let crate::MirFixedPreparationPhase::EvaluateSelector {
                        region,
                        destination: crate::MirFixedSelectorDestination::Scalar(local),
                        ..
                    } = &owner.phase
                    {
                        if region.steps.iter().any(|step| {
                            matches!(
                                step,
                                crate::MirExpressionStep::Let {
                                    value: MirRvalue::Future { .. },
                                    ..
                                }
                            )
                        }) {
                            assert_eq!(
                                state[local.0].len(),
                                1,
                                "final scalar receipt retains its region future target"
                            );
                            observed = true;
                        }
                    }
                }
            }
            let mut transferred = before;
            transfer_future_targets(body, block, &mut transferred);
            assert_eq!(
                transferred, state,
                "classification and CFG transfer must agree"
            );
        }
        assert!(observed);
    }

    #[test]
    fn conditional_overwrite_does_not_promote_unknown_to_known_safe() {
        let assembly = lower("a=1;");
        let body = assembly.bodies.values().next().unwrap();
        let mut state = vec![FutureTargets::from([FunctionId(7)])];
        let value = MirRvalue::ShortCircuit {
            left: MirOperand::Local(MirLocalId(0)),
            op: crate::MirShortCircuitOp::And,
            right_temps: vec![crate::MirStmt {
                kind: MirStmtKind::Assign {
                    place: MirPlace::Local(MirLocalId(0)),
                    value: MirRvalue::Colon,
                },
                span: Span::default(),
            }],
            right: MirOperand::Local(MirLocalId(0)),
        };
        walk_spawn_rvalue(body, &value, Span::default(), &mut state, &mut |_, _, _| {});
        assert!(
            state[0].is_empty(),
            "the executed branch erased known provenance"
        );
    }

    #[test]
    fn cfg_unknown_predecessor_is_not_an_absent_known_target() {
        let assembly = lower("function y=outer(flag); async function z=work(); z=1; end; if flag; fut=work(); else; fut=1; end; y=spawn(fut); end");
        let boundaries = analyze_assembly_spawn_boundaries(&assembly);
        let boundaries: Vec<_> = boundaries.values().flatten().collect();
        assert_eq!(boundaries.len(), 1);
        assert_eq!(
            boundaries[0].safety,
            SpawnSafetyFact::NotSpawnSafe {
                reason: SpawnSafetyReason::UnknownDynamicCapture
            }
        );
    }

    #[test]
    fn runtime_future_input_requires_its_issued_owner_and_store_discards_provenance() {
        for (store, expected) in [
            ("", SpawnSafetyFact::RequiresIsolation),
            (
                "fut=1;",
                SpawnSafetyFact::NotSpawnSafe {
                    reason: SpawnSafetyReason::UnknownDynamicCapture,
                },
            ),
        ] {
            let assembly = lower(&format!(
                "function y=consume(fut); {store} y=spawn(fut); end"
            ));
            let boundaries = analyze_assembly_spawn_boundaries(&assembly);
            let boundaries: Vec<_> = boundaries.values().flatten().collect();
            assert_eq!(boundaries.len(), 1);
            assert_eq!(boundaries[0].safety, expected, "{store}");
        }
    }

    #[test]
    fn multi_assignment_overwrite_does_not_reuse_old_safe_future() {
        for store in ["[fut,other]=deal(1,2);", "fut{:}=1;", "fut(1)=1;"] {
            let assembly = lower(&format!("function y=outer(); async function z=work(); z=1; end; fut=work(); {store} y=spawn(fut); end"));
            let boundaries = analyze_assembly_spawn_boundaries(&assembly);
            let boundaries: Vec<_> = boundaries.values().flatten().collect();
            assert_eq!(boundaries.len(), 1, "{store}");
            assert_eq!(
                boundaries[0].safety,
                SpawnSafetyFact::NotSpawnSafe {
                    reason: SpawnSafetyReason::UnknownDynamicCapture
                },
                "{store}"
            );
        }
    }

    #[test]
    fn mutation_annotation_does_not_erase_rhs_future_before_actual_store() {
        let assembly = lower("function y=outer(); async function z=work(); z=1; end; fut=work(); fut(1)=spawn(fut); y=spawn(fut); end");
        let boundaries = analyze_assembly_spawn_boundaries(&assembly);
        let boundaries: Vec<_> = boundaries.values().flatten().collect();
        assert_eq!(boundaries.len(), 2);
        assert_eq!(boundaries[0].safety, SpawnSafetyFact::SpawnSafe);
        assert_eq!(
            boundaries[1].safety,
            SpawnSafetyFact::NotSpawnSafe {
                reason: SpawnSafetyReason::UnknownDynamicCapture
            }
        );
    }

    #[test]
    fn writes_invalidate_only_their_roots_after_rhs_classification() {
        let assembly = lower("a=1; b=2;");
        let body = assembly.bodies.values().next().unwrap();
        let binding = super::super::workspace_effect::current_scope_bindings(body)[0];
        let mut state = vec![FutureTargets::from([FunctionId(7)]); body.locals.len()];
        let statement = crate::MirStmt {
            kind: MirStmtKind::WorkspaceEffect {
                effect: runmat_types::WorkspaceEffect::ClearsBinding,
                bindings: vec![binding],
            },
            span: Span::default(),
        };
        walk_spawn_statement(body, &statement, &mut state, &mut |_, _, _| {});
        for (index, fact) in state.iter().enumerate() {
            assert_eq!(fact.is_empty(), index == binding.0);
        }

        // Even a nested write changes its root's future provenance, but does
        // not alter the independent value retained in another scalar local.
        state[binding.0] = FutureTargets::from([FunctionId(7)]);
        let place = MirPlace::DynamicMember(
            Box::new(MirPlace::Local(binding)),
            MirOperand::Local(binding),
        );
        invalidate_place(body, &place, &mut state);
        assert!(state[binding.0].is_empty());
        assert!(state
            .iter()
            .enumerate()
            .all(|(index, fact)| index == binding.0 || fact.len() == 1));
    }

    #[test]
    fn current_workspace_effect_preserves_private_receipts_and_foreign_eval_facts() {
        let assembly = lower("a=sin(1); b=2;");
        let body = assembly.bodies.values().next().unwrap();
        let mut body = body.clone();
        let mut receipt = body.locals[0].clone();
        receipt.id = MirLocalId(body.locals.len());
        receipt.kind = MirLocalKind::Temporary;
        receipt.binding = None;
        body.locals.push(receipt);
        let body = &body;
        let mut state = vec![FutureTargets::from([FunctionId(7)]); body.locals.len()];
        let annotation = crate::MirStmt {
            kind: MirStmtKind::WorkspaceEffect {
                effect: runmat_types::WorkspaceEffect::DynamicEval,
                bindings: vec![],
            },
            span: Span::default(),
        };
        walk_spawn_statement(body, &annotation, &mut state, &mut |_, _, _| {});
        assert!(state.iter().all(|fact| fact.len() == 1));
        let mut call = body
            .blocks
            .iter()
            .flat_map(|block| &block.statements)
            .find_map(|statement| match &statement.kind {
                MirStmtKind::Assign {
                    value: MirRvalue::Call(call),
                    ..
                }
                | MirStmtKind::Expr(MirRvalue::Call(call)) => Some(call.clone()),
                _ => None,
            })
            .expect("ordinary call fixture");
        call.workspace_effect = Some(runmat_builtins::BuiltinWorkspaceEffect::DynamicEval);
        walk_spawn_rvalue(
            body,
            &MirRvalue::Call(call.clone()),
            Span::default(),
            &mut state,
            &mut |_, _, _| {},
        );
        assert!(
            state.iter().all(|fact| fact.len() == 1),
            "foreign evaluation does not rebind current values"
        );
        call.workspace_effect =
            Some(runmat_builtins::BuiltinWorkspaceEffect::DynamicEvalCurrentWorkspace);
        walk_spawn_rvalue(
            body,
            &MirRvalue::Call(call),
            Span::default(),
            &mut state,
            &mut |_, _, _| {},
        );
        let bindings = super::super::workspace_effect::current_scope_bindings(body);
        assert!(!bindings.is_empty());
        for local in &body.locals {
            assert_eq!(state[local.id.0].is_empty(), bindings.contains(&local.id));
        }
    }
}

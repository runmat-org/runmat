use crate::{
    BasicBlock, BasicBlockId, MirAggregateKind, MirAssembly, MirBody, MirCallArg, MirDiagnostic,
    MirDiagnosticSeverity, MirIndexComponent, MirIndexing, MirLocalId, MirLocalKind, MirOperand,
    MirPlace, MirRvalue, MirStmt, MirStmtKind, MirTerminatorKind,
};
use runmat_hir::Span;
use runmat_types::{
    infer_binary, infer_call, infer_cell_aggregate, infer_index, infer_index_mutation,
    infer_literal, infer_member_read, infer_member_write, infer_range, infer_struct,
    infer_tensor_aggregate, infer_unary, AliasFact, CallContract, CallRequest, CallableFact,
    CallableIdentity, CertaintyFact, ContiguityFact, DynamicReason, ExecutionFact, FactJoin,
    FutureStateFact, IndexSelectorFact, InvalidationVector, LayoutFact, LiteralContext,
    LiteralValue, MutationContract, MutationFact, OutputSelection, RangeStepFact, ResidencyFact,
    ShapeFact, SpawnSafetyFact, StorageFact, ValueFact, ValueKindFact, ViewFact,
};
use std::collections::{HashMap, VecDeque};

use super::{
    spawn_safety::{analyze_assembly_spawn_boundaries, diagnose_spawn_safety},
    AnalysisStore, InitFact, MirLocalFact, MirLocalKey,
};

#[derive(Debug, Clone)]
struct InitDataflowResult {
    in_states: Vec<Option<Vec<InitFact>>>,
}

type SimpleValueFact = ValueFact;

fn analyze_body(body: &MirBody, store: &mut AnalysisStore) {
    let local_facts = compute_simple_local_facts(body);

    for local in &body.locals {
        store.mir_locals.insert(
            MirLocalKey {
                function: body.function,
                local: local.id,
            },
            MirLocalFact {
                value: local_facts[local.id.0]
                    .clone()
                    .unwrap_or_else(dynamic_value),
            },
        );
    }
}

fn compute_simple_local_facts(body: &MirBody) -> Vec<Option<SimpleValueFact>> {
    let local_count = body.locals.len();
    if body.blocks.is_empty() {
        return Vec::new();
    }

    let block_index: HashMap<BasicBlockId, usize> = body
        .blocks
        .iter()
        .enumerate()
        .map(|(idx, block)| (block.id, idx))
        .collect();
    let mut in_states: Vec<Option<Vec<Option<SimpleValueFact>>>> = vec![None; body.blocks.len()];
    let mut out_states = vec![vec![None; local_count]; body.blocks.len()];
    in_states[0] = Some(vec![None; local_count]);

    let mut worklist = VecDeque::from([0usize]);
    while let Some(block_idx) = worklist.pop_front() {
        let Some(input) = in_states[block_idx].clone() else {
            continue;
        };
        let block = &body.blocks[block_idx];
        let output = transfer_fact_block(block, input);
        out_states[block_idx] = output.clone();

        for successor in successors(&block.terminator.kind) {
            let Some(&successor_idx) = block_index.get(&successor) else {
                continue;
            };
            let changed = match &mut in_states[successor_idx] {
                Some(existing) => join_fact_state(existing, &output),
                slot @ None => {
                    *slot = Some(output.clone());
                    true
                }
            };
            if changed {
                worklist.push_back(successor_idx);
            }
        }
    }

    let mut final_facts = vec![None; local_count];
    for facts in out_states {
        join_fact_state(&mut final_facts, &facts);
    }
    final_facts
}

fn transfer_fact_block(
    block: &BasicBlock,
    mut facts: Vec<Option<SimpleValueFact>>,
) -> Vec<Option<SimpleValueFact>> {
    let mut pending_mutation = None;
    for stmt in &block.statements {
        match &stmt.kind {
            MirStmtKind::Assign { place, value } => {
                let assigned = simple_rvalue_fact(value, &facts);
                assign_place_fact(
                    place,
                    assigned,
                    pending_mutation.take().as_ref(),
                    &mut facts,
                );
            }
            MirStmtKind::MultiAssign { targets, value } => {
                let outputs = if let MirRvalue::Call(call) = value {
                    let mut selection = OutputSelection::new(call.requested_outputs);
                    for (index, target) in targets.targets.iter().enumerate() {
                        if matches!(target, crate::MirOutputTarget::Discard) {
                            selection.discarded.insert(index);
                        }
                    }
                    call_outputs(call, &facts, selection)
                } else {
                    rvalue_outputs(value, &facts)
                };
                for (index, target) in targets.targets.iter().enumerate() {
                    if let crate::MirOutputTarget::Place(MirPlace::Local(local)) = target {
                        facts[local.0] =
                            Some(outputs.get(index).cloned().unwrap_or_else(dynamic_value));
                    }
                }
            }
            MirStmtKind::PlaceMutation(mutation) => pending_mutation = Some(mutation.clone()),
            MirStmtKind::Expr(_)
            | MirStmtKind::WorkspaceEffect { .. }
            | MirStmtKind::EnvironmentEffect(_) => {}
        }
    }
    match &block.terminator.kind {
        MirTerminatorKind::Await {
            result: Some(place),
            ..
        } => {
            if let Some(local) = place_root(place) {
                facts[local.0] = Some(dynamic_value());
            }
        }
        MirTerminatorKind::For { binding, .. } => {
            facts[binding.0] = Some(dynamic_value());
        }
        _ => {}
    }
    facts
}

fn assign_place_fact(
    place: &MirPlace,
    assigned: ValueFact,
    mutation: Option<&crate::MirPlaceMutation>,
    facts: &mut [Option<SimpleValueFact>],
) {
    match place {
        MirPlace::Local(local) => facts[local.0] = Some(assigned),
        MirPlace::Member(base, member) => {
            let current = place_fact(base, facts);
            let allow_creation = mutation.is_some_and(|mutation| {
                matches!(
                    mutation.creation_policy,
                    runmat_types::AssignmentCreationPolicy::CreateStructFieldPath
                )
            });
            let updated = infer_member_write(&current, member, &assigned, allow_creation).fact;
            assign_place_fact(base, updated, mutation, facts);
        }
        MirPlace::Index(base, indexing) => {
            let current = place_fact(base, facts);
            let mut contract = mutation.map_or(
                MutationContract {
                    kind: runmat_types::PlaceMutationKind::IndexedAssign,
                    creation: runmat_types::AssignmentCreationPolicy::CreateArrayByIndex,
                    shape: runmat_types::AssignmentShapePolicy::MatlabCompatible,
                },
                |mutation| MutationContract {
                    kind: mutation.kind,
                    creation: mutation.creation_policy,
                    shape: mutation.shape_policy,
                },
            );
            if matches!(indexing.kind, runmat_types::IndexKind::Brace)
                && !matches!(contract.kind, runmat_types::PlaceMutationKind::Delete)
            {
                contract.kind = runmat_types::PlaceMutationKind::CellAssign;
            }
            let updated = infer_index_mutation(
                &current,
                &index_selectors(indexing, facts),
                &assigned,
                contract,
            )
            .fact;
            assign_place_fact(base, updated, mutation, facts);
        }
        MirPlace::DynamicMember(base, _) => {
            assign_place_fact(
                base,
                ValueFact::unknown(DynamicReason::DynamicDispatch),
                mutation,
                facts,
            );
        }
        MirPlace::Binding(_) => {}
    }
}

fn place_fact(place: &MirPlace, facts: &[Option<SimpleValueFact>]) -> ValueFact {
    match place {
        MirPlace::Local(local) => facts
            .get(local.0)
            .and_then(Clone::clone)
            .unwrap_or_else(dynamic_value),
        MirPlace::Member(base, member) => infer_member_read(&place_fact(base, facts), member).fact,
        MirPlace::Index(base, indexing) => {
            infer_index(
                &place_fact(base, facts),
                indexing.kind,
                &index_selectors(indexing, facts),
                indexing.result_context,
            )
            .fact
        }
        MirPlace::DynamicMember(_, _) | MirPlace::Binding(_) => dynamic_value(),
    }
}

fn join_fact_state(
    existing: &mut [Option<SimpleValueFact>],
    incoming: &[Option<SimpleValueFact>],
) -> bool {
    let mut changed = false;
    for (slot, incoming) in existing.iter_mut().zip(incoming) {
        let mut joined = slot.clone();
        if let Some(incoming) = incoming.clone() {
            merge_simple_fact(&mut joined, incoming);
        }
        if *slot != joined {
            *slot = joined;
            changed = true;
        }
    }
    changed
}

fn simple_rvalue_fact(value: &MirRvalue, facts: &[Option<SimpleValueFact>]) -> SimpleValueFact {
    match value {
        MirRvalue::Use(operand) => simple_operand_fact(operand, facts),
        MirRvalue::Future { .. } => scalar_fact(ValueKindFact::Execution(ExecutionFact::Future {
            output: Box::new(dynamic_value()),
            state: FutureStateFact::Lazy,
        })),
        MirRvalue::Spawn(_) => scalar_fact(ValueKindFact::Execution(ExecutionFact::Task {
            output: Box::new(dynamic_value()),
            spawn_safety: SpawnSafetyFact::RequiresIsolation,
        })),
        MirRvalue::Aggregate {
            kind,
            rows,
            cols,
            elements,
        } => aggregate_fact(kind, *rows, *cols, elements, facts),
        MirRvalue::StructLiteral { fields } => {
            infer_struct(
                fields
                    .iter()
                    .map(|(name, operand)| (name.0.clone(), simple_operand_fact(operand, facts)))
                    .collect(),
            )
            .fact
        }
        MirRvalue::ObjectLiteral { class_name, fields } => {
            let properties = fields
                .iter()
                .map(|(name, operand)| (name.0.clone(), simple_operand_fact(operand, facts)))
                .collect();
            scalar_fact(ValueKindFact::Object(runmat_types::ObjectFact {
                class: None,
                runtime_class: Some(class_name.clone()),
                properties,
                properties_complete: true,
                handle_semantics: None,
            }))
        }
        MirRvalue::Binary(left, operator, right) => {
            infer_binary(
                *operator,
                &simple_operand_fact(left, facts),
                &simple_operand_fact(right, facts),
            )
            .fact
        }
        MirRvalue::ShortCircuit { .. } => scalar_fact(ValueKindFact::Logical),
        MirRvalue::Unary(operator, operand) => {
            infer_unary(*operator, &simple_operand_fact(operand, facts)).fact
        }
        MirRvalue::Range { start, step, end } => {
            let step = match step {
                None => RangeStepFact::Implicit,
                Some(step) => numeric_operand(step)
                    .map(RangeStepFact::Known)
                    .unwrap_or(RangeStepFact::Unknown),
            };
            infer_range(numeric_operand(start), step, numeric_operand(end)).fact
        }
        MirRvalue::Index { base, indexing } => {
            infer_index(
                &simple_operand_fact(base, facts),
                indexing.kind,
                &index_selectors(indexing, facts),
                indexing.result_context,
            )
            .fact
        }
        MirRvalue::Member { base, member } => {
            infer_member_read(&simple_operand_fact(base, facts), member).fact
        }
        MirRvalue::DynamicMember { .. }
        | MirRvalue::WorkspaceFirstStaticProperty { .. }
        | MirRvalue::MetaClass(_)
        | MirRvalue::Colon
        | MirRvalue::End => dynamic_value(),
        MirRvalue::Call(call) => rvalue_outputs(value, facts)
            .into_iter()
            .next()
            .unwrap_or_else(|| {
                if call.requested_outputs.fixed_count() == 0 {
                    scalar_fact(ValueKindFact::Void)
                } else {
                    dynamic_value()
                }
            }),
        MirRvalue::Distributed(_) | MirRvalue::Collective(_) => dynamic_value(),
    }
}

fn aggregate_fact(
    kind: &MirAggregateKind,
    rows: usize,
    cols: usize,
    elements: &[MirOperand],
    facts: &[Option<SimpleValueFact>],
) -> SimpleValueFact {
    let rows = if rows == 0 {
        Vec::new()
    } else {
        elements
            .chunks(cols.max(1))
            .map(|row| {
                row.iter()
                    .map(|operand| simple_operand_fact(operand, facts))
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>()
    };
    match kind {
        MirAggregateKind::Tensor => infer_tensor_aggregate(&rows).fact,
        MirAggregateKind::Cell => infer_cell_aggregate(&rows).fact,
    }
}

fn simple_operand_fact(operand: &MirOperand, facts: &[Option<SimpleValueFact>]) -> SimpleValueFact {
    match operand {
        MirOperand::Local(local) => facts
            .get(local.0)
            .and_then(Clone::clone)
            .unwrap_or_else(dynamic_value),
        MirOperand::Constant(constant) => infer_literal(&literal_value(constant)).fact,
        MirOperand::FunctionHandle(
            CallableIdentity::BoundFunction(function)
            | CallableIdentity::ExternalFunction { function, .. },
        )
        | MirOperand::FunctionHandle(CallableIdentity::AnonymousFunction(function)) => {
            scalar_fact(ValueKindFact::Callable(CallableFact {
                identity: Some(CallableIdentity::BoundFunction(*function)),
                parameters: Vec::new(),
                parameters_complete: false,
                outputs: Vec::new(),
                outputs_complete: false,
                variadic_inputs: true,
                variadic_outputs: true,
                captures: Vec::new(),
                captures_complete: true,
            }))
        }
        MirOperand::FunctionHandle(identity) => {
            scalar_fact(ValueKindFact::Callable(CallableFact {
                identity: Some(identity.clone()),
                parameters: Vec::new(),
                parameters_complete: false,
                outputs: Vec::new(),
                outputs_complete: false,
                variadic_inputs: true,
                variadic_outputs: true,
                captures: Vec::new(),
                captures_complete: true,
            }))
        }
    }
}

fn rvalue_outputs(value: &MirRvalue, facts: &[Option<SimpleValueFact>]) -> Vec<SimpleValueFact> {
    let MirRvalue::Call(call) = value else {
        let fact = simple_rvalue_fact(value, facts);
        return match fact.kind {
            ValueKindFact::OutputList(list) => list.outputs,
            _ => vec![fact],
        };
    };
    call_outputs(call, facts, OutputSelection::new(call.requested_outputs))
}

fn call_outputs(
    call: &crate::MirCall,
    facts: &[Option<SimpleValueFact>],
    output_selection: OutputSelection,
) -> Vec<SimpleValueFact> {
    let request = CallRequest {
        arguments: call
            .args
            .iter()
            .map(|argument| simple_operand_fact(argument.operand(), facts))
            .collect(),
        literals: LiteralContext::new(
            call.args
                .iter()
                .map(|argument| literal_operand(argument.operand()))
                .collect(),
        ),
        outputs: output_selection,
    };
    infer_call(
        &CallContract::dynamic(DynamicReason::UnresolvedCallable),
        &request,
    )
    .outputs
}

fn literal_value(constant: &crate::MirConstant) -> LiteralValue {
    match constant {
        crate::MirConstant::Number(value) => LiteralValue::Real {
            text: value.clone(),
            class: runmat_types::NumericClass::Double,
        },
        crate::MirConstant::String(value) => {
            if value.0.starts_with('\'') {
                LiteralValue::Character(value.0.trim_matches('\'').to_owned())
            } else {
                LiteralValue::String(value.0.trim_matches('"').to_owned())
            }
        }
        crate::MirConstant::Symbol(value) => LiteralValue::Symbolic(value.0.clone()),
        crate::MirConstant::Bool(value) => LiteralValue::Bool(*value),
        crate::MirConstant::EmptyArray => LiteralValue::Empty,
    }
}

fn literal_operand(operand: &MirOperand) -> LiteralValue {
    match operand {
        MirOperand::Constant(constant) => literal_value(constant),
        MirOperand::Local(_) | MirOperand::FunctionHandle(_) => LiteralValue::Unknown,
    }
}

fn numeric_operand(operand: &MirOperand) -> Option<f64> {
    match operand {
        MirOperand::Constant(crate::MirConstant::Number(value)) => value.parse().ok(),
        _ => None,
    }
}

fn index_selectors(
    indexing: &MirIndexing,
    facts: &[Option<SimpleValueFact>],
) -> Vec<IndexSelectorFact> {
    indexing
        .components
        .iter()
        .map(|component| match component {
            MirIndexComponent::Colon => IndexSelectorFact::Colon,
            MirIndexComponent::End { offset, .. } => IndexSelectorFact::End { offset: *offset },
            MirIndexComponent::Expr(operand) => {
                if let Some(index) =
                    numeric_operand(operand).filter(|index| *index >= 1.0 && index.fract() == 0.0)
                {
                    IndexSelectorFact::KnownOneBasedIndex(index as usize)
                } else {
                    let fact = simple_operand_fact(operand, facts);
                    if matches!(fact.kind, ValueKindFact::Logical) {
                        IndexSelectorFact::Logical(fact)
                    } else {
                        IndexSelectorFact::Numeric(fact)
                    }
                }
            }
        })
        .collect()
}

fn scalar_fact(kind: ValueKindFact) -> SimpleValueFact {
    value_fact(kind, ShapeFact::Scalar, StorageFact::Scalar)
}

fn value_fact(kind: ValueKindFact, shape: ShapeFact, storage: StorageFact) -> SimpleValueFact {
    ValueFact {
        kind,
        shape,
        storage,
        layout: LayoutFact::ColumnMajor,
        contiguity: ContiguityFact::Contiguous,
        view: ViewFact::Materialized,
        residency: ResidencyFact::Host,
        alias: AliasFact::Unique,
        mutation: MutationFact::ValueSemantics,
        certainty: CertaintyFact::Proven,
        invalidation: InvalidationVector::default(),
    }
}

fn dynamic_value() -> SimpleValueFact {
    ValueFact::unknown(DynamicReason::Unspecified)
}

fn merge_simple_fact(slot: &mut Option<SimpleValueFact>, incoming: SimpleValueFact) {
    match slot {
        Some(existing) => {
            *existing = existing.join(&incoming);
        }
        None => *slot = Some(incoming),
    }
}

pub fn analyze_assembly(assembly: &MirAssembly) -> AnalysisStore {
    let mut store = AnalysisStore::default();
    for body in assembly.bodies.values() {
        analyze_body(body, &mut store);
        store.diagnostics.extend(diagnose_uninitialized_reads(body));
        store.diagnostics.extend(diagnose_semantic_misuse(body));
    }
    let boundaries = analyze_assembly_spawn_boundaries(assembly);
    for boundaries in boundaries.values() {
        store.diagnostics.extend(diagnose_spawn_safety(boundaries));
    }
    store
}

pub(super) fn diagnose_uninitialized_reads(body: &MirBody) -> Vec<MirDiagnostic> {
    let result = compute_init_dataflow(body);
    let mut diagnostics = Vec::new();

    for (idx, block) in body.blocks.iter().enumerate() {
        let Some(mut state) = result.in_states[idx].clone() else {
            continue;
        };
        diagnose_block(block, &mut state, &mut diagnostics);
    }

    diagnostics
}

pub(super) fn diagnose_semantic_misuse(body: &MirBody) -> Vec<MirDiagnostic> {
    let mut diagnostics = Vec::new();
    for block in &body.blocks {
        for stmt in &block.statements {
            match &stmt.kind {
                MirStmtKind::Assign { value, .. }
                | MirStmtKind::MultiAssign { value, .. }
                | MirStmtKind::Expr(value) => {
                    diagnose_rvalue_semantics(value, stmt.span, &mut diagnostics)
                }
                MirStmtKind::PlaceMutation(mutation) => {
                    if matches!(mutation.kind, runmat_hir::PlaceMutationKind::Delete)
                        && !matches!(
                            mutation.creation_policy,
                            runmat_hir::AssignmentCreationPolicy::ExistingOnly
                        )
                    {
                        diagnostics.push(hir_diagnostic(
                            "RM-MIR0007",
                            "delete assignment cannot create a target",
                            "delete assignments require an existing indexed target",
                            stmt.span,
                            "assignment-semantics",
                        ));
                    }
                }
                MirStmtKind::WorkspaceEffect { effect, .. } => {
                    if matches!(effect, runmat_hir::WorkspaceEffect::DynamicEval) {
                        diagnostics.push(hir_diagnostic(
                            "RM-MIR0008",
                            "dynamic workspace evaluation blocks static analysis",
                            "prefer explicit bindings over eval-style workspace mutation",
                            stmt.span,
                            "workspace-effect",
                        ));
                    }
                }
                MirStmtKind::EnvironmentEffect(_) => diagnostics.push(hir_diagnostic(
                    "RM-MIR0009",
                    "environment mutation invalidates dynamic lookup assumptions",
                    "avoid path, cwd, or function-cache mutation in analyzable regions",
                    stmt.span,
                    "environment-effect",
                )),
            }
        }
    }
    diagnostics
}

fn diagnose_rvalue_semantics(value: &MirRvalue, span: Span, diagnostics: &mut Vec<MirDiagnostic>) {
    match value {
        MirRvalue::Call(call) => {
            if matches!(call.syntax, runmat_hir::CallSyntax::Command)
                && !matches!(
                    call.requested_outputs,
                    runmat_hir::RequestedOutputCount::Zero
                )
            {
                diagnostics.push(hir_diagnostic(
                    "RM-MIR0005",
                    "command syntax cannot request output values",
                    "use function-call syntax when outputs are required",
                    span,
                    "command-syntax",
                ));
            }
            if matches!(
                call.requested_outputs,
                runmat_hir::RequestedOutputCount::Zero
            ) && call
                .args
                .iter()
                .any(|arg| matches!(arg, MirCallArg::Expansion { .. }))
            {
                diagnostics.push(hir_diagnostic(
                    "RM-MIR0004",
                    "comma-list expansion is not valid for a zero-output call",
                    "consume comma-list expansions in value-producing call contexts",
                    span,
                    "comma-list",
                ));
            }
        }
        MirRvalue::Index { indexing, .. } => {
            if indexing
                .components
                .iter()
                .any(|component| matches!(component, MirIndexComponent::End { dim: None, .. }))
            {
                diagnostics.push(hir_diagnostic(
                    "RM-MIR0006",
                    "symbolic end requires an index dimension context",
                    "resolve end against the indexed value and dimension before runtime lowering",
                    span,
                    "indexing-semantics",
                ));
            }
        }
        _ => {}
    }
}

fn hir_diagnostic(
    code: &'static str,
    message: &'static str,
    help: &'static str,
    span: Span,
    category: &'static str,
) -> MirDiagnostic {
    MirDiagnostic::new(code, MirDiagnosticSeverity::Warning, message, span)
        .with_primary_label("semantic marker recorded here")
        .with_help(help)
        .with_category(category)
}

fn compute_init_dataflow(body: &MirBody) -> InitDataflowResult {
    let local_count = body.locals.len();
    if body.blocks.is_empty() {
        return InitDataflowResult {
            in_states: Vec::new(),
        };
    }

    let block_index: HashMap<BasicBlockId, usize> = body
        .blocks
        .iter()
        .enumerate()
        .map(|(idx, block)| (block.id, idx))
        .collect();

    let mut initial = vec![InitFact::Unassigned; local_count];
    for local in &body.locals {
        if matches!(local.kind, MirLocalKind::Parameter | MirLocalKind::Capture) {
            initial[local.id.0] = InitFact::DefinitelyAssigned;
        }
    }

    let mut in_states: Vec<Option<Vec<InitFact>>> = vec![None; body.blocks.len()];
    let mut out_states: Vec<Vec<InitFact>> =
        vec![vec![InitFact::Unassigned; local_count]; body.blocks.len()];
    in_states[0] = Some(initial);

    let mut worklist = VecDeque::from([0usize]);
    while let Some(block_idx) = worklist.pop_front() {
        let Some(input) = in_states[block_idx].clone() else {
            continue;
        };
        let block = &body.blocks[block_idx];
        let output = transfer_block(block, input);
        out_states[block_idx] = output.clone();

        for successor in successors(&block.terminator.kind) {
            let Some(&successor_idx) = block_index.get(&successor) else {
                continue;
            };
            let changed = match &mut in_states[successor_idx] {
                Some(existing) => join_into(existing, &output),
                slot @ None => {
                    *slot = Some(output.clone());
                    true
                }
            };
            if changed {
                worklist.push_back(successor_idx);
            }
        }
    }

    InitDataflowResult { in_states }
}

fn transfer_block(block: &crate::BasicBlock, mut state: Vec<InitFact>) -> Vec<InitFact> {
    for stmt in &block.statements {
        match &stmt.kind {
            MirStmtKind::Assign { place, .. } => mark_place_assigned(place, &mut state),
            MirStmtKind::MultiAssign { targets, .. } => {
                for target in &targets.targets {
                    if let crate::MirOutputTarget::Place(place) = target {
                        mark_place_assigned(place, &mut state);
                    }
                }
            }
            MirStmtKind::Expr(_)
            | MirStmtKind::PlaceMutation(_)
            | MirStmtKind::WorkspaceEffect { .. }
            | MirStmtKind::EnvironmentEffect(_) => {}
        }
    }

    match &block.terminator.kind {
        MirTerminatorKind::For { binding, .. } => {
            state[binding.0] = InitFact::DefinitelyAssigned;
        }
        MirTerminatorKind::Await {
            result: Some(place),
            ..
        } => {
            mark_place_assigned(place, &mut state);
        }
        MirTerminatorKind::Await { result: None, .. } => {}
        _ => {}
    }
    state
}

fn diagnose_block(
    block: &BasicBlock,
    state: &mut [InitFact],
    diagnostics: &mut Vec<MirDiagnostic>,
) {
    for stmt in &block.statements {
        diagnose_stmt(stmt, state, diagnostics);
    }

    match &block.terminator.kind {
        MirTerminatorKind::Branch { cond, .. } => {
            diagnose_operand_read(cond, state, block.terminator.span, diagnostics);
        }
        MirTerminatorKind::Switch { discr, cases, .. } => {
            diagnose_operand_read(discr, state, block.terminator.span, diagnostics);
            for (case, _) in cases {
                diagnose_operand_read(case, state, block.terminator.span, diagnostics);
            }
        }
        MirTerminatorKind::For {
            binding, iterable, ..
        } => {
            diagnose_rvalue_reads(iterable, state, block.terminator.span, diagnostics);
            state[binding.0] = InitFact::DefinitelyAssigned;
        }
        MirTerminatorKind::ParFor {
            binding,
            iterable,
            maximum_workers,
            ..
        } => {
            diagnose_rvalue_reads(iterable, state, block.terminator.span, diagnostics);
            if let Some(workers) = maximum_workers.as_deref() {
                diagnose_rvalue_reads(workers, state, block.terminator.span, diagnostics);
            }
            state[binding.0] = InitFact::DefinitelyAssigned;
        }
        MirTerminatorKind::Spmd { header, .. } => match header.as_ref() {
            crate::parallel::MirSpmdHeader::Default => {}
            crate::parallel::MirSpmdHeader::One(value) => {
                diagnose_rvalue_reads(value, state, block.terminator.span, diagnostics);
            }
            crate::parallel::MirSpmdHeader::Two(first, second) => {
                diagnose_rvalue_reads(first, state, block.terminator.span, diagnostics);
                diagnose_rvalue_reads(second, state, block.terminator.span, diagnostics);
            }
            crate::parallel::MirSpmdHeader::Three(first, second, third) => {
                diagnose_rvalue_reads(first, state, block.terminator.span, diagnostics);
                diagnose_rvalue_reads(second, state, block.terminator.span, diagnostics);
                diagnose_rvalue_reads(third, state, block.terminator.span, diagnostics);
            }
        },
        MirTerminatorKind::Return(outputs) => {
            for output in outputs {
                diagnose_operand_read(output, state, block.terminator.span, diagnostics);
            }
        }
        MirTerminatorKind::Await { future, result, .. } => {
            diagnose_operand_read(future, state, block.terminator.span, diagnostics);
            if let Some(place) = result {
                diagnose_place_reads(place, state, block.terminator.span, diagnostics);
                mark_place_assigned(place, state);
            }
        }
        MirTerminatorKind::Goto(_)
        | MirTerminatorKind::TryCatch { .. }
        | MirTerminatorKind::Unreachable => {}
    }
}

fn diagnose_stmt(stmt: &MirStmt, state: &mut [InitFact], diagnostics: &mut Vec<MirDiagnostic>) {
    match &stmt.kind {
        MirStmtKind::Assign { place, value } => {
            diagnose_rvalue_reads(value, state, stmt.span, diagnostics);
            diagnose_place_reads(place, state, stmt.span, diagnostics);
            mark_place_assigned(place, state);
        }
        MirStmtKind::MultiAssign { targets, value } => {
            diagnose_rvalue_reads(value, state, stmt.span, diagnostics);
            for target in &targets.targets {
                if let crate::MirOutputTarget::Place(place) = target {
                    diagnose_place_reads(place, state, stmt.span, diagnostics);
                    mark_place_assigned(place, state);
                }
            }
        }
        MirStmtKind::Expr(value) => diagnose_rvalue_reads(value, state, stmt.span, diagnostics),
        MirStmtKind::PlaceMutation(_)
        | MirStmtKind::WorkspaceEffect { .. }
        | MirStmtKind::EnvironmentEffect(_) => {}
    }
}

fn diagnose_rvalue_reads(
    value: &MirRvalue,
    state: &mut [InitFact],
    span: Span,
    diagnostics: &mut Vec<MirDiagnostic>,
) {
    match value {
        MirRvalue::Use(operand) | MirRvalue::Unary(_, operand) | MirRvalue::Spawn(operand) => {
            diagnose_operand_read(operand, state, span, diagnostics);
        }
        MirRvalue::Binary(left, _, right) => {
            diagnose_operand_read(left, state, span, diagnostics);
            diagnose_operand_read(right, state, span, diagnostics);
        }
        MirRvalue::ShortCircuit {
            left,
            right_temps,
            right,
            ..
        } => {
            diagnose_operand_read(left, state, span, diagnostics);
            for stmt in right_temps {
                diagnose_stmt(stmt, state, diagnostics);
            }
            diagnose_operand_read(right, state, span, diagnostics);
        }
        MirRvalue::Range { start, step, end } => {
            diagnose_operand_read(start, state, span, diagnostics);
            if let Some(step) = step {
                diagnose_operand_read(step, state, span, diagnostics);
            }
            diagnose_operand_read(end, state, span, diagnostics);
        }
        MirRvalue::Call(call) => {
            for arg in &call.args {
                diagnose_operand_read(arg.operand(), state, span, diagnostics);
            }
        }
        MirRvalue::Aggregate { elements, .. } => {
            for element in elements {
                diagnose_operand_read(element, state, span, diagnostics);
            }
        }
        MirRvalue::StructLiteral { fields } | MirRvalue::ObjectLiteral { fields, .. } => {
            for (_, value) in fields {
                diagnose_operand_read(value, state, span, diagnostics);
            }
        }
        MirRvalue::Index { base, indexing } => {
            diagnose_operand_read(base, state, span, diagnostics);
            diagnose_indexing_reads(indexing, state, span, diagnostics);
        }
        MirRvalue::Member { base, .. } => diagnose_operand_read(base, state, span, diagnostics),
        MirRvalue::DynamicMember { base, member } => {
            diagnose_operand_read(base, state, span, diagnostics);
            diagnose_operand_read(member, state, span, diagnostics);
        }
        MirRvalue::WorkspaceFirstStaticProperty { .. } => {}
        MirRvalue::Future { args, .. } => {
            for arg in args {
                diagnose_operand_read(arg.operand(), state, span, diagnostics);
            }
        }
        MirRvalue::MetaClass(_) | MirRvalue::Colon | MirRvalue::End => {}
        MirRvalue::Distributed(operation) => match operation {
            crate::parallel::MirDistributedOp::Create { input, .. } => {
                diagnose_operand_read(input, state, span, diagnostics);
            }
            crate::parallel::MirDistributedOp::LocalPart { .. }
            | crate::parallel::MirDistributedOp::Materialize { .. }
            | crate::parallel::MirDistributedOp::Redistribute { .. } => {}
        },
        MirRvalue::Collective(operation) => match operation {
            crate::parallel::MirCollectiveOp::Broadcast { input, .. }
            | crate::parallel::MirCollectiveOp::Gather { input, .. }
            | crate::parallel::MirCollectiveOp::Scatter { input, .. }
            | crate::parallel::MirCollectiveOp::AllGather { input, .. }
            | crate::parallel::MirCollectiveOp::Reduce { input, .. }
            | crate::parallel::MirCollectiveOp::AllReduce { input, .. }
            | crate::parallel::MirCollectiveOp::Send { input, .. } => {
                diagnose_operand_read(input, state, span, diagnostics);
            }
            crate::parallel::MirCollectiveOp::Barrier { .. }
            | crate::parallel::MirCollectiveOp::Receive { .. } => {}
        },
    }
}

fn diagnose_place_reads(
    place: &MirPlace,
    state: &[InitFact],
    span: Span,
    diagnostics: &mut Vec<MirDiagnostic>,
) {
    match place {
        MirPlace::Local(_) | MirPlace::Binding(_) => {}
        MirPlace::Member(base, _) => diagnose_place_reads(base, state, span, diagnostics),
        MirPlace::DynamicMember(base, member) => {
            diagnose_place_reads(base, state, span, diagnostics);
            diagnose_operand_read(member, state, span, diagnostics);
        }
        MirPlace::Index(base, indexing) => {
            diagnose_place_reads(base, state, span, diagnostics);
            diagnose_indexing_reads(indexing, state, span, diagnostics);
        }
    }
}

fn diagnose_indexing_reads(
    indexing: &MirIndexing,
    state: &[InitFact],
    span: Span,
    diagnostics: &mut Vec<MirDiagnostic>,
) {
    for component in &indexing.components {
        match component {
            MirIndexComponent::Expr(operand) => {
                diagnose_operand_read(operand, state, span, diagnostics);
            }
            MirIndexComponent::Colon | MirIndexComponent::End { .. } => {}
        }
    }
}

fn diagnose_operand_read(
    operand: &MirOperand,
    state: &[InitFact],
    span: Span,
    diagnostics: &mut Vec<MirDiagnostic>,
) {
    let MirOperand::Local(local) = operand else {
        return;
    };
    match state[local.0] {
        InitFact::DefinitelyAssigned => {}
        InitFact::Unassigned => diagnostics.push(init_diagnostic(
            "RM-MIR0001",
            "local may be read before it is assigned",
            "this local is read before any assignment reaches this point",
            span,
        )),
        InitFact::MaybeAssigned => diagnostics.push(init_diagnostic(
            "RM-MIR0002",
            "local may be read before assignment on some control-flow paths",
            "this local is not definitely assigned on every path reaching this point",
            span,
        )),
    }
}

fn init_diagnostic(
    code: &'static str,
    message: &'static str,
    label: &'static str,
    span: Span,
) -> MirDiagnostic {
    let help = match code {
        "RM-MIR0001" => "assign this local before reading it",
        "RM-MIR0002" => "assign this local on every control-flow path before reading it",
        _ => "ensure the local is initialized before use",
    };
    MirDiagnostic::new(code, MirDiagnosticSeverity::Error, message, span)
        .with_primary_label(label)
        .with_help(help)
        .with_category("definite-assignment")
}

fn mark_place_assigned(place: &MirPlace, state: &mut [InitFact]) {
    if let Some(local) = place_root(place) {
        state[local.0] = InitFact::DefinitelyAssigned;
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

fn join_into(existing: &mut [InitFact], incoming: &[InitFact]) -> bool {
    let mut changed = false;
    for (slot, incoming) in existing.iter_mut().zip(incoming) {
        let joined = join_init(*slot, *incoming);
        if *slot != joined {
            *slot = joined;
            changed = true;
        }
    }
    changed
}

fn join_init(left: InitFact, right: InitFact) -> InitFact {
    match (left, right) {
        (InitFact::Unassigned, InitFact::Unassigned) => InitFact::Unassigned,
        (InitFact::DefinitelyAssigned, InitFact::DefinitelyAssigned) => {
            InitFact::DefinitelyAssigned
        }
        _ => InitFact::MaybeAssigned,
    }
}

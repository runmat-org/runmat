use crate::{
    BasicBlock, BasicBlockId, MirAggregateKind, MirAssembly, MirBody, MirCallArg, MirDiagnostic,
    MirDiagnosticSeverity, MirIndexComponent, MirIndexing, MirLocalId, MirLocalKind, MirOperand,
    MirPlace, MirRvalue, MirStmtKind, MirTerminatorKind,
};
use runmat_hir::{
    AsyncValueFact, DimFact, EmptyArrayRole, ExpansionSemantics, FunctionHandleTarget, FutureFact,
    FutureStateFact, NumericClass, NumericDomain, OperatorKind, ShapeFact, Span, SpawnSafetyFact,
    TaskHandleFact, TensorElementDomainFact, TensorTypeFact, TypeFact, ValueFlowFact,
};
use std::collections::{HashMap, VecDeque};

use super::{
    analyze_liveness, analyze_spawn_boundaries_with_summaries, diagnose_spawn_safety,
    propagate_function_summaries, summarize_body, AnalysisStore, BindingFact, ExprFact, InitFact,
    MirLocalFact, MirLocalKey, ModuleSummary,
};

#[derive(Debug, Clone)]
struct InitDataflowResult {
    in_states: Vec<Option<Vec<InitFact>>>,
    final_state: Vec<InitFact>,
}

#[derive(Debug, Clone, PartialEq)]
struct SimpleValueFact {
    ty: TypeFact,
    shape: ShapeFact,
    value_flow: ValueFlowFact,
    async_value: Option<AsyncValueFact>,
    empty_array_role: Option<EmptyArrayRole>,
    expansion: Option<ExpansionSemantics>,
    operator: Option<OperatorKind>,
    tensor_element_domain: Option<TensorElementDomainFact>,
}

impl Default for SimpleValueFact {
    fn default() -> Self {
        Self {
            ty: TypeFact::Unknown,
            shape: ShapeFact::Unknown,
            value_flow: ValueFlowFact::UnknownList,
            async_value: None,
            empty_array_role: None,
            expansion: None,
            operator: None,
            tensor_element_domain: None,
        }
    }
}

pub fn analyze_body(body: &MirBody, store: &mut AnalysisStore) {
    let result = compute_init_dataflow(body);
    let local_facts = compute_simple_local_facts(body);

    for record in &body.source_map.statements {
        if let Some(expr) = record.expr {
            store
                .expressions
                .entry(expr)
                .or_insert_with(default_expr_fact);
        }
    }

    for local in &body.locals {
        let initialized = result.final_state[local.id.0];
        store.mir_locals.insert(
            MirLocalKey {
                function: body.function,
                local: local.id,
            },
            MirLocalFact {
                ty: local_facts[local.id.0].clone().unwrap_or_default().ty,
                shape: local_facts[local.id.0].clone().unwrap_or_default().shape,
                value_flow: local_facts[local.id.0]
                    .clone()
                    .unwrap_or_default()
                    .value_flow,
                async_value: local_facts[local.id.0]
                    .clone()
                    .unwrap_or_default()
                    .async_value,
                empty_array_role: local_facts[local.id.0]
                    .clone()
                    .unwrap_or_default()
                    .empty_array_role,
                expansion: local_facts[local.id.0]
                    .clone()
                    .unwrap_or_default()
                    .expansion,
                operator: local_facts[local.id.0].clone().unwrap_or_default().operator,
                tensor_element_domain: local_facts[local.id.0]
                    .clone()
                    .unwrap_or_default()
                    .tensor_element_domain,
                initialized,
            },
        );
        if let Some(binding) = local.binding {
            insert_binding_fact(
                store,
                binding,
                initialized,
                local_facts[local.id.0].clone().unwrap_or_default().ty,
            );
        }
    }

    for source in &body.source_map.locals {
        if let Some(expr) = source.expr {
            let fact = local_facts[source.local.0].clone().unwrap_or_default();
            store.expressions.insert(
                expr,
                ExprFact {
                    ty: fact.ty,
                    shape: fact.shape,
                    value_flow: fact.value_flow,
                    async_value: fact.async_value,
                    empty_array_role: fact.empty_array_role,
                    expansion: fact.expansion,
                    operator: fact.operator,
                    tensor_element_domain: fact.tensor_element_domain,
                },
            );
        }
    }
}

fn default_expr_fact() -> ExprFact {
    ExprFact {
        ty: TypeFact::Unknown,
        shape: ShapeFact::Unknown,
        value_flow: ValueFlowFact::UnknownList,
        async_value: None,
        empty_array_role: None,
        expansion: None,
        operator: None,
        tensor_element_domain: None,
    }
}

fn insert_binding_fact(
    store: &mut AnalysisStore,
    binding: runmat_hir::BindingId,
    initialized: InitFact,
    ty: TypeFact,
) {
    let incoming = BindingFact { ty, initialized };
    store
        .bindings
        .entry(binding)
        .and_modify(|existing| {
            existing.ty = join_type(&existing.ty, &incoming.ty);
            existing.initialized = join_init(existing.initialized, incoming.initialized);
        })
        .or_insert(incoming);
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
    for stmt in &block.statements {
        match &stmt.kind {
            MirStmtKind::Assign { place, value } => {
                if let MirPlace::Local(local) = place {
                    facts[local.0] = Some(simple_rvalue_fact(value));
                }
            }
            MirStmtKind::MultiAssign { targets, .. } => {
                for target in &targets.targets {
                    if let crate::MirOutputTarget::Place(MirPlace::Local(local)) = target {
                        facts[local.0] = Some(SimpleValueFact::default());
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
        MirTerminatorKind::Await {
            result: Some(place),
            ..
        } => {
            if let Some(local) = place_root(place) {
                facts[local.0] = Some(SimpleValueFact::default());
            }
        }
        MirTerminatorKind::For { binding, .. } => {
            facts[binding.0] = Some(SimpleValueFact::default());
        }
        _ => {}
    }
    facts
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

fn simple_rvalue_fact(value: &MirRvalue) -> SimpleValueFact {
    match value {
        MirRvalue::Use(operand) => simple_operand_fact(operand),
        MirRvalue::Future { .. } => SimpleValueFact {
            async_value: Some(AsyncValueFact::Future(FutureFact {
                output: Box::new(TypeFact::Unknown),
                state: FutureStateFact::Lazy,
            })),
            ..SimpleValueFact::default()
        },
        MirRvalue::Spawn(_) => SimpleValueFact {
            async_value: Some(AsyncValueFact::TaskHandle(TaskHandleFact {
                output: Box::new(TypeFact::Unknown),
                spawn_safety: SpawnSafetyFact::RequiresIsolation,
            })),
            ..SimpleValueFact::default()
        },
        MirRvalue::Aggregate {
            kind,
            rows,
            cols,
            elements,
        } => aggregate_fact(kind, *rows, *cols, elements),
        MirRvalue::Binary(_, op, _) | MirRvalue::Unary(op, _) => SimpleValueFact {
            operator: Some(op.clone()),
            expansion: Some(ExpansionSemantics::ImplicitExpansion),
            ..SimpleValueFact::default()
        },
        MirRvalue::Range { .. } => SimpleValueFact {
            expansion: Some(ExpansionSemantics::ExactShape),
            ..SimpleValueFact::default()
        },
        MirRvalue::Index { .. } => SimpleValueFact {
            value_flow: ValueFlowFact::UnknownList,
            ..SimpleValueFact::default()
        },
        MirRvalue::Member { .. }
        | MirRvalue::DynamicMember { .. }
        | MirRvalue::MetaClass(_)
        | MirRvalue::Colon
        | MirRvalue::End
        | MirRvalue::Call(_) => SimpleValueFact::default(),
    }
}

fn aggregate_fact(
    kind: &MirAggregateKind,
    rows: usize,
    cols: usize,
    elements: &[MirOperand],
) -> SimpleValueFact {
    let shape = ShapeFact::Shaped {
        dims: vec![DimFact::Known(rows), DimFact::Known(cols)],
    };
    match kind {
        MirAggregateKind::Tensor => {
            let element = tensor_element_domain(elements);
            let empty_array_role = (rows == 0 || cols == 0).then_some(EmptyArrayRole::EmptyValue);
            let mut fact = single_fact(
                TypeFact::Tensor(TensorTypeFact {
                    element: element.clone(),
                    shape: shape.clone(),
                }),
                shape.clone(),
            );
            fact.empty_array_role = empty_array_role;
            fact.tensor_element_domain = Some(element);
            fact
        }
        MirAggregateKind::Cell => single_fact(TypeFact::Cell, shape),
        MirAggregateKind::Struct | MirAggregateKind::ObjectArray(_) => SimpleValueFact::default(),
    }
}

fn tensor_element_domain(elements: &[MirOperand]) -> TensorElementDomainFact {
    if !elements.is_empty()
        && elements
            .iter()
            .all(|element| matches!(element, MirOperand::Constant(crate::MirConstant::Number(_))))
    {
        TensorElementDomainFact::Numeric {
            class: NumericClass::Double,
            domain: NumericDomain::Real,
        }
    } else {
        TensorElementDomainFact::Unknown
    }
}

fn simple_operand_fact(operand: &MirOperand) -> SimpleValueFact {
    match operand {
        MirOperand::Constant(crate::MirConstant::Number(_)) => {
            let ty = TypeFact::Numeric {
                class: NumericClass::Double,
                domain: NumericDomain::Real,
            };
            scalar_single_fact(ty)
        }
        MirOperand::Constant(crate::MirConstant::String(_)) => {
            scalar_single_fact(TypeFact::CharArray)
        }
        MirOperand::FunctionHandle(FunctionHandleTarget::Function(function))
        | MirOperand::FunctionHandle(FunctionHandleTarget::Anonymous(function)) => {
            scalar_single_fact(TypeFact::Function(*function))
        }
        _ => SimpleValueFact::default(),
    }
}

fn scalar_single_fact(ty: TypeFact) -> SimpleValueFact {
    single_fact(ty, ShapeFact::Scalar)
}

fn single_fact(ty: TypeFact, shape: ShapeFact) -> SimpleValueFact {
    SimpleValueFact {
        ty: ty.clone(),
        shape,
        value_flow: ValueFlowFact::Single(ty),
        async_value: None,
        empty_array_role: None,
        expansion: None,
        operator: None,
        tensor_element_domain: None,
    }
}

fn merge_simple_fact(slot: &mut Option<SimpleValueFact>, incoming: SimpleValueFact) {
    match slot {
        Some(existing) => {
            *existing = SimpleValueFact {
                ty: join_type(&existing.ty, &incoming.ty),
                shape: join_shape(&existing.shape, &incoming.shape),
                value_flow: join_value_flow(&existing.value_flow, &incoming.value_flow),
                async_value: join_async_value(&existing.async_value, &incoming.async_value),
                empty_array_role: join_option_fact(
                    &existing.empty_array_role,
                    &incoming.empty_array_role,
                ),
                expansion: join_option_fact(&existing.expansion, &incoming.expansion),
                operator: join_option_fact(&existing.operator, &incoming.operator),
                tensor_element_domain: join_option_fact(
                    &existing.tensor_element_domain,
                    &incoming.tensor_element_domain,
                ),
            };
        }
        None => *slot = Some(incoming),
    }
}

fn join_type(left: &TypeFact, right: &TypeFact) -> TypeFact {
    if left == right {
        left.clone()
    } else {
        TypeFact::Unknown
    }
}

fn join_shape(left: &ShapeFact, right: &ShapeFact) -> ShapeFact {
    if left == right {
        left.clone()
    } else {
        ShapeFact::Unknown
    }
}

fn join_value_flow(left: &ValueFlowFact, right: &ValueFlowFact) -> ValueFlowFact {
    if left == right {
        left.clone()
    } else {
        ValueFlowFact::UnknownList
    }
}

fn join_async_value(
    left: &Option<AsyncValueFact>,
    right: &Option<AsyncValueFact>,
) -> Option<AsyncValueFact> {
    if left == right {
        left.clone()
    } else {
        None
    }
}

fn join_option_fact<T: Clone + PartialEq>(left: &Option<T>, right: &Option<T>) -> Option<T> {
    if left == right {
        left.clone()
    } else {
        None
    }
}

pub fn analyze_assembly(assembly: &MirAssembly) -> AnalysisStore {
    let mut store = AnalysisStore::default();
    for body in assembly.bodies.values() {
        analyze_body(body, &mut store);
        summarize_body(body, &mut store);
        store.liveness.insert(body.function, analyze_liveness(body));
        store.diagnostics.extend(diagnose_uninitialized_reads(body));
        store.diagnostics.extend(diagnose_semantic_misuse(body));
    }
    propagate_function_summaries(&mut store);
    for body in assembly.bodies.values() {
        if let (Some(module), Some(summary)) = (
            body.source_map.module,
            store.functions.get(&body.function).cloned(),
        ) {
            insert_module_summary(&mut store, module, body, &summary);
        }
    }
    for body in assembly.bodies.values() {
        let boundaries = analyze_spawn_boundaries_with_summaries(body, &store.functions);
        store.diagnostics.extend(diagnose_spawn_safety(&boundaries));
        store.spawn_boundaries.insert(body.function, boundaries);
    }
    store
}

fn insert_module_summary(
    store: &mut AnalysisStore,
    module: runmat_hir::ModuleId,
    body: &MirBody,
    summary: &super::FunctionSummary,
) {
    store
        .modules
        .entry(module)
        .and_modify(|existing| {
            if !existing.functions.contains(&summary.function) {
                existing.functions.push(summary.function);
            }
            if existing.source_unit.is_none() {
                existing.source_unit = body.source_map.source_unit.clone();
            }
            existing.workspace.extend(summary.effects.workspace.clone());
            existing
                .environment
                .extend(summary.effects.environment.clone());
            existing.may_call_unknown |= summary.may_call_unknown;
        })
        .or_insert_with(|| ModuleSummary {
            module,
            functions: vec![summary.function],
            source_unit: body.source_map.source_unit.clone(),
            compatibility_mode: body.source_map.compatibility_mode.clone(),
            workspace: summary.effects.workspace.clone(),
            environment: summary.effects.environment.clone(),
            may_call_unknown: summary.may_call_unknown,
        });
}

pub fn diagnose_uninitialized_reads(body: &MirBody) -> Vec<MirDiagnostic> {
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

pub fn diagnose_semantic_misuse(body: &MirBody) -> Vec<MirDiagnostic> {
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
                        diagnostics.push(semantic_diagnostic(
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
                        diagnostics.push(semantic_diagnostic(
                            "RM-MIR0008",
                            "dynamic workspace evaluation blocks static analysis",
                            "prefer explicit bindings over eval-style workspace mutation",
                            stmt.span,
                            "workspace-effect",
                        ));
                    }
                }
                MirStmtKind::EnvironmentEffect(_) => diagnostics.push(semantic_diagnostic(
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
                diagnostics.push(semantic_diagnostic(
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
                diagnostics.push(semantic_diagnostic(
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
                diagnostics.push(semantic_diagnostic(
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

fn semantic_diagnostic(
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
            final_state: Vec::new(),
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

    let mut final_state: Option<Vec<InitFact>> = None;
    for (idx, block) in body.blocks.iter().enumerate() {
        if matches!(block.terminator.kind, MirTerminatorKind::Return(_)) {
            match &mut final_state {
                Some(state) => {
                    join_into(state, &out_states[idx]);
                }
                None => final_state = Some(out_states[idx].clone()),
            }
        }
    }
    InitDataflowResult {
        in_states,
        final_state: final_state.unwrap_or_else(|| vec![InitFact::Unassigned; local_count]),
    }
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
        MirTerminatorKind::Await { result, .. } => {
            if let Some(place) = result {
                mark_place_assigned(place, &mut state);
            }
        }
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

fn diagnose_rvalue_reads(
    value: &MirRvalue,
    state: &[InitFact],
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
        MirRvalue::Index { base, indexing } => {
            diagnose_operand_read(base, state, span, diagnostics);
            diagnose_indexing_reads(indexing, state, span, diagnostics);
        }
        MirRvalue::Member { base, .. } => diagnose_operand_read(base, state, span, diagnostics),
        MirRvalue::DynamicMember { base, member } => {
            diagnose_operand_read(base, state, span, diagnostics);
            diagnose_operand_read(member, state, span, diagnostics);
        }
        MirRvalue::Future { args, .. } => {
            for arg in args {
                diagnose_operand_read(arg.operand(), state, span, diagnostics);
            }
        }
        MirRvalue::MetaClass(_) | MirRvalue::Colon | MirRvalue::End => {}
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
            MirIndexComponent::Expr(operand) | MirIndexComponent::Logical(operand) => {
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
        MirTerminatorKind::TryCatch {
            try_block,
            catch_block,
            ..
        } => vec![*try_block, *catch_block],
        MirTerminatorKind::Await {
            resume, cleanup, ..
        } => cleanup.map_or_else(|| vec![*resume], |cleanup| vec![*resume, cleanup]),
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

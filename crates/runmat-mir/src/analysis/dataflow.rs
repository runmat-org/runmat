use crate::{
    BasicBlock, MirAggregateKind, MirAssembly, MirBody, MirDiagnostic, MirDiagnosticSeverity,
    MirIndexComponent, MirIndexing, MirLocalId, MirOperand, MirPlace, MirRvalue, MirStmt,
    MirStmtKind, MirTerminatorKind,
};
use runmat_hir::Span;
use runmat_types::{
    infer_binary, infer_cell_aggregate, infer_index, infer_index_mutation, infer_literal,
    infer_member_read, infer_member_write, infer_range, infer_struct, infer_tensor_aggregate,
    infer_unary, AliasFact, CallableFact, CallableIdentity, CertaintyFact, ContiguityFact,
    DynamicReason, ExecutionFact, FutureStateFact, IndexSelectorFact, InvalidationVector,
    LayoutFact, LiteralValue, MutationContract, MutationFact, NumericClass, RangeStepFact,
    ResidencyFact, ShapeFact, SpawnSafetyFact, StorageFact, ValueFact, ValueKindFact, ViewFact,
};

mod expression_regions;
mod initialization;
mod subscript_chain;
use super::workspace_effect::{current_scope_bindings, exposes_current_workspace};
use expression_regions::{diagnose_expression_steps, diagnose_subscript_chain_reads};
use initialization::compute_init_dataflow;
pub(crate) use subscript_chain::{SubscriptChainInference, SubscriptPathDisposition};

use super::{
    spawn_safety::{analyze_assembly_spawn_boundaries, diagnose_spawn_safety},
    InitFact,
};

#[cfg(test)]
#[path = "dataflow/workspace_binding_tests.rs"]
mod workspace_binding_tests;

pub(crate) fn assign_place_fact(
    place: &MirPlace,
    assigned: ValueFact,
    mutation: Option<&crate::MirPlaceMutation>,
    facts: &mut [Option<ValueFact>],
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

fn place_fact(place: &MirPlace, facts: &[Option<ValueFact>]) -> ValueFact {
    match place {
        MirPlace::Local(local) => facts
            .get(local.0)
            .and_then(Clone::clone)
            .unwrap_or_else(dynamic_value),
        MirPlace::Member(base, member) => {
            infer_member_read(
                &place_fact(base, facts),
                member,
                runmat_types::SequenceUse::RequireSingle,
            )
            .fact
        }
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

pub(crate) fn simple_rvalue_fact(value: &MirRvalue, facts: &[Option<ValueFact>]) -> ValueFact {
    simple_rvalue_inference(value, facts).fact
}

pub(crate) fn simple_rvalue_inference(
    value: &MirRvalue,
    facts: &[Option<ValueFact>],
) -> runmat_types::FactInference {
    match value {
        MirRvalue::Use(operand) => {
            runmat_types::FactInference::exact(simple_operand_fact(operand, facts))
        }
        MirRvalue::Future { .. } | MirRvalue::ExternalFuture { .. } => {
            runmat_types::FactInference::exact(scalar_fact(ValueKindFact::Execution(
                ExecutionFact::Future {
                    output: runmat_types::ValueSequenceFact::single(dynamic_value()),
                    state: FutureStateFact::Lazy,
                },
            )))
        }
        MirRvalue::Spawn(_) => runmat_types::FactInference::exact(scalar_fact(
            ValueKindFact::Execution(ExecutionFact::Task {
                output: runmat_types::ValueSequenceFact::single(dynamic_value()),
                spawn_safety: SpawnSafetyFact::RequiresIsolation,
            }),
        )),
        MirRvalue::Aggregate {
            kind,
            row_lengths,
            elements,
        } => aggregate_inference(kind, row_lengths, elements, facts),
        MirRvalue::StructLiteral { fields } => infer_struct(
            fields
                .iter()
                .map(|(name, operand)| (name.0.clone(), simple_operand_fact(operand, facts)))
                .collect(),
        ),
        MirRvalue::ObjectLiteral { class_name, fields } => {
            let properties = fields
                .iter()
                .map(|(name, operand)| (name.0.clone(), simple_operand_fact(operand, facts)))
                .collect();
            runmat_types::FactInference::exact(scalar_fact(ValueKindFact::Object(
                runmat_types::ObjectFact {
                    class: None,
                    runtime_class: Some(
                        runmat_types::ClassIdentity::from_qualified_name(class_name)
                            .expect("MIR object literals carry canonical class names"),
                    ),
                    properties,
                    properties_complete: true,
                    handle_semantics: None,
                },
            )))
        }
        MirRvalue::Binary(left, operator, right) => infer_binary(
            *operator,
            &simple_operand_fact(left, facts),
            &simple_operand_fact(right, facts),
        ),
        MirRvalue::ShortCircuit { .. } => {
            runmat_types::FactInference::exact(scalar_fact(ValueKindFact::Logical))
        }
        MirRvalue::Unary(operator, operand) => {
            infer_unary(*operator, &simple_operand_fact(operand, facts))
        }
        MirRvalue::Range { start, step, end } => {
            let step = match step {
                None => RangeStepFact::Implicit,
                Some(step) => numeric_operand(step)
                    .map(RangeStepFact::Known)
                    .unwrap_or(RangeStepFact::Unknown),
            };
            infer_range(numeric_operand(start), step, numeric_operand(end))
        }
        MirRvalue::Index { base, indexing }
            if matches!(
                indexing.result_context,
                runmat_types::IndexResultContext::ReadCommaList
                    | runmat_types::IndexResultContext::FunctionArgumentExpansion
            ) =>
        {
            legacy_sequence_as_value_inference(runmat_types::infer_index_sequence(
                &simple_operand_fact(base, facts),
                indexing.kind,
                &index_selectors(indexing, facts),
                indexing.result_context,
            ))
        }
        MirRvalue::Index { base, indexing } => infer_index(
            &simple_operand_fact(base, facts),
            indexing.kind,
            &index_selectors(indexing, facts),
            indexing.result_context,
        ),
        MirRvalue::Member {
            base,
            member,
            sequence_use,
        } if *sequence_use == runmat_types::SequenceUse::RequireSingle => {
            infer_member_read(&simple_operand_fact(base, facts), member, *sequence_use)
        }
        MirRvalue::Member { base, member, .. } => legacy_sequence_as_value_inference(
            runmat_types::infer_member_sequence(&simple_operand_fact(base, facts), member),
        ),
        MirRvalue::SubscriptChain(chain)
            if chain.sequence_use != runmat_types::SequenceUse::RequireSingle =>
        {
            legacy_sequence_as_value_inference(infer_subscript_chain_sequence(chain, facts))
        }
        MirRvalue::SubscriptChain(chain) => infer_subscript_chain(chain, facts),
        MirRvalue::DynamicMember { .. }
        | MirRvalue::WorkspaceFirstStaticProperty { .. }
        | MirRvalue::MetaClass(_)
        | MirRvalue::Colon
        | MirRvalue::End => runmat_types::FactInference::exact(dynamic_value()),
        MirRvalue::Call(call) if call.requested_outputs.known_count() == Some(0) => {
            runmat_types::FactInference::exact(scalar_fact(ValueKindFact::Void))
        }
        MirRvalue::Call(_) => runmat_types::FactInference::exact(dynamic_value()),
        MirRvalue::Distributed(_) | MirRvalue::Collective(_) => {
            runmat_types::FactInference::exact(dynamic_value())
        }
    }
}

/// Defensive fallback for old serialized MIR. Current lowering represents
/// every non-single sequence expression with `CaptureSequence`, so a value
/// rvalue cannot carry the sequence fact forward.
fn legacy_sequence_as_value_inference(
    inference: runmat_types::SequenceFactInference,
) -> runmat_types::FactInference {
    let fact = if !inference.sequence.variadic && inference.sequence.outputs.len() == 1 {
        inference
            .sequence
            .outputs
            .into_iter()
            .next()
            .expect("one sequence output was established")
    } else {
        ValueFact::unknown(DynamicReason::RuntimeValue)
    };
    runmat_types::FactInference {
        fact,
        diagnostics: inference.diagnostics,
    }
}

pub(crate) fn infer_subscript_chain(
    chain: &crate::MirSubscriptChain,
    facts: &[Option<ValueFact>],
) -> runmat_types::FactInference {
    subscript_chain::infer(chain, facts).inference
}

pub(crate) fn infer_subscript_chain_contract(
    chain: &crate::MirSubscriptChain,
    facts: &[Option<ValueFact>],
) -> SubscriptChainInference {
    subscript_chain::infer(chain, facts)
}

pub(crate) fn infer_subscript_chain_sequence(
    chain: &crate::MirSubscriptChain,
    facts: &[Option<ValueFact>],
) -> runmat_types::SequenceFactInference {
    subscript_chain::infer_sequence(chain, facts)
}

fn aggregate_inference(
    kind: &MirAggregateKind,
    row_lengths: &[usize],
    elements: &[crate::MirAggregateElement],
    facts: &[Option<ValueFact>],
) -> runmat_types::FactInference {
    let mut offset = 0usize;
    let mut rows = Vec::with_capacity(row_lengths.len());
    for &length in row_lengths {
        let Some(end) = offset.checked_add(length) else {
            return runmat_types::FactInference::exact(dynamic_value());
        };
        let Some(row) = elements.get(offset..end) else {
            return runmat_types::FactInference::exact(dynamic_value());
        };
        rows.push(
            row.iter()
                .map(|element| {
                    element
                        .operand()
                        .map(|operand| simple_operand_fact(operand, facts))
                        .unwrap_or_else(dynamic_value)
                })
                .collect::<Vec<_>>(),
        );
        offset = end;
    }
    if offset != elements.len() {
        return runmat_types::FactInference::exact(dynamic_value());
    }
    match kind {
        MirAggregateKind::Tensor => infer_tensor_aggregate(&rows),
        MirAggregateKind::Cell => infer_cell_aggregate(&rows),
    }
}

fn simple_operand_fact(operand: &MirOperand, facts: &[Option<ValueFact>]) -> ValueFact {
    match operand {
        MirOperand::Local(local) => facts
            .get(local.0)
            .and_then(Clone::clone)
            .unwrap_or_else(dynamic_value),
        MirOperand::Constant(constant) => constant_fact(constant),
        MirOperand::FunctionHandle(CallableIdentity::BoundFunction(function))
        | MirOperand::FunctionHandle(CallableIdentity::AnonymousFunction(function)) => {
            scalar_fact(ValueKindFact::Callable(CallableFact {
                identity: Some(CallableIdentity::BoundFunction(*function)),
                capabilities: Default::default(),
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
                capabilities: Default::default(),
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

pub(crate) fn constant_fact(constant: &crate::MirConstant) -> ValueFact {
    if let crate::MirConstant::Symbol(name) = constant {
        if let Some(constant) = runmat_builtins::builtin_constant_catalog_entry_by_name(&name.0) {
            return constant.fact();
        }
    }
    infer_literal(&literal_value(constant)).fact
}

pub(crate) fn literal_value(constant: &crate::MirConstant) -> LiteralValue {
    match constant {
        crate::MirConstant::Number(value) => LiteralValue::Real {
            text: value.clone(),
            class: runmat_types::NumericClass::Double,
        },
        crate::MirConstant::ImaginaryNumber(value) => LiteralValue::Complex {
            real: "0".into(),
            imaginary: value.clone(),
            class: runmat_types::NumericClass::Double,
        },
        crate::MirConstant::IntegerLiteral(value) => LiteralValue::Integer {
            text: value.text().to_owned(),
            class: integer_literal_numeric_class(value.class()),
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

fn integer_literal_numeric_class(class: runmat_types::IntegerLiteralClass) -> NumericClass {
    use runmat_types::IntegerLiteralClass;

    match class {
        IntegerLiteralClass::Int8 => NumericClass::Int8,
        IntegerLiteralClass::Int16 => NumericClass::Int16,
        IntegerLiteralClass::Int32 => NumericClass::Int32,
        IntegerLiteralClass::Int64 => NumericClass::Int64,
        IntegerLiteralClass::UInt8 => NumericClass::UInt8,
        IntegerLiteralClass::UInt16 => NumericClass::UInt16,
        IntegerLiteralClass::UInt32 => NumericClass::UInt32,
        IntegerLiteralClass::UInt64 => NumericClass::UInt64,
    }
}

fn numeric_operand(operand: &MirOperand) -> Option<f64> {
    match operand {
        MirOperand::Constant(crate::MirConstant::Number(value)) => value.parse().ok(),
        _ => None,
    }
}

pub(crate) fn index_selectors(
    indexing: &MirIndexing,
    facts: &[Option<ValueFact>],
) -> Vec<IndexSelectorFact> {
    indexing
        .components
        .iter()
        .map(|component| match component {
            MirIndexComponent::Colon => IndexSelectorFact::Colon,
            MirIndexComponent::Expr(operand) => index_selector_for_operand(operand, facts),
            MirIndexComponent::ContextualExpr(_)
            | MirIndexComponent::PreparedExpr(_)
            | MirIndexComponent::Expansion(_) => {
                IndexSelectorFact::Numeric(ValueFact::unknown(DynamicReason::Unspecified))
            }
        })
        .collect()
}

fn index_selector_for_operand(
    operand: &MirOperand,
    facts: &[Option<ValueFact>],
) -> IndexSelectorFact {
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

fn scalar_fact(kind: ValueKindFact) -> ValueFact {
    value_fact(kind, ShapeFact::Scalar, StorageFact::Scalar)
}

fn value_fact(kind: ValueKindFact, shape: ShapeFact, storage: StorageFact) -> ValueFact {
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

fn dynamic_value() -> ValueFact {
    ValueFact::unknown(DynamicReason::Unspecified)
}

pub(crate) fn diagnostics_for_assembly(assembly: &MirAssembly) -> Vec<MirDiagnostic> {
    let mut diagnostics = Vec::new();
    for (function, body) in &assembly.bodies {
        let source = assembly
            .functions
            .get(function)
            .map(|metadata| metadata.source);
        diagnostics.extend(
            diagnose_uninitialized_reads(
                body,
                super::constructor_entry::instance_output_local(assembly, body),
            )
            .into_iter()
            .map(|diagnostic| attach_source(diagnostic, source)),
        );
        diagnostics.extend(
            diagnose_semantic_misuse(body)
                .into_iter()
                .map(|diagnostic| attach_source(diagnostic, source)),
        );
    }
    let boundaries = analyze_assembly_spawn_boundaries(assembly);
    for (function, boundaries) in &boundaries {
        let source = assembly
            .functions
            .get(function)
            .map(|metadata| metadata.source);
        diagnostics.extend(
            diagnose_spawn_safety(boundaries)
                .into_iter()
                .map(|diagnostic| attach_source(diagnostic, source)),
        );
    }
    diagnostics
}

fn attach_source(
    diagnostic: MirDiagnostic,
    source: Option<runmat_types::ProgramSourceId>,
) -> MirDiagnostic {
    match source {
        Some(source) => diagnostic.with_source(source),
        None => diagnostic,
    }
}

pub(super) fn diagnose_uninitialized_reads(
    body: &MirBody,
    constructor_instance: Option<MirLocalId>,
) -> Vec<MirDiagnostic> {
    let result = compute_init_dataflow(body, constructor_instance);
    let current_bindings = current_scope_bindings(body);
    let mut diagnostics = Vec::new();

    for (idx, block) in body.blocks.iter().enumerate() {
        let Some(mut state) = result.in_states[idx].clone() else {
            continue;
        };
        diagnose_block(block, &mut state, &current_bindings, &mut diagnostics);
    }

    diagnostics
}

pub(super) fn diagnose_semantic_misuse(body: &MirBody) -> Vec<MirDiagnostic> {
    let mut diagnostics = Vec::new();
    for block in &body.blocks {
        for stmt in &block.statements {
            match &stmt.kind {
                MirStmtKind::PrepareFixedAssignment(owner) => {
                    for step in owner.steps() {
                        if let crate::MirExpressionStep::Let { value, span, .. } = step {
                            diagnose_rvalue_semantics(value, *span, &mut diagnostics);
                        }
                    }
                }
                MirStmtKind::PreparedAssign { value, .. }
                | MirStmtKind::Assign { value, .. }
                | MirStmtKind::MultiAssign { value, .. }
                | MirStmtKind::SequenceAssign { value, .. }
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
                MirStmtKind::CaptureSequence { source, .. } => {
                    source.visit_direct_expression_steps_dyn(&mut |steps| {
                        diagnose_region_semantics(steps, &mut diagnostics)
                    });
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
                MirStmtKind::StatementOutput(_) => {}
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
        }
        MirRvalue::Index { .. } => {}
        _ => {}
    }
    value.visit_direct_expression_steps_dyn(&mut |steps| {
        diagnose_region_semantics(steps, diagnostics)
    });
}

fn diagnose_region_semantics(
    steps: &[crate::MirExpressionStep],
    diagnostics: &mut Vec<MirDiagnostic>,
) {
    for step in steps {
        match step {
            crate::MirExpressionStep::Let { value, span, .. } => {
                diagnose_rvalue_semantics(value, *span, diagnostics)
            }
            crate::MirExpressionStep::CaptureSequence { source, .. } => {
                source.visit_direct_expression_steps_dyn(&mut |nested| {
                    diagnose_region_semantics(nested, diagnostics)
                });
            }
        }
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

fn diagnose_block(
    block: &BasicBlock,
    state: &mut [InitFact],
    current_bindings: &[MirLocalId],
    diagnostics: &mut Vec<MirDiagnostic>,
) {
    for stmt in &block.statements {
        diagnose_stmt(stmt, state, current_bindings, diagnostics);
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
            diagnose_rvalue_reads(
                iterable,
                state,
                current_bindings,
                block.terminator.span,
                diagnostics,
            );
            state[binding.0] = InitFact::DefinitelyAssigned;
        }
        MirTerminatorKind::ParFor {
            binding,
            iterable,
            maximum_workers,
            ..
        } => {
            diagnose_rvalue_reads(
                iterable,
                state,
                current_bindings,
                block.terminator.span,
                diagnostics,
            );
            if let Some(workers) = maximum_workers.as_deref() {
                diagnose_rvalue_reads(
                    workers,
                    state,
                    current_bindings,
                    block.terminator.span,
                    diagnostics,
                );
            }
            state[binding.0] = InitFact::DefinitelyAssigned;
        }
        MirTerminatorKind::Spmd { header, .. } => match header.as_ref() {
            crate::parallel::MirSpmdHeader::Default => {}
            crate::parallel::MirSpmdHeader::One(value) => {
                diagnose_rvalue_reads(
                    value,
                    state,
                    current_bindings,
                    block.terminator.span,
                    diagnostics,
                );
            }
            crate::parallel::MirSpmdHeader::Two(first, second) => {
                diagnose_rvalue_reads(
                    first,
                    state,
                    current_bindings,
                    block.terminator.span,
                    diagnostics,
                );
                diagnose_rvalue_reads(
                    second,
                    state,
                    current_bindings,
                    block.terminator.span,
                    diagnostics,
                );
            }
            crate::parallel::MirSpmdHeader::Three(first, second, third) => {
                diagnose_rvalue_reads(
                    first,
                    state,
                    current_bindings,
                    block.terminator.span,
                    diagnostics,
                );
                diagnose_rvalue_reads(
                    second,
                    state,
                    current_bindings,
                    block.terminator.span,
                    diagnostics,
                );
                diagnose_rvalue_reads(
                    third,
                    state,
                    current_bindings,
                    block.terminator.span,
                    diagnostics,
                );
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
                diagnose_place_reads(
                    place,
                    state,
                    current_bindings,
                    block.terminator.span,
                    diagnostics,
                );
                mark_place_assigned(place, state);
            }
        }
        MirTerminatorKind::Goto(_)
        | MirTerminatorKind::TryCatch { .. }
        | MirTerminatorKind::Unreachable => {}
    }
}

fn diagnose_stmt(
    stmt: &MirStmt,
    state: &mut [InitFact],
    current_bindings: &[MirLocalId],
    diagnostics: &mut Vec<MirDiagnostic>,
) {
    match &stmt.kind {
        MirStmtKind::PrepareFixedAssignment(owner) => {
            if let Some(region) = owner.selector_region() {
                let mut nested_state = state.to_vec();
                diagnose_expression_steps(
                    &region.steps,
                    &mut nested_state,
                    current_bindings,
                    diagnostics,
                );
                diagnose_call_argument_reads(
                    &region.argument,
                    &mut nested_state,
                    current_bindings,
                    region.span,
                    diagnostics,
                );
                transfer_region_workspace_lookup(state, &nested_state, current_bindings);
            } else if matches!(
                owner.phase,
                crate::MirFixedPreparationPhase::FinishStep { .. }
            ) {
                owner.visit_external_operands(|operand| {
                    diagnose_operand_read(operand, state, stmt.span, diagnostics);
                });
            }
            if let Some(crate::MirFixedSelectorDestination::Scalar(local)) = owner.output() {
                mark_place_assigned(&crate::MirPlace::Local(local), state);
            }
        }
        MirStmtKind::PreparedAssign { preparation, value } => {
            diagnose_rvalue_reads(value, state, current_bindings, stmt.span, diagnostics);
            mark_place_assigned(&preparation.mutation.place, state);
        }
        MirStmtKind::Assign { place, value } => {
            diagnose_rvalue_reads(value, state, current_bindings, stmt.span, diagnostics);
            diagnose_place_reads(place, state, current_bindings, stmt.span, diagnostics);
            mark_place_assigned(place, state);
        }
        MirStmtKind::MultiAssign { targets, value } => {
            diagnose_rvalue_reads(value, state, current_bindings, stmt.span, diagnostics);
            for target in &targets.targets {
                match target {
                    crate::MirOutputTarget::Place(place) => {
                        diagnose_place_reads(
                            place,
                            state,
                            current_bindings,
                            stmt.span,
                            diagnostics,
                        );
                        mark_place_assigned(place, state);
                    }
                    crate::MirOutputTarget::Sequence(target) => {
                        diagnose_sequence_target_reads(
                            target,
                            state,
                            current_bindings,
                            stmt.span,
                            diagnostics,
                        );
                        mark_place_assigned(target.base(), state);
                    }
                    crate::MirOutputTarget::Discard => {}
                }
            }
        }
        MirStmtKind::SequenceAssign { target, value } => {
            diagnose_rvalue_reads(value, state, current_bindings, stmt.span, diagnostics);
            let (base, member) = match target {
                crate::MirSequenceTarget::Member { base, .. } => (base, None),
                crate::MirSequenceTarget::DynamicMember { base, member } => (base, Some(member)),
                crate::MirSequenceTarget::CellContents { base, indexing } => {
                    diagnose_indexing_reads(
                        indexing,
                        state,
                        current_bindings,
                        stmt.span,
                        diagnostics,
                    );
                    (base, None)
                }
            };
            diagnose_place_reads(base, state, current_bindings, stmt.span, diagnostics);
            if let Some(member) = member {
                diagnose_operand_read(member, state, stmt.span, diagnostics);
            }
            mark_place_assigned(base, state);
        }
        MirStmtKind::CaptureSequence { source, .. } => {
            diagnose_expansion_source_reads(
                source,
                state,
                current_bindings,
                stmt.span,
                diagnostics,
            );
        }
        MirStmtKind::WorkspaceEffect { effect, bindings } => {
            mark_declared_storage(effect, bindings, state);
        }
        MirStmtKind::Expr(value) => {
            diagnose_rvalue_reads(value, state, current_bindings, stmt.span, diagnostics)
        }
        MirStmtKind::StatementOutput(output) => {
            if let crate::MirStatementOutputSource::Local(local) = output.source {
                diagnose_operand_read(
                    &crate::MirOperand::Local(local),
                    state,
                    stmt.span,
                    diagnostics,
                );
            }
        }
        MirStmtKind::PlaceMutation(_) | MirStmtKind::EnvironmentEffect(_) => {}
    }
}

fn mark_declared_storage(
    effect: &runmat_hir::WorkspaceEffect,
    bindings: &[MirLocalId],
    state: &mut [InitFact],
) {
    if matches!(
        effect,
        runmat_hir::WorkspaceEffect::MutatesGlobal | runmat_hir::WorkspaceEffect::MutatesPersistent
    ) {
        for binding in bindings {
            state[binding.0] = InitFact::DefinitelyAssigned;
        }
    }
}

fn diagnose_sequence_target_reads(
    target: &crate::MirSequenceTarget,
    state: &mut [InitFact],
    current_bindings: &[MirLocalId],
    span: Span,
    diagnostics: &mut Vec<MirDiagnostic>,
) {
    diagnose_place_reads(target.base(), state, current_bindings, span, diagnostics);
    match target {
        crate::MirSequenceTarget::Member { .. } => {}
        crate::MirSequenceTarget::DynamicMember { member, .. } => {
            diagnose_operand_read(member, state, span, diagnostics)
        }
        crate::MirSequenceTarget::CellContents { indexing, .. } => {
            diagnose_indexing_reads(indexing, state, current_bindings, span, diagnostics)
        }
    }
}

fn diagnose_rvalue_reads(
    value: &MirRvalue,
    state: &mut [InitFact],
    current_bindings: &[MirLocalId],
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
                diagnose_stmt(stmt, state, current_bindings, diagnostics);
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
                diagnose_call_argument_reads(arg, state, current_bindings, span, diagnostics);
            }
        }
        MirRvalue::Aggregate { elements, .. } => {
            for element in elements {
                if let Some(operand) = element.operand() {
                    diagnose_operand_read(operand, state, span, diagnostics);
                }
            }
        }
        MirRvalue::StructLiteral { fields } | MirRvalue::ObjectLiteral { fields, .. } => {
            for (_, value) in fields {
                diagnose_operand_read(value, state, span, diagnostics);
            }
        }
        MirRvalue::Index { base, indexing } => {
            diagnose_operand_read(base, state, span, diagnostics);
            diagnose_indexing_reads(indexing, state, current_bindings, span, diagnostics);
        }
        MirRvalue::SubscriptChain(chain) => {
            diagnose_subscript_chain_reads(chain, state, current_bindings, span, diagnostics)
        }
        MirRvalue::Member { base, .. } => diagnose_operand_read(base, state, span, diagnostics),
        MirRvalue::DynamicMember { base, member, .. } => {
            diagnose_operand_read(base, state, span, diagnostics);
            diagnose_operand_read(member, state, span, diagnostics);
        }
        MirRvalue::WorkspaceFirstStaticProperty { .. } => {}
        MirRvalue::Future { args, .. } | MirRvalue::ExternalFuture { args, .. } => {
            for arg in args {
                diagnose_call_argument_reads(arg, state, current_bindings, span, diagnostics);
            }
        }
        MirRvalue::MetaClass(_) | MirRvalue::Colon | MirRvalue::End => {}
        MirRvalue::Distributed(operation) => {
            for operand in operation.operands() {
                diagnose_operand_read(operand, state, span, diagnostics);
            }
        }
        MirRvalue::Collective(operation) => {
            for operand in operation.operands() {
                diagnose_operand_read(operand, state, span, diagnostics);
            }
        }
    }
    if exposes_current_workspace(value) {
        mark_runtime_workspace_lookup(state, current_bindings);
    }
}

fn mark_runtime_workspace_lookup(state: &mut [InitFact], current_bindings: &[MirLocalId]) {
    for binding in current_bindings {
        state[binding.0] = InitFact::RuntimeWorkspaceLookup;
    }
}

fn diagnose_call_argument_reads(
    argument: &crate::MirCallArg,
    state: &mut [InitFact],
    current_bindings: &[MirLocalId],
    span: Span,
    diagnostics: &mut Vec<MirDiagnostic>,
) {
    match argument {
        crate::MirCallArg::Single(operand) => {
            diagnose_operand_read(operand, state, span, diagnostics)
        }
        crate::MirCallArg::Expansion(source) => {
            diagnose_expansion_source_reads(source, state, current_bindings, span, diagnostics)
        }
        crate::MirCallArg::CapturedSequence(_) => {}
    }
}

fn diagnose_place_reads(
    place: &MirPlace,
    state: &mut [InitFact],
    current_bindings: &[MirLocalId],
    span: Span,
    diagnostics: &mut Vec<MirDiagnostic>,
) {
    match place {
        MirPlace::Local(_) | MirPlace::Binding(_) => {}
        MirPlace::Member(base, _) => {
            diagnose_place_reads(base, state, current_bindings, span, diagnostics)
        }
        MirPlace::DynamicMember(base, member) => {
            diagnose_place_reads(base, state, current_bindings, span, diagnostics);
            diagnose_operand_read(member, state, span, diagnostics);
        }
        MirPlace::Index(base, indexing) => {
            diagnose_place_reads(base, state, current_bindings, span, diagnostics);
            diagnose_indexing_reads(indexing, state, current_bindings, span, diagnostics);
        }
    }
}

fn diagnose_indexing_reads(
    indexing: &MirIndexing,
    state: &mut [InitFact],
    current_bindings: &[MirLocalId],
    span: Span,
    diagnostics: &mut Vec<MirDiagnostic>,
) {
    for component in &indexing.components {
        match component {
            MirIndexComponent::Expr(operand) => {
                diagnose_operand_read(operand, state, span, diagnostics);
            }
            MirIndexComponent::ContextualExpr(region) => {
                let mut nested_state = state.to_vec();
                diagnose_expression_steps(
                    region.steps(),
                    &mut nested_state,
                    current_bindings,
                    diagnostics,
                );
                diagnose_operand_read(region.result(), &nested_state, span, diagnostics);
                transfer_region_workspace_lookup(state, &nested_state, current_bindings);
            }
            MirIndexComponent::PreparedExpr(region) => {
                let mut nested_state = state.to_vec();
                diagnose_expression_steps(
                    region.steps(),
                    &mut nested_state,
                    current_bindings,
                    diagnostics,
                );
                diagnose_operand_read(region.result(), &nested_state, span, diagnostics);
                transfer_region_workspace_lookup(state, &nested_state, current_bindings);
            }
            MirIndexComponent::Expansion(region) => {
                let mut nested_state = state.to_vec();
                diagnose_expression_steps(
                    &region.steps,
                    &mut nested_state,
                    current_bindings,
                    diagnostics,
                );
                diagnose_call_argument_reads(
                    &region.argument,
                    &mut nested_state,
                    current_bindings,
                    region.span,
                    diagnostics,
                );
                transfer_region_workspace_lookup(state, &nested_state, current_bindings);
            }
            MirIndexComponent::Colon => {}
        }
    }
}

fn diagnose_expansion_source_reads(
    source: &crate::MirExpansionSource,
    state: &mut [InitFact],
    current_bindings: &[MirLocalId],
    span: Span,
    diagnostics: &mut Vec<MirDiagnostic>,
) {
    match source {
        crate::MirExpansionSource::SubscriptChain(chain) => {
            diagnose_subscript_chain_reads(chain, state, current_bindings, span, diagnostics)
        }
        crate::MirExpansionSource::CellContents { base, indexing } => {
            diagnose_operand_read(base, state, span, diagnostics);
            diagnose_indexing_reads(indexing, state, current_bindings, span, diagnostics);
        }
        crate::MirExpansionSource::ReturnedOutputs(base)
        | crate::MirExpansionSource::Member { base, .. } => {
            diagnose_operand_read(base, state, span, diagnostics);
        }
        crate::MirExpansionSource::DynamicMember { base, member } => {
            diagnose_operand_read(base, state, span, diagnostics);
            diagnose_operand_read(member, state, span, diagnostics);
        }
    }
}

fn transfer_region_workspace_lookup(
    state: &mut [InitFact],
    region_state: &[InitFact],
    current_bindings: &[MirLocalId],
) {
    for binding in current_bindings {
        if region_state[binding.0] == InitFact::RuntimeWorkspaceLookup {
            state[binding.0] = InitFact::RuntimeWorkspaceLookup;
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
    let Some(fact) = state.get(local.0) else {
        diagnostics.push(invalid_region_local_diagnostic(*local, span));
        return;
    };
    match fact {
        InitFact::DefinitelyAssigned | InitFact::RuntimeWorkspaceLookup => {}
        InitFact::Unassigned => {
            diagnostics.push(init_diagnostic(UninitializedRead::Definite, span))
        }
        InitFact::MaybeAssigned => {
            diagnostics.push(init_diagnostic(UninitializedRead::Possible, span))
        }
    }
}

fn invalid_region_local_diagnostic(local: MirLocalId, span: Span) -> MirDiagnostic {
    MirDiagnostic::new(
        "RM-MIR0021",
        MirDiagnosticSeverity::Error,
        format!("expression region references undeclared local {local:?}"),
        span,
    )
    .with_category("expression-region")
}

enum UninitializedRead {
    Definite,
    Possible,
}

fn init_diagnostic(kind: UninitializedRead, span: Span) -> MirDiagnostic {
    let (code, message, label, help) = match kind {
        UninitializedRead::Definite => (
            "RM-MIR0001",
            "local may be read before it is assigned",
            "this local is read before any assignment reaches this point",
            "assign this local before reading it",
        ),
        UninitializedRead::Possible => (
            "RM-MIR0002",
            "local may be read before assignment on some control-flow paths",
            "this local is not definitely assigned on every path reaching this point",
            "assign this local on every control-flow path before reading it",
        ),
    };
    MirDiagnostic::new(code, MirDiagnosticSeverity::Error, message, span)
        .with_runtime_check()
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

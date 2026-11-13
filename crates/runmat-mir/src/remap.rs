use std::collections::{BTreeMap, HashMap};

use runmat_types::{CallableIdentity, FunctionId};

use crate::{
    parallel::{MirCollectiveOp, MirDistributedOp, MirSpmdHeader},
    MirAssembly, MirCallArg, MirCallee, MirIndexComponent, MirIndexing, MirOperand, MirPlace,
    MirRvalue, MirStmt, MirStmtKind, MirTerminatorKind,
};

#[cfg(test)]
#[path = "remap/tests.rs"]
mod tests;

/// Remap unit-local HIR function ordinals into an immutable program/session
/// namespace after HIR lowering. MIR owns this traversal so new MIR constructs
/// cannot be silently omitted by Core, VM, or native codegen.
pub fn remap_function_ids(
    assembly: &mut MirAssembly,
    remap: &HashMap<FunctionId, FunctionId>,
) -> Result<(), String> {
    remap_with_policy(assembly, remap, RemapPolicy::EvalLocal)
}

/// Rebind a proved closed compilation product. Unlike eval-local lowering,
/// externally encoded references to its fresh IDs belong to this unit.
pub fn remap_closed_function_ids(
    assembly: &mut MirAssembly,
    remap: &HashMap<FunctionId, FunctionId>,
) -> Result<(), String> {
    remap_with_policy(assembly, remap, RemapPolicy::ClosedUnit)
}

#[derive(Clone, Copy)]
enum RemapPolicy {
    EvalLocal,
    ClosedUnit,
}

struct RemapContext<'a> {
    ids: &'a HashMap<FunctionId, FunctionId>,
    policy: RemapPolicy,
}

fn remap_with_policy(
    assembly: &mut MirAssembly,
    ids: &HashMap<FunctionId, FunctionId>,
    policy: RemapPolicy,
) -> Result<(), String> {
    let remap = &RemapContext { ids, policy };
    if remap.ids.is_empty() {
        return Ok(());
    }
    let destinations = remap
        .ids
        .values()
        .copied()
        .collect::<std::collections::BTreeSet<_>>();
    if destinations.len() != remap.ids.len()
        || assembly
            .functions
            .keys()
            .map(|id| mapped(*id, remap))
            .collect::<std::collections::BTreeSet<_>>()
            .len()
            != assembly.functions.len()
        || assembly
            .bodies
            .keys()
            .map(|id| mapped(*id, remap))
            .collect::<std::collections::BTreeSet<_>>()
            .len()
            != assembly.bodies.len()
    {
        return Err("MIR function rename collides with an existing local owner".into());
    }
    let mut bodies = BTreeMap::new();
    for (function, mut body) in std::mem::take(&mut assembly.bodies) {
        body.function = mapped(function, remap);
        for block in &mut body.blocks {
            for statement in &mut block.statements {
                remap_statement(statement, remap)?;
            }
            remap_terminator(&mut block.terminator.kind, remap)?;
        }
        bodies.insert(mapped(function, remap), body);
    }
    assembly.bodies = bodies;

    let mut functions = BTreeMap::new();
    for (function, mut metadata) in std::mem::take(&mut assembly.functions) {
        if let Some(parent) = &mut metadata.parent {
            *parent = mapped(*parent, remap);
        }
        for capture in &mut metadata.captures {
            capture.from_function = mapped(capture.from_function, remap);
        }
        functions.insert(mapped(function, remap), metadata);
    }
    assembly.functions = functions;
    for entrypoint in &mut assembly.entrypoints {
        *entrypoint = mapped(*entrypoint, remap);
    }
    for class in &mut assembly.classes {
        for method in &mut class.methods {
            method.function = mapped(method.function, remap);
        }
    }
    Ok(())
}

fn mapped(function: FunctionId, remap: &RemapContext<'_>) -> FunctionId {
    remap.ids.get(&function).copied().unwrap_or(function)
}

fn remap_identity(identity: &mut CallableIdentity, remap: &RemapContext<'_>) {
    match identity {
        CallableIdentity::BoundFunction(function)
        | CallableIdentity::AnonymousFunction(function) => *function = mapped(*function, remap),
        CallableIdentity::ExternalFunction { function, .. } => {
            if matches!(remap.policy, RemapPolicy::ClosedUnit) {
                *function = mapped(*function, remap);
            }
        }
        CallableIdentity::Builtin(_)
        | CallableIdentity::Imported(_)
        | CallableIdentity::Method(_)
        | CallableIdentity::DynamicName(_)
        | CallableIdentity::ExternalName(_) => {}
    }
}

fn remap_operand(operand: &mut MirOperand, remap: &RemapContext<'_>) {
    if let MirOperand::FunctionHandle(identity) = operand {
        remap_identity(identity, remap);
    }
}

fn remap_call_arg(argument: &mut MirCallArg, remap: &RemapContext<'_>) -> Result<(), String> {
    match argument {
        MirCallArg::Single(operand) => remap_operand(operand, remap),
        MirCallArg::Expansion(source) => remap_expansion_source(source, remap)?,
        MirCallArg::CapturedSequence(_) => {}
    }
    Ok(())
}

fn remap_expansion_source(
    source: &mut crate::MirExpansionSource,
    remap: &RemapContext<'_>,
) -> Result<(), String> {
    match source {
        crate::MirExpansionSource::SubscriptChain(chain) => {
            remap_chain(chain, remap)?;
        }
        crate::MirExpansionSource::CellContents { base, indexing } => {
            remap_operand(base, remap);
            remap_indexing(indexing, remap)?;
        }
        crate::MirExpansionSource::ReturnedOutputs(base)
        | crate::MirExpansionSource::Member { base, .. } => remap_operand(base, remap),
        crate::MirExpansionSource::DynamicMember { base, member } => {
            remap_operand(base, remap);
            remap_operand(member, remap);
        }
    }
    Ok(())
}

fn remap_indexing(indexing: &mut MirIndexing, remap: &RemapContext<'_>) -> Result<(), String> {
    for component in &mut indexing.components {
        match component {
            MirIndexComponent::Expr(operand) => remap_operand(operand, remap),
            MirIndexComponent::ContextualExpr(region) => {
                region.try_visit_rvalues_mut(|value| remap_rvalue(value, remap))?;
                region.try_visit_capture_sources_mut(|source| {
                    remap_expansion_source(source, remap)
                })?;
                remap_operand(region.result_mut(), remap);
            }
            MirIndexComponent::PreparedExpr(region) => {
                region.try_visit_rvalues_mut(|value| remap_rvalue(value, remap))?;
                region.try_visit_capture_sources_mut(|source| {
                    remap_expansion_source(source, remap)
                })?;
                remap_operand(region.result_mut(), remap);
            }
            MirIndexComponent::Expansion(region) => {
                for step in &mut region.steps {
                    match step {
                        crate::MirExpressionStep::Let { value, .. } => remap_rvalue(value, remap)?,
                        crate::MirExpressionStep::CaptureSequence { source, .. } => {
                            remap_expansion_source(source, remap)?
                        }
                    }
                }
                remap_call_arg(&mut region.argument, remap)?;
            }
            MirIndexComponent::Colon => {}
        }
    }
    Ok(())
}

fn remap_chain(
    chain: &mut crate::MirSubscriptChain,
    remap: &RemapContext<'_>,
) -> Result<(), String> {
    remap_operand(&mut chain.root, remap);
    for step in &mut chain.steps {
        match step {
            crate::MirSubscriptStep::Index(indexing) => remap_indexing(indexing, remap)?,
            crate::MirSubscriptStep::DottedInvoke {
                arguments,
                semantic_fallback,
                ..
            } => {
                if let Some(identity) = semantic_fallback {
                    // Compiler metadata mutation creates a new backing only
                    // when the function ID actually changes. Dispatch merely
                    // retains the already immutable source owner.
                    let changed = match identity.as_ref() {
                        CallableIdentity::BoundFunction(function)
                        | CallableIdentity::AnonymousFunction(function) => {
                            mapped(*function, remap) != *function
                        }
                        CallableIdentity::ExternalFunction { function, .. }
                            if matches!(remap.policy, RemapPolicy::ClosedUnit) =>
                        {
                            mapped(*function, remap) != *function
                        }
                        _ => false,
                    };
                    if changed {
                        remap_identity(std::sync::Arc::make_mut(identity), remap);
                    }
                }
                for argument in arguments {
                    match argument {
                        crate::MirDottedInvokeArgument::Colon => {}
                        crate::MirDottedInvokeArgument::Contextual(region) => {
                            remap_region(region, remap)?;
                        }
                        crate::MirDottedInvokeArgument::Call(region) => {
                            for step in &mut region.steps {
                                match step {
                                    crate::MirExpressionStep::Let { value, .. } => {
                                        remap_rvalue(value, remap)?;
                                    }
                                    crate::MirExpressionStep::CaptureSequence {
                                        source, ..
                                    } => {
                                        remap_expansion_source(source, remap)?;
                                    }
                                }
                            }
                            remap_call_arg(&mut region.argument, remap)?;
                        }
                    }
                }
            }
            crate::MirSubscriptStep::DynamicMember(member) => remap_operand(member, remap),
            crate::MirSubscriptStep::PreparedDynamicMember(region) => {
                region.try_visit_rvalues_mut(|value| remap_rvalue(value, remap))?;
                region.try_visit_capture_sources_mut(|source| {
                    remap_expansion_source(source, remap)
                })?;
                remap_operand(region.result_mut(), remap);
            }
            crate::MirSubscriptStep::Member(_) => {}
        }
    }
    Ok(())
}

fn remap_region(
    region: &mut crate::MirExpressionRegion,
    remap: &RemapContext<'_>,
) -> Result<(), String> {
    region.try_visit_rvalues_mut(|value| remap_rvalue(value, remap))?;
    region.try_visit_capture_sources_mut(|source| remap_expansion_source(source, remap))?;
    remap_operand(region.result_mut(), remap);
    Ok(())
}

fn remap_place(place: &mut MirPlace, remap: &RemapContext<'_>) -> Result<(), String> {
    match place {
        MirPlace::DynamicMember(base, member) => {
            remap_place(base, remap)?;
            remap_operand(member, remap);
        }
        MirPlace::Index(base, indexing) => {
            remap_place(base, remap)?;
            remap_indexing(indexing, remap)?;
        }
        MirPlace::Member(base, _) => remap_place(base, remap)?,
        MirPlace::Local(_) | MirPlace::Binding(_) => {}
    }
    Ok(())
}

fn remap_rvalue(value: &mut MirRvalue, remap: &RemapContext<'_>) -> Result<(), String> {
    match value {
        MirRvalue::Use(operand) | MirRvalue::Unary(_, operand) | MirRvalue::Spawn(operand) => {
            remap_operand(operand, remap)
        }
        MirRvalue::Binary(left, _, right) => {
            remap_operand(left, remap);
            remap_operand(right, remap);
        }
        MirRvalue::ShortCircuit {
            left,
            right_temps,
            right,
            ..
        } => {
            remap_operand(left, remap);
            for statement in right_temps {
                remap_statement(statement, remap)?;
            }
            remap_operand(right, remap);
        }
        MirRvalue::Range { start, step, end } => {
            remap_operand(start, remap);
            if let Some(step) = step {
                remap_operand(step, remap);
            }
            remap_operand(end, remap);
        }
        MirRvalue::Call(call) => {
            match &mut call.callee {
                MirCallee::Static(identity) => remap_identity(identity, remap),
                MirCallee::Dynamic(operand) => remap_operand(operand, remap),
                MirCallee::SuperConstructor { .. } | MirCallee::SuperMethod { .. } => {}
            }
            for argument in &mut call.args {
                remap_call_arg(argument, remap)?;
            }
        }
        MirRvalue::Aggregate { elements, .. } => {
            for element in elements {
                if let Some(operand) = element.operand_mut() {
                    remap_operand(operand, remap);
                }
            }
        }
        MirRvalue::StructLiteral { fields } | MirRvalue::ObjectLiteral { fields, .. } => {
            for (_, field) in fields {
                remap_operand(field, remap);
            }
        }
        MirRvalue::Index { base, indexing } => {
            remap_operand(base, remap);
            remap_indexing(indexing, remap)?;
        }
        MirRvalue::SubscriptChain(chain) => {
            remap_chain(chain, remap)?;
        }
        MirRvalue::Member { base, .. } => remap_operand(base, remap),
        MirRvalue::DynamicMember { base, member, .. } => {
            remap_operand(base, remap);
            remap_operand(member, remap);
        }
        MirRvalue::Future { function, args, .. } => {
            *function = mapped(*function, remap);
            for argument in args {
                remap_call_arg(argument, remap)?;
            }
        }
        MirRvalue::ExternalFuture { args, .. } => {
            for argument in args {
                remap_call_arg(argument, remap)?;
            }
        }
        MirRvalue::Distributed(operation) => remap_distributed(operation, remap)?,
        MirRvalue::Collective(operation) => remap_collective(operation, remap)?,
        MirRvalue::WorkspaceFirstStaticProperty { .. }
        | MirRvalue::MetaClass(_)
        | MirRvalue::Colon
        | MirRvalue::End => {}
    }
    Ok(())
}

fn remap_statement(statement: &mut MirStmt, remap: &RemapContext<'_>) -> Result<(), String> {
    match &mut statement.kind {
        MirStmtKind::PrepareFixedAssignment(owner) => {
            remap_prepared_assignment(&mut owner.assignment, remap)?;
            if let crate::MirFixedPreparationPhase::EvaluateSelector { region, .. } =
                &mut owner.phase
            {
                for step in &mut region.steps {
                    match step {
                        crate::MirExpressionStep::Let { value, .. } => remap_rvalue(value, remap)?,
                        crate::MirExpressionStep::CaptureSequence { source, .. } => {
                            remap_expansion_source(source, remap)?
                        }
                    }
                }
                remap_call_arg(&mut region.argument, remap)?;
            }
        }
        MirStmtKind::PreparedAssign { preparation, value } => {
            remap_prepared_assignment(preparation, remap)?;
            remap_rvalue(value, remap)?;
        }
        MirStmtKind::Assign { place, value } => {
            remap_place(place, remap)?;
            remap_rvalue(value, remap)?;
        }
        MirStmtKind::MultiAssign { targets, value } => {
            for target in &mut targets.targets {
                match target {
                    crate::MirOutputTarget::Place(place) => remap_place(place, remap)?,
                    crate::MirOutputTarget::Sequence(target) => {
                        remap_sequence_target(target, remap)?
                    }
                    crate::MirOutputTarget::Discard => {}
                }
            }
            remap_rvalue(value, remap)?;
        }
        MirStmtKind::SequenceAssign { target, value } => {
            match target {
                crate::MirSequenceTarget::Member { base, .. } => remap_place(base, remap)?,
                crate::MirSequenceTarget::DynamicMember { base, member } => {
                    remap_place(base, remap)?;
                    remap_operand(member, remap);
                }
                crate::MirSequenceTarget::CellContents { base, indexing } => {
                    remap_place(base, remap)?;
                    remap_indexing(indexing, remap)?;
                }
            }
            remap_rvalue(value, remap)?;
        }
        MirStmtKind::CaptureSequence { source, .. } => {
            remap_expansion_source(source, remap)?;
        }
        MirStmtKind::Expr(value) => remap_rvalue(value, remap)?,
        MirStmtKind::PlaceMutation(mutation) => remap_place(&mut mutation.place, remap)?,
        MirStmtKind::StatementOutput(_)
        | MirStmtKind::WorkspaceEffect { .. }
        | MirStmtKind::EnvironmentEffect(_) => {}
    }
    Ok(())
}

fn remap_prepared_assignment(
    owner: &mut crate::MirPreparedAssignment,
    remap: &RemapContext<'_>,
) -> Result<(), String> {
    remap_place(&mut owner.mutation.place, remap)?;
    for step in &mut owner.steps {
        match step {
            crate::MirExpressionStep::Let { value, .. } => remap_rvalue(value, remap)?,
            crate::MirExpressionStep::CaptureSequence { source, .. } => {
                remap_expansion_source(source, remap)?
            }
        }
    }
    Ok(())
}

fn remap_sequence_target(
    target: &mut crate::MirSequenceTarget,
    remap: &RemapContext<'_>,
) -> Result<(), String> {
    match target {
        crate::MirSequenceTarget::Member { base, .. } => remap_place(base, remap)?,
        crate::MirSequenceTarget::DynamicMember { base, member } => {
            remap_place(base, remap)?;
            remap_operand(member, remap);
        }
        crate::MirSequenceTarget::CellContents { base, indexing } => {
            remap_place(base, remap)?;
            remap_indexing(indexing, remap)?;
        }
    }
    Ok(())
}

fn remap_terminator(
    terminator: &mut MirTerminatorKind,
    remap: &RemapContext<'_>,
) -> Result<(), String> {
    match terminator {
        MirTerminatorKind::Branch { cond, .. } => remap_operand(cond, remap),
        MirTerminatorKind::Switch { discr, cases, .. } => {
            remap_operand(discr, remap);
            for (case, _) in cases {
                remap_operand(case, remap);
            }
        }
        MirTerminatorKind::For { iterable, .. } => remap_rvalue(iterable, remap)?,
        MirTerminatorKind::ParFor {
            iterable,
            maximum_workers,
            ..
        } => {
            remap_rvalue(iterable, remap)?;
            if let Some(maximum_workers) = maximum_workers {
                remap_rvalue(maximum_workers, remap)?;
            }
        }
        MirTerminatorKind::Spmd { header, .. } => match header.as_mut() {
            MirSpmdHeader::Default => {}
            MirSpmdHeader::One(first) => remap_rvalue(first, remap)?,
            MirSpmdHeader::Two(first, second) => {
                remap_rvalue(first, remap)?;
                remap_rvalue(second, remap)?;
            }
            MirSpmdHeader::Three(first, second, third) => {
                remap_rvalue(first, remap)?;
                remap_rvalue(second, remap)?;
                remap_rvalue(third, remap)?;
            }
        },
        MirTerminatorKind::Return(values) => {
            for value in values {
                remap_operand(value, remap);
            }
        }
        MirTerminatorKind::Await { future, result, .. } => {
            remap_operand(future, remap);
            if let Some(result) = result {
                remap_place(result, remap)?;
            }
        }
        MirTerminatorKind::Goto(_)
        | MirTerminatorKind::TryCatch { .. }
        | MirTerminatorKind::Unreachable => {}
    }
    Ok(())
}

fn remap_distributed(
    operation: &mut MirDistributedOp,
    remap: &RemapContext<'_>,
) -> Result<(), String> {
    match operation {
        MirDistributedOp::Create { input, .. }
        | MirDistributedOp::Codistributed { input, .. }
        | MirDistributedOp::Build {
            local_part: input, ..
        } => {
            remap_operand(input, remap);
            match operation {
                MirDistributedOp::Codistributed { overload, .. } => match overload {
                    crate::parallel::MirCodistributedOverload::ReplicatedInputDefault => {}
                    crate::parallel::MirCodistributedOverload::CodistributorOrDesignatedWorker {
                        operand,
                    } => remap_operand(operand, remap),
                    crate::parallel::MirCodistributedOverload::DesignatedWorkerWithCodistributor {
                        worker,
                        codistributor,
                    } => {
                        remap_operand(worker, remap);
                        remap_operand(codistributor, remap);
                    }
                },
                MirDistributedOp::Build {
                    codistributor,
                    validation,
                    ..
                } => {
                    if let Some(codistributor) = codistributor {
                        remap_operand(codistributor, remap);
                    }
                    if let crate::parallel::MirDistributedBuildValidation::RuntimeOption(option) =
                        validation
                    {
                        remap_operand(option, remap);
                    }
                }
                MirDistributedOp::Create { .. } => {}
                _ => unreachable!("construction variants matched above"),
            }
        }
        MirDistributedOp::LocalPart { value }
        | MirDistributedOp::Materialize { value }
        | MirDistributedOp::Codistributor { value } => {
            remap_operand(value, remap);
        }
        MirDistributedOp::GlobalIndices {
            value,
            dimension,
            lab,
            ..
        } => {
            remap_operand(value, remap);
            remap_operand(dimension, remap);
            if let Some(lab) = lab {
                remap_operand(lab, remap);
            }
        }
        MirDistributedOp::Redistribute {
            value,
            codistributor,
        } => {
            remap_operand(value, remap);
            remap_operand(codistributor, remap);
        }
    }
    Ok(())
}

fn remap_collective(
    operation: &mut MirCollectiveOp,
    remap: &RemapContext<'_>,
) -> Result<(), String> {
    operation.for_each_operand_mut(|operand| remap_operand(operand, remap));
    Ok(())
}

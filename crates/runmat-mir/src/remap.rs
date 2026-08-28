use std::collections::{BTreeMap, HashMap};

use runmat_hir::{CallableIdentity, FunctionId};

use crate::{
    parallel::{MirCollectiveOp, MirDistributedOp, MirSpmdHeader},
    MirAssembly, MirCallArg, MirCallee, MirIndexComponent, MirIndexing, MirOperand, MirPlace,
    MirRvalue, MirStmt, MirStmtKind, MirTerminatorKind,
};

/// Remap unit-local HIR function ordinals into an immutable program/session
/// namespace after HIR lowering. MIR owns this traversal so new MIR constructs
/// cannot be silently omitted by Core, VM, or native codegen.
pub fn remap_function_ids(
    assembly: &mut MirAssembly,
    remap: &HashMap<FunctionId, FunctionId>,
) -> Result<(), String> {
    if remap.is_empty() {
        return Ok(());
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

fn mapped(function: FunctionId, remap: &HashMap<FunctionId, FunctionId>) -> FunctionId {
    remap.get(&function).copied().unwrap_or(function)
}

fn remap_identity(identity: &mut CallableIdentity, remap: &HashMap<FunctionId, FunctionId>) {
    match identity {
        CallableIdentity::BoundFunction(function)
        | CallableIdentity::AnonymousFunction(function) => *function = mapped(*function, remap),
        CallableIdentity::ExternalFunction { .. }
        | CallableIdentity::Builtin(_)
        | CallableIdentity::Imported(_)
        | CallableIdentity::Method(_)
        | CallableIdentity::DynamicName(_)
        | CallableIdentity::ExternalName(_) => {}
    }
}

fn remap_operand(operand: &mut MirOperand, remap: &HashMap<FunctionId, FunctionId>) {
    if let MirOperand::FunctionHandle(identity) = operand {
        remap_identity(identity, remap);
    }
}

fn remap_call_arg(argument: &mut MirCallArg, remap: &HashMap<FunctionId, FunctionId>) {
    match argument {
        MirCallArg::Single(operand) => remap_operand(operand, remap),
        MirCallArg::Expansion { base, indices, .. } => {
            remap_operand(base, remap);
            for index in indices {
                remap_operand(index, remap);
            }
        }
    }
}

fn remap_indexing(indexing: &mut MirIndexing, remap: &HashMap<FunctionId, FunctionId>) {
    for component in &mut indexing.components {
        if let MirIndexComponent::Expr(operand) = component {
            remap_operand(operand, remap);
        }
    }
}

fn remap_place(place: &mut MirPlace, remap: &HashMap<FunctionId, FunctionId>) {
    match place {
        MirPlace::DynamicMember(base, member) => {
            remap_place(base, remap);
            remap_operand(member, remap);
        }
        MirPlace::Index(base, indexing) => {
            remap_place(base, remap);
            remap_indexing(indexing, remap);
        }
        MirPlace::Member(base, _) => remap_place(base, remap),
        MirPlace::Local(_) | MirPlace::Binding(_) => {}
    }
}

fn remap_rvalue(
    value: &mut MirRvalue,
    remap: &HashMap<FunctionId, FunctionId>,
) -> Result<(), String> {
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
                remap_call_arg(argument, remap);
            }
        }
        MirRvalue::Aggregate { elements, .. } => {
            for element in elements {
                remap_operand(element, remap);
            }
        }
        MirRvalue::StructLiteral { fields } | MirRvalue::ObjectLiteral { fields, .. } => {
            for (_, field) in fields {
                remap_operand(field, remap);
            }
        }
        MirRvalue::Index { base, indexing } => {
            remap_operand(base, remap);
            remap_indexing(indexing, remap);
        }
        MirRvalue::Member { base, .. } => remap_operand(base, remap),
        MirRvalue::DynamicMember { base, member } => {
            remap_operand(base, remap);
            remap_operand(member, remap);
        }
        MirRvalue::Future { function, args, .. } => {
            *function = mapped(*function, remap);
            for argument in args {
                remap_call_arg(argument, remap);
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

fn remap_statement(
    statement: &mut MirStmt,
    remap: &HashMap<FunctionId, FunctionId>,
) -> Result<(), String> {
    match &mut statement.kind {
        MirStmtKind::Assign { place, value } => {
            remap_place(place, remap);
            remap_rvalue(value, remap)?;
        }
        MirStmtKind::MultiAssign { targets, value } => {
            for target in &mut targets.targets {
                if let crate::MirOutputTarget::Place(place) = target {
                    remap_place(place, remap);
                }
            }
            remap_rvalue(value, remap)?;
        }
        MirStmtKind::Expr(value) => remap_rvalue(value, remap)?,
        MirStmtKind::PlaceMutation(mutation) => remap_place(&mut mutation.place, remap),
        MirStmtKind::WorkspaceEffect { .. } | MirStmtKind::EnvironmentEffect(_) => {}
    }
    Ok(())
}

fn remap_terminator(
    terminator: &mut MirTerminatorKind,
    remap: &HashMap<FunctionId, FunctionId>,
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
            region,
            iterable,
            maximum_workers,
            ..
        } => {
            remap_parallel_region(region, remap)?;
            remap_rvalue(iterable, remap)?;
            if let Some(maximum_workers) = maximum_workers {
                remap_rvalue(maximum_workers, remap)?;
            }
        }
        MirTerminatorKind::Spmd { region, header, .. } => {
            remap_parallel_region(region, remap)?;
            match header.as_mut() {
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
            }
        }
        MirTerminatorKind::Return(values) => {
            for value in values {
                remap_operand(value, remap);
            }
        }
        MirTerminatorKind::Await { future, result, .. } => {
            remap_operand(future, remap);
            if let Some(result) = result {
                remap_place(result, remap);
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
    remap: &HashMap<FunctionId, FunctionId>,
) -> Result<(), String> {
    match operation {
        MirDistributedOp::Create {
            id, owner, input, ..
        } => {
            remap_program_function(&mut id.function, remap)?;
            remap_parallel_region(owner, remap)?;
            remap_operand(input, remap);
        }
        MirDistributedOp::LocalPart { value }
        | MirDistributedOp::Materialize { value }
        | MirDistributedOp::Redistribute { value, .. } => {
            remap_program_function(&mut value.function, remap)?;
        }
    }
    Ok(())
}

fn remap_collective(
    operation: &mut MirCollectiveOp,
    remap: &HashMap<FunctionId, FunctionId>,
) -> Result<(), String> {
    let (id, input) = match operation {
        MirCollectiveOp::Barrier { id } | MirCollectiveOp::Receive { id, .. } => (id, None),
        MirCollectiveOp::Broadcast { id, input, .. }
        | MirCollectiveOp::Gather { id, input, .. }
        | MirCollectiveOp::Scatter { id, input, .. }
        | MirCollectiveOp::AllGather { id, input }
        | MirCollectiveOp::Reduce { id, input, .. }
        | MirCollectiveOp::AllReduce { id, input, .. }
        | MirCollectiveOp::Send { id, input, .. } => (id, Some(input)),
    };
    remap_parallel_region(&mut id.region, remap)?;
    if let Some(input) = input {
        remap_operand(input, remap);
    }
    Ok(())
}

fn remap_parallel_region(
    region: &mut runmat_types::ParallelRegionId,
    remap: &HashMap<FunctionId, FunctionId>,
) -> Result<(), String> {
    remap_program_function(&mut region.0.function, remap)
}

fn remap_program_function(
    function: &mut runmat_types::ProgramFunctionId,
    remap: &HashMap<FunctionId, FunctionId>,
) -> Result<(), String> {
    let old = FunctionId(function.0 as usize);
    let new = mapped(old, remap);
    function.0 = u32::try_from(new.0)
        .map_err(|_| "remapped function identity exceeds portable schema".to_string())?;
    Ok(())
}

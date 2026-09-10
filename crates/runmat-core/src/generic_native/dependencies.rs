use std::collections::BTreeSet;

use runmat_hir::{CallableIdentity, FunctionId};
use runmat_mir::{
    MirCallArg, MirCallee, MirIndexing, MirOperand, MirPlace, MirRvalue, MirStmt, MirStmtKind,
    MirSubscriptChain, MirTerminatorKind,
};

pub(super) struct ReferencedFunctions {
    pub functions: BTreeSet<FunctionId>,
    pub dynamic_catalog: bool,
}

pub(super) fn analyze(assembly: &runmat_mir::MirAssembly) -> ReferencedFunctions {
    let mut referenced = ReferencedFunctions {
        functions: BTreeSet::new(),
        dynamic_catalog: false,
    };
    for body in assembly.bodies.values() {
        for block in &body.blocks {
            for statement in &block.statements {
                visit_statement(statement, &mut referenced);
            }
            visit_terminator(&block.terminator.kind, &mut referenced);
        }
    }
    referenced
}

fn visit_identity(identity: &CallableIdentity, out: &mut ReferencedFunctions) {
    match identity {
        CallableIdentity::ExternalFunction { function, .. } => {
            out.functions.insert(*function);
        }
        CallableIdentity::BoundFunction(_) | CallableIdentity::AnonymousFunction(_) => {}
        CallableIdentity::ExternalName(_)
        | CallableIdentity::DynamicName(_)
        | CallableIdentity::Imported(_)
        | CallableIdentity::Method(_) => out.dynamic_catalog = true,
        CallableIdentity::Builtin(_) => {}
    }
}

fn visit_operand(operand: &MirOperand, out: &mut ReferencedFunctions) {
    if let MirOperand::FunctionHandle(identity) = operand {
        visit_identity(identity, out);
    }
}

fn visit_call_arg(argument: &MirCallArg, out: &mut ReferencedFunctions) {
    match argument {
        MirCallArg::Single(operand) => visit_operand(operand, out),
        MirCallArg::Expansion(source) => {
            source.visit_operands(|operand| visit_operand(operand, out));
        }
        MirCallArg::CapturedSequence(_) => {}
    }
}

fn visit_indexing(indexing: &MirIndexing, out: &mut ReferencedFunctions) {
    indexing.visit_operands(|operand| visit_operand(operand, out));
    indexing.visit_expression_regions(|region| {
        for step in region.steps() {
            match step {
                runmat_mir::MirExpressionStep::Let { value, .. } => visit_rvalue(value, out),
                runmat_mir::MirExpressionStep::CaptureSequence { source, .. } => {
                    source.visit_operands(|operand| visit_operand(operand, out));
                }
            }
        }
    });
}

fn visit_subscript_chain(chain: &MirSubscriptChain, out: &mut ReferencedFunctions) {
    chain.visit_operands(|operand| visit_operand(operand, out));
    chain.visit_expression_regions(|region| {
        for step in region.steps() {
            match step {
                runmat_mir::MirExpressionStep::Let { value, .. } => visit_rvalue(value, out),
                runmat_mir::MirExpressionStep::CaptureSequence { source, .. } => {
                    source.visit_operands(|operand| visit_operand(operand, out));
                }
            }
        }
    });
}

fn visit_place(place: &MirPlace, out: &mut ReferencedFunctions) {
    match place {
        MirPlace::DynamicMember(base, member) => {
            visit_place(base, out);
            visit_operand(member, out);
        }
        MirPlace::Index(base, indexing) => {
            visit_place(base, out);
            visit_indexing(indexing, out);
        }
        MirPlace::Member(base, _) => visit_place(base, out),
        MirPlace::Local(_) | MirPlace::Binding(_) => {}
    }
}

fn visit_rvalue(value: &MirRvalue, out: &mut ReferencedFunctions) {
    match value {
        MirRvalue::Use(operand) | MirRvalue::Unary(_, operand) | MirRvalue::Spawn(operand) => {
            visit_operand(operand, out)
        }
        MirRvalue::Binary(left, _, right) => {
            visit_operand(left, out);
            visit_operand(right, out);
        }
        MirRvalue::ShortCircuit {
            left,
            right_temps,
            right,
            ..
        } => {
            visit_operand(left, out);
            for statement in right_temps {
                visit_statement(statement, out);
            }
            visit_operand(right, out);
        }
        MirRvalue::Range { start, step, end } => {
            visit_operand(start, out);
            if let Some(step) = step {
                visit_operand(step, out);
            }
            visit_operand(end, out);
        }
        MirRvalue::Call(call) => {
            match &call.callee {
                MirCallee::Static(identity) => visit_identity(identity, out),
                MirCallee::Dynamic(operand) => {
                    out.dynamic_catalog = true;
                    visit_operand(operand, out);
                }
                MirCallee::SuperConstructor { .. } | MirCallee::SuperMethod { .. } => {}
            }
            for argument in &call.args {
                visit_call_arg(argument, out);
            }
        }
        MirRvalue::Aggregate { elements, .. } => {
            for element in elements {
                if let Some(operand) = element.operand() {
                    visit_operand(operand, out);
                }
            }
        }
        MirRvalue::StructLiteral { fields } | MirRvalue::ObjectLiteral { fields, .. } => {
            for (_, value) in fields {
                visit_operand(value, out);
            }
        }
        MirRvalue::Index { base, indexing } => {
            visit_operand(base, out);
            visit_indexing(indexing, out);
        }
        MirRvalue::SubscriptChain(chain) => visit_subscript_chain(chain, out),
        MirRvalue::Member { base, .. } => visit_operand(base, out),
        MirRvalue::DynamicMember { base, member, .. } => {
            visit_operand(base, out);
            visit_operand(member, out);
        }
        MirRvalue::Future { args, .. } => {
            for argument in args {
                visit_call_arg(argument, out);
            }
        }
        MirRvalue::WorkspaceFirstStaticProperty { .. }
        | MirRvalue::MetaClass(_)
        | MirRvalue::Colon
        | MirRvalue::End
        | MirRvalue::Distributed(_)
        | MirRvalue::Collective(_) => {}
    }
}

fn visit_statement(statement: &MirStmt, out: &mut ReferencedFunctions) {
    match &statement.kind {
        MirStmtKind::Assign { place, value } => {
            visit_place(place, out);
            visit_rvalue(value, out);
        }
        MirStmtKind::MultiAssign { targets, value } => {
            for target in &targets.targets {
                match target {
                    runmat_mir::MirOutputTarget::Place(place) => visit_place(place, out),
                    runmat_mir::MirOutputTarget::Sequence(target) => {
                        visit_place(target.base(), out);
                        target.visit_operands(|operand| visit_operand(operand, out));
                    }
                    runmat_mir::MirOutputTarget::Discard => {}
                }
            }
            visit_rvalue(value, out);
        }
        MirStmtKind::SequenceAssign { target, value } => {
            visit_place(target.base(), out);
            target.visit_operands(|operand| visit_operand(operand, out));
            visit_rvalue(value, out);
        }
        MirStmtKind::CaptureSequence { source, .. } => {
            source.visit_operands(|operand| visit_operand(operand, out));
        }
        MirStmtKind::Expr(value) => visit_rvalue(value, out),
        MirStmtKind::PlaceMutation(mutation) => visit_place(&mutation.place, out),
        MirStmtKind::WorkspaceEffect { .. } | MirStmtKind::EnvironmentEffect(_) => {}
    }
}

fn visit_terminator(terminator: &MirTerminatorKind, out: &mut ReferencedFunctions) {
    match terminator {
        MirTerminatorKind::Branch { cond, .. } => visit_operand(cond, out),
        MirTerminatorKind::Switch { discr, cases, .. } => {
            visit_operand(discr, out);
            for (case, _) in cases {
                visit_operand(case, out);
            }
        }
        MirTerminatorKind::For { iterable, .. } => visit_rvalue(iterable, out),
        MirTerminatorKind::ParFor {
            iterable,
            maximum_workers,
            ..
        } => {
            visit_rvalue(iterable, out);
            if let Some(maximum_workers) = maximum_workers {
                visit_rvalue(maximum_workers, out);
            }
        }
        MirTerminatorKind::Spmd { header, .. } => match header.as_ref() {
            runmat_mir::parallel::MirSpmdHeader::Default => {}
            runmat_mir::parallel::MirSpmdHeader::One(first) => visit_rvalue(first, out),
            runmat_mir::parallel::MirSpmdHeader::Two(first, second) => {
                visit_rvalue(first, out);
                visit_rvalue(second, out);
            }
            runmat_mir::parallel::MirSpmdHeader::Three(first, second, third) => {
                visit_rvalue(first, out);
                visit_rvalue(second, out);
                visit_rvalue(third, out);
            }
        },
        MirTerminatorKind::Return(values) => {
            for value in values {
                visit_operand(value, out);
            }
        }
        MirTerminatorKind::Await { future, result, .. } => {
            visit_operand(future, out);
            if let Some(result) = result {
                visit_place(result, out);
            }
        }
        MirTerminatorKind::Goto(_)
        | MirTerminatorKind::TryCatch { .. }
        | MirTerminatorKind::Unreachable => {}
    }
}

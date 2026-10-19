use runmat_hir::{
    FunctionId, HirBlock, HirCallableRef, HirDottedInvokeArgument, HirExpr, HirExprKind, HirPlace,
    HirSequenceTarget, HirStmtKind, HirSubscriptStep, IndexComponent, IndexingSemantics,
    OutputTarget,
};

use super::HirUnresolvedSite;

pub(super) fn visit_block(
    block: &HirBlock,
    caller: FunctionId,
    sites: &mut Vec<HirUnresolvedSite>,
) {
    for statement in &block.statements {
        match &statement.kind {
            HirStmtKind::ExprStmt(expr, _) | HirStmtKind::Assign(_, expr, _) => {
                visit_expr(expr, caller, sites);
                if let HirStmtKind::Assign(place, _, _) = &statement.kind {
                    visit_place(place, caller, sites);
                }
            }
            HirStmtKind::MultiAssign(outputs, expr, _) => {
                for target in &outputs.targets {
                    match target {
                        OutputTarget::Place(place) => visit_place(place, caller, sites),
                        OutputTarget::Sequence(sequence) => {
                            visit_sequence_target(sequence, caller, sites);
                        }
                        OutputTarget::Discard => {}
                    }
                }
                visit_expr(expr, caller, sites);
            }
            HirStmtKind::SequenceAssign { target, value, .. } => {
                visit_sequence_target(target, caller, sites);
                visit_expr(value, caller, sites);
            }
            HirStmtKind::If {
                cond,
                then_body,
                elseif_blocks,
                else_body,
            } => {
                visit_expr(cond, caller, sites);
                visit_block(then_body, caller, sites);
                for (condition, body) in elseif_blocks {
                    visit_expr(condition, caller, sites);
                    visit_block(body, caller, sites);
                }
                if let Some(body) = else_body {
                    visit_block(body, caller, sites);
                }
            }
            HirStmtKind::While { cond, body } => {
                visit_expr(cond, caller, sites);
                visit_block(body, caller, sites);
            }
            HirStmtKind::For { range, body, .. } => {
                visit_expr(range, caller, sites);
                visit_block(body, caller, sites);
            }
            HirStmtKind::ParFor {
                range,
                maximum_workers,
                body,
                ..
            } => {
                visit_expr(range, caller, sites);
                if let Some(maximum_workers) = maximum_workers {
                    visit_expr(maximum_workers, caller, sites);
                }
                visit_block(body, caller, sites);
            }
            HirStmtKind::Spmd { header, body, .. } => {
                match header {
                    runmat_hir::parallel::SpmdHeader::Default => {}
                    runmat_hir::parallel::SpmdHeader::One(first) => {
                        visit_expr(first, caller, sites);
                    }
                    runmat_hir::parallel::SpmdHeader::Two(first, second) => {
                        visit_expr(first, caller, sites);
                        visit_expr(second, caller, sites);
                    }
                    runmat_hir::parallel::SpmdHeader::Three(first, second, third) => {
                        visit_expr(first, caller, sites);
                        visit_expr(second, caller, sites);
                        visit_expr(third, caller, sites);
                    }
                }
                visit_block(body, caller, sites);
            }
            HirStmtKind::Switch {
                expr,
                cases,
                otherwise,
            } => {
                visit_expr(expr, caller, sites);
                for (condition, body) in cases {
                    visit_expr(condition, caller, sites);
                    visit_block(body, caller, sites);
                }
                if let Some(body) = otherwise {
                    visit_block(body, caller, sites);
                }
            }
            HirStmtKind::TryCatch {
                try_body,
                catch_body,
                ..
            } => {
                visit_block(try_body, caller, sites);
                visit_block(catch_body, caller, sites);
            }
            HirStmtKind::Global(_)
            | HirStmtKind::Persistent(_)
            | HirStmtKind::Break
            | HirStmtKind::Continue
            | HirStmtKind::Return
            | HirStmtKind::Import(_) => {}
        }
    }
}

fn visit_expr(expr: &HirExpr, caller: FunctionId, sites: &mut Vec<HirUnresolvedSite>) {
    match &expr.kind {
        HirExprKind::Call(call) => {
            if let HirCallableRef::Unresolved(name) = &call.callee {
                if let [single] = name.0.as_slice() {
                    sites.push(HirUnresolvedSite {
                        caller,
                        expression: expr.id,
                        name: single.0.clone(),
                    });
                }
            }
            if let HirCallableRef::DynamicExpr(callee) = &call.callee {
                visit_expr(callee, caller, sites);
            }
            for argument in &call.args {
                visit_expr(argument, caller, sites);
            }
        }
        HirExprKind::CommandCall(command) => {
            if let HirCallableRef::Unresolved(name) = &command.command {
                if let [single] = name.0.as_slice() {
                    sites.push(HirUnresolvedSite {
                        caller,
                        expression: expr.id,
                        name: single.0.clone(),
                    });
                }
            }
        }
        HirExprKind::Unary(_, value) | HirExprKind::Await(value) | HirExprKind::Spawn(value) => {
            visit_expr(value, caller, sites);
        }
        HirExprKind::Binary(left, _, right) | HirExprKind::Range(left, Some(right), _) => {
            visit_expr(left, caller, sites);
            visit_expr(right, caller, sites);
            if let HirExprKind::Range(_, _, final_value) = &expr.kind {
                visit_expr(final_value, caller, sites);
            }
        }
        HirExprKind::Range(first, None, final_value) => {
            visit_expr(first, caller, sites);
            visit_expr(final_value, caller, sites);
        }
        HirExprKind::Tensor(rows) | HirExprKind::Cell(rows) => {
            for row in rows {
                for value in row {
                    visit_expr(value, caller, sites);
                }
            }
        }
        HirExprKind::StructLiteral(fields) | HirExprKind::ObjectLiteral { fields, .. } => {
            for (_, value) in fields {
                visit_expr(value, caller, sites);
            }
        }
        HirExprKind::Index(base, indexing) => {
            visit_expr(base, caller, sites);
            visit_indexing(indexing, caller, sites);
        }
        HirExprKind::SubscriptChain(chain) => {
            visit_expr(&chain.root, caller, sites);
            for step in &chain.steps {
                match step {
                    HirSubscriptStep::Index(indexing) => visit_indexing(indexing, caller, sites),
                    HirSubscriptStep::DynamicMember(member) => visit_expr(member, caller, sites),
                    HirSubscriptStep::DottedInvoke { arguments, .. } => {
                        for argument in arguments {
                            if let HirDottedInvokeArgument::Value(value) = argument {
                                visit_expr(value, caller, sites);
                            }
                        }
                    }
                    HirSubscriptStep::Member(_) => {}
                }
            }
        }
        HirExprKind::Member { base, .. } => visit_expr(base, caller, sites),
        HirExprKind::MemberDynamic { base, member, .. } => {
            visit_expr(base, caller, sites);
            visit_expr(member, caller, sites);
        }
        HirExprKind::Number(_)
        | HirExprKind::ImaginaryNumber(_)
        | HirExprKind::IntegerLiteral(_)
        | HirExprKind::String(_)
        | HirExprKind::Constant(_)
        | HirExprKind::Binding(_)
        | HirExprKind::Colon
        | HirExprKind::End
        | HirExprKind::WorkspaceFirstStaticProperty { .. }
        | HirExprKind::FunctionHandle(_)
        | HirExprKind::AnonymousFunction(_)
        | HirExprKind::MetaClass(_) => {}
    }
}

fn visit_indexing(
    indexing: &IndexingSemantics,
    caller: FunctionId,
    sites: &mut Vec<HirUnresolvedSite>,
) {
    for component in &indexing.components {
        match component {
            IndexComponent::Expr(expr) | IndexComponent::Logical(expr) => {
                visit_expr(expr, caller, sites);
            }
            IndexComponent::Colon | IndexComponent::End { .. } => {}
        }
    }
}

fn visit_place(place: &HirPlace, caller: FunctionId, sites: &mut Vec<HirUnresolvedSite>) {
    match place {
        HirPlace::Binding(_) => {}
        HirPlace::Member(base, _) => visit_expr(base, caller, sites),
        HirPlace::MemberDynamic(base, member) => {
            visit_expr(base, caller, sites);
            visit_expr(member, caller, sites);
        }
        HirPlace::Index(base, indexing) | HirPlace::IndexCell(base, indexing) => {
            visit_expr(base, caller, sites);
            visit_indexing(indexing, caller, sites);
        }
    }
}

fn visit_sequence_target(
    target: &HirSequenceTarget,
    caller: FunctionId,
    sites: &mut Vec<HirUnresolvedSite>,
) {
    match target {
        HirSequenceTarget::Member { base, .. } => visit_expr(base, caller, sites),
        HirSequenceTarget::DynamicMember { base, member } => {
            visit_expr(base, caller, sites);
            visit_expr(member, caller, sites);
        }
        HirSequenceTarget::CellContents { base, indexing } => {
            visit_expr(base, caller, sites);
            visit_indexing(indexing, caller, sites);
        }
    }
}

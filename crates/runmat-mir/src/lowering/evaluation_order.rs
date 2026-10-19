use runmat_hir::{
    HirCallableRef, HirExpr, HirExprKind, HirPlace, HirSequenceTarget, HirStmt, HirStmtKind,
    IndexComponent, OutputTarget,
};

pub(super) fn expression_children(expression: &HirExpr) -> Vec<&HirExpr> {
    match &expression.kind {
        HirExprKind::Unary(_, inner) | HirExprKind::Await(inner) | HirExprKind::Spawn(inner) => {
            vec![inner]
        }
        HirExprKind::Binary(left, _, right) => vec![left, right],
        HirExprKind::Tensor(rows) | HirExprKind::Cell(rows) => rows.iter().flatten().collect(),
        HirExprKind::StructLiteral(fields) | HirExprKind::ObjectLiteral { fields, .. } => {
            fields.iter().map(|(_, value)| value).collect()
        }
        HirExprKind::Range(start, step, end) => {
            let mut result = vec![start.as_ref()];
            result.extend(step.iter().map(AsRef::as_ref));
            result.push(end);
            result
        }
        HirExprKind::Index(base, indexing) => std::iter::once(base.as_ref())
            .chain(indexing.components.iter().filter_map(component_expression))
            .collect(),
        HirExprKind::Member { base, .. } => vec![base],
        HirExprKind::MemberDynamic { base, member, .. } => vec![base, member],
        HirExprKind::Call(call) => {
            let mut result = Vec::with_capacity(call.args.len() + 1);
            if let HirCallableRef::DynamicExpr(callee) = &call.callee {
                result.push(callee.as_ref());
            }
            result.extend(call.args.iter());
            result
        }
        _ => Vec::new(),
    }
}

pub(super) fn statement_expressions(statement: &HirStmt) -> Vec<&HirExpr> {
    match &statement.kind {
        HirStmtKind::Assign(place, value, _) => place_expressions(place)
            .into_iter()
            .chain(std::iter::once(value))
            .collect(),
        HirStmtKind::MultiAssign(targets, value, _) => targets
            .targets
            .iter()
            .flat_map(output_target_expressions)
            .chain(std::iter::once(value))
            .collect(),
        HirStmtKind::SequenceAssign { target, value, .. } => sequence_expressions(target)
            .into_iter()
            .chain(std::iter::once(value))
            .collect(),
        HirStmtKind::ExprStmt(value, _)
        | HirStmtKind::If { cond: value, .. }
        | HirStmtKind::While { cond: value, .. }
        | HirStmtKind::For { range: value, .. }
        | HirStmtKind::Switch { expr: value, .. } => vec![value],
        HirStmtKind::ParFor {
            range,
            maximum_workers,
            ..
        } => {
            let mut result = vec![range];
            result.extend(maximum_workers.iter());
            result
        }
        HirStmtKind::Spmd { header, .. } => match header {
            runmat_hir::parallel::SpmdHeader::Default => Vec::new(),
            runmat_hir::parallel::SpmdHeader::One(first) => vec![first],
            runmat_hir::parallel::SpmdHeader::Two(first, second) => vec![first, second],
            runmat_hir::parallel::SpmdHeader::Three(first, second, third) => {
                vec![first, second, third]
            }
        },
        _ => Vec::new(),
    }
}

fn output_target_expressions(target: &OutputTarget) -> Vec<&HirExpr> {
    match target {
        OutputTarget::Place(place) => place_expressions(place),
        OutputTarget::Sequence(target) => sequence_expressions(target),
        OutputTarget::Discard => Vec::new(),
    }
}

fn sequence_expressions(target: &HirSequenceTarget) -> Vec<&HirExpr> {
    match target {
        HirSequenceTarget::Member { base, .. } => vec![base],
        HirSequenceTarget::DynamicMember { base, member } => vec![base, member],
        HirSequenceTarget::CellContents { base, indexing } => std::iter::once(base.as_ref())
            .chain(indexing.components.iter().filter_map(component_expression))
            .collect(),
    }
}

fn place_expressions(place: &HirPlace) -> Vec<&HirExpr> {
    match place {
        HirPlace::Binding(_) => Vec::new(),
        HirPlace::Member(base, _) => vec![base],
        HirPlace::MemberDynamic(base, member) => vec![base, member],
        HirPlace::Index(base, indexing) | HirPlace::IndexCell(base, indexing) => {
            std::iter::once(base.as_ref())
                .chain(indexing.components.iter().filter_map(component_expression))
                .collect()
        }
    }
}

fn component_expression(component: &IndexComponent) -> Option<&HirExpr> {
    match component {
        IndexComponent::Expr(value) | IndexComponent::Logical(value) => Some(value),
        IndexComponent::Colon | IndexComponent::End { .. } => None,
    }
}

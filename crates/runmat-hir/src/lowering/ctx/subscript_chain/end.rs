use super::AstExpr;

pub(super) fn expr_references_end(expr: &AstExpr) -> bool {
    match expr {
        AstExpr::EndKeyword(_) => true,
        AstExpr::Range(start, step, end, _) => {
            expr_references_end(start)
                || step.as_deref().is_some_and(expr_references_end)
                || expr_references_end(end)
        }
        AstExpr::Binary(left, _, right, _) => {
            expr_references_end(left) || expr_references_end(right)
        }
        AstExpr::Unary(_, inner, _) => expr_references_end(inner),
        AstExpr::Tensor(rows, _) | AstExpr::Cell(rows, _) => rows
            .iter()
            .flat_map(|row| row.iter())
            .any(expr_references_end),
        AstExpr::Index(base, indices, _) | AstExpr::IndexCell(base, indices, _) => {
            expr_references_end(base) || indices.iter().any(expr_references_end)
        }
        AstExpr::FuncCall(_, args, _)
        | AstExpr::MethodCall(_, _, args, _)
        | AstExpr::DottedInvoke(_, _, args, _) => args.iter().any(expr_references_end),
        AstExpr::Member(base, _, _) => expr_references_end(base),
        AstExpr::MemberDynamic(base, name, _) => {
            expr_references_end(base) || expr_references_end(name)
        }
        AstExpr::AnonFunc { body, .. } => expr_references_end(body),
        _ => false,
    }
}

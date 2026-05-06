use crate::{
    BasicBlock, BasicBlockId, MirAssembly, MirBody, MirOperand, MirSourceMap, MirSourceRecord,
    MirTerminator, MirTerminatorKind,
};
use runmat_hir::{HirAssembly, HirFunction, SemanticError};

use super::{expr::lower_operand, stmt::lower_stmt, MirLoweringContext};

pub fn lower_assembly(hir: &HirAssembly) -> Result<MirAssembly, SemanticError> {
    let mut assembly = MirAssembly::default();
    for function in &hir.functions {
        assembly
            .bodies
            .insert(function.id, lower_function(function)?);
    }
    Ok(assembly)
}

pub fn lower_function(function: &HirFunction) -> Result<MirBody, SemanticError> {
    let mut ctx = MirLoweringContext::new();
    let (locals, local_sources) = ctx.locals_for_function(function);
    let entry = BasicBlockId(0);
    let mut statements = Vec::new();
    let mut source_records = Vec::new();
    for stmt in &function.body.statements {
        source_records.push(MirSourceRecord {
            block: entry,
            stmt: Some(stmt.id),
            expr: None,
            span: stmt.span,
        });
        statements.extend(lower_stmt(&ctx, stmt)?);
    }
    let returns: Vec<MirOperand> = function
        .outputs
        .iter()
        .map(|binding| {
            let expr = runmat_hir::HirExpr {
                id: runmat_hir::ExprId(usize::MAX),
                kind: runmat_hir::HirExprKind::Binding(*binding),
                span: function.span,
            };
            lower_operand(&ctx, &expr)
        })
        .collect::<Result<_, _>>()?;
    Ok(MirBody {
        function: function.id,
        locals,
        blocks: vec![BasicBlock {
            id: entry,
            statements,
            terminator: MirTerminator {
                kind: MirTerminatorKind::Return(returns),
                span: function.span,
            },
        }],
        source_map: MirSourceMap {
            function: Some(function.id),
            statements: source_records,
            locals: local_sources,
        },
    })
}

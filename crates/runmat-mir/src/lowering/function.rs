use crate::{
    BasicBlock, BasicBlockId, MirAssembly, MirBody, MirSourceMap, MirTerminator, MirTerminatorKind,
};
use runmat_hir::{HirAssembly, HirFunction, SemanticError};

use super::MirLoweringContext;

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
    Ok(MirBody {
        function: function.id,
        locals,
        blocks: vec![BasicBlock {
            id: entry,
            statements: Vec::new(),
            terminator: MirTerminator {
                kind: MirTerminatorKind::Return(Vec::new()),
                span: function.span,
            },
        }],
        source_map: MirSourceMap {
            function: Some(function.id),
            statements: Vec::new(),
            locals: local_sources,
        },
    })
}

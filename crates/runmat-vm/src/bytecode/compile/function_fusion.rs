//! Function-local fusion metadata uses the same MIR gate and instruction
//! window derivation as entrypoint bytecode, but is never borrowed from an
//! unrelated entrypoint or assembled from test-only graph guesses.

use super::{derive_semantic_fusion_instruction_windows, derive_semantic_fusion_metadata};
use crate::bytecode::{program::FunctionFusionScaffold, Instr};

pub(super) fn derive_function_scaffold(
    mir: &runmat_mir::MirAssembly,
    function: runmat_hir::FunctionId,
    instructions: &[Instr],
    instr_spans: &[runmat_hir::Span],
) -> Option<FunctionFusionScaffold> {
    let metadata = derive_semantic_fusion_metadata(mir, Some(function));
    if metadata.mir_fusion_candidate_group_count == 0 {
        return None;
    }
    let mut windows = derive_semantic_fusion_instruction_windows(
        instructions,
        instr_spans,
        &metadata.mir_fusion_candidate_groups,
    );
    windows.sort_by_key(|window| (window.span.start, window.span.end));
    let scaffold = FunctionFusionScaffold {
        signal_count: metadata.mir_fusion_signal_count,
        candidates: metadata.mir_fusion_candidate_groups,
        windows,
    };
    scaffold
        .valid_for(function, instructions, instr_spans)
        .then_some(scaffold)
}

#[cfg(test)]
mod tests {
    #[test]
    fn function_local_semantic_windows_survive_executable_materialization() {
        let source = "function y = f()\npersistent p\nt = p .* 2;\np = p + t + 1;\ny = p;\nend\n";
        let ast = runmat_parser::parse(source).unwrap();
        let hir = runmat_hir::lower(&ast, &runmat_hir::LoweringContext::empty()).unwrap();
        let mir = runmat_mir::lowering::lower_assembly(&hir.assembly).unwrap();
        assert!(hir.assembly.entrypoints.is_empty());
        let function = hir.assembly.functions[0].id;
        let registry = crate::bytecode::FunctionRegistry::new(
            crate::bytecode::compile_semantic_function_registry(&hir.assembly, &mir).unwrap(),
        );
        let body = registry.get(function).unwrap();
        let scaffold = body
            .fusion_scaffold
            .as_ref()
            .expect("function-owned MIR candidates");
        assert!(scaffold.valid_for(function, &body.instructions, &body.instr_spans));
        body.validate_fusion_scaffold()
            .expect("selected statement admission");
        let layout = crate::layout::derive_layout(&hir.assembly, &mir).unwrap();
        let executable = crate::Bytecode::for_function(body, registry.clone(), layout.clone());
        assert_eq!(
            executable.fusion_metadata.mir_fusion_candidate_group_count,
            scaffold.candidates.len()
        );
        assert!(executable
            .fusion_metadata
            .instruction_windows
            .iter()
            .all(|window| {
                body.instructions[window.span.start..=window.span.end]
                    .iter()
                    .all(|instruction| {
                        !matches!(
                            instruction,
                            crate::Instr::StoreVar(_) | crate::Instr::StoreLocal(_)
                        )
                    })
            }));

        let mut remapped = body.clone();
        let destination = runmat_hir::FunctionId(function.0 + 10);
        let portable = remapped.portable_function;
        let source = remapped.source_id;
        crate::bytecode::remap_session_function(
            &mut remapped,
            destination,
            portable,
            source,
            &[(function, destination)].into_iter().collect(),
        )
        .expect("session-owned remap");
        assert!(remapped
            .fusion_scaffold
            .as_ref()
            .is_some_and(|scaffold| scaffold.valid_for(
                destination,
                &remapped.instructions,
                &remapped.instr_spans
            )));
        remapped
            .validate_fusion_scaffold()
            .expect("remapped statement admission");
        let mut legacy = body.clone();
        legacy.fusion_scaffold = None;
        assert_ne!(
            crate::bytecode::semantic_definition_fingerprint(body).unwrap(),
            crate::bytecode::semantic_definition_fingerprint(&legacy).unwrap()
        );

        let mut malformed = body.clone();
        malformed.fusion_scaffold.as_mut().unwrap().candidates[0].function =
            runmat_hir::FunctionId(function.0 + 1);
        let malformed_registry =
            crate::bytecode::FunctionRegistry::new([(function, malformed)].into_iter().collect());
        let rejected = crate::Bytecode::for_function(
            malformed_registry.get(function).unwrap(),
            malformed_registry.clone(),
            layout,
        );
        assert!(rejected.runtime_fusion_groups().is_empty());
    }
}

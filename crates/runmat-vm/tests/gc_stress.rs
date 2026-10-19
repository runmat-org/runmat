//! Interpreter/collector integration under repeated execution.
use runmat_gc::{gc_collect_minor, gc_test_context};

#[test]
fn test_gc_with_interpreter_integration() {
    gc_test_context(|| {
        // Test GC under interpreter load
        use futures::executor::block_on;
        use runmat_hir::LoweringContext;
        use runmat_parser::parse;

        // Program that creates many temporary values
        let program = r#"
        result = 0;
        for i = 1:100
            temp1 = i * 2;
            temp2 = temp1 + 1;
            temp3 = temp2 * temp2;
            result = result + temp3;
        end
    "#;

        fn compile_source(
            source: &str,
        ) -> std::result::Result<runmat_vm::Bytecode, runmat_vm::CompileError> {
            let ast = parse(source).map_err(|err| runmat_vm::CompileError::new(err.to_string()))?;
            let lowering = runmat_hir::lower(&ast, &LoweringContext::empty())
                .map_err(|err| runmat_vm::CompileError::new(format!("{err:?}")))?;
            let mir = runmat_mir::lowering::lower_assembly(&lowering.assembly)
                .map_err(|err| runmat_vm::CompileError::new(format!("{err:?}")))?;
            let entrypoint = lowering
                .assembly
                .entrypoints
                .first()
                .ok_or_else(|| runmat_vm::CompileError::new("missing semantic entrypoint"))?
                .id;
            runmat_vm::compile(&lowering.assembly, &mir, entrypoint)
        }

        // Run the program multiple times to stress the GC
        for run in 0..50 {
            let bytecode = compile_source(program).expect("semantic compile should succeed");
            let vars = block_on(runmat_vm::interpret(&bytecode)).expect("execution should succeed");

            // Verify the result is consistent
            let result: f64 = (&vars[0]).try_into().expect("result should be a number");
            assert!(result > 0.0);

            if run % 10 == 9 {
                let collected = gc_collect_minor().expect("collection should succeed");
                println!(
                    "After {} interpreter runs, collected {} objects",
                    run + 1,
                    collected
                );
            }
        }
    });
}

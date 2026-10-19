use super::semantic_function_bindings;
use runmat_hir::FunctionId;
use runmat_vm::{Bytecode, FunctionBytecode, FunctionRegistry, Instr};
use std::collections::HashMap;

#[test]
fn text_callbacks_remain_names_while_explicit_handles_receive_bindings() {
    let helper = FunctionId(7);
    let mut bytecode = Bytecode::with_instructions(
        vec![
            Instr::LoadCharRow("helper".into()),
            Instr::CallFevalMulti(0, 1),
            Instr::LoadCharRow("helper".into()),
            Instr::CallBuiltinMulti("cellfun".into(), 1, 1),
            Instr::LoadString("helper".into()),
            Instr::CallBuiltinMulti("arrayfun".into(), 1, 1),
            Instr::CreateFunctionHandle("helper".into()),
        ],
        0,
    );
    bytecode.function_registry = FunctionRegistry::new(HashMap::from([(
        helper,
        FunctionBytecode {
            function: helper,
            display_name: "helper".into(),
            ..Default::default()
        },
    )]));

    let bindings = semantic_function_bindings(&bytecode).expect("literal binding selection");
    assert_eq!(bindings.len(), 1);
    assert_eq!(bindings[0].pc, 6);
    assert_eq!(bindings[0].function, helper);
    assert_eq!(bindings[0].display_name, "helper");
    assert!(matches!(bytecode.instructions[0], Instr::LoadCharRow(_)));
    assert!(matches!(bytecode.instructions[2], Instr::LoadCharRow(_)));
    assert!(matches!(bytecode.instructions[4], Instr::LoadString(_)));
}

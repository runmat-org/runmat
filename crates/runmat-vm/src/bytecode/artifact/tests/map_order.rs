use super::*;

#[test]
fn envelopes_are_deterministic_across_map_insertion_orders() {
    let mut left = Bytecode::with_instructions(vec![Instr::Return], 2);
    left.var_names.insert(0, "first".into());
    left.var_names.insert(1, "second".into());
    let mut right = Bytecode::with_instructions(vec![Instr::Return], 2);
    right.var_names.insert(1, "second".into());
    right.var_names.insert(0, "first".into());
    assert_eq!(
        encode_interpreter_script_v2(&left).unwrap(),
        encode_interpreter_script_v2(&right).unwrap()
    );

    let function = |id, name: &str| FunctionBytecode {
        function: FunctionId(id),
        display_name: name.into(),
        instructions: vec![Instr::Return],
        ..FunctionBytecode::default()
    };
    let left = FunctionRegistry::new(HashMap::from([
        (FunctionId(1), function(1, "first")),
        (FunctionId(2), function(2, "second")),
    ]));
    let right = FunctionRegistry::new(HashMap::from([
        (FunctionId(2), function(2, "second")),
        (FunctionId(1), function(1, "first")),
    ]));
    assert_eq!(
        encode_interpreter_program_v2(&left).unwrap(),
        encode_interpreter_program_v2(&right).unwrap()
    );
}

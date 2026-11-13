use super::*;

#[test]
fn multi_slot_activation_presence_has_one_canonical_wire_order() {
    let slots = HashSet::from([3, 1, 0, 2]);
    let mut script = Bytecode::with_instructions(vec![Instr::Return], 4);
    script.initially_unassigned_slots = slots.clone();
    let script_value = serde_json::to_value(&script).unwrap();
    assert_eq!(
        script_value["initially_unassigned_slots"],
        serde_json::json!([0, 1, 2, 3])
    );
    let encoded_script = encode_interpreter_script_v2(&script).unwrap();
    assert_eq!(
        decode_interpreter_script_v2(&encoded_script)
            .unwrap()
            .initially_unassigned_slots,
        slots
    );

    let mut unsorted_script = script_value;
    unsorted_script["initially_unassigned_slots"] = serde_json::json!([3, 1, 0, 2]);
    let unsorted_script = super::super::envelope::encode_payload(
        InterpreterPayloadForm::ScriptV2,
        SCRIPT_V2_PREAMBLE,
        &[("bytecode", BYTECODE_SCHEMA_VERSION)],
        &unsorted_script,
    )
    .unwrap();
    assert!(matches!(
        decode_interpreter_script_v2(&unsorted_script),
        Err(InterpreterPayloadError::InvalidEnvelope {
            form: InterpreterPayloadForm::ScriptV2,
            ..
        })
    ));

    let registry = FunctionRegistry::new(HashMap::from([(
        FunctionId(1),
        FunctionBytecode {
            function: FunctionId(1),
            var_count: 4,
            instructions: vec![Instr::Return],
            initially_unassigned_slots: slots.clone(),
            ..Default::default()
        },
    )]));
    let mut program_value = serde_json::to_value(&registry).unwrap();
    let function = program_value["functions"]
        .as_object_mut()
        .unwrap()
        .values_mut()
        .next()
        .unwrap();
    assert_eq!(
        function["initially_unassigned_slots"],
        serde_json::json!([0, 1, 2, 3])
    );
    let encoded_program = encode_interpreter_program_v2(&registry).unwrap();
    assert_eq!(
        decode_interpreter_program_v2(&encoded_program)
            .unwrap()
            .functions[&FunctionId(1)]
            .initially_unassigned_slots,
        slots
    );

    function["initially_unassigned_slots"] = serde_json::json!([3, 1, 0, 2]);
    let unsorted_program = super::super::envelope::encode_payload(
        InterpreterPayloadForm::ProgramV2,
        b"runmat-interpreter-program-v2\n",
        &[
            ("bytecode", BYTECODE_SCHEMA_VERSION),
            ("function_registry", FUNCTION_REGISTRY_SCHEMA_VERSION),
        ],
        &program_value,
    )
    .unwrap();
    assert!(matches!(
        decode_interpreter_program_v2(&unsorted_program),
        Err(InterpreterPayloadError::InvalidEnvelope {
            form: InterpreterPayloadForm::ProgramV2,
            ..
        })
    ));
}

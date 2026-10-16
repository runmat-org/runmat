use super::envelope::decode_envelope_bounded;
use super::*;
use crate::bytecode::{
    Bytecode, FunctionRegistry, BYTECODE_SCHEMA_VERSION, FUNCTION_REGISTRY_SCHEMA_VERSION,
};
use crate::{FunctionBytecode, Instr};
use runmat_hir::FunctionId;
use std::collections::{HashMap, HashSet};

const SCRIPT_V2_PREAMBLE: &[u8] = b"runmat-interpreter-script-v2\n";

#[cfg(feature = "native-accel")]
#[path = "tests/stage.rs"]
mod stage;

#[test]
fn struct_assignment_opcode_round_trips_and_pre_creation_artifacts_reject() {
    let bytecode =
        Bytecode::with_instructions(vec![Instr::LoadVarForStructAssignment(0), Instr::Return], 1);
    let encoded = encode_interpreter_script_v2(&bytecode).unwrap();
    let decoded = decode_interpreter_script_v2(&encoded).unwrap();
    assert!(matches!(
        decoded.instructions[0],
        Instr::LoadVarForStructAssignment(0)
    ));
    // Preserve the historical bytes: this was the current producer before schema 8.
    let historical = include_bytes!("../../../tests/fixtures/interpreter-program-current.artifact");
    assert!(matches!(
        decode_interpreter_program_v2(historical),
        Err(InterpreterPayloadError::UnsupportedRevision {
            field: InterpreterRevisionField::Bytecode,
            actual: 7,
            expected: BYTECODE_SCHEMA_VERSION,
        })
    ));
    // Statement-PC projection now travels with the bytecode's canonical
    // resume-point authority; an older envelope cannot silently omit it.
    let previous = b"runmat-interpreter-script-v2\nbytecode=14\n\n{}\n";
    assert!(matches!(
        decode_interpreter_script_v2(previous),
        Err(InterpreterPayloadError::UnsupportedRevision {
            field: InterpreterRevisionField::Bytecode,
            actual: 14,
            expected: BYTECODE_SCHEMA_VERSION,
        })
    ));
    // Bytecode 17 has no root-creation field on prepared sequence targets.
    // Reject its envelope before deserializing even a well-formed payload.
    let stale_sequence = super::envelope::encode_payload(
        InterpreterPayloadForm::ScriptV2,
        SCRIPT_V2_PREAMBLE,
        &[("bytecode", 17)],
        &serde_json::to_value(Bytecode::with_instructions(vec![Instr::Return], 0)).unwrap(),
    )
    .unwrap();
    assert!(matches!(
        decode_interpreter_script_v2(&stale_sequence),
        Err(InterpreterPayloadError::UnsupportedRevision {
            field: InterpreterRevisionField::Bytecode,
            actual: 17,
            expected: BYTECODE_SCHEMA_VERSION,
        })
    ));
    let stale_registry = format!("runmat-interpreter-program-v2\nbytecode={BYTECODE_SCHEMA_VERSION}\nfunction_registry=4\n\n{{}}\n");
    assert!(matches!(
        decode_interpreter_program_v2(stale_registry.as_bytes()),
        Err(InterpreterPayloadError::UnsupportedRevision {
            field: InterpreterRevisionField::FunctionRegistry,
            actual: 4,
            expected: FUNCTION_REGISTRY_SCHEMA_VERSION,
        })
    ));
    let future = format!(
        "runmat-interpreter-script-v2\nbytecode={}\n\n{{}}\n",
        BYTECODE_SCHEMA_VERSION + 1
    );
    assert!(matches!(
        decode_interpreter_script_v2(future.as_bytes()),
        Err(InterpreterPayloadError::UnsupportedRevision {
            field: InterpreterRevisionField::Bytecode,
            ..
        })
    ));
}

#[test]
fn sequence_root_creation_instruction_round_trips_with_current_revision() {
    let compile_round_trip = |source| {
        let ast = runmat_parser::parse(source).unwrap();
        let hir = runmat_hir::lower(&ast, &runmat_hir::LoweringContext::empty()).unwrap();
        let mir = runmat_mir::lowering::lower_assembly(&hir.assembly).unwrap();
        let bytecode =
            crate::bytecode::compile(&hir.assembly, &mir, hir.assembly.entrypoints[0].id).unwrap();
        decode_interpreter_script_v2(&encode_interpreter_script_v2(&bytecode).unwrap()).unwrap()
    };
    let direct = compile_round_trip("c{2} = 9;");
    assert!(direct
        .instructions
        .iter()
        .any(|instruction| matches!(instruction, Instr::LoadVarForCellAssignment(0))));

    let old_direct = super::envelope::encode_payload(
        InterpreterPayloadForm::ScriptV2,
        SCRIPT_V2_PREAMBLE,
        &[("bytecode", 18)],
        &serde_json::to_value(&direct).unwrap(),
    )
    .unwrap();
    assert!(matches!(
        decode_interpreter_script_v2(&old_direct),
        Err(InterpreterPayloadError::UnsupportedRevision {
            field: InterpreterRevisionField::Bytecode,
            actual: 18,
            expected: BYTECODE_SCHEMA_VERSION,
        })
    ));

    let decoded = compile_round_trip("[c{2}, y] = deal(9, 4);");
    assert!(matches!(
        decoded
            .instructions
            .iter()
            .find(|instruction| matches!(instruction, Instr::BeginSequenceOutputTarget { .. })),
        Some(Instr::BeginSequenceOutputTarget {
            root_slot: 0,
            root_creation: runmat_mir::SequenceRootCreation::EmptyCell,
        })
    ));
}

#[test]
fn current_script_and_program_envelopes_round_trip() {
    let bytecode = Bytecode::with_instructions(vec![Instr::Return], 0);
    let encoded_script = encode_interpreter_script_v2(&bytecode).unwrap();
    assert_eq!(
        decode_interpreter_script_v2(&encoded_script)
            .unwrap()
            .instructions
            .len(),
        1
    );
    let registry = FunctionRegistry::default();
    let encoded_program = encode_interpreter_program_v2(&registry).unwrap();
    let decoded = decode_interpreter_program_v2(&encoded_program).unwrap();
    assert!(decoded.functions.is_empty());
}

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
    let unsorted_script = super::envelope::encode_payload(
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
    let unsorted_program = super::envelope::encode_payload(
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

#[test]
fn portable_artifacts_reject_session_source_references_on_encode_and_import() {
    use crate::BytecodeSourceRef;
    let runtime = runmat_runtime::context::RuntimeContext::new(std::rc::Rc::new(
        runmat_runtime::execution::RuntimeExecutionService::new(),
    ));
    let lease = runmat_runtime::source_context::lease_source_context_in_context(
        &runtime,
        "forged.m",
        None,
        "answer = 1;",
    )
    .expect("source owner");

    let mut bytecode = Bytecode::with_instructions(vec![Instr::Return], 0);
    bytecode.source_id = Some(BytecodeSourceRef::Runtime(lease.source_ref()));
    assert!(encode_interpreter_script_v2(&bytecode).is_err());
    let mut forged_bytecode = Bytecode::with_instructions(vec![Instr::Return], 0);
    forged_bytecode.source_id = Some(BytecodeSourceRef::Program(runmat_types::ProgramSourceId(0)));
    let mut forged_value = serde_json::to_value(forged_bytecode).expect("portable template");
    forged_value["source_id"] = serde_json::json!({"domain": "runtime", "id": usize::MAX - 1});
    let forged = super::envelope::encode_payload(
        InterpreterPayloadForm::ScriptV2,
        SCRIPT_V2_PREAMBLE,
        &[("bytecode", BYTECODE_SCHEMA_VERSION)],
        &forged_value,
    )
    .expect("canonical forged transport");
    assert!(decode_interpreter_script_v2(&forged).is_err());

    let mut registry = FunctionRegistry::new(HashMap::from([(
        FunctionId(1),
        FunctionBytecode {
            function: FunctionId(1),
            source_id: Some(BytecodeSourceRef::Program(runmat_types::ProgramSourceId(0))),
            ..Default::default()
        },
    )]));
    let mut forged_value = serde_json::to_value(&registry).expect("portable template");
    registry
        .functions
        .get_mut(&FunctionId(1))
        .expect("one function")
        .source_id = Some(BytecodeSourceRef::Runtime(lease.source_ref()));
    assert!(encode_interpreter_program_v2(&registry).is_err());
    let function = forged_value["functions"]
        .as_object_mut()
        .expect("registry functions")
        .values_mut()
        .next()
        .expect("one function");
    function["source_id"] = serde_json::json!({"domain": "runtime", "id": usize::MAX - 1});
    let forged = super::envelope::encode_payload(
        InterpreterPayloadForm::ProgramV2,
        b"runmat-interpreter-program-v2\n",
        &[
            ("bytecode", BYTECODE_SCHEMA_VERSION),
            ("function_registry", FUNCTION_REGISTRY_SCHEMA_VERSION),
        ],
        &forged_value,
    )
    .expect("canonical forged registry transport");
    assert!(decode_interpreter_program_v2(&forged).is_err());
}

#[test]
fn frozen_stale_and_legacy_payloads_are_rejected() {
    let script = include_bytes!("../../../tests/fixtures/interpreter-script-bytecode-6.artifact");
    assert!(matches!(
        decode_interpreter_script_v2(script),
        Err(InterpreterPayloadError::UnsupportedRevision {
            field: InterpreterRevisionField::Bytecode,
            actual: 6,
            expected: BYTECODE_SCHEMA_VERSION,
        })
    ));
    let program = include_bytes!("../../../tests/fixtures/interpreter-program-registry-4.artifact");
    assert!(matches!(
        decode_interpreter_program_v2(program),
        Err(InterpreterPayloadError::UnsupportedRevision {
            field: InterpreterRevisionField::Bytecode,
            actual: 7,
            expected: BYTECODE_SCHEMA_VERSION,
        })
    ));
    let legacy = include_bytes!("../../../tests/fixtures/interpreter-program-legacy-raw.json");
    assert!(matches!(
        decode_interpreter_program_v2(legacy),
        Err(InterpreterPayloadError::LegacyPayload {
            expected: InterpreterPayloadForm::ProgramV2,
        })
    ));
    let legacy_script =
        include_bytes!("../../../tests/fixtures/interpreter-script-legacy-raw.json");
    assert!(matches!(
        decode_interpreter_script_v2(legacy_script),
        Err(InterpreterPayloadError::LegacyPayload {
            expected: InterpreterPayloadForm::ScriptV2,
        })
    ));
}

#[test]
fn revisions_are_rejected_before_the_body_is_deserialized() {
    let stale = b"runmat-interpreter-script-v2\nbytecode=6\n\nnot-json\n";
    assert!(matches!(
        decode_interpreter_script_v2(stale),
        Err(InterpreterPayloadError::UnsupportedRevision {
            field: InterpreterRevisionField::Bytecode,
            actual: 6,
            expected: BYTECODE_SCHEMA_VERSION,
        })
    ));

    let stale_over_limit = b"runmat-interpreter-script-v2\nbytecode=6\n\nchanged-schema-body\n";
    assert!(matches!(
        decode_envelope_bounded(
            stale_over_limit,
            InterpreterPayloadForm::ScriptV2,
            SCRIPT_V2_PREAMBLE,
            &[(
                "bytecode",
                InterpreterRevisionField::Bytecode,
                BYTECODE_SCHEMA_VERSION,
            )],
            8,
        ),
        Err(InterpreterPayloadError::UnsupportedRevision {
            field: InterpreterRevisionField::Bytecode,
            actual: 6,
            expected: BYTECODE_SCHEMA_VERSION,
        })
    ));
}

#[test]
fn noncanonical_current_body_is_rejected() {
    let payload = format!("runmat-interpreter-script-v2\nbytecode={BYTECODE_SCHEMA_VERSION}\n\n{{ \"instructions\": [\"Return\"], \"var_count\": 0 }}\n");
    assert!(matches!(
        decode_interpreter_script_v2(payload.as_bytes()),
        Err(InterpreterPayloadError::InvalidEnvelope {
            form: InterpreterPayloadForm::ScriptV2,
            ..
        })
    ));
}

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

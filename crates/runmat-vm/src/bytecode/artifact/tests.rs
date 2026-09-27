use super::envelope::decode_envelope_bounded;
use super::*;
use crate::bytecode::{
    Bytecode, FunctionRegistry, BYTECODE_SCHEMA_VERSION, FUNCTION_REGISTRY_SCHEMA_VERSION,
};
use crate::{FunctionBytecode, Instr};
use runmat_hir::FunctionId;
use std::collections::HashMap;

const SCRIPT_V2_PREAMBLE: &[u8] = b"runmat-interpreter-script-v2\n";

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

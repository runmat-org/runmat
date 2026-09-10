use super::envelope::{decode_canonical_body, decode_envelope, encode_payload};
use super::{InterpreterPayloadError, InterpreterPayloadForm, InterpreterRevisionField};
use crate::bytecode::{
    Bytecode, FunctionRegistry, BYTECODE_SCHEMA_VERSION, FUNCTION_REGISTRY_SCHEMA_VERSION,
};

const SCRIPT_V2_PREAMBLE: &[u8] = b"runmat-interpreter-script-v2\n";
const PROGRAM_V2_PREAMBLE: &[u8] = b"runmat-interpreter-program-v2\n";

pub fn encode_interpreter_script_v2(
    bytecode: &Bytecode,
) -> Result<Vec<u8>, InterpreterPayloadError> {
    encode_payload(
        InterpreterPayloadForm::ScriptV2,
        SCRIPT_V2_PREAMBLE,
        &[("bytecode", BYTECODE_SCHEMA_VERSION)],
        bytecode,
    )
}

pub fn decode_interpreter_script_v2(bytes: &[u8]) -> Result<Bytecode, InterpreterPayloadError> {
    let body = decode_envelope(
        bytes,
        InterpreterPayloadForm::ScriptV2,
        SCRIPT_V2_PREAMBLE,
        &[(
            "bytecode",
            InterpreterRevisionField::Bytecode,
            BYTECODE_SCHEMA_VERSION,
        )],
    )?;
    decode_canonical_body(body, InterpreterPayloadForm::ScriptV2)
}

pub fn encode_interpreter_program_v2(
    registry: &FunctionRegistry,
) -> Result<Vec<u8>, InterpreterPayloadError> {
    encode_payload(
        InterpreterPayloadForm::ProgramV2,
        PROGRAM_V2_PREAMBLE,
        &[
            ("bytecode", BYTECODE_SCHEMA_VERSION),
            ("function_registry", FUNCTION_REGISTRY_SCHEMA_VERSION),
        ],
        registry,
    )
}

pub fn decode_interpreter_program_v2(
    bytes: &[u8],
) -> Result<FunctionRegistry, InterpreterPayloadError> {
    let body = decode_envelope(
        bytes,
        InterpreterPayloadForm::ProgramV2,
        PROGRAM_V2_PREAMBLE,
        &[
            (
                "bytecode",
                InterpreterRevisionField::Bytecode,
                BYTECODE_SCHEMA_VERSION,
            ),
            (
                "function_registry",
                InterpreterRevisionField::FunctionRegistry,
                FUNCTION_REGISTRY_SCHEMA_VERSION,
            ),
        ],
    )?;
    decode_canonical_body(body, InterpreterPayloadForm::ProgramV2)
}

use super::envelope::{decode_canonical_body, decode_envelope, encode_payload};
use super::error::invalid;
use super::{InterpreterPayloadError, InterpreterPayloadForm, InterpreterRevisionField};
use crate::bytecode::{
    Bytecode, FunctionRegistry, BYTECODE_SCHEMA_VERSION, FUNCTION_REGISTRY_SCHEMA_VERSION,
};

const SCRIPT_V2_PREAMBLE: &[u8] = b"runmat-interpreter-script-v2\n";
const PROGRAM_V2_PREAMBLE: &[u8] = b"runmat-interpreter-program-v2\n";

pub fn encode_interpreter_script_v2(
    bytecode: &Bytecode,
) -> Result<Vec<u8>, InterpreterPayloadError> {
    bytecode
        .validate_portable_source_origins()
        .map_err(|message| invalid(InterpreterPayloadForm::ScriptV2, message))?;
    bytecode
        .validate_catch_scope_metadata()
        .map_err(|message| invalid(InterpreterPayloadForm::ScriptV2, message))?;
    crate::bytecode::validate_sequence_register_flow(&bytecode.instructions)
        .map_err(|message| invalid(InterpreterPayloadForm::ScriptV2, message))?;
    crate::bytecode::validate_stack_geometry(&bytecode.instructions, bytecode.stack_geometry)
        .map_err(|message| invalid(InterpreterPayloadForm::ScriptV2, message))?;
    bytecode
        .statement_pc_ranges()
        .map_err(|error| invalid(InterpreterPayloadForm::ScriptV2, error.to_string()))?;
    #[cfg(feature = "native-accel")]
    bytecode
        .validate_fusion_stage_projection()
        .map_err(|message| invalid(InterpreterPayloadForm::ScriptV2, message))?;
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
    let bytecode: Bytecode = decode_canonical_body(body, InterpreterPayloadForm::ScriptV2)?;
    bytecode
        .validate_portable_source_origins()
        .map_err(|message| invalid(InterpreterPayloadForm::ScriptV2, message))?;
    bytecode
        .validate_catch_scope_metadata()
        .map_err(|message| invalid(InterpreterPayloadForm::ScriptV2, message))?;
    crate::bytecode::validate_sequence_register_flow(&bytecode.instructions)
        .map_err(|message| invalid(InterpreterPayloadForm::ScriptV2, message))?;
    crate::bytecode::validate_stack_geometry(&bytecode.instructions, bytecode.stack_geometry)
        .map_err(|message| invalid(InterpreterPayloadForm::ScriptV2, message))?;
    bytecode
        .statement_pc_ranges()
        .map_err(|error| invalid(InterpreterPayloadForm::ScriptV2, error.to_string()))?;
    #[cfg(feature = "native-accel")]
    bytecode
        .validate_fusion_stage_projection()
        .map_err(|message| invalid(InterpreterPayloadForm::ScriptV2, message))?;
    // A decoded artifact never transports a process-local selected descriptor.
    // Rebuild it only after the canonical stream/projection validators pass.
    #[cfg(feature = "native-accel")]
    let _ = bytecode.sealed_selected_image_regions();
    Ok(bytecode)
}

pub fn encode_interpreter_program_v2(
    registry: &FunctionRegistry,
) -> Result<Vec<u8>, InterpreterPayloadError> {
    registry
        .validate_portable_source_origins()
        .map_err(|message| invalid(InterpreterPayloadForm::ProgramV2, message))?;
    for function in registry.functions.values() {
        function
            .validate_catch_scope_metadata()
            .map_err(|message| invalid(InterpreterPayloadForm::ProgramV2, message))?;
        crate::bytecode::validate_sequence_register_flow(&function.instructions)
            .map_err(|message| invalid(InterpreterPayloadForm::ProgramV2, message))?;
        function
            .statement_pc_ranges()
            .map_err(|error| invalid(InterpreterPayloadForm::ProgramV2, error.to_string()))?;
        #[cfg(feature = "native-accel")]
        function
            .validate_fusion_stage_projection()
            .map_err(|message| invalid(InterpreterPayloadForm::ProgramV2, message))?;
    }
    #[cfg(feature = "native-accel")]
    registry
        .validate_fusion_scaffolds_for_artifact()
        .map_err(|message| invalid(InterpreterPayloadForm::ProgramV2, message))?;
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
    let registry: FunctionRegistry =
        decode_canonical_body(body, InterpreterPayloadForm::ProgramV2)?;
    registry
        .validate_portable_source_origins()
        .map_err(|message| invalid(InterpreterPayloadForm::ProgramV2, message))?;
    for function in registry.functions.values() {
        function
            .validate_catch_scope_metadata()
            .map_err(|message| invalid(InterpreterPayloadForm::ProgramV2, message))?;
        crate::bytecode::validate_sequence_register_flow(&function.instructions)
            .map_err(|message| invalid(InterpreterPayloadForm::ProgramV2, message))?;
        function
            .statement_pc_ranges()
            .map_err(|error| invalid(InterpreterPayloadForm::ProgramV2, error.to_string()))?;
        #[cfg(feature = "native-accel")]
        function
            .validate_fusion_stage_projection()
            .map_err(|message| invalid(InterpreterPayloadForm::ProgramV2, message))?;
    }
    #[cfg(feature = "native-accel")]
    registry
        .validate_fusion_scaffolds_for_artifact()
        .map_err(|message| invalid(InterpreterPayloadForm::ProgramV2, message))?;
    #[cfg(feature = "native-accel")]
    registry.finalize_selected_image_views();
    Ok(registry)
}

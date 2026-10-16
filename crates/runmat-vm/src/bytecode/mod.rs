mod artifact;
#[cfg(feature = "native-accel")]
mod fusion_effects;
pub mod compile;
#[cfg(feature = "native-accel")]
pub(crate) mod fusion_private_temp;
#[cfg(feature = "native-accel")]
mod fusion_stage;
pub mod instr;
#[cfg(feature = "native-accel")]
mod multi_store;
mod parallel;
pub mod program;
mod region;
mod sequence_register;
mod source_ref;
mod statement_pc;
mod subscript;

/// Portable schema for serialized [`Bytecode`] payloads. Version 21 adds an
/// internal capture-only statement-output label for isolated numeric text.
pub const BYTECODE_SCHEMA_VERSION: u16 = 21;
/// Portable schema for serialized [`FunctionRegistry`] payloads.
pub const FUNCTION_REGISTRY_SCHEMA_VERSION: u16 = 16;

pub use artifact::{
    decode_interpreter_program_v2, decode_interpreter_script_v2, encode_interpreter_program_v2,
    encode_interpreter_script_v2, InterpreterPayloadError, InterpreterPayloadForm,
    InterpreterRevisionField,
};
pub use compile::{
    compile, compile_semantic_function_registry, compile_semantic_function_registry_with_analysis,
    compile_with_analysis,
};
#[cfg(feature = "native-accel")]
pub(crate) use fusion_stage::FusionStageSelectionOrigin;
#[cfg(feature = "native-accel")]
pub use fusion_stage::{CompiledFusionStageStatement, FusionStageProjection};
pub use instr::{
    AggregateElementSpec, BytecodeCodistributedOverload, BytecodeCollectiveOp,
    BytecodeDistributedBuildValidation, BytecodeDistributedOp, BytecodeSpmdHeader, EmitLabel,
    Instr, StackEffect,
};
#[cfg(feature = "native-accel")]
pub use multi_store::{
    ImageCheckpoint, ImageLiveOut, ImageLiveOutSource, ImageNormalizePublicationJoin,
    ImageRegionStore, ImageStoreDestination, ImageStoreOrdinal, ProspectiveMultiStoreRegion,
    ProspectiveSemanticMultiStoreRegion, ProspectiveStore,
};
pub use parallel::{BytecodeParallelVariable, BytecodeParforRegion, BytecodeSpmdRegion};
pub use program::{
    rebind_session_source, remap_closed_session_function, remap_decorated_session_definitions,
    remap_session_entrypoint, remap_session_function, SessionCallableBinding,
};
pub use program::{
    semantic_definition_fingerprint, AsyncMetadata, AwaitSite, Bytecode, FunctionBytecode,
    FunctionRegistry, FunctionSourceProjectionError, InheritedStorageKind, InheritedStorageSlot,
    PrivateSourceAlias, PrivateSourceDirectory, PublishedNameResolutionView, SpawnSite,
};
#[cfg(feature = "native-accel")]
pub use program::{
    FusionCandidateGroup, FusionInstructionKind, FusionInstructionWindow, FusionMetadata,
};
pub use region::{BytecodeRegion, BytecodeRegionBoundary};
pub use sequence_register::validate_sequence_register_flow;
pub use source_ref::BytecodeSourceRef;
pub use statement_pc::{StatementPcProjectionError, StatementPcRange};
pub use subscript::{
    BytecodeDottedInvokeArgument, BytecodeSubscriptSelector, BytecodeSubscriptStep,
};

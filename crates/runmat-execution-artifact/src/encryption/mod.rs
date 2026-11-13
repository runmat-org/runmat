mod context;
mod envelope_codec;
mod frame;
mod manifest;
#[cfg(not(target_arch = "wasm32"))]
mod native;
mod object_codec;
mod portable;
mod program_chunk;
mod program_manifest;
mod provider;
mod recovery;
mod run_key;
mod suite;

pub use context::{EncryptionContext, EncryptionPurpose};
pub use envelope_codec::{decode_run_key_envelope, encode_run_key_envelope};
pub use frame::{
    decode_transfer_wire_frame, encode_transfer_wire_frame, open_transfer_frame,
    seal_transfer_frame, OpenedTransferFrame, TransferFrameAuthority, TransferWireFrame,
    TRANSFER_FRAME_ENCRYPTION_SUITE,
};
pub use manifest::{EncryptedArtifact, ExecutionRecipientKey};
#[cfg(not(target_arch = "wasm32"))]
pub use native::{NativeExecutionEncryption, NativeExecutionPrivateKey};
pub use object_codec::{decode_encrypted_run_object, encode_encrypted_run_object};
pub use portable::{PortableExecutionEncryption, PortableExecutionPrivateKey};
pub use program_chunk::{
    open_program_chunk, seal_program_chunk, ProgramChunkAuthority,
    MAX_PROGRAM_CHUNK_CBOR_OVERHEAD_BYTES, MAX_PROGRAM_CHUNK_ENCODED_OBJECT_BYTES,
    MAX_PROGRAM_CHUNK_PLAINTEXT_BYTES,
};
pub use program_manifest::{
    decode_program_chunk_manifest, encode_program_chunk_manifest, open_program_chunk_manifest,
    seal_program_chunk_manifest, OpenedProgramChunkManifest, ProgramChunkManifest,
    ProgramChunkObject, ProgramManifestAuthority, MAX_PROGRAM_CHUNK_MANIFEST_BYTES,
    MAX_PROGRAM_CHUNK_MANIFEST_ENCODED_OBJECT_BYTES,
};
pub use provider::ExecutionEncryptionProvider;
pub use recovery::{open_recovery_artifact, RecoveryArtifactScope};
pub use run_key::{
    open_run_key, seal_run_key, EncryptedRunObject, RunKeyEnvelope, RunKeyMaterial,
    RunObjectEncryption, RunObjectEncryptionSuite,
};
pub use suite::ExecutionHpkeSuite;

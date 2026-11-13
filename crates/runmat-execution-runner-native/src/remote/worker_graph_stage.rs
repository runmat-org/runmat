//! Worker-side admission of one run-scoped encrypted Graph Program. The
//! caller owns state transitions; this module never reads ambient file paths.

use runmat_execution::Digest;
use runmat_execution_artifact::encryption::{
    decode_encrypted_run_object, open_program_chunk_manifest, ProgramManifestAuthority,
    RunKeyMaterial, MAX_PROGRAM_CHUNK_MANIFEST_BYTES,
};
use runmat_execution_transport_native::control::{
    DriverArtifactDownload, SignedArtifactDownloader,
};

use super::program_staging::{GraphProgramStageWriter, StagedGraphProgram};
use crate::{NativeExecutionError, NativeExecutionResult};

pub(super) async fn begin(
    run_key: RunKeyMaterial,
    run_identity: &str,
    encrypted_manifest: &[u8],
    maximum_program_bytes: u64,
) -> NativeExecutionResult<GraphProgramStageWriter> {
    if encrypted_manifest.len()
        > runmat_execution_artifact::encryption::MAX_PROGRAM_CHUNK_MANIFEST_ENCODED_OBJECT_BYTES
    {
        return Err(protocol("encrypted Graph Program manifest is oversized"));
    }
    let object =
        decode_encrypted_run_object(encrypted_manifest, MAX_PROGRAM_CHUNK_MANIFEST_BYTES + 16)
            .map_err(protocol)?;
    let authority = ProgramManifestAuthority {
        run_identity: run_identity.into(),
        key_epoch: 1,
    };
    let opened = open_program_chunk_manifest(&run_key, &authority, &object, maximum_program_bytes)
        .map_err(protocol)?;
    GraphProgramStageWriter::new(opened, run_key, maximum_program_bytes).await
}

pub(super) async fn download_next(
    writer: &mut GraphProgramStageWriter,
    downloader: &SignedArtifactDownloader,
    program_digest: Digest,
    ordinal: u32,
    signed: &DriverArtifactDownload,
) -> NativeExecutionResult<()> {
    if writer.manifest().program_digest != program_digest {
        return Err(protocol("Graph Program chunk names another Program"));
    }
    writer
        .download_chunk(downloader, ordinal as usize, signed)
        .await
}

pub(super) async fn finish(
    writer: GraphProgramStageWriter,
    program_digest: Digest,
) -> NativeExecutionResult<StagedGraphProgram> {
    if writer.manifest().program_digest != program_digest {
        return Err(protocol("Graph Program finish names another Program"));
    }
    writer.finish().await
}

fn protocol(error: impl std::fmt::Display) -> NativeExecutionError {
    NativeExecutionError::Protocol(error.to_string())
}

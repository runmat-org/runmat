//! A host-originated Graph uses the same authenticated manifest/chunk staging
//! as a downloaded remote Program. There is no raw-Graph control-frame route.
use super::*;
use runmat_execution_artifact::encryption::{
    decode_encrypted_run_object, open_program_chunk_manifest, ProgramManifestAuthority,
    RunKeyMaterial, MAX_PROGRAM_CHUNK_MANIFEST_BYTES,
};

impl RemoteGraphTransfer {
    /// Authenticate the manifest and bind its exact plaintext digest/length to
    /// this selected Graph and ordered encrypted-object download inventory.
    /// The worker independently reopens the same manifest under its run key.
    pub fn admit(
        graph: &GraphExecutionRequest,
        encrypted_manifest: Vec<u8>,
        ordered_downloads: Vec<DriverArtifactDownload>,
        maximum_program_bytes: u64,
        run_key: &RunKeyMaterial,
        authority: &ProgramManifestAuthority,
    ) -> NativeExecutionResult<Self> {
        let invalid = |message: &str| NativeExecutionError::Protocol(message.into());
        graph
            .validate()
            .map_err(|error| invalid(&error.to_string()))?;
        if graph.assignment.is_some() || graph.job_id.is_some() {
            return Err(invalid("remote Graph must be unassigned before admission"));
        }
        let encrypted =
            decode_encrypted_run_object(&encrypted_manifest, MAX_PROGRAM_CHUNK_MANIFEST_BYTES + 16)
                .map_err(|error| invalid(&error.to_string()))?;
        let opened =
            open_program_chunk_manifest(run_key, authority, &encrypted, maximum_program_bytes)
                .map_err(|error| invalid(&error.to_string()))?;
        // RemoteProgramObject::Graph is untagged, so this is exactly its wire
        // representation. Hash/count by streaming; do not duplicate its code.
        let (digest, length) =
            crate::metadata_identity::json_identity(b"", graph, maximum_program_bytes)?;
        let manifest = opened.manifest();
        if digest != manifest.program_digest
            || length != manifest.program_length
            || ordered_downloads.len() != manifest.chunks.len()
        {
            return Err(invalid(
                "encrypted Graph manifest differs from original Program bytes",
            ));
        }
        for (download, chunk) in ordered_downloads.iter().zip(&manifest.chunks) {
            if download.kind != "program-chunk"
                || download.artifact_id != chunk.artifact_id
                || download.method != "GET"
                || download.ciphertext_size_bytes != chunk.encoded_object_length
                || download.ciphertext_digest != chunk.encoded_object_digest.to_string()
            {
                return Err(invalid(
                    "Graph download inventory differs from authenticated manifest",
                ));
            }
        }
        Ok(Self {
            encrypted_manifest: Arc::from(encrypted_manifest),
            ordered_downloads: Arc::from(ordered_downloads),
            maximum_program_bytes,
            program_digest: digest,
            program_length: length,
        })
    }
}

//! Private, run-scoped staging of one authenticated Graph Program object.
//! No task or assignment authority is issued here; the child must compare the
//! final request digest with its later exact assignment before effects.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use runmat_execution::{Digest, DigestStream};
use runmat_execution_artifact::encryption::{
    decode_encrypted_run_object, open_program_chunk, OpenedProgramChunkManifest,
    ProgramChunkAuthority, RunKeyMaterial, MAX_PROGRAM_CHUNK_PLAINTEXT_BYTES,
};
use runmat_execution_transport_native::control::{
    DriverArtifactDownload, SignedArtifactDownloader,
};
use tokio::io::AsyncWriteExt as _;

use crate::{NativeExecutionError, NativeExecutionResult};

pub(super) struct StagedGraphProgram {
    _root: tempfile::TempDir,
    program_path: PathBuf,
    digest: Digest,
    length: u64,
}

/// The one ordered, private staging owner used by both signed HTTP downloads
/// and bounded worker transfer frames. An incomplete writer only owns files;
/// it cannot issue a staged Program reference.
pub(super) struct GraphProgramStageWriter {
    opened: OpenedProgramChunkManifest,
    run_key: RunKeyMaterial,
    root: tempfile::TempDir,
    output: tokio::fs::File,
    program_path: PathBuf,
    hasher: DigestStream,
    length: u64,
    next_chunk: usize,
}

impl StagedGraphProgram {
    /// Only the owning worker may pass this file through its authenticated
    /// parent-created child root. Never serialize this absolute path in QUIC.
    pub(super) fn private_path(&self) -> &Path {
        // The owner retains TempDir until the selected child has retired.
        &self.program_path
    }

    pub(super) fn digest(&self) -> Digest {
        self.digest
    }

    pub(super) fn length(&self) -> u64 {
        self.length
    }
}

impl GraphProgramStageWriter {
    pub(super) async fn new(
        opened: OpenedProgramChunkManifest,
        run_key: RunKeyMaterial,
        maximum_program_bytes: u64,
    ) -> NativeExecutionResult<Self> {
        opened
            .manifest()
            .validate(maximum_program_bytes)
            .map_err(protocol)?;
        let root = tempfile::Builder::new()
            .prefix("runmat-graph-program-")
            .tempdir()
            .map_err(protocol)?;
        let program_path = root.path().join("program.bin");
        let output = tokio::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&program_path)
            .await
            .map_err(protocol)?;
        Ok(Self {
            opened,
            run_key,
            root,
            output,
            program_path,
            hasher: DigestStream::new(),
            length: 0,
            next_chunk: 0,
        })
    }

    pub(super) fn manifest(&self) -> &runmat_execution_artifact::encryption::ProgramChunkManifest {
        self.opened.manifest()
    }

    pub(super) async fn push_encoded_chunk(
        &mut self,
        ordinal: usize,
        encoded: &[u8],
    ) -> NativeExecutionResult<()> {
        if ordinal != self.next_chunk {
            return Err(protocol("Graph Program chunks are out of order"));
        }
        let manifest = self.opened.manifest();
        let entry = manifest
            .chunks
            .get(ordinal)
            .ok_or_else(|| protocol("Graph Program chunk ordinal is out of range"))?;
        if encoded.len() as u64 != entry.encoded_object_length
            || Digest::sha256(encoded) != entry.encoded_object_digest
        {
            return Err(protocol("Graph Program chunk object differs from manifest"));
        }
        let encrypted =
            decode_encrypted_run_object(encoded, MAX_PROGRAM_CHUNK_PLAINTEXT_BYTES + 16)
                .map_err(protocol)?;
        let authority = ProgramChunkAuthority {
            run_identity: self.opened.run_identity().into(),
            program_digest: manifest.program_digest,
            program_length: manifest.program_length,
            chunk_index: ordinal as u64,
            chunk_count: manifest.chunks.len() as u64,
            chunk_length: entry.plaintext_length,
            chunk_digest: entry.plaintext_digest,
            key_epoch: self.opened.key_epoch(),
        };
        let plaintext =
            open_program_chunk(&self.run_key, &authority, &encrypted).map_err(protocol)?;
        self.output.write_all(&plaintext).await.map_err(protocol)?;
        self.hasher.update(&plaintext);
        self.length = self
            .length
            .checked_add(plaintext.len() as u64)
            .ok_or_else(|| protocol("Graph Program length overflow"))?;
        self.next_chunk += 1;
        Ok(())
    }

    pub(super) async fn download_chunk(
        &mut self,
        downloader: &SignedArtifactDownloader,
        ordinal: usize,
        signed: &DriverArtifactDownload,
    ) -> NativeExecutionResult<()> {
        let entry = self
            .manifest()
            .chunks
            .get(ordinal)
            .ok_or_else(|| protocol("Graph Program chunk ordinal is out of range"))?;
        if signed.kind != "program-chunk"
            || signed.artifact_id != entry.artifact_id
            || signed.ciphertext_size_bytes != entry.encoded_object_length
            || signed.ciphertext_digest != entry.encoded_object_digest.to_string()
        {
            return Err(protocol("signed Graph Program chunk differs from manifest"));
        }
        let encoded_path = self.root.path().join(format!("chunk-{ordinal}.enc"));
        let mut encoded_file = tokio::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&encoded_path)
            .await
            .map_err(protocol)?;
        downloader
            .download_to(signed, &mut encoded_file, entry.encoded_object_length)
            .await
            .map_err(protocol)?;
        drop(encoded_file);
        let encoded = tokio::fs::read(&encoded_path).await.map_err(protocol)?;
        self.push_encoded_chunk(ordinal, &encoded).await?;
        tokio::fs::remove_file(&encoded_path)
            .await
            .map_err(protocol)?;
        Ok(())
    }

    pub(super) async fn finish(mut self) -> NativeExecutionResult<StagedGraphProgram> {
        let expected_count = self.opened.manifest().chunks.len();
        let expected_length = self.opened.manifest().program_length;
        let expected_digest = self.opened.manifest().program_digest;
        if self.next_chunk != expected_count
            || self.length != expected_length
            || self.hasher.finish() != expected_digest
        {
            return Err(protocol(
                "staged Graph Program differs from manifest identity",
            ));
        }
        self.output.flush().await.map_err(protocol)?;
        drop(self.output);
        Ok(StagedGraphProgram {
            _root: self.root,
            program_path: self.program_path,
            digest: expected_digest,
            length: self.length,
        })
    }
}

pub(super) async fn download_and_stage(
    downloader: &SignedArtifactDownloader,
    run_key: &RunKeyMaterial,
    opened: OpenedProgramChunkManifest,
    signed_downloads: &[DriverArtifactDownload],
    maximum_program_bytes: u64,
) -> NativeExecutionResult<StagedGraphProgram> {
    let mut stage =
        GraphProgramStageWriter::new(opened, run_key.clone(), maximum_program_bytes).await?;
    let mut downloads = HashMap::with_capacity(stage.manifest().chunks.len());
    for download in signed_downloads {
        if download.kind == "program-chunk" {
            if downloads
                .insert(download.artifact_id.as_str(), download)
                .is_some()
            {
                return Err(protocol("duplicate signed Graph Program chunk descriptor"));
            }
            if downloads.len() > stage.manifest().chunks.len() {
                return Err(protocol("too many signed Graph Program chunk descriptors"));
            }
        }
    }
    if downloads.len() != stage.manifest().chunks.len() {
        return Err(protocol(
            "Graph Program chunk download count differs from manifest",
        ));
    }
    for ordinal in 0..stage.manifest().chunks.len() {
        let entry = &stage.manifest().chunks[ordinal];
        let signed = downloads
            .get(entry.artifact_id.as_str())
            .ok_or_else(|| protocol("Graph Program chunk has no signed download"))?;
        stage.download_chunk(downloader, ordinal, signed).await?;
    }
    stage.finish().await
}

fn protocol(error: impl std::fmt::Display) -> NativeExecutionError {
    NativeExecutionError::Protocol(error.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use runmat_execution_artifact::encryption::{
        encode_encrypted_run_object, open_program_chunk_manifest, seal_program_chunk,
        seal_program_chunk_manifest, ProgramChunkManifest, ProgramChunkObject,
        ProgramManifestAuthority,
    };

    #[tokio::test]
    async fn staged_program_is_issued_only_after_ordered_authenticated_completion() {
        let key = RunKeyMaterial::from_entropy([7; 32]).expect("run key");
        let program = b"graph request";
        let full_digest = Digest::sha256(program);
        let chunk = ProgramChunkAuthority {
            run_identity: "run-staging".into(),
            program_digest: full_digest,
            program_length: program.len() as u64,
            chunk_index: 0,
            chunk_count: 1,
            chunk_length: program.len() as u32,
            chunk_digest: full_digest,
            key_epoch: 1,
        };
        let encrypted = seal_program_chunk(&key, [8; 32], &chunk, program).expect("seal chunk");
        let encoded = encode_encrypted_run_object(&encrypted).expect("encode chunk");
        let manifest = ProgramChunkManifest {
            schema_version: 1,
            program_digest: full_digest,
            program_length: program.len() as u64,
            chunks: vec![ProgramChunkObject {
                artifact_id: "chunk-0".into(),
                plaintext_digest: full_digest,
                plaintext_length: program.len() as u32,
                encoded_object_digest: Digest::sha256(&encoded),
                encoded_object_length: encoded.len() as u64,
            }],
        };
        let authority = ProgramManifestAuthority {
            run_identity: "run-staging".into(),
            key_epoch: 1,
        };
        let sealed = seal_program_chunk_manifest(&key, [9; 32], &authority, &manifest, 128)
            .expect("seal manifest");
        let opened =
            open_program_chunk_manifest(&key, &authority, &sealed, 128).expect("open manifest");
        let mut writer = GraphProgramStageWriter::new(opened, key, 128)
            .await
            .expect("prepare staging");
        assert!(writer.push_encoded_chunk(1, &encoded).await.is_err());
        writer
            .push_encoded_chunk(0, &encoded)
            .await
            .expect("append exact chunk");
        let staged = writer.finish().await.expect("issue completed stage");
        assert_eq!(staged.digest(), full_digest);
        assert_eq!(staged.length(), program.len() as u64);
        assert_eq!(
            tokio::fs::read(staged.private_path()).await.unwrap(),
            program
        );
    }
}

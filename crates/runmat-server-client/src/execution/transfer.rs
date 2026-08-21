use anyhow::{bail, Context, Result};
use reqwest::Method;
use sha2::{Digest as _, Sha256};

use crate::execution::{public_error, to_i64};
use crate::public_api::{self, types};

const CHUNK_BYTES: usize = 8 * 1024 * 1024;

#[derive(Clone)]
pub struct ExecutionClient {
    api: public_api::Client,
    transfer: reqwest::Client,
}

pub struct ExecutionArtifactUpload {
    pub kind: types::ArtifactKindRequest,
    pub ciphertext: Vec<u8>,
    pub media_type: String,
    pub encryption_suite: String,
    pub retain_for_seconds: u64,
}

impl ExecutionClient {
    pub fn new(server_url: &str, token: &str) -> Result<Self> {
        Ok(Self {
            api: crate::auth::build_public_client(server_url, token)?,
            // Signed blob targets carry their own authorization. Deliberately
            // do not attach the RunMat bearer token to this client.
            transfer: reqwest::Client::new(),
        })
    }

    pub fn api(&self) -> &public_api::Client {
        &self.api
    }

    pub async fn upload_artifact(
        &self,
        project_id: &str,
        endpoint_fingerprint: &str,
        upload: ExecutionArtifactUpload,
    ) -> Result<types::ArtifactResponse> {
        if upload.ciphertext.is_empty() {
            bail!("execution artifact ciphertext cannot be empty");
        }
        let size = upload.ciphertext.len() as u64;
        let digest = format!("sha256:{:x}", Sha256::digest(&upload.ciphertext));
        let artifact = self
            .api
            .register_execution_artifact(
                project_id,
                &types::RegisterArtifactRequest {
                    kind: upload.kind,
                    ciphertext_digest: digest,
                    ciphertext_size_bytes: to_i64(size, "ciphertext size")?,
                    media_type: upload.media_type,
                    encryption_suite: upload.encryption_suite,
                    retain_for_seconds: to_i64(upload.retain_for_seconds, "artifact retention")?,
                },
            )
            .await
            .map_err(public_error)?
            .into_inner();
        let grant = self
            .api
            .create_artifact_grant(
                project_id,
                &artifact.id,
                &types::CreateArtifactGrantRequest {
                    allocation_lease_id: None,
                    endpoint_fingerprint: endpoint_fingerprint.into(),
                    maximum_bytes: to_i64(size, "artifact grant size")?,
                    permission: types::ArtifactGrantPermissionRequest::Upload,
                    ttl_seconds: 3_600,
                },
            )
            .await
            .map_err(public_error)?
            .into_inner();
        let transfer = self
            .api
            .begin_artifact_transfer(
                project_id,
                &grant.id,
                &types::BeginTransferRequest {
                    endpoint_fingerprint: endpoint_fingerprint.into(),
                },
            )
            .await
            .map_err(public_error)?
            .into_inner();
        if transfer.method != "SESSION" {
            bail!("execution artifact upload returned an unsupported transfer mode");
        }

        let chunks = upload
            .ciphertext
            .chunks(CHUNK_BYTES)
            .enumerate()
            .map(|(index, bytes)| {
                Ok(types::UploadChunkRequestBody {
                    chunk_index: i32::try_from(index)
                        .context("execution artifact has too many chunks")?,
                    offset_bytes: to_i64((index * CHUNK_BYTES) as u64, "chunk offset")?,
                    size_bytes: to_i64(bytes.len() as u64, "chunk size")?,
                    ciphertext_sha256: format!("{:x}", Sha256::digest(bytes)),
                })
            })
            .collect::<Result<Vec<_>>>()?;
        let targets = self
            .api
            .prepare_artifact_transfer_chunks(
                project_id,
                &transfer.transfer_id,
                &types::PrepareTransferChunksRequest { chunks },
            )
            .await
            .map_err(public_error)?
            .into_inner()
            .targets;
        if targets.len() != upload.ciphertext.chunks(CHUNK_BYTES).count() {
            bail!("execution artifact upload returned an incomplete target set");
        }
        for target in targets {
            let index =
                usize::try_from(target.chunk_index).context("negative upload chunk index")?;
            let bytes = upload
                .ciphertext
                .chunks(CHUNK_BYTES)
                .nth(index)
                .context("upload target names an unknown chunk")?;
            self.send_target(&target.method, &target.url, &target.headers, bytes.to_vec())
                .await?;
        }
        self.api
            .complete_artifact_transfer(
                project_id,
                &transfer.transfer_id,
                &types::CompleteTransferRequest {
                    bytes_transferred: to_i64(size, "transferred bytes")?,
                },
            )
            .await
            .map_err(public_error)?;
        Ok(artifact)
    }

    pub async fn download_artifact(
        &self,
        project_id: &str,
        artifact_id: &str,
        endpoint_fingerprint: &str,
    ) -> Result<Vec<u8>> {
        let artifact = self
            .api
            .get_execution_artifact(project_id, artifact_id)
            .await
            .map_err(public_error)?
            .into_inner();
        let grant = self
            .api
            .create_artifact_grant(
                project_id,
                artifact_id,
                &types::CreateArtifactGrantRequest {
                    allocation_lease_id: None,
                    endpoint_fingerprint: endpoint_fingerprint.into(),
                    maximum_bytes: artifact.ciphertext_size_bytes,
                    permission: types::ArtifactGrantPermissionRequest::Download,
                    ttl_seconds: 3_600,
                },
            )
            .await
            .map_err(public_error)?
            .into_inner();
        let transfer = self
            .api
            .begin_artifact_transfer(
                project_id,
                &grant.id,
                &types::BeginTransferRequest {
                    endpoint_fingerprint: endpoint_fingerprint.into(),
                },
            )
            .await
            .map_err(public_error)?
            .into_inner();
        let bytes = self
            .send_target(
                &transfer.method,
                &transfer.url,
                &transfer.headers,
                Vec::new(),
            )
            .await?;
        if bytes.len() as i64 != artifact.ciphertext_size_bytes
            || format!("sha256:{:x}", Sha256::digest(&bytes)) != artifact.ciphertext_digest
        {
            bail!("downloaded execution artifact failed exact ciphertext verification");
        }
        self.api
            .complete_artifact_transfer(
                project_id,
                &transfer.transfer_id,
                &types::CompleteTransferRequest {
                    bytes_transferred: artifact.ciphertext_size_bytes,
                },
            )
            .await
            .map_err(public_error)?;
        Ok(bytes)
    }

    async fn send_target(
        &self,
        method: &str,
        url: &str,
        headers: &std::collections::HashMap<String, String>,
        body: Vec<u8>,
    ) -> Result<Vec<u8>> {
        let method = Method::from_bytes(method.as_bytes()).context("invalid transfer method")?;
        let mut request = self.transfer.request(method, url);
        for (name, value) in headers {
            request = request.header(name, value);
        }
        if !body.is_empty() {
            request = request.body(body);
        }
        let response = request
            .send()
            .await
            .context("execution artifact transfer failed")?;
        let status = response.status();
        let bytes = response.bytes().await.context("read transfer response")?;
        if !status.is_success() {
            bail!("execution artifact transfer failed with HTTP {status}");
        }
        Ok(bytes.to_vec())
    }
}

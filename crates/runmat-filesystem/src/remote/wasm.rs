use crate::{DirEntry, FileHandle, FsFileType, FsMetadata, FsProvider, OpenFlags};
use js_sys::{ArrayBuffer, Uint8Array};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::io::{self, ErrorKind, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::time::{Duration, SystemTime, UNIX_EPOCH};
use url::Url;
use uuid::Uuid;
use wasm_bindgen::JsCast;
use wasm_bindgen::JsValue;

const MANIFEST_HASH: &str = "manifest:v1";
const SHARD_PREFIX: &str = "/.runmat/shards";
use web_sys::{XmlHttpRequest, XmlHttpRequestResponseType};

#[derive(Clone, Debug)]
pub struct RemoteFsConfig {
    pub base_url: String,
    pub auth_token: Option<String>,
    pub chunk_bytes: usize,
    pub direct_read_threshold_bytes: u64,
    pub direct_write_threshold_bytes: u64,
    pub shard_threshold_bytes: u64,
    pub shard_size_bytes: u64,
    pub timeout_ms: u32,
    pub retry_max_attempts: usize,
    pub retry_base_delay_ms: u32,
    pub retry_max_delay_ms: u32,
}

impl Default for RemoteFsConfig {
    fn default() -> Self {
        Self {
            base_url: String::new(),
            auth_token: None,
            chunk_bytes: 16 * 1024 * 1024,
            direct_read_threshold_bytes: 64 * 1024 * 1024,
            direct_write_threshold_bytes: 64 * 1024 * 1024,
            shard_threshold_bytes: 4 * 1024 * 1024 * 1024,
            shard_size_bytes: 512 * 1024 * 1024,
            timeout_ms: 120_000,
            retry_max_attempts: 5,
            retry_base_delay_ms: 100,
            retry_max_delay_ms: 2_000,
        }
    }
}

pub struct RemoteFsProvider {
    base: Url,
    auth_header: Option<String>,
    chunk_bytes: usize,
    direct_read_threshold_bytes: u64,
    shard_threshold_bytes: u64,
    shard_size_bytes: u64,
    timeout_ms: u32,
}

impl RemoteFsProvider {
    pub fn new(config: RemoteFsConfig) -> io::Result<Self> {
        if config.base_url.is_empty() {
            return Err(io::Error::new(
                ErrorKind::InvalidInput,
                "RemoteFsConfig.base_url must be provided",
            ));
        }
        let base = Url::parse(&config.base_url).map_err(map_url_err)?;
        Ok(Self {
            base,
            auth_header: config.auth_token.map(|token| format!("Bearer {token}")),
            chunk_bytes: config.chunk_bytes.max(64 * 1024),
            direct_read_threshold_bytes: config.direct_read_threshold_bytes,
            shard_threshold_bytes: config.shard_threshold_bytes,
            shard_size_bytes: config.shard_size_bytes.max(8 * 1024 * 1024),
            timeout_ms: config.timeout_ms.max(1),
        })
    }

    fn endpoint(&self, route: &str, query: &[(&str, String)]) -> io::Result<String> {
        let mut url = self
            .base
            .join(route.trim_start_matches('/'))
            .map_err(map_url_err)?;
        {
            let mut pairs = url.query_pairs_mut();
            for (k, v) in query {
                pairs.append_pair(k, v);
            }
        }
        Ok(url.into())
    }

    fn send_text(
        &self,
        method: &str,
        route: &str,
        query: &[(&str, String)],
        body: Option<&[u8]>,
        content_type: Option<&str>,
    ) -> io::Result<String> {
        let url = self.endpoint(route, query)?;
        let xhr = self.prepare_xhr(method, &url, XmlHttpRequestResponseType::Text)?;
        self.apply_headers(&xhr, content_type)?;
        self.dispatch(&xhr, body)?;
        self.read_text(&xhr)
    }

    fn send_bytes(
        &self,
        method: &str,
        route: &str,
        query: &[(&str, String)],
        body: Option<&[u8]>,
        content_type: Option<&str>,
    ) -> io::Result<Vec<u8>> {
        let url = self.endpoint(route, query)?;
        let xhr = self.prepare_xhr(method, &url, XmlHttpRequestResponseType::Arraybuffer)?;
        self.apply_headers(&xhr, content_type)?;
        self.dispatch(&xhr, body)?;
        self.read_bytes(&xhr)
    }

    fn fetch_download_url(&self, path: &str) -> io::Result<DownloadUrlResponse> {
        let text = self.send_text(
            "GET",
            "/fs/download-url",
            &[("path", path.to_string())],
            None,
            None,
        )?;
        serde_json::from_str(&text).map_err(map_serde_err)
    }

    fn read_range_from_url(&self, url: &str, offset: u64, length: u64) -> io::Result<Vec<u8>> {
        if length == 0 {
            return Ok(Vec::new());
        }
        let end = offset + length - 1;
        let xhr = self.prepare_xhr("GET", url, XmlHttpRequestResponseType::Arraybuffer)?;
        let range = format!("bytes={offset}-{end}");
        xhr.set_request_header("Range", &range)
            .map_err(|err| map_js_error("set_request_header", err))?;
        self.apply_headers(&xhr, None)?;
        self.dispatch(&xhr, None)?;
        self.read_bytes(&xhr)
    }

    fn send_empty(&self, method: &str, route: &str, query: &[(&str, String)]) -> io::Result<()> {
        let url = self.endpoint(route, query)?;
        let xhr = self.prepare_xhr(method, &url, XmlHttpRequestResponseType::Text)?;
        self.apply_headers(&xhr, None)?;
        self.dispatch(&xhr, None)?;
        Ok(())
    }

    fn prepare_xhr(
        &self,
        method: &str,
        url: &str,
        response_type: XmlHttpRequestResponseType,
    ) -> io::Result<XmlHttpRequest> {
        let xhr = XmlHttpRequest::new().map_err(|err| map_js_error("XmlHttpRequest::new", err))?;
        xhr.open_with_async(method, url, false)
            .map_err(|err| map_js_error("XmlHttpRequest::open", err))?;
        xhr.set_response_type(response_type);
        xhr.set_timeout(self.timeout_ms);
        Ok(xhr)
    }

    fn apply_headers(&self, xhr: &XmlHttpRequest, content_type: Option<&str>) -> io::Result<()> {
        if let Some(auth) = &self.auth_header {
            xhr.set_request_header("Authorization", auth)
                .map_err(|err| map_js_error("XmlHttpRequest::set_request_header", err))?;
        }
        xhr.set_request_header("X-RunMat-Client", "remote-fs-wasm")
            .map_err(|err| map_js_error("XmlHttpRequest::set_request_header", err))?;
        if let Some(ct) = content_type {
            xhr.set_request_header("Content-Type", ct)
                .map_err(|err| map_js_error("XmlHttpRequest::set_request_header", err))?;
        }
        Ok(())
    }

    fn dispatch(&self, xhr: &XmlHttpRequest, body: Option<&[u8]>) -> io::Result<()> {
        match body {
            Some(bytes) => {
                let array = Uint8Array::new_with_length(bytes.len() as u32);
                array.copy_from(bytes);
                xhr.send_with_opt_array_buffer(Some(&array.buffer()))
                    .map_err(|err| map_js_error("XmlHttpRequest::send", err))?;
            }
            None => {
                xhr.send()
                    .map_err(|err| map_js_error("XmlHttpRequest::send", err))?;
            }
        }
        Ok(())
    }

    fn read_text(&self, xhr: &XmlHttpRequest) -> io::Result<String> {
        let status = xhr
            .status()
            .map_err(|err| map_js_error("XmlHttpRequest::status", err))?;
        if status < 200 || status >= 300 {
            return Err(self.status_error(xhr, status));
        }
        xhr.response_text()
            .map_err(|err| map_js_error("XmlHttpRequest::response_text", err))?
            .ok_or_else(|| {
                io::Error::new(
                    ErrorKind::Other,
                    "remote fs: empty text response from server",
                )
            })
    }

    fn read_bytes(&self, xhr: &XmlHttpRequest) -> io::Result<Vec<u8>> {
        let status = xhr
            .status()
            .map_err(|err| map_js_error("XmlHttpRequest::status", err))?;
        if status < 200 || status >= 300 {
            return Err(self.status_error(xhr, status));
        }
        let value = xhr
            .response()
            .map_err(|err| map_js_error("XmlHttpRequest::response", err))?;
        if value.is_null() || value.is_undefined() {
            return Ok(Vec::new());
        }
        let buffer = value.dyn_into::<js_sys::ArrayBuffer>().map_err(|_| {
            io::Error::new(ErrorKind::InvalidData, "remote fs: expected ArrayBuffer")
        })?;
        let view = Uint8Array::new(&buffer);
        let mut out = vec![0u8; view.length() as usize];
        view.copy_to(&mut out);
        Ok(out)
    }

    fn status_error(&self, xhr: &XmlHttpRequest, status: u16) -> io::Error {
        let message = xhr
            .response_text()
            .ok()
            .flatten()
            .filter(|s| !s.is_empty())
            .unwrap_or_else(|| xhr.status_text());
        let kind = match status {
            404 => ErrorKind::NotFound,
            401 | 403 => ErrorKind::PermissionDenied,
            409 => ErrorKind::AlreadyExists,
            400 => ErrorKind::InvalidInput,
            _ => ErrorKind::Other,
        };
        io::Error::new(kind, format!("remote fs http error ({status}): {message}"))
    }

    fn fetch_metadata(&self, path: &str) -> io::Result<MetadataResponse> {
        let text = self.send_text(
            "GET",
            "/fs/metadata",
            &[("path", path.to_string())],
            None,
            None,
        )?;
        serde_json::from_str(&text).map_err(map_serde_err)
    }

    fn fetch_dir(&self, path: &str) -> io::Result<Vec<DirEntryResponse>> {
        let text = self.send_text("GET", "/fs/dir", &[("path", path.to_string())], None, None)?;
        serde_json::from_str(&text).map_err(map_serde_err)
    }

    fn fetch_canonical_path(&self, path: &str) -> io::Result<String> {
        let text = self.send_text(
            "GET",
            "/fs/canonicalize",
            &[("path", path.to_string())],
            None,
            None,
        )?;
        let payload: CanonicalizeResponse = serde_json::from_str(&text).map_err(map_serde_err)?;
        Ok(payload.path)
    }

    fn download_chunk(&self, path: &str, offset: u64, length: usize) -> io::Result<Vec<u8>> {
        self.send_bytes(
            "GET",
            "/fs/read",
            &[
                ("path", path.to_string()),
                ("offset", offset.to_string()),
                ("length", length.to_string()),
            ],
            None,
            None,
        )
    }

    fn upload_chunk(
        &self,
        path: &str,
        offset: u64,
        truncate: bool,
        final_chunk: bool,
        data: &[u8],
        hash: Option<&str>,
    ) -> io::Result<Option<FsWriteSessionResponse>> {
        let mut query = vec![("path", path.to_string()), ("offset", offset.to_string())];
        if truncate {
            query.push(("truncate", "true".into()));
        }
        if final_chunk {
            query.push(("final", "true".into()));
        }
        if let Some(hash) = hash {
            query.push(("hash", hash.to_string()));
        }
        let text = self.send_text("PUT", "/fs/write", &query, Some(data), None)?;
        if text.is_empty() {
            return Ok(None);
        }
        let session: FsWriteSessionResponse = serde_json::from_str(&text).map_err(map_serde_err)?;
        Ok(Some(session))
    }

    fn upload_session_start(
        &self,
        path: &str,
        size_bytes: u64,
        content_sha256: &str,
    ) -> io::Result<UploadSessionStartResponse> {
        let body = UploadSessionStartRequest {
            path: path.to_string(),
            size_bytes: size_bytes as i64,
            content_type: None,
            content_sha256: content_sha256.to_string(),
        };
        let payload = serde_json::to_vec(&body).map_err(map_serde_err)?;
        let text = self.send_text(
            "POST",
            "/fs/upload-session/start",
            &[],
            Some(&payload),
            Some("application/json"),
        )?;
        serde_json::from_str(&text).map_err(map_serde_err)
    }

    fn upload_session_chunks(
        &self,
        session_id: &str,
        blob_key: &str,
        chunks: Vec<UploadSessionChunkDescriptor>,
    ) -> io::Result<UploadSessionChunksResponse> {
        let body = UploadSessionChunksRequest {
            session_id: session_id.to_string(),
            blob_key: blob_key.to_string(),
            chunks,
        };
        let payload = serde_json::to_vec(&body).map_err(map_serde_err)?;
        let text = self.send_text(
            "POST",
            "/fs/upload-session/chunks",
            &[],
            Some(&payload),
            Some("application/json"),
        )?;
        serde_json::from_str(&text).map_err(map_serde_err)
    }

    fn upload_session_complete(
        &self,
        path: &str,
        session_id: &str,
        blob_key: &str,
        size_bytes: u64,
        content_sha256: &str,
        chunk_count: usize,
        hash: Option<&str>,
    ) -> io::Result<()> {
        let body = UploadSessionCompleteRequest {
            path: path.to_string(),
            session_id: session_id.to_string(),
            blob_key: blob_key.to_string(),
            size_bytes: size_bytes as i64,
            content_sha256: content_sha256.to_string(),
            chunk_count,
            hash: hash.map(ToString::to_string),
        };
        let payload = serde_json::to_vec(&body).map_err(map_serde_err)?;
        let _ = self.send_text(
            "POST",
            "/fs/upload-session/complete",
            &[],
            Some(&payload),
            Some("application/json"),
        )?;
        Ok(())
    }

    fn upload_chunk_target(
        &self,
        method: &str,
        url: &str,
        headers: &HashMap<String, String>,
        data: &[u8],
    ) -> io::Result<()> {
        let xhr = self.prepare_xhr(method, url, XmlHttpRequestResponseType::Text)?;
        for (name, value) in headers {
            xhr.set_request_header(name, value)
                .map_err(|err| map_js_error("XmlHttpRequest::set_request_header", err))?;
        }
        self.dispatch(&xhr, Some(data))?;
        let status = xhr
            .status()
            .map_err(|err| map_js_error("XmlHttpRequest::status", err))?;
        if status < 200 || status >= 300 {
            return Err(self.status_error(&xhr, status));
        }
        Ok(())
    }

    fn normalize(&self, path: &Path) -> String {
        let mut normalized = PathBuf::new();
        normalized.push("/");
        normalized.push(path);
        normalized.to_string_lossy().replace('\\', "/").to_string()
    }

    fn ensure_parent_exists(&self, path: &Path) -> io::Result<()> {
        if let Some(parent) = path.parent() {
            self.create_dir_all(parent)?;
        }
        Ok(())
    }

    fn download_raw_file(&self, path: &str, len: u64) -> io::Result<Vec<u8>> {
        if len == 0 {
            return Ok(Vec::new());
        }
        if should_use_direct_read(len, self.direct_read_threshold_bytes) {
            let url = self.fetch_download_url(path)?.download_url;
            return self.read_range_from_url(&url, 0, len);
        }
        let mut buffer = Vec::with_capacity(len as usize);
        let mut offset = 0;
        while offset < len {
            let remaining = len - offset;
            let length = self.chunk_bytes.min(remaining as usize);
            let chunk = self.download_chunk(path, offset, length)?;
            buffer.extend_from_slice(&chunk);
            offset += chunk.len() as u64;
            if chunk.is_empty() {
                break;
            }
        }
        Ok(buffer)
    }

    fn download_sharded_file(&self, path: &str, len: u64) -> io::Result<Vec<u8>> {
        let manifest_bytes = self.download_raw_file(path, len)?;
        let manifest: ShardManifest = serde_json::from_slice(&manifest_bytes)
            .map_err(|_| io::Error::new(ErrorKind::InvalidData, "invalid shard manifest"))?;
        if manifest.version != 1 {
            return Err(io::Error::new(
                ErrorKind::InvalidData,
                "unsupported manifest",
            ));
        }
        let mut buffer = Vec::with_capacity(manifest.total_size as usize);
        for shard in manifest.shards {
            let meta = self.fetch_metadata(&shard.path)?;
            let bytes = self.download_raw_file(&shard.path, meta.len)?;
            buffer.extend_from_slice(&bytes);
        }
        Ok(buffer)
    }

    fn upload_entire_file(&self, path: &str, data: &[u8]) -> io::Result<()> {
        if data.len() as u64 >= self.shard_threshold_bytes {
            return self.upload_sharded_file(path, data);
        }
        self.upload_unsharded_file(path, data, None)
    }

    fn upload_unsharded_file_legacy(
        &self,
        path: &str,
        data: &[u8],
        hash: Option<&str>,
    ) -> io::Result<()> {
        if data.is_empty() {
            self.upload_chunk(path, 0, true, true, data, hash)?;
            return Ok(());
        }
        let mut offset = 0;
        let mut index = 0;
        while offset < data.len() {
            let end = std::cmp::min(offset + self.chunk_bytes, data.len());
            let chunk = &data[offset..end];
            let truncate = index == 0;
            let final_chunk = end == data.len();
            let session =
                self.upload_chunk(path, offset as u64, truncate, final_chunk, chunk, hash)?;
            if let Some(session) = session {
                let expected = offset as u64 + chunk.len() as u64;
                if session.next_offset as u64 != expected {
                    return Err(io::Error::new(ErrorKind::Other, "unexpected next offset"));
                }
            }
            offset = end;
            index += 1;
        }
        Ok(())
    }

    fn upload_unsharded_file(&self, path: &str, data: &[u8], hash: Option<&str>) -> io::Result<()> {
        if data.is_empty() {
            return self.upload_unsharded_file_legacy(path, data, hash);
        }
        let content_sha256 = sha256_hex(data);
        let session = match self.upload_session_start(path, data.len() as u64, &content_sha256) {
            Ok(session) => session,
            Err(err) if err.kind() == ErrorKind::NotFound => {
                return self.upload_unsharded_file_legacy(path, data, hash);
            }
            Err(err) => return Err(err),
        };
        let chunk_size = (session.chunk_size_bytes as usize).max(1);
        let mut chunks = Vec::new();
        let mut offset = 0usize;
        let mut index = 0usize;
        while offset < data.len() {
            let end = std::cmp::min(offset + chunk_size, data.len());
            let slice = &data[offset..end];
            chunks.push(UploadSessionChunkDescriptor {
                chunk_index: index,
                offset_bytes: offset as i64,
                size_bytes: (end - offset) as i64,
                chunk_sha256: sha256_hex(slice),
            });
            offset = end;
            index += 1;
        }
        let chunk_targets =
            self.upload_session_chunks(&session.session_id, &session.blob_key, chunks.clone())?;
        let targets_by_index: HashMap<usize, UploadChunkTarget> = chunk_targets
            .targets
            .into_iter()
            .map(|target| (target.chunk_index, target))
            .collect();
        for chunk in &chunks {
            let target = targets_by_index.get(&chunk.chunk_index).ok_or_else(|| {
                io::Error::other(format!("missing target for chunk {}", chunk.chunk_index))
            })?;
            let start = chunk.offset_bytes as usize;
            let end = start + chunk.size_bytes as usize;
            self.upload_chunk_target(
                &target.method,
                &target.upload_url,
                &target.headers,
                &data[start..end],
            )?;
        }
        self.upload_session_complete(
            path,
            &session.session_id,
            &session.blob_key,
            data.len() as u64,
            &content_sha256,
            chunks.len(),
            hash,
        )?;
        Ok(())
    }

    fn upload_sharded_file(&self, path: &str, data: &[u8]) -> io::Result<()> {
        let shard_size = self.shard_size_bytes as usize;
        let mut shards = Vec::new();
        let mut offset = 0usize;
        while offset < data.len() {
            let end = std::cmp::min(offset + shard_size, data.len());
            let slice = &data[offset..end];
            let shard_path = format!("{}/{}", SHARD_PREFIX, Uuid::new_v4());
            self.upload_unsharded_file(&shard_path, slice, None)?;
            shards.push(ShardEntry {
                path: shard_path,
                size: slice.len() as u64,
            });
            offset = end;
        }
        let manifest = ShardManifest {
            version: 1,
            total_size: data.len() as u64,
            shard_size: self.shard_size_bytes,
            shards,
        };
        let bytes = serde_json::to_vec(&manifest)
            .map_err(|_| io::Error::new(ErrorKind::InvalidData, "invalid manifest"))?;
        self.upload_unsharded_file(path, &bytes, Some(MANIFEST_HASH))
    }
}

fn should_use_direct_read(length: u64, threshold: u64) -> bool {
    length >= threshold
}

impl FsProvider for RemoteFsProvider {
    fn open(&self, path: &Path, flags: &OpenFlags) -> io::Result<Box<dyn FileHandle>> {
        let mut data = Vec::new();
        let normalized = self.normalize(path);
        let mut exists = false;

        if flags.read || flags.append || (!flags.create && !flags.create_new) {
            match self.fetch_metadata(&normalized) {
                Ok(meta) => {
                    if meta.file_type != "file" {
                        return Err(io::Error::new(
                            ErrorKind::Other,
                            "remote path is not a file",
                        ));
                    }
                    data = self.download_raw_file(&normalized, meta.len)?;
                    exists = true;
                }
                Err(err) if err.kind() == ErrorKind::NotFound => {
                    exists = false;
                }
                Err(err) => return Err(err),
            }
        }

        if flags.create_new && exists {
            return Err(io::Error::new(
                ErrorKind::AlreadyExists,
                format!("File already exists: {}", path.display()),
            ));
        }

        if flags.truncate {
            data.clear();
        }

        if flags.create && !exists {
            exists = true;
        }

        if !exists && !flags.create {
            return Err(io::Error::new(
                ErrorKind::NotFound,
                format!("File not found: {}", path.display()),
            ));
        }

        let cursor = if flags.append { data.len() } else { 0 };

        let handle = RemoteFileHandle {
            provider: self.clone(),
            path: normalized,
            data,
            cursor,
            flags: flags.clone(),
            dirty: false,
        };
        Ok(Box::new(handle))
    }

    fn read(&self, path: &Path) -> io::Result<Vec<u8>> {
        let normalized = self.normalize(path);
        let meta = self.fetch_metadata(&normalized)?;
        if meta.file_type != "file" {
            return Err(io::Error::new(
                ErrorKind::Other,
                "remote path is not a file",
            ));
        }
        self.download_raw_file(&normalized, meta.len)
    }

    fn write(&self, path: &Path, data: &[u8]) -> io::Result<()> {
        let normalized = self.normalize(path);
        self.ensure_parent_exists(path)?;
        self.upload_entire_file(&normalized, data)
    }

    fn remove_file(&self, path: &Path) -> io::Result<()> {
        let normalized = self.normalize(path);
        self.send_empty("DELETE", "/fs/file", &[("path", normalized)])
    }

    fn metadata(&self, path: &Path) -> io::Result<FsMetadata> {
        let normalized = self.normalize(path);
        let resp = self.fetch_metadata(&normalized)?;
        Ok(resp.into())
    }

    fn symlink_metadata(&self, path: &Path) -> io::Result<FsMetadata> {
        self.metadata(path)
    }

    fn read_dir(&self, path: &Path) -> io::Result<Vec<DirEntry>> {
        let normalized = self.normalize(path);
        let resp = self.fetch_dir(&normalized)?;
        Ok(resp
            .into_iter()
            .map(|entry| DirEntry {
                path: PathBuf::from(entry.path),
                file_name: entry.file_name.into(),
                file_type: entry.file_type.into(),
            })
            .collect())
    }

    fn canonicalize(&self, path: &Path) -> io::Result<PathBuf> {
        let normalized = self.normalize(path);
        let canonical = self.fetch_canonical_path(&normalized)?;
        Ok(PathBuf::from(canonical))
    }

    fn create_dir(&self, path: &Path) -> io::Result<()> {
        self.send_json(
            "POST",
            "/fs/mkdir",
            &CreateDirRequest {
                path: self.normalize(path),
                recursive: false,
            },
        )
    }

    fn create_dir_all(&self, path: &Path) -> io::Result<()> {
        self.send_json(
            "POST",
            "/fs/mkdir",
            &CreateDirRequest {
                path: self.normalize(path),
                recursive: true,
            },
        )
    }

    fn remove_dir(&self, path: &Path) -> io::Result<()> {
        self.send_empty(
            "DELETE",
            "/fs/dir",
            &[
                ("path", self.normalize(path)),
                ("recursive", "false".into()),
            ],
        )
    }

    fn remove_dir_all(&self, path: &Path) -> io::Result<()> {
        self.send_empty(
            "DELETE",
            "/fs/dir",
            &[("path", self.normalize(path)), ("recursive", "true".into())],
        )
    }

    fn rename(&self, from: &Path, to: &Path) -> io::Result<()> {
        self.send_json(
            "POST",
            "/fs/rename",
            &RenameRequest {
                from: self.normalize(from),
                to: self.normalize(to),
            },
        )
    }

    fn set_readonly(&self, path: &Path, readonly: bool) -> io::Result<()> {
        self.send_json(
            "POST",
            "/fs/set-readonly",
            &SetReadonlyRequest {
                path: self.normalize(path),
                readonly,
            },
        )
    }
}

impl RemoteFsProvider {
    fn send_json<T: Serialize>(&self, method: &str, route: &str, body: &T) -> io::Result<()> {
        let payload =
            serde_json::to_vec(body).map_err(|err| io::Error::new(ErrorKind::Other, err))?;
        let _ = self.send_text(method, route, &[], Some(&payload), Some("application/json"))?;
        Ok(())
    }
}

impl Clone for RemoteFsProvider {
    fn clone(&self) -> Self {
        Self {
            base: self.base.clone(),
            auth_header: self.auth_header.clone(),
            chunk_bytes: self.chunk_bytes,
            direct_read_threshold_bytes: self.direct_read_threshold_bytes,
            shard_threshold_bytes: self.shard_threshold_bytes,
            shard_size_bytes: self.shard_size_bytes,
            timeout_ms: self.timeout_ms,
        }
    }
}

struct RemoteFileHandle {
    provider: RemoteFsProvider,
    path: String,
    data: Vec<u8>,
    cursor: usize,
    flags: OpenFlags,
    dirty: bool,
}

impl RemoteFileHandle {
    fn flush_remote(&mut self) -> io::Result<()> {
        if !self.dirty {
            return Ok(());
        }
        self.provider.upload_entire_file(&self.path, &self.data)?;
        self.dirty = false;
        Ok(())
    }
}

impl Drop for RemoteFileHandle {
    fn drop(&mut self) {
        if self.dirty {
            let _ = self.provider.upload_entire_file(&self.path, &self.data);
        }
    }
}

impl Read for RemoteFileHandle {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        if !self.flags.read {
            return Err(io::Error::new(
                ErrorKind::PermissionDenied,
                "remote file not opened for reading",
            ));
        }
        let remaining = self.data.len().saturating_sub(self.cursor);
        let to_read = remaining.min(buf.len());
        buf[..to_read].copy_from_slice(&self.data[self.cursor..self.cursor + to_read]);
        self.cursor += to_read;
        Ok(to_read)
    }
}

impl Write for RemoteFileHandle {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        if !self.flags.write && !self.flags.append {
            return Err(io::Error::new(
                ErrorKind::PermissionDenied,
                "remote file not opened for writing",
            ));
        }
        if self.flags.append {
            self.cursor = self.data.len();
        }
        let required = self.cursor + buf.len();
        if required > self.data.len() {
            self.data.resize(required, 0);
        }
        self.data[self.cursor..self.cursor + buf.len()].copy_from_slice(buf);
        self.cursor += buf.len();
        self.dirty = true;
        Ok(buf.len())
    }

    fn flush(&mut self) -> io::Result<()> {
        self.flush_remote()
    }
}

impl Seek for RemoteFileHandle {
    fn seek(&mut self, pos: SeekFrom) -> io::Result<u64> {
        let new_pos = match pos {
            SeekFrom::Start(offset) => offset as i64,
            SeekFrom::End(offset) => self.data.len() as i64 + offset,
            SeekFrom::Current(offset) => self.cursor as i64 + offset,
        };
        if new_pos < 0 {
            return Err(io::Error::new(
                ErrorKind::InvalidInput,
                "seek before start of file",
            ));
        }
        self.cursor = new_pos as usize;
        Ok(self.cursor as u64)
    }
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct MetadataResponse {
    #[serde(rename = "fileType")]
    file_type: String,
    len: u64,
    modified: Option<u64>,
    readonly: bool,
    hash: Option<String>,
}

impl From<MetadataResponse> for FsMetadata {
    fn from(value: MetadataResponse) -> Self {
        FsMetadata::new_with_hash(
            value.file_type.into(),
            value.len,
            value
                .modified
                .map(|secs| UNIX_EPOCH + Duration::from_secs(secs)),
            value.readonly,
            value.hash,
        )
    }
}

#[derive(Debug, Deserialize)]
struct DirEntryResponse {
    path: String,
    #[serde(rename = "fileName")]
    file_name: String,
    #[serde(rename = "fileType")]
    file_type: String,
}

impl From<String> for FsFileType {
    fn from(value: String) -> Self {
        match value.as_str() {
            "dir" => FsFileType::Directory,
            "file" => FsFileType::File,
            "symlink" => FsFileType::Symlink,
            "other" => FsFileType::Other,
            _ => FsFileType::Unknown,
        }
    }
}

#[derive(Debug, Serialize)]
struct CreateDirRequest {
    path: String,
    recursive: bool,
}

#[derive(Debug, Serialize)]
struct RenameRequest {
    from: String,
    to: String,
}

#[derive(Debug, Serialize)]
struct SetReadonlyRequest {
    path: String,
    readonly: bool,
}

#[derive(Debug, Serialize, Deserialize)]
struct ShardManifest {
    version: u32,
    total_size: u64,
    shard_size: u64,
    shards: Vec<ShardEntry>,
}

#[derive(Debug, Serialize, Deserialize)]
struct ShardEntry {
    path: String,
    size: u64,
}

#[derive(Debug, Deserialize)]
struct CanonicalizeResponse {
    path: String,
}

#[derive(Debug, Deserialize)]
struct DownloadUrlResponse {
    #[serde(rename = "downloadUrl")]
    download_url: String,
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
struct UploadSessionStartRequest {
    path: String,
    size_bytes: i64,
    content_type: Option<String>,
    content_sha256: String,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct UploadSessionStartResponse {
    session_id: String,
    blob_key: String,
    chunk_size_bytes: i64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
struct UploadSessionChunkDescriptor {
    chunk_index: usize,
    offset_bytes: i64,
    size_bytes: i64,
    chunk_sha256: String,
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
struct UploadSessionChunksRequest {
    session_id: String,
    blob_key: String,
    chunks: Vec<UploadSessionChunkDescriptor>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct UploadSessionChunksResponse {
    targets: Vec<UploadChunkTarget>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
struct UploadChunkTarget {
    chunk_index: usize,
    method: String,
    upload_url: String,
    headers: HashMap<String, String>,
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
struct UploadSessionCompleteRequest {
    path: String,
    session_id: String,
    blob_key: String,
    size_bytes: i64,
    content_sha256: String,
    chunk_count: usize,
    hash: Option<String>,
}

#[derive(Debug, Deserialize)]
struct FsWriteSessionResponse {
    #[serde(rename = "sessionId")]
    _session_id: String,
    #[serde(rename = "nextOffset")]
    next_offset: i64,
}

fn sha256_hex(data: &[u8]) -> String {
    let digest = Sha256::digest(data);
    let mut out = String::with_capacity(digest.len() * 2);
    for byte in digest {
        use std::fmt::Write as _;
        let _ = write!(out, "{byte:02x}");
    }
    out
}

#[cfg(all(test, target_arch = "wasm32"))]
mod tests {
    use super::*;
    use wasm_bindgen_test::wasm_bindgen_test;
    use wasm_bindgen_test::wasm_bindgen_test_configure;

    wasm_bindgen_test_configure!(run_in_browser);

    #[wasm_bindgen_test]
    fn download_url_parses() {
        let json = r#"{\"downloadUrl\":\"https://example.com/obj\"}"#;
        let parsed: DownloadUrlResponse = serde_json::from_str(json).unwrap();
        assert_eq!(parsed.download_url, "https://example.com/obj");
    }

    #[wasm_bindgen_test]
    fn direct_read_threshold_checks() {
        let threshold = RemoteFsConfig::default().direct_read_threshold_bytes;
        assert!(should_use_direct_read(threshold, threshold));
        assert!(!should_use_direct_read(threshold - 1, threshold));
    }

    #[wasm_bindgen_test]
    fn sha256_hex_matches_known_value() {
        assert_eq!(
            sha256_hex(b"abc"),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
    }

    #[wasm_bindgen_test]
    fn upload_chunk_descriptor_serializes_checksum_fields() {
        let request = UploadSessionChunksRequest {
            session_id: "session-1".to_string(),
            blob_key: "blob-key".to_string(),
            chunks: vec![UploadSessionChunkDescriptor {
                chunk_index: 3,
                offset_bytes: 512,
                size_bytes: 256,
                chunk_sha256: "deadbeef".to_string(),
            }],
        };
        let value = serde_json::to_value(&request).expect("serialize request");
        assert_eq!(value["sessionId"], "session-1");
        assert_eq!(value["blobKey"], "blob-key");
        assert_eq!(value["chunks"][0]["chunkIndex"], 3);
        assert_eq!(value["chunks"][0]["chunkSha256"], "deadbeef");
    }
}

fn map_js_error(op: &str, err: JsValue) -> io::Error {
    let msg = err
        .as_string()
        .or_else(|| err.dyn_ref::<js_sys::Error>().map(|e| e.message().into()))
        .unwrap_or_else(|| format!("{err:?}"));
    io::Error::new(ErrorKind::Other, format!("{op}: {msg}"))
}

fn map_url_err(err: url::ParseError) -> io::Error {
    io::Error::new(ErrorKind::InvalidInput, err)
}

fn map_serde_err(err: serde_json::Error) -> io::Error {
    io::Error::new(ErrorKind::InvalidData, err)
}

use crate::{DirEntry, FileHandle, FsFileType, FsMetadata, FsProvider, OpenFlags};
use crossbeam_utils::thread;
use once_cell::sync::Lazy;
use chrono::DateTime;
use reqwest::blocking::{Client, Response};
use reqwest::StatusCode;
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;
use std::fmt;
use std::io::{self, ErrorKind, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::Duration;
use url::Url;
use uuid::Uuid;

const MANIFEST_HASH: &str = "manifest:v1";
const SHARD_PREFIX: &str = "/.runmat/shards";

static USER_AGENT: Lazy<String> = Lazy::new(|| {
    format!(
        "runmat-remote-fs/{} ({})",
        env!("CARGO_PKG_VERSION"),
        std::env::consts::OS
    )
});

#[derive(Clone, Debug)]
pub struct RemoteFsConfig {
    pub base_url: String,
    pub auth_token: Option<String>,
    pub chunk_bytes: usize,
    pub parallel_requests: usize,
    pub direct_read_threshold_bytes: u64,
    pub direct_write_threshold_bytes: u64,
    pub shard_threshold_bytes: u64,
    pub shard_size_bytes: u64,
    pub timeout: Duration,
    pub retry_max_attempts: usize,
    pub retry_base_delay: Duration,
    pub retry_max_delay: Duration,
}

impl Default for RemoteFsConfig {
    fn default() -> Self {
        Self {
            base_url: String::new(),
            auth_token: None,
            chunk_bytes: 16 * 1024 * 1024,
            parallel_requests: 4,
            direct_read_threshold_bytes: 64 * 1024 * 1024,
            direct_write_threshold_bytes: 64 * 1024 * 1024,
            shard_threshold_bytes: 4 * 1024 * 1024 * 1024,
            shard_size_bytes: 512 * 1024 * 1024,
            timeout: Duration::from_secs(120),
            retry_max_attempts: 5,
            retry_base_delay: Duration::from_millis(100),
            retry_max_delay: Duration::from_secs(2),
        }
    }
}

struct RemoteInner {
    client: Client,
    base: Url,
    auth_header: Option<String>,
    chunk_bytes: usize,
    parallel_requests: usize,
    direct_read_threshold_bytes: u64,
    direct_write_threshold_bytes: u64,
    shard_threshold_bytes: u64,
    shard_size_bytes: u64,
    retry_max_attempts: usize,
    retry_base_delay: Duration,
    retry_max_delay: Duration,
}

impl RemoteInner {
    fn new(config: RemoteFsConfig) -> io::Result<Self> {
        if config.base_url.is_empty() {
            return Err(io::Error::new(
                ErrorKind::InvalidInput,
                "RemoteFsConfig.base_url must be provided",
            ));
        }
        let base = Url::parse(&config.base_url).map_err(map_url_err)?;
        let client = Client::builder()
            .timeout(config.timeout)
            .user_agent(USER_AGENT.clone())
            .build()
            .map_err(map_http_err)?;
        Ok(Self {
            client,
            base,
            auth_header: config.auth_token.map(|token| format!("Bearer {token}")),
            chunk_bytes: config.chunk_bytes.max(64 * 1024),
            parallel_requests: config.parallel_requests.max(1),
            direct_read_threshold_bytes: config.direct_read_threshold_bytes,
            direct_write_threshold_bytes: config.direct_write_threshold_bytes,
            shard_threshold_bytes: config.shard_threshold_bytes,
            shard_size_bytes: config.shard_size_bytes.max(8 * 1024 * 1024),
            retry_max_attempts: config.retry_max_attempts.max(1),
            retry_base_delay: config.retry_base_delay,
            retry_max_delay: config.retry_max_delay,
        })
    }

    fn should_retry(&self, status: StatusCode) -> bool {
        matches!(
            status,
            StatusCode::TOO_MANY_REQUESTS
                | StatusCode::INTERNAL_SERVER_ERROR
                | StatusCode::BAD_GATEWAY
                | StatusCode::SERVICE_UNAVAILABLE
                | StatusCode::GATEWAY_TIMEOUT
        )
    }

    fn retry_delay(&self, attempt: usize) -> Duration {
        let factor = 1u64.checked_shl(attempt as u32).unwrap_or(u64::MAX);
        let delay = self.retry_base_delay.as_millis() as u64 * factor;
        let capped = delay.min(self.retry_max_delay.as_millis() as u64);
        Duration::from_millis(capped)
    }

    fn send_with_retry(
        &self,
        method: reqwest::Method,
        route: &str,
        query: &[(&str, String)],
    ) -> io::Result<Response> {
        for attempt in 0..self.retry_max_attempts {
            let resp = self
                .request(method.clone(), route, query)
                .send()
                .map_err(map_http_err)?;
            if !self.should_retry(resp.status()) || attempt + 1 == self.retry_max_attempts {
                return Ok(resp);
            }
            std::thread::sleep(self.retry_delay(attempt));
        }
        Err(io::Error::new(
            ErrorKind::Other,
            "request retries exhausted",
        ))
    }

    fn get_url_with_retry(&self, url: &str, range: Option<String>) -> io::Result<Response> {
        for attempt in 0..self.retry_max_attempts {
            let mut request = self.client.get(url);
            if let Some(range) = &range {
                request = request.header(reqwest::header::RANGE, range);
            }
            let resp = request.send().map_err(map_http_err)?;
            if !self.should_retry(resp.status()) || attempt + 1 == self.retry_max_attempts {
                return Ok(resp);
            }
            std::thread::sleep(self.retry_delay(attempt));
        }
        Err(io::Error::new(
            ErrorKind::Other,
            "request retries exhausted",
        ))
    }

    fn endpoint(&self, route: &str) -> Url {
        self.base
            .join(route.trim_start_matches('/'))
            .expect("failed to join remote route")
    }

    fn request(
        &self,
        method: reqwest::Method,
        route: &str,
        query: &[(&str, String)],
    ) -> reqwest::blocking::RequestBuilder {
        let mut url = self.endpoint(route);
        {
            let mut pairs = url.query_pairs_mut();
            for (k, v) in query {
                pairs.append_pair(k, v);
            }
        }
        let mut builder = self.client.request(method, url);
        if let Some(auth) = &self.auth_header {
            builder = builder.header("Authorization", auth);
        }
        builder
    }

    fn get_json<T: for<'a> Deserialize<'a>>(
        &self,
        route: &str,
        query: &[(&str, String)],
    ) -> io::Result<T> {
        let resp = self
            .send_with_retry(reqwest::Method::GET, route, query)?;
        handle_error(resp)?.json().map_err(map_http_err)
    }

    fn post_empty<B: Serialize>(&self, route: &str, body: &B) -> io::Result<()> {
        let resp = self
            .request(reqwest::Method::POST, route, &[])
            .json(body)
            .send()
            .map_err(map_http_err)?;
        handle_error(resp)?;
        Ok(())
    }

    fn post_json<T: for<'a> Deserialize<'a>, B: Serialize>(
        &self,
        route: &str,
        body: &B,
    ) -> io::Result<T> {
        let resp = self
            .request(reqwest::Method::POST, route, &[])
            .json(body)
            .send()
            .map_err(map_http_err)?;
        handle_error(resp)?.json().map_err(map_http_err)
    }

    fn delete_empty(&self, route: &str, query: &[(&str, String)]) -> io::Result<()> {
        let resp = self
            .send_with_retry(reqwest::Method::DELETE, route, query)?;
        handle_error(resp)?;
        Ok(())
    }

    fn download_chunk(&self, path: &str, offset: u64, length: usize) -> io::Result<Vec<u8>> {
        let resp = self.send_with_retry(
            reqwest::Method::GET,
            "/fs/read",
            &[
                ("path", path.to_string()),
                ("offset", offset.to_string()),
                ("length", length.to_string()),
            ],
        )?;
        let mut body = handle_error(resp)?;
        if let Some(content_type) = body.headers().get(reqwest::header::CONTENT_TYPE) {
            if content_type
                .to_str()
                .map(|value| value.contains("application/json"))
                .unwrap_or(false)
            {
                let json: DownloadUrlResponse = body.json().map_err(map_http_err)?;
                return self.download_range_from_url(&json.download_url, offset, length as u64);
            }
        }
        let mut buf = Vec::with_capacity(length);
        body.copy_to(&mut buf).map_err(map_http_err)?;
        Ok(buf)
    }

    fn download_range_from_url(
        &self,
        url: &str,
        offset: u64,
        length: u64,
    ) -> io::Result<Vec<u8>> {
        if length == 0 {
            return Ok(Vec::new());
        }
        let end = offset + length - 1;
        let resp = self.get_url_with_retry(url, Some(format!("bytes={offset}-{end}")))?;
        let mut body = handle_error(resp)?;
        let mut buf = Vec::with_capacity(length as usize);
        body.copy_to(&mut buf).map_err(map_http_err)?;
        Ok(buf)
    }

    fn fetch_download_url(&self, path: &str) -> io::Result<DownloadUrlResponse> {
        self.get_json("/fs/download-url", &[("path", path.to_string())])
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
            query.push(("truncate", "true".to_string()));
        }
        if final_chunk {
            query.push(("final", "true".to_string()));
        }
        if let Some(hash) = hash {
            query.push(("hash", hash.to_string()));
        }
        let resp = self
            .request(reqwest::Method::PUT, "/fs/write", &query)
            .body(data.to_vec())
            .send()
            .map_err(map_http_err)?;
        if resp.status() == StatusCode::ACCEPTED {
            let session = handle_error(resp)?.json().map_err(map_http_err)?;
            return Ok(Some(session));
        }
        handle_error(resp)?;
        Ok(None)
    }

    fn multipart_create(&self, path: &str, size_bytes: u64) -> io::Result<MultipartUploadResponse> {
        self.post_json(
            "/fs/multipart-upload",
            &MultipartUploadRequest {
                path: path.to_string(),
                size_bytes: size_bytes as i64,
                content_type: None,
            },
        )
    }

    fn multipart_presign_part(
        &self,
        session_id: &str,
        blob_key: &str,
        upload_id: &str,
        part_number: i32,
        size_bytes: u64,
    ) -> io::Result<String> {
        let response: MultipartUploadPartResponse = self.post_json(
            "/fs/multipart-upload/part",
            &MultipartUploadPartRequest {
                session_id: session_id.to_string(),
                blob_key: blob_key.to_string(),
                upload_id: upload_id.to_string(),
                part_number,
                size_bytes: size_bytes as i64,
            },
        )?;
        Ok(response.upload_url)
    }

    fn multipart_complete(
        &self,
        path: &str,
        session_id: &str,
        blob_key: &str,
        upload_id: &str,
        size_bytes: u64,
        parts: Vec<MultipartPart>,
    ) -> io::Result<()> {
        let resp = self
            .request(reqwest::Method::POST, "/fs/multipart-upload/complete", &[])
            .json(&MultipartUploadCompleteRequest {
                path: path.to_string(),
                session_id: session_id.to_string(),
                blob_key: blob_key.to_string(),
                upload_id: upload_id.to_string(),
                size_bytes: size_bytes as i64,
                hash: None,
                parts,
            })
            .send()
            .map_err(map_http_err)?;
        handle_error(resp)?;
        Ok(())
    }

    fn put_upload_url_with_etag(&self, url: &str, data: &[u8]) -> io::Result<String> {
        let resp = self
            .client
            .put(url)
            .body(data.to_vec())
            .send()
            .map_err(map_http_err)?;
        let resp = handle_error(resp)?;
        let etag = resp
            .headers()
            .get(reqwest::header::ETAG)
            .and_then(|value| value.to_str().ok())
            .unwrap_or("")
            .to_string();
        Ok(etag)
    }

    fn fetch_metadata(&self, path: &str) -> io::Result<MetadataResponse> {
        self.get_json("/fs/metadata", &[("path", path.to_string())])
    }

    fn fetch_dir(&self, path: &str) -> io::Result<Vec<DirEntryResponse>> {
        self.get_json("/fs/dir", &[("path", path.to_string())])
    }

    fn canonicalize_path(&self, path: &str) -> io::Result<String> {
        let resp: CanonicalizeResponse =
            self.get_json("/fs/canonicalize", &[("path", path.to_string())])?;
        Ok(resp.path)
    }
}

pub struct RemoteFsProvider {
    inner: Arc<RemoteInner>,
}

impl RemoteFsProvider {
    pub fn new(config: RemoteFsConfig) -> io::Result<Self> {
        Ok(Self {
            inner: Arc::new(RemoteInner::new(config)?),
        })
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
        if len >= self.inner.direct_read_threshold_bytes {
            let url = self.inner.fetch_download_url(path)?.download_url;
            return self.download_entire_file_from_url(&url, len);
        }
        let chunk = self.inner.chunk_bytes as u64;
        let mut tasks = Vec::new();
        let mut offset = 0;
        let mut index = 0;
        while offset < len {
            let remaining = len - offset;
            let length = std::cmp::min(chunk, remaining);
            tasks.push(ChunkTask {
                offset,
                length: length as usize,
                index,
            });
            offset += length;
            index += 1;
        }
        let mut buffer = vec![0u8; len as usize];
        let path = path.to_string();
        let inner = self.inner.clone();
        let queue = Arc::new(Mutex::new(VecDeque::from(tasks.clone())));
        let results = Arc::new(Mutex::new(vec![None::<Vec<u8>>; tasks.len()]));
        let error: Arc<Mutex<Option<io::Error>>> = Arc::new(Mutex::new(None));
        let threads = inner
            .parallel_requests
            .min(queue.lock().unwrap().len())
            .max(1);
        thread::scope(|scope| {
            for _ in 0..threads {
                let queue = queue.clone();
                let error = error.clone();
                let results = results.clone();
                let inner = inner.clone();
                let path = path.clone();
                scope.spawn(move |_| loop {
                    let task_opt = {
                        let mut guard = queue.lock().unwrap();
                        guard.pop_front()
                    };
                    let task = match task_opt {
                        Some(task) => task,
                        None => break,
                    };
                    match inner.download_chunk(&path, task.offset, task.length) {
                        Ok(bytes) => {
                            let mut guard = results.lock().unwrap();
                            guard[task.index] = Some(bytes);
                        }
                        Err(err) => {
                            let mut guard = error.lock().unwrap();
                            if guard.is_none() {
                                *guard = Some(err);
                            }
                            break;
                        }
                    }
                });
            }
        })
        .expect("remote download scope");
        if let Some(err) = error.lock().unwrap().take() {
            return Err(err);
        }
        let chunks = Arc::try_unwrap(results)
            .expect("results still in use")
            .into_inner()
            .expect("results poisoned");
        for (task, maybe) in tasks.iter().zip(chunks.into_iter()) {
            let bytes = maybe.expect("missing downloaded chunk for remote file");
            let start = task.offset as usize;
            buffer[start..start + bytes.len()].copy_from_slice(&bytes);
        }
        Ok(buffer)
    }

    fn download_sharded_file(&self, path: &str, len: u64) -> io::Result<Vec<u8>> {
        let manifest_bytes = self.download_raw_file(path, len)?;
        let manifest: ShardManifest = serde_json::from_slice(&manifest_bytes)
            .map_err(|_| io::Error::new(ErrorKind::InvalidData, "invalid shard manifest"))?;
        if manifest.version != 1 {
            return Err(io::Error::new(ErrorKind::InvalidData, "unsupported manifest"));
        }
        let mut buffer = Vec::with_capacity(manifest.total_size as usize);
        for shard in manifest.shards {
            let meta = self.inner.fetch_metadata(&shard.path)?;
            let bytes = self.download_raw_file(&shard.path, meta.len)?;
            buffer.extend_from_slice(&bytes);
        }
        Ok(buffer)
    }

    fn download_entire_file_from_url(&self, url: &str, len: u64) -> io::Result<Vec<u8>> {
        let chunk = self.inner.chunk_bytes as u64;
        let mut tasks = Vec::new();
        let mut offset = 0;
        let mut index = 0;
        while offset < len {
            let remaining = len - offset;
            let length = std::cmp::min(chunk, remaining);
            tasks.push(ChunkTask {
                offset,
                length: length as usize,
                index,
            });
            offset += length;
            index += 1;
        }
        let mut buffer = vec![0u8; len as usize];
        let url = url.to_string();
        let inner = self.inner.clone();
        let queue = Arc::new(Mutex::new(VecDeque::from(tasks.clone())));
        let results = Arc::new(Mutex::new(vec![None::<Vec<u8>>; tasks.len()]));
        let error: Arc<Mutex<Option<io::Error>>> = Arc::new(Mutex::new(None));
        let threads = inner
            .parallel_requests
            .min(queue.lock().unwrap().len())
            .max(1);
        thread::scope(|scope| {
            for _ in 0..threads {
                let queue = queue.clone();
                let error = error.clone();
                let results = results.clone();
                let inner = inner.clone();
                let url = url.clone();
                scope.spawn(move |_| loop {
                    let task_opt = {
                        let mut guard = queue.lock().unwrap();
                        guard.pop_front()
                    };
                    let task = match task_opt {
                        Some(task) => task,
                        None => break,
                    };
                    match inner.download_range_from_url(&url, task.offset, task.length as u64) {
                        Ok(bytes) => {
                            let mut guard = results.lock().unwrap();
                            guard[task.index] = Some(bytes);
                        }
                        Err(err) => {
                            let mut guard = error.lock().unwrap();
                            if guard.is_none() {
                                *guard = Some(err);
                            }
                            break;
                        }
                    }
                });
            }
        })
        .expect("remote download scope");
        if let Some(err) = error.lock().unwrap().take() {
            return Err(err);
        }
        let chunks = Arc::try_unwrap(results)
            .expect("results still in use")
            .into_inner()
            .expect("results poisoned");
        for (task, maybe) in tasks.iter().zip(chunks.into_iter()) {
            let bytes = maybe.expect("missing downloaded chunk for remote file");
            let start = task.offset as usize;
            buffer[start..start + bytes.len()].copy_from_slice(&bytes);
        }
        Ok(buffer)
    }

    fn upload_entire_file(&self, path: &str, data: &[u8]) -> io::Result<()> {
        if data.len() as u64 >= self.inner.shard_threshold_bytes {
            return self.upload_sharded_file(path, data);
        }
        self.upload_unsharded_file(path, data, None)
    }

    fn upload_unsharded_file(
        &self,
        path: &str,
        data: &[u8],
        hash: Option<&str>,
    ) -> io::Result<()> {
        if data.len() as u64 >= self.inner.direct_write_threshold_bytes {
            return self.upload_multipart_file(path, data);
        }
        if data.is_empty() {
            self.inner.upload_chunk(path, 0, true, true, data, hash)?;
            return Ok(());
        }
        let chunk = self.inner.chunk_bytes;
        let mut offset = 0usize;
        while offset < data.len() {
            let end = std::cmp::min(offset + chunk, data.len());
            let slice = &data[offset..end];
            let truncate = offset == 0;
            let final_chunk = end == data.len();
            let result = self.inner.upload_chunk(
                path,
                offset as u64,
                truncate,
                final_chunk,
                slice,
                hash,
            )?;
            if let Some(session) = result {
                let expected = offset as u64 + slice.len() as u64;
                if session.next_offset as u64 != expected {
                    return Err(io::Error::new(
                        ErrorKind::Other,
                        "unexpected next offset",
                    ));
                }
            }
            offset = end;
        }
        Ok(())
    }

    fn upload_sharded_file(&self, path: &str, data: &[u8]) -> io::Result<()> {
        let shard_size = self.inner.shard_size_bytes as usize;
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
            shard_size: self.inner.shard_size_bytes,
            shards,
        };
        let bytes = serde_json::to_vec(&manifest)
            .map_err(|_| io::Error::new(ErrorKind::InvalidData, "invalid manifest"))?;
        self.upload_unsharded_file(path, &bytes, Some(MANIFEST_HASH))
    }

    fn upload_multipart_file(&self, path: &str, data: &[u8]) -> io::Result<()> {
        if data.is_empty() {
            self.inner.upload_chunk(path, 0, true, true, data, None)?;
            return Ok(());
        }
        let session = self.inner.multipart_create(path, data.len() as u64)?;
        let part_size = session.part_size_bytes as usize;
        let mut tasks = std::collections::VecDeque::new();
        let mut offset = 0usize;
        let mut part_number = 1;
        let mut index = 0usize;
        while offset < data.len() {
            let end = std::cmp::min(offset + part_size, data.len());
            let length = end - offset;
            tasks.push_back(MultipartTask {
                index,
                part_number,
                offset,
                length,
            });
            offset = end;
            part_number += 1;
            index += 1;
        }
        let tasks = Arc::new(Mutex::new(tasks));
        let task_len = tasks.lock().unwrap().len();
        let mut result_vec: Vec<MultipartResult> = Vec::with_capacity(task_len);
        for _ in 0..task_len {
            result_vec.push(None);
        }
        let results = Arc::new(Mutex::new(result_vec));
        let error = Arc::new(Mutex::new(None));
        let data = Arc::new(data.to_vec());
        thread::scope(|scope| {
            for _ in 0..self.inner.parallel_requests {
                let tasks = Arc::clone(&tasks);
                let results = Arc::clone(&results);
                let error = Arc::clone(&error);
                let inner = Arc::clone(&self.inner);
                let blob_key = session.blob_key.clone();
                let upload_id = session.upload_id.clone();
                let session_id = session.session_id.clone();
                let data = Arc::clone(&data);
                scope.spawn(move |_| loop {
                    let task = {
                        let mut guard = tasks.lock().unwrap();
                        guard.pop_front()
                    };
                    let Some(task) = task else { break };
                    let slice = &data[task.offset..task.offset + task.length];
                    let result = (|| {
                        let url = inner.multipart_presign_part(
                            &session_id,
                            &blob_key,
                            &upload_id,
                            task.part_number,
                            task.length as u64,
                        )?;
                        let etag = inner.put_upload_url_with_etag(&url, slice)?;
                        Ok(MultipartPart {
                            part_number: task.part_number,
                            etag,
                        })
                    })();
                    let mut guard = results.lock().unwrap();
                    guard[task.index] = Some(result);
                    if let Some(Err(err)) = guard[task.index].as_ref() {
                        let mut err_guard = error.lock().unwrap();
                        if err_guard.is_none() {
                            *err_guard = Some(io::Error::new(err.kind(), err.to_string()));
                        }
                        break;
                    }
                });
            }
        })
        .expect("multipart upload scope");
        if let Some(err) = error.lock().unwrap().take() {
            return Err(err);
        }
        let mut parts = Vec::with_capacity(results.lock().unwrap().len());
        for maybe in Arc::try_unwrap(results)
            .expect("results still in use")
            .into_inner()
            .expect("results poisoned")
        {
            let part = maybe.expect("missing part result")?;
            if part.etag.is_empty() {
                return Err(io::Error::other("missing etag"));
            }
            parts.push(part);
        }
        parts.sort_by_key(|part| part.part_number);
        self.inner.multipart_complete(
            path,
            &session.session_id,
            &session.blob_key,
            &session.upload_id,
            data.len() as u64,
            parts,
        )?;
        Ok(())
    }
}

#[derive(Clone, Copy)]
struct ChunkTask {
    offset: u64,
    length: usize,
    index: usize,
}

type MultipartResult = Option<Result<MultipartPart, io::Error>>;

#[derive(Clone, Copy)]
struct MultipartTask {
    index: usize,
    part_number: i32,
    offset: usize,
    length: usize,
}

impl FsProvider for RemoteFsProvider {
    fn open(&self, path: &Path, flags: &OpenFlags) -> io::Result<Box<dyn FileHandle>> {
        let normalized = self.normalize(path);
        let mut data = Vec::new();
        if flags.read {
            let meta = self.inner.fetch_metadata(&normalized)?;
            if meta.file_type != "file" {
                return Err(io::Error::other("remote path is not a file"));
            }
            data = if meta.hash.as_deref() == Some(MANIFEST_HASH) {
                self.download_sharded_file(&normalized, meta.len)?
            } else {
                self.download_raw_file(&normalized, meta.len)?
            };
        }
        if flags.truncate {
            data.clear();
        }
        if flags.create {
            self.ensure_parent_exists(path)?;
        }
        let handle = RemoteFileHandle {
            provider: self.clone(),
            path: normalized,
            data,
            position: 0,
            flags: flags.clone(),
            dirty: false,
        };
        Ok(Box::new(handle))
    }

    fn read(&self, path: &Path) -> io::Result<Vec<u8>> {
        let normalized = self.normalize(path);
        let meta = self.inner.fetch_metadata(&normalized)?;
        if meta.file_type != "file" {
            return Err(io::Error::other("remote path is not a file"));
        }
        if meta.hash.as_deref() == Some(MANIFEST_HASH) {
            self.download_sharded_file(&normalized, meta.len)
        } else {
            self.download_raw_file(&normalized, meta.len)
        }
    }

    fn write(&self, path: &Path, data: &[u8]) -> io::Result<()> {
        let normalized = self.normalize(path);
        self.ensure_parent_exists(path)?;
        self.upload_entire_file(&normalized, data)
    }

    fn remove_file(&self, path: &Path) -> io::Result<()> {
        let normalized = self.normalize(path);
        self.inner
            .delete_empty("/fs/file", &[("path", normalized)])?;
        Ok(())
    }

    fn metadata(&self, path: &Path) -> io::Result<FsMetadata> {
        let normalized = self.normalize(path);
        let resp = self.inner.fetch_metadata(&normalized)?;
        Ok(resp.into())
    }

    fn symlink_metadata(&self, path: &Path) -> io::Result<FsMetadata> {
        self.metadata(path)
    }

    fn read_dir(&self, path: &Path) -> io::Result<Vec<DirEntry>> {
        let normalized = self.normalize(path);
        let resp = self.inner.fetch_dir(&normalized)?;
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
        let canonical = self.inner.canonicalize_path(&normalized)?;
        Ok(PathBuf::from(canonical))
    }

    fn create_dir(&self, path: &Path) -> io::Result<()> {
        let normalized = self.normalize(path);
        self.inner.post_empty(
            "/fs/mkdir",
            &CreateDirRequest {
                path: normalized,
                recursive: false,
            },
        )
    }

    fn create_dir_all(&self, path: &Path) -> io::Result<()> {
        let normalized = self.normalize(path);
        self.inner.post_empty(
            "/fs/mkdir",
            &CreateDirRequest {
                path: normalized,
                recursive: true,
            },
        )
    }

    fn remove_dir(&self, path: &Path) -> io::Result<()> {
        let normalized = self.normalize(path);
        self.inner.delete_empty(
            "/fs/dir",
            &[("path", normalized), ("recursive", "false".into())],
        )
    }

    fn remove_dir_all(&self, path: &Path) -> io::Result<()> {
        let normalized = self.normalize(path);
        self.inner.delete_empty(
            "/fs/dir",
            &[("path", normalized), ("recursive", "true".into())],
        )
    }

    fn rename(&self, from: &Path, to: &Path) -> io::Result<()> {
        self.inner.post_empty(
            "/fs/rename",
            &RenameRequest {
                from: self.normalize(from),
                to: self.normalize(to),
            },
        )
    }

    fn set_readonly(&self, path: &Path, readonly: bool) -> io::Result<()> {
        self.inner.post_empty(
            "/fs/set-readonly",
            &SetReadonlyRequest {
                path: self.normalize(path),
                readonly,
            },
        )
    }
}

impl Clone for RemoteFsProvider {
    fn clone(&self) -> Self {
        Self {
            inner: Arc::clone(&self.inner),
        }
    }
}

struct RemoteFileHandle {
    provider: RemoteFsProvider,
    path: String,
    data: Vec<u8>,
    position: usize,
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

impl Read for RemoteFileHandle {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        let remaining = &self.data[self.position..];
        let amt = remaining.len().min(buf.len());
        buf[..amt].copy_from_slice(&remaining[..amt]);
        self.position += amt;
        Ok(amt)
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
            self.position = self.data.len();
        }
        let required = self.position + buf.len();
        if required > self.data.len() {
            self.data.resize(required, 0);
        }
        self.data[self.position..self.position + buf.len()].copy_from_slice(buf);
        self.position += buf.len();
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
            SeekFrom::Current(offset) => self.position as i64 + offset,
        };
        if new_pos < 0 {
            return Err(io::Error::new(ErrorKind::InvalidInput, "seek before start"));
        }
        self.position = new_pos as usize;
        Ok(self.position as u64)
    }
}

impl Drop for RemoteFileHandle {
    fn drop(&mut self) {
        if self.dirty {
            if let Err(err) = self.provider.upload_entire_file(&self.path, &self.data) {
                eprintln!("remote fs flush failed: {err}");
            }
        }
    }
}

#[derive(Debug, Deserialize)]
struct MetadataResponse {
    #[serde(rename = "fileType")]
    file_type: String,
    len: u64,
    #[serde(rename = "modifiedAt")]
    modified_at: Option<String>,
    readonly: bool,
    hash: Option<String>,
}

impl From<MetadataResponse> for FsMetadata {
    fn from(value: MetadataResponse) -> Self {
        FsMetadata::new_with_hash(
            value.file_type.into(),
            value.len,
            parse_modified_at(value.modified_at.as_deref()),
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

#[derive(Debug, Serialize, Deserialize)]
struct CreateDirRequest {
    path: String,
    recursive: bool,
}

#[derive(Debug, Serialize, Deserialize)]
struct RenameRequest {
    from: String,
    to: String,
}

#[derive(Debug, Serialize, Deserialize)]
struct SetReadonlyRequest {
    path: String,
    readonly: bool,
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
struct MultipartUploadRequest {
    path: String,
    #[serde(rename = "sizeBytes")]
    size_bytes: i64,
    #[serde(rename = "contentType")]
    content_type: Option<String>,
}

#[derive(Debug, Deserialize)]
struct MultipartUploadResponse {
    #[serde(rename = "sessionId")]
    session_id: String,
    #[serde(rename = "blobKey")]
    blob_key: String,
    #[serde(rename = "uploadId")]
    upload_id: String,
    #[serde(rename = "partSizeBytes")]
    part_size_bytes: i64,
}

#[derive(Debug, Serialize, Deserialize)]
struct MultipartUploadPartRequest {
    #[serde(rename = "sessionId")]
    session_id: String,
    #[serde(rename = "blobKey")]
    blob_key: String,
    #[serde(rename = "uploadId")]
    upload_id: String,
    #[serde(rename = "partNumber")]
    part_number: i32,
    #[serde(rename = "sizeBytes")]
    size_bytes: i64,
}

#[derive(Debug, Deserialize)]
struct MultipartUploadPartResponse {
    #[serde(rename = "upload_url")]
    upload_url: String,
}

#[derive(Debug, Serialize, Deserialize)]
struct MultipartUploadCompleteRequest {
    path: String,
    #[serde(rename = "sessionId")]
    session_id: String,
    #[serde(rename = "blobKey")]
    blob_key: String,
    #[serde(rename = "uploadId")]
    upload_id: String,
    #[serde(rename = "sizeBytes")]
    size_bytes: i64,
    hash: Option<String>,
    parts: Vec<MultipartPart>,
}

#[derive(Debug, Serialize, Deserialize)]
struct MultipartPart {
    #[serde(rename = "partNumber")]
    part_number: i32,
    etag: String,
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
struct FsWriteSessionResponse {
    #[serde(rename = "sessionId")]
    _session_id: String,
    #[serde(rename = "nextOffset")]
    next_offset: i64,
}

fn map_http_err(err: reqwest::Error) -> io::Error {
    io::Error::other(err)
}

fn parse_modified_at(value: Option<&str>) -> Option<std::time::SystemTime> {
    let value = value?;
    let parsed = DateTime::parse_from_rfc3339(value).ok()?;
    let millis = parsed.timestamp_millis();
    if millis < 0 {
        return None;
    }
    Some(std::time::UNIX_EPOCH + Duration::from_millis(millis as u64))
}

fn map_url_err(err: url::ParseError) -> io::Error {
    io::Error::new(ErrorKind::InvalidInput, err)
}

fn handle_error(resp: Response) -> io::Result<Response> {
    let status = resp.status();
    if status.is_success() {
        return Ok(resp);
    }
    let text = resp.text().unwrap_or_else(|_| status.to_string());
    let kind = match status {
        StatusCode::NOT_FOUND => ErrorKind::NotFound,
        StatusCode::FORBIDDEN | StatusCode::UNAUTHORIZED => ErrorKind::PermissionDenied,
        StatusCode::CONFLICT => ErrorKind::AlreadyExists,
        StatusCode::BAD_REQUEST => ErrorKind::InvalidInput,
        _ => ErrorKind::Other,
    };
    Err(io::Error::new(kind, text))
}

impl fmt::Debug for RemoteFsProvider {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("RemoteFsProvider").finish_non_exhaustive()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::extract::{Query, State};
    use axum::http::{HeaderMap, StatusCode};
    use axum::routing::{delete, get, post, put};
    use axum::{Json, Router};
    use serde::Deserialize;
    use std::net::TcpListener as StdTcpListener;
    use std::sync::Arc;
    use tempfile::tempdir;
    use tokio::net::TcpListener as TokioTcpListener;
    use tokio::runtime::Runtime;

    #[derive(Clone)]
    struct Harness {
        root: Arc<PathBuf>,
        _keeper: Arc<tempfile::TempDir>,
        base_url: Arc<Mutex<Option<String>>>,
    }

    impl Harness {
        fn new() -> Self {
            let dir = tempdir().expect("tempdir");
            let path = dir.path().to_path_buf();
            Self {
                root: Arc::new(path),
                _keeper: Arc::new(dir),
                base_url: Arc::new(Mutex::new(None)),
            }
        }

        fn resolve(&self, remote_path: &str) -> PathBuf {
            let trimmed = remote_path.trim_start_matches('/');
            self.root.join(trimmed)
        }
    }

    #[derive(Deserialize)]
    struct PathParams {
        path: String,
        offset: Option<u64>,
        length: Option<usize>,
        truncate: Option<String>,
        recursive: Option<String>,
    }

    async fn metadata_handler(
        State(harness): State<Harness>,
        Query(params): Query<PathParams>,
    ) -> Result<Json<serde_json::Value>, StatusCode> {
        let meta =
            std::fs::metadata(harness.resolve(&params.path)).map_err(|_| StatusCode::NOT_FOUND)?;
        let file_type = if meta.is_dir() {
            "dir"
        } else if meta.is_file() {
            "file"
        } else {
            "other"
        };
        Ok(Json(serde_json::json!({
            "fileType": file_type,
            "len": meta.len(),
            "modified": meta.modified().ok().and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok()).map(|d| d.as_secs()),
            "readonly": meta.permissions().readonly()
        })))
    }

    async fn dir_handler(
        State(harness): State<Harness>,
        Query(params): Query<PathParams>,
    ) -> Result<Json<Vec<serde_json::Value>>, StatusCode> {
        let mut entries = Vec::new();
        for entry in
            std::fs::read_dir(harness.resolve(&params.path)).map_err(|_| StatusCode::NOT_FOUND)?
        {
            let entry = entry.map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;
            let file_type = entry
                .file_type()
                .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;
            let kind = if file_type.is_dir() {
                "dir"
            } else if file_type.is_file() {
                "file"
            } else {
                "other"
            };
            entries.push(serde_json::json!({
                "path": format!("/{}", entry.path().strip_prefix(&*harness.root).unwrap().display()),
                "fileName": entry.file_name().to_string_lossy(),
                "fileType": kind
            }));
        }
        Ok(Json(entries))
    }

    async fn canonicalize_handler(
        State(harness): State<Harness>,
        Query(params): Query<PathParams>,
    ) -> Result<Json<serde_json::Value>, StatusCode> {
        let path = harness.resolve(&params.path);
        let canonical = std::fs::canonicalize(path).map_err(|_| StatusCode::NOT_FOUND)?;
        let rel = canonical.strip_prefix(&*harness.root).unwrap_or(&canonical);
        Ok(Json(serde_json::json!({
            "path": format!("/{}", rel.display())
        })))
    }

    async fn read_handler(
        State(harness): State<Harness>,
        Query(params): Query<PathParams>,
    ) -> Result<Vec<u8>, StatusCode> {
        let mut data =
            std::fs::read(harness.resolve(&params.path)).map_err(|_| StatusCode::NOT_FOUND)?;
        let offset = params.offset.unwrap_or(0) as usize;
        let length = params.length.unwrap_or(data.len().saturating_sub(offset));
        let end = std::cmp::min(offset + length, data.len());
        if offset < data.len() {
            data = data[offset..end].to_vec();
        } else {
            data.clear();
        }
        Ok(data)
    }

    async fn write_handler(
        State(harness): State<Harness>,
        Query(params): Query<PathParams>,
        body: axum::body::Bytes,
    ) -> Result<(), StatusCode> {
        let path = harness.resolve(&params.path);
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;
        }
        let mut data = if params.truncate.as_deref() == Some("true") || !path.exists() {
            Vec::new()
        } else {
            std::fs::read(&path).map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?
        };
        let offset = params.offset.unwrap_or(0) as usize;
        let required = offset + body.len();
        if required > data.len() {
            data.resize(required, 0);
        }
        data[offset..offset + body.len()].copy_from_slice(&body);
        std::fs::write(path, data).map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;
        Ok(())
    }

    async fn mkdir_handler(
        State(harness): State<Harness>,
        Json(req): Json<CreateDirRequest>,
    ) -> Result<(), StatusCode> {
        let path = harness.resolve(&req.path);
        if req.recursive {
            std::fs::create_dir_all(path).map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;
        } else {
            std::fs::create_dir(path).map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;
        }
        Ok(())
    }

    async fn delete_file_handler(
        State(harness): State<Harness>,
        Query(params): Query<PathParams>,
    ) -> Result<(), StatusCode> {
        std::fs::remove_file(harness.resolve(&params.path)).map_err(|_| StatusCode::NOT_FOUND)?;
        Ok(())
    }

    async fn delete_dir_handler(
        State(harness): State<Harness>,
        Query(params): Query<PathParams>,
    ) -> Result<(), StatusCode> {
        let recursive = params.recursive.as_deref() == Some("true");
        if recursive {
            std::fs::remove_dir_all(harness.resolve(&params.path))
                .map_err(|_| StatusCode::NOT_FOUND)?;
        } else {
            std::fs::remove_dir(harness.resolve(&params.path))
                .map_err(|_| StatusCode::NOT_FOUND)?;
        }
        Ok(())
    }

    async fn rename_handler(
        State(harness): State<Harness>,
        Json(req): Json<RenameRequest>,
    ) -> Result<(), StatusCode> {
        let from = harness.resolve(&req.from);
        let to = harness.resolve(&req.to);
        if let Some(parent) = to.parent() {
            std::fs::create_dir_all(parent).map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;
        }
        std::fs::rename(from, to).map_err(|_| StatusCode::NOT_FOUND)?;
        Ok(())
    }

    async fn set_readonly_handler(
        State(harness): State<Harness>,
        Json(req): Json<SetReadonlyRequest>,
    ) -> Result<(), StatusCode> {
        let path = harness.resolve(&req.path);
        let mut perms = std::fs::metadata(&path)
            .map_err(|_| StatusCode::NOT_FOUND)?
            .permissions();
        perms.set_readonly(req.readonly);
        std::fs::set_permissions(path, perms).map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;
        Ok(())
    }

    async fn read_with_download_handler(
        State(harness): State<Harness>,
        Query(params): Query<PathParams>,
    ) -> Result<Json<serde_json::Value>, StatusCode> {
        let base = harness
            .base_url
            .lock()
            .unwrap()
            .clone()
            .ok_or(StatusCode::INTERNAL_SERVER_ERROR)?;
        let download_url = format!("{base}/download?path={}", params.path);
        Ok(Json(serde_json::json!({
            "downloadUrl": download_url,
            "expiresAt": 0
        })))
    }

    async fn download_handler(
        State(harness): State<Harness>,
        Query(params): Query<PathParams>,
        headers: HeaderMap,
    ) -> Result<Vec<u8>, StatusCode> {
        let mut data =
            std::fs::read(harness.resolve(&params.path)).map_err(|_| StatusCode::NOT_FOUND)?;
        if let Some(range) = headers.get("range").and_then(|value| value.to_str().ok()) {
            if let Some((start, end)) = parse_range(range) {
                let start = start.min(data.len());
                let end = end.min(data.len().saturating_sub(1));
                if start < data.len() {
                    data = data[start..=end].to_vec();
                } else {
                    data.clear();
                }
            }
        }
        Ok(data)
    }

    fn parse_range(value: &str) -> Option<(usize, usize)> {
        let value = value.strip_prefix("bytes=")?;
        let (start, end) = value.split_once('-')?;
        let start = start.parse::<usize>().ok()?;
        let end = end.parse::<usize>().ok()?;
        Some((start, end))
    }

    fn spawn_server() -> (String, Harness, Runtime) {
        let harness = Harness::new();
        let router = Router::new()
            .route("/fs/metadata", get(metadata_handler))
            .route("/fs/dir", get(dir_handler).delete(delete_dir_handler))
            .route("/fs/file", delete(delete_file_handler))
            .route("/fs/canonicalize", get(canonicalize_handler))
            .route("/fs/read", get(read_handler))
            .route("/fs/write", put(write_handler))
            .route("/fs/mkdir", post(mkdir_handler))
            .route("/fs/rename", post(rename_handler))
            .route("/fs/set-readonly", post(set_readonly_handler))
            .with_state(harness.clone());
        let std_listener = StdTcpListener::bind("127.0.0.1:0").unwrap();
        std_listener.set_nonblocking(true).expect("nonblocking");
        let addr = std_listener.local_addr().unwrap();
        let service = router.into_make_service();
        let rt = Runtime::new().unwrap();
        rt.spawn(async move {
            let listener = TokioTcpListener::from_std(std_listener).unwrap();
            axum::serve(listener, service).await.unwrap();
        });
        (format!("http://{addr}"), harness, rt)
    }

    fn spawn_server_with_download_url() -> (String, Harness, Runtime) {
        let harness = Harness::new();
        let router = Router::new()
            .route("/fs/metadata", get(metadata_handler))
            .route("/fs/dir", get(dir_handler).delete(delete_dir_handler))
            .route("/fs/file", delete(delete_file_handler))
            .route("/fs/canonicalize", get(canonicalize_handler))
            .route("/fs/read", get(read_with_download_handler))
            .route("/fs/write", put(write_handler))
            .route("/fs/mkdir", post(mkdir_handler))
            .route("/fs/rename", post(rename_handler))
            .route("/fs/set-readonly", post(set_readonly_handler))
            .route("/download", get(download_handler))
            .with_state(harness.clone());
        let std_listener = StdTcpListener::bind("127.0.0.1:0").unwrap();
        std_listener.set_nonblocking(true).expect("nonblocking");
        let addr = std_listener.local_addr().unwrap();
        let service = router.into_make_service();
        let rt = Runtime::new().unwrap();
        let base = format!("http://{addr}");
        *harness.base_url.lock().unwrap() = Some(base.clone());
        rt.spawn(async move {
            let listener = TokioTcpListener::from_std(std_listener).unwrap();
            axum::serve(listener, service).await.unwrap();
        });
        (base, harness, rt)
    }

    #[test]
    fn remote_provider_roundtrip() {
        let (base, harness, _rt) = spawn_server();
        let provider = RemoteFsProvider::new(RemoteFsConfig {
            base_url: base,
            auth_token: None,
            chunk_bytes: 1024,
            parallel_requests: 4,
            direct_read_threshold_bytes: u64::MAX,
            timeout: Duration::from_secs(30),
            ..RemoteFsConfig::default()
        })
        .expect("provider");

        let data = (0..16_384u32)
            .flat_map(|v| v.to_le_bytes())
            .collect::<Vec<_>>();
        provider
            .write(Path::new("/reports/data.bin"), &data)
            .expect("write");
        let read_back = provider.read(Path::new("/reports/data.bin")).expect("read");
        assert_eq!(data, read_back);
        provider
            .remove_file(Path::new("/reports/data.bin"))
            .expect("remove");
        assert!(!harness.resolve("/reports/data.bin").exists());
    }

    #[test]
    fn remote_provider_download_url_read() {
        let (base, harness, _rt) = spawn_server_with_download_url();
        let provider = RemoteFsProvider::new(RemoteFsConfig {
            base_url: base,
            auth_token: None,
            chunk_bytes: 128,
            parallel_requests: 2,
            direct_read_threshold_bytes: u64::MAX,
            timeout: Duration::from_secs(30),
            ..RemoteFsConfig::default()
        })
        .expect("provider");

        let data = (0..1024u32)
            .flat_map(|v| v.to_le_bytes())
            .collect::<Vec<_>>();
        std::fs::create_dir_all(harness.resolve("/reports")).expect("mkdir");
        std::fs::write(harness.resolve("/reports/data.bin"), &data).expect("write");

        let read_back = provider.read(Path::new("/reports/data.bin")).expect("read");
        assert_eq!(data, read_back);
    }
}

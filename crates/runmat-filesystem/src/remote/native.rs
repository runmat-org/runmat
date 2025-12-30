use crate::{DirEntry, FileHandle, FsFileType, FsMetadata, FsProvider, OpenFlags};
use crossbeam_utils::thread;
use once_cell::sync::Lazy;
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
    pub timeout: Duration,
}

impl Default for RemoteFsConfig {
    fn default() -> Self {
        Self {
            base_url: String::new(),
            auth_token: None,
            chunk_bytes: 8 * 1024 * 1024,
            parallel_requests: 4,
            timeout: Duration::from_secs(120),
        }
    }
}

struct RemoteInner {
    client: Client,
    base: Url,
    auth_header: Option<String>,
    chunk_bytes: usize,
    parallel_requests: usize,
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
        })
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
            .request(reqwest::Method::GET, route, query)
            .send()
            .map_err(map_http_err)?;
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

    fn delete_empty(&self, route: &str, query: &[(&str, String)]) -> io::Result<()> {
        let resp = self
            .request(reqwest::Method::DELETE, route, query)
            .send()
            .map_err(map_http_err)?;
        handle_error(resp)?;
        Ok(())
    }

    fn download_chunk(&self, path: &str, offset: u64, length: usize) -> io::Result<Vec<u8>> {
        let resp = self
            .request(
                reqwest::Method::GET,
                "/fs/read",
                &[
                    ("path", path.to_string()),
                    ("offset", offset.to_string()),
                    ("length", length.to_string()),
                ],
            )
            .send()
            .map_err(map_http_err)?;
        let mut body = handle_error(resp)?;
        let mut buf = Vec::with_capacity(length);
        body.copy_to(&mut buf).map_err(map_http_err)?;
        Ok(buf)
    }

    fn upload_chunk(&self, path: &str, offset: u64, truncate: bool, data: &[u8]) -> io::Result<()> {
        let mut query = vec![("path", path.to_string()), ("offset", offset.to_string())];
        if truncate {
            query.push(("truncate", "true".to_string()));
        }
        let resp = self
            .request(reqwest::Method::PUT, "/fs/write", &query)
            .body(data.to_vec())
            .send()
            .map_err(map_http_err)?;
        handle_error(resp)?;
        Ok(())
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

    fn download_entire_file(&self, path: &str, len: u64) -> io::Result<Vec<u8>> {
        if len == 0 {
            return Ok(Vec::new());
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

    fn upload_entire_file(&self, path: &str, data: &[u8]) -> io::Result<()> {
        if data.is_empty() {
            self.inner.upload_chunk(path, 0, true, data)?;
            return Ok(());
        }
        let chunk = self.inner.chunk_bytes;
        let mut tasks = Vec::new();
        let mut offset = 0usize;
        let mut index = 0;
        while offset < data.len() {
            let end = std::cmp::min(offset + chunk, data.len());
            tasks.push(ChunkTask {
                offset: offset as u64,
                length: end - offset,
                index,
            });
            offset = end;
            index += 1;
        }
        let path = path.to_string();
        let inner = self.inner.clone();
        let queue = Arc::new(Mutex::new(VecDeque::from(tasks.clone())));
        let error: Arc<Mutex<Option<io::Error>>> = Arc::new(Mutex::new(None));
        let threads = inner
            .parallel_requests
            .min(queue.lock().unwrap().len())
            .max(1);
        thread::scope(|scope| {
            for _ in 0..threads {
                let queue = queue.clone();
                let error = error.clone();
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
                    let start = task.offset as usize;
                    let end = start + task.length;
                    let slice = &data[start..end];
                    let truncate = task.offset == 0;
                    if let Err(err) = inner.upload_chunk(&path, task.offset, truncate, slice) {
                        let mut guard = error.lock().unwrap();
                        if guard.is_none() {
                            *guard = Some(err);
                        }
                        break;
                    }
                });
            }
        })
        .expect("remote upload scope");
        if let Some(err) = error.lock().unwrap().take() {
            return Err(err);
        }
        Ok(())
    }
}

#[derive(Clone, Copy)]
struct ChunkTask {
    offset: u64,
    length: usize,
    index: usize,
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
            data = self.download_entire_file(&normalized, meta.len)?;
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
        self.download_entire_file(&normalized, meta.len)
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
    modified: Option<u64>,
    readonly: bool,
}

impl From<MetadataResponse> for FsMetadata {
    fn from(value: MetadataResponse) -> Self {
        FsMetadata {
            file_type: value.file_type.into(),
            len: value.len,
            modified: value
                .modified
                .map(|secs| std::time::UNIX_EPOCH + Duration::from_secs(secs)),
            readonly: value.readonly,
        }
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

fn map_http_err(err: reqwest::Error) -> io::Error {
    io::Error::other(err)
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
    use axum::http::StatusCode;
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
    }

    impl Harness {
        fn new() -> Self {
            let dir = tempdir().expect("tempdir");
            let path = dir.path().to_path_buf();
            Self {
                root: Arc::new(path),
                _keeper: Arc::new(dir),
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

    #[test]
    fn remote_provider_roundtrip() {
        let (base, harness, _rt) = spawn_server();
        let provider = RemoteFsProvider::new(RemoteFsConfig {
            base_url: base,
            auth_token: None,
            chunk_bytes: 1024,
            parallel_requests: 4,
            timeout: Duration::from_secs(30),
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
}

use std::collections::{BTreeMap, HashMap};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Mutex, OnceLock};

use chrono::Utc;
use runmat_builtins::{ObjectInstance, Tensor, Value};
use runmat_filesystem as fs;
use runmat_filesystem::data_contract::{
    DataChunkDescriptor, DataChunkUploadRequest, DataChunkUploadTarget,
};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::{build_runtime_error, BuiltinResult, RuntimeError};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataManifest {
    pub schema_version: u32,
    pub format: String,
    pub dataset_id: String,
    pub name: Option<String>,
    pub created_at: String,
    pub updated_at: String,
    pub arrays: BTreeMap<String, DataArrayMeta>,
    pub attrs: BTreeMap<String, serde_json::Value>,
    pub txn_sequence: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataArrayMeta {
    pub dtype: String,
    pub shape: Vec<usize>,
    pub chunk_shape: Vec<usize>,
    #[serde(default = "default_array_order")]
    pub order: String,
    pub codec: String,
    #[serde(default)]
    pub chunk_index_path: Option<String>,
    pub data_path: String,
}

fn default_array_order() -> String {
    "column_major".to_string()
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataArrayPayload {
    pub dtype: String,
    pub shape: Vec<usize>,
    pub values: Vec<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataChunkIndex {
    pub schema_version: u32,
    pub array: String,
    pub chunks: Vec<DataChunkIndexEntry>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataChunkIndexEntry {
    pub key: String,
    pub object_id: String,
    pub hash: String,
    pub bytes_raw: u64,
    pub bytes_stored: u64,
    #[serde(default)]
    pub coords: Vec<usize>,
    #[serde(default)]
    pub shape: Vec<usize>,
    pub data_path: String,
}

#[derive(Debug, Clone)]
pub struct DataSchema {
    pub arrays: BTreeMap<String, DataArrayMeta>,
}

#[derive(Debug, Clone)]
pub struct PendingTxn {
    pub dataset_path: String,
    pub base_sequence: u64,
    pub writes: Vec<PendingWrite>,
    pub resizes: Vec<PendingResize>,
    pub fills: Vec<PendingFill>,
    pub create_arrays: Vec<PendingCreateArray>,
    pub delete_arrays: Vec<String>,
    pub attrs: BTreeMap<String, Value>,
    pub status: TxnStatus,
}

#[derive(Debug, Clone)]
pub struct PendingWrite {
    pub array: String,
    pub slice_spec: Option<Value>,
    pub value: Value,
}

#[derive(Debug, Clone)]
pub struct PendingResize {
    pub array: String,
    pub shape: Vec<usize>,
}

#[derive(Debug, Clone)]
pub struct PendingFill {
    pub array: String,
    pub slice_spec: Option<Value>,
    pub value: Value,
}

#[derive(Debug, Clone)]
pub struct PendingCreateArray {
    pub array: String,
    pub meta: DataArrayMeta,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TxnStatus {
    Open,
    Committed,
    Aborted,
}

fn tx_registry() -> &'static Mutex<HashMap<String, PendingTxn>> {
    static REG: OnceLock<Mutex<HashMap<String, PendingTxn>>> = OnceLock::new();
    REG.get_or_init(|| Mutex::new(HashMap::new()))
}

pub fn data_error(message: impl Into<String>) -> RuntimeError {
    build_runtime_error(message)
        .with_identifier("RUNMAT:Data:Error")
        .with_builtin("data")
        .build()
}

pub fn parse_string(value: &Value, context: &str) -> BuiltinResult<String> {
    match value {
        Value::String(s) => Ok(s.clone()),
        Value::CharArray(ca) => Ok(ca.to_string()),
        _ => Err(data_error(format!("{context}: expected string value"))),
    }
}

pub fn dataset_root(path: &str) -> PathBuf {
    PathBuf::from(path)
}

pub fn manifest_path(root: &Path) -> PathBuf {
    root.join("manifest.json")
}

pub fn arrays_root(root: &Path) -> PathBuf {
    root.join("arrays")
}

pub async fn write_manifest_async(root: &Path, manifest: &DataManifest) -> BuiltinResult<()> {
    fs::create_dir_all_async(root).await.map_err(|err| {
        data_error(format!(
            "failed to create dataset root '{}': {err}",
            root.display()
        ))
    })?;
    let path = manifest_path(root);
    let bytes = serde_json::to_vec_pretty(manifest)
        .map_err(|err| data_error(format!("failed to encode manifest json: {err}")))?;
    fs::write_async(&path, &bytes).await.map_err(|err| {
        data_error(format!(
            "failed to write manifest '{}': {err}",
            path.display()
        ))
    })?;
    Ok(())
}

pub async fn read_manifest_async(root: &Path) -> BuiltinResult<DataManifest> {
    let path = manifest_path(root);
    let bytes = fs::read_async(&path).await.map_err(|err| {
        data_error(format!(
            "failed to read manifest '{}': {err}",
            path.display()
        ))
    })?;
    let manifest = serde_json::from_slice::<DataManifest>(&bytes).map_err(|err| {
        data_error(format!(
            "failed to parse manifest '{}': {err}",
            path.display()
        ))
    })?;
    Ok(manifest)
}

pub async fn write_array_payload_async(
    root: &Path,
    array: &str,
    payload: &DataArrayPayload,
    chunk_shape: &[usize],
) -> BuiltinResult<(PathBuf, PathBuf)> {
    let array_dir = arrays_root(root).join(array);
    fs::create_dir_all_async(&array_dir).await.map_err(|err| {
        data_error(format!(
            "failed to create array dir '{}': {err}",
            array_dir.display()
        ))
    })?;
    let payload_path = array_dir.join("data.f64.json");
    let bytes = serde_json::to_vec(payload)
        .map_err(|err| data_error(format!("failed to encode array payload json: {err}")))?;
    fs::write_async(&payload_path, &bytes)
        .await
        .map_err(|err| {
            data_error(format!(
                "failed to write payload '{}': {err}",
                payload_path.display()
            ))
        })?;

    let chunk_dir = array_dir.join("chunks");
    fs::create_dir_all_async(&chunk_dir).await.map_err(|err| {
        data_error(format!(
            "failed to create chunk dir '{}': {err}",
            chunk_dir.display()
        ))
    })?;

    let mut index = DataChunkIndex {
        schema_version: 1,
        array: array.to_string(),
        chunks: Vec::new(),
    };
    let mut upload_chunks = Vec::new();
    let grid_shape = chunk_grid_shape(&payload.shape, chunk_shape);
    let mut coords = vec![0usize; payload.shape.len()];
    loop {
        let chunk_start = chunk_start_for_coords(&coords, chunk_shape);
        let chunk_extent = chunk_extent_for_start(&chunk_start, chunk_shape, &payload.shape);
        let chunk_payload = DataArrayPayload {
            dtype: payload.dtype.clone(),
            shape: chunk_extent.clone(),
            values: collect_chunk_values(payload, &chunk_start, &chunk_extent)?,
        };
        let key = chunk_key(&coords);
        let object_id = format!("obj_{}", key.replace('.', "_"));
        let chunk_bytes = serde_json::to_vec(&chunk_payload)
            .map_err(|err| data_error(format!("failed to encode chunk payload: {err}")))?;
        let data_path = chunk_dir.join(format!("{object_id}.json"));
        fs::write_async(&data_path, &chunk_bytes)
            .await
            .map_err(|err| {
                data_error(format!(
                    "failed to write chunk '{}': {err}",
                    data_path.display()
                ))
            })?;
        let hash = sha256_hex(&chunk_bytes);
        let rel_chunk_path = data_path
            .strip_prefix(root)
            .map_err(|err| data_error(format!("failed to compute chunk relative path: {err}")))?
            .to_string_lossy()
            .to_string();
        index.chunks.push(DataChunkIndexEntry {
            key: key.clone(),
            object_id: object_id.clone(),
            hash: hash.clone(),
            bytes_raw: chunk_bytes.len() as u64,
            bytes_stored: chunk_bytes.len() as u64,
            coords: coords.clone(),
            shape: chunk_extent,
            data_path: rel_chunk_path,
        });
        upload_chunks.push((
            DataChunkDescriptor {
                key,
                object_id,
                hash,
                bytes_raw: chunk_bytes.len() as u64,
                bytes_stored: chunk_bytes.len() as u64,
            },
            chunk_bytes,
        ));
        if !advance_index(&mut coords, &grid_shape) {
            break;
        }
    }

    maybe_upload_chunks_async(root, array, upload_chunks).await?;

    tracing::info!(
        target: "runmat.data",
        dataset = %root.display(),
        array = array,
        chunks = index.chunks.len(),
        payload_bytes = bytes.len(),
        "data chunk write planned"
    );

    let chunk_index_path = chunk_dir.join("index.json");
    let chunk_index_bytes = serde_json::to_vec(&index)
        .map_err(|err| data_error(format!("failed to encode chunk index json: {err}")))?;
    fs::write_async(&chunk_index_path, &chunk_index_bytes)
        .await
        .map_err(|err| {
            data_error(format!(
                "failed to write chunk index '{}': {err}",
                chunk_index_path.display()
            ))
        })?;
    Ok((payload_path, chunk_index_path))
}

pub async fn read_array_payload_async(
    root: &Path,
    meta: &DataArrayMeta,
) -> BuiltinResult<DataArrayPayload> {
    if let Some(index_path) = &meta.chunk_index_path {
        let path = root.join(index_path);
        if fs::metadata_async(&path).await.is_ok() {
            return read_array_payload_chunked_async(root, meta, &path).await;
        }
    }
    let payload_path = root.join(&meta.data_path);
    let bytes = fs::read_async(&payload_path).await.map_err(|err| {
        data_error(format!(
            "failed to read payload '{}': {err}",
            payload_path.display()
        ))
    })?;
    serde_json::from_slice::<DataArrayPayload>(&bytes).map_err(|err| {
        data_error(format!(
            "failed to parse payload '{}': {err}",
            payload_path.display()
        ))
    })
}

async fn read_array_payload_chunked_async(
    root: &Path,
    meta: &DataArrayMeta,
    index_path: &Path,
) -> BuiltinResult<DataArrayPayload> {
    let bytes = fs::read_async(index_path).await.map_err(|err| {
        data_error(format!(
            "failed to read chunk index '{}': {err}",
            index_path.display()
        ))
    })?;
    let index: DataChunkIndex = serde_json::from_slice(&bytes).map_err(|err| {
        data_error(format!(
            "failed to parse chunk index '{}': {err}",
            index_path.display()
        ))
    })?;
    let mut values = vec![0.0; meta.shape.iter().copied().product::<usize>()];
    for chunk in index.chunks {
        let chunk_path = root.join(&chunk.data_path);
        let bytes = fs::read_async(&chunk_path).await.map_err(|err| {
            data_error(format!(
                "failed to read chunk payload '{}': {err}",
                chunk_path.display()
            ))
        })?;
        let payload: DataArrayPayload = serde_json::from_slice(&bytes).map_err(|err| {
            data_error(format!(
                "failed to parse chunk payload '{}': {err}",
                chunk_path.display()
            ))
        })?;
        let coords = chunk_coords_from_entry(&chunk, meta.shape.len())?;
        let chunk_start = chunk_start_for_coords(&coords, &meta.chunk_shape);
        let chunk_extent = if chunk.shape.is_empty() {
            chunk_extent_for_start(&chunk_start, &meta.chunk_shape, &meta.shape)
        } else {
            chunk.shape.clone()
        };
        if payload.shape != chunk_extent {
            return Err(data_error(format!(
                "chunk payload shape mismatch for key '{}': {:?} != {:?}",
                chunk.key, payload.shape, chunk_extent
            )));
        }
        let mut local = vec![0usize; chunk_extent.len()];
        loop {
            let mut global = Vec::with_capacity(chunk_extent.len());
            for dim in 0..chunk_extent.len() {
                global.push(chunk_start[dim] + local[dim]);
            }
            let src_linear = linear_index_column_major(&local, &chunk_extent)?;
            let dst_linear = linear_index_column_major(&global, &meta.shape)?;
            values[dst_linear] = payload.values[src_linear];
            if !advance_index(&mut local, &chunk_extent) {
                break;
            }
        }
    }
    Ok(DataArrayPayload {
        dtype: meta.dtype.clone(),
        shape: meta.shape.clone(),
        values,
    })
}

async fn maybe_upload_chunks_async(
    root: &Path,
    array: &str,
    chunks: Vec<(DataChunkDescriptor, Vec<u8>)>,
) -> BuiltinResult<()> {
    if chunks.is_empty() {
        return Ok(());
    }
    let request = DataChunkUploadRequest {
        dataset_path: root.to_string_lossy().to_string(),
        array: array.to_string(),
        chunks: chunks.iter().map(|(desc, _)| desc.clone()).collect(),
    };
    let targets = match fs::data_chunk_upload_targets_async(&request).await {
        Ok(targets) => targets,
        Err(err) if err.kind() == std::io::ErrorKind::Unsupported => return Ok(()),
        Err(err) => {
            return Err(data_error(format!(
                "failed to request data chunk upload targets: {err}"
            )))
        }
    };
    for (descriptor, bytes) in chunks {
        let target = find_chunk_target(&targets, &descriptor.key)?;
        fs::data_upload_chunk_async(target, &bytes)
            .await
            .map_err(|err| {
                data_error(format!(
                    "failed to upload chunk '{}': {err}",
                    descriptor.key
                ))
            })?;
        tracing::info!(
            target: "runmat.data",
            dataset = %root.display(),
            array = array,
            chunk_key = descriptor.key,
            bytes = bytes.len(),
            "data chunk uploaded"
        );
    }
    Ok(())
}

fn find_chunk_target<'a>(
    targets: &'a [DataChunkUploadTarget],
    key: &str,
) -> BuiltinResult<&'a DataChunkUploadTarget> {
    targets
        .iter()
        .find(|target| target.key == key)
        .ok_or_else(|| data_error(format!("missing upload target for chunk '{key}'")))
}

pub fn sha256_hex(bytes: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    let digest = hasher.finalize();
    format!("sha256:{:x}", digest)
}

fn chunk_key(coords: &[usize]) -> String {
    coords
        .iter()
        .map(|v| v.to_string())
        .collect::<Vec<_>>()
        .join(".")
}

fn chunk_grid_shape(shape: &[usize], chunk_shape: &[usize]) -> Vec<usize> {
    shape
        .iter()
        .enumerate()
        .map(|(idx, extent)| {
            let chunk = chunk_shape.get(idx).copied().unwrap_or(1).max(1);
            extent.div_ceil(chunk)
        })
        .collect()
}

fn chunk_start_for_coords(coords: &[usize], chunk_shape: &[usize]) -> Vec<usize> {
    coords
        .iter()
        .enumerate()
        .map(|(idx, coord)| coord * chunk_shape.get(idx).copied().unwrap_or(1).max(1))
        .collect()
}

fn chunk_extent_for_start(
    start: &[usize],
    chunk_shape: &[usize],
    full_shape: &[usize],
) -> Vec<usize> {
    start
        .iter()
        .enumerate()
        .map(|(idx, start)| {
            let chunk = chunk_shape.get(idx).copied().unwrap_or(1).max(1);
            let end = (*start + chunk).min(full_shape[idx]);
            end.saturating_sub(*start)
        })
        .collect()
}

fn collect_chunk_values(
    payload: &DataArrayPayload,
    chunk_start: &[usize],
    chunk_extent: &[usize],
) -> BuiltinResult<Vec<f64>> {
    let mut local = vec![0usize; chunk_extent.len()];
    let mut values = Vec::with_capacity(chunk_extent.iter().copied().product());
    loop {
        let mut global = Vec::with_capacity(chunk_extent.len());
        for dim in 0..chunk_extent.len() {
            global.push(chunk_start[dim] + local[dim]);
        }
        let linear = linear_index_column_major(&global, &payload.shape)?;
        values.push(payload.values[linear]);
        if !advance_index(&mut local, chunk_extent) {
            break;
        }
    }
    Ok(values)
}

fn chunk_coords_from_entry(entry: &DataChunkIndexEntry, rank: usize) -> BuiltinResult<Vec<usize>> {
    if !entry.coords.is_empty() {
        if entry.coords.len() != rank {
            return Err(data_error(format!(
                "chunk coords rank mismatch for key '{}': expected {rank}, got {}",
                entry.key,
                entry.coords.len()
            )));
        }
        return Ok(entry.coords.clone());
    }
    let coords = entry
        .key
        .split('.')
        .map(|part| {
            part.parse::<usize>()
                .map_err(|_| data_error(format!("invalid chunk key '{}'", entry.key)))
        })
        .collect::<BuiltinResult<Vec<_>>>()?;
    if coords.len() != rank {
        return Err(data_error(format!(
            "chunk key rank mismatch for key '{}': expected {rank}, got {}",
            entry.key,
            coords.len()
        )));
    }
    Ok(coords)
}

fn linear_index_column_major(index: &[usize], shape: &[usize]) -> BuiltinResult<usize> {
    if index.len() != shape.len() {
        return Err(data_error("chunk index rank mismatch"));
    }
    let mut stride = 1usize;
    let mut linear = 0usize;
    for (idx, extent) in index.iter().zip(shape.iter()) {
        if *idx >= *extent {
            return Err(data_error("chunk index out of bounds"));
        }
        linear += idx * stride;
        stride = stride.saturating_mul(*extent);
    }
    Ok(linear)
}

fn advance_index(index: &mut [usize], shape: &[usize]) -> bool {
    if shape.is_empty() {
        return false;
    }
    for dim in 0..shape.len() {
        index[dim] += 1;
        if index[dim] < shape[dim] {
            return true;
        }
        index[dim] = 0;
    }
    false
}

pub fn parse_schema(schema: &Value) -> BuiltinResult<DataSchema> {
    let Value::Struct(schema_struct) = schema else {
        return Err(data_error("data.create: schema must be a struct"));
    };
    let arrays_value = schema_struct
        .fields
        .get("arrays")
        .ok_or_else(|| data_error("data.create: schema missing 'arrays' field"))?;
    let Value::Struct(arrays_struct) = arrays_value else {
        return Err(data_error("data.create: schema.arrays must be a struct"));
    };

    let mut arrays = BTreeMap::new();
    for (name, meta_value) in &arrays_struct.fields {
        let Value::Struct(meta_struct) = meta_value else {
            return Err(data_error(format!(
                "data.create: schema.arrays.{name} must be a struct"
            )));
        };
        let dtype = meta_struct
            .fields
            .get("dtype")
            .map(|v| parse_string(v, "data.create schema dtype"))
            .transpose()?
            .unwrap_or_else(|| "f64".to_string());
        let shape = meta_struct
            .fields
            .get("shape")
            .map(parse_usize_vector)
            .transpose()?
            .unwrap_or_else(|| vec![0, 0]);
        let chunk_shape = meta_struct
            .fields
            .get("chunk")
            .map(parse_usize_vector)
            .transpose()?
            .unwrap_or_else(|| default_chunk_shape(&shape));
        let codec = meta_struct
            .fields
            .get("codec")
            .map(|v| parse_string(v, "data.create schema codec"))
            .transpose()?
            .unwrap_or_else(|| "zstd".to_string());
        let data_path = format!("arrays/{name}/data.f64.json");
        let chunk_index_path = format!("arrays/{name}/chunks/index.json");
        arrays.insert(
            name.clone(),
            DataArrayMeta {
                dtype,
                shape,
                chunk_shape,
                order: default_array_order(),
                codec,
                chunk_index_path: Some(chunk_index_path),
                data_path,
            },
        );
    }

    Ok(DataSchema { arrays })
}

fn default_chunk_shape(shape: &[usize]) -> Vec<usize> {
    if shape.is_empty() {
        return vec![1024];
    }
    let mut out = shape.to_vec();
    if out.len() == 1 {
        out[0] = out[0].clamp(1, 65_536);
        return out;
    }
    out[0] = out[0].clamp(1, 256);
    out[1] = out[1].clamp(1, 256);
    for dim in out.iter_mut().skip(2) {
        *dim = (*dim).clamp(1, 8);
    }
    out
}

fn parse_usize_vector(value: &Value) -> BuiltinResult<Vec<usize>> {
    match value {
        Value::Tensor(t) => tensor_to_usize_vector(t),
        Value::Num(n) => {
            if *n < 0.0 || !n.is_finite() {
                return Err(data_error(
                    "data schema dimensions must be non-negative finite numbers",
                ));
            }
            Ok(vec![*n as usize])
        }
        Value::Int(i) => {
            let n = i.to_i64();
            if n < 0 {
                return Err(data_error("data schema dimensions must be non-negative"));
            }
            Ok(vec![n as usize])
        }
        _ => Err(data_error(
            "data schema dimension field must be numeric tensor/vector",
        )),
    }
}

fn tensor_to_usize_vector(t: &Tensor) -> BuiltinResult<Vec<usize>> {
    let mut out = Vec::with_capacity(t.data.len());
    for value in &t.data {
        if !value.is_finite() || *value < 0.0 {
            return Err(data_error(
                "data schema dimensions must be non-negative finite numbers",
            ));
        }
        out.push(*value as usize);
    }
    Ok(out)
}

pub fn dataset_object(path: &str, manifest: &DataManifest) -> Value {
    let mut obj = ObjectInstance::new("Dataset".to_string());
    obj.properties
        .insert("__data_path".to_string(), Value::String(path.to_string()));
    obj.properties.insert(
        "__data_id".to_string(),
        Value::String(manifest.dataset_id.clone()),
    );
    obj.properties.insert(
        "__data_version".to_string(),
        Value::String(manifest_version_token(manifest)),
    );
    Value::Object(obj)
}

pub fn manifest_version_token(manifest: &DataManifest) -> String {
    format!("{}:{}", manifest.updated_at, manifest.txn_sequence)
}

pub fn ensure_manifest_sequence(expected: u64, manifest: &DataManifest) -> BuiltinResult<()> {
    if manifest.txn_sequence != expected {
        tracing::warn!(
            target: "runmat.data",
            expected_sequence = expected,
            actual_sequence = manifest.txn_sequence,
            "manifest conflict detected"
        );
        return Err(data_error(
            "MANIFEST_CONFLICT: dataset changed since transaction begin",
        ));
    }
    Ok(())
}

pub fn array_object(dataset_path: &str, array_name: &str) -> Value {
    let mut obj = ObjectInstance::new("DataArray".to_string());
    obj.properties.insert(
        "__data_path".to_string(),
        Value::String(dataset_path.to_string()),
    );
    obj.properties.insert(
        "__array_name".to_string(),
        Value::String(array_name.to_string()),
    );
    Value::Object(obj)
}

pub fn transaction_object(dataset_path: &str, tx_id: &str) -> Value {
    let mut obj = ObjectInstance::new("DataTransaction".to_string());
    obj.properties.insert(
        "__data_path".to_string(),
        Value::String(dataset_path.to_string()),
    );
    obj.properties
        .insert("__tx_id".to_string(), Value::String(tx_id.to_string()));
    Value::Object(obj)
}

pub fn get_object_prop<'a>(obj: &'a ObjectInstance, key: &str) -> BuiltinResult<&'a Value> {
    obj.properties
        .get(key)
        .ok_or_else(|| data_error(format!("object missing internal property '{key}'")))
}

pub fn now_rfc3339() -> String {
    Utc::now().to_rfc3339_opts(chrono::SecondsFormat::Secs, true)
}

pub fn new_dataset_id() -> String {
    static NEXT_DATASET_ID: AtomicU64 = AtomicU64::new(1);
    let seq = NEXT_DATASET_ID.fetch_add(1, Ordering::Relaxed);
    format!("ds_{}_{}", Utc::now().timestamp_millis(), seq)
}

pub fn new_tx_id() -> String {
    static NEXT_TX_ID: AtomicU64 = AtomicU64::new(1);
    let seq = NEXT_TX_ID.fetch_add(1, Ordering::Relaxed);
    format!("tx_{}_{}", Utc::now().timestamp_millis(), seq)
}

pub fn start_tx(dataset_path: String, base_sequence: u64) -> String {
    let tx_id = new_tx_id();
    let pending = PendingTxn {
        dataset_path,
        base_sequence,
        writes: Vec::new(),
        resizes: Vec::new(),
        fills: Vec::new(),
        create_arrays: Vec::new(),
        delete_arrays: Vec::new(),
        attrs: BTreeMap::new(),
        status: TxnStatus::Open,
    };
    let mut guard = tx_registry().lock().expect("tx registry lock poisoned");
    guard.insert(tx_id.clone(), pending);
    tx_id
}

pub fn with_tx_mut<T>(
    tx_id: &str,
    f: impl FnOnce(&mut PendingTxn) -> BuiltinResult<T>,
) -> BuiltinResult<T> {
    let mut guard = tx_registry().lock().expect("tx registry lock poisoned");
    let tx = guard
        .get_mut(tx_id)
        .ok_or_else(|| data_error(format!("transaction '{tx_id}' not found")))?;
    f(tx)
}

pub fn with_tx<T>(
    tx_id: &str,
    f: impl FnOnce(&PendingTxn) -> BuiltinResult<T>,
) -> BuiltinResult<T> {
    let guard = tx_registry().lock().expect("tx registry lock poisoned");
    let tx = guard
        .get(tx_id)
        .ok_or_else(|| data_error(format!("transaction '{tx_id}' not found")))?;
    f(tx)
}

pub fn remove_tx(tx_id: &str) {
    let mut guard = tx_registry().lock().expect("tx registry lock poisoned");
    let _ = guard.remove(tx_id);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ensure_manifest_sequence_accepts_matching_sequence() {
        let manifest = DataManifest {
            schema_version: 1,
            format: "runmat-data".to_string(),
            dataset_id: "ds_test".to_string(),
            name: Some("test".to_string()),
            created_at: "2026-03-01T00:00:00Z".to_string(),
            updated_at: "2026-03-01T00:00:00Z".to_string(),
            arrays: BTreeMap::new(),
            attrs: BTreeMap::new(),
            txn_sequence: 5,
        };
        ensure_manifest_sequence(5, &manifest).expect("expected sequence match");
    }

    #[test]
    fn ensure_manifest_sequence_rejects_conflict() {
        let manifest = DataManifest {
            schema_version: 1,
            format: "runmat-data".to_string(),
            dataset_id: "ds_test".to_string(),
            name: Some("test".to_string()),
            created_at: "2026-03-01T00:00:00Z".to_string(),
            updated_at: "2026-03-01T00:00:00Z".to_string(),
            arrays: BTreeMap::new(),
            attrs: BTreeMap::new(),
            txn_sequence: 6,
        };
        let err = ensure_manifest_sequence(5, &manifest).expect_err("expected conflict error");
        assert!(err.message().contains("MANIFEST_CONFLICT"));
    }

    #[test]
    fn transaction_registry_roundtrip() {
        let tx_id = start_tx("/datasets/test.data".to_string(), 7);
        let status = with_tx(&tx_id, |tx| Ok(tx.status.clone())).expect("tx lookup");
        assert_eq!(status, TxnStatus::Open);
        remove_tx(&tx_id);
        let err = with_tx(&tx_id, |_| Ok(())).expect_err("expected missing tx");
        assert!(err.message().contains("not found"));
    }

    #[test]
    fn sha256_hash_format_matches_expected_prefix() {
        let hash = sha256_hex(b"runmat");
        assert!(hash.starts_with("sha256:"));
        assert_eq!(hash.len(), "sha256:".len() + 64);
    }
}

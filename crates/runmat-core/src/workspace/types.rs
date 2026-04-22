use runmat_builtins::Value;
use uuid::Uuid;

use super::materialize::MATERIALIZE_DEFAULT_LIMIT;

#[derive(Debug, Clone)]
pub struct WorkspacePreview {
    pub values: Vec<f64>,
    pub truncated: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WorkspaceResidency {
    Cpu,
    Gpu,
    Unknown,
}

impl WorkspaceResidency {
    pub fn as_str(&self) -> &'static str {
        match self {
            WorkspaceResidency::Cpu => "cpu",
            WorkspaceResidency::Gpu => "gpu",
            WorkspaceResidency::Unknown => "unknown",
        }
    }
}

#[derive(Debug, Clone)]
pub struct WorkspaceEntry {
    pub name: String,
    pub class_name: String,
    pub dtype: Option<String>,
    pub shape: Vec<usize>,
    pub is_gpu: bool,
    pub size_bytes: Option<u64>,
    pub preview: Option<WorkspacePreview>,
    pub residency: WorkspaceResidency,
    pub preview_token: Option<Uuid>,
}

#[derive(Debug, Clone)]
pub struct WorkspaceSnapshot {
    pub full: bool,
    pub version: u64,
    pub values: Vec<WorkspaceEntry>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WorkspaceExportMode {
    Off,
    Auto,
    Force,
}

#[derive(Debug, Clone)]
pub struct MaterializedVariable {
    pub name: String,
    pub class_name: String,
    pub dtype: Option<String>,
    pub shape: Vec<usize>,
    pub is_gpu: bool,
    pub residency: WorkspaceResidency,
    pub size_bytes: Option<u64>,
    pub preview: Option<WorkspacePreview>,
    pub value: Value,
}

#[derive(Debug, Clone)]
pub enum WorkspaceMaterializeTarget {
    Name(String),
    Token(Uuid),
}

#[derive(Debug, Clone)]
pub struct WorkspaceSliceOptions {
    pub start: Vec<usize>,
    pub shape: Vec<usize>,
}

impl WorkspaceSliceOptions {
    pub(crate) fn sanitized(&self, tensor_shape: &[usize]) -> Option<WorkspaceSliceOptions> {
        if tensor_shape.is_empty() {
            return None;
        }
        let mut start = Vec::with_capacity(tensor_shape.len());
        let mut shape = Vec::with_capacity(tensor_shape.len());
        for (axis_idx, axis_len) in tensor_shape.iter().enumerate() {
            let axis_len = *axis_len;
            if axis_len == 0 {
                return None;
            }
            let requested_start = self.start.get(axis_idx).copied().unwrap_or(0);
            let clamped_start = requested_start.min(axis_len.saturating_sub(1));
            let requested_count = self.shape.get(axis_idx).copied().unwrap_or(axis_len);
            let clamped_count = requested_count.max(1).min(axis_len - clamped_start);
            start.push(clamped_start);
            shape.push(clamped_count);
        }
        Some(WorkspaceSliceOptions { start, shape })
    }
}

#[derive(Debug, Clone)]
pub struct WorkspaceMaterializeOptions {
    pub max_elements: usize,
    pub slice: Option<WorkspaceSliceOptions>,
}

impl Default for WorkspaceMaterializeOptions {
    fn default() -> Self {
        Self {
            max_elements: MATERIALIZE_DEFAULT_LIMIT,
            slice: None,
        }
    }
}

use std::collections::BTreeMap;
use std::sync::{
    atomic::{AtomicBool, AtomicU64, Ordering},
    Arc, Mutex, OnceLock,
};

#[cxx::bridge(namespace = "runmat_geometry_io::occt_backend")]
pub(crate) mod bridge {
    #[repr(u8)]
    #[derive(Debug, Clone, Copy)]
    enum OcctCadFormat {
        Step,
        Iges,
        Brep,
    }

    #[derive(Debug, Clone, Copy)]
    struct OcctImportOptions {
        linear_deflection: f64,
        angular_deflection: f64,
        relative_deflection: bool,
        max_triangles: u64,
        truncate_at_max_triangles: bool,
        cancel_token_id: u64,
    }

    #[derive(Debug, Clone)]
    struct OcctFaceSemanticPayload {
        face_id: u64,
        label_entry: String,
        label_name: String,
        label_kind: String,
        owner_entries: Vec<String>,
        owner_names: Vec<String>,
        owner_kinds: Vec<String>,
        layer_names: Vec<String>,
        color_type: String,
        color_hex_rgba: String,
        material_label_entry: String,
        material_name: String,
        material_description: String,
        material_density: String,
        material_density_name: String,
        material_density_value_type: String,
    }

    #[derive(Debug, Clone)]
    struct OcctAssemblyNodePayload {
        node_id: String,
        parent_node_id: String,
        label: String,
    }

    #[derive(Debug, Clone)]
    struct OcctImportPayload {
        backend: String,
        format_name: String,
        truncated: bool,
        triangle_budget: u64,
        vertices: Vec<f64>,
        triangles: Vec<u32>,
        triangle_face_ids: Vec<u64>,
        face_ids: Vec<u64>,
        face_names: Vec<String>,
        face_semantics: Vec<OcctFaceSemanticPayload>,
        assembly_nodes: Vec<OcctAssemblyNodePayload>,
        warnings: Vec<String>,
    }

    unsafe extern "C++" {
        include!("runmat-geometry-io/src/occt/occt_bridge.hxx");

        fn import_cad_bytes(
            path: &str,
            bytes: &[u8],
            format: OcctCadFormat,
            options: OcctImportOptions,
        ) -> Result<OcctImportPayload>;
    }

    extern "Rust" {
        fn occt_import_cancelled(cancel_token_id: u64) -> bool;
    }
}

static NEXT_CANCEL_TOKEN_ID: AtomicU64 = AtomicU64::new(1);
static CANCEL_TOKENS: OnceLock<Mutex<BTreeMap<u64, Arc<AtomicBool>>>> = OnceLock::new();

pub(crate) struct OcctCancelTokenRegistration {
    id: u64,
}

impl OcctCancelTokenRegistration {
    pub(crate) fn new(flag: Option<Arc<AtomicBool>>) -> Self {
        let Some(flag) = flag else {
            return Self { id: 0 };
        };
        let id = NEXT_CANCEL_TOKEN_ID.fetch_add(1, Ordering::Relaxed);
        if let Ok(mut tokens) = cancel_tokens().lock() {
            tokens.insert(id, flag);
            Self { id }
        } else {
            Self { id: 0 }
        }
    }

    pub(crate) fn id(&self) -> u64 {
        self.id
    }
}

impl Drop for OcctCancelTokenRegistration {
    fn drop(&mut self) {
        if self.id == 0 {
            return;
        }
        if let Ok(mut tokens) = cancel_tokens().lock() {
            tokens.remove(&self.id);
        }
    }
}

fn occt_import_cancelled(cancel_token_id: u64) -> bool {
    if cancel_token_id == 0 {
        return false;
    }
    cancel_tokens()
        .lock()
        .ok()
        .and_then(|tokens| tokens.get(&cancel_token_id).cloned())
        .map(|flag| flag.load(Ordering::Relaxed))
        .unwrap_or(false)
}

fn cancel_tokens() -> &'static Mutex<BTreeMap<u64, Arc<AtomicBool>>> {
    CANCEL_TOKENS.get_or_init(|| Mutex::new(BTreeMap::new()))
}

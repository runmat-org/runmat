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
        max_exact_representation_bytes: u64,
        max_exact_entities: u64,
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
    struct OcctFaceEvaluationSamplePayload {
        face_id: u64,
        u: f64,
        v: f64,
        point_x: f64,
        point_y: f64,
        point_z: f64,
        normal_x: f64,
        normal_y: f64,
        normal_z: f64,
        projection_error: f64,
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
        face_evaluation_samples: Vec<OcctFaceEvaluationSamplePayload>,
        assembly_nodes: Vec<OcctAssemblyNodePayload>,
        warnings: Vec<String>,
    }

    #[derive(Debug, Clone)]
    struct OcctExactShapePayload {
        kernel_version: String,
        kernel_abi: String,
        representation: Vec<u8>,
        compound_count: u64,
        compsolid_count: u64,
        solid_count: u64,
        shell_count: u64,
        face_count: u64,
        wire_count: u64,
        edge_count: u64,
        vertex_count: u64,
        kernel_valid: bool,
        has_volume_properties: bool,
        volume: f64,
        surface_area: f64,
        centroid_x: f64,
        centroid_y: f64,
        centroid_z: f64,
        inertia_xx: f64,
        inertia_yy: f64,
        inertia_zz: f64,
        inertia_xy: f64,
        inertia_xz: f64,
        inertia_yz: f64,
        vertices: Vec<OcctExactVertexPayload>,
        edges: Vec<OcctExactEdgePayload>,
        faces: Vec<OcctExactFacePayload>,
        wires: Vec<OcctExactWirePayload>,
        coedges: Vec<OcctExactCoedgePayload>,
        shells: Vec<OcctExactShellPayload>,
        solids: Vec<OcctExactSolidPayload>,
    }

    #[derive(Debug, Clone)]
    struct OcctExactVertexPayload {
        shape_key: u64,
        point_x: f64,
        point_y: f64,
        point_z: f64,
        tolerance: f64,
    }

    #[derive(Debug, Clone)]
    struct OcctExactEdgePayload {
        shape_key: u64,
        start_vertex_key: u64,
        end_vertex_key: u64,
        closed: bool,
        periodic: bool,
        degenerate: bool,
    }

    #[derive(Debug, Clone)]
    struct OcctExactFacePayload {
        shape_key: u64,
        reversed: bool,
        outer_wire_key: u64,
        inner_wire_keys: Vec<u64>,
        periodic_u: bool,
        periodic_v: bool,
        singular: bool,
    }

    #[derive(Debug, Clone)]
    struct OcctExactWirePayload {
        shape_key: u64,
        face_key: u64,
        reversed: bool,
        coedge_keys: Vec<u64>,
    }

    #[derive(Debug, Clone)]
    struct OcctExactCoedgePayload {
        coedge_key: u64,
        face_key: u64,
        wire_key: u64,
        edge_key: u64,
        reversed: bool,
        has_pcurve: bool,
        seam_image: i8,
    }

    #[derive(Debug, Clone)]
    struct OcctExactShellPayload {
        shape_key: u64,
        reversed: bool,
        face_keys: Vec<u64>,
        face_reversed: Vec<bool>,
    }

    #[derive(Debug, Clone)]
    struct OcctExactSolidPayload {
        shape_key: u64,
        outer_shell_key: u64,
        void_shell_keys: Vec<u64>,
    }

    #[derive(Debug, Clone, Copy)]
    struct OcctCurveRangePayload {
        start: f64,
        end: f64,
    }

    #[derive(Debug, Clone, Copy)]
    struct OcctCurveDerivativesPayload {
        point_x: f64,
        point_y: f64,
        point_z: f64,
        first_x: f64,
        first_y: f64,
        first_z: f64,
        second_x: f64,
        second_y: f64,
        second_z: f64,
    }

    #[derive(Debug, Clone, Copy)]
    struct OcctCurveProjectionPayload {
        parameter: f64,
        point_x: f64,
        point_y: f64,
        point_z: f64,
        distance: f64,
    }

    #[derive(Debug, Clone, Copy)]
    struct OcctPcurveDerivativesPayload {
        range_start: f64,
        range_end: f64,
        point_u: f64,
        point_v: f64,
        first_u: f64,
        first_v: f64,
        second_u: f64,
        second_v: f64,
    }

    #[derive(Debug, Clone)]
    struct OcctPreviewSessionStartPayload {
        session_id: u64,
        face_count: u64,
    }

    #[derive(Debug, Clone, Copy)]
    struct OcctPreviewSessionChunkOptions {
        target_triangles: u64,
        max_faces: u64,
        cancel_token_id: u64,
    }

    #[derive(Debug, Clone)]
    struct OcctPreviewSessionChunkPayload {
        session_id: u64,
        done: bool,
        face_cursor: u64,
        face_count: u64,
        topology: OcctImportPayload,
    }

    unsafe extern "C++" {
        include!("runmat-geometry-io/src/occt/occt_bridge.hxx");

        fn import_cad_bytes(
            path: &str,
            bytes: &[u8],
            format: OcctCadFormat,
            options: OcctImportOptions,
        ) -> Result<OcctImportPayload>;

        fn import_exact_cad_bytes(
            path: &str,
            bytes: &[u8],
            format: OcctCadFormat,
            options: OcctImportOptions,
        ) -> Result<OcctExactShapePayload>;

        fn start_exact_evaluator_session(
            representation: &[u8],
            meters_per_source_unit: f64,
        ) -> Result<u64>;

        fn exact_curve_range(session_id: u64, shape_key: u64) -> Result<OcctCurveRangePayload>;

        fn exact_curve_derivatives(
            session_id: u64,
            shape_key: u64,
            parameter: f64,
        ) -> Result<OcctCurveDerivativesPayload>;

        fn exact_curve_arc_length(
            session_id: u64,
            shape_key: u64,
            start: f64,
            end: f64,
            absolute_error_m: f64,
        ) -> Result<f64>;

        fn exact_curve_inverse_project(
            session_id: u64,
            shape_key: u64,
            point_m: &[f64],
            absolute_error_m: f64,
        ) -> Result<OcctCurveProjectionPayload>;

        fn exact_pcurve_derivatives(
            session_id: u64,
            face_key: u64,
            wire_key: u64,
            coedge_position: u64,
            seam_image: i8,
            parameter: f64,
        ) -> Result<OcctPcurveDerivativesPayload>;

        fn exact_pcurve_range(
            session_id: u64,
            face_key: u64,
            wire_key: u64,
            coedge_position: u64,
            seam_image: i8,
        ) -> Result<OcctCurveRangePayload>;

        fn close_exact_evaluator_session(session_id: u64);

        fn start_cad_preview_session(
            path: &str,
            bytes: &[u8],
            format: OcctCadFormat,
            options: OcctImportOptions,
        ) -> Result<OcctPreviewSessionStartPayload>;

        fn read_cad_preview_session_chunk(
            session_id: u64,
            options: OcctPreviewSessionChunkOptions,
        ) -> Result<OcctPreviewSessionChunkPayload>;

        fn close_cad_preview_session(session_id: u64);
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

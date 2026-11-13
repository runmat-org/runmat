use crate::import::CancellationProbe;
use sha2::Digest as _;
use std::collections::BTreeMap;
use std::sync::{
    atomic::{AtomicU64, Ordering},
    Mutex, OnceLock,
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

    #[repr(u8)]
    #[derive(Debug, Clone, Copy)]
    enum OcctHealingEntityKind {
        Vertex,
        Edge,
        Face,
    }

    #[repr(u8)]
    #[derive(Debug, Clone, Copy)]
    enum OcctFeatureKind {
        Box,
        Cylinder,
        Sphere,
        Transform,
        Instance,
        Extrude,
        Revolve,
        Union,
        Subtract,
        Intersect,
        Hole,
        Pocket,
        Fillet,
        Chamfer,
    }

    #[repr(u8)]
    #[derive(Debug, Clone, Copy)]
    enum OcctFeatureEntityKind {
        Vertex,
        Edge,
        Face,
        Solid,
    }

    #[repr(u8)]
    #[derive(Debug, Clone, Copy)]
    enum OcctFeatureRelationKind {
        Modified,
        Generated,
    }

    #[derive(Debug, Clone)]
    struct OcctFeatureOperand {
        representation: Vec<u8>,
    }

    #[derive(Debug, Clone)]
    struct OcctFeatureSelection {
        kind: OcctFeatureEntityKind,
        identity_digest: Vec<u8>,
    }

    #[derive(Debug, Clone)]
    struct OcctFeatureSourcePayload {
        operand_index: u64,
        kind: OcctFeatureEntityKind,
        identity_digest: Vec<u8>,
        deleted: bool,
    }

    #[derive(Debug, Clone)]
    struct OcctFeatureRelationPayload {
        operand_index: u64,
        source_kind: OcctFeatureEntityKind,
        source_identity_digest: Vec<u8>,
        result_kind: OcctFeatureEntityKind,
        result_shape_key: u64,
        relation: OcctFeatureRelationKind,
    }

    #[derive(Debug, Clone)]
    struct OcctFeatureExecutionPayload {
        representation: Vec<u8>,
        sources: Vec<OcctFeatureSourcePayload>,
        relations: Vec<OcctFeatureRelationPayload>,
    }

    #[derive(Debug, Clone)]
    struct OcctHealingRelationPayload {
        kind: OcctHealingEntityKind,
        path_segments: Vec<String>,
        source_digest: Vec<u8>,
        target_digest: Vec<u8>,
    }

    #[derive(Debug, Clone, Copy)]
    struct OcctStepContextUnitPayload {
        entity_id: u64,
        length_done: bool,
        meters_per_unit: f64,
    }

    #[derive(Debug, Clone, Copy)]
    struct OcctImportOptions {
        // Metres per coordinate unit requested for STEP XCAF transfer.
        step_coordinate_unit_m: f64,
        require_declared_step_units: bool,
        // Explicit interpretation of a unitless BREP coordinate.
        unitless_source_unit_m: f64,
        linear_deflection: f64,
        angular_deflection: f64,
        relative_deflection: bool,
        max_triangles: u64,
        truncate_at_max_triangles: bool,
        max_exact_representation_bytes: u64,
        max_exact_entities: u64,
        max_exact_identity_bytes: u64,
        heal_sew: bool,
        heal_orientation: bool,
        heal_duplicates: bool,
        heal_gaps: bool,
        heal_short_edges_and_sliver_faces: bool,
        /// Join touching solids so their coincident faces become shared interfaces.
        share_interfaces: bool,
        maximum_healing_displacement: f64,
        /// Largest curve/pcurve gap, in metres, that exact import admits by raising the edge's
        /// tolerance after rebuilding its pcurves: the request's geometric tolerance (requested
        /// deviation). Larger gaps stay defects. Unused by display-only imports.
        admissible_edge_gap_m: f64,
        cancel_token_id: u64,
    }

    /// One edge whose measured curve/pcurve gap exceeded its written tolerance: where it lies
    /// (curve start, parameter midpoint, end; kernel units), the gap before and after rebuilding
    /// its pcurves, and the tolerance import admitted (kernel units).
    #[derive(Debug, Clone)]
    struct OcctEdgeToleranceAdmissionPayload {
        start_x: f64,
        start_y: f64,
        start_z: f64,
        middle_x: f64,
        middle_y: f64,
        middle_z: f64,
        end_x: f64,
        end_y: f64,
        end_z: f64,
        written_tolerance: f64,
        gap_before: f64,
        gap_after: f64,
        admitted_tolerance: f64,
        repaired: bool,
        within_bound: bool,
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
        original_geometry_digest: Vec<u8>,
        orientation_repaired: bool,
        duplicates_consolidated: bool,
        original_kernel_valid: bool,
        post_duplicate_kernel_valid: bool,
        post_sewing_kernel_valid: bool,
        post_small_topology_kernel_valid: bool,
        healing_identity_work_bytes: u64,
        sewn: bool,
        gaps_repaired: bool,
        short_edges_simplified: bool,
        sliver_faces_simplified: bool,
        healing_relations: Vec<OcctHealingRelationPayload>,
        edge_tolerance_admissions: Vec<OcctEdgeToleranceAdmissionPayload>,
        maximum_healing_displacement: f64,
        displacement_original_x: f64,
        displacement_original_y: f64,
        displacement_original_z: f64,
        displacement_proposed_x: f64,
        displacement_proposed_y: f64,
        displacement_proposed_z: f64,
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
        lumps: Vec<OcctExactLumpPayload>,
        definitions: Vec<OcctExactDefinitionPayload>,
        occurrences: Vec<OcctExactOccurrencePayload>,
    }

    #[derive(Debug, Clone)]
    struct OcctExactDefinitionPayload {
        definition_index: u64,
        representation: Vec<u8>,
    }

    #[derive(Debug, Clone)]
    struct OcctExactOccurrencePayload {
        occurrence_index: u64,
        parent_occurrence_index: u64,
        path_segments: Vec<String>,
        definition_index: u64,
        transform: Vec<f64>,
    }

    #[derive(Debug, Clone)]
    struct OcctExactVertexPayload {
        occurrence_index: u64,
        shape_key: u64,
        identity_digest: Vec<u8>,
        point_x: f64,
        point_y: f64,
        point_z: f64,
        tolerance: f64,
        /// Authored names, in precedence order; selector metadata outside persistent identity.
        names: Vec<String>,
        /// Presentation layers (semantic tags); selector metadata outside persistent identity.
        tags: Vec<String>,
    }

    #[derive(Debug, Clone)]
    struct OcctExactEdgePayload {
        occurrence_index: u64,
        shape_key: u64,
        identity_digest: Vec<u8>,
        start_vertex_key: u64,
        end_vertex_key: u64,
        tolerance: f64,
        closed: bool,
        periodic: bool,
        degenerate: bool,
        /// Authored names, in precedence order; selector metadata outside persistent identity.
        names: Vec<String>,
        /// Presentation layers (semantic tags); selector metadata outside persistent identity.
        tags: Vec<String>,
    }

    #[derive(Debug, Clone)]
    struct OcctExactFacePayload {
        occurrence_index: u64,
        source_face_index: u64,
        shape_key: u64,
        identity_digest: Vec<u8>,
        reversed: bool,
        outer_wire_key: u64,
        inner_wire_keys: Vec<u64>,
        periodic_u: bool,
        periodic_v: bool,
        singular: bool,
        /// Authored names, in precedence order; selector metadata outside persistent identity.
        names: Vec<String>,
        /// Presentation layers (semantic tags); selector metadata outside persistent identity.
        tags: Vec<String>,
    }

    #[derive(Debug, Clone)]
    struct OcctExactWirePayload {
        occurrence_index: u64,
        shape_key: u64,
        identity_digest: Vec<u8>,
        face_key: u64,
        reversed: bool,
        coedge_keys: Vec<u64>,
    }

    #[derive(Debug, Clone)]
    struct OcctExactCoedgePayload {
        occurrence_index: u64,
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
        occurrence_index: u64,
        shape_key: u64,
        identity_digest: Vec<u8>,
        reversed: bool,
        face_keys: Vec<u64>,
        face_reversed: Vec<bool>,
    }

    #[derive(Debug, Clone)]
    struct OcctExactSolidPayload {
        occurrence_index: u64,
        shape_key: u64,
        identity_digest: Vec<u8>,
        outer_shell_key: u64,
        void_shell_keys: Vec<u64>,
        /// Authored solid or product names; selector metadata outside persistent identity.
        names: Vec<String>,
        /// Presentation layers (semantic tags); selector metadata outside persistent identity.
        tags: Vec<String>,
    }

    #[derive(Debug, Clone)]
    struct OcctExactLumpPayload {
        occurrence_index: u64,
        shape_key: u64,
        identity_digest: Vec<u8>,
        from_compsolid: bool,
        solid_keys: Vec<u64>,
    }

    #[derive(Debug, Clone, Copy)]
    struct OcctCurveRangePayload {
        start: f64,
        end: f64,
    }

    #[derive(Debug, Clone, Copy)]
    struct OcctCurveDerivativesPayload {
        parameter_unit: u8,
        parameter_scale: f64,
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
        parameter_unit: u8,
        parameter_scale: f64,
        parameter: f64,
        point_x: f64,
        point_y: f64,
        point_z: f64,
        distance: f64,
    }

    #[derive(Debug, Clone, Copy)]
    struct OcctPcurveDerivativesPayload {
        parameter_unit: u8,
        u_parameter_unit: u8,
        v_parameter_unit: u8,
        parameter_scale: f64,
        range_start: f64,
        range_end: f64,
        point_u: f64,
        point_v: f64,
        first_u: f64,
        first_v: f64,
        second_u: f64,
        second_v: f64,
    }

    #[derive(Debug, Clone, Copy)]
    struct OcctSurfacePropertiesPayload {
        u_start: f64,
        u_end: f64,
        v_start: f64,
        v_end: f64,
        u_periodic: bool,
        u_period: f64,
        v_periodic: bool,
        v_period: f64,
    }

    #[derive(Debug, Clone, Copy)]
    struct OcctSurfaceDerivativesPayload {
        u_parameter_unit: u8,
        v_parameter_unit: u8,
        parameter_scale: f64,
        point_x: f64,
        point_y: f64,
        point_z: f64,
        du_x: f64,
        du_y: f64,
        du_z: f64,
        dv_x: f64,
        dv_y: f64,
        dv_z: f64,
        duu_x: f64,
        duu_y: f64,
        duu_z: f64,
        duv_x: f64,
        duv_y: f64,
        duv_z: f64,
        dvv_x: f64,
        dvv_y: f64,
        dvv_z: f64,
    }

    #[derive(Debug, Clone, Copy)]
    struct OcctSurfaceProjectionPayload {
        u_parameter_unit: u8,
        v_parameter_unit: u8,
        parameter_scale: f64,
        u: f64,
        v: f64,
        point_x: f64,
        point_y: f64,
        point_z: f64,
        distance: f64,
    }

    #[derive(Debug, Clone, Copy)]
    struct OcctMassPropertiesPayload {
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
        include!("src/occt/occt_bridge.hxx");

        fn exact_kernel_abi() -> String;

        fn import_cad_bytes(
            path: &str,
            bytes: &[u8],
            materialized_path: &str,
            format: OcctCadFormat,
            options: OcctImportOptions,
        ) -> Result<OcctImportPayload>;

        fn import_exact_cad_bytes(
            path: &str,
            bytes: &[u8],
            materialized_path: &str,
            format: OcctCadFormat,
            options: OcctImportOptions,
        ) -> Result<OcctExactShapePayload>;

        fn execute_exact_feature(
            kind: OcctFeatureKind,
            scalars: &[f64],
            matrices: &[f64],
            selections: Vec<OcctFeatureSelection>,
            operands: Vec<OcctFeatureOperand>,
            options: OcctImportOptions,
        ) -> Result<OcctFeatureExecutionPayload>;

        fn start_exact_evaluator_session(
            representation: &[u8],
            meters_per_source_unit: f64,
            face_keys: &[u64],
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
            face_reversed: bool,
            wire_key: u64,
            coedge_position: u64,
            seam_image: i8,
            parameter: f64,
        ) -> Result<OcctPcurveDerivativesPayload>;

        fn exact_pcurve_range(
            session_id: u64,
            face_key: u64,
            face_reversed: bool,
            wire_key: u64,
            coedge_position: u64,
            seam_image: i8,
        ) -> Result<OcctCurveRangePayload>;

        fn exact_trim_classify(
            session_id: u64,
            face_key: u64,
            u: f64,
            v: f64,
            boundary_tolerance_uv: f64,
        ) -> Result<i8>;

        fn exact_surface_properties(
            session_id: u64,
            face_key: u64,
        ) -> Result<OcctSurfacePropertiesPayload>;

        fn exact_surface_derivatives(
            session_id: u64,
            face_key: u64,
            u: f64,
            v: f64,
        ) -> Result<OcctSurfaceDerivativesPayload>;

        fn exact_surface_closest_point(
            session_id: u64,
            face_key: u64,
            point_m: &[f64],
            absolute_error_m: f64,
        ) -> Result<OcctSurfaceProjectionPayload>;

        fn exact_mass_properties(
            session_id: u64,
            shape_keys: &[u64],
            is_sheet_body: bool,
        ) -> Result<OcctMassPropertiesPayload>;

        fn close_exact_evaluator_session(session_id: u64);

        fn start_cad_preview_session(
            path: &str,
            bytes: &[u8],
            materialized_path: &str,
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
        fn occt_exact_identity_digest(bytes: &[u8]) -> Vec<u8>;
        fn occt_geometry_digest(bytes: &[u8]) -> Vec<u8>;
        fn occt_validate_step_source_units(
            bytes: &[u8],
            contexts: &[OcctStepContextUnitPayload],
            require_declared: bool,
        ) -> Result<()>;
    }
}

static NEXT_CANCEL_TOKEN_ID: AtomicU64 = AtomicU64::new(1);
type CancelTokenMap = BTreeMap<u64, CancellationProbe>;
static CANCEL_TOKENS: OnceLock<Mutex<CancelTokenMap>> = OnceLock::new();

pub(crate) struct OcctCancelTokenRegistration {
    id: u64,
}

impl OcctCancelTokenRegistration {
    pub(crate) fn new(flag: Option<CancellationProbe>) -> Self {
        let Some(flag) = flag else {
            return Self { id: 0 };
        };
        let id = NEXT_CANCEL_TOKEN_ID.fetch_add(1, Ordering::Relaxed);
        lock_cancel_tokens(cancel_tokens()).insert(id, flag);
        Self { id }
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
        // Release the probe after the registry lock; its drop may run caller code.
        let retired = lock_cancel_tokens(cancel_tokens()).remove(&self.id);
        drop(retired);
    }
}

fn occt_import_cancelled(cancel_token_id: u64) -> bool {
    if cancel_token_id == 0 {
        return false;
    }
    // Clone under the lock, then probe outside it: probes may re-enter the registry.
    let probe = lock_cancel_tokens(cancel_tokens())
        .get(&cancel_token_id)
        .cloned();
    probe.is_some_and(|is_cancelled| is_cancelled())
}

fn occt_exact_identity_digest(bytes: &[u8]) -> Vec<u8> {
    let mut hasher = sha2::Sha256::new();
    sha2::Digest::update(&mut hasher, b"runmat.occt-persistent-shape-name\0");
    sha2::Digest::update(&mut hasher, bytes);
    sha2::Digest::finalize(hasher).to_vec()
}

fn occt_geometry_digest(bytes: &[u8]) -> Vec<u8> {
    sha2::Sha256::digest(bytes).to_vec()
}

fn cancel_tokens() -> &'static Mutex<CancelTokenMap> {
    CANCEL_TOKENS.get_or_init(|| Mutex::new(BTreeMap::new()))
}

fn lock_cancel_tokens(
    registry: &Mutex<CancelTokenMap>,
) -> std::sync::MutexGuard<'_, CancelTokenMap> {
    registry.lock().unwrap_or_else(|poisoned| {
        // Integer-keyed map mutations preserve Rust's map invariants on unwind.
        // Keep the original token inventory; poisoning cannot disable cancellation.
        registry.clear_poison();
        poisoned.into_inner()
    })
}

fn occt_validate_step_source_units(
    bytes: &[u8],
    contexts: &[bridge::OcctStepContextUnitPayload],
    require_declared: bool,
) -> Result<(), String> {
    use runmat_geometry_core::physical::{LengthUnit, QuantityKind, UnitDefinition};
    let mut checked = Vec::with_capacity(contexts.len());
    for context in contexts {
        let unit = if context.length_done {
            let definition = UnitDefinition::try_new(
                QuantityKind::Length.dimension(),
                context.meters_per_unit,
                0.0,
            )
            .map_err(|error| {
                format!(
                    "invalid OCCT source-unit factor in STEP #{}: {error}",
                    context.entity_id
                )
            })?;
            Some(LengthUnit::try_from_definition(definition).map_err(|error| error.to_string())?)
        } else {
            None
        };
        checked.push(crate::source_units::StepContextUnits {
            entity_id: context.entity_id,
            unit,
        });
    }
    crate::source_units::verify_step_context_units(bytes, &checked, require_declared)
}

#[cfg(test)]
mod control_tests;

#[cfg(test)]
mod cancellation_tests {
    use super::*;
    use std::sync::Arc;

    #[test]
    fn probe_runs_outside_registry_lock_and_registration_drop_removes_it() {
        let probe: CancellationProbe = Arc::new(|| {
            drop(cancel_tokens().lock().unwrap());
            true
        });
        let registration = OcctCancelTokenRegistration::new(Some(probe.clone()));
        let id = registration.id();
        assert_ne!(id, 0);
        assert!(occt_import_cancelled(id));
        assert_eq!(Arc::strong_count(&probe), 2);
        drop(registration);
        assert_eq!(Arc::strong_count(&probe), 1);
        assert!(!occt_import_cancelled(id));
    }
}

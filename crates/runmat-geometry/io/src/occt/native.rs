use super::{
    exact_healing_projection, exact_projection, ffi, import_validation, topology_from_raw,
    OcctCadFormat, OcctCadPreviewSessionChunk, OcctCadPreviewSessionStart, OcctCadTopology,
    OcctRawAssemblyNode, OcctRawFaceEvaluationSample, OcctRawFaceSemantic, OcctRawTopology,
};
use crate::exact::{ExactCadImportOptions, ImportedExactCad};
use crate::import::{
    GeometryImportBudgetPolicy, GeometryImportContext, GeometryImportError, GeometryImportOptions,
};
use runmat_geometry_core::{BodyMassProperties, GeometryDigest, GeometrySourceFormat, UnitSystem};
use sha2::{Digest, Sha256};
use std::sync::atomic::{AtomicBool, Ordering};

const DEFAULT_LINEAR_DEFLECTION: f64 = 0.01;
const DEFAULT_ANGULAR_DEFLECTION: f64 = 0.5;
const OCCT_IMPORT_CANCELLED_MESSAGE: &str = "OCCT CAD import cancelled";
static NATIVE_CAD_BACKEND_USED: AtomicBool = AtomicBool::new(false);

pub(crate) fn native_cad_backend_was_used() -> bool {
    NATIVE_CAD_BACKEND_USED.load(Ordering::Relaxed)
}

pub(crate) fn import_cad_topology(
    path: &str,
    bytes: &[u8],
    format: OcctCadFormat,
    options: &GeometryImportOptions,
    context: &GeometryImportContext,
) -> Result<OcctCadTopology, GeometryImportError> {
    NATIVE_CAD_BACKEND_USED.store(true, Ordering::Relaxed);
    context.check_cancelled()?;
    let cancel_token = ffi::OcctCancelTokenRegistration::new(context.cancellation_flag());
    let payload = ffi::bridge::import_cad_bytes(
        path,
        bytes,
        ffi_format(format),
        ffi_import_options(options, cancel_token.id()),
    )
    .map_err(|err| occt_bridge_error("OCCT CAD import failed", err))?;
    context.check_cancelled()?;

    payload_to_topology(payload, options, context)
}

pub(crate) fn import_exact_cad_shape(
    path: &str,
    bytes: &[u8],
    format: OcctCadFormat,
    options: &ExactCadImportOptions,
    context: &GeometryImportContext,
) -> Result<ImportedExactCad, GeometryImportError> {
    NATIVE_CAD_BACKEND_USED.store(true, Ordering::Relaxed);
    context.check_cancelled()?;
    let meters_per_source_unit = meters_per_unit(options.source_units)?;
    let cancel_token = ffi::OcctCancelTokenRegistration::new(context.cancellation_flag());
    let payload = ffi::bridge::import_exact_cad_bytes(
        path,
        bytes,
        ffi_format(format),
        ffi_exact_import_options(options, cancel_token.id(), meters_per_source_unit),
    )
    .map_err(|err| exact_bridge_error(err, options))?;
    context.check_cancelled()?;
    if !payload.kernel_valid {
        return Err(GeometryImportError::InvalidGeometry(
            "OCCT BRepCheck rejected the imported exact shape".into(),
        ));
    }
    if payload.face_count == 0
        || payload.wire_count == 0
        || payload.edge_count == 0
        || payload.vertex_count == 0
    {
        return Err(GeometryImportError::InvalidGeometry(
            "OCCT exact shape has incomplete boundary topology".into(),
        ));
    }
    if payload.solid_count > 0 && !payload.has_volume_properties {
        return Err(GeometryImportError::InvalidGeometry(
            "OCCT exact solid has non-positive or non-finite oriented volume".into(),
        ));
    }
    let measured_values = [
        payload.volume,
        payload.surface_area,
        payload.centroid_x,
        payload.centroid_y,
        payload.centroid_z,
        payload.inertia_xx,
        payload.inertia_yy,
        payload.inertia_zz,
        payload.inertia_xy,
        payload.inertia_xz,
        payload.inertia_yz,
    ];
    if payload.surface_area <= 0.0 || measured_values.iter().any(|value| !value.is_finite()) {
        return Err(GeometryImportError::InvalidGeometry(
            "OCCT exact shape produced invalid mass-property evidence".into(),
        ));
    }
    let mass_properties = payload.has_volume_properties.then(|| {
        let length2 = meters_per_source_unit * meters_per_source_unit;
        let length3 = length2 * meters_per_source_unit;
        let length5 = length3 * length2;
        BodyMassProperties {
            volume_m3: payload.volume * length3,
            surface_area_m2: payload.surface_area * length2,
            centroid_m: [
                payload.centroid_x * meters_per_source_unit,
                payload.centroid_y * meters_per_source_unit,
                payload.centroid_z * meters_per_source_unit,
            ],
            inertia_about_centroid_m5: [
                payload.inertia_xx * length5,
                payload.inertia_yy * length5,
                payload.inertia_zz * length5,
                payload.inertia_xy * length5,
                payload.inertia_xz * length5,
                payload.inertia_yz * length5,
            ],
        }
    });
    let projection = exact_projection::project_exact_contracts(
        &payload,
        meters_per_source_unit,
        mass_properties.as_ref(),
    )?;
    let orientation_repaired = payload.orientation_repaired;
    let duplicates_consolidated = payload.duplicates_consolidated;
    let original_geometry_digest = payload.original_geometry_digest.clone();
    let original_kernel_valid = payload.original_kernel_valid;
    let post_duplicate_kernel_valid = payload.post_duplicate_kernel_valid;
    let post_sewing_kernel_valid = payload.post_sewing_kernel_valid;
    let post_small_topology_kernel_valid = payload.post_small_topology_kernel_valid;
    let sewn = payload.sewn;
    let gaps_repaired = payload.gaps_repaired;
    let short_edges_simplified = payload.short_edges_simplified;
    let sliver_faces_simplified = payload.sliver_faces_simplified;
    let healing_relations = payload.healing_relations.clone();
    let maximum_healing_displacement_m =
        payload.maximum_healing_displacement * meters_per_source_unit;
    let displacement_original_m = [
        payload.displacement_original_x,
        payload.displacement_original_y,
        payload.displacement_original_z,
    ]
    .map(|coordinate| coordinate * meters_per_source_unit);
    let displacement_proposed_m = [
        payload.displacement_proposed_x,
        payload.displacement_proposed_y,
        payload.displacement_proposed_z,
    ]
    .map(|coordinate| coordinate * meters_per_source_unit);
    let mut imported = ImportedExactCad {
        source_digest: GeometryDigest::from_bytes(Sha256::digest(bytes).into()),
        source_format: match format {
            OcctCadFormat::Step => GeometrySourceFormat::Step,
            OcctCadFormat::Iges => GeometrySourceFormat::Iges,
            OcctCadFormat::Brep => GeometrySourceFormat::Brep,
        },
        source_units: options.source_units,
        kernel_version: payload.kernel_version,
        meters_per_source_unit,
        representation: payload.representation,
        topology: projection.topology,
        evaluators: projection.evaluators,
        model: projection.model,
        healing_report: None,
        analysis: options.analysis.clone(),
        kernel_body_shapes: projection.kernel_body_shapes,
    };
    let tolerance_m = imported
        .topology
        .vertices
        .iter()
        .map(|vertex| vertex.tolerance_m)
        .fold(0.0_f64, f64::max)
        .max(f64::EPSILON);
    let evaluator = super::evaluator::OcctExactEvaluator::new(&imported)
        .map_err(import_validation::map_validation_error)?;
    evaluator
        .validate_incidence_consistency(
            &imported.topology,
            tolerance_m,
            &import_validation::ImportEvaluationControl::new(context, options),
        )
        .map_err(import_validation::map_validation_error)?;
    if orientation_repaired
        || duplicates_consolidated
        || sewn
        || gaps_repaired
        || short_edges_simplified
        || sliver_faces_simplified
    {
        let report = exact_healing_projection::healing_report(
            exact_healing_projection::NativeHealingEvidence {
                original_digest: &original_geometry_digest,
                original_kernel_valid,
                post_duplicate_kernel_valid,
                duplicates_consolidated,
                orientation_repaired,
                sewn,
                gaps_repaired,
                post_sewing_kernel_valid,
                short_edges_simplified,
                sliver_faces_simplified,
                post_small_topology_kernel_valid,
                relations: &healing_relations,
                maximum_displacement_m: maximum_healing_displacement_m,
                displacement_original_m,
                displacement_proposed_m,
            },
            &imported,
        )?;
        imported.analysis.revision = report.revision_map.target_revision.clone();
        imported.healing_report = Some(report);
    } else if !original_geometry_digest.is_empty() {
        return Err(GeometryImportError::InvalidGeometry(
            "OCCT returned original geometry identity without a healing operation".into(),
        ));
    }
    Ok(imported)
}

fn meters_per_unit(units: UnitSystem) -> Result<f64, GeometryImportError> {
    match units {
        UnitSystem::Meter => Ok(1.0),
        UnitSystem::Millimeter => Ok(0.001),
        UnitSystem::Inch => Ok(0.0254),
        UnitSystem::Unspecified => Err(GeometryImportError::InvalidOptions(
            "exact CAD import requires explicit source units".into(),
        )),
    }
}

pub(crate) fn start_cad_preview_session(
    path: &str,
    bytes: &[u8],
    format: OcctCadFormat,
    options: &GeometryImportOptions,
    context: &GeometryImportContext,
) -> Result<OcctCadPreviewSessionStart, GeometryImportError> {
    NATIVE_CAD_BACKEND_USED.store(true, Ordering::Relaxed);
    context.check_cancelled()?;
    let cancel_token = ffi::OcctCancelTokenRegistration::new(context.cancellation_flag());
    let payload = ffi::bridge::start_cad_preview_session(
        path,
        bytes,
        ffi_format(format),
        ffi_import_options(options, cancel_token.id()),
    )
    .map_err(|err| occt_bridge_error("OCCT CAD preview session failed", err))?;
    context.check_cancelled()?;
    Ok(OcctCadPreviewSessionStart {
        session_id: payload.session_id,
        face_count: payload.face_count,
    })
}

pub(crate) fn read_cad_preview_session_chunk(
    session_id: u64,
    target_triangles: u64,
    max_faces: u64,
    options: &GeometryImportOptions,
    context: &GeometryImportContext,
) -> Result<OcctCadPreviewSessionChunk, GeometryImportError> {
    NATIVE_CAD_BACKEND_USED.store(true, Ordering::Relaxed);
    context.check_cancelled()?;
    let cancel_token = ffi::OcctCancelTokenRegistration::new(context.cancellation_flag());
    let payload = ffi::bridge::read_cad_preview_session_chunk(
        session_id,
        ffi::bridge::OcctPreviewSessionChunkOptions {
            target_triangles,
            max_faces,
            cancel_token_id: cancel_token.id(),
        },
    )
    .map_err(|err| occt_bridge_error("OCCT CAD preview session failed", err))?;
    context.check_cancelled()?;
    let topology = payload_to_topology(payload.topology, options, context)?;
    Ok(OcctCadPreviewSessionChunk {
        session_id: payload.session_id,
        done: payload.done,
        face_cursor: payload.face_cursor,
        face_count: payload.face_count,
        topology,
    })
}

pub(crate) fn close_cad_preview_session(session_id: u64) {
    ffi::bridge::close_cad_preview_session(session_id);
}

fn occt_bridge_error(operation: &str, err: impl std::fmt::Display) -> GeometryImportError {
    let message = err.to_string();
    if message.contains(OCCT_IMPORT_CANCELLED_MESSAGE) {
        GeometryImportError::Cancelled
    } else {
        GeometryImportError::ParseFailed(format!("{operation}: {message}"))
    }
}

fn exact_bridge_error(
    err: impl std::fmt::Display,
    options: &ExactCadImportOptions,
) -> GeometryImportError {
    let message = err.to_string();
    if message.contains(OCCT_IMPORT_CANCELLED_MESSAGE) {
        GeometryImportError::Cancelled
    } else if message.contains("exact representation exceeded") {
        GeometryImportError::ExactRepresentationCapacityExceeded {
            limit: options.max_representation_bytes,
        }
    } else if message.contains("exact topology exceeded") {
        GeometryImportError::ExactEntityCapacityExceeded {
            limit: options.max_entities,
        }
    } else if message.contains("exact persistent identity exceeded") {
        GeometryImportError::ExactValidationBudgetExceeded(format!(
            "persistent identity serialization exceeded {} bytes of work",
            options.max_identity_work_bytes
        ))
    } else if message.contains("requires definition-aware XCAF mutation") {
        GeometryImportError::BackendUnavailable(message)
    } else if message.contains("OCCT exact outer shell")
        || message.contains("OCCT exact void shell")
        || message.contains("OCCT exact solid shell has no nesting witness")
    {
        GeometryImportError::InvalidGeometry(message)
    } else {
        GeometryImportError::ParseFailed(format!("OCCT exact CAD import failed: {message}"))
    }
}

fn payload_to_topology(
    payload: ffi::bridge::OcctImportPayload,
    options: &GeometryImportOptions,
    context: &GeometryImportContext,
) -> Result<OcctCadTopology, GeometryImportError> {
    topology_from_raw(
        OcctRawTopology {
            backend: payload.backend,
            format_name: payload.format_name,
            truncated: payload.truncated,
            triangle_budget: payload.triangle_budget,
            vertices: payload.vertices,
            triangles: payload.triangles,
            triangle_face_ids: payload.triangle_face_ids,
            face_ids: payload.face_ids,
            face_names: payload.face_names,
            face_semantics: payload
                .face_semantics
                .into_iter()
                .map(|item| OcctRawFaceSemantic {
                    face_id: item.face_id,
                    label_entry: item.label_entry,
                    label_name: item.label_name,
                    label_kind: item.label_kind,
                    owner_entries: item.owner_entries,
                    owner_names: item.owner_names,
                    owner_kinds: item.owner_kinds,
                    layer_names: item.layer_names,
                    color_type: item.color_type,
                    color_hex_rgba: item.color_hex_rgba,
                    material_label_entry: item.material_label_entry,
                    material_name: item.material_name,
                    material_description: item.material_description,
                    material_density: item.material_density,
                    material_density_name: item.material_density_name,
                    material_density_value_type: item.material_density_value_type,
                })
                .collect(),
            face_evaluation_samples: payload
                .face_evaluation_samples
                .into_iter()
                .map(|item| OcctRawFaceEvaluationSample {
                    face_id: item.face_id,
                    u: item.u,
                    v: item.v,
                    point_m: [item.point_x, item.point_y, item.point_z],
                    unit_normal: [item.normal_x, item.normal_y, item.normal_z],
                    projection_error_m: item.projection_error,
                })
                .collect(),
            assembly_nodes: payload
                .assembly_nodes
                .into_iter()
                .map(|item| OcctRawAssemblyNode {
                    node_id: item.node_id,
                    parent_node_id: item.parent_node_id,
                    label: item.label,
                })
                .collect(),
            warnings: payload.warnings,
        },
        options,
        context,
    )
}

fn ffi_import_options(
    options: &GeometryImportOptions,
    cancel_token_id: u64,
) -> ffi::bridge::OcctImportOptions {
    let linear_deflection = options
        .tessellation_profile
        .chord_tolerance
        .filter(|value| value.is_finite() && *value > 0.0)
        .unwrap_or(DEFAULT_LINEAR_DEFLECTION);
    let angular_deflection = options
        .tessellation_profile
        .angle_tolerance_deg
        .filter(|value| value.is_finite() && *value > 0.0)
        .map(f64::to_radians)
        .unwrap_or(DEFAULT_ANGULAR_DEFLECTION);
    ffi::bridge::OcctImportOptions {
        linear_deflection,
        angular_deflection,
        relative_deflection: options.relative_deflection,
        max_triangles: options.max_triangles.unwrap_or(u64::MAX),
        truncate_at_max_triangles: options.budget_policy == GeometryImportBudgetPolicy::Truncate,
        max_exact_representation_bytes: u64::MAX,
        max_exact_entities: u64::MAX,
        max_exact_identity_bytes: u64::MAX,
        heal_sew: false,
        heal_orientation: false,
        heal_duplicates: false,
        heal_gaps: false,
        heal_short_edges_and_sliver_faces: false,
        maximum_healing_displacement: 0.0,
        cancel_token_id,
    }
}

fn ffi_exact_import_options(
    options: &ExactCadImportOptions,
    cancel_token_id: u64,
    meters_per_source_unit: f64,
) -> ffi::bridge::OcctImportOptions {
    ffi::bridge::OcctImportOptions {
        linear_deflection: DEFAULT_LINEAR_DEFLECTION,
        angular_deflection: DEFAULT_ANGULAR_DEFLECTION,
        relative_deflection: false,
        max_triangles: u64::MAX,
        truncate_at_max_triangles: false,
        max_exact_representation_bytes: options.max_representation_bytes,
        max_exact_entities: options.max_entities,
        max_exact_identity_bytes: options.max_identity_work_bytes,
        heal_sew: options.analysis.healing.sew,
        heal_orientation: options.analysis.healing.repair_orientation,
        heal_duplicates: options.analysis.healing.consolidate_duplicates,
        heal_gaps: options.analysis.healing.repair_tolerance_scale_gaps,
        heal_short_edges_and_sliver_faces: options
            .analysis
            .healing
            .simplify_short_edges_and_sliver_faces,
        maximum_healing_displacement: options.analysis.maximum_healing_displacement_m
            / meters_per_source_unit,
        cancel_token_id,
    }
}

fn ffi_format(format: OcctCadFormat) -> ffi::bridge::OcctCadFormat {
    match format {
        OcctCadFormat::Step => ffi::bridge::OcctCadFormat::Step,
        OcctCadFormat::Iges => ffi::bridge::OcctCadFormat::Iges,
        OcctCadFormat::Brep => ffi::bridge::OcctCadFormat::Brep,
    }
}

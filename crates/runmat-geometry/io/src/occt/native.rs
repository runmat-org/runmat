use super::{
    ffi, topology_from_raw, OcctCadFormat, OcctCadPreviewSessionChunk, OcctCadPreviewSessionStart,
    OcctCadTopology, OcctRawAssemblyNode, OcctRawFaceEvaluationSample, OcctRawFaceSemantic,
    OcctRawTopology,
};
use crate::exact::{ExactCadImportOptions, ExactCadKernelShape, ExactCadTopologyInventory};
use crate::import::{
    GeometryImportBudgetPolicy, GeometryImportContext, GeometryImportError, GeometryImportOptions,
};
use runmat_geometry_core::{BodyMassProperties, UnitSystem};
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
) -> Result<ExactCadKernelShape, GeometryImportError> {
    NATIVE_CAD_BACKEND_USED.store(true, Ordering::Relaxed);
    context.check_cancelled()?;
    let meters_per_source_unit = meters_per_unit(options.source_units)?;
    let cancel_token = ffi::OcctCancelTokenRegistration::new(context.cancellation_flag());
    let payload = ffi::bridge::import_exact_cad_bytes(
        path,
        bytes,
        ffi_format(format),
        ffi_exact_import_options(options, cancel_token.id()),
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
    Ok(ExactCadKernelShape {
        kernel_version: payload.kernel_version,
        kernel_abi: payload.kernel_abi,
        representation: payload.representation,
        topology: ExactCadTopologyInventory {
            compound_count: payload.compound_count,
            compsolid_count: payload.compsolid_count,
            solid_count: payload.solid_count,
            shell_count: payload.shell_count,
            face_count: payload.face_count,
            wire_count: payload.wire_count,
            edge_count: payload.edge_count,
            vertex_count: payload.vertex_count,
        },
        mass_properties,
    })
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
        cancel_token_id,
    }
}

fn ffi_exact_import_options(
    options: &ExactCadImportOptions,
    cancel_token_id: u64,
) -> ffi::bridge::OcctImportOptions {
    ffi::bridge::OcctImportOptions {
        linear_deflection: DEFAULT_LINEAR_DEFLECTION,
        angular_deflection: DEFAULT_ANGULAR_DEFLECTION,
        relative_deflection: false,
        max_triangles: u64::MAX,
        truncate_at_max_triangles: false,
        max_exact_representation_bytes: options.max_representation_bytes,
        max_exact_entities: options.max_entities,
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

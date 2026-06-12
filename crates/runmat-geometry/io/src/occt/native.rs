use super::{
    ffi, topology_from_raw, OcctCadFormat, OcctCadTopology, OcctRawAssemblyNode,
    OcctRawFaceSemantic, OcctRawTopology,
};
use crate::import::{GeometryImportError, GeometryImportOptions};

const DEFAULT_LINEAR_DEFLECTION: f64 = 0.01;
const DEFAULT_ANGULAR_DEFLECTION: f64 = 0.5;

pub(crate) fn import_cad_topology(
    path: &str,
    bytes: &[u8],
    format: OcctCadFormat,
    options: &GeometryImportOptions,
) -> Result<OcctCadTopology, GeometryImportError> {
    let payload = ffi::ffi::import_cad_bytes(
        path,
        bytes,
        ffi_format(format),
        ffi::ffi::OcctImportOptions {
            linear_deflection: DEFAULT_LINEAR_DEFLECTION,
            angular_deflection: DEFAULT_ANGULAR_DEFLECTION,
            relative_deflection: false,
            max_triangles: options.max_triangles.unwrap_or(u64::MAX),
        },
    )
    .map_err(|err| GeometryImportError::ParseFailed(format!("OCCT CAD import failed: {err}")))?;

    topology_from_raw(
        OcctRawTopology {
            backend: payload.backend,
            format_name: payload.format_name,
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
    )
}

fn ffi_format(format: OcctCadFormat) -> ffi::ffi::OcctCadFormat {
    match format {
        OcctCadFormat::Step => ffi::ffi::OcctCadFormat::Step,
        OcctCadFormat::Iges => ffi::ffi::OcctCadFormat::Iges,
        OcctCadFormat::Brep => ffi::ffi::OcctCadFormat::Brep,
    }
}

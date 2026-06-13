use crate::{
    import::{GeometryImportContext, GeometryImportError, GeometryImportOptions},
    sniff::GeometryFormat,
};
use runmat_geometry_core::{AssemblyNode, CadRegionOwnership};
#[cfg(any(
    all(not(target_arch = "wasm32"), feature = "occt-native"),
    all(target_arch = "wasm32", feature = "occt-wasm-host")
))]
use runmat_geometry_core::{
    CadColorEvidence, CadLabelRef, CadPhysicalMaterialEvidence, CadSemanticKind,
};
#[cfg(any(
    all(not(target_arch = "wasm32"), feature = "occt-native"),
    all(target_arch = "wasm32", feature = "occt-wasm-host")
))]
use std::collections::BTreeMap;

#[cfg(all(not(target_arch = "wasm32"), feature = "occt-native"))]
mod ffi;
#[cfg(all(not(target_arch = "wasm32"), feature = "occt-native"))]
mod native;
#[cfg(all(target_arch = "wasm32", feature = "occt-wasm-host"))]
mod wasm;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum OcctCadFormat {
    Step,
    Iges,
    Brep,
}

impl OcctCadFormat {
    pub(crate) fn from_geometry_format(format: GeometryFormat) -> Option<Self> {
        match format {
            GeometryFormat::Step => Some(Self::Step),
            GeometryFormat::Iges => Some(Self::Iges),
            GeometryFormat::Brep => Some(Self::Brep),
            _ => None,
        }
    }

    pub(crate) fn as_str(self) -> &'static str {
        match self {
            Self::Step => "step",
            Self::Iges => "iges",
            Self::Brep => "brep",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct OcctCadFace {
    pub face_id: u64,
    pub name: String,
    pub ownership: Option<CadRegionOwnership>,
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct OcctCadTopology {
    pub backend: String,
    pub format_name: String,
    pub truncated: bool,
    pub triangle_budget: Option<u64>,
    pub vertices: Vec<[f64; 3]>,
    pub triangles: Vec<[u32; 3]>,
    pub triangle_face_ids: Vec<u64>,
    pub faces: Vec<OcctCadFace>,
    pub assembly: Option<AssemblyNode>,
    pub warnings: Vec<String>,
}

#[cfg(any(
    all(not(target_arch = "wasm32"), feature = "occt-native"),
    all(target_arch = "wasm32", feature = "occt-wasm-host")
))]
pub(crate) struct OcctRawTopology {
    pub backend: String,
    pub format_name: String,
    pub truncated: bool,
    pub triangle_budget: u64,
    pub vertices: Vec<f64>,
    pub triangles: Vec<u32>,
    pub triangle_face_ids: Vec<u64>,
    pub face_ids: Vec<u64>,
    pub face_names: Vec<String>,
    pub face_semantics: Vec<OcctRawFaceSemantic>,
    pub assembly_nodes: Vec<OcctRawAssemblyNode>,
    pub warnings: Vec<String>,
}

#[cfg(any(
    all(not(target_arch = "wasm32"), feature = "occt-native"),
    all(target_arch = "wasm32", feature = "occt-wasm-host")
))]
pub(crate) struct OcctRawFaceSemantic {
    pub face_id: u64,
    pub label_entry: String,
    pub label_name: String,
    pub label_kind: String,
    pub owner_entries: Vec<String>,
    pub owner_names: Vec<String>,
    pub owner_kinds: Vec<String>,
    pub layer_names: Vec<String>,
    pub color_type: String,
    pub color_hex_rgba: String,
    pub material_label_entry: String,
    pub material_name: String,
    pub material_description: String,
    pub material_density: String,
    pub material_density_name: String,
    pub material_density_value_type: String,
}

#[cfg(any(
    all(not(target_arch = "wasm32"), feature = "occt-native"),
    all(target_arch = "wasm32", feature = "occt-wasm-host")
))]
pub(crate) struct OcctRawAssemblyNode {
    pub node_id: String,
    pub parent_node_id: String,
    pub label: String,
}

#[cfg(all(not(target_arch = "wasm32"), feature = "occt-native"))]
pub(crate) fn import_cad_topology(
    path: &str,
    bytes: &[u8],
    format: OcctCadFormat,
    options: &GeometryImportOptions,
    context: &GeometryImportContext,
) -> Result<Option<OcctCadTopology>, GeometryImportError> {
    native::import_cad_topology(path, bytes, format, options, context).map(Some)
}

#[cfg(all(target_arch = "wasm32", feature = "occt-wasm-host"))]
pub(crate) fn import_cad_topology(
    path: &str,
    bytes: &[u8],
    format: OcctCadFormat,
    options: &GeometryImportOptions,
    context: &GeometryImportContext,
) -> Result<Option<OcctCadTopology>, GeometryImportError> {
    wasm::import_cad_topology(path, bytes, format, options, context).map(Some)
}

#[cfg(not(any(
    all(not(target_arch = "wasm32"), feature = "occt-native"),
    all(target_arch = "wasm32", feature = "occt-wasm-host")
)))]
pub(crate) fn import_cad_topology(
    path: &str,
    bytes: &[u8],
    format: OcctCadFormat,
    options: &GeometryImportOptions,
    context: &GeometryImportContext,
) -> Result<Option<OcctCadTopology>, GeometryImportError> {
    let _ = (path, bytes, format, options, context);
    Ok(None)
}

#[cfg(any(
    all(not(target_arch = "wasm32"), feature = "occt-native"),
    all(target_arch = "wasm32", feature = "occt-wasm-host")
))]
pub(crate) fn topology_from_raw(
    payload: OcctRawTopology,
    options: &GeometryImportOptions,
    context: &GeometryImportContext,
) -> Result<OcctCadTopology, GeometryImportError> {
    context.check_cancelled()?;
    if !payload.vertices.len().is_multiple_of(3) {
        return Err(GeometryImportError::ParseFailed(
            "OCCT backend returned a vertex buffer whose length is not divisible by 3".to_string(),
        ));
    }
    if !payload.triangles.len().is_multiple_of(3) {
        return Err(GeometryImportError::ParseFailed(
            "OCCT backend returned an index buffer whose length is not divisible by 3".to_string(),
        ));
    }

    let triangle_count = payload.triangles.len() / 3;
    if payload.triangle_face_ids.len() != triangle_count {
        return Err(GeometryImportError::ParseFailed(format!(
            "OCCT backend returned {} face ids for {triangle_count} triangles",
            payload.triangle_face_ids.len()
        )));
    }

    crate::import::capacity_guard(triangle_count as u64, options)?;

    let vertices = payload
        .vertices
        .chunks_exact(3)
        .enumerate()
        .map(|(index, chunk)| {
            crate::import::check_cancelled_periodic(context, index)?;
            Ok([chunk[0], chunk[1], chunk[2]])
        })
        .collect::<Result<Vec<_>, GeometryImportError>>()?;

    context.check_cancelled()?;
    if vertices
        .iter()
        .flatten()
        .any(|coordinate| !coordinate.is_finite())
    {
        return Err(GeometryImportError::ParseFailed(
            "OCCT backend returned a non-finite vertex coordinate".to_string(),
        ));
    }

    let triangles = payload
        .triangles
        .chunks_exact(3)
        .enumerate()
        .map(|(index, chunk)| {
            crate::import::check_cancelled_periodic(context, index)?;
            Ok([chunk[0], chunk[1], chunk[2]])
        })
        .collect::<Result<Vec<_>, GeometryImportError>>()?;
    let vertex_count = vertices.len();
    if triangles
        .iter()
        .flatten()
        .any(|index| *index as usize >= vertex_count)
    {
        return Err(GeometryImportError::ParseFailed(
            "OCCT backend returned a triangle index outside the vertex buffer".to_string(),
        ));
    }

    context.check_cancelled()?;
    let ownership_by_face_id = face_ownership_by_id(payload.face_semantics, context)?;
    let faces = payload
        .face_ids
        .iter()
        .enumerate()
        .map(|(index, face_id)| {
            crate::import::check_cancelled_periodic(context, index)?;
            let fallback = format!("Face {}", index + 1);
            Ok(OcctCadFace {
                face_id: *face_id,
                name: payload
                    .face_names
                    .get(index)
                    .filter(|name| !name.trim().is_empty())
                    .cloned()
                    .unwrap_or(fallback),
                ownership: ownership_by_face_id.get(face_id).cloned(),
            })
        })
        .collect::<Result<Vec<_>, GeometryImportError>>()?;
    let assembly = build_assembly(payload.assembly_nodes, context)?;

    Ok(OcctCadTopology {
        backend: payload.backend,
        format_name: payload.format_name,
        truncated: payload.truncated,
        triangle_budget: payload.truncated.then_some(payload.triangle_budget),
        vertices,
        triangles,
        triangle_face_ids: payload.triangle_face_ids,
        faces,
        assembly,
        warnings: payload.warnings,
    })
}

#[cfg(any(
    all(not(target_arch = "wasm32"), feature = "occt-native"),
    all(target_arch = "wasm32", feature = "occt-wasm-host")
))]
fn face_ownership_by_id(
    rows: Vec<OcctRawFaceSemantic>,
    context: &GeometryImportContext,
) -> Result<BTreeMap<u64, CadRegionOwnership>, GeometryImportError> {
    let mut ownership = BTreeMap::new();
    for (index, row) in rows.into_iter().enumerate() {
        crate::import::check_cancelled_periodic(context, index)?;
        if row.owner_entries.len() != row.owner_names.len()
            || row.owner_entries.len() != row.owner_kinds.len()
        {
            return Err(GeometryImportError::ParseFailed(
                "OCCT backend returned inconsistent CAD owner path arrays".to_string(),
            ));
        }

        let label = cad_label_ref(row.label_entry, row.label_name, row.label_kind);
        let owner_path = row
            .owner_entries
            .into_iter()
            .zip(row.owner_names)
            .zip(row.owner_kinds)
            .filter_map(|((entry, name), kind)| cad_label_ref(entry, name, kind))
            .collect();
        let color = (!row.color_hex_rgba.trim().is_empty()).then(|| CadColorEvidence {
            source: "occt_xcaf".to_string(),
            color_type: row.color_type,
            hex_rgba: row.color_hex_rgba,
        });
        let material =
            (!row.material_name.trim().is_empty()).then(|| CadPhysicalMaterialEvidence {
                label_entry: row.material_label_entry,
                name: row.material_name,
                description: non_empty(row.material_description),
                density: non_empty(row.material_density),
                density_name: non_empty(row.material_density_name),
                density_value_type: non_empty(row.material_density_value_type),
            });

        let entry = CadRegionOwnership {
            face_id: Some(row.face_id),
            label,
            owner_path,
            layers: row.layer_names.into_iter().filter_map(non_empty).collect(),
            color,
            material,
        };
        ownership.insert(row.face_id, entry);
    }
    Ok(ownership)
}

#[cfg(any(
    all(not(target_arch = "wasm32"), feature = "occt-native"),
    all(target_arch = "wasm32", feature = "occt-wasm-host")
))]
fn cad_label_ref(entry: String, name: String, kind: String) -> Option<CadLabelRef> {
    let label_entry = non_empty(entry)?;
    Some(CadLabelRef {
        label_entry,
        name: non_empty(name).unwrap_or_default(),
        kind: parse_cad_kind(&kind),
    })
}

#[cfg(any(
    all(not(target_arch = "wasm32"), feature = "occt-native"),
    all(target_arch = "wasm32", feature = "occt-wasm-host")
))]
fn parse_cad_kind(kind: &str) -> CadSemanticKind {
    match kind {
        "assembly" => CadSemanticKind::Assembly,
        "component" => CadSemanticKind::Component,
        "reference" => CadSemanticKind::Reference,
        "body" => CadSemanticKind::Body,
        "compound" => CadSemanticKind::Compound,
        "face" => CadSemanticKind::Face,
        "subshape" => CadSemanticKind::Subshape,
        "layer" => CadSemanticKind::Layer,
        "color" => CadSemanticKind::Color,
        "material" => CadSemanticKind::Material,
        "shape" => CadSemanticKind::Shape,
        _ => CadSemanticKind::Unknown,
    }
}

#[cfg(any(
    all(not(target_arch = "wasm32"), feature = "occt-native"),
    all(target_arch = "wasm32", feature = "occt-wasm-host")
))]
fn build_assembly(
    rows: Vec<OcctRawAssemblyNode>,
    context: &GeometryImportContext,
) -> Result<Option<AssemblyNode>, GeometryImportError> {
    if rows.is_empty() {
        return Ok(None);
    }

    let mut by_parent = BTreeMap::<String, Vec<OcctRawAssemblyNode>>::new();
    for (index, row) in rows.into_iter().enumerate() {
        crate::import::check_cancelled_periodic(context, index)?;
        by_parent
            .entry(row.parent_node_id.clone())
            .or_default()
            .push(row);
    }
    for children in by_parent.values_mut() {
        children.sort_by(|left, right| left.node_id.cmp(&right.node_id));
    }

    let mut roots = build_assembly_children("", &mut by_parent, 0, context)?;
    if roots.len() == 1 {
        Ok(roots.pop())
    } else {
        Ok(Some(AssemblyNode {
            node_id: "assembly_root".to_string(),
            label: "CAD Assembly".to_string(),
            children: roots,
        }))
    }
}

#[cfg(any(
    all(not(target_arch = "wasm32"), feature = "occt-native"),
    all(target_arch = "wasm32", feature = "occt-wasm-host")
))]
fn build_assembly_children(
    parent: &str,
    by_parent: &mut BTreeMap<String, Vec<OcctRawAssemblyNode>>,
    depth: usize,
    context: &GeometryImportContext,
) -> Result<Vec<AssemblyNode>, GeometryImportError> {
    if depth > 256 {
        return Err(GeometryImportError::ParseFailed(
            "OCCT backend returned an assembly tree deeper than 256 levels".to_string(),
        ));
    }

    let Some(rows) = by_parent.remove(parent) else {
        return Ok(Vec::new());
    };
    rows.into_iter()
        .enumerate()
        .map(|(index, row)| {
            crate::import::check_cancelled_periodic(context, index)?;
            let children = build_assembly_children(&row.node_id, by_parent, depth + 1, context)?;
            Ok(AssemblyNode {
                node_id: row.node_id,
                label: if row.label.trim().is_empty() {
                    "Unnamed CAD Node".to_string()
                } else {
                    row.label
                },
                children,
            })
        })
        .collect()
}

#[cfg(any(
    all(not(target_arch = "wasm32"), feature = "occt-native"),
    all(target_arch = "wasm32", feature = "occt-wasm-host")
))]
fn non_empty(value: String) -> Option<String> {
    let trimmed = value.trim();
    (!trimmed.is_empty()).then(|| trimmed.to_string())
}

//! Exact CAD import boundary. This path returns kernel B-rep bytes and kernel-derived evidence;
//! it never exposes or consumes display tessellation.

use runmat_geometry_core::{BodyMassProperties, UnitSystem};

use crate::{
    import::{GeometryImportContext, GeometryImportError},
    occt::{self, OcctCadFormat},
    GeometryFormat,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ExactCadTopologyInventory {
    pub compound_count: u64,
    pub compsolid_count: u64,
    pub solid_count: u64,
    pub shell_count: u64,
    pub face_count: u64,
    pub wire_count: u64,
    pub edge_count: u64,
    pub vertex_count: u64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ExactCadKernelShape {
    pub kernel_version: String,
    pub kernel_abi: String,
    /// Canonical OCCT B-rep with all derived polygonal caches removed.
    pub representation: Vec<u8>,
    pub topology: ExactCadTopologyInventory,
    /// Present for solid-bearing shapes. Sheet-only geometry has no volume properties.
    pub mass_properties: Option<BodyMassProperties>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ExactCadImportOptions {
    pub source_units: UnitSystem,
    /// Hard bound applied before the kernel representation crosses the FFI boundary.
    pub max_representation_bytes: u64,
    /// Hard per-kind bound for compounds, solids, shells, faces, wires, edges, and vertices.
    pub max_entities: u64,
}

impl Default for ExactCadImportOptions {
    fn default() -> Self {
        Self {
            source_units: UnitSystem::Meter,
            max_representation_bytes: 512 * 1024 * 1024,
            max_entities: 10_000_000,
        }
    }
}

pub fn import_exact_cad(
    source_name: &str,
    bytes: &[u8],
    format: GeometryFormat,
    options: &ExactCadImportOptions,
    context: &GeometryImportContext,
) -> Result<ExactCadKernelShape, GeometryImportError> {
    context.check_cancelled()?;
    if bytes.is_empty() {
        return Err(GeometryImportError::ParseFailed(
            "exact CAD payload is empty".into(),
        ));
    }
    if options.max_representation_bytes == 0 || options.max_entities == 0 {
        return Err(GeometryImportError::InvalidOptions(
            "exact representation and entity budgets must be nonzero".into(),
        ));
    }
    let format = OcctCadFormat::from_geometry_format(format)
        .ok_or(GeometryImportError::UnsupportedFormat)?;
    occt::import_exact_cad_shape(source_name, bytes, format, options, context)
}

#[cfg(test)]
mod tests;

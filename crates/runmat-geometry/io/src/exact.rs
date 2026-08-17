//! Exact CAD import boundary. This path returns kernel B-rep bytes and kernel-derived evidence;
//! it never exposes or consumes display tessellation.

use runmat_geometry_core::{ExactBRepTopology, ExactEvaluatorRegistry, UnitSystem};

use crate::{
    import::{GeometryImportContext, GeometryImportError},
    occt::{self, OcctCadFormat},
    GeometryFormat,
};

#[derive(Debug, Clone, PartialEq)]
pub struct ImportedExactCad {
    pub kernel_version: String,
    /// Canonical OCCT B-rep with all derived polygonal caches removed.
    pub representation: Vec<u8>,
    /// Authoritative topology extracted directly from the kernel B-rep.
    pub topology: ExactBRepTopology,
    /// Exact evaluator bindings into `representation`; no display samples are admitted.
    pub evaluators: ExactEvaluatorRegistry,
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
) -> Result<ImportedExactCad, GeometryImportError> {
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

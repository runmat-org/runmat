//! Exact CAD import boundary. This path returns kernel B-rep bytes and kernel-derived evidence;
//! it never exposes or consumes display tessellation.

use std::collections::BTreeMap;

use runmat_geometry_core::{
    build_exact_geometry_closure, EncodedExactGeometryClosure, ExactBRepModel, ExactBRepTopology,
    ExactEvaluatorRegistry, GeometryDigest, GeometryDocument, GeometryHealingPolicy, GeometryModel,
    GeometryRevisionIdentity, GeometrySourceFormat, GeometrySourceIdentity,
    GeometryTolerancePolicy, UnitSystem, GEOMETRY_DOCUMENT_SCHEMA_VERSION,
};
use sha2::{Digest, Sha256};

use crate::{
    import::{GeometryImportContext, GeometryImportError},
    occt::{self, OcctCadFormat},
    GeometryFormat,
};

#[derive(Debug, Clone, PartialEq)]
pub struct ImportedExactCad {
    pub source_digest: GeometryDigest,
    pub source_format: GeometrySourceFormat,
    pub source_units: UnitSystem,
    pub kernel_version: String,
    pub meters_per_source_unit: f64,
    /// Canonical OCCT B-rep with all derived polygonal caches removed.
    pub representation: Vec<u8>,
    /// Authoritative topology extracted directly from the kernel B-rep.
    pub topology: ExactBRepTopology,
    /// Exact evaluator bindings into `representation`; no display samples are admitted.
    pub evaluators: ExactEvaluatorRegistry,
    pub model: ExactBRepModel,
    /// Kernel-private shape handles for body evaluators. Semantic identity remains exclusively in
    /// the topology; these ordinals never enter serialized geometry contracts.
    pub(crate) kernel_body_shapes:
        BTreeMap<runmat_geometry_core::MassPropertiesEvaluatorId, Vec<u64>>,
}

impl ImportedExactCad {
    pub fn representation_digest(&self) -> [u8; 32] {
        exact_representation_digest(&self.representation)
    }

    pub fn build_closure(
        &self,
        options: &ExactCadClosureOptions,
    ) -> Result<EncodedExactGeometryClosure, runmat_geometry_core::GeometryContractError> {
        let tolerance = GeometryTolerancePolicy {
            source_tolerance_m: self
                .topology
                .vertices
                .iter()
                .map(|vertex| vertex.tolerance_m)
                .fold(0.0_f64, f64::max),
            absolute_floor_m: options.absolute_tolerance_floor_m,
            model_relative_term: options.model_relative_tolerance,
            requested_deviation_m: options.requested_deviation_m,
            maximum_healing_displacement_m: options.maximum_healing_displacement_m,
        };
        let document = GeometryDocument {
            schema_version: GEOMETRY_DOCUMENT_SCHEMA_VERSION,
            source: GeometrySourceIdentity {
                content_digest: self.source_digest,
                format: self.source_format,
                importer_version: "runmat-exact-cad-import/1".into(),
                kernel_version: Some(self.kernel_version.clone()),
                source_units: self.source_units,
                meters_per_source_unit: self.meters_per_source_unit,
            },
            revision: options.revision.clone(),
            tolerance,
            healing: options.healing.clone(),
            model: GeometryModel::ExactBRep {
                model: self.model.clone(),
            },
            display_tessellations: Vec::new(),
        };
        build_exact_geometry_closure(
            document,
            &self.topology,
            &self.evaluators,
            Some(&self.representation),
            None,
        )
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct ExactCadClosureOptions {
    pub revision: GeometryRevisionIdentity,
    pub absolute_tolerance_floor_m: f64,
    pub model_relative_tolerance: f64,
    pub requested_deviation_m: f64,
    pub maximum_healing_displacement_m: f64,
    pub healing: GeometryHealingPolicy,
}

pub(crate) fn exact_representation_digest(representation: &[u8]) -> [u8; 32] {
    Sha256::digest(representation).into()
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ExactCadImportOptions {
    pub source_units: UnitSystem,
    /// Hard aggregate bound for the kernel representation and unique assembly-definition
    /// identity evidence before either crosses the FFI boundary.
    pub max_representation_bytes: u64,
    /// Hard per-kind and expanded-occurrence aggregate topology bound.
    pub max_entities: u64,
    /// Hard aggregate byte-work bound for canonical standalone subshape serialization used only
    /// to derive persistent names. Serialized subshapes are hashed and discarded immediately.
    pub max_identity_work_bytes: u64,
    /// Hard work bounds for independent exact-incidence admission after kernel import.
    pub max_validation_iterations: u64,
    pub max_validation_search_work: u64,
    pub max_validation_allocation_bytes: u64,
}

impl Default for ExactCadImportOptions {
    fn default() -> Self {
        Self {
            source_units: UnitSystem::Meter,
            max_representation_bytes: 512 * 1024 * 1024,
            max_entities: 10_000_000,
            max_identity_work_bytes: 2 * 1024 * 1024 * 1024,
            max_validation_iterations: 10_000_000,
            max_validation_search_work: 100_000_000,
            max_validation_allocation_bytes: 512 * 1024 * 1024,
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
    if options.max_representation_bytes == 0
        || options.max_entities == 0
        || options.max_identity_work_bytes == 0
        || options.max_validation_iterations == 0
        || options.max_validation_search_work == 0
        || options.max_validation_allocation_bytes == 0
    {
        return Err(GeometryImportError::InvalidOptions(
            "exact representation, entity, identity-work, and validation budgets must be nonzero"
                .into(),
        ));
    }
    let format = OcctCadFormat::from_geometry_format(format)
        .ok_or(GeometryImportError::UnsupportedFormat)?;
    occt::import_exact_cad_shape(source_name, bytes, format, options, context)
}

#[cfg(test)]
mod tests;

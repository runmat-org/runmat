//! Exact CAD import boundary. This path returns kernel B-rep bytes and kernel-derived evidence;
//! it never exposes or consumes display tessellation.

use std::collections::BTreeMap;

use runmat_geometry_core::{
    author_exact_contacts, build_exact_geometry_closure, EncodedExactGeometryClosure,
    ExactBRepModel, ExactBRepTopology, ExactContactDefinition, ExactEvaluatorRegistry,
    GeometryDigest, GeometryDocument, GeometryHealingPolicy, GeometryHealingReport, GeometryModel,
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
    /// Present only when the kernel changed authoritative topology under the admitted policy.
    pub healing_report: Option<GeometryHealingReport>,
    /// Kernel-authored mapping from display-import face ordinals to exact persistent faces.
    /// This selector metadata does not enter canonical geometry identity.
    pub source_face_ids: BTreeMap<u64, runmat_geometry_core::PersistentEntityId>,
    /// Analysis identity and mutation policy admitted before the kernel import. Keeping this on
    /// the imported object prevents closure construction from claiming a different tolerance or
    /// healing policy than the one under which topology was produced.
    pub(crate) analysis: ExactCadAnalysisOptions,
    /// Kernel-private shape handles for body evaluators. Semantic identity remains exclusively in
    /// the topology; these ordinals never enter serialized geometry contracts.
    pub(crate) kernel_body_shapes:
        BTreeMap<runmat_geometry_core::MassPropertiesEvaluatorId, Vec<u64>>,
}

impl ImportedExactCad {
    pub fn analysis_options(&self) -> &ExactCadAnalysisOptions {
        &self.analysis
    }

    pub fn representation_digest(&self) -> [u8; 32] {
        exact_representation_digest(&self.representation)
    }

    /// Replaces the analysis contact model by resolving explicit persistent source-face sides.
    /// This does not re-run the kernel or infer contact from geometric proximity.
    pub fn with_contacts(
        mut self,
        definitions: &[ExactContactDefinition],
    ) -> Result<Self, runmat_geometry_core::GeometryContractError> {
        self.topology.contacts = author_exact_contacts(&self.topology, definitions)?;
        self.model.contact_count = self.topology.contacts.len() as u64;
        self.topology.validate_against(&self.model)?;
        Ok(self)
    }

    pub fn geometry_document(
        &self,
    ) -> Result<GeometryDocument, runmat_geometry_core::GeometryContractError> {
        let tolerance = GeometryTolerancePolicy {
            source_tolerance_m: self
                .topology
                .vertices
                .iter()
                .map(|vertex| vertex.tolerance_m)
                .fold(0.0_f64, f64::max),
            absolute_floor_m: self.analysis.absolute_tolerance_floor_m,
            model_relative_term: self.analysis.model_relative_tolerance,
            requested_deviation_m: self.analysis.requested_deviation_m,
            maximum_healing_displacement_m: self.analysis.maximum_healing_displacement_m,
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
            revision: self.analysis.revision.clone(),
            tolerance,
            healing: self.analysis.healing.clone(),
            model: GeometryModel::ExactBRep {
                model: self.model.clone(),
            },
            display_tessellations: Vec::new(),
        };
        document.validate()?;
        Ok(document)
    }

    pub fn build_closure(
        &self,
    ) -> Result<EncodedExactGeometryClosure, runmat_geometry_core::GeometryContractError> {
        build_exact_geometry_closure(
            self.geometry_document()?,
            &self.topology,
            &self.evaluators,
            Some(&self.representation),
            self.healing_report.as_ref(),
        )
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct ExactCadAnalysisOptions {
    pub revision: GeometryRevisionIdentity,
    pub absolute_tolerance_floor_m: f64,
    pub model_relative_tolerance: f64,
    pub requested_deviation_m: f64,
    pub maximum_healing_displacement_m: f64,
    pub healing: GeometryHealingPolicy,
}

impl Default for ExactCadAnalysisOptions {
    fn default() -> Self {
        Self {
            revision: GeometryRevisionIdentity {
                revision: 1,
                persistent_mapping_version: 1,
                parent_document_digest: None,
            },
            absolute_tolerance_floor_m: 1.0e-12,
            model_relative_tolerance: 1.0e-12,
            requested_deviation_m: 1.0e-4,
            maximum_healing_displacement_m: 1.0e-6,
            healing: GeometryHealingPolicy {
                algorithm_version: "occt-healing/1".into(),
                sew: false,
                repair_orientation: false,
                consolidate_duplicates: false,
                repair_tolerance_scale_gaps: false,
                simplify_short_edges_and_sliver_faces: false,
            },
        }
    }
}

pub(crate) fn exact_representation_digest(representation: &[u8]) -> [u8; 32] {
    Sha256::digest(representation).into()
}

#[derive(Debug, Clone, PartialEq)]
pub struct ExactCadImportOptions {
    pub source_units: UnitSystem,
    /// Geometry identity, tolerance, and healing authority for this import. These settings are
    /// inseparable from the exact topology produced by the kernel.
    pub analysis: ExactCadAnalysisOptions,
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
            analysis: ExactCadAnalysisOptions::default(),
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
    let tolerance = GeometryTolerancePolicy {
        source_tolerance_m: 0.0,
        absolute_floor_m: options.analysis.absolute_tolerance_floor_m,
        model_relative_term: options.analysis.model_relative_tolerance,
        requested_deviation_m: options.analysis.requested_deviation_m,
        maximum_healing_displacement_m: options.analysis.maximum_healing_displacement_m,
    };
    tolerance
        .validate()
        .map_err(|error| GeometryImportError::InvalidOptions(error.to_string()))?;
    options
        .analysis
        .healing
        .validate()
        .map_err(|error| GeometryImportError::InvalidOptions(error.to_string()))?;
    let topology_changing_modes = u8::from(options.analysis.healing.sew)
        + u8::from(options.analysis.healing.consolidate_duplicates)
        + u8::from(options.analysis.healing.repair_tolerance_scale_gaps)
        + u8::from(
            options
                .analysis
                .healing
                .simplify_short_edges_and_sliver_faces,
        );
    if topology_changing_modes > 1 {
        return Err(GeometryImportError::InvalidOptions(
            "sewing, duplicate consolidation, gap repair, and small-topology repair must be requested as separate geometry revisions".into(),
        ));
    }
    if (options.analysis.healing.repair_tolerance_scale_gaps
        || options
            .analysis
            .healing
            .simplify_short_edges_and_sliver_faces)
        && options.analysis.maximum_healing_displacement_m <= 0.0
    {
        return Err(GeometryImportError::InvalidOptions(
            "gap or small-topology repair requires a positive maximum healing displacement".into(),
        ));
    }
    if options.analysis.revision.revision == 0
        || options.analysis.revision.persistent_mapping_version == 0
    {
        return Err(GeometryImportError::InvalidOptions(
            "exact geometry revision and persistent mapping version must be nonzero".into(),
        ));
    }
    let format = OcctCadFormat::from_geometry_format(format)
        .ok_or(GeometryImportError::UnsupportedFormat)?;
    occt::import_exact_cad_shape(source_name, bytes, format, options, context)
}

#[cfg(test)]
mod tests;

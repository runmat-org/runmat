//! Exact CAD import boundary. This path returns kernel B-rep bytes and kernel-derived evidence;
//! it never exposes or consumes display tessellation.

use std::collections::BTreeMap;

use runmat_geometry_core::{
    author_exact_contacts, build_exact_geometry_closure, EncodedExactGeometryClosure,
    ExactBRepModel, ExactBRepTopology, ExactContactDefinition, ExactEvaluatorRegistry,
    GeometryDigest, GeometryDocument, GeometryHealingPolicy, GeometryHealingReport, GeometryModel,
    GeometryRevisionIdentity, GeometrySourceFormat, GeometrySourceIdentity,
    GeometryTolerancePolicy, LengthUnit, GEOMETRY_DOCUMENT_SCHEMA_VERSION,
};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::{
    import::{GeometryImportContext, GeometryImportError},
    occt::{self, OcctCadFormat},
    GeometryFormat,
};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ImportedExactCad {
    pub source_digest: GeometryDigest,
    pub source_format: GeometrySourceFormat,
    /// File-declared contexts or an explicitly applied unitless-source fallback.
    pub source_unit_evidence: runmat_geometry_core::physical::LengthUnitEvidence,
    pub kernel_version: String,
    /// Scale of the retained kernel representation, independent of the authored fallback.
    /// STEP and IGES are retained in millimetres; unitless BREP uses `source_units`.
    pub kernel_units: runmat_geometry_core::physical::CoordinateUnits<
        runmat_geometry_core::physical::KernelCoordinates,
    >,
    /// Canonical OCCT B-rep with all derived polygonal caches removed.
    #[serde(with = "representation_wire")]
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
    /// Exact result-table join used only while translating kernel feature history. Shape-table
    /// keys never become semantic identity and never cross the geometry contract boundary.
    #[serde(with = "kernel_entity_id_wire")]
    pub(crate) kernel_entity_ids: BTreeMap<
        (u64, runmat_geometry_core::PersistentEntityKind, u64),
        runmat_geometry_core::PersistentEntityId,
    >,
}

/// Resolve source evidence and retained kernel coordinate scale once for every host.
/// A browser host response must agree with independently inspected input bytes.
#[cfg(any(
    feature = "occt-native",
    all(target_arch = "wasm32", feature = "occt-wasm-host"),
    test
))]
pub(crate) fn exact_import_units(
    path: &str,
    bytes: &[u8],
    format: OcctCadFormat,
    options: &ExactCadImportOptions,
) -> Result<
    (
        runmat_geometry_core::physical::LengthUnitEvidence,
        runmat_geometry_core::physical::CoordinateUnits<
            runmat_geometry_core::physical::KernelCoordinates,
        >,
    ),
    GeometryImportError,
> {
    let evidence = crate::source_units::inspect_source_units(path, bytes, format)
        .resolve_fallback(options.source_units, "explicit exact-import source units");
    evidence
        .validate()
        .map_err(|e| GeometryImportError::InvalidOptions(e.to_string()))?;
    if !evidence.physical_scale_resolved() {
        return Err(GeometryImportError::InvalidOptions("CAD source units are missing, unsupported or conflicting; explicit fallback units are admitted only for unitless formats".into()));
    }
    let kernel_unit = match format {
        OcctCadFormat::Step | OcctCadFormat::Iges => LengthUnit::Millimeter,
        OcctCadFormat::Brep => evidence.uniform_unit().ok_or_else(|| {
            GeometryImportError::InvalidOptions("BREP source length unit is required".into())
        })?,
    };
    Ok((
        evidence,
        runmat_geometry_core::physical::CoordinateUnits::new(kernel_unit),
    ))
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
            source_tolerance_m: runmat_types::physical::Length::new(
                self.topology
                    .vertices
                    .iter()
                    .map(|vertex| vertex.tolerance_m.canonical_value())
                    .fold(0.0_f64, f64::max),
            ),
            absolute_floor_m: runmat_types::physical::Length::new(
                self.analysis.absolute_tolerance_floor_m.canonical_value(),
            ),
            model_relative_term: self.analysis.model_relative_tolerance,
            requested_deviation_m: runmat_types::physical::Length::new(
                self.analysis.requested_deviation_m.canonical_value(),
            ),
            maximum_healing_displacement_m: runmat_types::physical::Length::new(
                self.analysis
                    .maximum_healing_displacement_m
                    .canonical_value(),
            ),
        };
        let document = GeometryDocument {
            schema_version: GEOMETRY_DOCUMENT_SCHEMA_VERSION,
            source: GeometrySourceIdentity {
                content_digest: self.source_digest,
                format: self.source_format,
                importer_version: "runmat-exact-cad-import/1".into(),
                kernel_version: Some(self.kernel_version.clone()),
                unit_evidence: self.source_unit_evidence.clone(),
                representation_units: self.kernel_units,
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

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ExactCadAnalysisOptions {
    pub revision: GeometryRevisionIdentity,
    #[serde(with = "runmat_types::physical::canonical_scalar")]
    pub absolute_tolerance_floor_m: runmat_types::physical::Length,
    pub model_relative_tolerance: f64,
    #[serde(with = "runmat_types::physical::canonical_scalar")]
    pub requested_deviation_m: runmat_types::physical::Length,
    #[serde(with = "runmat_types::physical::canonical_scalar")]
    pub maximum_healing_displacement_m: runmat_types::physical::Length,
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
            absolute_tolerance_floor_m: runmat_types::physical::Length::new(1.0e-12),
            model_relative_tolerance: 1.0e-12,
            requested_deviation_m: runmat_types::physical::Length::new(1.0e-4),
            maximum_healing_displacement_m: runmat_types::physical::Length::new(1.0e-6),
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

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ExactCadImportOptions {
    pub source_units: Option<LengthUnit>,
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
            source_units: None,
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

impl ExactCadImportOptions {
    pub fn validate(&self) -> Result<(), GeometryImportError> {
        if self.max_representation_bytes == 0
            || self.max_entities == 0
            || self.max_identity_work_bytes == 0
            || self.max_validation_iterations == 0
            || self.max_validation_search_work == 0
            || self.max_validation_allocation_bytes == 0
        {
            return Err(GeometryImportError::InvalidOptions(
                "exact representation, entity, identity-work, and validation budgets must be nonzero"
                    .into(),
            ));
        }
        let tolerance = GeometryTolerancePolicy {
            source_tolerance_m: runmat_types::physical::Length::new(0.0),
            absolute_floor_m: runmat_types::physical::Length::new(
                self.analysis.absolute_tolerance_floor_m.canonical_value(),
            ),
            model_relative_term: self.analysis.model_relative_tolerance,
            requested_deviation_m: runmat_types::physical::Length::new(
                self.analysis.requested_deviation_m.canonical_value(),
            ),
            maximum_healing_displacement_m: runmat_types::physical::Length::new(
                self.analysis
                    .maximum_healing_displacement_m
                    .canonical_value(),
            ),
        };
        tolerance
            .validate()
            .map_err(|error| GeometryImportError::InvalidOptions(error.to_string()))?;
        self.analysis
            .healing
            .validate()
            .map_err(|error| GeometryImportError::InvalidOptions(error.to_string()))?;
        let topology_changing_modes = u8::from(self.analysis.healing.sew)
            + u8::from(self.analysis.healing.consolidate_duplicates)
            + u8::from(self.analysis.healing.repair_tolerance_scale_gaps)
            + u8::from(self.analysis.healing.simplify_short_edges_and_sliver_faces);
        if topology_changing_modes > 1 {
            return Err(GeometryImportError::InvalidOptions(
                "sewing, duplicate consolidation, gap repair, and small-topology repair must be requested as separate geometry revisions".into(),
            ));
        }
        if (self.analysis.healing.repair_tolerance_scale_gaps
            || self.analysis.healing.simplify_short_edges_and_sliver_faces)
            && self
                .analysis
                .maximum_healing_displacement_m
                .canonical_value()
                <= 0.0
        {
            return Err(GeometryImportError::InvalidOptions(
                "gap or small-topology repair requires a positive maximum healing displacement"
                    .into(),
            ));
        }
        if self.analysis.revision.revision == 0
            || self.analysis.revision.persistent_mapping_version == 0
        {
            return Err(GeometryImportError::InvalidOptions(
                "exact geometry revision and persistent mapping version must be nonzero".into(),
            ));
        }
        Ok(())
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
    options.validate()?;
    let format = OcctCadFormat::from_geometry_format(format)
        .ok_or(GeometryImportError::UnsupportedFormat)?;
    occt::import_exact_cad_shape(source_name, bytes, format, options, context)
}

#[cfg(test)]
mod tests;

// JSON object keys cannot represent the kernel-private composite join key.
// Keep every entry when transferring the validated native import to a WASM host.
mod kernel_entity_id_wire {
    use super::*;
    type Key = (u64, runmat_geometry_core::PersistentEntityKind, u64);
    type Id = runmat_geometry_core::PersistentEntityId;
    pub fn serialize<S: serde::Serializer>(
        value: &BTreeMap<Key, Id>,
        serializer: S,
    ) -> Result<S::Ok, S::Error> {
        value.iter().collect::<Vec<_>>().serialize(serializer)
    }
    pub fn deserialize<'de, D: serde::Deserializer<'de>>(
        deserializer: D,
    ) -> Result<BTreeMap<Key, Id>, D::Error> {
        let entries = Vec::<(Key, Id)>::deserialize(deserializer)?;
        let count = entries.len();
        let result: BTreeMap<_, _> = entries.into_iter().collect();
        if result.len() != count {
            return Err(serde::de::Error::custom("duplicate kernel entity identity"));
        }
        Ok(result)
    }
}

mod representation_wire {
    use base64::{engine::general_purpose::STANDARD, Engine};
    use serde::{Deserialize, Serialize};
    pub fn serialize<S: serde::Serializer>(value: &[u8], serializer: S) -> Result<S::Ok, S::Error> {
        STANDARD.encode(value).serialize(serializer)
    }
    pub fn deserialize<'de, D: serde::Deserializer<'de>>(
        deserializer: D,
    ) -> Result<Vec<u8>, D::Error> {
        STANDARD
            .decode(String::deserialize(deserializer)?)
            .map_err(serde::de::Error::custom)
    }
}

#[cfg(test)]
mod unit_resolution_tests {
    use super::*;
    #[test]
    fn unitless_exact_source_requires_an_explicit_interpretation() {
        let mut options = ExactCadImportOptions::default();
        assert!(exact_import_units(
            "part.brep",
            b"DBRep_DrawableShape",
            OcctCadFormat::Brep,
            &options
        )
        .is_err());
        options.source_units = Some(LengthUnit::Millimeter);
        let (source, kernel) = exact_import_units(
            "part.brep",
            b"DBRep_DrawableShape",
            OcctCadFormat::Brep,
            &options,
        )
        .unwrap();
        assert_eq!(source.uniform_unit(), Some(LengthUnit::Millimeter));
        assert_eq!(kernel.unit(), LengthUnit::Millimeter);
        assert!(matches!(
            source,
            runmat_geometry_core::physical::LengthUnitEvidence::ExplicitFallback { .. }
        ));
    }
    #[test]
    fn absent_step_context_cannot_be_replaced_by_a_requested_fallback_unit() {
        let mut options = ExactCadImportOptions::default();
        options.source_units = Some(LengthUnit::Meter);
        let source =
            b"ISO-10303-21;HEADER;FILE_SCHEMA(('AP242'));ENDSEC;DATA;ENDSEC;END-ISO-10303-21;";
        assert!(
            exact_import_units("unitless.step", source, OcctCadFormat::Step, &options).is_err()
        );
    }
}

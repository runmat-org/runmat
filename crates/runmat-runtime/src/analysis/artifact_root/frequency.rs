//! Complete storage inventory of a shared frequency-sample publication.
//!
//! Solver results, receipts, state and resource closure remain independently authenticated
//! study outputs. This index owns only publication metadata and its field/observation roots.
use runmat_analysis_fea::execution::{
    FrequencySamplePublicationAuthority, FrequencySamplePublicationManifest,
};
use runmat_execution::{value::PublishedObjectCatalog, Digest};

use super::*;
use crate::analysis::storage::AnalysisFieldPageLookup;
use crate::analysis::{AnalysisFieldRequestOptions, AnalysisFieldSelection};

/// Borrowed authenticated semantic and physical authorities; no numerical arrays are copied.
#[derive(Clone, Copy)]
pub struct FrequencySamplePublicationAccess<'a> {
    pub publication: &'a FrequencySamplePublicationManifest,
    pub authority: &'a FrequencySamplePublicationAuthority,
    pub object_catalog: &'a PublishedObjectCatalog,
}

#[derive(Clone, Debug, Eq, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FrequencySampleArtifactRootInventory {
    publication: AnalysisArtifactRoot,
    samples: Vec<FrequencySampleArtifactRoots>,
}

/// Semantic authority and the complete physical manifest inventory admitted together.
/// Numerical partitions are authenticated lazily by the shared page reader.
pub struct AdmittedFrequencySamplePublication<'a> {
    inventory: &'a FrequencySampleArtifactRootInventory,
    access: FrequencySamplePublicationAccess<'a>,
}

impl FrequencySampleArtifactRootInventory {
    pub fn new(
        publication: AnalysisArtifactRoot,
        samples: Vec<FrequencySampleArtifactRoots>,
        access: &FrequencySamplePublicationAccess<'_>,
    ) -> Result<Self, String> {
        let inventory = Self {
            publication,
            samples,
        };
        inventory.validate(access)?;
        Ok(inventory)
    }

    pub const fn publication(&self) -> &AnalysisArtifactRoot {
        &self.publication
    }

    pub fn samples(&self) -> &[FrequencySampleArtifactRoots] {
        &self.samples
    }

    pub fn validate(&self, access: &FrequencySamplePublicationAccess<'_>) -> Result<(), String> {
        access
            .publication
            .validate_for(access.authority)
            .map_err(|error| error.to_string())?;
        self.publication.validate_role(
            AnalysisNumericalObjectKind::FrequencySamplePublication,
            access.publication.identity().digest(),
            "frequency-publication",
        )?;
        if self.publication.expected_payload_digest()
            != Digest::sha256(
                access
                    .publication
                    .canonical_bytes()
                    .map_err(|error| error.to_string())?,
            )
            || self.samples.len() != access.publication.samples().len()
        {
            return Err("frequency storage inventory differs from its complete publication".into());
        }
        let mut declared = vec![self.publication.reference().clone()];
        for (roots, sample) in self.samples.iter().zip(access.publication.samples()) {
            roots.validate_catalogs(&sample.fields, sample.observations.as_ref())?;
            declared.push(roots.fields().reference().clone());
            if let Some(observations) = roots.observations() {
                declared.push(observations.reference().clone());
            }
        }
        validate_complete_root_inventory(declared, access.object_catalog)
    }

    /// Admit every original manifest and its exact declared chunk closure without reading
    /// numerical chunks. No unchecked page handle is exposed.
    pub fn admit<'a>(
        &'a self,
        access: FrequencySamplePublicationAccess<'a>,
        read: impl FnMut(Digest) -> Result<Option<Vec<u8>>, String>,
    ) -> Result<AdmittedFrequencySamplePublication<'a>, String> {
        self.validate(&access)?;
        let mut roots = vec![&self.publication];
        for sample in &self.samples {
            roots.push(sample.fields());
            roots.extend(sample.observations());
        }
        super::closure::validate_closure(&roots, access.object_catalog, read)?;
        Ok(AdmittedFrequencySamplePublication {
            inventory: self,
            access,
        })
    }
}

impl AdmittedFrequencySamplePublication<'_> {
    /// Exact scientific field paging over original immutable partitions using the same
    /// reader as structural frequency publications.
    pub fn load_field_page(
        &self,
        run_id: &str,
        selection: &AnalysisFieldSelection,
        options: AnalysisFieldRequestOptions,
        read: impl FnMut(Digest) -> Result<Option<Vec<u8>>, String>,
    ) -> Result<AnalysisFieldPageLookup, String> {
        crate::analysis::storage::frequency_page::load_sample_page(
            run_id,
            &self.inventory.samples,
            self.access.publication,
            selection,
            options,
            read,
        )
    }
}

#[cfg(test)]
mod tests;

//! Typed roots for retained observation publications; numerical expectations live in the catalog.
use runmat_analysis_fea::execution::{
    ANALYSIS_MANIFEST_MEDIA_TYPE, ANALYSIS_MANIFEST_VALUE_SCHEMA,
};
use runmat_analysis_fea::results::observation::{
    ObservationArtifactIdentity, ObservationHistoryIdentity, ObservationPublicationCatalog,
    ObservationPublicationEntry, SeriesReductionResultIdentity,
};
use runmat_execution::{
    value::{PublishedObjectCatalog, ValueRef, ValueRefKind},
    Digest,
};
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ThermalObservationObjectRoot {
    pub reference: ValueRef,
    pub payload_digest: Digest,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ThermalObservationSampleRoot {
    pub artifact: ObservationArtifactIdentity,
    pub object: ThermalObservationObjectRoot,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ThermalObservationSeriesRoot {
    pub identity: SeriesReductionResultIdentity,
    pub object: ThermalObservationObjectRoot,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ThermalObservationHistoryRoot {
    pub history: ObservationHistoryIdentity,
    pub object: ThermalObservationObjectRoot,
    pub series: Option<ThermalObservationSeriesRoot>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ThermalObservationPublication {
    pub catalog: ObservationPublicationCatalog,
    pub samples: Vec<ThermalObservationSampleRoot>,
    pub histories: Vec<ThermalObservationHistoryRoot>,
}

impl ThermalObservationPublication {
    /// Validate exact semantic-root coverage and membership in the authenticated physical closure.
    /// Source/model validation remains `catalog.validate_for`; payload readers verify each root.
    pub fn validate(&self, objects: &PublishedObjectCatalog) -> Result<(), String> {
        self.catalog.validate().map_err(|error| error.to_string())?;
        objects.validate().map_err(|error| error.to_string())?;
        let mut expected_samples = BTreeSet::new();
        let mut expected_histories = BTreeSet::new();
        for entry in self.catalog.entries() {
            match entry {
                ObservationPublicationEntry::Sample(sample) => {
                    expected_samples.insert(sample.manifest.identity.digest());
                }
                ObservationPublicationEntry::History(history) => {
                    expected_histories.insert(history.history.identity.digest());
                }
            }
        }
        let samples = self
            .samples
            .iter()
            .map(|sample| sample.artifact.digest())
            .collect::<BTreeSet<_>>();
        let histories = self
            .histories
            .iter()
            .map(|history| history.history.digest())
            .collect::<BTreeSet<_>>();
        if samples != expected_samples
            || histories != expected_histories
            || samples.len() != self.samples.len()
            || histories.len() != self.histories.len()
        {
            return Err("thermal observation roots differ from publication catalog".into());
        }
        let mut root_ids = BTreeSet::new();
        let mut validate_root = |root: &ThermalObservationObjectRoot| -> Result<(), String> {
            if root.payload_digest == Digest::from_bytes([0; 32])
                || root.reference.kind != ValueRefKind::ResultObject
                || root.reference.media_type != ANALYSIS_MANIFEST_MEDIA_TYPE
                || root.reference.value_schema != ANALYSIS_MANIFEST_VALUE_SCHEMA
                || !objects.objects.contains(&root.reference)
                || !root_ids.insert(root.reference.id)
            {
                return Err(
                    "thermal observation root is absent, repeated, or outside its authority".into(),
                );
            }
            Ok(())
        };
        for sample in &self.samples {
            validate_root(&sample.object)?;
        }
        for history in &self.histories {
            validate_root(&history.object)?;
            let published = self
                .catalog
                .entries()
                .iter()
                .find_map(|entry| match entry {
                    ObservationPublicationEntry::History(entry)
                        if entry.history.identity == history.history =>
                    {
                        Some(entry)
                    }
                    _ => None,
                })
                .ok_or("thermal history publication is absent")?;
            if published.series.as_ref().map(|series| series.identity)
                != history.series.as_ref().map(|series| series.identity)
            {
                return Err("thermal series root differs from history publication".into());
            }
            if let Some(series) = &history.series {
                validate_root(&series.object)?;
            }
        }
        Ok(())
    }
}

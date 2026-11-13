use std::collections::BTreeSet;

use runmat_analysis_fea::{
    execution::{FrequencySamplePublicationManifest, StructuralFrequencyPublicationManifest},
    results::field::{RecoveredFieldArtifactEntry, RecoveredFieldNumericRepresentation},
};

use crate::analysis::{
    AnalysisFieldDescriptor, AnalysisFieldPageResult, AnalysisFieldPageValues,
    AnalysisFieldRequestOptions, AnalysisFieldSelection,
};

use super::AnalysisFieldPageLookup;

pub(super) struct SelectedField<'a> {
    pub(super) sample_index: usize,
    pub(super) entry: &'a RecoveredFieldArtifactEntry,
}

pub(super) fn select_field<'a>(
    publication: &'a StructuralFrequencyPublicationManifest,
    selection: &AnalysisFieldSelection,
) -> Result<SelectedField<'a>, AnalysisFieldPageLookup> {
    let candidates = publication
        .samples()
        .iter()
        .enumerate()
        .flat_map(|(sample_index, sample)| {
            sample.fields.fields.iter().map(move |entry| SelectedField {
                sample_index,
                entry,
            })
        })
        .collect::<Vec<_>>();
    select_candidates(candidates, selection)
}

pub(super) fn select_candidates<'a>(
    candidates: Vec<SelectedField<'a>>,
    selection: &AnalysisFieldSelection,
) -> Result<SelectedField<'a>, AnalysisFieldPageLookup> {
    if selection.transient_sample_index.is_some() {
        // Frequency publications have no retained transient states.
        return Err(AnalysisFieldPageLookup::SampleNotFound);
    }
    let mut candidates = candidates
        .into_iter()
        .filter(|candidate| {
            crate::analysis::field_selection::field_id_matches(
                candidate.entry.definition.field_id(),
                &selection.field_id,
            )
        })
        .collect::<Vec<_>>();
    if candidates.is_empty() {
        return Err(AnalysisFieldPageLookup::FieldNotFound);
    }

    let has_exact = candidates
        .iter()
        .any(|candidate| candidate.entry.definition.field_id() == selection.field_id);
    if has_exact {
        candidates.retain(|candidate| candidate.entry.definition.field_id() == selection.field_id);
    } else {
        let field_ids = candidates
            .iter()
            .map(|candidate| candidate.entry.definition.field_id().to_string())
            .collect::<BTreeSet<_>>();
        if field_ids.len() != 1 {
            return Err(AnalysisFieldPageLookup::FieldSelectionAmbiguous(
                field_ids.into_iter().collect(),
            ));
        }
    }

    match selection.sample {
        Some(sample) => candidates
            .into_iter()
            .find(|candidate| candidate.entry.sample == sample)
            .ok_or(AnalysisFieldPageLookup::SampleNotFound),
        None if candidates.len() == 1 => Ok(candidates.remove(0)),
        None => {
            let mut samples = Vec::new();
            for candidate in candidates {
                if !samples.contains(&candidate.entry.sample) {
                    samples.push(candidate.entry.sample);
                }
            }
            Err(AnalysisFieldPageLookup::SampleSelectionRequired(samples))
        }
    }
}

pub(super) fn select_sample_field<'a>(
    publication: &'a FrequencySamplePublicationManifest,
    selection: &AnalysisFieldSelection,
) -> Result<SelectedField<'a>, AnalysisFieldPageLookup> {
    if selection.sample.is_none() && publication.samples().len() > 1 {
        return Err(AnalysisFieldPageLookup::SampleSelectionRequired(
            publication
                .samples()
                .iter()
                .map(|sample| {
                    runmat_analysis_fea::results::field::RecoveredFieldSample::FrequencyHertz(
                        sample.entry.frequency_hz(),
                    )
                })
                .collect(),
        ));
    }
    let sample = publication
        .select_sample(selection.sample)
        .map_err(|_| AnalysisFieldPageLookup::SampleNotFound)?;
    let sample_index = sample.entry.ordinal() as usize;
    select_candidates(
        sample
            .fields
            .fields
            .iter()
            .map(|entry| SelectedField {
                sample_index,
                entry,
            })
            .collect(),
        selection,
    )
}

#[derive(Clone, Copy)]
pub(super) struct LogicalPage {
    pub(super) offset: usize,
    pub(super) count: usize,
    pub(super) total_count: usize,
}

impl LogicalPage {
    pub(super) fn new(
        total_count: usize,
        options: AnalysisFieldRequestOptions,
    ) -> Result<Self, String> {
        let offset = options.offset.unwrap_or(0).min(total_count);
        let limit = options
            .limit
            .unwrap_or(crate::analysis::ANALYSIS_FIELD_DEFAULT_MATERIALIZE_LIMIT)
            .min(crate::analysis::ANALYSIS_FIELD_DEFAULT_PAGE_SIZE);
        let count = limit.min(total_count.saturating_sub(offset));
        offset
            .checked_add(count)
            .ok_or_else(|| "analysis field page range overflow".to_string())?;
        Ok(Self {
            offset,
            count,
            total_count,
        })
    }

    pub(super) fn end(self) -> usize {
        self.offset + self.count
    }

    pub(super) fn truncated(self) -> bool {
        self.offset > 0 || self.end() < self.total_count
    }
}

pub(super) fn page_result(
    run_id: &str,
    entry: &RecoveredFieldArtifactEntry,
    descriptor: AnalysisFieldDescriptor,
    page: LogicalPage,
    values: AnalysisFieldPageValues,
) -> AnalysisFieldPageLookup {
    AnalysisFieldPageLookup::Page(Box::new(AnalysisFieldPageResult {
        run_id: run_id.to_string(),
        field_id: entry.definition.field_id().to_string(),
        descriptor,
        values,
        offset: page.offset,
        count: page.count,
        total_count: page.total_count,
        truncated: page.truncated(),
    }))
}

pub(super) fn empty_values(
    representation: RecoveredFieldNumericRepresentation,
) -> AnalysisFieldPageValues {
    match representation {
        RecoveredFieldNumericRepresentation::Real => AnalysisFieldPageValues::Real(Vec::new()),
        RecoveredFieldNumericRepresentation::ComplexInterleaved => {
            AnalysisFieldPageValues::ComplexInterleaved(Vec::new())
        }
    }
}

//! Size refinement against the shared accepted volume state. Candidate cavities
//! and quality deltas remain local; canonical publication is outside this owner.
use super::*;
use crate::cdt::insertion::prepare_delaunay_volume_node_insertion_with_prepared_barriers_from_hint;
use crate::cdt::insertion::topology_access::TetrahedralTopology;
use crate::cdt::volume_quality::{working::QualityCursor, QualityObjective};
use crate::cdt::volume_state::{
    SearchState, VolumeStateError, VolumeStateErrorKind, VolumeStateOptions, WorkingState,
};
use std::collections::BTreeSet;

const MAXIMUM_BATCH_SIZE: usize = 128;
const CANDIDATES_PER_SOURCE_WINDOW: usize = 16;
#[derive(Default)]
pub(in crate::cdt) struct WorkingRefinement {
    after: Option<QualityCursor>,
    progress_policy: super::admission::RefinementProgressPolicy,
    probe_before_repair: bool,
}
pub(in crate::cdt) enum WorkingRefinementProgress {
    Accepted(Vec<DelaunayVolumeNode>),
    Yielded,
    Blocked,
}
impl From<DelaunayVolumeRefinementStepOptions> for VolumeStateOptions {
    fn from(options: DelaunayVolumeRefinementStepOptions) -> Self {
        Self {
            insertion: options.insertion,
            cancellation_check_interval: options.candidate.cancellation_check_interval,
        }
    }
}
impl From<VolumeStateError> for DelaunayVolumeRefinementStepError {
    fn from(failure: VolumeStateError) -> Self {
        let kind = match failure.kind {
            VolumeStateErrorKind::Accounting(cause) => {
                DelaunayVolumeRefinementStepErrorKind::Accounting(cause)
            }
            VolumeStateErrorKind::InvalidOptions => {
                DelaunayVolumeRefinementStepErrorKind::InvalidOptions
            }
            VolumeStateErrorKind::InvalidInput => {
                DelaunayVolumeRefinementStepErrorKind::InvalidInput
            }
            VolumeStateErrorKind::InvalidTopology => {
                DelaunayVolumeRefinementStepErrorKind::InvalidTopology
            }
            VolumeStateErrorKind::InvalidQuality => {
                DelaunayVolumeRefinementStepErrorKind::InvalidQuality
            }
            VolumeStateErrorKind::ResourceLimit => {
                DelaunayVolumeRefinementStepErrorKind::ResourceLimit
            }
            VolumeStateErrorKind::InvalidProvenance => {
                DelaunayVolumeRefinementStepErrorKind::InvalidProvenance
            }
            VolumeStateErrorKind::Cancelled => DelaunayVolumeRefinementStepErrorKind::Cancelled,
        };
        super::insertion::error(kind, failure.reason)
    }
}
impl WorkingRefinement {
    /// Complete convergence owns both terminal bounds, unlike a standalone subdivision.
    pub(in crate::cdt) fn coupled() -> Self {
        Self {
            after: None,
            progress_policy: super::admission::RefinementProgressPolicy::Coupled,
            probe_before_repair: true,
        }
    }
    /// An accepted repair changes candidate geometry and invalidates the cursor.
    pub(in crate::cdt) fn repaired(&mut self) {
        self.after = None;
        self.probe_before_repair = true;
    }
    pub(in crate::cdt) fn step(
        &mut self,
        state: &mut WorkingState<'_>,
        options: DelaunayVolumeRefinementStepOptions,
        remaining: u64,
        cancellation: &dyn MeshingCancellationSignal,
    ) -> Result<WorkingRefinementProgress, DelaunayVolumeRefinementStepError> {
        super::validate_options(options.candidate).map_err(super::insertion::candidate_error)?;
        if remaining == 0 {
            return Err(super::insertion::error(
                DelaunayVolumeRefinementStepErrorKind::ResourceLimit,
                "coupled refinement exhausted its insertion budget",
            ));
        }
        let yield_after_source =
            self.probe_before_repair && state.quality().worst(QualityObjective::Shape).is_some();
        let maximum = MAXIMUM_BATCH_SIZE.min(usize::try_from(remaining).unwrap_or(usize::MAX));
        let mut work = CandidateWork::new(options.candidate, cancellation);
        let mut excluded = BTreeSet::new();
        let mut attempts = 0_u64;
        let mut rejected = 0_u64;
        let mut rejection_reasons = std::collections::BTreeMap::<&'static str, u64>::new();
        let mut pass_constructed = false;
        let mut complete_pass = self.after.is_none();
        loop {
            let window = state.quality().refinement_window(self.after, maximum);
            if window.is_empty() {
                if self.after.take().is_some() {
                    if complete_pass
                        && (!pass_constructed
                            || !excluded.is_empty()
                                && state.quality().worst(QualityObjective::Shape).is_some())
                    {
                        return Ok(WorkingRefinementProgress::Blocked);
                    }
                    pass_constructed = false;
                    complete_pass = true;
                    continue;
                }
                return Ok(WorkingRefinementProgress::Blocked);
            }
            // Only the bounded scheduling window is copied. Each actual trial
            // uses current topology and the shared resolved-field quality owner.
            let sources = window
                .into_iter()
                .map(|(cursor, source)| (cursor, source.node_identities))
                .collect::<Vec<_>>();
            let mut batch = state.trial();
            let mut inserted = Vec::new();
            for (cursor, identities) in sources {
                if !work.can_construct() {
                    break;
                }
                self.after = Some(cursor);
                // Earlier accepted points can remove this cell or alter its
                // graded metric. Never construct from a stale copied record.
                let Some(source) = batch.quality().quality_for_cell(identities).cloned() else {
                    continue;
                };
                for _ in 0..CANDIDATES_PER_SOURCE_WINDOW {
                    if !work.can_construct() {
                        break;
                    }
                    let candidate =
                        match super::candidate::construct_candidate_on_topology(
                            batch.topology(),
                            &source,
                            batch.quality_options(),
                            &[],
                            &excluded,
                            &mut work,
                        ) {
                            Ok(candidate) => candidate,
                            Err(failure)
                                if matches!(failure.kind,
                            DelaunayVolumeRefinementCandidateErrorKind::NumericalFailure |
                            DelaunayVolumeRefinementCandidateErrorKind::InvalidCandidate) =>
                            {
                                break
                            }
                            Err(failure) => return Err(super::insertion::candidate_error(failure)),
                        };
                    pass_constructed = true;
                    attempts += 1;
                    let vertices = source.node_identities.map(|id| {
                        batch
                            .topology()
                            .node_index(id)
                            .expect("current quality corner")
                    });
                    let hint = batch.topology().tetrahedron_index(vertices);
                    let insertion = match prepare_delaunay_volume_node_insertion_with_prepared_barriers_from_hint(
                        batch.topology(), candidate.node, batch.preparation().protected_facets(), true,
                        hint, options.insertion, cancellation,
                    ) {
                        Ok(insertion) => insertion,
                        Err(failure) if matches!(failure.kind,
                            crate::cdt::DelaunayInsertionErrorKind::InvalidNode |
                            crate::cdt::DelaunayInsertionErrorKind::InvalidTopology |
                            crate::cdt::DelaunayInsertionErrorKind::PointOutsideTopology) => {
                                excluded.insert(candidate.node.identity); rejected += 1;
                                *rejection_reasons.entry("insertion rejected").or_default() += 1;
                                report_rejection(attempts, rejected, "insertion", &failure.reason, &source);
                                continue;
                            },
                        Err(failure) => return Err(super::insertion::insertion_error(failure)),
                    };
                    let node = insertion.node;
                    // This snapshots changed priority bins, not the mesh/field.
                    // The nested transaction includes complete grading support
                    // and legalization edits, and rolls both owners back on reject.
                    let parent = batch.quality().capture_spectrum();
                    let mut trial = batch.trial();
                    if let Err(failure) = trial.insert_prepared(insertion, options, cancellation) {
                        if matches!(
                            failure.kind,
                            VolumeStateErrorKind::InvalidTopology
                                | VolumeStateErrorKind::InvalidInput
                                | VolumeStateErrorKind::InvalidQuality
                        ) {
                            excluded.insert(candidate.node.identity);
                            rejected += 1;
                            *rejection_reasons.entry("trial rejected").or_default() += 1;
                            report_rejection(attempts, rejected, "trial", &failure.reason, &source);
                            continue;
                        }
                        return Err(failure.into());
                    }
                    if let Some(reason) = self.progress_policy.rejection(trial.quality(), &parent) {
                        excluded.insert(candidate.node.identity);
                        rejected += 1;
                        *rejection_reasons.entry(reason).or_default() += 1;
                        report_rejection(attempts, rejected, "quality_spectrum", reason, &source);
                        continue;
                    }
                    trial.commit();
                    inserted.push(node);
                    tracing::debug!(target:"runmat_meshing_tetrahedron::refinement::provisional",
                        accepted_count=inserted.len(), "local refinement provisional point admitted");
                    // Rejection facts only describe the preceding unchanged state.
                    excluded.clear();
                    break;
                }
                // One existing source window is enough to expose a shape
                // dependency. Preserve the admitted work limit: exhaustion
                // remains an error, never a handoff that bypasses admission.
                if inserted.is_empty() && yield_after_source && work.can_construct() {
                    drop(batch);
                    self.probe_before_repair = false;
                    tracing::info!(
                        attempts,
                        rejected,
                        "size frontier yielded to coupled repair"
                    );
                    return Ok(WorkingRefinementProgress::Yielded);
                }
            }
            if !inserted.is_empty() {
                let change = batch.commit_prepared(cancellation)?;
                state.rebase_committed(change);
                self.probe_before_repair = true;
                let (shape_violations, refinement_violations) = state.quality().violation_counts();
                tracing::info!(
                    attempts,
                    rejected,
                    shape_violations,
                    refinement_violations,
                    cells = state.quality().cell_count(),
                    candidate_evaluations = work.evaluations(),
                    accepted_count = inserted.len(),
                    "local refinement batch admitted"
                );
                return Ok(WorkingRefinementProgress::Accepted(inserted));
            }
            drop(batch);
            if !work.can_construct() {
                return Err(super::insertion::error(DelaunayVolumeRefinementStepErrorKind::ResourceLimit,
                    format!("working refinement exhausted candidate work after {attempts} attempts and {rejected} rejections; rejection reasons {rejection_reasons:?}")));
            }
        }
    }
}

/// Bounded live diagnosis of an unchanged search; never a reason to accept a
/// candidate or modify its quality policy. Avoid per-predicate/per-cell logging.
fn report_rejection(
    attempts: u64,
    rejected: u64,
    phase: &str,
    reason: &str,
    source: &DelaunayTetrahedronQuality,
) {
    if attempts == 1 || attempts.is_multiple_of(128) {
        tracing::info!(attempts, rejected, phase, reason,
            source = ?source.node_identities,
            refinement_violation = source.refinement_violation_ratio,
            metric_edge = source.maximum_metric_edge_length,
            radius_edge = source.metric_radius_edge_ratio,
            "local refinement search progress");
    }
}

#[cfg(test)]
mod tests;

//! Admission projection for accuracy evaluation. Workspace comes from a shape projection; the
//! output bound also covers the exact encoding of the artifact the evaluator produces.

use std::mem::size_of;

use runmat_canonical_codec::profile_serialized_shape;
use serde::ser::{SerializeMap, SerializeSeq};
use serde::{Serialize, Serializer};

use super::*;
use crate::execution::AccuracyEvaluationRequest;

/// Requested-heap admission estimates, not allocator overhead or measured process RSS.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct AccuracyEvaluationResourceProfile {
    pub workspace_bytes_upper_bound: u64,
    pub output_encoded_bytes_upper_bound: u64,
}

/// Projects the largest report shape allowed by these criteria and evidence. Matching evidence
/// is retained before deduplication, and every possible measured value is represented rather
/// than selecting a governing value. Thus this traversal performs no accuracy decision.
///
/// The shared canonical profiler accounts value-tree/string/encoding overlap. Additional typed
/// collection staging covers contribution storage and preallocated matching/assumption lists. Caller-owned input
/// objects and transport output construction are accounted separately by the phase planner.
pub fn accuracy_evaluation_resource_profile(
    request: &AccuracyEvaluationRequest,
    evidence: &[AccuracyEvidence],
) -> Result<AccuracyEvaluationResourceProfile, AccuracyEvaluationError> {
    let projection = Projection { request, evidence };
    let output = profile_serialized_shape(&projection, ACCURACY_REPORT_LIMITS)
        .map_err(|error| AccuracyEvaluationError::IdentityEncoding(error.to_string()))?;
    let input = profile_serialized_shape(&(&request.plan, evidence), ACCURACY_REPORT_LIMITS)
        .map_err(|error| AccuracyEvaluationError::IdentityEncoding(error.to_string()))?;
    let contributions = bytes::<AccuracyContribution>(request.plan.criteria.len())?;
    let matching = bytes::<&AccuracyEvidence>(evidence.len())?;
    // Evaluation preallocates matching pointers and pre-deduplication assumptions, then sorts
    // them in place. Count retained String slots and clones explicitly, without relying on
    // incidental slack in the shared canonical materialization estimate.
    let mut assumptions = 0_u64;
    for criterion in &request.plan.criteria {
        for item in evidence.iter().filter(|item| {
            item.target == criterion.target
                && item.contribution == criterion.contribution
                && item.kind == criterion.evidence
        }) {
            assumptions = add(assumptions, bytes::<String>(item.assumptions.len())?)?;
            for assumption in &item.assumptions {
                assumptions = add(assumptions, bytes::<u8>(assumption.len())?)?;
            }
        }
    }
    let typed = add(contributions, add(matching, assumptions)?)?;
    // Output identity recomputation and final serialization are sequential, not repeated
    // simultaneous copies. Include both input-validation and output-encoding workspaces so
    // their overlap remains admitted even if validation is moved into the output handoff.
    let workspace = add(
        add(
            output.materialization_resident_bytes_estimate,
            input.materialization_resident_bytes_estimate,
        )?,
        add(typed, size_of::<AccuracyArtifact>() as u64)?,
    )?;
    // The output bound comes from the same typed encoding the evaluator emits: evaluation is a
    // pure function of the request and its authenticated evidence, so encoding the artifact it
    // produces gives the exact size. The shape projection remains a floor for workspace sizing.
    let exact = evaluate_accuracy_artifact(request, evidence.to_vec())
        .and_then(|artifact| artifact.canonical_bytes())
        .map(|bytes| bytes.len() as u64)?;
    Ok(AccuracyEvaluationResourceProfile {
        workspace_bytes_upper_bound: workspace,
        output_encoded_bytes_upper_bound: output.encoded_bytes_upper_bound.max(exact),
    })
}

fn add(left: u64, right: u64) -> Result<u64, AccuracyEvaluationError> {
    left.checked_add(right).ok_or_else(|| {
        AccuracyEvaluationError::IdentityEncoding("accuracy resource projection overflows".into())
    })
}

fn bytes<T>(count: usize) -> Result<u64, AccuracyEvaluationError> {
    u64::try_from(count)
        .ok()
        .and_then(|count| count.checked_mul(size_of::<T>() as u64))
        .ok_or_else(|| {
            AccuracyEvaluationError::IdentityEncoding(
                "accuracy resource projection overflows".into(),
            )
        })
}

struct Projection<'a> {
    request: &'a AccuracyEvaluationRequest,
    evidence: &'a [AccuracyEvidence],
}

impl Serialize for Projection<'_> {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let mut map = serializer.serialize_map(Some(10))?;
        map.serialize_entry("schema_version", &2_u16)?;
        map.serialize_entry("identity", &Digest::sha256([]))?;
        map.serialize_entry("evaluation_request", &self.request.identity())?;
        map.serialize_entry("producer", &self.request.producer)?;
        map.serialize_entry("source_kind", &self.request.source_kind)?;
        map.serialize_entry("source_result", &self.request.source_result)?;
        map.serialize_entry("source_observations", &self.request.source_observations)?;
        map.serialize_entry("evidence", &self.evidence)?;
        map.serialize_entry("evaluation", &EvaluationProjection(self))?;
        map.serialize_entry("gate", &GateProjection(self.longest_id()))?;
        map.end()
    }
}

impl Projection<'_> {
    fn longest_id(&self) -> &str {
        self.request
            .plan
            .criteria
            .iter()
            .map(|criterion| criterion.criterion_id.as_str())
            .max_by_key(|id| id.len())
            .unwrap_or("")
    }
}

struct GateProjection<'a>(&'a str);
impl Serialize for GateProjection<'_> {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let mut map = serializer.serialize_map(Some(3))?;
        map.serialize_entry("decision", "block")?;
        map.serialize_entry("status", "indeterminate")?;
        map.serialize_entry("governing_criterion_id", &Some(self.0))?;
        map.end()
    }
}

struct EvaluationProjection<'a, 'b>(&'a Projection<'b>);
impl Serialize for EvaluationProjection<'_, '_> {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let mut map = serializer.serialize_map(Some(2))?;
        map.serialize_entry("identity", &Digest::sha256([]))?;
        map.serialize_entry("report", &ReportProjection(self.0))?;
        map.end()
    }
}

struct ReportProjection<'a, 'b>(&'a Projection<'b>);
impl Serialize for ReportProjection<'_, '_> {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let mut map = serializer.serialize_map(Some(5))?;
        map.serialize_entry("accuracy_plan_id", &self.0.request.plan.accuracy_plan_id)?;
        map.serialize_entry("accuracy_plan_digest", &Digest::sha256([]))?;
        map.serialize_entry("status", "indeterminate")?;
        map.serialize_entry("contributions", &Contributions(self.0))?;
        map.serialize_entry("governing_criterion_id", &Some(self.0.longest_id()))?;
        map.end()
    }
}

struct Contributions<'a, 'b>(&'a Projection<'b>);
impl Serialize for Contributions<'_, '_> {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let mut sequence = serializer.serialize_seq(Some(self.0.request.plan.criteria.len()))?;
        for criterion in &self.0.request.plan.criteria {
            sequence.serialize_element(&ContributionProjection {
                criterion,
                evidence: self.0.evidence,
            })?;
        }
        sequence.end()
    }
}

struct ContributionProjection<'a> {
    criterion: &'a runmat_analysis_core::AccuracyCriterion,
    evidence: &'a [AccuracyEvidence],
}

impl Serialize for ContributionProjection<'_> {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let mut map = serializer.serialize_map(Some(10))?;
        map.serialize_entry("criterion_id", &self.criterion.criterion_id)?;
        map.serialize_entry("contribution", &self.criterion.contribution)?;
        map.serialize_entry("required_evidence", &self.criterion.evidence)?;
        map.serialize_entry("mandatory", &self.criterion.mandatory)?;
        map.serialize_entry("status", "indeterminate")?;
        map.serialize_entry("normalized_error", &Some(1.0_f64))?;
        // Each list dominates the corresponding selected/deduplicated output, including null
        // when there is no evidence. No criterion is evaluated to create this projection.
        map.serialize_entry("measured", &Matched(self, Part::Measure))?;
        map.serialize_entry("assumptions", &Matched(self, Part::Assumptions))?;
        map.serialize_entry("scopes", &Matched(self, Part::Scope))?;
        map.serialize_entry("evidence_identities", &Matched(self, Part::Identity))?;
        map.end()
    }
}

enum Part {
    Measure,
    Assumptions,
    Scope,
    Identity,
}
struct Matched<'a, 'b>(&'a ContributionProjection<'b>, Part);
impl Serialize for Matched<'_, '_> {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let mut sequence = serializer.serialize_seq(None)?;
        for item in self.0.evidence.iter().filter(|item| {
            item.target == self.0.criterion.target
                && item.contribution == self.0.criterion.contribution
                && item.kind == self.0.criterion.evidence
        }) {
            match self.1 {
                Part::Measure => sequence.serialize_element(&item.measured)?,
                Part::Assumptions => {
                    for assumption in &item.assumptions {
                        sequence.serialize_element(assumption)?;
                    }
                }
                Part::Scope => sequence.serialize_element(&item.scope)?,
                Part::Identity => sequence.serialize_element(&item.identity())?,
            }
        }
        sequence.end()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture() -> (AccuracyEvaluationRequest, AccuracyEvidence) {
        let sequence =
            crate::execution::SequenceIdentity::new(Digest::sha256("resource sequence")).unwrap();
        let producer = StepIdentity::new(sequence, Digest::sha256("resource step")).unwrap();
        let result =
            SolverResultIdentity::from_step_result(producer, Digest::sha256("resource result"))
                .unwrap();
        let request = AccuracyEvaluationRequest::new(
            producer,
            crate::execution::AccuracySourceKind::StructuralLinear,
            result,
            None,
            AccuracyPlan::baseline("resource accuracy"),
        )
        .unwrap();
        let evidence = AccuracyEvidence::new(
            AccuracyTarget::GlobalEquilibrium,
            ErrorContributionKind::Algebraic,
            RequiredAccuracyEvidence::ResidualBound,
            AccuracyMeasure::Relative(1.0e-12),
            vec!["canonical residual norm".into()],
            AccuracyEvidenceScope::Step(producer),
        )
        .unwrap();
        (request, evidence)
    }

    fn assert_bounds(
        request: &AccuracyEvaluationRequest,
        evidence: &[AccuracyEvidence],
    ) -> AccuracyEvaluationResourceProfile {
        let profile = accuracy_evaluation_resource_profile(request, evidence).unwrap();
        let artifact = evaluate_accuracy_artifact(request, evidence.to_vec()).unwrap();
        let encoded = artifact.canonical_bytes().unwrap();
        assert!(profile.output_encoded_bytes_upper_bound >= encoded.len() as u64);
        let actual_profile = profile_serialized_shape(&artifact, ACCURACY_REPORT_LIMITS).unwrap();
        assert!(
            profile.output_encoded_bytes_upper_bound >= actual_profile.encoded_bytes_upper_bound
        );
        assert!(
            profile.workspace_bytes_upper_bound
                >= actual_profile.materialization_resident_bytes_estimate
        );
        profile
    }

    #[test]
    fn bounds_supported_and_missing_evidence_without_evaluating_in_planner() {
        let (request, evidence) = fixture();
        assert_bounds(&request, &[]);
        assert_bounds(&request, &[evidence]);
    }

    #[test]
    fn accounts_matching_criterion_and_assumption_amplification() {
        let (request, evidence) = fixture();
        let baseline = assert_bounds(&request, std::slice::from_ref(&evidence));
        let mut plan = request.plan.clone();
        let criterion = plan.criteria[0].clone();
        for index in 0..32 {
            let mut criterion = criterion.clone();
            criterion.criterion_id = format!("criterion-{index}");
            plan.criteria.push(criterion);
        }
        let request = AccuracyEvaluationRequest::new(
            request.producer,
            request.source_kind,
            request.source_result,
            None,
            plan,
        )
        .unwrap();
        let evidence = (0..16)
            .map(|index| {
                AccuracyEvidence::new(
                    evidence.target.clone(),
                    evidence.contribution,
                    evidence.kind,
                    evidence.measured,
                    vec![format!(
                        "independent-assumption-{index}-{}",
                        "x".repeat(256)
                    )],
                    evidence.scope,
                )
                .unwrap()
            })
            .collect::<Vec<_>>();
        let amplified = assert_bounds(&request, &evidence);
        assert!(
            amplified.output_encoded_bytes_upper_bound > baseline.output_encoded_bytes_upper_bound
        );
        assert!(amplified.workspace_bytes_upper_bound > baseline.workspace_bytes_upper_bound);
    }

    #[test]
    fn bounds_absolute_measure_and_failed_production_gate() {
        let (request, original) = fixture();
        let mut plan = request.plan.clone();
        let quantity = ObservationQuantity::ThermalEnergyStorageRateDensity;
        plan.criteria[0].tolerance = AccuracyTolerance::Absolute {
            quantity,
            canonical_value: 1.0,
        };
        let request = AccuracyEvaluationRequest::new(
            request.producer,
            request.source_kind,
            request.source_result,
            None,
            plan,
        )
        .unwrap();
        let evidence = AccuracyEvidence::new(
            original.target,
            original.contribution,
            original.kind,
            AccuracyMeasure::Absolute {
                quantity,
                canonical_value: 2.0,
            },
            original.assumptions,
            original.scope,
        )
        .unwrap();
        assert_bounds(&request, &[evidence]);
    }

    #[test]
    fn excessive_strings_and_arithmetic_fail_closed() {
        let (request, mut evidence) = fixture();
        evidence.assumptions = vec!["x".repeat(ACCURACY_REPORT_LIMITS.maximum_string_bytes + 1)];
        assert!(accuracy_evaluation_resource_profile(&request, &[evidence]).is_err());
        assert!(add(u64::MAX, 1).is_err());
    }
}

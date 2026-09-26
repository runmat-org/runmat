//! Shared borrowed assignment reporting for actual output and preassembly sizing.
use std::fmt;

use runmat_analysis_core::{EvidenceConfidence, MaterialAssignment};
use runmat_canonical_codec::{
    profile_serialized, CanonicalCodecError, CanonicalLimits, SerializedCanonicalProfile,
};
use serde::{ser::SerializeSeq, Serialize, Serializer};

use super::{DiagnosticSerialization, FeaDiagnostic, FeaDiagnosticSeverity};

struct Message<'a>(&'a MaterialAssignment);

impl fmt::Display for Message<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let assignment = self.0;
        write!(
            f,
            "region={} expected_material={} assigned_material={} confidence={:?}",
            assignment.target_label(),
            assignment.expected_material_id,
            assignment.assigned_material_id,
            assignment.confidence
        )
    }
}

impl Serialize for Message<'_> {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.collect_str(self)
    }
}

fn records(
    assignments: &[MaterialAssignment],
) -> impl Iterator<Item = DiagnosticSerialization<'_, Message<'_>>> + Clone {
    assignments
        .iter()
        .filter(|assignment| assignment.expected_material_id != assignment.assigned_material_id)
        .map(|assignment| {
            let (code, severity) = match assignment.confidence {
                EvidenceConfidence::Verified => (
                    "ANALYSIS_MATERIAL_ASSIGNMENT_CONFLICT_VERIFIED",
                    FeaDiagnosticSeverity::Error,
                ),
                EvidenceConfidence::Probable => (
                    "ANALYSIS_MATERIAL_ASSIGNMENT_CONFLICT_PROBABLE",
                    FeaDiagnosticSeverity::Warning,
                ),
                EvidenceConfidence::Inferred => (
                    "ANALYSIS_MATERIAL_ASSIGNMENT_CONFLICT_INFERRED",
                    FeaDiagnosticSeverity::Warning,
                ),
            };
            DiagnosticSerialization {
                code,
                severity,
                message: Message(assignment),
            }
        })
}

pub(crate) fn diagnostics(assignments: &[MaterialAssignment]) -> Vec<FeaDiagnostic> {
    records(assignments)
        .map(|record| FeaDiagnostic {
            code: record.code.into(),
            severity: record.severity,
            message: record.message.to_string(),
        })
        .collect()
}

pub(crate) fn record_count(assignments: &[MaterialAssignment]) -> usize {
    records(assignments).count()
}

pub(crate) fn profile(
    assignments: &[MaterialAssignment],
) -> Result<SerializedCanonicalProfile, CanonicalCodecError> {
    struct Records<'a>(&'a [MaterialAssignment]);
    impl Serialize for Records<'_> {
        fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
            let records = records(self.0);
            let mut sequence = serializer.serialize_seq(Some(records.clone().count()))?;
            for record in records {
                sequence.serialize_element(&record)?;
            }
            sequence.end()
        }
    }
    profile_serialized(
        &Records(assignments),
        CanonicalLimits::new(usize::MAX, usize::MAX, usize::MAX, 32),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn projection_matches_conflicts_and_ignores_matching_assignments() {
        let mut assignments = Vec::new();
        for confidence in [
            EvidenceConfidence::Verified,
            EvidenceConfidence::Probable,
            EvidenceConfidence::Inferred,
        ] {
            let mut assignment = MaterialAssignment::to_solid_region(
                "領域\"\\\n".repeat(1000),
                "expected",
                confidence,
            );
            assignment.assigned_material_id = "actual".repeat(1000);
            assignments.push(assignment);
        }
        let actual = diagnostics(&assignments);
        assert_eq!(record_count(&assignments), actual.len());
        assert_eq!(actual.len(), 3);
        assert_eq!(actual[0].severity, FeaDiagnosticSeverity::Error);
        for (record, assignment) in actual.iter().zip(&assignments) {
            assert_eq!(
                record.message,
                format!(
                    "region={} expected_material={} assigned_material={} confidence={:?}",
                    assignment.target_label(),
                    assignment.expected_material_id,
                    assignment.assigned_material_id,
                    assignment.confidence
                )
            );
        }
        let projected = profile(&assignments).unwrap();
        assert_eq!(
            projected,
            profile_serialized(
                &actual,
                CanonicalLimits::new(usize::MAX, usize::MAX, usize::MAX, 32)
            )
            .unwrap()
        );
        assignments.push(MaterialAssignment::to_solid_region(
            "matching",
            "same",
            EvidenceConfidence::Verified,
        ));
        assert_eq!(profile(&assignments).unwrap(), projected);
        assert_eq!(diagnostics(&assignments), actual);
        assert_eq!(record_count(&assignments), actual.len());
        assert_eq!(record_count(&[]), 0);
        assert!(
            profile(&[]).unwrap().encoded_bytes_upper_bound < projected.encoded_bytes_upper_bound
        );
    }
}

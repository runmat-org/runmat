//! Move accepted backend records into the admitted retained inventory.
use crate::{
    planning::resource::{ResourcePlanningError, ThermalBackendEvidenceResourceFacts},
    solve::accelerator::AcceleratorSolveEvidence,
    FeaRunError,
};

pub(super) fn retain(
    records: &mut Vec<AcceleratorSolveEvidence>,
    current: &mut Option<AcceleratorSolveEvidence>,
    facts: &ThermalBackendEvidenceResourceFacts,
) -> Result<(), FeaRunError> {
    let Some(evidence) = current.as_ref() else {
        return Ok(());
    };
    if records.len() >= facts.maximum_records {
        return Err(ResourcePlanningError::Invalid(
            "thermal backend evidence inventory exceeds admitted solve count".into(),
        )
        .into());
    }
    facts.verify(evidence)?;
    if records.capacity() == 0 {
        records
            .try_reserve_exact(facts.maximum_records)
            .map_err(|error| {
                ResourcePlanningError::Invalid(format!(
                    "cannot reserve thermal backend evidence: {error}"
                ))
            })?;
    }
    records.push(current.take().expect("validated backend evidence exists"));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn accepted_records_are_moved_and_cpu_does_not_allocate() {
        let facts = ThermalBackendEvidenceResourceFacts {
            maximum_records: 3,
            record: AcceleratorSolveEvidence::completed_profile().unwrap(),
        };
        let mut records = Vec::new();
        retain(&mut records, &mut None, &facts).unwrap();
        assert_eq!(records.capacity(), 0);
        for _ in 0..3 {
            let mut current = Some(AcceleratorSolveEvidence::completed_evidence_fixture());
            retain(&mut records, &mut current, &facts).unwrap();
            assert!(current.is_none());
        }
        assert_eq!(records.len(), 3);
        let mut extra = Some(AcceleratorSolveEvidence::completed_evidence_fixture());
        assert!(retain(&mut records, &mut extra, &facts).is_err());
        assert!(extra.is_some());
        assert_eq!(records.len(), 3);
    }

    #[test]
    fn invalid_or_unaccepted_records_fail_before_retained_allocation() {
        let facts = ThermalBackendEvidenceResourceFacts {
            maximum_records: 1,
            record: AcceleratorSolveEvidence::completed_profile().unwrap(),
        };
        for residual in [f64::NAN, 2.0] {
            let mut evidence = AcceleratorSolveEvidence::completed_evidence_fixture();
            let AcceleratorSolveEvidence::Completed {
                independently_computed_relative_residual,
                ..
            } = &mut evidence
            else {
                unreachable!()
            };
            *independently_computed_relative_residual = residual;
            let mut current = Some(evidence);
            let mut records = Vec::new();
            assert!(retain(&mut records, &mut current, &facts).is_err());
            assert!(current.is_some());
            assert_eq!(records.capacity(), 0);
        }
    }
}

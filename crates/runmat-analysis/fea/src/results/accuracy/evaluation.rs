use super::*;

pub fn evaluate_accuracy(
    plan: &AccuracyPlan,
    evidence: &[AccuracyEvidence],
) -> Result<AccuracyEvaluation, AccuracyEvaluationError> {
    validate_plan_boundary(plan)?;
    validate_evidence_boundary(evidence)?;
    if evidence.iter().any(|item| {
        !plan.criteria.iter().any(|criterion| {
            item.target == criterion.target
                && item.contribution == criterion.contribution
                && item.kind == criterion.evidence
        })
    }) {
        return Err(AccuracyEvaluationError::InvalidEvidence(
            "every submitted evidence item must satisfy a declared accuracy criterion".into(),
        ));
    }
    let accuracy_plan_digest = plan_digest(plan)?;

    let mut contributions = Vec::with_capacity(plan.criteria.len());
    for criterion in &plan.criteria {
        let mut matching = Vec::with_capacity(evidence.len());
        matching.extend(evidence.iter().filter(|item| {
            item.target == criterion.target
                && item.contribution == criterion.contribution
                && item.kind == criterion.evidence
        }));
        matching.sort_unstable_by_key(|item| item.identity());

        if matching.is_empty() {
            contributions.push(AccuracyContribution {
                criterion_id: criterion.criterion_id.clone(),
                contribution: criterion.contribution,
                required_evidence: criterion.evidence,
                mandatory: criterion.mandatory,
                status: AccuracyStatus::Indeterminate,
                measured: None,
                normalized_error: None,
                assumptions: Vec::new(),
                scopes: Vec::new(),
                evidence_identities: Vec::new(),
            });
            continue;
        }

        let mut selected = None;
        for item in &matching {
            let normalized = normalized_error(item.measured, criterion.tolerance)?;
            if selected
                .as_ref()
                .is_none_or(|(_, current): &(&AccuracyEvidence, f64)| normalized > *current)
            {
                selected = Some((*item, normalized));
            }
        }
        let (selected, normalized) = selected.ok_or_else(|| {
            AccuracyEvaluationError::InvalidEvidence(
                "matching evidence could not be reduced to a finite measure".into(),
            )
        })?;
        let assumption_count = matching.iter().try_fold(0_usize, |count, item| {
            count.checked_add(item.assumptions.len()).ok_or_else(|| {
                AccuracyEvaluationError::InvalidEvidence(
                    "matched assumption count overflows".into(),
                )
            })
        })?;
        let mut assumptions = Vec::with_capacity(assumption_count);
        assumptions.extend(
            matching
                .iter()
                .flat_map(|item| item.assumptions.iter().cloned()),
        );
        assumptions.sort_unstable();
        assumptions.dedup();
        let mut scopes = matching.iter().map(|item| item.scope).collect::<Vec<_>>();
        scopes.sort_unstable();
        scopes.dedup();
        contributions.push(AccuracyContribution {
            criterion_id: criterion.criterion_id.clone(),
            contribution: criterion.contribution,
            required_evidence: criterion.evidence,
            mandatory: criterion.mandatory,
            status: if normalized <= 1.0 {
                AccuracyStatus::Satisfied
            } else {
                AccuracyStatus::NotSatisfied
            },
            measured: Some(selected.measured),
            normalized_error: Some(normalized),
            assumptions,
            scopes,
            evidence_identities: matching.iter().map(|item| item.identity()).collect(),
        });
    }

    let status = if contributions
        .iter()
        .any(|item| item.mandatory && item.status == AccuracyStatus::NotSatisfied)
    {
        AccuracyStatus::NotSatisfied
    } else if contributions
        .iter()
        .any(|item| item.mandatory && item.status == AccuracyStatus::Indeterminate)
    {
        AccuracyStatus::Indeterminate
    } else {
        AccuracyStatus::Satisfied
    };
    let governing_criterion_id = governing_criterion(&contributions, status);
    let report = AccuracyReport {
        accuracy_plan_id: plan.accuracy_plan_id.clone(),
        accuracy_plan_digest,
        status,
        contributions,
        governing_criterion_id,
    };
    Ok(AccuracyEvaluation {
        identity: report_identity(&report)?,
        report,
    })
}

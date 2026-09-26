use super::*;
pub(super) fn inferred_order(group: &LabelVector, grouphat: &LabelVector) -> Vec<Label> {
    match group.kind {
        LabelKind::Logical => vec![Label::Logical(false), Label::Logical(true)],
        LabelKind::Categorical => categorical_order(group, grouphat),
        kind => {
            let mut combined = Vec::with_capacity(group.labels.len() + grouphat.labels.len());
            combined.extend(group.labels.iter().cloned());
            combined.extend(grouphat.labels.iter().cloned());
            let mut out = dedupe_labels(combined);
            if matches!(kind, LabelKind::Numeric | LabelKind::Integer(_)) {
                sort_labels(&mut out, kind);
            }
            out
        }
    }
}

pub(super) fn categorical_order(group: &LabelVector, grouphat: &LabelVector) -> Vec<Label> {
    let mut out = Vec::new();
    for category in group
        .categories
        .iter()
        .chain(grouphat.categories.iter())
        .flatten()
    {
        let label = Label::Text(category.clone());
        if !is_missing_label(&label)
            && group_or_grouphat_contains(&label, group, grouphat)
            && !out.iter().any(|existing| same_label(existing, &label))
        {
            out.push(label);
        }
    }
    for label in group.labels.iter().chain(grouphat.labels.iter()) {
        if !is_missing_label(label) && !out.iter().any(|existing| same_label(existing, label)) {
            out.push(label.clone());
        }
    }
    out
}

pub(super) fn group_or_grouphat_contains(
    label: &Label,
    group: &LabelVector,
    grouphat: &LabelVector,
) -> bool {
    group
        .labels
        .iter()
        .chain(grouphat.labels.iter())
        .any(|candidate| same_label(candidate, label))
}

pub(super) fn dedupe_labels<I>(labels: I) -> Vec<Label>
where
    I: IntoIterator<Item = Label>,
{
    let mut out = Vec::new();
    for label in labels {
        if is_missing_label(&label) {
            continue;
        }
        if !out.iter().any(|existing| same_label(existing, &label)) {
            out.push(label);
        }
    }
    out
}

pub(super) fn sort_labels(labels: &mut [Label], kind: LabelKind) {
    match kind {
        LabelKind::Numeric => labels.sort_by(|left, right| match (left, right) {
            (Label::Numeric(a), Label::Numeric(b)) => a.partial_cmp(b).unwrap_or(Ordering::Equal),
            _ => Ordering::Equal,
        }),
        LabelKind::Integer(_) => labels.sort_by(|left, right| match (left, right) {
            (Label::Integer(a), Label::Integer(b)) => integer_sort_key(a).cmp(&integer_sort_key(b)),
            _ => Ordering::Equal,
        }),
        LabelKind::String | LabelKind::Char | LabelKind::Cell | LabelKind::Categorical => labels
            .sort_by(|left, right| match (left, right) {
                (Label::Text(a), Label::Text(b)) => a.cmp(b),
                _ => Ordering::Equal,
            }),
        LabelKind::Logical => labels.sort_by_key(|label| match label {
            Label::Logical(value) => *value,
            _ => false,
        }),
    }
}

pub(super) fn ensure_order_covers(
    order: &[Label],
    group: &[Label],
    grouphat: &[Label],
) -> BuiltinResult<()> {
    for label in group.iter().chain(grouphat.iter()) {
        if is_missing_label(label) {
            continue;
        }
        if !order
            .iter()
            .any(|order_label| same_label(order_label, label))
        {
            return Err(invalid(
                "confusionmat: Order must contain all nonmissing labels in group and grouphat",
            ));
        }
    }
    Ok(())
}

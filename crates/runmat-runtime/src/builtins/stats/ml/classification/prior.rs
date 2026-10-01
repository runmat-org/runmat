use super::*;
use runmat_value::StructValue;

pub(super) fn parse_prior(
    prior: Option<&Value>,
    labels: &[ClassLabel],
    kind: LabelKind,
    rows_by_class: &[Vec<usize>],
) -> BuiltinResult<Vec<f64>> {
    let raw = match prior {
        None => vec![1.0; labels.len()],
        Some(value) if string_matches(value, "empirical") => rows_by_class
            .iter()
            .map(|rows| rows.len() as f64)
            .collect::<Vec<_>>(),
        Some(Value::Struct(st)) => prior_from_struct(st, labels, kind)?,
        Some(value) => numeric_vector(value, "prior")?,
    };
    if raw.len() != labels.len() {
        return Err(invalid_argument(
            "classify: prior must contain one probability per class",
        ));
    }
    if raw.iter().any(|value| !value.is_finite() || *value < 0.0) {
        return Err(invalid_argument(
            "classify: prior probabilities must be finite and nonnegative",
        ));
    }
    let total = raw.iter().sum::<f64>();
    if total <= 0.0 {
        return Err(invalid_argument(
            "classify: prior probabilities must have positive total",
        ));
    }
    Ok(raw.into_iter().map(|value| value / total).collect())
}

pub(super) fn prior_from_struct(
    st: &StructValue,
    labels: &[ClassLabel],
    kind: LabelKind,
) -> BuiltinResult<Vec<f64>> {
    let group_value = st
        .fields
        .get("group")
        .or_else(|| st.fields.get("Group"))
        .ok_or_else(|| invalid_argument("classify: prior struct must contain group field"))?
        .clone();
    let prob_value = st
        .fields
        .get("prob")
        .or_else(|| st.fields.get("Prob"))
        .ok_or_else(|| invalid_argument("classify: prior struct must contain prob field"))?;
    let groups = labels_from_value(group_value, "prior.group")?;
    if !label_kinds_compatible(groups.kind, kind) {
        return Err(invalid_argument(
            "classify: prior group labels must match group label type",
        ));
    }
    let probs = numeric_vector(prob_value, "prior.prob")?;
    if groups.labels.len() != probs.len() {
        return Err(invalid_argument(
            "classify: prior group and prob lengths must match",
        ));
    }
    let mut out = vec![0.0; labels.len()];
    for (group, prob) in groups.labels.iter().zip(probs.iter()) {
        let Some(index) = labels.iter().position(|label| same_label(label, group)) else {
            return Err(invalid_argument(
                "classify: prior struct contains unknown group label",
            ));
        };
        out[index] = *prob;
    }
    Ok(out)
}

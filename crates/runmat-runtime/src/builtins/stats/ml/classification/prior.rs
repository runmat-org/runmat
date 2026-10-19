//! One admission authority for public prior controls and deferred resident payloads.
use super::*;
use runmat_builtins::PriorKeyword;

pub(super) enum PriorSpec {
    Uniform,
    Empirical,
    Vector(PriorValues),
    ByGroup {
        groups: PriorGroups,
        probabilities: PriorValues,
    },
}

pub(super) enum PriorValues {
    Host(Vec<f64>),
    Resident(Value),
}

pub(super) enum PriorGroups {
    Host(LabelVector),
    Resident(Value),
}

impl PriorSpec {
    pub(super) fn prepare(value: Option<&Value>) -> BuiltinResult<Self> {
        let Some(value) = value else {
            return Ok(Self::Uniform);
        };
        if scalar_text(value, "prior")
            .ok()
            .and_then(|text| PriorKeyword::from_public(&text))
            .is_some()
        {
            return Ok(Self::Empirical);
        }
        if let Value::Struct(structure) = value {
            let group = structure
                .fields
                .get("group")
                .or_else(|| structure.fields.get("Group"))
                .ok_or_else(|| {
                    invalid_argument("classify: prior struct must contain group field")
                })?;
            let probability = structure
                .fields
                .get("prob")
                .or_else(|| structure.fields.get("Prob"))
                .ok_or_else(|| {
                    invalid_argument("classify: prior struct must contain prob field")
                })?;
            let groups = PriorGroups::prepare(group)?;
            let probabilities = PriorValues::prepare(probability, "prior.prob")?;
            if let (Some(group_count), Some(probability_count)) =
                (groups.known_len(), probabilities.known_len())
            {
                if group_count != probability_count {
                    return Err(invalid_argument(
                        "classify: prior group and prob lengths must match",
                    ));
                }
            }
            return Ok(Self::ByGroup {
                groups,
                probabilities,
            });
        }
        Ok(Self::Vector(PriorValues::prepare(value, "prior")?))
    }

    pub(super) async fn materialize(self) -> BuiltinResult<Self> {
        match self {
            Self::Vector(values) => Ok(Self::Vector(values.materialize("prior").await?)),
            Self::ByGroup {
                groups,
                probabilities,
            } => Ok(Self::ByGroup {
                groups: groups.materialize().await?,
                probabilities: probabilities.materialize("prior.prob").await?,
            }),
            other => Ok(other),
        }
    }

    pub(super) fn resolve(
        self,
        labels: &[ClassLabel],
        kind: LabelKind,
        rows_by_class: &[Vec<usize>],
    ) -> BuiltinResult<Vec<f64>> {
        let raw = match self {
            Self::Uniform => vec![1.0; labels.len()],
            Self::Empirical => rows_by_class.iter().map(|rows| rows.len() as f64).collect(),
            Self::Vector(values) => values.into_host()?,
            Self::ByGroup {
                groups,
                probabilities,
            } => {
                let groups = groups.into_host()?;
                if !label_kinds_compatible(groups.kind, kind) {
                    return Err(invalid_argument(
                        "classify: prior group labels must match group label type",
                    ));
                }
                let probabilities = probabilities.into_host()?;
                if groups.labels.len() != probabilities.len() {
                    return Err(invalid_argument(
                        "classify: prior group and prob lengths must match",
                    ));
                }
                let mut ordered = vec![0.0; labels.len()];
                for (group, probability) in groups.labels.iter().zip(probabilities) {
                    let Some(index) = labels.iter().position(|label| same_label(label, group))
                    else {
                        return Err(invalid_argument(
                            "classify: prior struct contains unknown group label",
                        ));
                    };
                    ordered[index] = probability;
                }
                ordered
            }
        };
        if raw.len() != labels.len() {
            return Err(invalid_argument(
                "classify: prior must contain one probability per class",
            ));
        }
        let (scale, scaled_total) = probability_scale(&raw)?;
        Ok(raw
            .into_iter()
            .map(|value| (value / scale) / scaled_total)
            .collect())
    }
}

impl PriorValues {
    fn prepare(value: &Value, role: &str) -> BuiltinResult<Self> {
        header::preflight_numeric_vector(value, role)?;
        match value {
            Value::GpuTensor(_) => Ok(Self::Resident(value.clone())),
            _ => {
                let values = numeric_vector(value, role)?;
                probability_scale(&values)?;
                Ok(Self::Host(values))
            }
        }
    }

    fn known_len(&self) -> Option<usize> {
        match self {
            Self::Host(values) => Some(values.len()),
            Self::Resident(Value::GpuTensor(handle)) => handle
                .shape()
                .iter()
                .try_fold(1usize, |count, dimension| count.checked_mul(*dimension)),
            Self::Resident(_) => None,
        }
    }

    async fn materialize(self, role: &str) -> BuiltinResult<Self> {
        match self {
            Self::Resident(value) => {
                let value = gathered(value).await?;
                Self::prepare(&value, role)
            }
            host => Ok(host),
        }
    }

    fn into_host(self) -> BuiltinResult<Vec<f64>> {
        match self {
            Self::Host(values) => Ok(values),
            Self::Resident(_) => Err(internal_error(
                "classify: prior probabilities were not materialized",
            )),
        }
    }
}

impl PriorGroups {
    fn prepare(value: &Value) -> BuiltinResult<Self> {
        if header::group_count(value)?.is_none() {
            return Err(invalid_argument("classify: prior.group must be numeric, logical, string, character, or cellstr labels"));
        }
        if matches!(value, Value::GpuTensor(_)) {
            return Ok(Self::Resident(value.clone()));
        }
        Ok(Self::Host(labels_from_value(value.clone(), "prior.group")?))
    }

    fn known_len(&self) -> Option<usize> {
        match self {
            Self::Host(groups) => Some(groups.labels.len()),
            Self::Resident(value) => header::group_count(value).ok().flatten(),
        }
    }

    async fn materialize(self) -> BuiltinResult<Self> {
        match self {
            Self::Resident(value) => {
                let value = gathered(value).await?;
                Self::prepare(&value)
            }
            host => Ok(host),
        }
    }

    fn into_host(self) -> BuiltinResult<LabelVector> {
        match self {
            Self::Host(groups) => Ok(groups),
            Self::Resident(_) => Err(internal_error(
                "classify: prior groups were not materialized",
            )),
        }
    }
}

fn probability_scale(probabilities: &[f64]) -> BuiltinResult<(f64, f64)> {
    if probabilities
        .iter()
        .any(|value| !value.is_finite() || *value < 0.0)
    {
        return Err(invalid_argument(
            "classify: prior probabilities must be finite and nonnegative",
        ));
    }
    let scale = probabilities.iter().copied().fold(0.0_f64, f64::max);
    if scale <= 0.0 {
        return Err(invalid_argument(
            "classify: prior probabilities must have positive total",
        ));
    }
    let scaled_total = probabilities.iter().map(|value| value / scale).sum::<f64>();
    Ok((scale, scaled_total))
}

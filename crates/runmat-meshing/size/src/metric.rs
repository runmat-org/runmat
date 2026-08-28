use std::cmp::Ordering;
use std::collections::{BTreeMap, BTreeSet};

use runmat_geometry_core::{PersistentEntityId, PersistentEntityKind};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MetricContractError {
    pub field: String,
    pub reason: String,
}

impl MetricContractError {
    fn invalid(field: impl Into<String>, reason: impl Into<String>) -> Self {
        Self {
            field: field.into(),
            reason: reason.into(),
        }
    }
}

impl std::fmt::Display for MetricContractError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "invalid {}: {}", self.field, self.reason)
    }
}

impl std::error::Error for MetricContractError {}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MetricTensor3 {
    pub xx: f64,
    pub yy: f64,
    pub zz: f64,
    pub xy: f64,
    pub xz: f64,
    pub yz: f64,
}

impl MetricTensor3 {
    pub fn isotropic_length_m(length_m: f64) -> Result<Self, MetricContractError> {
        if !length_m.is_finite() || length_m <= 0.0 {
            return Err(MetricContractError::invalid(
                "isotropic metric length",
                "must be finite and greater than zero",
            ));
        }
        let diagonal = 1.0 / (length_m * length_m);
        let metric = Self {
            xx: diagonal,
            yy: diagonal,
            zz: diagonal,
            xy: 0.0,
            xz: 0.0,
            yz: 0.0,
        };
        metric.validate()?;
        Ok(metric)
    }

    pub fn validate(&self) -> Result<(), MetricContractError> {
        for (field, value) in [
            ("metric.xx", self.xx),
            ("metric.yy", self.yy),
            ("metric.zz", self.zz),
            ("metric.xy", self.xy),
            ("metric.xz", self.xz),
            ("metric.yz", self.yz),
        ] {
            if !value.is_finite() {
                return Err(MetricContractError::invalid(field, "must be finite"));
            }
        }
        let leading_2 = self.xx * self.yy - self.xy * self.xy;
        let determinant = self.xx * self.yy * self.zz + 2.0 * self.xy * self.xz * self.yz
            - self.xx * self.yz * self.yz
            - self.yy * self.xz * self.xz
            - self.zz * self.xy * self.xy;
        if !leading_2.is_finite()
            || !determinant.is_finite()
            || self.xx <= 0.0
            || leading_2 <= 0.0
            || determinant <= 0.0
        {
            return Err(MetricContractError::invalid(
                "metric tensor",
                "must be symmetric positive definite",
            ));
        }
        Ok(())
    }

    pub fn conservative_minimum_length_m(&self) -> Result<f64, MetricContractError> {
        self.validate()?;
        // The maximum absolute row sum bounds the largest eigenvalue from above. Its reciprocal
        // square root is therefore a conservative lower bound on the finest directional length.
        let maximum_density = [
            self.xx + self.xy.abs() + self.xz.abs(),
            self.yy + self.xy.abs() + self.yz.abs(),
            self.zz + self.xz.abs() + self.yz.abs(),
        ]
        .into_iter()
        .fold(0.0_f64, f64::max);
        let length = maximum_density.sqrt().recip();
        if !length.is_finite() || length <= 0.0 {
            return Err(MetricContractError::invalid(
                "metric characteristic length",
                "must be finite and greater than zero",
            ));
        }
        Ok(length)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MetricCombinationRule {
    MostRestrictiveIntersection,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MetricSourceKind {
    Global,
    Region,
    Point,
    Curve,
    Face,
    Volume,
    Proximity,
    Feature,
    Load,
    Contact,
    SolutionIndicator,
}

impl MetricSourceKind {
    pub const fn canonical_rank(self) -> u8 {
        match self {
            Self::Global => 0,
            Self::Region => 1,
            Self::Point => 2,
            Self::Curve => 3,
            Self::Face => 4,
            Self::Volume => 5,
            Self::Proximity => 6,
            Self::Feature => 7,
            Self::Load => 8,
            Self::Contact => 9,
            Self::SolutionIndicator => 10,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "scope", rename_all = "snake_case", deny_unknown_fields)]
pub enum MetricContributionScope {
    Region { region_id: PersistentEntityId },
    Entity { entity_id: PersistentEntityId },
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MetricContribution {
    pub source: MetricSourceKind,
    pub scope: MetricContributionScope,
    pub metric: MetricTensor3,
}

impl MetricContribution {
    pub fn validate(&self) -> Result<(), MetricContractError> {
        self.metric.validate()?;
        if self.source == MetricSourceKind::Global {
            return Err(MetricContractError::invalid(
                "metric contribution source",
                "global sizing is represented only by global_metric",
            ));
        }
        match &self.scope {
            MetricContributionScope::Region { region_id } => {
                region_id
                    .validate()
                    .map_err(|error| MetricContractError::invalid(error.field, error.reason))?;
                if region_id.kind != PersistentEntityKind::Region {
                    return Err(MetricContractError::invalid(
                        "metric contribution region",
                        "region scope requires a persistent region identity",
                    ));
                }
            }
            MetricContributionScope::Entity { entity_id } => entity_id
                .validate()
                .map_err(|error| MetricContractError::invalid(error.field, error.reason))?,
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MetricFieldRequest {
    pub combination: MetricCombinationRule,
    pub global_metric: MetricTensor3,
    pub maximum_grading_ratio: f64,
    #[serde(default)]
    pub contributions: Vec<MetricContribution>,
}

impl MetricFieldRequest {
    pub fn validate(&self) -> Result<(), MetricContractError> {
        self.global_metric.validate()?;
        if !self.maximum_grading_ratio.is_finite() || self.maximum_grading_ratio < 1.0 {
            return Err(MetricContractError::invalid(
                "maximum metric grading ratio",
                "must be finite and at least one",
            ));
        }
        if self.contributions.len() > 65_536 {
            return Err(MetricContractError::invalid(
                "metric contributions",
                "must contain at most 65536 entries",
            ));
        }
        for contribution in &self.contributions {
            contribution.validate()?;
        }
        if self
            .contributions
            .windows(2)
            .any(|pair| compare_contributions(&pair[0], &pair[1]) != Ordering::Less)
        {
            return Err(MetricContractError::invalid(
                "metric contributions",
                "must be unique and canonically ordered by source, scope, identity, and tensor",
            ));
        }
        Ok(())
    }

    pub fn intersect_contributions(
        &self,
        additional: impl IntoIterator<Item = MetricContribution>,
    ) -> Result<Self, MetricContractError> {
        self.validate()?;
        let mut resolved = self.clone();
        for contribution in additional {
            contribution.validate()?;
            resolved.contributions.push(contribution);
        }
        resolved.contributions.sort_by(compare_contributions);
        resolved
            .contributions
            .dedup_by(|left, right| compare_contributions(left, right) == Ordering::Equal);
        resolved.validate()?;
        Ok(resolved)
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct ResolvedMetricEvaluation {
    pub metric: MetricTensor3,
    pub active_sources: Vec<MetricSourceKind>,
    pub applied_contribution_count: u32,
    pub clipped_contribution_count: u32,
    pub rejected_contribution_count: u32,
}

/// Stage-neutral deterministic metric resolver. Each meshing stage supplies only the exact
/// entities incident to its query; combination and provenance remain owned here.
#[derive(Debug, Clone, PartialEq)]
pub struct ResolvedMetricField {
    request: MetricFieldRequest,
}

impl ResolvedMetricField {
    pub fn new(request: &MetricFieldRequest) -> Result<Self, MetricContractError> {
        request.validate()?;
        Ok(Self {
            request: request.clone(),
        })
    }

    pub fn resolve(
        &self,
        incident_entities: &BTreeSet<PersistentEntityId>,
    ) -> Result<ResolvedMetricEvaluation, MetricContractError> {
        let mut metric = self.request.global_metric;
        let mut sources = BTreeMap::from([(
            MetricSourceKind::Global.canonical_rank(),
            MetricSourceKind::Global,
        )]);
        let mut applied_contribution_count = 0u32;
        for contribution in &self.request.contributions {
            let entity_id = match &contribution.scope {
                MetricContributionScope::Region { region_id } => region_id,
                MetricContributionScope::Entity { entity_id } => entity_id,
            };
            if incident_entities.contains(entity_id) {
                metric = add_metric(metric, contribution.metric).ok_or_else(|| {
                    MetricContractError::invalid(
                        "resolved metric intersection",
                        "applicable tensors overflow their conservative intersection",
                    )
                })?;
                sources.insert(contribution.source.canonical_rank(), contribution.source);
                applied_contribution_count = applied_contribution_count.saturating_add(1);
            }
        }
        metric.validate()?;
        Ok(ResolvedMetricEvaluation {
            metric,
            active_sources: sources.into_values().collect(),
            applied_contribution_count,
            clipped_contribution_count: 0,
            rejected_contribution_count: 0,
        })
    }
}

fn add_metric(left: MetricTensor3, right: MetricTensor3) -> Option<MetricTensor3> {
    let values = [
        left.xx + right.xx,
        left.yy + right.yy,
        left.zz + right.zz,
        left.xy + right.xy,
        left.xz + right.xz,
        left.yz + right.yz,
    ];
    values
        .iter()
        .all(|value| value.is_finite())
        .then_some(MetricTensor3 {
            xx: values[0],
            yy: values[1],
            zz: values[2],
            xy: values[3],
            xz: values[4],
            yz: values[5],
        })
}

fn compare_contributions(left: &MetricContribution, right: &MetricContribution) -> Ordering {
    left.source
        .canonical_rank()
        .cmp(&right.source.canonical_rank())
        .then_with(|| compare_scopes(&left.scope, &right.scope))
        .then_with(|| compare_metrics(left.metric, right.metric))
}

fn compare_scopes(left: &MetricContributionScope, right: &MetricContributionScope) -> Ordering {
    match (left, right) {
        (
            MetricContributionScope::Region { region_id: left },
            MetricContributionScope::Region { region_id: right },
        )
        | (
            MetricContributionScope::Entity { entity_id: left },
            MetricContributionScope::Entity { entity_id: right },
        ) => left.cmp(right),
        (MetricContributionScope::Region { .. }, MetricContributionScope::Entity { .. }) => {
            Ordering::Less
        }
        (MetricContributionScope::Entity { .. }, MetricContributionScope::Region { .. }) => {
            Ordering::Greater
        }
    }
}

fn compare_metrics(left: MetricTensor3, right: MetricTensor3) -> Ordering {
    for (left, right) in [left.xx, left.yy, left.zz, left.xy, left.xz, left.yz]
        .into_iter()
        .zip([right.xx, right.yy, right.zz, right.xy, right.xz, right.yz])
    {
        let ordering = left.total_cmp(&right);
        if ordering != Ordering::Equal {
            return ordering;
        }
    }
    Ordering::Equal
}

#[cfg(test)]
mod tests;

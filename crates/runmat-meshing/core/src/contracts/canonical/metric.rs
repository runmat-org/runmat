use serde::{Deserialize, Serialize};

use super::{validate_finite, MeshingContractError, PersistentEntityId};

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
    pub fn isotropic_length_m(length_m: f64) -> Result<Self, MeshingContractError> {
        validate_finite("isotropic metric length", length_m)?;
        if length_m <= 0.0 {
            return Err(MeshingContractError::invalid(
                "isotropic metric length",
                "must be greater than zero",
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

    pub fn validate(&self) -> Result<(), MeshingContractError> {
        for (field, value) in [
            ("metric.xx", self.xx),
            ("metric.yy", self.yy),
            ("metric.zz", self.zz),
            ("metric.xy", self.xy),
            ("metric.xz", self.xz),
            ("metric.yz", self.yz),
        ] {
            validate_finite(field, value)?;
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
            return Err(MeshingContractError::invalid(
                "metric tensor",
                "must be symmetric positive definite",
            ));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MetricCombinationRule {
    MostRestrictiveIntersection,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
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

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "scope", rename_all = "snake_case", deny_unknown_fields)]
pub enum MetricContributionScope {
    Global,
    Region { region_id: String },
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
    pub fn validate(&self) -> Result<(), MeshingContractError> {
        self.metric.validate()?;
        match &self.scope {
            MetricContributionScope::Global => {
                if self.source != MetricSourceKind::Global {
                    return Err(MeshingContractError::invalid(
                        "metric contribution scope",
                        "global scope requires the global source kind",
                    ));
                }
            }
            MetricContributionScope::Region { region_id } => {
                super::validate_token("metric region id", region_id, 512)?;
            }
            MetricContributionScope::Entity { entity_id } => entity_id.validate()?,
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
    pub fn validate(&self) -> Result<(), MeshingContractError> {
        self.global_metric.validate()?;
        validate_finite("maximum metric grading ratio", self.maximum_grading_ratio)?;
        if self.maximum_grading_ratio < 1.0 {
            return Err(MeshingContractError::invalid(
                "maximum metric grading ratio",
                "must be at least one",
            ));
        }
        if self.contributions.len() > 65_536 {
            return Err(MeshingContractError::invalid(
                "metric contributions",
                "must contain at most 65536 entries",
            ));
        }
        for contribution in &self.contributions {
            contribution.validate()?;
        }
        Ok(())
    }
}

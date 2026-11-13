use serde::{Deserialize, Serialize};

use super::{validate_finite, validate_token, MeshingContractError, PersistentEntityId};

pub const MESHING_FAILURE_SCHEMA_VERSION: u16 = 2;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MeshingStageKind {
    GeometryAdmission,
    Healing,
    Sizing,
    CurveMesh,
    SurfaceMesh,
    ProtectedBoundaryComplex,
    Tetrahedralization,
    ConstraintRecovery,
    Refinement,
    Optimization,
    OrderElevation,
    Validation,
    Serialization,
    Publication,
}

impl MeshingStageKind {
    pub const ALL: [Self; 14] = [
        Self::GeometryAdmission,
        Self::Healing,
        Self::Sizing,
        Self::CurveMesh,
        Self::SurfaceMesh,
        Self::ProtectedBoundaryComplex,
        Self::Tetrahedralization,
        Self::ConstraintRecovery,
        Self::Refinement,
        Self::Optimization,
        Self::OrderElevation,
        Self::Validation,
        Self::Serialization,
        Self::Publication,
    ];

    /// Canonical operation reported when this stage fails before a more specific operation exists.
    pub const fn operation(self) -> MeshingOperation {
        match self {
            Self::GeometryAdmission => MeshingOperation::AdmitGeometry,
            Self::Healing => MeshingOperation::HealGeometry,
            Self::Sizing => MeshingOperation::ResolveMetric,
            Self::CurveMesh => MeshingOperation::DiscretizeCurve,
            Self::SurfaceMesh => MeshingOperation::TriangulateSurface,
            Self::ProtectedBoundaryComplex => MeshingOperation::BuildProtectedBoundaryComplex,
            Self::Tetrahedralization => MeshingOperation::Tetrahedralize,
            Self::ConstraintRecovery => MeshingOperation::RecoverConstraint,
            Self::Refinement => MeshingOperation::Refine,
            Self::Optimization => MeshingOperation::Optimize,
            Self::OrderElevation => MeshingOperation::ElevateOrder,
            Self::Validation => MeshingOperation::Validate,
            Self::Serialization => MeshingOperation::Serialize,
            Self::Publication => MeshingOperation::Publish,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MeshingOperation {
    AdmitGeometry,
    HealGeometry,
    ResolveMetric,
    DiscretizeCurve,
    TriangulateSurface,
    BuildProtectedBoundaryComplex,
    Tetrahedralize,
    RecoverConstraint,
    Refine,
    Optimize,
    ElevateOrder,
    Validate,
    Serialize,
    Publish,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MeshingFailureCategory {
    InvalidGeometry,
    HealingLimitExceeded,
    UnsatisfiableConstraints,
    SizingConflict,
    QualityTargetUnreachable,
    NodeBudgetExceeded,
    ElementBudgetExceeded,
    MemoryBudgetExceeded,
    ScratchBudgetExceeded,
    TimeBudgetExceeded,
    ArtifactBudgetExceeded,
    SearchWorkBudgetExceeded,
    RecursionBudgetExceeded,
    IterationBudgetExceeded,
    Cancelled,
    NumericalAccounting,
    NumericalFailure,
    InternalInvariantViolation,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum GeometricWitness {
    Point {
        #[serde(with = "runmat_types::physical::canonical_storage")]
        coordinates_m:
            runmat_types::physical::Quantity<runmat_types::physical::kind::Position, [f64; 3]>,
    },
    Segment {
        endpoints_m: [[f64; 3]; 2],
    },
    Triangle {
        vertices_m: [[f64; 3]; 3],
    },
    CurveParameter {
        parameter: f64,
    },
    SurfaceParameter {
        uv: [f64; 2],
    },
}

impl GeometricWitness {
    fn validate(&self) -> Result<(), MeshingContractError> {
        let values: Vec<f64> = match self {
            Self::Point { coordinates_m } => coordinates_m.canonical_value().to_vec(),
            Self::Segment { endpoints_m } => endpoints_m.iter().flatten().copied().collect(),
            Self::Triangle { vertices_m } => vertices_m.iter().flatten().copied().collect(),
            Self::CurveParameter { parameter } => vec![*parameter],
            Self::SurfaceParameter { uv } => uv.to_vec(),
        };
        for value in values {
            validate_finite("geometric witness", value)?;
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(
    tag = "type",
    content = "value",
    rename_all = "snake_case",
    deny_unknown_fields
)]
pub enum MeshingDiagnosticValue {
    Count(u64),
    Integer(i64),
    Scalar(f64),
    Text(String),
    Interval { minimum: f64, maximum: f64 },
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MeshingDiagnosticEntry {
    pub name: String,
    pub value: MeshingDiagnosticValue,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub unit: Option<String>,
}

impl MeshingDiagnosticEntry {
    fn validate(&self) -> Result<(), MeshingContractError> {
        validate_token("diagnostic name", &self.name, 128)?;
        if let Some(unit) = &self.unit {
            validate_token("diagnostic unit", unit, 64)?;
        }
        match &self.value {
            MeshingDiagnosticValue::Scalar(value) => validate_finite("diagnostic scalar", *value),
            MeshingDiagnosticValue::Interval { minimum, maximum } => {
                validate_finite("diagnostic interval minimum", *minimum)?;
                validate_finite("diagnostic interval maximum", *maximum)?;
                if minimum > maximum {
                    return Err(MeshingContractError::invalid(
                        "diagnostic interval",
                        "minimum must not exceed maximum",
                    ));
                }
                Ok(())
            }
            MeshingDiagnosticValue::Text(value) => validate_token("diagnostic text", value, 4096),
            MeshingDiagnosticValue::Count(_) | MeshingDiagnosticValue::Integer(_) => Ok(()),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MeshingFailure {
    pub schema_version: u16,
    pub category: MeshingFailureCategory,
    pub stage: MeshingStageKind,
    pub operation: MeshingOperation,
    #[serde(default)]
    pub entity_ids: Vec<PersistentEntityId>,
    #[serde(default)]
    pub witnesses: Vec<GeometricWitness>,
    #[serde(default)]
    pub request_values: Vec<MeshingDiagnosticEntry>,
    #[serde(default)]
    pub achieved_values: Vec<MeshingDiagnosticEntry>,
    /// The original numerical-admission error, present only for accounting failures.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub accounting_error: Option<runmat_simulation_usage::LedgerError>,
    pub remediation: String,
}

impl MeshingFailure {
    pub fn validate(&self) -> Result<(), MeshingContractError> {
        if self.schema_version != MESHING_FAILURE_SCHEMA_VERSION {
            return Err(MeshingContractError::invalid(
                "meshing failure schema version",
                format!("expected {MESHING_FAILURE_SCHEMA_VERSION}"),
            ));
        }
        if self.entity_ids.len() > 4096
            || self.witnesses.len() > 4096
            || self.request_values.len() > 4096
            || self.achieved_values.len() > 4096
        {
            return Err(MeshingContractError::invalid(
                "meshing failure diagnostics",
                "each diagnostic collection is limited to 4096 entries",
            ));
        }
        if self.operation.stage() != self.stage {
            return Err(MeshingContractError::invalid(
                "meshing failure operation",
                "operation does not belong to the reported stage",
            ));
        }
        if (self.category == MeshingFailureCategory::NumericalAccounting)
            != self.accounting_error.is_some()
        {
            return Err(MeshingContractError::invalid(
                "meshing failure accounting error",
                "numerical accounting failures must retain exactly one typed accounting cause",
            ));
        }
        for entity in &self.entity_ids {
            entity.validate()?;
        }
        for witness in &self.witnesses {
            witness.validate()?;
        }
        validate_diagnostics("request diagnostics", &self.request_values)?;
        validate_diagnostics("achieved diagnostics", &self.achieved_values)?;
        validate_token("meshing failure remediation", &self.remediation, 8192)
    }

    /// Returns the most specific textual diagnostic recorded by the failing stage.
    ///
    /// Stage adapters retain the originating kernel error in `achieved_values` so
    /// callers can inspect failures without depending on an implementation-specific
    /// error type. Prefer the last text entry: diagnostics are sorted by name and
    /// adapters conventionally place their stage failure detail there.
    pub fn detail(&self) -> Option<&str> {
        self.achieved_values
            .iter()
            .rev()
            .find_map(|entry| match &entry.value {
                MeshingDiagnosticValue::Text(value) if !value.is_empty() => Some(value.as_str()),
                _ => None,
            })
    }
}

impl MeshingOperation {
    pub const fn stage(self) -> MeshingStageKind {
        match self {
            Self::AdmitGeometry => MeshingStageKind::GeometryAdmission,
            Self::HealGeometry => MeshingStageKind::Healing,
            Self::ResolveMetric => MeshingStageKind::Sizing,
            Self::DiscretizeCurve => MeshingStageKind::CurveMesh,
            Self::TriangulateSurface => MeshingStageKind::SurfaceMesh,
            Self::BuildProtectedBoundaryComplex => MeshingStageKind::ProtectedBoundaryComplex,
            Self::Tetrahedralize => MeshingStageKind::Tetrahedralization,
            Self::RecoverConstraint => MeshingStageKind::ConstraintRecovery,
            Self::Refine => MeshingStageKind::Refinement,
            Self::Optimize => MeshingStageKind::Optimization,
            Self::ElevateOrder => MeshingStageKind::OrderElevation,
            Self::Validate => MeshingStageKind::Validation,
            Self::Serialize => MeshingStageKind::Serialization,
            Self::Publish => MeshingStageKind::Publication,
        }
    }
}

fn validate_diagnostics(
    field: &str,
    entries: &[MeshingDiagnosticEntry],
) -> Result<(), MeshingContractError> {
    for entry in entries {
        entry.validate()?;
    }
    if entries
        .windows(2)
        .any(|pair| pair[0].name.as_str() >= pair[1].name.as_str())
    {
        return Err(MeshingContractError::invalid(
            field,
            "entries must be sorted by unique name",
        ));
    }
    Ok(())
}

impl std::fmt::Display for MeshingFailure {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            formatter,
            "{:?} during {:?}/{:?}",
            self.category, self.stage, self.operation
        )?;
        if let Some(detail) = self.detail() {
            write!(formatter, ": {detail}")?;
        }
        for achieved in &self.achieved_values {
            if matches!(achieved.value, MeshingDiagnosticValue::Text(_)) {
                continue;
            }
            write!(
                formatter,
                "; achieved {}={}",
                achieved.name,
                diagnostic_value(&achieved.value)
            )?;
            if let Some(required) = self
                .request_values
                .iter()
                .find(|required| required.name == achieved.name)
            {
                write!(formatter, " (limit {})", diagnostic_value(&required.value))?;
            }
        }
        write!(formatter, "; remediation: {}", self.remediation)
    }
}

fn diagnostic_value(value: &MeshingDiagnosticValue) -> String {
    match value {
        MeshingDiagnosticValue::Count(value) => value.to_string(),
        MeshingDiagnosticValue::Integer(value) => value.to_string(),
        MeshingDiagnosticValue::Scalar(value) => value.to_string(),
        MeshingDiagnosticValue::Text(value) => value.clone(),
        MeshingDiagnosticValue::Interval { minimum, maximum } => {
            format!("[{minimum}, {maximum}]")
        }
    }
}

impl std::error::Error for MeshingFailure {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        self.accounting_error
            .as_ref()
            .map(|error| error as &(dyn std::error::Error + 'static))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn display_includes_retained_kernel_detail() {
        let failure = MeshingFailure {
            schema_version: MESHING_FAILURE_SCHEMA_VERSION,
            category: MeshingFailureCategory::InvalidGeometry,
            stage: MeshingStageKind::CurveMesh,
            operation: MeshingOperation::DiscretizeCurve,
            entity_ids: Vec::new(),
            witnesses: Vec::new(),
            request_values: Vec::new(),
            achieved_values: vec![MeshingDiagnosticEntry {
                name: "curve_failure".into(),
                value: MeshingDiagnosticValue::Text("edge evaluation failed".into()),
                unit: None,
            }],
            accounting_error: None,
            remediation: "repair the exact boundary".into(),
        };

        assert_eq!(
            failure.to_string(),
            "InvalidGeometry during CurveMesh/DiscretizeCurve: edge evaluation failed; \
             remediation: repair the exact boundary"
        );
    }

    #[test]
    fn display_remains_useful_without_text_diagnostics() {
        let failure = MeshingFailure {
            schema_version: MESHING_FAILURE_SCHEMA_VERSION,
            category: MeshingFailureCategory::ElementBudgetExceeded,
            stage: MeshingStageKind::Tetrahedralization,
            operation: MeshingOperation::Tetrahedralize,
            entity_ids: Vec::new(),
            witnesses: Vec::new(),
            request_values: Vec::new(),
            achieved_values: Vec::new(),
            accounting_error: None,
            remediation: "raise the element budget".into(),
        };

        assert_eq!(
            failure.to_string(),
            "ElementBudgetExceeded during Tetrahedralization/Tetrahedralize; \
             remediation: raise the element budget"
        );
    }

    #[test]
    fn accounting_failure_keeps_its_typed_cause_across_serialization() {
        let mut failure = MeshingFailure {
            schema_version: MESHING_FAILURE_SCHEMA_VERSION,
            category: MeshingFailureCategory::NumericalAccounting,
            stage: MeshingStageKind::OrderElevation,
            operation: MeshingOperation::ElevateOrder,
            entity_ids: Vec::new(),
            witnesses: Vec::new(),
            request_values: Vec::new(),
            achieved_values: Vec::new(),
            accounting_error: Some(runmat_simulation_usage::LedgerError::CountOverflow),
            remediation: "raise the operation spending cap".into(),
        };
        failure.validate().unwrap();
        let encoded = serde_json::to_vec(&failure).unwrap();
        let decoded: MeshingFailure = serde_json::from_slice(&encoded).unwrap();
        assert_eq!(decoded.accounting_error, failure.accounting_error);
        assert_eq!(
            decoded.category,
            MeshingFailureCategory::NumericalAccounting
        );
        failure.accounting_error = None;
        assert!(failure.validate().is_err());
        failure.category = MeshingFailureCategory::InvalidGeometry;
        failure.accounting_error = Some(runmat_simulation_usage::LedgerError::CountOverflow);
        assert!(failure.validate().is_err());
    }
}

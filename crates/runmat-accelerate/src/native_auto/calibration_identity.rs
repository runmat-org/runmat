//! Qualification of existing calibration data, not a second timing registry.
use super::*;

/// Closed versioned interpretation of imported rate denominators. Text exists
/// only at serialization boundaries, never as executable policy lookup.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum CalibrationUnitPolicy {
    OriginalScanMacV1,
    DiagnosticResidualV1,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum CalibrationSource {
    AutomaticTemplates,
    ImportedOriginalTemplates,
    ResidualDiagnostic,
}

impl CalibrationRecord {
    pub(super) fn matches(&self, info: &ApiDeviceInfo, precision: ProviderPrecision) -> bool {
        self.version == CALIBRATION_VERSION
            && self.precision == precision
            && self.provider.name == info.name
            && self.provider.vendor == info.vendor
            && self.provider.backend == info.backend
            && self.provider.device_id == info.device_id
            && [
                self.thresholds.cpu_elem_per_elem,
                self.thresholds.cpu_reduction_per_elem,
                self.thresholds.cpu_matmul_per_flop,
            ]
            .into_iter()
            .all(|rate| rate.is_finite() && rate > 0.0)
    }
}

impl CalibrationSample {
    pub(super) fn require_current(
        &self,
        info: &ApiDeviceInfo,
        precision: ProviderPrecision,
    ) -> Result<()> {
        if self.schema_version != CALIBRATION_VERSION
            || self.precision != Some(precision)
            || self.unit_policy != Some(CalibrationUnitPolicy::OriginalScanMacV1)
            || self.provider.is_none()
            || self.provider_conflict
        {
            return Err(anyhow!("unqualified calibration sample: schema 2, exact provider/precision and original-scan-mac-v1 unit policy are required; legacy category measurements are not current execution evidence"));
        }
        let provider = self
            .provider
            .as_ref()
            .expect("checked calibration provider");
        if provider.name != info.name
            || provider.vendor != info.vendor
            || provider.backend != info.backend
            || provider.device_id != info.device_id
        {
            return Err(anyhow!(
                "calibration sample provider identity does not match the selected provider"
            ));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn unit_policy_is_closed_and_legacy_residuals_are_not_current_templates() {
        assert_eq!(
            serde_json::from_str::<CalibrationUnitPolicy>("\"diagnostic-residual-v1\"").unwrap(),
            CalibrationUnitPolicy::DiagnosticResidualV1
        );
        assert!(serde_json::from_str::<CalibrationUnitPolicy>("\"arbitrary-policy\"").is_err());
        let legacy: CalibrationSample = serde_json::from_str(r#"{"runs":1}"#).unwrap();
        let info = ApiDeviceInfo {
            device_id: 1,
            name: "fixture".into(),
            vendor: "fixture".into(),
            memory_bytes: None,
            backend: None,
        };
        assert!(legacy
            .require_current(&info, ProviderPrecision::F32)
            .is_err());
        let diagnostic: CalibrationSample = serde_json::from_str(
            r#"{"schema_version":2,"unit_policy":"diagnostic-residual-v1","runs":1}"#,
        )
        .unwrap();
        assert!(diagnostic
            .require_current(&info, ProviderPrecision::F64)
            .is_err());
    }
}

use serde::{Deserialize, Serialize};

use super::{invalid, token, Ap242Measure};
use crate::{GeometryContractError, GeometryDigest};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "kind")]
pub enum Ap242Change {
    SetParameter {
        parameter_id: String,
        previous: Ap242Measure,
        value: Ap242Measure,
    },
    SetEngineeringProperty {
        property_id: String,
        previous: Ap242Measure,
        value: Ap242Measure,
    },
    SetMaterialAssociation {
        association_id: String,
        material_id: String,
    },
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Ap242ChangeSet {
    pub source_digest: GeometryDigest,
    pub changes: Vec<Ap242Change>,
}

impl Ap242ChangeSet {
    pub fn validate(&self) -> Result<(), GeometryContractError> {
        self.source_digest
            .validate_nonzero("AP242 change-set source digest")?;
        if self.changes.is_empty() {
            return Err(invalid(
                "AP242 change set",
                "must contain at least one typed change",
            ));
        }
        for change in &self.changes {
            match change {
                Ap242Change::SetParameter {
                    parameter_id,
                    previous,
                    value,
                } => {
                    token("AP242 changed parameter", parameter_id)?;
                    previous.validate()?;
                    value.validate()?;
                    if previous.kind() != value.kind() {
                        return Err(invalid(
                            "AP242 parameter change",
                            "previous and replacement dimensions must match",
                        ));
                    }
                }
                Ap242Change::SetEngineeringProperty {
                    property_id,
                    previous,
                    value,
                } => {
                    token("AP242 changed property", property_id)?;
                    previous.validate()?;
                    value.validate()?;
                    if previous.kind() != value.kind() {
                        return Err(invalid(
                            "AP242 property change",
                            "previous and replacement dimensions must match",
                        ));
                    }
                }
                Ap242Change::SetMaterialAssociation {
                    association_id,
                    material_id,
                } => {
                    token("AP242 changed association", association_id)?;
                    token("AP242 replacement material", material_id)?;
                }
            }
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Ap242ExportDisposition {
    Applied,
    SidecarOnly,
    Rejected,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Ap242ExportItemReport {
    pub change_index: u64,
    pub disposition: Ap242ExportDisposition,
    pub detail: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Ap242ExportReport {
    pub source_digest: GeometryDigest,
    pub output_digest: Option<GeometryDigest>,
    pub source_preserved: bool,
    pub lossless: bool,
    pub items: Vec<Ap242ExportItemReport>,
}

impl Ap242ExportReport {
    pub fn validate(&self) -> Result<(), GeometryContractError> {
        self.source_digest
            .validate_nonzero("AP242 export source digest")?;
        if let Some(output_digest) = self.output_digest {
            output_digest.validate_nonzero("AP242 export output digest")?;
        }
        if self.items.is_empty() {
            return Err(invalid(
                "AP242 export report",
                "must report every requested change",
            ));
        }
        for (index, item) in self.items.iter().enumerate() {
            if item.change_index != index as u64 {
                return Err(invalid(
                    "AP242 export report",
                    "change indexes must be contiguous and ordered",
                ));
            }
            token("AP242 export item detail", &item.detail)?;
        }
        if self.lossless
            && (!self.source_preserved
                || self
                    .items
                    .iter()
                    .any(|item| item.disposition != Ap242ExportDisposition::Applied))
        {
            return Err(invalid(
                "AP242 export report",
                "lossless export requires preserved source and every change applied",
            ));
        }
        Ok(())
    }
}

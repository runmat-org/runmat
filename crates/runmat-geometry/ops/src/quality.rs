use runmat_geometry_core::GeometryAsset;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct QualityReport {
    pub warnings: Vec<String>,
    pub errors: Vec<String>,
}

pub fn evaluate_quality(asset: &GeometryAsset) -> QualityReport {
    let mut warnings = Vec::new();
    let mut errors = Vec::new();

    if !asset.source.unit_evidence.physical_scale_resolved() {
        warnings.push("geometry source units are not verified".to_string());
    }
    if asset.meshes.is_empty() {
        errors.push("geometry contains no meshes".to_string());
    }

    QualityReport { warnings, errors }
}

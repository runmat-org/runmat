use runmat_geometry_core::{GeometryAsset, UnitSystem};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct QualityReport {
    pub warnings: Vec<String>,
    pub errors: Vec<String>,
}

pub fn evaluate_quality(asset: &GeometryAsset) -> QualityReport {
    let mut warnings = Vec::new();
    let mut errors = Vec::new();

    if asset.units == UnitSystem::Unspecified {
        warnings.push("geometry units are unspecified".to_string());
    }
    if asset.meshes.is_empty() {
        errors.push("geometry contains no meshes".to_string());
    }

    QualityReport { warnings, errors }
}

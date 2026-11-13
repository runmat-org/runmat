use super::{AuthoredModelError, AuthoredRegionSource};
use crate::AuthoredInputType;
use runmat_geometry_core::GeometryRegionRole;

impl AuthoredRegionSource {
    pub fn role(&self) -> GeometryRegionRole {
        match self {
            Self::Geometry(region) => region.role,
            Self::ModelInput(input) => match input.value_type {
                AuthoredInputType::Region(role) => role,
                _ => GeometryRegionRole::Vertex,
            },
        }
    }

    pub fn validate(&self) -> Result<(), AuthoredModelError> {
        match self {
            Self::Geometry(region) => {
                required("region module path", &region.module_path)?;
                required("region geometry id", &region.geometry_id)?;
                required("region id", &region.region_id)
            }
            Self::ModelInput(input) => {
                required("model input id", &input.input_id)?;
                if !matches!(input.value_type, AuthoredInputType::Region(_)) {
                    Err(AuthoredModelError::RegionInputRequired)
                } else {
                    Ok(())
                }
            }
        }
    }
}

fn required(field: &'static str, value: &str) -> Result<(), AuthoredModelError> {
    if value.trim().is_empty() {
        Err(AuthoredModelError::InvalidToken(field))
    } else {
        Ok(())
    }
}

use std::collections::BTreeSet;

use serde::{Deserialize, Serialize};

use super::model_authoring::{
    AuthoredMaterialSource, AuthoredPropertyTarget, AuthoredRegionSource,
};
use super::structure::{
    HomogeneousShellReferenceSurface, PlyOrderDirection, ShellReferenceSurface,
};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "kind")]
pub enum AuthoredSectionDefinition {
    GeneralBeam(GeneralBeamSection),
    RectangularBeam(RectangularBeamSection),
    CircularBeam(CircularBeamSection),
    TubularBeam(TubularBeamSection),
    HomogeneousShell(HomogeneousShellSection),
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GeneralBeamSection {
    pub section_id: String,
    #[serde(with = "runmat_types::physical::canonical_scalar")]
    pub area_m2: runmat_types::physical::Area,
    #[serde(with = "runmat_types::physical::canonical_scalar")]
    pub iy_m4: runmat_types::physical::SecondMomentOfArea,
    #[serde(with = "runmat_types::physical::canonical_scalar")]
    pub iz_m4: runmat_types::physical::SecondMomentOfArea,
    #[serde(with = "runmat_types::physical::canonical_scalar")]
    pub torsion_j_m4: runmat_types::physical::SecondMomentOfArea,
    #[serde(with = "runmat_types::physical::canonical_scalar")]
    pub outer_fiber_y_m: runmat_types::physical::Length,
    #[serde(with = "runmat_types::physical::canonical_scalar")]
    pub outer_fiber_z_m: runmat_types::physical::Length,
    #[serde(with = "runmat_types::physical::canonical_scalar")]
    pub torsion_outer_radius_m: runmat_types::physical::Length,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RectangularBeamSection {
    pub section_id: String,
    #[serde(with = "runmat_types::physical::canonical_scalar")]
    pub width_m: runmat_types::physical::Length,
    #[serde(with = "runmat_types::physical::canonical_scalar")]
    pub height_m: runmat_types::physical::Length,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CircularBeamSection {
    pub section_id: String,
    #[serde(with = "runmat_types::physical::canonical_scalar")]
    pub radius_m: runmat_types::physical::Length,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TubularBeamSection {
    pub section_id: String,
    #[serde(with = "runmat_types::physical::canonical_scalar")]
    pub outer_radius_m: runmat_types::physical::Length,
    #[serde(with = "runmat_types::physical::canonical_scalar")]
    pub inner_radius_m: runmat_types::physical::Length,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct HomogeneousShellSection {
    pub section_id: String,
    #[serde(with = "runmat_types::physical::canonical_scalar")]
    pub thickness_m: runmat_types::physical::Length,
    pub reference_surface: HomogeneousShellReferenceSurface,
    pub shear_correction: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AuthoredLayupDefinition {
    pub layup_id: String,
    pub reference_orientation: AuthoredPropertyDefinitionRef,
    pub ply_order: PlyOrderDirection,
    pub reference_surface: ShellReferenceSurface,
    pub shear_correction: Option<f64>,
    pub plies: Vec<AuthoredPly>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AuthoredPly {
    pub ply_id: String,
    pub material: AuthoredMaterialSource,
    #[serde(with = "runmat_types::physical::canonical_scalar")]
    pub thickness_m: runmat_types::physical::Length,
    #[serde(with = "runmat_types::physical::canonical_storage")]
    pub angle_rad: runmat_types::physical::Quantity<runmat_types::physical::kind::Angle, f64>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "kind")]
pub enum AuthoredOrientationDefinition {
    GlobalCartesian { orientation_id: String },
    Cartesian(CartesianOrientation),
    Cylindrical(AxisOrientation),
    Spherical(AxisOrientation),
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CartesianOrientation {
    pub orientation_id: String,
    #[serde(with = "runmat_types::physical::canonical_storage")]
    pub origin_m:
        runmat_types::physical::Quantity<runmat_types::physical::kind::Position, [f64; 3]>,
    pub x_axis: [f64; 3],
    pub y_axis: [f64; 3],
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AxisOrientation {
    pub orientation_id: String,
    #[serde(with = "runmat_types::physical::canonical_storage")]
    pub origin_m:
        runmat_types::physical::Quantity<runmat_types::physical::kind::Position, [f64; 3]>,
    pub axis: [f64; 3],
    pub reference_direction: [f64; 3],
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AuthoredPropertyDefinitionRef {
    pub module_path: String,
    pub model_id: String,
    pub property_id: String,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "kind", content = "reference")]
pub enum AuthoredSectionSource {
    Definition(AuthoredPropertyDefinitionRef),
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "kind", content = "reference")]
pub enum AuthoredLayupSource {
    Definition(AuthoredPropertyDefinitionRef),
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "kind", content = "reference")]
pub enum AuthoredOrientationSource {
    Definition(AuthoredPropertyDefinitionRef),
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "assignment")]
pub enum AuthoredPropertyAssignment {
    Section {
        section: AuthoredSectionSource,
        target: AuthoredPropertyTarget,
    },
    Layup {
        layup: AuthoredLayupSource,
        target: AuthoredRegionSource,
    },
    Orientation {
        orientation: AuthoredOrientationSource,
        target: AuthoredPropertyTarget,
    },
}

impl AuthoredSectionDefinition {
    pub fn id(&self) -> &str {
        match self {
            Self::GeneralBeam(value) => &value.section_id,
            Self::RectangularBeam(value) => &value.section_id,
            Self::CircularBeam(value) => &value.section_id,
            Self::TubularBeam(value) => &value.section_id,
            Self::HomogeneousShell(value) => &value.section_id,
        }
    }

    pub fn validate(&self) -> Result<(), AuthoredPropertyError> {
        token("section id", self.id())?;
        match self {
            Self::GeneralBeam(value) => {
                positive("area", value.area_m2.canonical_value())?;
                positive("iy", value.iy_m4.canonical_value())?;
                positive("iz", value.iz_m4.canonical_value())?;
                positive("torsion constant", value.torsion_j_m4.canonical_value())?;
                nonnegative("outer fiber y", value.outer_fiber_y_m.canonical_value())?;
                nonnegative("outer fiber z", value.outer_fiber_z_m.canonical_value())?;
                nonnegative(
                    "torsion outer radius",
                    value.torsion_outer_radius_m.canonical_value(),
                )
            }
            Self::RectangularBeam(value) => {
                positive("width", value.width_m.canonical_value())?;
                positive("height", value.height_m.canonical_value())
            }
            Self::CircularBeam(value) => positive("radius", value.radius_m.canonical_value()),
            Self::TubularBeam(value) => {
                positive("outer radius", value.outer_radius_m.canonical_value())?;
                positive("inner radius", value.inner_radius_m.canonical_value())?;
                if value.inner_radius_m.canonical_value() >= value.outer_radius_m.canonical_value()
                {
                    return Err(AuthoredPropertyError::InvalidTubeRadii);
                }
                Ok(())
            }
            Self::HomogeneousShell(value) => {
                positive("thickness", value.thickness_m.canonical_value())?;
                if let HomogeneousShellReferenceSurface::OffsetFromMidSurface(value) =
                    value.reference_surface
                {
                    finite("shell reference-surface offset", value)?;
                }
                positive("shear correction", value.shear_correction)
            }
        }
    }

    pub fn is_beam(&self) -> bool {
        !matches!(self, Self::HomogeneousShell(_))
    }
}

impl AuthoredLayupDefinition {
    pub fn validate(&self) -> Result<(), AuthoredPropertyError> {
        token("layup id", &self.layup_id)?;
        self.reference_orientation.validate()?;
        if let Some(value) = self.shear_correction {
            positive("layup shear correction", value)?;
        }
        if let ShellReferenceSurface::OffsetFromMidSurface(value) = self.reference_surface {
            finite("layup reference-surface offset", value)?;
        }
        if self.plies.is_empty() {
            return Err(AuthoredPropertyError::EmptyLayup);
        }
        let mut ids = BTreeSet::new();
        for ply in &self.plies {
            token("ply id", &ply.ply_id)?;
            if !ids.insert(&ply.ply_id) {
                return Err(AuthoredPropertyError::DuplicatePly(ply.ply_id.clone()));
            }
            positive("ply thickness", ply.thickness_m.canonical_value())?;
            finite("ply angle", ply.angle_rad.canonical_value())?;
        }
        Ok(())
    }
}

impl AuthoredPropertyDefinitionRef {
    pub fn validate(&self) -> Result<(), AuthoredPropertyError> {
        token("property module path", &self.module_path)?;
        token("property model id", &self.model_id)?;
        token("property id", &self.property_id)
    }
}

impl AuthoredOrientationDefinition {
    pub fn id(&self) -> &str {
        match self {
            Self::GlobalCartesian { orientation_id } => orientation_id,
            Self::Cartesian(value) => &value.orientation_id,
            Self::Cylindrical(value) | Self::Spherical(value) => &value.orientation_id,
        }
    }

    pub fn validate(&self) -> Result<(), AuthoredPropertyError> {
        token("orientation id", self.id())?;
        match self {
            Self::GlobalCartesian { .. } => Ok(()),
            Self::Cartesian(value) => {
                finite_vector("orientation origin", value.origin_m.canonical_value())?;
                independent(value.x_axis, value.y_axis)
            }
            Self::Cylindrical(value) | Self::Spherical(value) => {
                finite_vector("orientation origin", value.origin_m.canonical_value())?;
                independent(value.axis, value.reference_direction)
            }
        }
    }
}

fn independent(a: [f64; 3], b: [f64; 3]) -> Result<(), AuthoredPropertyError> {
    finite_vector("orientation axis", a)?;
    finite_vector("orientation reference direction", b)?;
    let norm_a = a.iter().map(|value| value * value).sum::<f64>().sqrt();
    let norm_b = b.iter().map(|value| value * value).sum::<f64>().sqrt();
    let cross = [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ];
    let cross_norm = cross.iter().map(|value| value * value).sum::<f64>().sqrt();
    if norm_a <= f64::EPSILON
        || norm_b <= f64::EPSILON
        || cross_norm <= f64::EPSILON * norm_a * norm_b
    {
        Err(AuthoredPropertyError::DegenerateOrientation)
    } else {
        Ok(())
    }
}

fn token(field: &'static str, value: &str) -> Result<(), AuthoredPropertyError> {
    if value.trim().is_empty() {
        Err(AuthoredPropertyError::InvalidToken(field))
    } else {
        Ok(())
    }
}

fn finite(field: &'static str, value: f64) -> Result<(), AuthoredPropertyError> {
    if value.is_finite() {
        Ok(())
    } else {
        Err(AuthoredPropertyError::InvalidNumber(field))
    }
}

fn positive(field: &'static str, value: f64) -> Result<(), AuthoredPropertyError> {
    finite(field, value)?;
    if value > 0.0 {
        Ok(())
    } else {
        Err(AuthoredPropertyError::NonPositive(field))
    }
}

fn nonnegative(field: &'static str, value: f64) -> Result<(), AuthoredPropertyError> {
    finite(field, value)?;
    if value >= 0.0 {
        Ok(())
    } else {
        Err(AuthoredPropertyError::Negative(field))
    }
}

fn finite_vector(field: &'static str, value: [f64; 3]) -> Result<(), AuthoredPropertyError> {
    if value.iter().all(|component| component.is_finite()) {
        Ok(())
    } else {
        Err(AuthoredPropertyError::InvalidNumber(field))
    }
}

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum AuthoredPropertyError {
    #[error("{0} must not be empty")]
    InvalidToken(&'static str),
    #[error("{0} must be finite")]
    InvalidNumber(&'static str),
    #[error("{0} must be positive")]
    NonPositive(&'static str),
    #[error("{0} must be nonnegative")]
    Negative(&'static str),
    #[error("tube inner radius must be less than its outer radius")]
    InvalidTubeRadii,
    #[error("layup must contain at least one ply")]
    EmptyLayup,
    #[error("layup contains duplicate ply `{0}`")]
    DuplicatePly(String),
    #[error("orientation axes must be finite, nonzero, and nonparallel")]
    DegenerateOrientation,
}

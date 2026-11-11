//! A board cross-section: the 2D section a transmission-line study solves.
//!
//! The section is a stackup listed from the top of the board down, plus the traces on one copper
//! layer. Lengths are canonical metres and angles radians. The stackup is either authored inline
//! or read from a board layout file at check time; either way the resolved layers are part of the
//! definition, so the study digest covers the values the solver uses.
use serde::{Deserialize, Serialize};

use super::{GeometryContractError, GeometryDigest};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CrossSectionDefinition {
    /// Layers from the top of the board to the bottom.
    pub layers: Vec<CrossSectionLayer>,
    /// The layout file the layers were read from, when the section names one.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub stackup: Option<CrossSectionStackupSource>,
    pub traces: Vec<CrossSectionTrace>,
    /// Open-region extent; `None` sizes it from the stackup.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub enclosure: Option<CrossSectionEnclosure>,
}

/// The board layout file a section's stackup was read from.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CrossSectionStackupSource {
    pub logical_path: String,
    /// The stackup chosen from a layout with several (rigid-flex); `None` takes the only one.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub stackup_name: Option<String>,
    pub content_digest: GeometryDigest,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CrossSectionLayer {
    pub name: String,
    pub thickness_m: f64,
    pub role: CrossSectionLayerRole,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "kind")]
pub enum CrossSectionLayerRole {
    /// Core or prepreg with its relative permittivity (Dk).
    Dielectric { permittivity: f64 },
    /// Conformal solder mask over the outer copper next to it.
    SolderMask { permittivity: f64 },
    /// The routed copper layer. `fill_permittivity` is what fills it between traces; `None`
    /// takes the mask or air on an outer layer and the dielectric above on an inner layer.
    Signal {
        #[serde(default, skip_serializing_if = "Option::is_none")]
        fill_permittivity: Option<f64>,
    },
    /// A full reference plane at 0 V.
    Plane,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CrossSectionTraceRole {
    /// A line whose parameters are reported.
    Signal,
    /// Coplanar ground copper held at 0 V.
    Ground,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CrossSectionTraceBase {
    Top,
    Bottom,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CrossSectionTrace {
    pub name: String,
    pub role: CrossSectionTraceRole,
    /// The copper layer the trace is etched on.
    pub layer: String,
    /// Lateral position of the trace centre.
    pub center_m: f64,
    /// Width of the base face.
    pub width_m: f64,
    /// Sidewall angle from the base face; π/2 is a rectangular trace.
    pub etch_angle_rad: f64,
    /// `None` takes the laminate side.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub base: Option<CrossSectionTraceBase>,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CrossSectionEnclosure {
    pub half_width_m: f64,
    pub air_m: f64,
}

impl CrossSectionDefinition {
    /// The signal traces in authored order: the conductors whose parameters a study reports.
    pub fn signal_traces(&self) -> impl Iterator<Item = &CrossSectionTrace> {
        self.traces
            .iter()
            .filter(|trace| trace.role == CrossSectionTraceRole::Signal)
    }

    /// Contract checks. The solver checks the physics of the section (overlap, a return
    /// conductor, the etch angle against the copper thickness) when it builds the domain.
    pub fn validate(&self) -> Result<(), GeometryContractError> {
        let invalid = |field: &str, message: &str| GeometryContractError::invalid(field, message);
        if self.layers.is_empty() {
            return Err(invalid("cross-section layers", "must not be empty"));
        }
        let mut names = std::collections::BTreeSet::new();
        for layer in &self.layers {
            if layer.name.trim().is_empty() {
                return Err(invalid("cross-section layer name", "must not be empty"));
            }
            if !names.insert(layer.name.as_str()) {
                return Err(invalid(
                    "cross-section layers",
                    "layer names must be unique",
                ));
            }
            if !layer.thickness_m.is_finite() || layer.thickness_m < 0.0 {
                return Err(invalid(
                    "cross-section layer thickness",
                    "must be finite and not negative",
                ));
            }
            let permittivity = match layer.role {
                CrossSectionLayerRole::Dielectric { permittivity }
                | CrossSectionLayerRole::SolderMask { permittivity } => Some(permittivity),
                CrossSectionLayerRole::Signal { fill_permittivity } => fill_permittivity,
                CrossSectionLayerRole::Plane => None,
            };
            if permittivity.is_some_and(|value| !value.is_finite() || value < 1.0) {
                return Err(invalid(
                    "cross-section permittivity",
                    "must be finite and at least 1",
                ));
            }
        }
        if self.signal_traces().next().is_none() {
            return Err(invalid(
                "cross-section traces",
                "need at least one signal trace",
            ));
        }
        let mut traces = std::collections::BTreeSet::new();
        for trace in &self.traces {
            if trace.name.trim().is_empty() {
                return Err(invalid("cross-section trace name", "must not be empty"));
            }
            if !traces.insert(trace.name.as_str()) {
                return Err(invalid(
                    "cross-section traces",
                    "trace names must be unique",
                ));
            }
            let Some(layer) = self.layers.iter().find(|layer| layer.name == trace.layer) else {
                return Err(invalid(
                    "cross-section trace layer",
                    "must name a layer of the section",
                ));
            };
            if !matches!(layer.role, CrossSectionLayerRole::Signal { .. }) {
                return Err(invalid(
                    "cross-section trace layer",
                    "must name a `signal` copper layer",
                ));
            }
            if !(trace.center_m.is_finite() && trace.width_m.is_finite() && trace.width_m > 0.0) {
                return Err(invalid(
                    "cross-section trace",
                    "center must be finite and width positive",
                ));
            }
            if !(trace.etch_angle_rad > 0.0 && trace.etch_angle_rad <= std::f64::consts::FRAC_PI_2)
            {
                return Err(invalid(
                    "cross-section trace etch angle",
                    "must be above 0° and at most 90°",
                ));
            }
        }
        if let Some(enclosure) = self.enclosure {
            if !(enclosure.half_width_m.is_finite()
                && enclosure.half_width_m > 0.0
                && enclosure.air_m.is_finite()
                && enclosure.air_m > 0.0)
            {
                return Err(invalid(
                    "cross-section enclosure",
                    "half_width and air must be positive",
                ));
            }
        }
        if let Some(stackup) = &self.stackup {
            if stackup.logical_path.trim().is_empty() {
                return Err(invalid("cross-section stackup", "path must not be empty"));
            }
        }
        Ok(())
    }
}

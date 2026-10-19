//! Presentation plans selected from the actual admitted procedure, never from filenames.
use anyhow::{bail, Result};
use runmat_analysis_core::{AnalysisProcedure, IndependentDomain, PhysicsFamily, ProcedureStep};
use runmat_analysis_fea::contracts::*;

pub(crate) enum CaptureProfile {
    Static,
    Thermal,
    Modal { count: usize },
}
pub(super) struct ViewSpec {
    pub id: &'static str,
    pub label: &'static str,
    pub field: Option<String>,
    pub unit: &'static str,
    pub shape: &'static str,
    pub relative: bool,
}

/// Keep the frequency with its original saved mode, using the retained basis.
pub(super) fn display_label(
    id: &str,
    label: &str,
    results: &runmat_runtime::analysis::AnalysisResultsData,
) -> Result<String> {
    let ordinal = match id {
        "mode-1" => 0,
        "mode-2" => 1,
        _ => return Ok(label.into()),
    };
    let modal = results
        .modal_results
        .as_ref()
        .ok_or_else(|| anyhow::anyhow!("saved mode has no retained modal basis"))?;
    let frequency = modal
        .mode_indices()
        .nth(ordinal)
        .and_then(|index| modal.basis.modes().get(index))
        .ok_or_else(|| anyhow::anyhow!("saved mode is missing from its retained basis"))?
        .frequency_hz();
    anyhow::ensure!(
        frequency.is_finite() && frequency >= 0.0,
        "saved mode has an invalid natural frequency"
    );
    let frequency = if frequency != 0.0 && !(0.01..100_000.0).contains(&frequency) {
        format!("{frequency:.3e}")
    } else {
        format!("{frequency:.2}")
    };
    Ok(format!("{label} · {frequency} Hz"))
}
impl CaptureProfile {
    pub(crate) fn for_step(step: &ProcedureStep) -> Result<Self> {
        match (step.family, step.procedure) {
            (PhysicsFamily::Structural, AnalysisProcedure::Static) => Ok(Self::Static),
            (PhysicsFamily::Structural, AnalysisProcedure::ModalExtraction) => {
                let IndependentDomain::Eigenvalue { requested_count } = step.independent_domain
                else {
                    bail!("modal snapshot requires an eigenvalue inventory");
                };
                if requested_count == 0 {
                    bail!("modal snapshot requires at least one mode");
                }
                Ok(Self::Modal {
                    count: requested_count.min(2),
                })
            }
            (PhysicsFamily::Thermal, AnalysisProcedure::Static) => Ok(Self::Thermal),
            _ => bail!(
                "snapshot supports standalone structural static/modal or thermal steady studies"
            ),
        }
    }
    pub(super) fn views(&self) -> Vec<ViewSpec> {
        let mut views = vec![ViewSpec {
            id: "geometry",
            label: "Solver mesh",
            field: None,
            unit: "m",
            shape: "undeformed",
            relative: false,
        }];
        let fields = match self {
            Self::Static => vec![
                (
                    "displacement",
                    "Displacement magnitude",
                    FEA_FIELD_STRUCTURAL_DISPLACEMENT.into(),
                    "m",
                    "exaggerated",
                    false,
                ),
                (
                    "stress",
                    "Von Mises stress",
                    FEA_FIELD_STRUCTURAL_VON_MISES.into(),
                    "Pa",
                    "undeformed",
                    false,
                ),
            ],
            Self::Thermal => vec![
                (
                    "temperature",
                    "Temperature",
                    fea_thermal_temperature_field_id(0),
                    "K",
                    "undeformed",
                    false,
                ),
                (
                    "heat-flux",
                    "Heat-flux magnitude",
                    fea_thermal_heat_flux_field_id(0),
                    "W/m²",
                    "undeformed",
                    false,
                ),
            ],
            Self::Modal { .. } => vec![
                (
                    "mode-1",
                    "Mode 1 relative amplitude",
                    fea_modal_mode_shape_field_id(1),
                    "relative",
                    "exaggerated",
                    true,
                ),
                (
                    "mode-2",
                    "Mode 2 relative amplitude",
                    fea_modal_mode_shape_field_id(2),
                    "relative",
                    "exaggerated",
                    true,
                ),
            ],
        };
        views.extend(
            fields
                .into_iter()
                .take(match self {
                    Self::Modal { count } => *count,
                    _ => usize::MAX,
                })
                .map(|(id, label, field, unit, shape, relative)| ViewSpec {
                    id,
                    label,
                    field: Some(field),
                    unit,
                    shape,
                    relative,
                }),
        );
        views
    }
}

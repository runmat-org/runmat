//! What a scene needs beyond its topology and fields: the colour scale, the deformation scale
//! and the solved bodies it draws.
use super::{deformation, legend, regions, FieldData, RenderViewSelection};

#[derive(Default)]
pub(super) struct SceneStats {
    pub(super) element_count: usize,
    pub(super) scale: legend::ScalarScale,
    pub(super) automatic_scale: Option<f64>,
    /// The study's fixed deformation scale (`display <output> { deformation = … }`); it replaces
    /// the automatic scale.
    pub(super) fixed_deformation: Option<f64>,
    pub(super) regions: regions::ViewRegions,
}

impl SceneStats {
    pub(super) fn from_selection(
        selection: &RenderViewSelection,
        field: Option<&FieldData>,
        deformation: Option<&FieldData>,
    ) -> Self {
        let extremes = field.and_then(|field| {
            selection
                .field_summaries
                .iter()
                .find(|summary| summary.field_id == field.label)
                .map(|summary| (summary.minimum, summary.maximum))
        });
        let [minimum, maximum] = selection.model_bounds;
        let diagonal = (0..3)
            .map(|axis| (maximum[axis] - minimum[axis]).powi(2))
            .sum::<f64>()
            .sqrt();
        let automatic_scale = deformation.and_then(|field| {
            selection
                .field_summaries
                .iter()
                .find(|summary| summary.field_id == field.label)
                .map(|summary| {
                    deformation::automatic_scale(diagonal, summary.maximum, field.mode.is_some())
                })
        });
        Self {
            element_count: selection.volume_element_count,
            scale: legend::ScalarScale::of_extremes(extremes),
            automatic_scale,
            ..Self::default()
        }
    }

    /// The displacement scale of an exaggerated shape: the study's fixed scale, else the
    /// automatic one, times the viewer's `exaggeration`.
    pub(super) fn exaggerated_scale(&self, exaggeration: f64) -> Option<f64> {
        self.fixed_deformation
            .or(self.automatic_scale)
            .map(|scale| scale * exaggeration)
    }
}

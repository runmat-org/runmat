//! Descriptive text-only overlay for a proposed immutable Figure snapshot.
use super::{Figure, TextStyle};

/// References actual prepared strings/styles; this is not a scene or lease issuer.
#[derive(Clone, Copy)]
pub struct FigureMetadataOverlay<'a> {
    pub name: Option<&'a String>,
    pub sg_title: Option<&'a String>,
    pub sg_title_style: &'a TextStyle,
    /// Existing axes metadata used by CurrentAxes' legacy header projection.
    pub active_axes: Option<usize>,
}
impl FigureMetadataOverlay<'_> {
    pub(super) fn retained_bytes(&self, figure: &Figure) -> Option<usize> {
        let mut bytes = self
            .name
            .map_or(0, String::capacity)
            .checked_add(self.sg_title.map_or(0, String::capacity))?;
        super::super::clone_budget::add(
            &mut bytes,
            self.sg_title_style.retained_snapshot_bytes()?,
        )?;
        let (labels, colormap) = match self.active_axes {
            Some(index) => {
                let metadata = figure.axes_metadata.get(index)?;
                (
                    [
                        metadata.title.as_ref(),
                        metadata.x_label.as_ref(),
                        metadata.y_label.as_ref(),
                        metadata.z_label.as_ref(),
                    ],
                    &metadata.colormap,
                )
            }
            None => (
                [
                    figure.title.as_ref(),
                    figure.x_label.as_ref(),
                    figure.y_label.as_ref(),
                    figure.z_label.as_ref(),
                ],
                &figure.colormap,
            ),
        };
        for label in labels.into_iter().flatten() {
            super::super::clone_budget::add(&mut bytes, label.capacity())?;
        }
        super::super::clone_budget::add(&mut bytes, colormap.retained_snapshot_bytes()?)?;
        Some(bytes)
    }
}

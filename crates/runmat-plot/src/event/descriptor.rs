//! Plot descriptors and the shape checks a scene plot's data passes before it is restored.

use super::*;

/// Descriptor for a single plot element within the figure.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct PlotDescriptor {
    pub kind: PlotKind,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub label: Option<String>,
    pub axes_index: u32,
    pub color_rgba: [f32; 4],
    pub visible: bool,
}

impl PlotDescriptor {
    pub(super) fn from_plot(plot: &PlotElement, axes_index: u32) -> Self {
        Self {
            kind: PlotKind::from(plot.plot_type()),
            label: plot.label(),
            axes_index,
            color_rgba: vec4_to_rgba(plot.color()),
            visible: plot.is_visible(),
        }
    }
}

pub(super) fn validate_required_equal_lengths(
    kind: &str,
    fields: &[(&str, usize)],
) -> Result<(), SceneExportError> {
    let Some((first_name, first_len)) = fields.first().copied() else {
        return Ok(());
    };
    if first_len == 0 {
        return Err(SceneExportError::unexportable(format!(
            "{kind} is missing required {first_name} values"
        )));
    }
    for (name, len) in fields.iter().copied().skip(1) {
        if len == 0 {
            return Err(SceneExportError::unexportable(format!(
                "{kind} is missing required {name} values"
            )));
        }
        if len != first_len {
            return Err(SceneExportError::unexportable(format!(
                "{kind} length mismatch: {name} has {len} values, expected {first_len}"
            )));
        }
    }
    Ok(())
}

pub(super) fn validate_surface_grid<T>(
    kind: &str,
    x: &[f64],
    y: &[f64],
    grid: &[Vec<T>],
) -> Result<(), SceneExportError> {
    if x.is_empty() {
        return Err(SceneExportError::unexportable(format!(
            "{kind} is missing required x values"
        )));
    }
    if y.is_empty() {
        return Err(SceneExportError::unexportable(format!(
            "{kind} is missing required y values"
        )));
    }
    if grid.is_empty() {
        return Err(SceneExportError::unexportable(format!(
            "{kind} is missing required grid rows"
        )));
    }
    if grid.len() != x.len() {
        return Err(SceneExportError::unexportable(format!(
            "{kind} row count ({}) must match x length ({})",
            grid.len(),
            x.len()
        )));
    }
    for (row_idx, row) in grid.iter().enumerate() {
        if row.len() != y.len() {
            return Err(SceneExportError::unexportable(format!(
                "{kind} row {row_idx} length ({}) must match y length ({})",
                row.len(),
                y.len()
            )));
        }
    }
    Ok(())
}

pub(super) fn validate_matching_grid_shape<T, U>(
    kind: &str,
    reference: &[Vec<T>],
    grid: &[Vec<U>],
) -> Result<(), SceneExportError> {
    if grid.len() != reference.len() {
        return Err(SceneExportError::unexportable(format!(
            "{kind} row count ({}) must match surface row count ({})",
            grid.len(),
            reference.len()
        )));
    }
    for (row_idx, (reference_row, row)) in reference.iter().zip(grid).enumerate() {
        if row.len() != reference_row.len() {
            return Err(SceneExportError::unexportable(format!(
                "{kind} row {row_idx} length ({}) must match surface row length ({})",
                row.len(),
                reference_row.len()
            )));
        }
    }
    Ok(())
}

pub(super) fn validate_surface_coordinate_grids(
    kind: &str,
    x_grid: &[Vec<f64>],
    y_grid: &[Vec<f64>],
    z_grid: &[Vec<f64>],
) -> Result<(), SceneExportError> {
    if z_grid.is_empty() {
        return Err(SceneExportError::unexportable(format!(
            "{kind} is missing required grid rows"
        )));
    }
    validate_matching_grid_shape(kind, z_grid, x_grid)?;
    validate_matching_grid_shape(kind, z_grid, y_grid)?;
    Ok(())
}

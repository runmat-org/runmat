/// One exact cell of a row-major subplot grid, without allocating the grid.
pub(crate) fn tiled_viewport_px(
    total_width: u32,
    total_height: u32,
    rows: u32,
    columns: u32,
    index: u32,
) -> Option<(u32, u32, u32, u32)> {
    let count = rows.checked_mul(columns)?;
    if rows == 0 || columns == 0 || index >= count {
        return None;
    }
    let row = index / columns;
    let column = index % columns;
    Some(tiled_cell_px(
        total_width,
        total_height,
        rows,
        columns,
        row,
        column,
    ))
}

fn tiled_cell_px(
    total_width: u32,
    total_height: u32,
    rows: u32,
    columns: u32,
    row: u32,
    column: u32,
) -> (u32, u32, u32, u32) {
    let cell_width = (total_width / columns).max(1);
    let cell_height = (total_height / rows).max(1);
    let x = column * cell_width;
    let y = row * cell_height;
    let width = if column + 1 == columns {
        total_width.saturating_sub(x).max(1)
    } else {
        cell_width
    };
    let height = if row + 1 == rows {
        total_height.saturating_sub(y).max(1)
    } else {
        cell_height
    };
    (x, y, width, height)
}

/// Resolve every axes viewport from one immutable figure layout.
pub(crate) fn figure_viewports_px(
    figure: &crate::plots::Figure,
    total_width: u32,
    total_height: u32,
) -> Vec<(u32, u32, u32, u32)> {
    let total_width = total_width.max(1);
    let total_height = total_height.max(1);
    let axes_count = figure.axes_count().max(1);
    let full = (0, 0, total_width, total_height);
    (0..axes_count)
        .map(|axes_index| {
            let Some(owner) = figure.axes_layout_owner(axes_index) else {
                return full;
            };
            if let Some(metadata) = figure
                .axes_metadata(owner)
                .filter(|metadata| metadata.position_explicit)
            {
                return position_to_viewport_px(
                    total_width,
                    total_height,
                    metadata.position,
                    &metadata.units,
                );
            }
            figure
                .resolved_axes_grid_cell(axes_index)
                .and_then(|cell| grid_region_viewport_px(total_width, total_height, cell))
                .unwrap_or(full)
        })
        .collect()
}

fn grid_region_viewport_px(
    total_width: u32,
    total_height: u32,
    cell: crate::plots::AxesGridCell,
) -> Option<(u32, u32, u32, u32)> {
    if !cell.is_valid() {
        return None;
    }
    let (x, y, _, _) = tiled_viewport_px(
        total_width,
        total_height,
        cell.rows,
        cell.columns,
        cell.index,
    )?;
    let last = cell.index + (cell.row_span - 1) * cell.columns + cell.column_span - 1;
    let (last_x, last_y, last_width, last_height) =
        tiled_viewport_px(total_width, total_height, cell.rows, cell.columns, last)?;
    Some((x, y, last_x + last_width - x, last_y + last_height - y))
}

pub(crate) fn position_to_viewport_px(
    total_width: u32,
    total_height: u32,
    position: [f64; 4],
    units: &str,
) -> (u32, u32, u32, u32) {
    let total_width = f64::from(total_width.max(1));
    let total_height = f64::from(total_height.max(1));
    let (left, bottom, width, height) = if units.eq_ignore_ascii_case("normalized") {
        (
            position[0] * total_width,
            position[1] * total_height,
            position[2] * total_width,
            position[3] * total_height,
        )
    } else {
        (position[0], position[1], position[2], position[3])
    };
    let x = left.clamp(0.0, total_width - 1.0);
    let height = height.max(1.0).min(total_height);
    let y = (total_height - bottom - height).clamp(0.0, total_height - 1.0);
    let width = width.max(1.0).min(total_width - x);
    let height = height.min(total_height - y);
    (
        x.round() as u32,
        y.round() as u32,
        width.round().max(1.0) as u32,
        height.round().max(1.0) as u32,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn spanned_grid_viewports_include_inner_cells_and_outer_remainders() {
        let cell = crate::plots::AxesGridCell::from_slots(2, 3, [0, 1, 3, 4]).unwrap();
        assert_eq!(
            grid_region_viewport_px(601, 401, cell),
            Some((0, 0, 400, 401))
        );
        let right = crate::plots::AxesGridCell::from_slots(2, 3, [2, 5]).unwrap();
        assert_eq!(
            grid_region_viewport_px(601, 401, right),
            Some((400, 0, 201, 401))
        );
    }

    #[test]
    fn tiled_viewports_preserve_odd_remainder_and_small_extents() {
        assert_eq!(
            (0..2)
                .map(|index| tiled_viewport_px(401, 5, 1, 2, index).unwrap())
                .collect::<Vec<_>>(),
            vec![(0, 0, 200, 5), (200, 0, 201, 5)]
        );
        assert_eq!(tiled_viewport_px(401, 5, 1, 2, 1), Some((200, 0, 201, 5)));
        assert_eq!(
            (0..9)
                .map(|index| tiled_viewport_px(2, 2, 3, 3, index).unwrap())
                .collect::<Vec<_>>(),
            vec![
                (0, 0, 1, 1),
                (1, 0, 1, 1),
                (2, 0, 1, 1),
                (0, 1, 1, 1),
                (1, 1, 1, 1),
                (2, 1, 1, 1),
                (0, 2, 1, 1),
                (1, 2, 1, 1),
                (2, 2, 1, 1),
            ]
        );
    }

    #[test]
    fn overlay_viewports_follow_parent_layout_and_allow_an_explicit_override() {
        use crate::plots::{AxesGridCell, Figure};
        let mut figure = Figure::new();
        figure
            .set_axes_grid_cell(0, AxesGridCell::single(1, 2, 0))
            .unwrap();
        let overlay = figure.ensure_overlay_axes(0);
        assert_eq!(
            figure_viewports_px(&figure, 600, 400)[overlay],
            (0, 0, 300, 400)
        );
        figure
            .set_axes_grid_cell(0, AxesGridCell::single(2, 2, 3))
            .unwrap();
        assert_eq!(
            figure_viewports_px(&figure, 600, 400)[overlay],
            (300, 200, 300, 200)
        );
        figure.set_axes_position(0, [0.1, 0.2, 0.5, 0.5]);
        assert_eq!(
            figure_viewports_px(&figure, 600, 400)[overlay],
            (60, 120, 300, 200)
        );
        figure.set_axes_position(overlay, [0.0, 0.0, 0.25, 0.25]);
        assert_eq!(
            figure_viewports_px(&figure, 600, 400)[overlay],
            (0, 300, 150, 100)
        );
    }
}

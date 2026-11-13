/// Pixel viewports for a row-major subplot grid.
pub(crate) fn tiled_viewports_px(
    total_width: u32,
    total_height: u32,
    rows: usize,
    cols: usize,
) -> Vec<(u32, u32, u32, u32)> {
    if rows == 0 || cols == 0 {
        return vec![(0, 0, total_width.max(1), total_height.max(1))];
    }
    let rows_u32 = rows as u32;
    let cols_u32 = cols as u32;
    let mut out = Vec::with_capacity(rows.saturating_mul(cols));
    for row in 0..rows_u32 {
        for column in 0..cols_u32 {
            out.push(tiled_cell_px(
                total_width,
                total_height,
                rows_u32,
                cols_u32,
                row,
                column,
            ));
        }
    }
    out
}

/// One exact cell from [`tiled_viewports_px`] without allocating the grid.
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
    let (rows, columns) = figure.axes_grid();
    let axes_count = figure.axes_count().max(1);
    let full = (0, 0, total_width, total_height);
    let mut viewports = if rows.saturating_mul(columns) > 1 {
        tiled_viewports_px(total_width, total_height, rows, columns)
    } else {
        vec![full; axes_count]
    };
    if viewports.len() < axes_count {
        let grid_len = viewports.len().max(1);
        for axes_index in viewports.len()..axes_count {
            let parent = figure
                .axes_overlay_parent(axes_index)
                .unwrap_or(axes_index)
                .min(grid_len - 1);
            viewports.push(viewports.get(parent).copied().unwrap_or(full));
        }
    }
    viewports.truncate(axes_count);
    for (axes_index, viewport) in viewports.iter_mut().enumerate() {
        if let Some(metadata) = figure
            .axes_metadata(axes_index)
            .filter(|metadata| metadata.position_explicit)
        {
            *viewport = position_to_viewport_px(
                total_width,
                total_height,
                metadata.position,
                &metadata.units,
            );
        } else if let Some(cell) = figure.axes_grid_cell(axes_index) {
            *viewport = tiled_viewport_px(
                total_width,
                total_height,
                cell.rows,
                cell.columns,
                cell.index,
            )
            .unwrap_or(full);
        }
    }
    viewports
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
    fn tiled_viewports_preserve_odd_remainder_and_small_extents() {
        assert_eq!(
            tiled_viewports_px(401, 5, 1, 2),
            vec![(0, 0, 200, 5), (200, 0, 201, 5)]
        );
        assert_eq!(tiled_viewport_px(401, 5, 1, 2, 1), Some((200, 0, 201, 5)));
        assert_eq!(
            tiled_viewports_px(2, 2, 3, 3),
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
        let all = tiled_viewports_px(2, 2, 3, 3);
        for index in 0..9 {
            assert_eq!(
                tiled_viewport_px(2, 2, 3, 3, index),
                Some(all[index as usize])
            );
        }
    }
}

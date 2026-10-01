//! Scatterhist grouping domain behavior.
use super::*;

pub(super) fn grouped_series(
    x: &[f64],
    y: &[f64],
    options: &ScatterhistOptions,
) -> BuiltinResult<Vec<GroupSeries>> {
    let labels = if options.plot_group {
        options
            .group
            .clone()
            .unwrap_or_else(|| vec![Some(String::new()); x.len()])
    } else {
        vec![Some(String::new()); x.len()]
    };
    let mut grouped: Vec<(String, GroupSeries)> = Vec::new();
    for ((&x_value, &y_value), label) in x.iter().zip(y.iter()).zip(labels.iter()) {
        let Some(label) = label else {
            continue;
        };
        let key = if label.is_empty() {
            "Data".to_string()
        } else {
            label.clone()
        };
        let idx = if let Some(idx) = grouped.iter().position(|(existing, _)| existing == &key) {
            idx
        } else {
            if grouped.len() >= MAX_GROUPS {
                return Err(invalid(format!(
                    "scatterhist: Group creates more than {MAX_GROUPS} groups"
                )));
            }
            let color_index = grouped.len();
            let style = series_style_for_group(color_index, options);
            grouped.push((
                key.clone(),
                GroupSeries {
                    label: key.clone(),
                    scatter_x: Vec::new(),
                    scatter_y: Vec::new(),
                    marginal_x: Vec::new(),
                    marginal_y: Vec::new(),
                    style,
                },
            ));
            grouped.len() - 1
        };
        let series = &mut grouped[idx].1;
        if x_value.is_finite() {
            series.marginal_x.push(x_value);
        }
        if y_value.is_finite() {
            series.marginal_y.push(y_value);
        }
        if x_value.is_finite() && y_value.is_finite() {
            series.scatter_x.push(x_value);
            series.scatter_y.push(y_value);
        }
    }
    if grouped.iter().all(|(_, group)| {
        group.scatter_x.is_empty() && group.marginal_x.is_empty() && group.marginal_y.is_empty()
    }) {
        return Err(invalid("scatterhist: no finite observations are available"));
    }
    Ok(grouped.into_iter().map(|(_, group)| group).collect())
}

pub(super) fn series_style_for_group(index: usize, options: &ScatterhistOptions) -> SeriesStyle {
    let color = options
        .colors
        .as_ref()
        .and_then(|colors| colors.get(index % colors.len()).copied())
        .unwrap_or_else(|| GROUP_COLORS[index % GROUP_COLORS.len()]);
    SeriesStyle {
        color,
        marker: options.markers[index % options.markers.len()],
        marker_size: options.marker_sizes[index % options.marker_sizes.len()],
        line_width: options.line_widths[index % options.line_widths.len()],
        line_style: options.line_styles[index % options.line_styles.len()],
        bandwidth: if options.bandwidths.is_empty() {
            None
        } else {
            Some(options.bandwidths[index % options.bandwidths.len()])
        },
    }
}

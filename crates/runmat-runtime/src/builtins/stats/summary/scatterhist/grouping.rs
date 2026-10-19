//! Scatterhist grouping domain behavior.
use super::*;

pub(super) fn grouped_series(
    x: &[f64],
    y: &[f64],
    options: &ScatterhistOptions,
) -> BuiltinResult<Vec<GroupSeries>> {
    let grouped_by_value = options.plot_group && options.group.is_some();
    let default_key = KeyAtom::Text(String::new());
    let mut grouped: Vec<(KeyAtom, GroupSeries)> = Vec::new();
    for (index, (&x_value, &y_value)) in x.iter().zip(y.iter()).enumerate() {
        let label = if grouped_by_value {
            options
                .group
                .as_ref()
                .and_then(|labels| labels.get(index))
                .and_then(Option::as_ref)
        } else {
            Some(&default_key)
        };
        let Some(label) = label else {
            continue;
        };
        let key = label.clone();
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
                    label: if grouped_by_value {
                        key.label()
                    } else {
                        "Data".to_string()
                    },
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

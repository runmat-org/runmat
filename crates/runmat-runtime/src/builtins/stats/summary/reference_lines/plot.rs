//! Axes span, plotted-data fit, and line geometry.
use super::*;

pub(super) fn refline_x_span(
    handle: FigureHandle,
    axes_index: usize,
    owner: ReferenceLineOwner,
) -> BuiltinResult<(f64, f64)> {
    if let Some(limits) = axes_metadata_snapshot(handle, axes_index)
        .map_err(|err| owner.internal(format!("{}: {err}", owner.name())))?
        .x_limits
        .filter(|limits| valid_span(*limits))
    {
        return Ok(limits);
    }

    if let Some(mut figure) = clone_figure(handle) {
        let bounds = figure.data_bounds_for_axes(axes_index);
        let span = finite_or_padded_span(bounds.min.x as f64, bounds.max.x as f64);
        if let Some(span) = span {
            return Ok(span);
        }
    }

    Ok((0.0, 1.0))
}

pub(super) async fn least_squares_specs(
    handle: FigureHandle,
    axes_index: usize,
    owner: ReferenceLineOwner,
) -> BuiltinResult<Vec<ReflineLineSpec>> {
    let x_span = refline_x_span(handle, axes_index, owner)?;
    let Some(figure) = clone_figure(handle) else {
        return Ok(Vec::new());
    };

    let mut specs = Vec::new();
    for (plot_index, plot) in figure.plots().enumerate() {
        if figure
            .plot_axes_indices()
            .get(plot_index)
            .copied()
            .unwrap_or(0)
            != axes_index
        {
            continue;
        }
        if !plot.is_visible() {
            continue;
        }

        let data = match plot {
            PlotElement::Scatter(plot) => {
                Some(plot.export_scene_xy_data_typed().await.map_err(|err| {
                    owner.internal_with_plot_source(
                        format!("{}: unable to read scatter data: {err}", owner.name()),
                        err,
                    )
                })?)
            }
            PlotElement::Line(plot)
                if plot.marker.is_some() && matches!(plot.line_style, LineStyle::None) =>
            {
                Some(plot.export_scene_xy_data_typed().await.map_err(|err| {
                    owner.internal_with_plot_source(
                        format!("{}: unable to read line data: {err}", owner.name()),
                        err,
                    )
                })?)
            }
            _ => None,
        };
        let Some((x, y)) = data else {
            continue;
        };
        if let Some((slope, intercept)) = least_squares_coefficients(&x, &y, owner)? {
            specs.push(ReflineLineSpec {
                slope,
                intercept,
                x_span,
            });
        }
    }

    Ok(specs)
}

pub(super) fn line_from_spec(
    spec: ReflineLineSpec,
    style: &crate::builtins::plotting::style::ParsedLineStyle,
    owner: ReferenceLineOwner,
) -> BuiltinResult<LinePlot> {
    let y_span = line_endpoints(spec);
    if !y_span.0.is_finite() || !y_span.1.is_finite() {
        return Err(owner.invalid(format!(
            "{}: computed line coordinates must be finite",
            owner.name()
        )));
    }

    let mut line = LinePlot::new(vec![spec.x_span.0, spec.x_span.1], vec![y_span.0, y_span.1])
        .map_err(|err| owner.internal(format!("{}: {err}", owner.name())))?
        .with_style(
            style.appearance.color,
            style.appearance.line_width,
            style.appearance.line_style,
        );
    line.set_marker(marker_metadata_from_appearance(&style.appearance));
    if let Some(visible) = style.appearance.visible {
        line.set_visible(visible);
    }
    if let Some(label) = style.label.clone() {
        line = line.with_label(label);
    }
    Ok(line)
}

fn line_endpoints(spec: ReflineLineSpec) -> (f64, f64) {
    (
        spec.slope.mul_add(spec.x_span.0, spec.intercept),
        spec.slope.mul_add(spec.x_span.1, spec.intercept),
    )
}

fn finite_or_padded_span(min: f64, max: f64) -> Option<(f64, f64)> {
    if !min.is_finite() || !max.is_finite() {
        return None;
    }
    if min < max {
        return Some((min, max));
    }
    if min == max {
        let pad = min.abs().max(1.0) * 0.5;
        return Some((min - pad, max + pad));
    }
    None
}

fn valid_span((min, max): (f64, f64)) -> bool {
    min.is_finite() && max.is_finite() && min < max
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn finite_endpoints_survive_overflowing_unfused_product() {
        let y = line_endpoints(ReflineLineSpec {
            slope: 1.0e308,
            intercept: -1.0e308,
            x_span: (2.0, 2.5),
        });
        assert_eq!(y, (1.0e308, 1.5e308));
    }

    #[test]
    fn opposite_signed_extreme_fit_has_finite_line_endpoints() {
        let (slope, intercept) = super::super::fit::least_squares_coefficients(
            &[-f64::MAX, f64::MAX],
            &[f64::MAX, -f64::MAX],
            ReferenceLineOwner::Lsline,
        )
        .unwrap()
        .unwrap();
        let endpoints = line_endpoints(ReflineLineSpec {
            slope,
            intercept,
            x_span: (-f64::MAX, f64::MAX),
        });
        assert!(endpoints.0.is_finite() && endpoints.1.is_finite());
        assert!((endpoints.0 / f64::MAX - 1.0).abs() < 1.0e-12);
        assert!((endpoints.1 / -f64::MAX - 1.0).abs() < 1.0e-12);
    }
}

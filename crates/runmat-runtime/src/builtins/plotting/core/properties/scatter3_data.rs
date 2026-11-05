//! Scatter3 property transaction. Parse the whole pair set before source
//! readback, then publish one checked plot successor.

use super::*;
use runmat_plot::plots::Scatter3Plot;

struct Prepared {
    x: Option<NumericPlotData>,
    y: Option<NumericPlotData>,
    z: Option<NumericPlotData>,
    cdata: Option<NumericPlotData>,
    sizedata: Option<NumericPlotData>,
    styles: Vec<Style>,
}

enum Style {
    Marker(runmat_plot::plots::MarkerStyle),
    FaceColor(glam::Vec4),
    EdgeColor(glam::Vec4),
    LineWidth(f32),
    DisplayName(String),
}

impl Prepared {
    fn new(pairs: &[(String, &Value)], builtin: &'static str) -> BuiltinResult<Self> {
        let mut prepared = Self {
            x: None,
            y: None,
            z: None,
            cdata: None,
            sizedata: None,
            styles: Vec::new(),
        };
        for (key, value) in pairs {
            match key.as_str() {
                "xdata" => prepared.x = Some(line_numeric_data(value, "XData", builtin)?),
                "ydata" => prepared.y = Some(line_numeric_data(value, "YData", builtin)?),
                "zdata" => prepared.z = Some(line_numeric_data(value, "ZData", builtin)?),
                "colordata" => prepared.cdata = Some(line_numeric_data(value, "CData", builtin)?),
                "sizedata" => {
                    prepared.sizedata = Some(line_numeric_data(value, "SizeData", builtin)?)
                }
                "marker" => {
                    let name = ValidatedStyleValue::new(value, builtin)?
                        .text()
                        .ok_or_else(|| plotting_error(builtin, "Marker must be text"))?;
                    let style = match name.trim() {
                        "o" => runmat_plot::plots::MarkerStyle::Circle,
                        "s" => runmat_plot::plots::MarkerStyle::Square,
                        "^" => runmat_plot::plots::MarkerStyle::Triangle,
                        "d" => runmat_plot::plots::MarkerStyle::Diamond,
                        "+" => runmat_plot::plots::MarkerStyle::Plus,
                        "x" => runmat_plot::plots::MarkerStyle::Cross,
                        "*" => runmat_plot::plots::MarkerStyle::Star,
                        "h" => runmat_plot::plots::MarkerStyle::Hexagon,
                        _ => return Err(plotting_error(builtin, "unsupported scatter3 Marker")),
                    };
                    prepared.styles.push(Style::Marker(style));
                }
                "markerfacecolor" | "markeredgecolor" => {
                    let color = parse_color_value(&LineStyleParseOptions::generic(builtin), value)?;
                    prepared.styles.push(if key == "markerfacecolor" {
                        Style::FaceColor(color)
                    } else {
                        Style::EdgeColor(color)
                    });
                }
                "linewidth" => {
                    let width = value_as_f64(value)
                        .filter(|width| {
                            width.is_finite() && *width >= 0.0 && *width <= f32::MAX as f64
                        })
                        .ok_or_else(|| {
                            plotting_error(builtin, "LineWidth must be a nonnegative finite scalar")
                        })?;
                    prepared.styles.push(Style::LineWidth(width as f32));
                }
                "displayname" => prepared.styles.push(Style::DisplayName(
                    ValidatedStyleValue::new(value, builtin)?
                        .text()
                        .ok_or_else(|| plotting_error(builtin, "DisplayName must be text"))?,
                )),
                _ => {
                    return Err(plotting_error(
                        builtin,
                        format!("{builtin}: unsupported scatter3 property `{key}`"),
                    ))
                }
            }
        }
        if prepared.cdata.is_some()
            && prepared
                .styles
                .iter()
                .any(|style| matches!(style, Style::FaceColor(_)))
        {
            return Err(plotting_error(
                builtin,
                "CData and MarkerFaceColor cannot be set together",
            ));
        }
        Ok(prepared)
    }

    fn requires_data(&self) -> bool {
        self.x.is_some()
            || self.y.is_some()
            || self.z.is_some()
            || self.cdata.is_some()
            || self.sizedata.is_some()
            || self
                .styles
                .iter()
                .any(|style| matches!(style, Style::FaceColor(_)))
    }

    fn needs_existing_colors(&self) -> bool {
        self.cdata.is_none()
            && !self
                .styles
                .iter()
                .any(|style| matches!(style, Style::FaceColor(_)))
    }
}

struct ExistingAttributes {
    colors: Option<Vec<glam::Vec4>>,
    sizes: Option<Option<Vec<f32>>>,
}

fn host_attributes(
    current: &Scatter3Plot,
    prepared: &Prepared,
    builtin: &'static str,
) -> BuiltinResult<ExistingAttributes> {
    let colors = if prepared.needs_existing_colors() {
        if current.has_resident_color_source() {
            return Err(plotting_error(
                builtin,
                "scatter3 CData requires async property access",
            ));
        }
        let count = current.point_count();
        Some(match current.colors.as_slice() {
            colors if colors.len() == count => colors.to_vec(),
            [color] => vec![*color; count],
            _ => {
                return Err(plotting_error(
                    builtin,
                    "scatter3 has no exportable color source",
                ))
            }
        })
    } else {
        None
    };
    let sizes = if prepared.sizedata.is_none() {
        if current.has_resident_size_source() {
            return Err(plotting_error(
                builtin,
                "scatter3 SizeData requires async property access",
            ));
        }
        Some(current.point_sizes.clone())
    } else {
        None
    };
    Ok(ExistingAttributes { colors, sizes })
}

async fn async_attributes(
    current: &Scatter3Plot,
    prepared: &Prepared,
    builtin: &'static str,
) -> BuiltinResult<ExistingAttributes> {
    let colors = if prepared.needs_existing_colors() {
        Some(
            current
                .export_scene_colors(current.point_count())
                .await
                .map_err(|err| map_figure_error(builtin, err))?,
        )
    } else {
        None
    };
    let sizes = if prepared.sizedata.is_none() {
        Some(
            current
                .export_scene_sizes()
                .await
                .map_err(|err| map_figure_error(builtin, err))?,
        )
    } else {
        None
    };
    Ok(ExistingAttributes { colors, sizes })
}

fn apply_styles(plot: &mut Scatter3Plot, styles: Vec<Style>) {
    for style in styles {
        match style {
            // Marker, edge and width are material/metadata channels, so an
            // unchanged GPU vertex source remains valid.
            Style::Marker(value) => plot.marker_style = value,
            Style::EdgeColor(value) => plot.edge_color = value,
            Style::LineWidth(value) => plot.edge_thickness = value,
            Style::DisplayName(value) => plot.label = Some(value),
            Style::FaceColor(value) => {
                plot.colors.fill(value);
                plot.color_values = None;
                plot.color_limits = None;
            }
        }
    }
}

fn source_xyz(plot: &Scatter3Plot, builtin: &'static str) -> BuiltinResult<PlotXYZ> {
    plot.source_data()
        .map(|(x, y, z)| (x.clone(), y.clone(), z.clone()))
        .ok_or_else(|| {
            plotting_error(
                builtin,
                format!("{builtin}: scatter3 source data requires async property access"),
            )
        })
}

pub(super) async fn export_xyz(
    plot: &Scatter3Plot,
    builtin: &'static str,
) -> BuiltinResult<PlotXYZ> {
    plot.export_numeric_xyz_data()
        .await
        .map_err(|err| map_figure_error(builtin, err))?
        .ok_or_else(|| plotting_error(builtin, format!("{builtin}: scatter3 has no source data")))
}

pub(super) fn get(
    handle: &super::super::state::SimplePlotHandleState,
    property: Option<&str>,
    builtin: &'static str,
    source: PropertyReadSource<'_>,
) -> BuiltinResult<Value> {
    let plot = get_simple_plot_from_source(handle, builtin, source)?;
    let runmat_plot::plots::figure::PlotElement::Scatter3(plot) = plot.as_ref() else {
        return Err(plotting_error(builtin, "invalid scatter3 handle"));
    };
    let needs_xyz = matches!(
        property.map(canonical_property_name).as_deref(),
        None | Some("xdata" | "ydata" | "zdata")
    );
    let xyz = needs_xyz.then(|| source_xyz(plot, builtin)).transpose()?;
    read(handle, property, builtin, plot, xyz, None, None)
}

pub(super) async fn get_async(
    handle: &super::super::state::SimplePlotHandleState,
    plot: &Scatter3Plot,
    property: Option<&str>,
    builtin: &'static str,
) -> BuiltinResult<Value> {
    let key = property.map(canonical_property_name);
    let xyz = if matches!(key.as_deref(), None | Some("xdata" | "ydata" | "zdata")) {
        Some(export_xyz(plot, builtin).await?)
    } else {
        None
    };
    let sizes = if matches!(key.as_deref(), None | Some("sizedata")) {
        Some(
            plot.export_scene_sizes()
                .await
                .map_err(|err| map_figure_error(builtin, err))?,
        )
    } else {
        None
    };
    let colors =
        if matches!(key.as_deref(), None | Some("colordata")) && plot.color_values.is_none() {
            Some(
                plot.export_scene_colors(plot.point_count())
                    .await
                    .map_err(|err| map_figure_error(builtin, err))?,
            )
        } else {
            None
        };
    read(handle, property, builtin, plot, xyz, sizes, colors)
}

fn read(
    handle: &super::super::state::SimplePlotHandleState,
    property: Option<&str>,
    builtin: &'static str,
    plot: &Scatter3Plot,
    xyz: Option<PlotXYZ>,
    sizes: Option<Option<Vec<f32>>>,
    colors: Option<Vec<glam::Vec4>>,
) -> BuiltinResult<Value> {
    let key = property.map(canonical_property_name);
    let xyz = xyz.map(|(x, y, z)| {
        (
            numeric_plot_data_value(&x),
            numeric_plot_data_value(&y),
            numeric_plot_data_value(&z),
        )
    });
    let size_value = || -> BuiltinResult<Value> {
        let source = match sizes.as_ref() {
            Some(value) => value.as_deref(),
            None => plot.point_sizes.as_deref(),
        };
        if sizes.is_none() && plot.has_resident_size_source() {
            return Err(plotting_error(
                builtin,
                "scatter3 SizeData requires async property access",
            ));
        }
        Ok(match source {
            Some(values) => tensor_from_vec(
                values
                    .iter()
                    .map(|value| marker_diameter_px_to_area_points2(*value))
                    .collect(),
            ),
            None => Value::Num(marker_diameter_px_to_area_points2(plot.point_size)),
        })
    };
    let color_value = || -> BuiltinResult<Value> {
        if let Some(values) = plot.color_values.as_ref() {
            return Ok(tensor_from_vec(values.clone()));
        }
        if colors.is_none() && plot.has_resident_color_source() {
            return Err(plotting_error(
                builtin,
                "scatter3 CData requires async property access",
            ));
        }
        let colors = colors.as_deref().unwrap_or(&plot.colors);
        if colors.len() == 1 && colors[0].w == 1.0 {
            return Ok(rgb_triplet_value(colors[0]));
        }
        let components = if colors.iter().any(|color| color.w != 1.0) {
            4
        } else {
            3
        };
        let value_count = colors
            .len()
            .checked_mul(components)
            .ok_or_else(|| plotting_error(builtin, "scatter3 CData size overflow"))?;
        let mut values =
            runmat_value::exact_capacity_vec::<f64>(value_count).map_err(|source| {
                plotting_error_with_source(
                    builtin,
                    "scatter3 CData output allocation failed",
                    source,
                )
            })?;
        for component in 0..components {
            for color in colors {
                values.push(color.to_array()[component] as f64);
            }
        }
        Ok(Value::Tensor(
            Tensor::new(values, vec![colors.len(), components])
                .map_err(|err| plotting_error(builtin, format!("scatter3 CData: {err}")))?,
        ))
    };
    match key.as_deref() {
        None => {
            let (x, y, z) =
                xyz.ok_or_else(|| plotting_error(builtin, "scatter3 has no source data"))?;
            let mut result = child_base_struct("scatter", handle.figure, handle.axes_index);
            result.insert("XData", x);
            result.insert("YData", y);
            result.insert("ZData", z);
            result.insert("CData", color_value()?);
            result.insert("SizeData", size_value()?);
            result.insert(
                "Marker",
                Value::String(marker_style_name(plot.marker_style).into()),
            );
            result.insert(
                "MarkerFaceColor",
                rgb_triplet_value(plot.colors.first().copied().unwrap_or_default()),
            );
            result.insert("MarkerEdgeColor", rgb_triplet_value(plot.edge_color));
            result.insert("LineWidth", Value::Num(plot.edge_thickness as f64));
            result.insert(
                "DisplayName",
                Value::string(plot.label.clone().unwrap_or_default()),
            );
            Ok(Value::Struct(result.finish()))
        }
        Some("type") => Ok(Value::String("scatter".into())),
        Some("parent") => Ok(child_parent_handle(handle.figure, handle.axes_index)),
        Some("children") => Ok(handles_value(Vec::new())),
        Some("xdata") => Ok(xyz
            .ok_or_else(|| plotting_error(builtin, "scatter3 XData requires source"))?
            .0),
        Some("ydata") => Ok(xyz
            .ok_or_else(|| plotting_error(builtin, "scatter3 YData requires source"))?
            .1),
        Some("zdata") => Ok(xyz
            .ok_or_else(|| plotting_error(builtin, "scatter3 ZData requires source"))?
            .2),
        Some("colordata") => color_value(),
        Some("sizedata") => size_value(),
        Some("marker") => Ok(Value::String(marker_style_name(plot.marker_style).into())),
        Some("markerfacecolor") => Ok(rgb_triplet_value(
            plot.colors.first().copied().unwrap_or_default(),
        )),
        Some("markeredgecolor") => Ok(rgb_triplet_value(plot.edge_color)),
        Some("linewidth") => Ok(Value::Num(plot.edge_thickness as f64)),
        Some("displayname") => Ok(Value::string(plot.label.clone().unwrap_or_default())),
        Some(other) => Err(plotting_error(
            builtin,
            format!("{builtin}: unsupported scatter3 property `{other}`"),
        )),
    }
}

pub(super) fn apply(
    authority: &super::super::state::PropertyMutationTarget,
    handle: &super::super::state::SimplePlotHandleState,
    pairs: &[(String, &Value)],
    builtin: &'static str,
) -> BuiltinResult<()> {
    let current = authority
        .plot_snapshot(handle.figure, handle.plot_index)
        .map_err(|err| map_figure_error(builtin, err))?;
    let runmat_plot::plots::figure::PlotElement::Scatter3(current) = current else {
        return Err(plotting_error(builtin, "invalid scatter3 handle"));
    };
    let prepared = Prepared::new(pairs, builtin)?;
    if !prepared.requires_data() {
        return update_property_plot(
            authority,
            handle.figure,
            handle.plot_index,
            None,
            builtin,
            |plot| {
                if let runmat_plot::plots::figure::PlotElement::Scatter3(plot) = plot {
                    apply_styles(plot, prepared.styles);
                }
            },
        );
    }
    let xyz = source_xyz(&current, builtin)?;
    let existing = host_attributes(&current, &prepared, builtin)?;
    apply_with_data(
        authority, handle, builtin, &current, xyz, prepared, existing, None,
    )
}

pub(super) async fn apply_async(
    authority: &super::super::state::PropertyMutationTarget,
    handle: &super::super::state::SimplePlotHandleState,
    current: &Scatter3Plot,
    pairs: &[(String, &Value)],
    builtin: &'static str,
    guard: Option<PlotUpdateGuard>,
) -> BuiltinResult<()> {
    let prepared = Prepared::new(pairs, builtin)?;
    if !prepared.requires_data() {
        return update_property_plot(
            authority,
            handle.figure,
            handle.plot_index,
            guard,
            builtin,
            |plot| {
                if let runmat_plot::plots::figure::PlotElement::Scatter3(plot) = plot {
                    apply_styles(plot, prepared.styles);
                }
            },
        );
    }
    let xyz = export_xyz(current, builtin).await?;
    let existing = async_attributes(current, &prepared, builtin).await?;
    apply_with_data(
        authority, handle, builtin, current, xyz, prepared, existing, guard,
    )
}

// Prepared inputs and exported resident attributes are host-owned here. Keep
// this successor preparation synchronous: provider readback finishes in the
// async entry point before the checked revision can be committed.
fn apply_with_data(
    authority: &super::super::state::PropertyMutationTarget,
    handle: &super::super::state::SimplePlotHandleState,
    builtin: &'static str,
    current: &Scatter3Plot,
    (current_x, current_y, current_z): PlotXYZ,
    prepared: Prepared,
    existing: ExistingAttributes,
    guard: Option<PlotUpdateGuard>,
) -> BuiltinResult<()> {
    let old_count = current_x.len();
    let x = prepared.x.unwrap_or(current_x);
    let y = prepared.y.unwrap_or(current_y);
    let z = prepared.z.unwrap_or(current_z);
    let count = x.len();
    if count == 0 || count != y.len() || count != z.len() {
        return Err(plotting_error(
            builtin,
            format!("{builtin}: XData, YData, and ZData must contain the same nonzero number of elements"),
        ));
    }
    let face_color = prepared.styles.iter().find_map(|style| match style {
        Style::FaceColor(color) => Some(*color),
        _ => None,
    });
    let (colors, color_values, color_limits) = if let Some(numeric) = prepared.cdata {
        let shape = numeric.shape();
        if shape.len() == 2 && matches!(shape[1], 3 | 4) && (shape[0] == 1 || shape[0] == count) {
            (
                super::super::point::convert_rgb_color_matrix(
                    &numeric_plot_data_value(&numeric),
                    count,
                    builtin,
                )?,
                None,
                None,
            )
        } else {
            let mut values = numeric.materialize_f64();
            if values.len() == 1 && count > 1 {
                values = vec![values[0]; count];
            }
            if values.len() != count {
                return Err(plotting_error(
                    builtin,
                    "CData must be scalar, per-point values, or an RGB/RGBA matrix",
                ));
            }
            let colormap = super::super::state::axes_metadata_snapshot_for_property(
                authority,
                handle.figure,
                handle.axes_index,
            )
            .map_err(|err| map_figure_error(builtin, err))?
            .colormap;
            let (colors, limits) =
                super::super::point::map_scalar_values_to_colors(&values, colormap);
            (colors, Some(values), Some(limits))
        }
    } else if let Some(color) = face_color {
        (vec![color; count], None, None)
    } else {
        let old_colors = existing.colors.expect("existing colors were prepared");
        if count == old_count {
            (
                old_colors,
                current.color_values.clone(),
                current.color_limits,
            )
        } else if current.color_values.is_none()
            && old_colors
                .first()
                .is_some_and(|first| old_colors.iter().all(|color| color == first))
        {
            (vec![old_colors[0]; count], None, None)
        } else {
            return Err(plotting_error(
                builtin,
                "resizing scatter3 requires matching CData",
            ));
        }
    };
    let (point_size, point_sizes) = if let Some(numeric) = prepared.sizedata {
        if numeric.len() == 1 {
            let value = numeric.materialize_f64()[0];
            if !value.is_finite() || value < 0.0 {
                return Err(plotting_error(
                    builtin,
                    "SizeData must be nonnegative and finite",
                ));
            }
            let diameter = marker_area_points2_to_diameter_px(value);
            if !diameter.is_finite() {
                return Err(plotting_error(
                    builtin,
                    "SizeData diameter exceeds finite f32 range",
                ));
            }
            (diameter, None)
        } else {
            let values = numeric.materialize_f64();
            if values.len() != count
                || values
                    .iter()
                    .any(|value| !value.is_finite() || *value < 0.0)
            {
                return Err(plotting_error(
                    builtin,
                    "SizeData must match the point count and be finite/nonnegative",
                ));
            }
            let sizes = super::super::point::convert_size_vector(
                &numeric_plot_data_value(&numeric),
                count,
                builtin,
            )?;
            if sizes.iter().any(|diameter| !diameter.is_finite()) {
                return Err(plotting_error(
                    builtin,
                    "SizeData diameter exceeds finite f32 range",
                ));
            }
            (current.point_size, Some(sizes))
        }
    } else {
        let old_sizes = existing.sizes.expect("existing sizes were prepared");
        if count != old_count && old_sizes.is_some() {
            return Err(plotting_error(
                builtin,
                "resizing scatter3 requires matching SizeData",
            ));
        }
        (current.point_size, old_sizes)
    };
    update_property_plot(
        authority,
        handle.figure,
        handle.plot_index,
        guard,
        builtin,
        |plot| {
            if let runmat_plot::plots::figure::PlotElement::Scatter3(plot) = plot {
                // Lengths and source conversions were checked before the
                // transaction. This call cannot fail for those exact values.
                plot.update_numeric_data(x, y, z)
                    .expect("validated scatter3 coordinate lengths");
                plot.colors = colors;
                plot.color_values = color_values;
                plot.color_limits = color_limits;
                plot.point_size = point_size;
                plot.point_sizes = point_sizes;
                apply_styles(plot, prepared.styles);
            }
        },
    )
}

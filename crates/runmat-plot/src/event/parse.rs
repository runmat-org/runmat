//! Parsing serialized style names and the lossy float decoders scene JSON uses.

use super::*;

pub(super) fn parse_line_style(value: &str) -> crate::plots::LineStyle {
    match value {
        "None" | "none" => crate::plots::LineStyle::None,
        "Dashed" => crate::plots::LineStyle::Dashed,
        "Dotted" => crate::plots::LineStyle::Dotted,
        "DashDot" => crate::plots::LineStyle::DashDot,
        _ => crate::plots::LineStyle::Solid,
    }
}

pub(super) fn parse_bar_orientation(value: &str) -> crate::plots::bar::Orientation {
    match value {
        "Horizontal" => crate::plots::bar::Orientation::Horizontal,
        _ => crate::plots::bar::Orientation::Vertical,
    }
}

pub(super) fn parse_reference_line_orientation(
    value: &str,
) -> Result<ReferenceLineOrientation, String> {
    match value.to_ascii_lowercase().as_str() {
        "horizontal" => Ok(ReferenceLineOrientation::Horizontal),
        "vertical" => Ok(ReferenceLineOrientation::Vertical),
        _ => Err(format!(
            "unknown reference line orientation '{value}'; expected 'horizontal' or 'vertical'"
        )),
    }
}

pub(super) fn parse_marker_style(value: &str) -> MarkerStyle {
    match value {
        "Square" => MarkerStyle::Square,
        "Triangle" => MarkerStyle::Triangle,
        "Diamond" => MarkerStyle::Diamond,
        "Plus" => MarkerStyle::Plus,
        "Cross" => MarkerStyle::Cross,
        "Star" => MarkerStyle::Star,
        "Hexagon" => MarkerStyle::Hexagon,
        _ => MarkerStyle::Circle,
    }
}

pub(super) fn parse_colormap(value: &str) -> ColorMap {
    ColorMap::from_serialized_token(value).unwrap_or(ColorMap::Parula)
}

pub(super) fn parse_shading_mode(value: &str) -> ShadingMode {
    match value {
        "Flat" => ShadingMode::Flat,
        "Smooth" => ShadingMode::Smooth,
        "Faceted" => ShadingMode::Faceted,
        "None" => ShadingMode::None,
        _ => ShadingMode::Smooth,
    }
}

pub(super) fn parse_patch_face_color_mode(value: &str) -> PatchFaceColorMode {
    match value {
        "None" => PatchFaceColorMode::None,
        "Flat" => PatchFaceColorMode::Flat,
        "Interp" => PatchFaceColorMode::Interp,
        _ => PatchFaceColorMode::Color,
    }
}

pub(super) fn parse_patch_edge_color_mode(value: &str) -> PatchEdgeColorMode {
    match value {
        "None" => PatchEdgeColorMode::None,
        _ => PatchEdgeColorMode::Color,
    }
}

pub(super) fn parse_mesh_edge_mode(value: &str) -> MeshEdgeMode {
    MeshEdgeMode::parse(value).unwrap_or_default()
}

pub(super) fn xyz_to_vec3(value: [f32; 3]) -> Vec3 {
    Vec3::new(value[0], value[1], value[2])
}

pub(super) fn serialized_bounds(min: [f32; 3], max: [f32; 3]) -> BoundingBox {
    BoundingBox::new(xyz_to_vec3(min), xyz_to_vec3(max))
}

pub(super) fn vec3_to_xyz(value: Vec3) -> [f32; 3] {
    [value.x, value.y, value.z]
}

pub(super) fn rgba_to_vec4(value: [f32; 4]) -> Vec4 {
    Vec4::new(value[0], value[1], value[2], value[3])
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SerializedVertex {
    pub(super) position: [f32; 3],
    pub(super) color_rgba: [f32; 4],
    pub(super) normal: [f32; 3],
    pub(super) tex_coords: [f32; 2],
}

impl From<Vertex> for SerializedVertex {
    fn from(value: Vertex) -> Self {
        Self {
            position: value.position,
            color_rgba: value.color,
            normal: value.normal,
            tex_coords: value.tex_coords,
        }
    }
}

impl From<SerializedVertex> for Vertex {
    fn from(value: SerializedVertex) -> Self {
        Self {
            position: value.position,
            color: value.color_rgba,
            normal: value.normal,
            tex_coords: value.tex_coords,
        }
    }
}

/// Serialized legend entry for frontend rendering.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct FigureLegendEntry {
    pub label: String,
    pub plot_type: PlotKind,
    pub color_rgba: [f32; 4],
}

impl From<LegendEntry> for FigureLegendEntry {
    fn from(entry: LegendEntry) -> Self {
        Self {
            label: entry.label,
            plot_type: PlotKind::from(entry.plot_type),
            color_rgba: vec4_to_rgba(entry.color),
        }
    }
}

/// Serializable plot kind values consumed by UI + transports.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum PlotKind {
    Line,
    Line3,
    Scatter,
    Bar,
    ErrorBar,
    Stairs,
    Stem,
    Area,
    Quiver,
    Pie,
    Image,
    Surface,
    Mesh,
    Patch,
    Scatter3,
    Contour,
    ContourFill,
    ReferenceLine,
}

impl From<PlotType> for PlotKind {
    fn from(value: PlotType) -> Self {
        match value {
            PlotType::Line => Self::Line,
            PlotType::Line3 => Self::Line3,
            PlotType::Scatter => Self::Scatter,
            PlotType::Bar => Self::Bar,
            PlotType::ErrorBar => Self::ErrorBar,
            PlotType::Stairs => Self::Stairs,
            PlotType::Stem => Self::Stem,
            PlotType::Area => Self::Area,
            PlotType::Quiver => Self::Quiver,
            PlotType::Pie => Self::Pie,
            PlotType::Surface => Self::Surface,
            PlotType::Mesh => Self::Mesh,
            PlotType::Patch => Self::Patch,
            PlotType::Scatter3 => Self::Scatter3,
            PlotType::Contour => Self::Contour,
            PlotType::ContourFill => Self::ContourFill,
            PlotType::ReferenceLine => Self::ReferenceLine,
        }
    }
}

pub(super) fn parse_line_style_name(name: &str) -> crate::plots::line::LineStyle {
    match name.to_ascii_lowercase().as_str() {
        "none" => crate::plots::line::LineStyle::None,
        "dashed" => crate::plots::line::LineStyle::Dashed,
        "dotted" => crate::plots::line::LineStyle::Dotted,
        "dashdot" => crate::plots::line::LineStyle::DashDot,
        _ => crate::plots::line::LineStyle::Solid,
    }
}

pub(super) fn parse_colormap_name(name: &str) -> crate::plots::surface::ColorMap {
    crate::plots::surface::ColorMap::from_serialized_token(name)
        .unwrap_or(crate::plots::surface::ColorMap::Parula)
}

pub(super) fn vec4_to_rgba(value: Vec4) -> [f32; 4] {
    [value.x, value.y, value.z, value.w]
}

pub(super) fn deserialize_f64_lossy<'de, D>(deserializer: D) -> Result<f64, D::Error>
where
    D: serde::Deserializer<'de>,
{
    let value = Option::<f64>::deserialize(deserializer)?;
    Ok(value.unwrap_or(f64::NAN))
}

/// One float read leniently: a number, the lossless nonfinite tag `runmat_types` writes, or
/// JSON `null` (NaN), as scenes saved before lossless encoding wrote NaN.
pub(super) struct LossyFloat<T>(T);

impl<'de> Deserialize<'de> for LossyFloat<f64> {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        if !deserializer.is_human_readable() {
            return f64::deserialize(deserializer).map(Self);
        }
        let value = Option::<LosslessFloat<f64>>::deserialize(deserializer)?;
        Ok(Self(value.map_or(f64::NAN, |value| value.0)))
    }
}

impl<'de> Deserialize<'de> for LossyFloat<f32> {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        if !deserializer.is_human_readable() {
            return f32::deserialize(deserializer).map(Self);
        }
        let value = Option::<LosslessFloat<f32>>::deserialize(deserializer)?;
        Ok(Self(value.map_or(f32::NAN, |value| value.0)))
    }
}

pub(super) struct LosslessFloat<T>(T);

impl<'de> Deserialize<'de> for LosslessFloat<f64> {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        runmat_types::serde_f64::deserialize(deserializer).map(Self)
    }
}

impl<'de> Deserialize<'de> for LosslessFloat<f32> {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        runmat_types::serde_f32::deserialize(deserializer).map(Self)
    }
}

pub(super) fn deserialize_vec_f64_lossy<'de, D>(deserializer: D) -> Result<Vec<f64>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    let values = Vec::<LossyFloat<f64>>::deserialize(deserializer)?;
    Ok(values.into_iter().map(|value| value.0).collect())
}

pub(super) fn deserialize_vec_f32_lossy<'de, D>(deserializer: D) -> Result<Vec<f32>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    let values = Vec::<LossyFloat<f32>>::deserialize(deserializer)?;
    Ok(values.into_iter().map(|value| value.0).collect())
}

pub(super) fn deserialize_option_vec_f64_lossy<'de, D>(
    deserializer: D,
) -> Result<Option<Vec<f64>>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    let values = Option::<Vec<Option<f64>>>::deserialize(deserializer)?;
    Ok(values.map(|items| {
        items
            .into_iter()
            .map(|value| value.unwrap_or(f64::NAN))
            .collect()
    }))
}

pub(super) fn deserialize_matrix_f64_lossy<'de, D>(
    deserializer: D,
) -> Result<Vec<Vec<f64>>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    let rows = Vec::<Vec<Option<f64>>>::deserialize(deserializer)?;
    Ok(rows
        .into_iter()
        .map(|row| {
            row.into_iter()
                .map(|value| value.unwrap_or(f64::NAN))
                .collect()
        })
        .collect())
}

pub(super) fn deserialize_option_pair_f64_lossy<'de, D>(
    deserializer: D,
) -> Result<Option<[f64; 2]>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    let value = Option::<[Option<f64>; 2]>::deserialize(deserializer)?;
    Ok(value.map(|pair| [pair[0].unwrap_or(f64::NAN), pair[1].unwrap_or(f64::NAN)]))
}

pub(super) fn deserialize_option_vec_f32_lossy<'de, D>(
    deserializer: D,
) -> Result<Option<Vec<f32>>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    let values = Option::<Vec<Option<f32>>>::deserialize(deserializer)?;
    Ok(values.map(|items| {
        items
            .into_iter()
            .map(|value| value.unwrap_or(f32::NAN))
            .collect()
    }))
}

pub(super) fn deserialize_vec_xyz_f32_lossy<'de, D>(
    deserializer: D,
) -> Result<Vec<[f32; 3]>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    let values = Vec::<[Option<f32>; 3]>::deserialize(deserializer)?;
    Ok(values
        .into_iter()
        .map(|xyz| {
            [
                xyz[0].unwrap_or(f32::NAN),
                xyz[1].unwrap_or(f32::NAN),
                xyz[2].unwrap_or(f32::NAN),
            ]
        })
        .collect())
}

pub(super) fn deserialize_vec_rgba_f32_lossy<'de, D>(
    deserializer: D,
) -> Result<Vec<[f32; 4]>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    let values = Vec::<[Option<f32>; 4]>::deserialize(deserializer)?;
    Ok(values
        .into_iter()
        .map(|rgba| {
            [
                rgba[0].unwrap_or(f32::NAN),
                rgba[1].unwrap_or(f32::NAN),
                rgba[2].unwrap_or(f32::NAN),
                rgba[3].unwrap_or(f32::NAN),
            ]
        })
        .collect())
}

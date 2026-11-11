//! Serialized mesh annotations: text, regions, triangle ranges, scalar and vector fields, deformation.

use super::*;

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SerializedTextAnnotation {
    pub position: [f32; 3],
    pub text: String,
    pub style: SerializedTextStyle,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SerializedMeshRegion {
    pub region_id: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub label: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tag: Option<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub triangle_ranges: Vec<SerializedMeshTriangleRange>,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SerializedMeshTriangleRange {
    pub start: u32,
    pub count: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SerializedMeshScalarField {
    pub field_id: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub label: Option<String>,
    pub location: String,
    #[serde(deserialize_with = "deserialize_vec_f32_lossy")]
    pub values: Vec<f32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub color_limits: Option<[f32; 2]>,
    pub colormap: String,
    pub alpha: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SerializedMeshVectorField {
    pub field_id: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub label: Option<String>,
    pub location: String,
    #[serde(deserialize_with = "deserialize_vec_xyz_f32_lossy")]
    pub vectors: Vec<[f32; 3]>,
    pub scale: f32,
    pub stride: usize,
    pub color_rgba: [f32; 4],
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SerializedMeshDeformation {
    pub field_id: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub label: Option<String>,
    #[serde(deserialize_with = "deserialize_vec_xyz_f32_lossy")]
    pub displacements: Vec<[f32; 3]>,
    pub scale: f32,
}

impl From<AxesMetadata> for SerializedAxesMetadata {
    fn from(value: AxesMetadata) -> Self {
        Self {
            y_direction: value.y_direction,
            axes_kind: value.axes_kind.into(),
            overlay_parent: value.overlay_parent,
            y_axis_location: value.y_axis_location,
            position: value.position,
            position_explicit: value.position_explicit,
            units: value.units,
            title: value.title,
            subtitle: value.subtitle,
            x_label: value.x_label,
            y_label: value.y_label,
            z_label: value.z_label,
            x_ticks: value.x_ticks,
            y_ticks: value.y_ticks,
            x_tick_labels: value.x_tick_labels,
            y_tick_labels: value.y_tick_labels,
            x_tick_format: value.x_tick_format,
            y_tick_format: value.y_tick_format,
            x_tick_label_rotation: value.x_tick_label_rotation,
            y_tick_label_rotation: value.y_tick_label_rotation,
            x_limits: value.x_limits.map(|(a, b)| [a, b]),
            y_limits: value.y_limits.map(|(a, b)| [a, b]),
            z_limits: value.z_limits.map(|(a, b)| [a, b]),
            x_limit_method: value.x_limit_method,
            y_limit_method: value.y_limit_method,
            z_limit_method: value.z_limit_method,
            x_log: value.x_log,
            y_log: value.y_log,
            view_azimuth_deg: value.view_azimuth_deg,
            view_elevation_deg: value.view_elevation_deg,
            grid_enabled: value.grid_enabled,
            minor_grid_enabled: value.minor_grid_enabled,
            minor_grid_explicit: value.minor_grid_explicit,
            hidden_line_removal: value.hidden_line_removal,
            box_enabled: value.box_enabled,
            axis_equal: value.axis_equal,
            data_aspect_ratio: value.data_aspect_ratio,
            data_aspect_ratio_mode: value.data_aspect_ratio_mode,
            legend_enabled: value.legend_enabled,
            colorbar_enabled: value.colorbar_enabled,
            colormap: value.colormap.to_serialized_token(),
            colormap_length: value.colormap_length,
            color_order: value
                .color_order
                .map(|colors| colors.into_iter().map(|c| [c.x, c.y, c.z]).collect()),
            color_limits: value.color_limits.map(|(a, b)| [a, b]),
            axes_style: value.axes_style.into(),
            title_style: value.title_style.into(),
            subtitle_style: value.subtitle_style.into(),
            x_label_style: value.x_label_style.into(),
            y_label_style: value.y_label_style.into(),
            z_label_style: value.z_label_style.into(),
            legend_style: value.legend_style.into(),
            world_text_annotations: value
                .world_text_annotations
                .into_iter()
                .map(Into::into)
                .collect(),
        }
    }
}

impl From<SerializedAxesMetadata> for AxesMetadata {
    fn from(value: SerializedAxesMetadata) -> Self {
        Self {
            y_direction: value.y_direction,
            axes_kind: value.axes_kind.into(),
            overlay_parent: value.overlay_parent,
            y_axis_location: value.y_axis_location,
            position: value.position,
            position_explicit: value.position_explicit,
            units: value.units,
            title: value.title,
            subtitle: value.subtitle,
            x_label: value.x_label,
            y_label: value.y_label,
            z_label: value.z_label,
            x_ticks: value.x_ticks,
            y_ticks: value.y_ticks,
            x_tick_labels: value.x_tick_labels,
            y_tick_labels: value.y_tick_labels,
            x_tick_format: value.x_tick_format,
            y_tick_format: value.y_tick_format,
            x_tick_label_rotation: value.x_tick_label_rotation,
            y_tick_label_rotation: value.y_tick_label_rotation,
            x_limits: value.x_limits.map(|[a, b]| (a, b)),
            y_limits: value.y_limits.map(|[a, b]| (a, b)),
            z_limits: value.z_limits.map(|[a, b]| (a, b)),
            x_limit_method: value.x_limit_method,
            y_limit_method: value.y_limit_method,
            z_limit_method: value.z_limit_method,
            x_log: value.x_log,
            y_log: value.y_log,
            view_azimuth_deg: value.view_azimuth_deg,
            view_elevation_deg: value.view_elevation_deg,
            view_revision: 0,
            grid_enabled: value.grid_enabled,
            minor_grid_enabled: value.minor_grid_enabled,
            minor_grid_explicit: value.minor_grid_explicit || value.minor_grid_enabled,
            hidden_line_removal: value.hidden_line_removal,
            box_enabled: value.box_enabled,
            axis_equal: value.axis_equal,
            data_aspect_ratio: value.data_aspect_ratio,
            data_aspect_ratio_mode: value.data_aspect_ratio_mode,
            legend_enabled: value.legend_enabled,
            colorbar_enabled: value.colorbar_enabled,
            colormap: parse_colormap_name(&value.colormap),
            colormap_length: value.colormap_length,
            color_order: value.color_order.map(|colors| {
                colors
                    .into_iter()
                    .map(|[r, g, b]| glam::Vec4::new(r, g, b, 1.0))
                    .collect()
            }),
            color_limits: value.color_limits.map(|[a, b]| (a, b)),
            axes_style: value.axes_style.into(),
            title_style: value.title_style.into(),
            subtitle_style: value.subtitle_style.into(),
            x_label_style: value.x_label_style.into(),
            y_label_style: value.y_label_style.into(),
            z_label_style: value.z_label_style.into(),
            legend_style: value.legend_style.into(),
            world_text_annotations: value
                .world_text_annotations
                .into_iter()
                .map(Into::into)
                .collect(),
        }
    }
}

impl From<crate::plots::figure::TextAnnotation> for SerializedTextAnnotation {
    fn from(value: crate::plots::figure::TextAnnotation) -> Self {
        Self {
            position: value.position.to_array(),
            text: value.text,
            style: value.style.into(),
        }
    }
}

impl From<SerializedTextAnnotation> for crate::plots::figure::TextAnnotation {
    fn from(value: SerializedTextAnnotation) -> Self {
        Self {
            position: glam::Vec3::from_array(value.position),
            text: value.text,
            style: value.style.into(),
        }
    }
}

impl From<&MeshRegion> for SerializedMeshRegion {
    fn from(value: &MeshRegion) -> Self {
        Self {
            region_id: value.region_id.clone(),
            label: value.label.clone(),
            tag: value.tag.clone(),
            triangle_ranges: value
                .triangle_ranges
                .iter()
                .copied()
                .map(Into::into)
                .collect(),
        }
    }
}

impl From<&crate::geometry_scene::GeometrySceneRegion> for SerializedMeshRegion {
    fn from(value: &crate::geometry_scene::GeometrySceneRegion) -> Self {
        Self {
            region_id: value.region_id.clone(),
            label: value.label.clone(),
            tag: value.tag.clone(),
            triangle_ranges: value
                .triangle_ranges
                .iter()
                .copied()
                .map(Into::into)
                .collect(),
        }
    }
}

impl From<SerializedMeshRegion> for MeshRegion {
    fn from(value: SerializedMeshRegion) -> Self {
        MeshRegion {
            region_id: value.region_id,
            label: value.label,
            tag: value.tag,
            triangle_ranges: value.triangle_ranges.into_iter().map(Into::into).collect(),
        }
    }
}

impl From<SerializedMeshRegion> for crate::geometry_scene::GeometrySceneRegion {
    fn from(value: SerializedMeshRegion) -> Self {
        crate::geometry_scene::GeometrySceneRegion::new(
            value.region_id,
            value.label,
            value.tag,
            value.triangle_ranges.into_iter().map(Into::into).collect(),
        )
    }
}

impl From<MeshTriangleRange> for SerializedMeshTriangleRange {
    fn from(value: MeshTriangleRange) -> Self {
        Self {
            start: value.start,
            count: value.count,
        }
    }
}

impl From<crate::geometry_scene::GeometrySceneTriangleRange> for SerializedMeshTriangleRange {
    fn from(value: crate::geometry_scene::GeometrySceneTriangleRange) -> Self {
        Self {
            start: value.start,
            count: value.count,
        }
    }
}

impl From<SerializedMeshTriangleRange> for MeshTriangleRange {
    fn from(value: SerializedMeshTriangleRange) -> Self {
        Self::new(value.start, value.count)
    }
}

impl From<SerializedMeshTriangleRange> for crate::geometry_scene::GeometrySceneTriangleRange {
    fn from(value: SerializedMeshTriangleRange) -> Self {
        Self::new(value.start, value.count)
    }
}

impl From<&MeshScalarField> for SerializedMeshScalarField {
    fn from(value: &MeshScalarField) -> Self {
        Self {
            field_id: value.field_id.clone(),
            label: value.label.clone(),
            location: value.location.as_str().to_string(),
            values: value.values.clone(),
            color_limits: value.color_limits,
            colormap: value.colormap.clone(),
            alpha: value.alpha,
        }
    }
}

impl TryFrom<SerializedMeshScalarField> for MeshScalarField {
    type Error = String;

    fn try_from(value: SerializedMeshScalarField) -> Result<Self, Self::Error> {
        Ok(Self {
            field_id: value.field_id,
            label: value.label,
            location: MeshFieldLocation::parse(&value.location).ok_or_else(|| {
                format!("unknown mesh scalar field location '{}'", value.location)
            })?,
            values: value.values,
            color_limits: value.color_limits,
            colormap: value.colormap,
            alpha: value.alpha,
        })
    }
}

impl From<&MeshVectorField> for SerializedMeshVectorField {
    fn from(value: &MeshVectorField) -> Self {
        Self {
            field_id: value.field_id.clone(),
            label: value.label.clone(),
            location: value.location.as_str().to_string(),
            vectors: value
                .vectors
                .iter()
                .map(|vector| vector.to_array())
                .collect(),
            scale: value.scale,
            stride: value.stride,
            color_rgba: vec4_to_rgba(value.color),
        }
    }
}

impl TryFrom<SerializedMeshVectorField> for MeshVectorField {
    type Error = String;

    fn try_from(value: SerializedMeshVectorField) -> Result<Self, Self::Error> {
        Ok(Self {
            field_id: value.field_id,
            label: value.label,
            location: MeshFieldLocation::parse(&value.location).ok_or_else(|| {
                format!("unknown mesh vector field location '{}'", value.location)
            })?,
            vectors: value.vectors.into_iter().map(Vec3::from_array).collect(),
            scale: value.scale,
            stride: value.stride,
            color: rgba_to_vec4(value.color_rgba),
        })
    }
}

impl From<&MeshDeformation> for SerializedMeshDeformation {
    fn from(value: &MeshDeformation) -> Self {
        Self {
            field_id: value.field_id.clone(),
            label: value.label.clone(),
            displacements: value
                .displacements
                .iter()
                .map(|displacement| displacement.to_array())
                .collect(),
            scale: value.scale,
        }
    }
}

impl From<SerializedMeshDeformation> for MeshDeformation {
    fn from(value: SerializedMeshDeformation) -> Self {
        Self {
            field_id: value.field_id,
            label: value.label,
            displacements: value
                .displacements
                .into_iter()
                .map(Vec3::from_array)
                .collect(),
            scale: value.scale,
        }
    }
}

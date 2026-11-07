mod compact_mesh;
mod geometry_lines;
mod packed;
mod export;
mod scene_plot;
mod figure_scene;
mod metadata;
mod mesh_meta;
mod descriptor;
mod plot_export;
mod plot_from;
mod plot_apply;
mod parse;
#[cfg(test)]
mod tests;
use crate::core::{BoundingBox, Vertex};
use crate::plots::{
    AreaPlot, AxesKind, AxesMetadata, BarChart, ColorMap, ContourFillPlot, ContourPlot, ErrorBar,
    Figure, LegendEntry, LegendStyle, Line3Plot, LinePlot, MarkerStyle, MeshDeformation,
    MeshEdgeMode, MeshFieldLocation, MeshPlot, MeshRegion, MeshScalarField, MeshTriangleRange,
    MeshVectorField, NumericPlotData, PatchEdgeColorMode, PatchFaceColorMode, PatchPlot,
    PlotElement, PlotType, QuiverPlot, ReferenceLine, ReferenceLineOrientation, Scatter3Plot,
    ScatterPlot, ShadingMode, StairsPlot, StemPlot, SurfacePlot, TextStyle,
};
pub use geometry_lines::GeometryLines;
use glam::{Vec3, Vec4};
use runmat_value::NumericStorage;
use serde::{Deserialize, Serialize};
use std::{error::Error, fmt};

/// High-level event emitted whenever a figure changes.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct FigureEvent {
    pub handle: u32,
    pub kind: FigureEventKind,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub fingerprint: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub figure: Option<FigureSnapshot>,
}

/// Event kind for figure lifecycle + updates.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum FigureEventKind {
    Created,
    Updated,
    Cleared,
    Closed,
}

/// Snapshot of the figure state describing layout + plots.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct FigureSnapshot {
    pub layout: FigureLayout,
    pub metadata: FigureMetadata,
    pub plots: Vec<PlotDescriptor>,
}

fn orientation_triad_default() -> bool {
    true
}

fn orientation_triad_is_default(shown: &bool) -> bool {
    *shown
}

/// Full replay scene payload capable of reconstructing a figure.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct FigureScene {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub geometry_physical_context: Option<crate::geometry_scene::GeometryPhysicalContext>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub geometry_overlay: Option<crate::geometry_scene::GeometrySceneOverlay>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub geometry_lines: Vec<GeometryLines>,
    /// A geometry scene's corner orientation triad (see
    /// [`crate::geometry_scene::GeometryScene::show_orientation_triad`]); written only when off.
    #[serde(
        default = "orientation_triad_default",
        skip_serializing_if = "orientation_triad_is_default"
    )]
    pub geometry_orientation_triad: bool,
    pub schema_version: u32,
    pub layout: FigureLayout,
    pub metadata: FigureMetadata,
    pub plots: Vec<ScenePlot>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "dtype", content = "values", rename_all = "lowercase")]
pub enum SceneNumericStorage {
    // Nonfinite values are written losslessly; scenes saved before that wrote NaN as JSON
    // `null` (NaN marks gaps, e.g. holes in a surf), which still reads back as NaN.
    F64(
        #[serde(
            serialize_with = "runmat_types::serde_f64::vec::serialize",
            deserialize_with = "deserialize_vec_f64_lossy"
        )]
        Vec<f64>,
    ),
    F32(
        #[serde(
            serialize_with = "runmat_types::serde_f32::vec::serialize",
            deserialize_with = "deserialize_vec_f32_lossy"
        )]
        Vec<f32>,
    ),
    I8(Vec<i8>),
    I16(Vec<i16>),
    I32(Vec<i32>),
    I64(Vec<i64>),
    U8(Vec<u8>),
    U16(Vec<u16>),
    U32(Vec<u32>),
    U64(Vec<u64>),
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct SceneNumericData {
    pub storage: SceneNumericStorage,
    pub shape: Vec<usize>,
}

impl From<&NumericPlotData> for SceneNumericData {
    fn from(data: &NumericPlotData) -> Self {
        let storage = match data.storage() {
            NumericStorage::F64(values) => SceneNumericStorage::F64(values.clone()),
            NumericStorage::F32(values) => SceneNumericStorage::F32(values.clone()),
            NumericStorage::I8(values) => SceneNumericStorage::I8(values.clone()),
            NumericStorage::I16(values) => SceneNumericStorage::I16(values.clone()),
            NumericStorage::I32(values) => SceneNumericStorage::I32(values.clone()),
            NumericStorage::I64(values) => SceneNumericStorage::I64(values.clone()),
            NumericStorage::U8(values) => SceneNumericStorage::U8(values.clone()),
            NumericStorage::U16(values) => SceneNumericStorage::U16(values.clone()),
            NumericStorage::U32(values) => SceneNumericStorage::U32(values.clone()),
            NumericStorage::U64(values) => SceneNumericStorage::U64(values.clone()),
        };
        Self {
            storage,
            shape: data.shape().to_vec(),
        }
    }
}

impl TryFrom<SceneNumericData> for NumericPlotData {
    type Error = String;

    fn try_from(data: SceneNumericData) -> Result<Self, Self::Error> {
        let storage = match data.storage {
            SceneNumericStorage::F64(values) => NumericStorage::F64(values),
            SceneNumericStorage::F32(values) => NumericStorage::F32(values),
            SceneNumericStorage::I8(values) => NumericStorage::I8(values),
            SceneNumericStorage::I16(values) => NumericStorage::I16(values),
            SceneNumericStorage::I32(values) => NumericStorage::I32(values),
            SceneNumericStorage::I64(values) => NumericStorage::I64(values),
            SceneNumericStorage::U8(values) => NumericStorage::U8(values),
            SceneNumericStorage::U16(values) => NumericStorage::U16(values),
            SceneNumericStorage::U32(values) => NumericStorage::U32(values),
            SceneNumericStorage::U64(values) => NumericStorage::U64(values),
        };
        NumericPlotData::new(storage, data.shape)
    }
}

#[allow(unused_imports)]
pub use descriptor::*;
#[allow(unused_imports)]
pub use export::*;
#[allow(unused_imports)]
pub use figure_scene::*;
#[allow(unused_imports)]
pub use mesh_meta::*;
#[allow(unused_imports)]
pub use metadata::*;
#[allow(unused_imports)]
pub use parse::*;
#[allow(unused_imports)]
pub use plot_apply::*;
#[allow(unused_imports)]
pub use plot_export::*;
#[allow(unused_imports)]
pub use plot_from::*;
#[allow(unused_imports)]
pub use scene_plot::*;

use super::{ColorMap, NumericPlotData, PatchFaceColorMode, SurfaceColorMapping};
use crate::plots::color_data::{indexed_color, rgb_component, rgb_component_valid};
use glam::Vec4;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ColorLocation {
    Uniform,
    Face,
    Vertex,
    FaceOrVertex,
    CoordinateGrid,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub(super) struct PatchColorLayout {
    count: usize,
    rgb: bool,
    location: ColorLocation,
    extent: Option<(f64, f64)>,
}

pub(super) struct PatchColorContext<'a> {
    pub colormap: &'a ColorMap,
    pub colormap_length: usize,
    pub limits: Option<(f64, f64)>,
    pub mapping: SurfaceColorMapping,
    pub alpha: f32,
    pub source_indices: Option<&'a [usize]>,
}

impl PatchColorLayout {
    pub(super) fn validate(
        data: &NumericPlotData,
        face_count: usize,
        vertex_count: usize,
        coordinate_shape: Option<&[usize]>,
    ) -> Result<Self, String> {
        let shape = data.shape();
        let (count, rgb, location) = if let Some(coordinate_shape) = coordinate_shape {
            let count = coordinate_shape
                .iter()
                .try_fold(1_usize, |total, extent| total.checked_mul(*extent))
                .ok_or("patch: coordinate color extent overflows")?;
            let rgb = shape.len() == coordinate_shape.len() + 1
                && &shape[..coordinate_shape.len()] == coordinate_shape
                && shape.last() == Some(&3);
            let matching_vector =
                is_vector_shape(shape) && is_vector_shape(coordinate_shape) && data.len() == count;
            if shape != coordinate_shape && !matching_vector && !rgb {
                return Err("patch: coordinate CData must match XData or add an RGB plane".into());
            }
            (count, rgb, ColorLocation::CoordinateGrid)
        } else {
            Self::face_vertex_extent(shape, face_count, vertex_count)?
        };
        if count == 0 || count.checked_mul(if rgb { 3 } else { 1 }) != Some(data.len()) {
            return Err("patch: color data extent is invalid".into());
        }
        let extent = if rgb {
            for index in 0..data.len() {
                let value = data
                    .storage()
                    .value_at(index)
                    .expect("validated color extent");
                if !rgb_component_valid(value) {
                    return Err(
                        "patch: RGB components must be within their class range or NaN".into(),
                    );
                }
            }
            None
        } else {
            let mut lower = f64::INFINITY;
            let mut upper = f64::NEG_INFINITY;
            for index in 0..data.len() {
                let value = data
                    .storage()
                    .value_at(index)
                    .expect("validated color extent")
                    .materialize_f64();
                if value.is_infinite() {
                    return Err("patch: color indices must not be infinite".into());
                }
                if !value.is_nan() {
                    lower = lower.min(value);
                    upper = upper.max(value);
                }
            }
            (lower <= upper).then_some((lower, upper))
        };
        Ok(Self {
            count,
            rgb,
            location,
            extent,
        })
    }

    fn face_vertex_extent(
        shape: &[usize],
        face_count: usize,
        vertex_count: usize,
    ) -> Result<(usize, bool, ColorLocation), String> {
        let (count, rgb) = match shape {
            [rows, 3] => (*rows, true),
            [count] | [count, 1] | [1, count] => (*count, false),
            _ => {
                return Err(
                    "patch: color data must be a vector of indices or n-by-3 RGB rows".into(),
                )
            }
        };
        let location = match (count == face_count, count == vertex_count, count == 1) {
            (_, _, true) => ColorLocation::Uniform,
            (true, true, false) => ColorLocation::FaceOrVertex,
            (true, false, false) => ColorLocation::Face,
            _ if count == vertex_count => ColorLocation::Vertex,
            _ => return Err("patch: color data must match face or vertex count".into()),
        };
        Ok((count, rgb, location))
    }

    pub(super) fn supports(self, mode: PatchFaceColorMode) -> bool {
        !matches!(mode, PatchFaceColorMode::Interp)
            || matches!(
                self.location,
                ColorLocation::Vertex | ColorLocation::FaceOrVertex | ColorLocation::CoordinateGrid
            )
    }

    pub(super) fn color_for(
        self,
        data: &NumericPlotData,
        face_index: usize,
        vertex_index: usize,
        mode: PatchFaceColorMode,
        context: &PatchColorContext<'_>,
    ) -> Vec4 {
        let index = match self.location {
            ColorLocation::Uniform => 0,
            ColorLocation::Face => face_index,
            ColorLocation::Vertex => vertex_index,
            ColorLocation::FaceOrVertex => {
                if mode == PatchFaceColorMode::Interp {
                    vertex_index
                } else {
                    face_index
                }
            }
            ColorLocation::CoordinateGrid => context
                .source_indices
                .expect("validated coordinate color indices")[vertex_index],
        };
        let index = index.min(self.count - 1);
        if self.rgb {
            let red_sample = data.storage().value_at(index).expect("validated RGB");
            let green_sample = data
                .storage()
                .value_at(index + self.count)
                .expect("validated RGB");
            let blue_sample = data
                .storage()
                .value_at(index + self.count * 2)
                .expect("validated RGB");
            if [red_sample, green_sample, blue_sample]
                .into_iter()
                .any(|sample| sample.materialize_f64().is_nan())
            {
                return Vec4::ZERO;
            }
            let red = rgb_component(red_sample);
            let green = rgb_component(green_sample);
            let blue = rgb_component(blue_sample);
            Vec4::new(red, green, blue, context.alpha)
        } else {
            let value = data.storage().value_at(index).expect("validated index");
            indexed_color(
                value,
                context.colormap,
                context.colormap_length,
                (context.mapping == SurfaceColorMapping::Scaled)
                    .then(|| context.limits.or(self.extent))
                    .flatten(),
                context.alpha,
            )
        }
    }
}

pub(super) fn is_vector_shape(shape: &[usize]) -> bool {
    matches!(shape, [_] | [1, _] | [_, 1])
}

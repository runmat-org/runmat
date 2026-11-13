//! Compact export of geometry-scene surfaces and lines.
//!
//! Scene producers emit one vertex per triangle corner (flat normals, baked colors). Exported
//! as is, a dense quadratic FEA surface repeats every surface node about six times. The export
//! shares corners that coincide after quantization, keeps creases sharp by sharing only across
//! faces within the crease angle, and stores flat per-triangle colors once per triangle. Field
//! values are not touched here: colors are display data, and every saved legend keeps the
//! producer's extrema.

use crate::core::Vertex;
use std::collections::HashMap;

/// Faces meeting at more than this angle keep separate corners so the viewer's averaged
/// normals do not round a crease. Matches the feature-edge crease of FEA field scenes.
const CREASE_COSINE: f32 = 0.866_025_4; // cos 30°

/// Display colors keep three decimals: finer than an 8-bit display channel.
const COLOR_STEP: f32 = 1.0e-3;

/// Decimal quantization of exported positions.
#[derive(Clone, Copy, Debug)]
pub(crate) struct PositionGrid {
    scale: f64,
}

impl PositionGrid {
    /// A grid at about one millionth of the scene diagonal, in whole decimal digits, so the
    /// saved numbers stay short and the shape error stays far below a pixel.
    pub(crate) fn for_diagonal(diagonal: f32) -> Self {
        let diagonal = f64::from(diagonal);
        let decimals = if diagonal.is_finite() && diagonal > 0.0 {
            (-(diagonal * 1.0e-6).log10()).ceil().clamp(0.0, 12.0)
        } else {
            9.0
        };
        Self {
            scale: 10f64.powi(decimals as i32),
        }
    }

    /// The largest distance between an original and a quantized position.
    pub(crate) fn maximum_error(self) -> f64 {
        0.5 * 3f64.sqrt() / self.scale
    }

    fn quantize(self, position: [f32; 3]) -> [f32; 3] {
        position.map(|value| {
            let rounded = (f64::from(value) * self.scale).round() / self.scale;
            if rounded.is_finite() {
                rounded as f32
            } else {
                value
            }
        })
    }
}

fn quantize_color(color: [f32; 4]) -> [f32; 4] {
    color.map(|value| (value / COLOR_STEP).round() * COLOR_STEP)
}

fn bits3(value: [f32; 3]) -> [u32; 3] {
    value.map(f32::to_bits)
}

fn bits4(value: [f32; 4]) -> [u32; 4] {
    value.map(f32::to_bits)
}

fn face_normal(positions: [[f32; 3]; 3]) -> [f32; 3] {
    let [a, b, c] = positions.map(glam::Vec3::from_array);
    (b - a).cross(c - a).normalize_or_zero().to_array()
}

fn same_side(left: [f32; 3], right: [f32; 3]) -> bool {
    let dot = left[0] * right[0] + left[1] * right[1] + left[2] * right[2];
    dot >= CREASE_COSINE
}

/// A shared-vertex triangle surface ready for `ScenePlot::Mesh`.
pub(crate) struct CompactSurface {
    pub vertices: Vec<[f32; 3]>,
    pub triangles: Vec<[u32; 3]>,
    pub vertex_colors: Vec<[f32; 4]>,
    pub triangle_colors: Vec<[f32; 4]>,
}

#[derive(Clone, Copy)]
enum ColorMode {
    /// Every shared corner has one color.
    Vertex,
    /// Every triangle has one color.
    Triangle,
    /// Corners are shared only where position and color agree.
    Corner,
}

struct Corner {
    position: [f32; 3],
    color: [f32; 4],
    normal: [f32; 3],
}

pub(crate) fn compact_surface(
    vertices: &[Vertex],
    indices: &[u32],
    grid: PositionGrid,
) -> Option<CompactSurface> {
    let corners = indices
        .chunks_exact(3)
        .map(|triangle| {
            let source = [0, 1, 2].map(|corner| vertices.get(triangle[corner] as usize));
            let [Some(a), Some(b), Some(c)] = source else {
                return None;
            };
            let normal = face_normal([a.position, b.position, c.position]);
            Some(
                [a, b, c]
                    .iter()
                    .map(|vertex| Corner {
                        position: grid.quantize(vertex.position),
                        color: quantize_color(vertex.color),
                        normal,
                    })
                    .collect::<Vec<_>>(),
            )
        })
        .collect::<Option<Vec<_>>>()?;
    if corners.is_empty() {
        return None;
    }
    let flat = corners.iter().all(|triangle| {
        bits4(triangle[0].color) == bits4(triangle[1].color)
            && bits4(triangle[0].color) == bits4(triangle[2].color)
    });
    for mode in [ColorMode::Vertex, ColorMode::Triangle, ColorMode::Corner] {
        if matches!(mode, ColorMode::Triangle) && !flat {
            continue;
        }
        if let Some(surface) = weld(&corners, mode) {
            return Some(surface);
        }
    }
    None
}

/// Share corners by quantized position (and color in `Corner` mode), splitting at creases.
/// Returns `None` in `Vertex` mode when a shared corner would need two colors.
fn weld(corners: &[Vec<Corner>], mode: ColorMode) -> Option<CompactSurface> {
    let mut groups: HashMap<([u32; 3], [u32; 4]), Vec<u32>> = HashMap::new();
    let mut vertices = Vec::new();
    let mut normals: Vec<[f32; 3]> = Vec::new();
    let mut vertex_colors = Vec::new();
    let mut triangles = Vec::with_capacity(corners.len());
    let mut triangle_colors = Vec::new();
    for triangle in corners {
        let mut shared = [0u32; 3];
        for (slot, corner) in triangle.iter().enumerate() {
            let color_key = match mode {
                ColorMode::Corner => bits4(corner.color),
                ColorMode::Vertex | ColorMode::Triangle => [0; 4],
            };
            let candidates = groups
                .entry((bits3(corner.position), color_key))
                .or_default();
            let found = candidates
                .iter()
                .copied()
                .find(|&index| same_side(normals[index as usize], corner.normal));
            let index = match found {
                Some(index) => {
                    if matches!(mode, ColorMode::Vertex)
                        && bits4(vertex_colors[index as usize]) != bits4(corner.color)
                    {
                        return None;
                    }
                    index
                }
                None => {
                    let index = u32::try_from(vertices.len()).ok()?;
                    vertices.push(corner.position);
                    normals.push(corner.normal);
                    vertex_colors.push(corner.color);
                    candidates.push(index);
                    index
                }
            };
            shared[slot] = index;
        }
        triangles.push(shared);
        if matches!(mode, ColorMode::Triangle) {
            triangle_colors.push(triangle[0].color);
        }
    }
    if matches!(mode, ColorMode::Triangle) {
        vertex_colors.clear();
    }
    Some(CompactSurface {
        vertices,
        triangles,
        vertex_colors,
        triangle_colors,
    })
}

/// Shared, quantized line endpoints for `GeometryLines`.
pub(crate) fn compact_lines(
    vertices: &[Vertex],
    indices: Option<&[u32]>,
    grid: PositionGrid,
) -> (Vec<[f32; 3]>, Vec<[f32; 4]>, Vec<u32>) {
    let order: Vec<u32> = match indices {
        Some(indices) => indices.to_vec(),
        None => (0..vertices.len() as u32).collect(),
    };
    let mut lookup: HashMap<([u32; 3], [u32; 4]), u32> = HashMap::new();
    let mut positions = Vec::new();
    let mut colors = Vec::new();
    let mut shared = Vec::with_capacity(order.len());
    for index in order {
        let Some(vertex) = vertices.get(index as usize) else {
            continue;
        };
        let position = grid.quantize(vertex.position);
        let color = quantize_color(vertex.color);
        let next = positions.len() as u32;
        let slot = *lookup
            .entry((bits3(position), bits4(color)))
            .or_insert_with(|| {
                positions.push(position);
                colors.push(color);
                next
            });
        shared.push(slot);
    }
    (positions, colors, shared)
}

#[cfg(test)]
mod tests {
    use super::*;
    use glam::{Vec3, Vec4};

    fn vertex(position: [f32; 3], color: [f32; 4]) -> Vertex {
        Vertex::new(Vec3::from_array(position), Vec4::from_array(color))
    }

    fn corners(triangles: &[[[f32; 3]; 3]], colors: &[[f32; 4]; 3]) -> (Vec<Vertex>, Vec<u32>) {
        let mut vertices = Vec::new();
        for triangle in triangles {
            for (corner, position) in triangle.iter().enumerate() {
                vertices.push(vertex(*position, colors[corner]));
            }
        }
        let indices = (0..vertices.len() as u32).collect();
        (vertices, indices)
    }

    #[test]
    fn coplanar_corners_share_and_creases_split() {
        let red = [1.0, 0.0, 0.0, 1.0];
        let grid = PositionGrid::for_diagonal(1.0);
        // Two coplanar triangles share an edge: four vertices.
        let (vertices, indices) = corners(
            &[
                [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
                [[1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]],
            ],
            &[red; 3],
        );
        let surface = compact_surface(&vertices, &indices, grid).unwrap();
        assert_eq!(surface.vertices.len(), 4);
        assert_eq!(surface.vertex_colors.len(), 4);
        // A 90° fold keeps the shared edge's corners apart.
        let (vertices, indices) = corners(
            &[
                [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
                [[1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
            ],
            &[red; 3],
        );
        let surface = compact_surface(&vertices, &indices, grid).unwrap();
        assert_eq!(surface.vertices.len(), 6);
    }

    #[test]
    fn flat_triangle_colors_are_stored_once_per_triangle() {
        let grid = PositionGrid::for_diagonal(1.0);
        let mut vertices = Vec::new();
        let colors = [[1.0, 0.0, 0.0, 1.0], [0.0, 0.0, 1.0, 1.0]];
        let triangles = [
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            [[1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]],
        ];
        for (triangle, color) in triangles.iter().zip(colors) {
            for position in triangle {
                vertices.push(vertex(*position, color));
            }
        }
        let indices: Vec<u32> = (0..6).collect();
        let surface = compact_surface(&vertices, &indices, grid).unwrap();
        assert_eq!(surface.vertices.len(), 4);
        assert!(surface.vertex_colors.is_empty());
        assert_eq!(surface.triangle_colors, colors.to_vec());
    }

    #[test]
    fn quantization_error_is_bounded_by_the_grid() {
        let grid = PositionGrid::for_diagonal(0.1);
        let original = [0.012_345_678_f32, -0.098_765_43, 0.05];
        let quantized = grid.quantize(original);
        let error = original
            .iter()
            .zip(quantized)
            .map(|(a, b)| f64::from(a - b).powi(2))
            .sum::<f64>()
            .sqrt();
        assert!(error <= grid.maximum_error() + 1e-9, "{error}");
        assert!(grid.maximum_error() < 1.0e-7);
    }
}

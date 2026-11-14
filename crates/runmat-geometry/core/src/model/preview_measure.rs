//! Size-independent measurements of an imported preview surface, taken at check time so a
//! solver mesh's size can be estimated before meshing.
//!
//! The measure holds the enclosed volume, the surface area, and an area spectrum: how much of
//! the surface lies at each curvature radius, wall thickness and facing-surface gap. Those three
//! lengths are the ones a mesher's sizing answers to (chordal deviation on curved faces, element
//! layers across thin walls and narrow gaps), so the spectrum lets a mesh estimate apply any
//! mesh settings without re-reading the surface. Everything is in canonical metres.
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashMap};

use super::GeometryAsset;
use crate::selection::EntityKind;

#[path = "preview_measure/ray_grid.rs"]
mod ray_grid;

/// Rays cast per measure at most. Larger previews sample every k-th triangle; the sampled
/// spectrum is scaled to the whole surface area.
const MAXIMUM_RAYS: usize = 40_000;
/// Spectrum bins per octave of each length.
const BINS_PER_OCTAVE: f64 = 4.0;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PreviewMeasure {
    /// Volume the preview surface encloses, m^3.
    pub volume_m3: f64,
    /// Area of the preview surface, m^2.
    pub area_m2: f64,
    /// The surface area by curvature radius, wall thickness and gap. Areas sum to `area_m2`.
    pub patches: Vec<PreviewSurfacePatch>,
}

/// Surface area that shares one curvature radius, wall thickness and gap (within a quarter
/// octave of each).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PreviewSurfacePatch {
    pub area_m2: f64,
    /// The smaller principal radius of curvature; absent on flat surface.
    pub radius_m: Option<f64>,
    /// Material thickness behind the surface, along the inward normal; absent when the ray
    /// leaves the preview (an open or inconsistently oriented surface).
    pub thickness_m: Option<f64>,
    /// Distance to the nearest surface facing it across empty space, along the outward normal.
    pub gap_m: Option<f64>,
}

impl PreviewMeasure {
    /// Measures the preview surface meshes of `asset`. `None` without a resolved physical
    /// scale, without triangles, or when the surface encloses no volume.
    pub fn measure(asset: &GeometryAsset) -> Option<Self> {
        let scale = asset.coordinate_units?.meters_per_unit();
        let surface = Surface::collect(asset, scale)?;
        let signed_volume: f64 = surface.triangles.iter().map(|t| t.signed_volume).sum();
        let area_m2: f64 = surface.triangles.iter().map(|t| t.area).sum();
        let volume_m3 = signed_volume.abs();
        if !(volume_m3.is_finite() && volume_m3 > 0.0 && area_m2.is_finite() && area_m2 > 0.0) {
            return None;
        }
        // Outward normals enclose positive volume; a reversed surface flips every normal.
        let outward = signed_volume.signum();
        let radii = surface.curvature_radii();
        let grid = ray_grid::RayGrid::build(&surface.positions, &surface.triangles, outward);
        let stride = surface.triangles.len().div_ceil(MAXIMUM_RAYS).max(1);
        let mut bins = BTreeMap::<[i32; 3], [f64; 4]>::new();
        let mut sampled_area = 0.0;
        for (index, triangle) in surface.triangles.iter().enumerate().step_by(stride) {
            let normal = triangle.unit_normal.map(|x| x * outward);
            let inward = normal.map(|x| -x);
            let thickness = grid.first_hit(index, triangle.centroid, inward, true);
            let gap = grid.first_hit(index, triangle.centroid, normal, false);
            let radius = radii[index];
            let key = [bin(radius), bin(thickness), bin(gap)];
            let entry = bins.entry(key).or_insert([0.0; 4]);
            entry[0] += triangle.area;
            entry[1] += triangle.area * radius.unwrap_or(0.0);
            entry[2] += triangle.area * thickness.unwrap_or(0.0);
            entry[3] += triangle.area * gap.unwrap_or(0.0);
            sampled_area += triangle.area;
        }
        let to_whole = area_m2 / sampled_area;
        let patches = bins
            .into_iter()
            .filter(|(_, sums)| sums[0] > 0.0)
            .map(|(key, sums)| {
                let mean = |slot: usize, bin: i32| (bin != i32::MAX).then(|| sums[slot] / sums[0]);
                PreviewSurfacePatch {
                    area_m2: sums[0] * to_whole,
                    radius_m: mean(1, key[0]),
                    thickness_m: mean(2, key[1]),
                    gap_m: mean(3, key[2]),
                }
            })
            .collect();
        Some(Self {
            volume_m3,
            area_m2,
            patches,
        })
    }
}

fn bin(length: Option<f64>) -> i32 {
    length.map_or(i32::MAX, |x| (x.log2() * BINS_PER_OCTAVE).floor() as i32)
}

struct Triangle {
    corners: [usize; 3],
    /// Smooth-surface identity: triangles of one CAD face share it.
    face: u64,
    area: f64,
    signed_volume: f64,
    unit_normal: [f64; 3],
    centroid: [f64; 3],
}

struct Surface {
    positions: Vec<[f64; 3]>,
    triangles: Vec<Triangle>,
}

impl Surface {
    fn collect(asset: &GeometryAsset, scale: f64) -> Option<Self> {
        let mut positions = Vec::new();
        let mut triangles = Vec::new();
        for (mesh_index, mesh) in asset.surface_meshes.iter().enumerate() {
            let faces = triangle_faces(asset, &mesh.mesh_id, mesh.triangles.len());
            let base = positions.len();
            positions.extend(mesh.vertices.iter().map(|p| p.map(|x| x * scale)));
            for (index, corners) in mesh.triangles.iter().enumerate() {
                let corners = corners.map(|corner| base + corner as usize);
                let [a, b, c] = corners.map(|corner| positions[corner]);
                let normal = cross(sub(b, a), sub(c, a));
                let doubled_area = norm(normal);
                if !(doubled_area.is_finite() && doubled_area > 0.0) {
                    continue;
                }
                // A triangle outside every face region is its own smooth patch: no curvature
                // is read across an unknown crease.
                let face = faces
                    .get(index)
                    .copied()
                    .flatten()
                    .map_or(u64::MAX - triangles.len() as u64, |region| {
                        ((mesh_index as u64) << 32) | region as u64
                    });
                triangles.push(Triangle {
                    corners,
                    face,
                    area: doubled_area / 2.0,
                    signed_volume: dot(a, cross(b, c)) / 6.0,
                    unit_normal: normal.map(|x| x / doubled_area),
                    centroid: std::array::from_fn(|axis| (a[axis] + b[axis] + c[axis]) / 3.0),
                });
            }
        }
        (!triangles.is_empty()).then_some(Self {
            positions,
            triangles,
        })
    }

    /// The smaller principal radius at each triangle: the largest normal turn per unit length
    /// along its edges, from area-weighted vertex normals within its CAD face. A cylinder of
    /// radius R reads 1/R along its circumference for any tessellation density.
    fn curvature_radii(&self) -> Vec<Option<f64>> {
        let key = |triangle: &Triangle, corner: usize| {
            let p = self.positions[triangle.corners[corner]];
            (triangle.face, p.map(f64::to_bits))
        };
        let mut normals = HashMap::<(u64, [u64; 3]), [f64; 3]>::new();
        for triangle in &self.triangles {
            for corner in 0..3 {
                let sum = normals.entry(key(triangle, corner)).or_insert([0.0; 3]);
                for (axis, value) in sum.iter_mut().enumerate() {
                    *value += triangle.unit_normal[axis] * triangle.area;
                }
            }
        }
        for normal in normals.values_mut() {
            let length = norm(*normal);
            if length > 0.0 {
                *normal = normal.map(|x| x / length);
            }
        }
        self.triangles
            .iter()
            .map(|triangle| {
                let curvature = (0..3)
                    .map(|edge| {
                        let (u, v) = (edge, (edge + 1) % 3);
                        let length = norm(sub(
                            self.positions[triangle.corners[u]],
                            self.positions[triangle.corners[v]],
                        ));
                        let turn =
                            norm(sub(normals[&key(triangle, u)], normals[&key(triangle, v)]));
                        turn / length
                    })
                    .fold(0.0_f64, f64::max);
                // Below a thousandth of a radian across an edge the face is flat.
                let edge = (triangle.area * 2.0).sqrt();
                (curvature.is_finite() && curvature * edge > 1.0e-3).then(|| 1.0 / curvature)
            })
            .collect()
    }
}

/// The face region of each triangle of `mesh_id`, from the asset's face mappings. A region
/// that spans several faces (a body or a named group) also maps triangles; each triangle takes
/// the smallest region that holds it, its own CAD face.
fn triangle_faces(asset: &GeometryAsset, mesh_id: &str, count: usize) -> Vec<Option<u32>> {
    let mut faces = vec![None; count];
    let mut mappings: Vec<_> = asset
        .region_entity_mappings
        .iter()
        .filter(|mapping| mapping.mesh_id == mesh_id && mapping.entity_kind == EntityKind::Face)
        .enumerate()
        .collect();
    let size = |mapping: &crate::model::RegionEntityMapping| -> u64 {
        mapping.ranges.iter().map(|range| range.count).sum()
    };
    mappings.sort_by_key(|(region, mapping)| (size(mapping), *region));
    for (region, mapping) in mappings {
        for range in &mapping.ranges {
            let start = usize::try_from(range.start)
                .unwrap_or(usize::MAX)
                .min(count);
            let end = usize::try_from(range.start.saturating_add(range.count))
                .unwrap_or(usize::MAX)
                .min(count);
            for face in &mut faces[start..end] {
                face.get_or_insert(region as u32);
            }
        }
    }
    faces
}

fn sub(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

fn cross(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

fn dot(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn norm(a: [f64; 3]) -> f64 {
    dot(a, a).sqrt()
}

#[cfg(test)]
#[path = "preview_measure_tests.rs"]
mod tests;

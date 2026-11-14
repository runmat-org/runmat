use super::PreviewMeasure;
use crate::model::{
    GeometryAsset, GeometrySource, MeshDescriptor, MeshKind, RegionEntityMapping, SourceGeometry,
    SourceGeometryKind, SurfaceMesh, TessellationProfile,
};
use crate::physical::{CoordinateUnits, LengthUnit};
use crate::selection::EntityKind;
use crate::EntityIdRange;
use std::f64::consts::TAU;

/// One CAD face as a (u, v) grid; the triangle normal follows d/du x d/dv, reversed by `flip`.
struct Faces {
    vertices: Vec<[f64; 3]>,
    triangles: Vec<[u32; 3]>,
    ranges: Vec<(u64, u64)>,
}

impl Faces {
    fn face(&mut self, nu: usize, nv: usize, flip: bool, at: impl Fn(f64, f64) -> [f64; 3]) {
        let base = self.vertices.len() as u32;
        let first = self.triangles.len() as u64;
        for j in 0..=nv {
            for i in 0..=nu {
                self.vertices
                    .push(at(i as f64 / nu as f64, j as f64 / nv as f64));
            }
        }
        let index = |i: usize, j: usize| base + (j * (nu + 1) + i) as u32;
        for j in 0..nv {
            for i in 0..nu {
                let (a, b, c, d) = (
                    index(i, j),
                    index(i + 1, j),
                    index(i + 1, j + 1),
                    index(i, j + 1),
                );
                for triangle in [[a, b, c], [a, c, d]] {
                    self.triangles.push(if flip {
                        [triangle[0], triangle[2], triangle[1]]
                    } else {
                        triangle
                    });
                }
            }
        }
        self.ranges
            .push((first, self.triangles.len() as u64 - first));
    }

    fn asset(self) -> GeometryAsset {
        let mappings = self
            .ranges
            .iter()
            .enumerate()
            .map(|(face, (start, count))| {
                RegionEntityMapping::new(
                    format!("face_{face}"),
                    "mesh_1",
                    EntityKind::Face,
                    vec![EntityIdRange::new(*start, *count)],
                )
            })
            .collect();
        GeometryAsset {
            geometry_id: "fixture".into(),
            source: GeometrySource {
                unit_evidence: Default::default(),
                path: "fixture.step".into(),
                sha256: String::new(),
                importer_version: "fixture".into(),
            },
            source_geometry: SourceGeometry {
                kind: SourceGeometryKind::Cad,
                assembly: None,
                material_evidence: vec![],
                exchange: None,
                cad_evaluators: Vec::new(),
            },
            tessellation_profile: TessellationProfile::default(),
            coordinate_units: Some(CoordinateUnits::new(LengthUnit::Millimeter)),
            revision: 1,
            meshes: vec![MeshDescriptor {
                mesh_id: "mesh_1".into(),
                kind: MeshKind::Surface,
                vertex_count: self.vertices.len() as u64,
                element_count: self.triangles.len() as u64,
            }],
            surface_meshes: vec![SurfaceMesh::new("mesh_1", self.vertices, self.triangles)],
            regions: Vec::new(),
            region_entity_mappings: mappings,
            diagnostics: vec![],
        }
    }
}

/// A tube in millimetres: outer radius 10, bore radius 8, length 20, 64 segments.
fn tube() -> GeometryAsset {
    let (outer, inner, length, n) = (10.0, 8.0, 20.0, 64);
    // The seam (u = 1) repeats the u = 0 points bit for bit, as a periodic face's seam does.
    let ring = |r: f64, u: f64, z: f64| {
        let angle = TAU * (u % 1.0);
        [r * angle.cos(), r * angle.sin(), z]
    };
    let mut faces = Faces {
        vertices: Vec::new(),
        triangles: Vec::new(),
        ranges: Vec::new(),
    };
    faces.face(n, 4, false, |u, v| ring(outer, u, v * length));
    faces.face(n, 4, true, |u, v| ring(inner, u, v * length));
    faces.face(4, n, false, |u, v| {
        ring(inner + u * (outer - inner), v, length)
    });
    faces.face(4, n, true, |u, v| ring(inner + u * (outer - inner), v, 0.0));
    faces.asset()
}

fn area_where(measure: &PreviewMeasure, keep: impl Fn(&super::PreviewSurfacePatch) -> bool) -> f64 {
    measure
        .patches
        .iter()
        .filter(|p| keep(p))
        .map(|p| p.area_m2)
        .sum()
}

#[test]
fn tube_measures_volume_area_curvature_thickness_and_bore_gap() {
    let measure = PreviewMeasure::measure(&tube()).expect("closed tube");
    let polygon = 64.0 / 2.0 * (TAU / 64.0).sin();
    let volume = polygon * (100.0 - 64.0) * 20.0 * 1.0e-9;
    assert!(
        (measure.volume_m3 - volume).abs() < 1.0e-6 * volume,
        "{}",
        measure.volume_m3
    );
    let total: f64 = measure.patches.iter().map(|p| p.area_m2).sum();
    assert!((total - measure.area_m2).abs() < 1.0e-9 * measure.area_m2);
    // Outer wall: radius 10 mm, 2 mm of material behind it, nothing facing it.
    let outer = area_where(&measure, |p| {
        p.radius_m.is_some_and(|r| (r - 0.010).abs() < 0.001)
            && p.thickness_m.is_some_and(|t| (t - 0.002).abs() < 2.0e-4)
            && p.gap_m.is_none()
    });
    let lateral = TAU * 0.010 * 0.020;
    assert!(
        (outer - lateral).abs() < 0.02 * lateral,
        "{outer} vs {lateral}: {measure:?}"
    );
    // Bore: radius 8 mm, the opposite bore wall 16 mm away.
    let bore = area_where(&measure, |p| {
        p.radius_m.is_some_and(|r| (r - 0.008).abs() < 0.001)
            && p.gap_m.is_some_and(|g| (g - 0.016).abs() < 0.001)
    });
    assert!(
        (bore - TAU * 0.008 * 0.020).abs() < 0.03 * bore,
        "{bore}: {measure:?}"
    );
    // End faces are flat with the tube length behind them.
    let ends = area_where(&measure, |p| {
        p.radius_m.is_none() && p.thickness_m.is_some_and(|t| (t - 0.020).abs() < 0.002)
    });
    let annulus = 2.0 * polygon * (100.0 - 64.0) * 1.0e-6;
    assert!(
        (ends - annulus).abs() < 0.02 * annulus,
        "{ends} vs {annulus}"
    );
}

#[test]
fn reversed_surface_measures_the_same_solid() {
    let mut asset = tube();
    for triangle in &mut asset.surface_meshes[0].triangles {
        triangle.swap(1, 2);
    }
    let reversed = PreviewMeasure::measure(&asset).unwrap();
    let measure = PreviewMeasure::measure(&tube()).unwrap();
    assert!((reversed.volume_m3 - measure.volume_m3).abs() < 1.0e-15);
    assert_eq!(reversed.patches.len(), measure.patches.len());
}

#[test]
fn unscaled_or_empty_previews_have_no_measure() {
    let mut asset = tube();
    asset.coordinate_units = None;
    assert_eq!(PreviewMeasure::measure(&asset), None);
    let mut empty = tube();
    empty.surface_meshes.clear();
    assert_eq!(PreviewMeasure::measure(&empty), None);
}

#[test]
fn a_region_spanning_every_face_reads_no_curvature_across_edges() {
    let mut asset = tube();
    let triangles = asset.surface_meshes[0].triangles.len() as u64;
    asset.region_entity_mappings.insert(
        0,
        RegionEntityMapping::new(
            "body",
            "mesh_1",
            EntityKind::Face,
            vec![EntityIdRange::new(0, triangles)],
        ),
    );
    assert_eq!(
        PreviewMeasure::measure(&asset),
        PreviewMeasure::measure(&tube())
    );
}

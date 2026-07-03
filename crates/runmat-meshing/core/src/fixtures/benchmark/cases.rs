use super::*;
use crate::{
    size::field::{MeshSizingField, SizingSample},
    validation::AnalysisMeshValidationOptions,
    MeshTargetSize, RefinementFocusLevel, VolumeMeshingOptions,
};
use runmat_geometry_core::{
    EntityIdRange, EntityKind, GeometryAsset, GeometrySource, MeshDescriptor, MeshKind, Region,
    RegionEntityMapping, SourceGeometry, SourceGeometryKind, SurfaceMesh, TessellationProfile,
    UnitSystem,
};

pub fn generic_mesh_benchmark_cases() -> Vec<MeshBenchmarkCase> {
    vec![
        solid_box_benchmark_case(
            "solid_cube",
            MeshBenchmarkTier::Solid3d,
            [1.0, 1.0, 1.0],
            1.0,
            6.0,
            1,
        ),
        solid_box_benchmark_case(
            "thin_slab",
            MeshBenchmarkTier::ThinFeature,
            [1.0, 1.0, 0.1],
            0.1,
            2.4,
            1,
        ),
        through_hole_block_benchmark_case(),
        faceted_cylinder_benchmark_case(),
        tapered_arm_benchmark_case(),
        disconnected_boxes_benchmark_case(),
        boundary_load_patch_benchmark_case(),
        adaptive_refinement_benchmark_case(),
    ]
}

fn solid_box_benchmark_case(
    benchmark_id: &str,
    tier: MeshBenchmarkTier,
    dimensions_m: [f64; 3],
    expected_volume_m3: f64,
    expected_boundary_area_m2: f64,
    max_volume_component_count: usize,
) -> MeshBenchmarkCase {
    let geometry = box_geometry(benchmark_id, dimensions_m, [0.0, 0.0, 0.0]);
    benchmark_case(
        benchmark_id,
        tier,
        geometry,
        expected_volume_m3,
        expected_boundary_area_m2,
        max_volume_component_count,
    )
}

fn disconnected_boxes_benchmark_case() -> MeshBenchmarkCase {
    let first = box_surface([1.0, 1.0, 1.0], [0.0, 0.0, 0.0], 0);
    let second = box_surface([1.0, 1.0, 1.0], [1.6, 0.0, 0.0], 8);
    let mut vertices = first.0;
    vertices.extend(second.0);
    let mut triangles = first.1;
    triangles.extend(second.1);
    let geometry = geometry_from_surface(
        "disconnected_boxes",
        "generic_disconnected_boxes_surface",
        vertices,
        triangles,
    );
    benchmark_case(
        "disconnected_boxes",
        MeshBenchmarkTier::MultiBody,
        geometry,
        2.0,
        12.0,
        2,
    )
}

fn through_hole_block_benchmark_case() -> MeshBenchmarkCase {
    let outer = [1.0, 1.0, 1.0];
    let hole_min = [0.35, 0.35];
    let hole_max = [0.65, 0.65];
    let (vertices, triangles) = through_hole_block_surface(outer, hole_min, hole_max);
    let hole_width = hole_max[0] - hole_min[0];
    let hole_depth = hole_max[1] - hole_min[1];
    let expected_volume_m3 = outer[0] * outer[1] * outer[2] - hole_width * hole_depth * outer[2];
    let expected_boundary_area_m2 = 2.0 * (outer[0] * outer[1] - hole_width * hole_depth)
        + 2.0 * (outer[0] + outer[1]) * outer[2]
        + 2.0 * (hole_width + hole_depth) * outer[2];
    benchmark_case(
        "through_hole_block",
        MeshBenchmarkTier::HoleFeature,
        geometry_from_surface(
            "through_hole_block",
            "generic_through_hole_block_surface",
            vertices,
            triangles,
        ),
        expected_volume_m3,
        expected_boundary_area_m2,
        1,
    )
}

#[cfg(test)]
fn annular_bore_block_benchmark_case() -> MeshBenchmarkCase {
    let dimensions_m = [1.0, 1.0, 0.8];
    let bore_radius_m = 0.18;
    let segment_count = 12_usize;
    let (vertices, triangles) =
        annular_bore_block_surface(dimensions_m, bore_radius_m, segment_count);
    let expected_volume_m3 = closed_surface_volume_m3(&vertices, &triangles);
    let expected_boundary_area_m2 = closed_surface_area_m2(&vertices, &triangles);
    benchmark_case(
        "annular_bore_block",
        MeshBenchmarkTier::HoleFeature,
        geometry_from_surface(
            "annular_bore_block",
            "generic_annular_bore_block_surface",
            vertices,
            triangles,
        ),
        expected_volume_m3,
        expected_boundary_area_m2,
        1,
    )
}

fn faceted_cylinder_benchmark_case() -> MeshBenchmarkCase {
    let segment_count = 16_usize;
    let radius_m = 0.5_f64;
    let height_m = 1.0_f64;
    let (vertices, triangles) = faceted_cylinder_surface(segment_count, radius_m, height_m);
    let polygon_area = 0.5
        * segment_count as f64
        * radius_m.powi(2)
        * (std::f64::consts::TAU / segment_count as f64).sin();
    let polygon_perimeter =
        2.0 * segment_count as f64 * radius_m * (std::f64::consts::PI / segment_count as f64).sin();
    benchmark_case(
        "faceted_cylinder",
        MeshBenchmarkTier::CurvedSurface,
        geometry_from_surface(
            "faceted_cylinder",
            "generic_faceted_cylinder_surface",
            vertices,
            triangles,
        ),
        polygon_area * height_m,
        2.0 * polygon_area + polygon_perimeter * height_m,
        1,
    )
}

fn tapered_arm_benchmark_case() -> MeshBenchmarkCase {
    let (vertices, triangles) = tapered_rectangular_prism_surface([0.8, 0.5], [0.55, 0.35], 0.8);
    let expected_volume_m3 = closed_surface_volume_m3(&vertices, &triangles);
    let expected_boundary_area_m2 = closed_surface_area_m2(&vertices, &triangles);
    benchmark_case(
        "tapered_arm",
        MeshBenchmarkTier::Solid3d,
        geometry_from_surface(
            "tapered_arm",
            "generic_tapered_arm_surface",
            vertices,
            triangles,
        ),
        expected_volume_m3,
        expected_boundary_area_m2,
        1,
    )
}

fn adaptive_refinement_benchmark_case() -> MeshBenchmarkCase {
    let mut case = benchmark_case(
        "adaptive_refinement_marker",
        MeshBenchmarkTier::AdaptiveRefinement,
        box_geometry(
            "adaptive_refinement_marker",
            [1.0, 1.0, 1.0],
            [0.0, 0.0, 0.0],
        ),
        1.0,
        6.0,
        1,
    );
    case.options.target_size = MeshTargetSize::LengthM(2.0);
    case.options.refinement.focus.curvature = false;
    case.options.refinement.focus.small_features = false;
    case.options.refinement.focus.interfaces = RefinementFocusLevel::Off;
    case.sizing = Some(MeshSizingField {
        samples: vec![SizingSample {
            position_m: [0.25, 0.25, 0.25],
            target_size_m: 0.50,
            reason: Some("structural.stress_gradient".to_string()),
        }],
        ..MeshSizingField::default()
    });
    case
}

fn boundary_load_patch_benchmark_case() -> MeshBenchmarkCase {
    let mut case = benchmark_case(
        "boundary_load_patch",
        MeshBenchmarkTier::SizingField,
        box_geometry("boundary_load_patch", [1.0, 1.0, 1.0], [0.0, 0.0, 0.0]),
        1.0,
        6.0,
        1,
    );
    case.options.target_size = MeshTargetSize::LengthM(1.0);
    case.options.refinement.focus.curvature = false;
    case.options.refinement.focus.small_features = false;
    case.options.refinement.focus.interfaces = RefinementFocusLevel::Off;
    case.validation.required_boundary_region_ids =
        vec!["benchmark_root".to_string(), "benchmark_tip".to_string()];
    case.validation.required_material_region_ids =
        vec!["benchmark_root".to_string(), "benchmark_tip".to_string()];
    case.sizing = Some(MeshSizingField {
        samples: vec![
            SizingSample {
                position_m: [1.0, 0.5, 0.5],
                target_size_m: 0.25,
                reason: Some("structural.load_regions".to_string()),
            },
            SizingSample {
                position_m: [0.5, 0.0, 0.5],
                target_size_m: 0.25,
                reason: Some("structural.constraint_regions".to_string()),
            },
        ],
        ..MeshSizingField::default()
    });
    case
}

fn benchmark_case(
    benchmark_id: &str,
    tier: MeshBenchmarkTier,
    geometry: GeometryAsset,
    expected_volume_m3: f64,
    expected_boundary_area_m2: f64,
    max_volume_component_count: usize,
) -> MeshBenchmarkCase {
    let characteristic_size = expected_volume_m3.cbrt() / 2.0;
    let options = VolumeMeshingOptions {
        target_size: MeshTargetSize::LengthM(characteristic_size.max(0.02)),
        max_elements: GENERIC_BENCHMARK_MAX_ELEMENTS,
        ..VolumeMeshingOptions::default()
    };
    MeshBenchmarkCase {
        benchmark_id: benchmark_id.to_string(),
        tier,
        geometry,
        options,
        sizing: None,
        validation: AnalysisMeshValidationOptions {
            max_volume_element_count: Some(GENERIC_BENCHMARK_MAX_ELEMENTS),
            expected_volume_m3: Some(expected_volume_m3),
            expected_boundary_area_m2: Some(expected_boundary_area_m2),
            max_volume_component_count: Some(max_volume_component_count),
            min_boundary_face_recovery_ratio: 1.0,
            min_boundary_edge_recovery_ratio: 1.0,
            require_no_fan_fallback: true,
            require_no_unrepaired_exact_quality: true,
            ..AnalysisMeshValidationOptions::default()
        },
    }
}

fn box_geometry(benchmark_id: &str, dimensions_m: [f64; 3], origin_m: [f64; 3]) -> GeometryAsset {
    let (vertices, triangles) = box_surface(dimensions_m, origin_m, 0);
    geometry_from_surface(
        benchmark_id,
        &format!("generic_{benchmark_id}_surface"),
        vertices,
        triangles,
    )
}

fn geometry_from_surface(
    geometry_suffix: &str,
    mesh_id: &str,
    vertices: Vec<[f64; 3]>,
    triangles: Vec<[u32; 3]>,
) -> GeometryAsset {
    let face_count = triangles.len() as u64;
    GeometryAsset {
        geometry_id: format!("geo_benchmark_{geometry_suffix}"),
        source: GeometrySource {
            path: format!("/fixtures/{geometry_suffix}.step"),
            sha256: format!("generic-{geometry_suffix}"),
            importer_version: "benchmark-fixture/v1".to_string(),
        },
        source_geometry: SourceGeometry {
            kind: SourceGeometryKind::Cad,
            assembly: None,
            material_evidence: Vec::new(),
            cad_evaluators: Vec::new(),
        },
        tessellation_profile: TessellationProfile::default(),
        units: UnitSystem::Meter,
        revision: 1,
        meshes: vec![MeshDescriptor {
            mesh_id: mesh_id.to_string(),
            kind: MeshKind::Surface,
            vertex_count: vertices.len() as u64,
            element_count: face_count,
        }],
        surface_meshes: vec![SurfaceMesh::new(mesh_id, vertices, triangles)],
        regions: vec![
            Region {
                region_id: "benchmark_root".to_string(),
                name: "benchmark_root".to_string(),
                tag: Some("support".to_string()),
                cad_ownership: None,
            },
            Region {
                region_id: "benchmark_tip".to_string(),
                name: "benchmark_tip".to_string(),
                tag: Some("load".to_string()),
                cad_ownership: None,
            },
        ],
        region_entity_mappings: vec![
            RegionEntityMapping::new(
                "benchmark_root",
                mesh_id,
                EntityKind::Face,
                vec![EntityIdRange::new(0, face_count / 2)],
            ),
            RegionEntityMapping::new(
                "benchmark_tip",
                mesh_id,
                EntityKind::Face,
                vec![EntityIdRange::new(
                    face_count / 2,
                    face_count - face_count / 2,
                )],
            ),
        ],
        diagnostics: Vec::new(),
    }
}

fn box_surface(
    dimensions_m: [f64; 3],
    origin_m: [f64; 3],
    node_offset: u32,
) -> (Vec<[f64; 3]>, Vec<[u32; 3]>) {
    let [sx, sy, sz] = dimensions_m;
    let [ox, oy, oz] = origin_m;
    let vertices = vec![
        [ox, oy, oz],
        [ox + sx, oy, oz],
        [ox + sx, oy + sy, oz],
        [ox, oy + sy, oz],
        [ox, oy, oz + sz],
        [ox + sx, oy, oz + sz],
        [ox + sx, oy + sy, oz + sz],
        [ox, oy + sy, oz + sz],
    ];
    let triangles = [
        [0, 2, 1],
        [0, 3, 2],
        [4, 5, 6],
        [4, 6, 7],
        [0, 1, 5],
        [0, 5, 4],
        [1, 2, 6],
        [1, 6, 5],
        [2, 3, 7],
        [2, 7, 6],
        [3, 0, 4],
        [3, 4, 7],
    ]
    .into_iter()
    .map(|triangle| {
        [
            triangle[0] + node_offset,
            triangle[1] + node_offset,
            triangle[2] + node_offset,
        ]
    })
    .collect();
    (vertices, triangles)
}

fn tapered_rectangular_prism_surface(
    base_size_m: [f64; 2],
    tip_size_m: [f64; 2],
    length_m: f64,
) -> (Vec<[f64; 3]>, Vec<[u32; 3]>) {
    let [base_y, base_z] = base_size_m;
    let [tip_y, tip_z] = tip_size_m;
    let vertices = vec![
        [0.0, -base_y / 2.0, -base_z / 2.0],
        [0.0, base_y / 2.0, -base_z / 2.0],
        [0.0, base_y / 2.0, base_z / 2.0],
        [0.0, -base_y / 2.0, base_z / 2.0],
        [length_m, -tip_y / 2.0, -tip_z / 2.0],
        [length_m, tip_y / 2.0, -tip_z / 2.0],
        [length_m, tip_y / 2.0, tip_z / 2.0],
        [length_m, -tip_y / 2.0, tip_z / 2.0],
    ];
    let mut triangles = Vec::<[u32; 3]>::new();
    for quad in [
        [0, 1, 2, 3],
        [4, 7, 6, 5],
        [0, 4, 5, 1],
        [1, 5, 6, 2],
        [2, 6, 7, 3],
        [3, 7, 4, 0],
    ] {
        push_quad(&mut triangles, quad);
    }
    (vertices, triangles)
}

fn closed_surface_area_m2(vertices: &[[f64; 3]], triangles: &[[u32; 3]]) -> f64 {
    triangles
        .iter()
        .map(|triangle| triangle_area_m2(vertices, *triangle))
        .sum()
}

fn closed_surface_volume_m3(vertices: &[[f64; 3]], triangles: &[[u32; 3]]) -> f64 {
    triangles
        .iter()
        .map(|triangle| {
            let a = vertices[triangle[0] as usize];
            let b = vertices[triangle[1] as usize];
            let c = vertices[triangle[2] as usize];
            dot3(a, cross3(b, c)) / 6.0
        })
        .sum::<f64>()
        .abs()
}

fn triangle_area_m2(vertices: &[[f64; 3]], triangle: [u32; 3]) -> f64 {
    let a = vertices[triangle[0] as usize];
    let b = vertices[triangle[1] as usize];
    let c = vertices[triangle[2] as usize];
    let ab = sub3(b, a);
    let ac = sub3(c, a);
    0.5 * norm3(cross3(ab, ac))
}

fn sub3(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

fn cross3(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

fn dot3(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn norm3(a: [f64; 3]) -> f64 {
    dot3(a, a).sqrt()
}

fn through_hole_block_surface(
    outer: [f64; 3],
    hole_min: [f64; 2],
    hole_max: [f64; 2],
) -> (Vec<[f64; 3]>, Vec<[u32; 3]>) {
    let [sx, sy, sz] = outer;
    let [hx0, hy0] = hole_min;
    let [hx1, hy1] = hole_max;
    let vertices = vec![
        [0.0, 0.0, 0.0],
        [sx, 0.0, 0.0],
        [sx, sy, 0.0],
        [0.0, sy, 0.0],
        [hx0, hy0, 0.0],
        [hx1, hy0, 0.0],
        [hx1, hy1, 0.0],
        [hx0, hy1, 0.0],
        [0.0, 0.0, sz],
        [sx, 0.0, sz],
        [sx, sy, sz],
        [0.0, sy, sz],
        [hx0, hy0, sz],
        [hx1, hy0, sz],
        [hx1, hy1, sz],
        [hx0, hy1, sz],
    ];
    let mut triangles = Vec::<[u32; 3]>::new();
    for quad in [
        [0, 1, 5, 4],
        [1, 2, 6, 5],
        [2, 3, 7, 6],
        [3, 0, 4, 7],
        [8, 12, 13, 9],
        [9, 13, 14, 10],
        [10, 14, 15, 11],
        [11, 15, 12, 8],
        [0, 8, 9, 1],
        [1, 9, 10, 2],
        [2, 10, 11, 3],
        [3, 11, 8, 0],
        [4, 5, 13, 12],
        [5, 6, 14, 13],
        [6, 7, 15, 14],
        [7, 4, 12, 15],
    ] {
        push_quad(&mut triangles, quad);
    }
    (vertices, triangles)
}

#[cfg(test)]
fn annular_bore_block_surface(
    dimensions_m: [f64; 3],
    bore_radius_m: f64,
    segment_count: usize,
) -> (Vec<[f64; 3]>, Vec<[u32; 3]>) {
    let [sx, sy, sz] = dimensions_m;
    let center = [sx * 0.5, sy * 0.5];
    let mut vertices = Vec::<[f64; 3]>::with_capacity(segment_count * 4);
    for z in [0.0, sz] {
        for radius in [None, Some(bore_radius_m)] {
            for index in 0..segment_count {
                let theta = std::f64::consts::TAU * index as f64 / segment_count as f64;
                let direction = [theta.cos(), theta.sin()];
                let radius = radius.unwrap_or_else(|| {
                    let x_limit = if direction[0].abs() > f64::EPSILON {
                        sx * 0.5 / direction[0].abs()
                    } else {
                        f64::INFINITY
                    };
                    let y_limit = if direction[1].abs() > f64::EPSILON {
                        sy * 0.5 / direction[1].abs()
                    } else {
                        f64::INFINITY
                    };
                    x_limit.min(y_limit)
                });
                vertices.push([
                    center[0] + direction[0] * radius,
                    center[1] + direction[1] * radius,
                    z,
                ]);
            }
        }
    }

    let bottom_outer = 0_u32;
    let bottom_inner = segment_count as u32;
    let top_outer = (segment_count * 2) as u32;
    let top_inner = (segment_count * 3) as u32;
    let mut triangles = Vec::<[u32; 3]>::with_capacity(segment_count * 8);
    for index in 0..segment_count as u32 {
        let next = (index + 1) % segment_count as u32;
        push_quad(
            &mut triangles,
            [
                bottom_outer + index,
                bottom_outer + next,
                top_outer + next,
                top_outer + index,
            ],
        );
        push_quad(
            &mut triangles,
            [
                bottom_inner + next,
                bottom_inner + index,
                top_inner + index,
                top_inner + next,
            ],
        );
        push_quad(
            &mut triangles,
            [
                top_outer + next,
                top_inner + next,
                top_inner + index,
                top_outer + index,
            ],
        );
        push_quad(
            &mut triangles,
            [
                bottom_outer + index,
                bottom_inner + index,
                bottom_inner + next,
                bottom_outer + next,
            ],
        );
    }
    (vertices, triangles)
}

fn faceted_cylinder_surface(
    segment_count: usize,
    radius_m: f64,
    height_m: f64,
) -> (Vec<[f64; 3]>, Vec<[u32; 3]>) {
    let mut vertices = Vec::<[f64; 3]>::with_capacity(segment_count * 2 + 2);
    for z in [0.0, height_m] {
        for index in 0..segment_count {
            let theta = std::f64::consts::TAU * index as f64 / segment_count as f64;
            vertices.push([radius_m * theta.cos(), radius_m * theta.sin(), z]);
        }
    }
    let bottom_center = vertices.len() as u32;
    vertices.push([0.0, 0.0, 0.0]);
    let top_center = vertices.len() as u32;
    vertices.push([0.0, 0.0, height_m]);

    let mut triangles = Vec::<[u32; 3]>::with_capacity(segment_count * 4);
    let top_offset = segment_count as u32;
    for index in 0..segment_count as u32 {
        let next = (index + 1) % segment_count as u32;
        push_quad(
            &mut triangles,
            [index, next, top_offset + next, top_offset + index],
        );
        triangles.push([bottom_center, next, index]);
        triangles.push([top_center, top_offset + index, top_offset + next]);
    }
    (vertices, triangles)
}

fn push_quad(triangles: &mut Vec<[u32; 3]>, quad: [u32; 4]) {
    triangles.push([quad[0], quad[1], quad[2]]);
    triangles.push([quad[0], quad[2], quad[3]]);
}

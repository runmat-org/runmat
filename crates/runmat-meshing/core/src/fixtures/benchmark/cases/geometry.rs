use runmat_geometry_core::{
    EntityIdRange, EntityKind, GeometryAsset, GeometrySource, MeshDescriptor, MeshKind, Region,
    RegionEntityMapping, SourceGeometry, SourceGeometryKind, SurfaceMesh, TessellationProfile,
    UnitSystem,
};

pub(super) fn box_geometry(
    benchmark_id: &str,
    dimensions_m: [f64; 3],
    origin_m: [f64; 3],
) -> GeometryAsset {
    let (vertices, triangles) = box_surface(dimensions_m, origin_m, 0);
    geometry_from_surface(
        benchmark_id,
        &format!("generic_{benchmark_id}_surface"),
        vertices,
        triangles,
    )
}

pub(super) fn geometry_from_surface(
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

pub(super) fn box_surface(
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

pub(super) fn tapered_rectangular_prism_surface(
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

pub(super) fn closed_surface_area_m2(vertices: &[[f64; 3]], triangles: &[[u32; 3]]) -> f64 {
    triangles
        .iter()
        .map(|triangle| triangle_area_m2(vertices, *triangle))
        .sum()
}

pub(super) fn closed_surface_volume_m3(vertices: &[[f64; 3]], triangles: &[[u32; 3]]) -> f64 {
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

pub(super) fn through_hole_block_surface(
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
pub(super) fn annular_bore_block_surface(
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

pub(super) fn faceted_cylinder_surface(
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

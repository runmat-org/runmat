use std::collections::{BTreeMap, BTreeSet};

use runmat_meshing_core::{
    predicate::{tetrahedron_circumsphere_contains_point, tetrahedron_signed_volume},
    tolerance::MeshingTolerance,
};

use super::{ConnectivityPoint, ConnectivityTetrahedron};

pub(super) fn tetrahedralize_points(
    input_points: &[ConnectivityPoint],
) -> Vec<ConnectivityTetrahedron> {
    if input_points.len() < 4 {
        return Vec::new();
    }
    let mut points = input_points.to_vec();
    let super_start = points.len();
    points.extend(super_tetrahedron_points(input_points));
    let mut tetrahedra = vec![ConnectivityTetrahedron {
        vertices: [
            super_start,
            super_start + 1,
            super_start + 2,
            super_start + 3,
        ],
    }];

    for point_index in 0..input_points.len() {
        let point = points[point_index].coordinates_m;
        let mut bad_indices = Vec::<usize>::new();
        for (tetrahedron_index, tetrahedron) in tetrahedra.iter().enumerate() {
            if tetrahedron_circumsphere_contains_point(
                tetrahedron
                    .vertices
                    .map(|index| points[index].coordinates_m),
                point,
                MeshingTolerance::default(),
            ) {
                bad_indices.push(tetrahedron_index);
            }
        }
        if bad_indices.is_empty() {
            continue;
        }

        let bad_set = bad_indices.iter().copied().collect::<BTreeSet<_>>();
        let mut face_counts = BTreeMap::<[usize; 3], usize>::new();
        for tetrahedron_index in &bad_indices {
            for face in connectivity_tetrahedron_faces(tetrahedra[*tetrahedron_index].vertices) {
                *face_counts
                    .entry(connectivity_sorted_face(face))
                    .or_default() += 1;
            }
        }
        let cavity_faces = face_counts
            .into_iter()
            .filter_map(|(face, count)| (count == 1).then_some(face))
            .collect::<Vec<_>>();

        tetrahedra = tetrahedra
            .into_iter()
            .enumerate()
            .filter_map(|(tetrahedron_index, tetrahedron)| {
                (!bad_set.contains(&tetrahedron_index)).then_some(tetrahedron)
            })
            .collect();
        for face in cavity_faces {
            let vertices = [face[0], face[1], face[2], point_index];
            let points_for_tetrahedron = vertices.map(|index| points[index].coordinates_m);
            if tetrahedron_signed_volume(points_for_tetrahedron).abs()
                > MeshingTolerance::default().volume_epsilon(1.0)
            {
                tetrahedra.push(ConnectivityTetrahedron { vertices });
            }
        }
    }

    tetrahedra
        .into_iter()
        .filter(|tetrahedron| {
            !tetrahedron
                .vertices
                .iter()
                .any(|index| points[*index].is_super)
        })
        .collect()
}

fn super_tetrahedron_points(points: &[ConnectivityPoint]) -> [ConnectivityPoint; 4] {
    let mut min = points[0].coordinates_m;
    let mut max = points[0].coordinates_m;
    for point in points {
        for axis in 0..3 {
            min[axis] = min[axis].min(point.coordinates_m[axis]);
            max[axis] = max[axis].max(point.coordinates_m[axis]);
        }
    }
    let center = [
        (min[0] + max[0]) * 0.5,
        (min[1] + max[1]) * 0.5,
        (min[2] + max[2]) * 0.5,
    ];
    let span = (0..3)
        .map(|axis| max[axis] - min[axis])
        .fold(0.0_f64, f64::max)
        .max(1.0);
    let radius = span * 16.0;
    [
        ConnectivityPoint {
            node_id: u32::MAX - 3,
            coordinates_m: [center[0] + radius, center[1], center[2] - radius],
            is_super: true,
        },
        ConnectivityPoint {
            node_id: u32::MAX - 2,
            coordinates_m: [center[0] - radius, center[1] + radius, center[2] - radius],
            is_super: true,
        },
        ConnectivityPoint {
            node_id: u32::MAX - 1,
            coordinates_m: [center[0] - radius, center[1] - radius, center[2] - radius],
            is_super: true,
        },
        ConnectivityPoint {
            node_id: u32::MAX,
            coordinates_m: [center[0], center[1], center[2] + radius],
            is_super: true,
        },
    ]
}

fn connectivity_tetrahedron_faces(vertices: [usize; 4]) -> [[usize; 3]; 4] {
    [
        [vertices[0], vertices[1], vertices[2]],
        [vertices[0], vertices[1], vertices[3]],
        [vertices[0], vertices[2], vertices[3]],
        [vertices[1], vertices[2], vertices[3]],
    ]
}

fn connectivity_sorted_face(mut face: [usize; 3]) -> [usize; 3] {
    face.sort();
    face
}

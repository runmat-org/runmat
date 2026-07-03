use std::collections::{BTreeMap, BTreeSet};

use crate::{
    predicate::{tet_circumsphere_contains_point, tet_signed_volume},
    tolerance::MeshingTolerance,
};

use super::{ConnectivityPoint, ConnectivityTet};

pub(super) fn tetrahedralize_points(input_points: &[ConnectivityPoint]) -> Vec<ConnectivityTet> {
    if input_points.len() < 4 {
        return Vec::new();
    }
    let mut points = input_points.to_vec();
    let super_start = points.len();
    points.extend(super_tetrahedron_points(input_points));
    let mut tets = vec![ConnectivityTet {
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
        for (tet_index, tet) in tets.iter().enumerate() {
            if tet_circumsphere_contains_point(
                tet.vertices.map(|index| points[index].coordinates_m),
                point,
                MeshingTolerance::default(),
            ) {
                bad_indices.push(tet_index);
            }
        }
        if bad_indices.is_empty() {
            continue;
        }

        let bad_set = bad_indices.iter().copied().collect::<BTreeSet<_>>();
        let mut face_counts = BTreeMap::<[usize; 3], usize>::new();
        for tet_index in &bad_indices {
            for face in connectivity_tet_faces(tets[*tet_index].vertices) {
                *face_counts
                    .entry(connectivity_sorted_face(face))
                    .or_default() += 1;
            }
        }
        let cavity_faces = face_counts
            .into_iter()
            .filter_map(|(face, count)| (count == 1).then_some(face))
            .collect::<Vec<_>>();

        tets = tets
            .into_iter()
            .enumerate()
            .filter_map(|(tet_index, tet)| (!bad_set.contains(&tet_index)).then_some(tet))
            .collect();
        for face in cavity_faces {
            let vertices = [face[0], face[1], face[2], point_index];
            let points_for_tet = vertices.map(|index| points[index].coordinates_m);
            if tet_signed_volume(points_for_tet).abs()
                > MeshingTolerance::default().volume_epsilon(1.0)
            {
                tets.push(ConnectivityTet { vertices });
            }
        }
    }

    tets.into_iter()
        .filter(|tet| !tet.vertices.iter().any(|index| points[*index].is_super))
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

fn connectivity_tet_faces(vertices: [usize; 4]) -> [[usize; 3]; 4] {
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

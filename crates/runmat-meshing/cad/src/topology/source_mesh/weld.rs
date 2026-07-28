use std::collections::BTreeMap;

use super::geometry::distance;

pub(super) fn weld_surface_vertices(
    vertices: &[[f64; 3]],
    bounds_min_m: [f64; 3],
    bounds_max_m: [f64; 3],
) -> (Vec<[f64; 3]>, Vec<u32>) {
    let tolerance = weld_tolerance_m(bounds_min_m, bounds_max_m);
    let mut buckets = BTreeMap::<[i64; 3], Vec<u32>>::new();
    let mut welded_vertices = Vec::<[f64; 3]>::new();
    let mut vertex_map = Vec::<u32>::with_capacity(vertices.len());

    for vertex in vertices {
        let key = weld_key(*vertex, tolerance);
        let mut welded_id = None;
        for neighbor_key in neighboring_weld_keys(key) {
            let Some(candidates) = buckets.get(&neighbor_key) else {
                continue;
            };
            for candidate_id in candidates {
                let candidate = welded_vertices[*candidate_id as usize];
                if distance(candidate, *vertex) <= tolerance {
                    welded_id = Some(*candidate_id);
                    break;
                }
            }
            if welded_id.is_some() {
                break;
            }
        }

        let welded_id = match welded_id {
            Some(welded_id) => welded_id,
            None => {
                let welded_id = welded_vertices.len() as u32;
                welded_vertices.push(*vertex);
                buckets.entry(key).or_default().push(welded_id);
                welded_id
            }
        };
        vertex_map.push(welded_id);
    }

    (welded_vertices, vertex_map)
}

fn weld_tolerance_m(bounds_min_m: [f64; 3], bounds_max_m: [f64; 3]) -> f64 {
    let span = (0..3)
        .map(|axis| bounds_max_m[axis] - bounds_min_m[axis])
        .fold(0.0_f64, f64::max);
    (span * 1.0e-8).max(1.0e-9)
}

fn weld_key(vertex: [f64; 3], tolerance: f64) -> [i64; 3] {
    [
        (vertex[0] / tolerance).round() as i64,
        (vertex[1] / tolerance).round() as i64,
        (vertex[2] / tolerance).round() as i64,
    ]
}

fn neighboring_weld_keys(key: [i64; 3]) -> impl Iterator<Item = [i64; 3]> {
    (-1..=1).flat_map(move |dx| {
        (-1..=1).flat_map(move |dy| (-1..=1).map(move |dz| [key[0] + dx, key[1] + dy, key[2] + dz]))
    })
}

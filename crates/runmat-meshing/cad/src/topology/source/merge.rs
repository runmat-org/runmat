use std::collections::BTreeMap;

use super::CadFace;

pub(super) fn merge_stable_cad_faces(face_seeds: Vec<CadFace>) -> Vec<CadFace> {
    let mut merged = BTreeMap::<String, (CadFace, BTreeMap<u32, usize>)>::new();
    for face in face_seeds {
        let key = face.entity_id.id.clone();
        let source_edge_counts = face
            .source_edge_ids
            .iter()
            .map(|edge_id| (*edge_id, 1_usize))
            .collect::<BTreeMap<_, _>>();
        match merged.get_mut(&key) {
            Some((existing, edge_counts)) => {
                existing
                    .source_face_ids
                    .extend(face.source_face_ids.iter().copied());
                existing
                    .source_edge_ids
                    .extend(face.source_edge_ids.iter().copied());
                existing.region_ids.extend(face.region_ids.iter().cloned());
                existing
                    .material_region_ids
                    .extend(face.material_region_ids.iter().cloned());
                existing.unit_normal = area_weighted_normal(
                    existing.unit_normal,
                    existing.area_m2,
                    face.unit_normal,
                    face.area_m2,
                );
                existing.area_m2 += face.area_m2;
                for (source_edge_id, count) in source_edge_counts {
                    *edge_counts.entry(source_edge_id).or_default() += count;
                }
            }
            None => {
                merged.insert(key, (face, source_edge_counts));
            }
        }
    }
    merged
        .into_values()
        .map(|(mut face, edge_counts)| {
            face.source_face_ids.sort_unstable();
            face.source_face_ids.dedup();
            face.source_edge_ids.sort_unstable();
            face.source_edge_ids.dedup();
            face.region_ids.sort();
            face.region_ids.dedup();
            face.material_region_ids.sort();
            face.material_region_ids.dedup();
            face.loop_edge_ids = edge_counts
                .into_iter()
                .filter_map(|(edge_id, count)| (count == 1).then(|| format!("cad_edge_{edge_id}")))
                .collect();
            face
        })
        .collect()
}

fn area_weighted_normal(
    left_normal: [f64; 3],
    left_area_m2: f64,
    right_normal: [f64; 3],
    right_area_m2: f64,
) -> [f64; 3] {
    let combined = [
        left_normal[0] * left_area_m2 + right_normal[0] * right_area_m2,
        left_normal[1] * left_area_m2 + right_normal[1] * right_area_m2,
        left_normal[2] * left_area_m2 + right_normal[2] * right_area_m2,
    ];
    let length =
        (combined[0] * combined[0] + combined[1] * combined[1] + combined[2] * combined[2]).sqrt();
    if !length.is_finite() || length <= f64::EPSILON {
        left_normal
    } else {
        [
            combined[0] / length,
            combined[1] / length,
            combined[2] / length,
        ]
    }
}

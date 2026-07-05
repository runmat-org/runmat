use std::collections::BTreeSet;

use super::builder::PartitionBuilder;

#[derive(Debug, Clone)]
pub(super) struct PartitionCell {
    pub(super) node_ids: Vec<u32>,
    lower_bounds: [f64; 4],
    upper_bounds: [f64; 4],
}

pub(super) fn partition_cells(
    divisions: usize,
    inner_lower_bounds: [f64; 4],
    builder: &mut PartitionBuilder,
) -> Vec<PartitionCell> {
    let mut cells = Vec::<PartitionCell>::new();
    for first in 0..divisions {
        for second in 0..divisions {
            for third in 0..divisions {
                for fourth in 0..divisions {
                    let lower_bounds = [
                        first as f64 / divisions as f64,
                        second as f64 / divisions as f64,
                        third as f64 / divisions as f64,
                        fourth as f64 / divisions as f64,
                    ];
                    let upper_bounds = [
                        (first + 1) as f64 / divisions as f64,
                        (second + 1) as f64 / divisions as f64,
                        (third + 1) as f64 / divisions as f64,
                        (fourth + 1) as f64 / divisions as f64,
                    ];
                    if lower_bounds.iter().sum::<f64>() > 1.0 + 1.0e-12
                        || upper_bounds.iter().sum::<f64>() < 1.0 - 1.0e-12
                        || lower_bounds
                            .iter()
                            .enumerate()
                            .all(|(index, value)| *value >= inner_lower_bounds[index] - 1.0e-12)
                    {
                        continue;
                    }
                    let mut node_ids = BTreeSet::<u32>::new();
                    for active in three_active_coordinates() {
                        for mask in 0..8 {
                            let mut barycentric = [f64::NAN; 4];
                            for (bit_index, coordinate_index) in active.iter().enumerate() {
                                barycentric[*coordinate_index] = if (mask & (1 << bit_index)) == 0 {
                                    lower_bounds[*coordinate_index]
                                } else {
                                    upper_bounds[*coordinate_index]
                                };
                            }
                            let free_index = (0..4)
                                .find(|index| barycentric[*index].is_nan())
                                .expect("one free barycentric coordinate");
                            barycentric[free_index] = 1.0
                                - barycentric
                                    .iter()
                                    .filter(|value| !value.is_nan())
                                    .sum::<f64>();
                            if barycentric.iter().enumerate().all(|(index, value)| {
                                *value >= lower_bounds[index] - 1.0e-12
                                    && *value <= upper_bounds[index] + 1.0e-12
                            }) {
                                node_ids.insert(builder.insert_node(barycentric));
                            }
                        }
                    }
                    if node_ids.len() >= 4 {
                        cells.push(PartitionCell {
                            node_ids: node_ids.into_iter().collect(),
                            lower_bounds,
                            upper_bounds,
                        });
                    }
                }
            }
        }
    }
    cells
}

pub(super) fn cell_centroid(cell: &PartitionCell, builder: &PartitionBuilder) -> [f64; 4] {
    let mut barycentric = [0.0; 4];
    for node_id in &cell.node_ids {
        let node_barycentric = builder.barycentric_by_id[node_id];
        for index in 0..4 {
            barycentric[index] += node_barycentric[index];
        }
    }
    for value in &mut barycentric {
        *value /= cell.node_ids.len() as f64;
    }
    barycentric
}

pub(super) fn cell_faces(cell: &PartitionCell, builder: &PartitionBuilder) -> Vec<Vec<u32>> {
    let mut faces = Vec::<Vec<u32>>::new();
    for coordinate_index in 0..4 {
        for value in [
            cell.lower_bounds[coordinate_index],
            cell.upper_bounds[coordinate_index],
        ] {
            let face = cell
                .node_ids
                .iter()
                .copied()
                .filter(|node_id| {
                    let barycentric = builder.barycentric_by_id[node_id];
                    (barycentric[coordinate_index] - value).abs() <= 1.0e-12
                })
                .collect::<Vec<_>>();
            if face.len() >= 3 {
                faces.push(face);
            }
        }
    }
    faces
}

fn three_active_coordinates() -> [[usize; 3]; 4] {
    [[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]]
}

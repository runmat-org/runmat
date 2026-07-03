use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct SolidElementQuality {
    pub volume_m3: f64,
    pub min_edge_length_m: f64,
    pub max_edge_length_m: f64,
    pub aspect_ratio: f64,
}

impl SolidElementQuality {
    pub fn from_tetrahedron4_nodes(nodes_m: [[f64; 3]; 4], volume_m3: f64) -> Self {
        let mut min_edge_length_m = f64::INFINITY;
        let mut max_edge_length_m = 0.0_f64;
        for (left, right) in [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)] {
            let length = distance(nodes_m[left], nodes_m[right]);
            min_edge_length_m = min_edge_length_m.min(length);
            max_edge_length_m = max_edge_length_m.max(length);
        }
        let aspect_ratio = max_edge_length_m / min_edge_length_m.max(f64::EPSILON);
        Self {
            volume_m3,
            min_edge_length_m,
            max_edge_length_m,
            aspect_ratio,
        }
    }
}

fn distance(left: [f64; 3], right: [f64; 3]) -> f64 {
    ((right[0] - left[0]).powi(2) + (right[1] - left[1]).powi(2) + (right[2] - left[2]).powi(2))
        .sqrt()
}

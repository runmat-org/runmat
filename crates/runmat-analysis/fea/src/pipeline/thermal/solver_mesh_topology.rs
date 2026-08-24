use std::collections::{BTreeMap, BTreeSet};

use runmat_meshing_core::SolverMeshArtifact;

#[derive(Debug, Clone)]
pub(super) struct ThermalMeshTopology {
    coordinates_m: Vec<[f64; 3]>,
    elements: Vec<Vec<usize>>,
    neighbors: Vec<Vec<usize>>,
    axis_bounds_m: [[f64; 2]; 3],
    coordinate_counts: [usize; 3],
    active_dimension_count: usize,
    characteristic_length_m: f64,
    boundary_face_count: usize,
}

impl ThermalMeshTopology {
    pub(super) fn derive(mesh: &SolverMeshArtifact) -> Option<Self> {
        let node_indices = mesh
            .topology
            .nodes
            .iter()
            .enumerate()
            .map(|(index, node)| (node.node_id, index))
            .collect::<BTreeMap<_, _>>();
        let coordinates_m = mesh
            .topology
            .nodes
            .iter()
            .map(|node| node.coordinates_m)
            .collect::<Vec<_>>();
        let elements = mesh
            .topology
            .volume_elements
            .iter()
            .map(|element| {
                element
                    .node_ids
                    .iter()
                    .map(|node_id| node_indices.get(node_id).copied())
                    .collect::<Option<Vec<_>>>()
            })
            .collect::<Option<Vec<_>>>()?;
        if coordinates_m.is_empty() || elements.is_empty() {
            return None;
        }

        let mut neighbor_sets = vec![BTreeSet::new(); coordinates_m.len()];
        for element in &elements {
            for left in 0..element.len() {
                for right in (left + 1)..element.len() {
                    neighbor_sets[element[left]].insert(element[right]);
                    neighbor_sets[element[right]].insert(element[left]);
                }
            }
        }
        let neighbors = neighbor_sets
            .into_iter()
            .map(|neighbors| neighbors.into_iter().collect::<Vec<_>>())
            .collect::<Vec<_>>();
        let axis_bounds_m = std::array::from_fn(|axis| {
            let min = coordinates_m
                .iter()
                .map(|point| point[axis])
                .fold(f64::INFINITY, f64::min);
            let max = coordinates_m
                .iter()
                .map(|point| point[axis])
                .fold(f64::NEG_INFINITY, f64::max);
            [min, max]
        });
        let coordinate_counts = std::array::from_fn(|axis| {
            let mut values = coordinates_m
                .iter()
                .map(|point| point[axis])
                .collect::<Vec<_>>();
            values.sort_by(f64::total_cmp);
            values.dedup_by(|left, right| left.to_bits() == right.to_bits());
            values.len()
        });
        let active_dimension_count = axis_bounds_m
            .iter()
            .filter(|bounds| bounds[1] - bounds[0] > 1.0e-12)
            .count();
        let characteristic_length_m = axis_bounds_m
            .iter()
            .map(|bounds| (bounds[1] - bounds[0]).powi(2))
            .sum::<f64>()
            .sqrt();

        Some(Self {
            coordinates_m,
            elements,
            neighbors,
            axis_bounds_m,
            coordinate_counts,
            active_dimension_count,
            characteristic_length_m,
            boundary_face_count: mesh.topology.boundary_faces.len(),
        })
    }

    pub(super) fn coordinate_counts(&self) -> [usize; 3] {
        self.coordinate_counts
    }

    pub(super) fn active_dimension_count(&self) -> usize {
        self.active_dimension_count
    }

    pub(super) fn characteristic_length_m(&self) -> f64 {
        self.characteristic_length_m
    }

    pub(super) fn boundary_face_count(&self) -> usize {
        self.boundary_face_count
    }

    pub(super) fn elements(&self) -> Vec<Vec<usize>> {
        self.elements.clone()
    }

    pub(super) fn normalized_coordinate(&self, node_index: usize, axis: usize) -> f64 {
        let bounds = self.axis_bounds_m[axis];
        let span = bounds[1] - bounds[0];
        if span <= 1.0e-12 {
            0.0
        } else {
            (self.coordinates_m[node_index][axis] - bounds[0]) / span
        }
    }

    pub(super) fn edges(&self) -> Vec<[usize; 2]> {
        let mut edges = BTreeSet::new();
        for (node, neighbors) in self.neighbors.iter().enumerate() {
            for neighbor in neighbors {
                edges.insert([node.min(*neighbor), node.max(*neighbor)]);
            }
        }
        edges.into_iter().collect()
    }

    pub(super) fn gradient(&self, values: &[f64], node_index: usize) -> [f64; 3] {
        let origin = self.coordinates_m[node_index];
        let origin_value = values.get(node_index).copied().unwrap_or(0.0);
        let mut normal = [[0.0_f64; 3]; 3];
        let mut rhs = [0.0_f64; 3];
        for neighbor in &self.neighbors[node_index] {
            let delta = std::array::from_fn::<_, 3, _>(|axis| {
                self.coordinates_m[*neighbor][axis] - origin[axis]
            });
            let value_delta = values.get(*neighbor).copied().unwrap_or(origin_value) - origin_value;
            for row in 0..3 {
                rhs[row] += delta[row] * value_delta;
                for column in 0..3 {
                    normal[row][column] += delta[row] * delta[column];
                }
            }
        }
        solve_three_by_three(normal, rhs).unwrap_or([0.0; 3])
    }

    pub(super) fn graph_laplacian(&self, values: &[f64], node_index: usize) -> f64 {
        let center = values.get(node_index).copied().unwrap_or(0.0);
        let neighbors = &self.neighbors[node_index];
        if neighbors.is_empty() {
            return 0.0;
        }
        neighbors
            .iter()
            .map(|neighbor| values.get(*neighbor).copied().unwrap_or(center) - center)
            .sum::<f64>()
            / neighbors.len() as f64
    }

    pub(super) fn boundary_axis_flux(&self, heat_flux: &[f64], axis: usize) -> [f64; 2] {
        let bounds = self.axis_bounds_m[axis];
        let tolerance = ((bounds[1] - bounds[0]).abs() * 1.0e-9).max(1.0e-12);
        let mut sums = [0.0_f64; 2];
        let mut counts = [0usize; 2];
        for (node_index, point) in self.coordinates_m.iter().enumerate() {
            for side in 0..2 {
                if (point[axis] - bounds[side]).abs() <= tolerance {
                    sums[side] += heat_flux.get(node_index * 3 + axis).copied().unwrap_or(0.0);
                    counts[side] += 1;
                }
            }
        }
        std::array::from_fn(|side| {
            if counts[side] == 0 {
                0.0
            } else {
                sums[side] / counts[side] as f64
            }
        })
    }
}

fn solve_three_by_three(mut matrix: [[f64; 3]; 3], mut rhs: [f64; 3]) -> Option<[f64; 3]> {
    let scale = matrix
        .iter()
        .flatten()
        .map(|value| value.abs())
        .fold(0.0_f64, f64::max)
        .max(1.0);
    for column in 0..3 {
        let pivot = (column..3).max_by(|left, right| {
            matrix[*left][column]
                .abs()
                .total_cmp(&matrix[*right][column].abs())
        })?;
        if matrix[pivot][column].abs() <= scale * 1.0e-12 {
            return None;
        }
        matrix.swap(column, pivot);
        rhs.swap(column, pivot);
        let divisor = matrix[column][column];
        for entry in column..3 {
            matrix[column][entry] /= divisor;
        }
        rhs[column] /= divisor;
        for row in 0..3 {
            if row == column {
                continue;
            }
            let factor = matrix[row][column];
            for entry in column..3 {
                matrix[row][entry] -= factor * matrix[column][entry];
            }
            rhs[row] -= factor * rhs[column];
        }
    }
    rhs.iter().all(|value| value.is_finite()).then_some(rhs)
}

#[cfg(test)]
mod tests {
    use runmat_meshing_core::{fixtures, ElementOrder};

    use super::ThermalMeshTopology;

    #[test]
    fn canonical_tetrahedron_recovers_linear_temperature_gradient() {
        let mesh = fixtures::canonical_tetrahedron_solver_mesh(ElementOrder::Tet4);
        let topology = ThermalMeshTopology::derive(&mesh).expect("thermal topology");
        let temperatures = mesh
            .topology
            .nodes
            .iter()
            .map(|node| {
                2.0 * node.coordinates_m[0] + 3.0 * node.coordinates_m[1] - node.coordinates_m[2]
            })
            .collect::<Vec<_>>();

        for node_index in 0..temperatures.len() {
            let gradient = topology.gradient(&temperatures, node_index);
            assert!((gradient[0] - 2.0).abs() <= 1.0e-12);
            assert!((gradient[1] - 3.0).abs() <= 1.0e-12);
            assert!((gradient[2] + 1.0).abs() <= 1.0e-12);
        }
        assert_eq!(topology.active_dimension_count(), 3);
        assert_eq!(topology.elements().len(), 1);
    }
}

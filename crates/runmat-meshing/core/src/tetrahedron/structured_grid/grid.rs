use super::*;

#[derive(Debug, Clone, PartialEq)]
pub(super) struct StructuredGrid {
    pub(super) x: Vec<f64>,
    pub(super) y: Vec<f64>,
    pub(super) z: Vec<f64>,
}

impl StructuredGrid {
    pub(super) fn uniform(input: &BoundaryMeshInput, divisions: usize) -> Self {
        Self {
            x: uniform_axis(input.bounds_min_m[0], input.bounds_max_m[0], divisions),
            y: uniform_axis(input.bounds_min_m[1], input.bounds_max_m[1], divisions),
            z: uniform_axis(input.bounds_min_m[2], input.bounds_max_m[2], divisions),
        }
    }

    pub(super) fn nx(&self) -> usize {
        self.x.len().saturating_sub(1)
    }

    pub(super) fn ny(&self) -> usize {
        self.y.len().saturating_sub(1)
    }

    pub(super) fn nz(&self) -> usize {
        self.z.len().saturating_sub(1)
    }

    pub(super) fn element_count(&self) -> usize {
        6 * self.nx() * self.ny() * self.nz()
    }

    pub(super) fn cell_count(&self) -> usize {
        self.nx() * self.ny() * self.nz()
    }

    pub(super) fn cell_index(&self, i: usize, j: usize, k: usize) -> usize {
        i + self.nx() * (j + self.ny() * k)
    }

    pub(super) fn cell_coordinates(&self, index: usize) -> (usize, usize, usize) {
        let nx = self.nx().max(1);
        let ny = self.ny().max(1);
        let k = index / (nx * ny);
        let rem = index % (nx * ny);
        let j = rem / nx;
        let i = rem % nx;
        (i, j, k)
    }

    pub(super) fn cell_neighbors(&self, i: usize, j: usize, k: usize) -> Vec<usize> {
        let mut neighbors = Vec::with_capacity(6);
        if i > 0 {
            neighbors.push(self.cell_index(i - 1, j, k));
        }
        if i + 1 < self.nx() {
            neighbors.push(self.cell_index(i + 1, j, k));
        }
        if j > 0 {
            neighbors.push(self.cell_index(i, j - 1, k));
        }
        if j + 1 < self.ny() {
            neighbors.push(self.cell_index(i, j + 1, k));
        }
        if k > 0 {
            neighbors.push(self.cell_index(i, j, k - 1));
        }
        if k + 1 < self.nz() {
            neighbors.push(self.cell_index(i, j, k + 1));
        }
        neighbors
    }

    pub(super) fn min_cell_size(&self) -> Option<f64> {
        [
            min_axis_spacing(&self.x),
            min_axis_spacing(&self.y),
            min_axis_spacing(&self.z),
        ]
        .into_iter()
        .flatten()
        .reduce(f64::min)
    }

    pub(super) fn max_cell_aspect_ratio(&self) -> Option<f64> {
        let mut max_ratio = 0.0_f64;
        let mut saw_cell = false;
        for dx in axis_spacings(&self.x) {
            for dy in axis_spacings(&self.y) {
                for dz in axis_spacings(&self.z) {
                    let min_edge = dx.min(dy).min(dz);
                    if !min_edge.is_finite() || min_edge <= 0.0 {
                        continue;
                    }
                    let diagonal = (dx * dx + dy * dy + dz * dz).sqrt();
                    max_ratio = max_ratio.max(diagonal / min_edge);
                    saw_cell = true;
                }
            }
        }
        saw_cell.then_some(max_ratio)
    }
}

fn uniform_axis(min: f64, max: f64, divisions: usize) -> Vec<f64> {
    (0..=divisions)
        .map(|index| lerp(min, max, index as f64 / divisions as f64))
        .collect()
}

fn min_axis_spacing(axis: &[f64]) -> Option<f64> {
    axis_spacings(axis)
        .filter(|value| value.is_finite() && *value > 0.0)
        .reduce(f64::min)
}

fn axis_spacings(axis: &[f64]) -> impl Iterator<Item = f64> + '_ {
    axis.windows(2).map(|pair| pair[1] - pair[0])
}

pub(super) fn grid_nodes(grid: &StructuredGrid) -> Vec<AnalysisMeshNode> {
    let mut nodes = Vec::with_capacity(grid.x.len() * grid.y.len() * grid.z.len());
    for z in &grid.z {
        for y in &grid.y {
            for x in &grid.x {
                nodes.push(AnalysisMeshNode {
                    node_id: nodes.len() as u32 + 1,
                    coordinates_m: [*x, *y, *z],
                    provenance: Vec::new(),
                });
            }
        }
    }
    nodes
}

pub(super) fn node_id_at(grid: &StructuredGrid, i: usize, j: usize, k: usize) -> u32 {
    (1 + i + grid.x.len() * (j + grid.y.len() * k)) as u32
}

//! Graph partitioning of the element dual graph.
//!
//! A recursive coordinate bisection of element centroids gives compact, exactly balanced
//! initial parts. Boundary refinement on the face-adjacency dual graph then moves elements
//! that reduce the cut (the number of element faces shared between parts, which bounds the
//! interface size every Krylov iteration communicates) while holding the balance tolerance.

use std::collections::HashMap;

use crate::error::{DistributedError, DistributedResult};
use crate::problem::PartitionMesh;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PartitionOptions {
    pub parts: usize,
    /// Greedy boundary refinement passes over the dual graph.
    pub refinement_passes: usize,
    /// Allowed relative excess of the largest part over the mean element count.
    pub imbalance_tolerance: f64,
}

impl PartitionOptions {
    pub fn new(parts: usize) -> Self {
        Self {
            parts,
            refinement_passes: 8,
            imbalance_tolerance: 0.03,
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct PartitionQuality {
    pub parts: usize,
    /// Element faces shared by two different parts.
    pub cut_faces: usize,
    /// Nodes that belong to elements of more than one part.
    pub interface_nodes: usize,
    pub smallest_part: usize,
    pub largest_part: usize,
    pub refinement_moves: usize,
}

/// Assign every element to one of `options.parts` parts.
pub fn partition_elements(
    mesh: &PartitionMesh,
    options: PartitionOptions,
) -> DistributedResult<(Vec<u32>, PartitionQuality)> {
    mesh.validate()?;
    let elements = mesh.element_count();
    if options.parts == 0 || options.parts > elements || options.parts > u32::MAX as usize {
        return Err(DistributedError::InvalidProblem(format!(
            "cannot split {elements} elements into {} parts",
            options.parts
        )));
    }
    let centroids = (0..elements)
        .map(|element| {
            let nodes = mesh.element(element);
            let mut centroid = [0.0; 3];
            for &node in &nodes[..4] {
                for (axis, coordinate) in centroid.iter_mut().enumerate() {
                    *coordinate += mesh.coordinates[node as usize][axis] * 0.25;
                }
            }
            centroid
        })
        .collect::<Vec<_>>();
    let mut parts = vec![0u32; elements];
    let mut order = (0..elements as u32).collect::<Vec<_>>();
    bisect(&centroids, &mut order, 0, options.parts, &mut parts);
    let adjacency = dual_graph(mesh);
    let moves = refine(&adjacency, &mut parts, options);
    let quality = quality(mesh, &adjacency, &parts, options.parts, moves);
    Ok((parts, quality))
}

fn bisect(
    centroids: &[[f64; 3]],
    elements: &mut [u32],
    first_part: usize,
    part_count: usize,
    parts: &mut [u32],
) {
    if part_count == 1 {
        for &element in elements.iter() {
            parts[element as usize] = first_part as u32;
        }
        return;
    }
    let mut lower = [f64::INFINITY; 3];
    let mut upper = [f64::NEG_INFINITY; 3];
    for &element in elements.iter() {
        for axis in 0..3 {
            let value = centroids[element as usize][axis];
            lower[axis] = lower[axis].min(value);
            upper[axis] = upper[axis].max(value);
        }
    }
    let axis = (0..3)
        .max_by(|&a, &b| {
            (upper[a] - lower[a])
                .partial_cmp(&(upper[b] - lower[b]))
                .unwrap_or(std::cmp::Ordering::Equal)
        })
        .unwrap_or(0);
    let left_parts = part_count / 2;
    let split = elements.len() * left_parts / part_count;
    elements.select_nth_unstable_by(split, |&a, &b| {
        centroids[a as usize][axis]
            .partial_cmp(&centroids[b as usize][axis])
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(a.cmp(&b))
    });
    let (left, right) = elements.split_at_mut(split);
    bisect(centroids, left, first_part, left_parts, parts);
    bisect(
        centroids,
        right,
        first_part + left_parts,
        part_count - left_parts,
        parts,
    );
}

/// Face adjacency of tetrahedral elements (CSR), from their corner nodes.
fn dual_graph(mesh: &PartitionMesh) -> (Vec<usize>, Vec<u32>) {
    let elements = mesh.element_count();
    let mut faces: HashMap<[u32; 3], u32> = HashMap::with_capacity(elements * 2);
    let mut edges: Vec<(u32, u32)> = Vec::with_capacity(elements * 2);
    const FACES: [[usize; 3]; 4] = [[1, 2, 3], [0, 2, 3], [0, 1, 3], [0, 1, 2]];
    for element in 0..elements {
        let nodes = mesh.element(element);
        for face in FACES {
            let mut key = [nodes[face[0]], nodes[face[1]], nodes[face[2]]];
            key.sort_unstable();
            match faces.remove(&key) {
                Some(other) => edges.push((other, element as u32)),
                None => {
                    faces.insert(key, element as u32);
                }
            }
        }
    }
    let mut degree = vec![0usize; elements + 1];
    for &(a, b) in &edges {
        degree[a as usize + 1] += 1;
        degree[b as usize + 1] += 1;
    }
    for index in 0..elements {
        degree[index + 1] += degree[index];
    }
    let mut cursor = degree.clone();
    let mut neighbors = vec![0u32; degree[elements]];
    for &(a, b) in &edges {
        neighbors[cursor[a as usize]] = b;
        cursor[a as usize] += 1;
        neighbors[cursor[b as usize]] = a;
        cursor[b as usize] += 1;
    }
    (degree, neighbors)
}

fn refine(
    (offsets, neighbors): &(Vec<usize>, Vec<u32>),
    parts: &mut [u32],
    options: PartitionOptions,
) -> usize {
    let elements = parts.len();
    let mut sizes = vec![0usize; options.parts];
    for &part in parts.iter() {
        sizes[part as usize] += 1;
    }
    let mean = elements as f64 / options.parts as f64;
    let maximum = (mean * (1.0 + options.imbalance_tolerance)).ceil() as usize;
    let minimum = (mean * (1.0 - options.imbalance_tolerance))
        .floor()
        .max(1.0) as usize;
    let mut moves = 0;
    let mut counts: Vec<(u32, u32)> = Vec::with_capacity(8);
    for _ in 0..options.refinement_passes {
        let mut pass_moves = 0;
        for element in 0..elements {
            let own = parts[element];
            counts.clear();
            let mut internal = 0u32;
            for &neighbor in &neighbors[offsets[element]..offsets[element + 1]] {
                let part = parts[neighbor as usize];
                if part == own {
                    internal += 1;
                } else if let Some(entry) = counts.iter_mut().find(|entry| entry.0 == part) {
                    entry.1 += 1;
                } else {
                    counts.push((part, 1));
                }
            }
            let Some(&(target, external)) = counts
                .iter()
                .max_by(|a, b| a.1.cmp(&b.1).then(b.0.cmp(&a.0)))
            else {
                continue;
            };
            if external > internal
                && sizes[target as usize] < maximum
                && sizes[own as usize] > minimum
            {
                parts[element] = target;
                sizes[target as usize] += 1;
                sizes[own as usize] -= 1;
                pass_moves += 1;
            }
        }
        moves += pass_moves;
        if pass_moves == 0 {
            break;
        }
    }
    moves
}

fn quality(
    mesh: &PartitionMesh,
    (offsets, neighbors): &(Vec<usize>, Vec<u32>),
    parts: &[u32],
    part_count: usize,
    refinement_moves: usize,
) -> PartitionQuality {
    let mut cut_faces = 0;
    for element in 0..parts.len() {
        for &neighbor in &neighbors[offsets[element]..offsets[element + 1]] {
            if (neighbor as usize) > element && parts[neighbor as usize] != parts[element] {
                cut_faces += 1;
            }
        }
    }
    let mut first_part = vec![u32::MAX; mesh.node_count()];
    let mut shared = vec![false; mesh.node_count()];
    for (element, &part) in parts.iter().enumerate() {
        for &node in mesh.element(element) {
            let node = node as usize;
            if first_part[node] == u32::MAX {
                first_part[node] = part;
            } else if first_part[node] != part {
                shared[node] = true;
            }
        }
    }
    let mut sizes = vec![0usize; part_count];
    for &part in parts {
        sizes[part as usize] += 1;
    }
    PartitionQuality {
        parts: part_count,
        cut_faces,
        interface_nodes: shared.iter().filter(|&&shared| shared).count(),
        smallest_part: sizes.iter().copied().min().unwrap_or(0),
        largest_part: sizes.iter().copied().max().unwrap_or(0),
        refinement_moves,
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    /// Structured box of `n^3` cubes, each split into six tetrahedra.
    pub(crate) fn box_mesh(n: usize, size: f64) -> PartitionMesh {
        let stride = n + 1;
        let mut coordinates = Vec::with_capacity(stride * stride * stride);
        for k in 0..=n {
            for j in 0..=n {
                for i in 0..=n {
                    coordinates.push([
                        size * i as f64 / n as f64,
                        size * j as f64 / n as f64,
                        size * k as f64 / n as f64,
                    ]);
                }
            }
        }
        let index = |i: usize, j: usize, k: usize| (i + stride * (j + stride * k)) as u32;
        let mut element_nodes = Vec::with_capacity(n * n * n * 24);
        for k in 0..n {
            for j in 0..n {
                for i in 0..n {
                    let v = [
                        index(i, j, k),
                        index(i + 1, j, k),
                        index(i + 1, j + 1, k),
                        index(i, j + 1, k),
                        index(i, j, k + 1),
                        index(i + 1, j, k + 1),
                        index(i + 1, j + 1, k + 1),
                        index(i, j + 1, k + 1),
                    ];
                    for tet in [
                        [0, 1, 2, 6],
                        [0, 2, 3, 6],
                        [0, 3, 7, 6],
                        [0, 7, 4, 6],
                        [0, 4, 5, 6],
                        [0, 5, 1, 6],
                    ] {
                        for corner in tet {
                            element_nodes.push(v[corner]);
                        }
                    }
                }
            }
        }
        PartitionMesh {
            coordinates,
            nodes_per_element: 4,
            element_nodes,
        }
    }

    #[test]
    fn partitions_are_balanced_and_refinement_never_increases_the_cut() {
        let mesh = box_mesh(10, 1.0);
        for parts in [1, 2, 3, 5, 8] {
            let mut unrefined = PartitionOptions::new(parts);
            unrefined.refinement_passes = 0;
            let (_, coarse) = partition_elements(&mesh, unrefined).unwrap();
            let (assignment, refined) =
                partition_elements(&mesh, PartitionOptions::new(parts)).unwrap();
            assert_eq!(assignment.len(), mesh.element_count());
            assert!(assignment.iter().all(|&part| (part as usize) < parts));
            assert!(
                refined.cut_faces <= coarse.cut_faces,
                "{parts}: {refined:?} {coarse:?}"
            );
            let mean = mesh.element_count() as f64 / parts as f64;
            assert!(refined.largest_part as f64 <= (mean * 1.03).ceil());
            assert!(refined.smallest_part > 0);
            if parts == 1 {
                assert_eq!(refined.cut_faces, 0);
            }
        }
    }

    #[test]
    fn partitioning_is_deterministic() {
        let mesh = box_mesh(6, 2.0);
        let first = partition_elements(&mesh, PartitionOptions::new(4)).unwrap();
        let second = partition_elements(&mesh, PartitionOptions::new(4)).unwrap();
        assert_eq!(first, second);
    }
}

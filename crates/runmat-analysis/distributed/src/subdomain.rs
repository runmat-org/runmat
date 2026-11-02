//! Overlapping subdomain extraction and the neighbor communication plan.
//!
//! Part `p` owns the elements `E_p` assigned by the partitioner. Every node is owned by the
//! lowest-numbered part among the parts whose elements contain it. Rank `p` receives `E_p` plus
//! every element sharing a node with `E_p` (one overlapping layer). Rows of the global operator
//! for the nodes of `E_p` are therefore complete on rank `p`: matrix-vector products for owned
//! rows and the restriction of the operator to the nodes of `E_p` (the overlapping Schwarz block)
//! are exact without any operator communication.

use crate::error::{DistributedError, DistributedResult};
use crate::krylov::KrylovSettings;
use crate::problem::PartitionedProblem;
use crate::rank::{NeighborPlan, SubdomainInput};

/// Sizes of every rank's piece, for evidence and diagnostics.
#[derive(Clone, Debug, PartialEq)]
pub struct SubdomainLayout {
    pub ranks: usize,
    pub owned_nodes: Vec<usize>,
    pub local_nodes: Vec<usize>,
    pub owned_elements: Vec<usize>,
    pub local_elements: Vec<usize>,
    pub neighbors: Vec<usize>,
}

/// Split one global problem into one input per subdomain (`ranks` subdomains).
pub fn extract_rank_inputs(
    problem: &PartitionedProblem,
    element_parts: &[u32],
    ranks: usize,
    settings: KrylovSettings,
) -> DistributedResult<(Vec<SubdomainInput>, SubdomainLayout)> {
    problem.validate()?;
    let mesh = &problem.mesh;
    let elements = mesh.element_count();
    let nodes = mesh.node_count();
    let dofs_per_node = problem.field.dofs_per_node();
    if element_parts.len() != elements || element_parts.iter().any(|&part| part as usize >= ranks) {
        return Err(DistributedError::InvalidProblem(
            "element partition does not match the mesh and rank count".into(),
        ));
    }
    // Node to element incidence (CSR).
    let mut node_offsets = vec![0usize; nodes + 1];
    for &node in &mesh.element_nodes {
        node_offsets[node as usize + 1] += 1;
    }
    for index in 0..nodes {
        node_offsets[index + 1] += node_offsets[index];
    }
    let mut cursor = node_offsets.clone();
    let mut node_elements = vec![0u32; mesh.element_nodes.len()];
    for element in 0..elements {
        for &node in mesh.element(element) {
            node_elements[cursor[node as usize]] = element as u32;
            cursor[node as usize] += 1;
        }
    }
    let mut owner = vec![u32::MAX; nodes];
    for (element, &part) in element_parts.iter().enumerate() {
        for &node in mesh.element(element) {
            let node = node as usize;
            owner[node] = owner[node].min(part);
        }
    }
    if owner.contains(&u32::MAX) {
        return Err(DistributedError::InvalidProblem(
            "every node must belong to at least one element".into(),
        ));
    }
    let mut part_elements = vec![Vec::new(); ranks];
    for (element, &part) in element_parts.iter().enumerate() {
        part_elements[part as usize].push(element as u32);
    }
    if part_elements.iter().any(Vec::is_empty) {
        return Err(DistributedError::InvalidProblem(
            "every rank needs at least one element".into(),
        ));
    }

    // Per part: interior nodes N_p (nodes of E_p), overlapped elements, local nodes L_p.
    let mut mark = vec![u32::MAX; nodes];
    let mut element_mark = vec![u32::MAX; elements];
    let mut interior = Vec::with_capacity(ranks);
    let mut overlap = Vec::with_capacity(ranks);
    let mut local = Vec::with_capacity(ranks);
    for (part, elements) in part_elements.iter().enumerate() {
        let stamp = part as u32;
        let mut interior_nodes = Vec::new();
        for &element in elements {
            element_mark[element as usize] = stamp;
            for &node in mesh.element(element as usize) {
                if mark[node as usize] != stamp {
                    mark[node as usize] = stamp;
                    interior_nodes.push(node);
                }
            }
        }
        let mut overlap_elements = Vec::new();
        for &node in &interior_nodes {
            for &element in
                &node_elements[node_offsets[node as usize]..node_offsets[node as usize + 1]]
            {
                if element_mark[element as usize] != stamp {
                    element_mark[element as usize] = stamp;
                    overlap_elements.push(element);
                }
            }
        }
        overlap_elements.sort_unstable();
        let mut local_nodes = interior_nodes.clone();
        for &element in &overlap_elements {
            for &node in mesh.element(element as usize) {
                if mark[node as usize] != stamp {
                    mark[node as usize] = stamp;
                    local_nodes.push(node);
                }
            }
        }
        interior_nodes.sort_unstable();
        local_nodes.sort_unstable();
        interior.push(interior_nodes);
        overlap.push(overlap_elements);
        local.push(local_nodes);
    }

    // Communication lists, ordered by global node id on both sides.
    let mut ghost_recv = vec![vec![Vec::new(); ranks]; ranks];
    let mut ghost_send = vec![vec![Vec::new(); ranks]; ranks];
    let mut accum_send = vec![vec![Vec::new(); ranks]; ranks];
    let mut accum_recv = vec![vec![Vec::new(); ranks]; ranks];
    for part in 0..ranks {
        for &node in &local[part] {
            let owning = owner[node as usize] as usize;
            if owning != part {
                ghost_recv[part][owning].push(node);
                ghost_send[owning][part].push(node);
            }
        }
        for &node in &interior[part] {
            let owning = owner[node as usize] as usize;
            if owning != part {
                accum_send[part][owning].push(node);
                accum_recv[owning][part].push(node);
            }
        }
    }

    // Subdomain centroids and length scales for the rigid-body coarse modes.
    let mut centroid = vec![[0.0f64; 3]; ranks];
    let mut owned_count = vec![0usize; ranks];
    for (node, &part) in owner.iter().enumerate() {
        let part = part as usize;
        owned_count[part] += 1;
        for axis in 0..3 {
            centroid[part][axis] += mesh.coordinates[node][axis];
        }
    }
    for part in 0..ranks {
        if owned_count[part] == 0 {
            return Err(DistributedError::InvalidProblem(format!(
                "rank {part} owns no nodes; use fewer ranks"
            )));
        }
        for axis in 0..3 {
            centroid[part][axis] /= owned_count[part] as f64;
        }
    }
    let mut scale = vec![0.0f64; ranks];
    for (node, &part) in owner.iter().enumerate() {
        let part = part as usize;
        let c = centroid[part];
        let x = mesh.coordinates[node];
        let distance =
            ((x[0] - c[0]).powi(2) + (x[1] - c[1]).powi(2) + (x[2] - c[2]).powi(2)).sqrt();
        scale[part] = scale[part].max(distance);
    }
    for value in &mut scale {
        if *value <= 0.0 {
            *value = 1.0;
        }
    }

    let mut prescribed = problem.prescribed.clone();
    prescribed.sort_by_key(|entry| entry.dof);
    prescribed.dedup_by_key(|entry| entry.dof);
    let mut loads = std::collections::BTreeMap::<u64, f64>::new();
    for load in &problem.loads {
        *loads.entry(load.dof).or_insert(0.0) += load.value;
    }

    let mut inputs = Vec::with_capacity(ranks);
    let mut layout = SubdomainLayout {
        ranks,
        owned_nodes: owned_count.clone(),
        local_nodes: Vec::with_capacity(ranks),
        owned_elements: Vec::with_capacity(ranks),
        local_elements: Vec::with_capacity(ranks),
        neighbors: Vec::with_capacity(ranks),
    };
    for part in 0..ranks {
        let local_nodes = &local[part];
        let local_index = |node: u32| -> u32 {
            local_nodes
                .binary_search(&node)
                .expect("communication node is local") as u32
        };
        let mut is_interior = vec![false; local_nodes.len()];
        for &node in &interior[part] {
            is_interior[local_index(node) as usize] = true;
        }
        let owners = local_nodes
            .iter()
            .map(|&node| owner[node as usize])
            .collect::<Vec<_>>();
        let element_ids = part_elements[part]
            .iter()
            .chain(&overlap[part])
            .copied()
            .collect::<Vec<_>>();
        let mut connectivity = Vec::with_capacity(element_ids.len() * mesh.nodes_per_element);
        let mut materials = Vec::with_capacity(element_ids.len());
        for &element in &element_ids {
            for &node in mesh.element(element as usize) {
                connectivity.push(local_index(node));
            }
            materials.push(problem.element_materials[element as usize]);
        }
        let mut local_prescribed = Vec::new();
        for (index, &node) in local_nodes.iter().enumerate() {
            for component in 0..dofs_per_node {
                let dof = node as u64 * dofs_per_node as u64 + component as u64;
                if let Ok(position) = prescribed.binary_search_by_key(&dof, |entry| entry.dof) {
                    local_prescribed.push((
                        (index * dofs_per_node + component) as u32,
                        prescribed[position].value,
                    ));
                }
            }
        }
        let mut local_loads = Vec::new();
        for (index, &node) in local_nodes.iter().enumerate() {
            if owners[index] as usize != part {
                continue;
            }
            for component in 0..dofs_per_node {
                let dof = node as u64 * dofs_per_node as u64 + component as u64;
                if let Some(value) = loads.get(&dof) {
                    local_loads.push(((index * dofs_per_node + component) as u32, *value));
                }
            }
        }
        let mut neighbors = Vec::new();
        for other in 0..ranks {
            if other == part
                || (ghost_recv[part][other].is_empty() && ghost_send[part][other].is_empty())
            {
                continue;
            }
            neighbors.push(NeighborPlan {
                subdomain: other as u32,
                ghost_recv: ghost_recv[part][other]
                    .iter()
                    .map(|&n| local_index(n))
                    .collect(),
                ghost_send: ghost_send[part][other]
                    .iter()
                    .map(|&n| local_index(n))
                    .collect(),
                accum_send: accum_send[part][other]
                    .iter()
                    .map(|&n| local_index(n))
                    .collect(),
                accum_recv: accum_recv[part][other]
                    .iter()
                    .map(|&n| local_index(n))
                    .collect(),
            });
        }
        layout.local_nodes.push(local_nodes.len());
        layout.owned_elements.push(part_elements[part].len());
        layout.local_elements.push(element_ids.len());
        layout.neighbors.push(neighbors.len());
        inputs.push(SubdomainInput {
            subdomain: part as u32,
            subdomains: ranks as u32,
            field: problem.field,
            nodes_per_element: mesh.nodes_per_element as u32,
            settings,
            global_nodes: local_nodes.iter().map(|&node| node as u64).collect(),
            coordinates: local_nodes
                .iter()
                .map(|&node| mesh.coordinates[node as usize])
                .collect(),
            node_owner: owners,
            interior: is_interior,
            elements: connectivity,
            element_materials: materials,
            global_elements: element_ids.iter().map(|&element| element as u64).collect(),
            owned_element_count: part_elements[part].len() as u32,
            prescribed: local_prescribed,
            loads: local_loads,
            neighbors,
            centroids: centroid.clone(),
            scales: scale.clone(),
        });
    }
    Ok((inputs, layout))
}

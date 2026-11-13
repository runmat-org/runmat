//! Multilevel nested dissection.
//!
//! For each subgraph: coarsen by heavy-edge matching, bisect the coarsest graph by greedy graph
//! growing from several seeds, project back with Fiduccia–Mattheyses refinement at every level,
//! turn the edge cut into a minimum vertex separator (König's theorem on the bipartite cut graph),
//! and recurse on both sides; separators are numbered last. Leaves are ordered by AMD. Every
//! choice is deterministic (no randomness, ties broken by vertex index), so the ordering — and with
//! it the factor — is identical across runs and thread counts.

use std::collections::{BinaryHeap, VecDeque};

use super::graph::Graph;

/// Subgraphs at or below this size are ordered by AMD.
const LEAF_SIZE: usize = 256;
/// Coarsening stops at this size.
const COARSEST_SIZE: usize = 160;
/// Maximum part weight relative to half the total.
const IMBALANCE: f64 = 1.10;
/// Subgraphs larger than this recurse in parallel.
const PARALLEL_SIZE: usize = 4096;

pub(crate) fn nested_dissection(graph: &Graph) -> Vec<usize> {
    let ids: Vec<usize> = (0..graph.vertex_count()).collect();
    dissect(graph.clone(), ids)
}

fn amd_or_identity(graph: &Graph, ids: &[usize]) -> Vec<usize> {
    match graph.amd_order() {
        Ok(order) => order.into_iter().map(|v| ids[v]).collect(),
        Err(_) => ids.to_vec(),
    }
}

fn dissect(graph: Graph, ids: Vec<usize>) -> Vec<usize> {
    let n = graph.vertex_count();
    if n <= LEAF_SIZE || graph.adjncy.is_empty() {
        return amd_or_identity(&graph, &ids);
    }
    // Disconnected graphs: dissect each component independently.
    let components = connected_components(&graph);
    if components.len() > 1 {
        let mut local = vec![usize::MAX; n];
        let parts: Vec<(Graph, Vec<usize>)> = components
            .into_iter()
            .map(|component| {
                let sub = graph.induced(&component, &mut local);
                let sub_ids = component.iter().map(|&v| ids[v]).collect();
                (sub, sub_ids)
            })
            .collect();
        drop(graph);
        let mut order = Vec::with_capacity(n);
        for (sub, sub_ids) in parts {
            order.extend(dissect(sub, sub_ids));
        }
        return order;
    }
    let part = separator(&graph);
    let (mut left, mut right, mut sep) = (Vec::new(), Vec::new(), Vec::new());
    for (v, &side) in part.iter().enumerate() {
        match side {
            0 => left.push(v),
            1 => right.push(v),
            _ => sep.push(v),
        }
    }
    if left.is_empty() || right.is_empty() {
        return amd_or_identity(&graph, &ids);
    }
    let mut local = vec![usize::MAX; n];
    let left_graph = graph.induced(&left, &mut local);
    let right_graph = graph.induced(&right, &mut local);
    let left_ids: Vec<usize> = left.iter().map(|&v| ids[v]).collect();
    let right_ids: Vec<usize> = right.iter().map(|&v| ids[v]).collect();
    let sep_ids: Vec<usize> = sep.iter().map(|&v| ids[v]).collect();
    drop(graph);
    let (left_order, right_order) = if n >= PARALLEL_SIZE && crate::par::threads() > 1 {
        rayon::join(
            || dissect(left_graph, left_ids),
            || dissect(right_graph, right_ids),
        )
    } else {
        (
            dissect(left_graph, left_ids),
            dissect(right_graph, right_ids),
        )
    };
    let mut order = left_order;
    order.extend(right_order);
    order.extend(sep_ids);
    order
}

fn connected_components(graph: &Graph) -> Vec<Vec<usize>> {
    let n = graph.vertex_count();
    let mut seen = vec![false; n];
    let mut components = Vec::new();
    let mut queue = VecDeque::new();
    for start in 0..n {
        if seen[start] {
            continue;
        }
        let mut component = Vec::new();
        seen[start] = true;
        queue.push_back(start);
        while let Some(v) = queue.pop_front() {
            component.push(v);
            for &u in graph.neighbors(v) {
                if !seen[u] {
                    seen[u] = true;
                    queue.push_back(u);
                }
            }
        }
        component.sort_unstable();
        components.push(component);
    }
    components
}

/// Returns part labels 0/1 and 2 for separator vertices.
fn separator(graph: &Graph) -> Vec<u8> {
    let part = edge_bisection(graph);
    vertex_separator(graph, part)
}

/// Multilevel balanced edge bisection (labels 0/1).
pub(crate) fn edge_bisection(graph: &Graph) -> Vec<u8> {
    // Coarsening hierarchy.
    let mut levels: Vec<Graph> = Vec::new();
    let mut maps: Vec<Vec<usize>> = Vec::new();
    let mut current = graph.clone();
    let max_vertex_weight =
        ((graph.total_weight() as f64 / COARSEST_SIZE as f64) * 1.5).ceil() as u64;
    while current.vertex_count() > COARSEST_SIZE {
        let (coarse, cmap) = coarsen(&current, max_vertex_weight.max(1));
        if coarse.vertex_count() as f64 > 0.9 * current.vertex_count() as f64 {
            break;
        }
        levels.push(current);
        maps.push(cmap);
        current = coarse;
    }
    let mut part = initial_bisection(&current);
    refine(&current, &mut part);
    while let Some(fine) = levels.pop() {
        let cmap = maps.pop().unwrap();
        let mut projected = vec![0u8; fine.vertex_count()];
        for v in 0..fine.vertex_count() {
            projected[v] = part[cmap[v]];
        }
        part = projected;
        refine(&fine, &mut part);
        current = fine;
    }
    let _ = current;
    part
}

/// Heavy-edge matching; returns the coarse graph and the fine→coarse map.
fn coarsen(graph: &Graph, max_vertex_weight: u64) -> (Graph, Vec<usize>) {
    let n = graph.vertex_count();
    // Visit low-degree vertices first (counting sort by degree; ties by index).
    let max_degree = (0..n).map(|v| graph.neighbors(v).len()).max().unwrap_or(0);
    let mut buckets = vec![0usize; max_degree + 2];
    for v in 0..n {
        buckets[graph.neighbors(v).len() + 1] += 1;
    }
    for d in 0..=max_degree {
        buckets[d + 1] += buckets[d];
    }
    let mut visit = vec![0usize; n];
    for v in 0..n {
        let d = graph.neighbors(v).len();
        visit[buckets[d]] = v;
        buckets[d] += 1;
    }
    let mut mate = vec![usize::MAX; n];
    for &v in &visit {
        if mate[v] != usize::MAX {
            continue;
        }
        let mut best = usize::MAX;
        let mut best_weight = 0u32;
        for (&u, &w) in graph.neighbors(v).iter().zip(graph.edge_weights(v)) {
            if mate[u] == usize::MAX
                && u != v
                && (graph.vwgt[v] as u64 + graph.vwgt[u] as u64) <= max_vertex_weight
                && (w > best_weight
                    || (w == best_weight && best != usize::MAX && graph.vwgt[u] < graph.vwgt[best]))
            {
                best = u;
                best_weight = w;
            }
        }
        if best == usize::MAX {
            mate[v] = v;
        } else {
            mate[v] = best;
            mate[best] = v;
        }
    }
    let mut cmap = vec![usize::MAX; n];
    let mut coarse_count = 0usize;
    for v in 0..n {
        if cmap[v] == usize::MAX {
            cmap[v] = coarse_count;
            cmap[mate[v]] = coarse_count;
            coarse_count += 1;
        }
    }
    let mut xadj = vec![0usize; coarse_count + 1];
    let mut adjncy = Vec::with_capacity(graph.adjncy.len() / 2);
    let mut adjwgt = Vec::with_capacity(graph.adjncy.len() / 2);
    let mut vwgt = vec![0u32; coarse_count];
    let mut slot = vec![usize::MAX; coarse_count];
    let mut c = 0usize;
    for v in 0..n {
        if cmap[v] != c || (mate[v] < v) {
            continue;
        }
        let start = adjncy.len();
        let members = if mate[v] == v {
            [v, usize::MAX]
        } else {
            [v, mate[v]]
        };
        for &m in members.iter().filter(|&&m| m != usize::MAX) {
            vwgt[c] += graph.vwgt[m];
            for (&u, &w) in graph.neighbors(m).iter().zip(graph.edge_weights(m)) {
                let cu = cmap[u];
                if cu == c {
                    continue;
                }
                if slot[cu] == usize::MAX || slot[cu] < start {
                    slot[cu] = adjncy.len();
                    adjncy.push(cu);
                    adjwgt.push(w);
                } else {
                    adjwgt[slot[cu]] += w;
                }
            }
        }
        xadj[c + 1] = adjncy.len();
        c += 1;
    }
    debug_assert_eq!(c, coarse_count);
    (
        Graph {
            xadj,
            adjncy,
            adjwgt,
            vwgt,
        },
        cmap,
    )
}

fn cut_weight(graph: &Graph, part: &[u8]) -> u64 {
    let mut cut = 0u64;
    for v in 0..graph.vertex_count() {
        for (&u, &w) in graph.neighbors(v).iter().zip(graph.edge_weights(v)) {
            if part[u] != part[v] {
                cut += w as u64;
            }
        }
    }
    cut / 2
}

fn pseudo_peripheral(graph: &Graph, start: usize) -> usize {
    let n = graph.vertex_count();
    let mut current = start;
    let mut distance = vec![usize::MAX; n];
    for _ in 0..4 {
        distance.iter_mut().for_each(|d| *d = usize::MAX);
        let mut queue = VecDeque::new();
        distance[current] = 0;
        queue.push_back(current);
        let mut last = current;
        while let Some(v) = queue.pop_front() {
            last = v;
            for &u in graph.neighbors(v) {
                if distance[u] == usize::MAX {
                    distance[u] = distance[v] + 1;
                    queue.push_back(u);
                }
            }
        }
        if last == current {
            break;
        }
        current = last;
    }
    current
}

/// Greedy graph growing from several seeds; keeps the smallest refined cut.
fn initial_bisection(graph: &Graph) -> Vec<u8> {
    let n = graph.vertex_count();
    let total = graph.total_weight();
    let mut seeds = vec![pseudo_peripheral(graph, 0)];
    for k in 1..4 {
        seeds.push((k * n) / 4);
    }
    let mut best: Option<(u64, Vec<u8>)> = None;
    for &seed in &seeds {
        let mut part = vec![1u8; n];
        let mut grown = 0u64;
        // gain[v] = (edges into region) - (edges out of region), for frontier vertices.
        let mut heap = BinaryHeap::new();
        let mut gain = vec![i64::MIN; n];
        let mut in_heap = vec![false; n];
        let push = |v: usize, g: i64, heap: &mut BinaryHeap<(i64, std::cmp::Reverse<usize>)>| {
            heap.push((g, std::cmp::Reverse(v)));
        };
        gain[seed] = 0;
        push(seed, 0, &mut heap);
        in_heap[seed] = true;
        while grown * 2 < total {
            let Some((g, std::cmp::Reverse(v))) = heap.pop() else {
                // Disconnected remainder: take any unassigned vertex.
                match (0..n).find(|&v| part[v] == 1) {
                    Some(v) => {
                        gain[v] = 0;
                        push(v, 0, &mut heap);
                        continue;
                    }
                    None => break,
                }
            };
            if part[v] == 0 || g != gain[v] {
                continue;
            }
            part[v] = 0;
            grown += graph.vwgt[v] as u64;
            for (&u, &w) in graph.neighbors(v).iter().zip(graph.edge_weights(v)) {
                if part[u] == 1 {
                    if !in_heap[u] {
                        // Initialize: -(external degree)
                        let mut g0 = 0i64;
                        for (&x, &wx) in graph.neighbors(u).iter().zip(graph.edge_weights(u)) {
                            g0 += if part[x] == 0 {
                                wx as i64
                            } else {
                                -(wx as i64)
                            };
                        }
                        gain[u] = g0;
                        in_heap[u] = true;
                    } else {
                        gain[u] += 2 * w as i64;
                    }
                    push(u, gain[u], &mut heap);
                }
            }
        }
        refine(graph, &mut part);
        let cut = cut_weight(graph, &part);
        if best.as_ref().is_none_or(|(c, _)| cut < *c) {
            best = Some((cut, part));
        }
    }
    best.map(|(_, part)| part).unwrap_or_else(|| vec![0; n])
}

/// Boundary Fiduccia–Mattheyses refinement of a 2-way edge partition under the balance bound.
fn refine(graph: &Graph, part: &mut [u8]) {
    let n = graph.vertex_count();
    let total = graph.total_weight();
    let max_part = ((total as f64 / 2.0) * IMBALANCE).ceil() as u64;
    let mut pw = [0u64; 2];
    for v in 0..n {
        pw[part[v] as usize] += graph.vwgt[v] as u64;
    }
    // External minus internal degree per vertex.
    let mut gain = vec![0i64; n];
    let compute_gain = |v: usize, part: &[u8]| -> i64 {
        let mut g = 0i64;
        for (&u, &w) in graph.neighbors(v).iter().zip(graph.edge_weights(v)) {
            g += if part[u] != part[v] {
                w as i64
            } else {
                -(w as i64)
            };
        }
        g
    };
    for (v, gain) in gain.iter_mut().enumerate() {
        *gain = compute_gain(v, part);
    }
    // Rebalance first if needed: move best-gain vertices off the heavy side.
    for _ in 0..n {
        let heavy = if pw[0] > max_part {
            0u8
        } else if pw[1] > max_part {
            1u8
        } else {
            break;
        };
        let mut best = usize::MAX;
        for v in 0..n {
            if part[v] == heavy && (best == usize::MAX || gain[v] > gain[best]) {
                // Prefer boundary vertices.
                best = v;
            }
        }
        if best == usize::MAX {
            break;
        }
        move_vertex(graph, part, &mut gain, &mut pw, best);
    }
    let move_limit = (n / 100).max(25);
    for _pass in 0..8 {
        let mut heap: BinaryHeap<(i64, std::cmp::Reverse<usize>)> = BinaryHeap::new();
        for v in 0..n {
            if graph.neighbors(v).iter().any(|&u| part[u] != part[v]) {
                heap.push((gain[v], std::cmp::Reverse(v)));
            }
        }
        let mut locked = vec![false; n];
        let mut moves: Vec<usize> = Vec::new();
        let mut cut_delta = 0i64;
        let (mut best_delta, mut best_len) = (0i64, 0usize);
        let mut best_balance = pw[0].abs_diff(pw[1]);
        while let Some((g, std::cmp::Reverse(v))) = heap.pop() {
            if locked[v] || g != gain[v] {
                continue;
            }
            let from = part[v] as usize;
            let to = 1 - from;
            if pw[to] + graph.vwgt[v] as u64 > max_part {
                continue;
            }
            locked[v] = true;
            cut_delta -= gain[v];
            move_vertex(graph, part, &mut gain, &mut pw, v);
            moves.push(v);
            for &u in graph.neighbors(v) {
                if !locked[u] {
                    heap.push((gain[u], std::cmp::Reverse(u)));
                }
            }
            let balance = pw[0].abs_diff(pw[1]);
            if cut_delta < best_delta || (cut_delta == best_delta && balance < best_balance) {
                best_delta = cut_delta;
                best_len = moves.len();
                best_balance = balance;
            }
            if moves.len() - best_len > move_limit {
                break;
            }
        }
        // Roll back moves past the best prefix.
        while moves.len() > best_len {
            let v = moves.pop().unwrap();
            move_vertex(graph, part, &mut gain, &mut pw, v);
        }
        if best_len == 0 {
            break;
        }
    }
}

fn move_vertex(graph: &Graph, part: &mut [u8], gain: &mut [i64], pw: &mut [u64; 2], v: usize) {
    let from = part[v] as usize;
    let to = 1 - from;
    part[v] = to as u8;
    pw[from] -= graph.vwgt[v] as u64;
    pw[to] += graph.vwgt[v] as u64;
    gain[v] = -gain[v];
    for (&u, &w) in graph.neighbors(v).iter().zip(graph.edge_weights(v)) {
        // Edge (u,v) flipped between internal and external for u.
        if part[u] as usize == to {
            gain[u] -= 2 * w as i64;
        } else {
            gain[u] += 2 * w as i64;
        }
    }
}

/// Minimum vertex cover of the cut edges (Hopcroft–Karp + König), then trims redundant
/// separator vertices.
fn vertex_separator(graph: &Graph, mut part: Vec<u8>) -> Vec<u8> {
    let n = graph.vertex_count();
    let mut left_index = vec![usize::MAX; n];
    let mut right_index = vec![usize::MAX; n];
    let (mut left, mut right) = (Vec::new(), Vec::new());
    for v in 0..n {
        let boundary = graph.neighbors(v).iter().any(|&u| part[u] != part[v]);
        if boundary {
            if part[v] == 0 {
                left_index[v] = left.len();
                left.push(v);
            } else {
                right_index[v] = right.len();
                right.push(v);
            }
        }
    }
    let adjacency: Vec<Vec<usize>> = left
        .iter()
        .map(|&v| {
            graph
                .neighbors(v)
                .iter()
                .filter(|&&u| part[u] == 1)
                .map(|&u| right_index[u])
                .collect()
        })
        .collect();
    let (match_left, match_right) = hopcroft_karp(&adjacency, right.len());
    // König: Z = vertices reachable from unmatched left vertices by alternating paths.
    let mut visited_left = vec![false; left.len()];
    let mut visited_right = vec![false; right.len()];
    let mut queue = VecDeque::new();
    for (l, &m) in match_left.iter().enumerate() {
        if m == usize::MAX {
            visited_left[l] = true;
            queue.push_back(l);
        }
    }
    while let Some(l) = queue.pop_front() {
        for &r in &adjacency[l] {
            if !visited_right[r] && match_left[l] != r {
                visited_right[r] = true;
                let m = match_right[r];
                if m != usize::MAX && !visited_left[m] {
                    visited_left[m] = true;
                    queue.push_back(m);
                }
            }
        }
    }
    for (l, &v) in left.iter().enumerate() {
        if !visited_left[l] {
            part[v] = 2;
        }
    }
    for (r, &v) in right.iter().enumerate() {
        if visited_right[r] {
            part[v] = 2;
        }
    }
    // Trim: a separator vertex touching only one side (plus separator) can join that side.
    for v in 0..n {
        if part[v] != 2 {
            continue;
        }
        let (mut touches0, mut touches1) = (false, false);
        for &u in graph.neighbors(v) {
            match part[u] {
                0 => touches0 = true,
                1 => touches1 = true,
                _ => {}
            }
        }
        if !touches1 {
            part[v] = 0;
        } else if !touches0 {
            part[v] = 1;
        }
    }
    part
}

fn hopcroft_karp(adjacency: &[Vec<usize>], right_count: usize) -> (Vec<usize>, Vec<usize>) {
    let left_count = adjacency.len();
    let mut match_left = vec![usize::MAX; left_count];
    let mut match_right = vec![usize::MAX; right_count];
    // Greedy initialization.
    for l in 0..left_count {
        for &r in &adjacency[l] {
            if match_right[r] == usize::MAX {
                match_left[l] = r;
                match_right[r] = l;
                break;
            }
        }
    }
    let mut dist = vec![usize::MAX; left_count];
    loop {
        // BFS layers from free left vertices.
        let mut queue = VecDeque::new();
        for l in 0..left_count {
            if match_left[l] == usize::MAX {
                dist[l] = 0;
                queue.push_back(l);
            } else {
                dist[l] = usize::MAX;
            }
        }
        let mut found = false;
        while let Some(l) = queue.pop_front() {
            for &r in &adjacency[l] {
                let m = match_right[r];
                if m == usize::MAX {
                    found = true;
                } else if dist[m] == usize::MAX {
                    dist[m] = dist[l] + 1;
                    queue.push_back(m);
                }
            }
        }
        if !found {
            break;
        }
        // Iterative DFS augment along layered graph.
        let mut iter = vec![0usize; left_count];
        for start in 0..left_count {
            if match_left[start] != usize::MAX {
                continue;
            }
            let mut stack = vec![start];
            let mut path_right: Vec<usize> = Vec::new();
            while let Some(&l) = stack.last() {
                if iter[l] < adjacency[l].len() {
                    let r = adjacency[l][iter[l]];
                    iter[l] += 1;
                    let m = match_right[r];
                    if m == usize::MAX {
                        // Augment.
                        path_right.push(r);
                        for (depth, &pl) in stack.iter().enumerate() {
                            let pr = path_right[depth];
                            match_left[pl] = pr;
                            match_right[pr] = pl;
                        }
                        stack.clear();
                        break;
                    } else if dist[m] == dist[l] + 1 {
                        path_right.push(r);
                        stack.push(m);
                    }
                } else {
                    dist[l] = usize::MAX;
                    stack.pop();
                    path_right.pop();
                }
            }
        }
    }
    (match_left, match_right)
}

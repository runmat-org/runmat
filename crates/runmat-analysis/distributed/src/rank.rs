//! One worker process of the decomposition, owning one or more subdomains.
//!
//! Every subdomain is assembled, factorized and iterated independently of where it is placed.
//! Messages between subdomains of the same process are copied in memory; messages to other
//! processes are batched per destination process. Reductions combine per-subdomain
//! contributions in global subdomain order, so for a fixed partition the result is bitwise
//! identical for any number of processes and any assignment of subdomains to processes.

use std::collections::BTreeMap;
use std::time::Instant;

use faer::linalg::solvers::Solve;
use faer::{Mat, Side};

use crate::comm::Communicator;
use crate::error::{DistributedError, DistributedResult};
use crate::frame::{Decoder, Encoder};
use crate::krylov::{IterationRecord, KrylovSettings, KrylovSummary};
use crate::local_solver::{SubdomainFactorization, SubdomainSolverFactory};
use crate::problem::{ElementKernel, FieldKind};

/// Exchange lists with one neighbor subdomain, as local node indices ordered by global node id.
#[derive(Clone, Debug, PartialEq)]
pub struct NeighborPlan {
    pub subdomain: u32,
    /// Local nodes owned by the neighbor whose values this subdomain needs.
    pub ghost_recv: Vec<u32>,
    /// Owned nodes whose values the neighbor needs.
    pub ghost_send: Vec<u32>,
    /// Interior (overlap) nodes owned by the neighbor: Schwarz corrections to send.
    pub accum_send: Vec<u32>,
    /// Owned nodes inside the neighbor's overlap: Schwarz corrections to receive.
    pub accum_recv: Vec<u32>,
}

/// Everything one subdomain needs; no process receives the global mesh or operator.
#[derive(Clone, Debug, PartialEq)]
pub struct SubdomainInput {
    pub subdomain: u32,
    pub subdomains: u32,
    pub field: FieldKind,
    pub nodes_per_element: u32,
    pub settings: KrylovSettings,
    pub global_nodes: Vec<u64>,
    pub coordinates: Vec<[f64; 3]>,
    /// Owning subdomain of each local node.
    pub node_owner: Vec<u32>,
    /// Nodes of this subdomain's own elements (the overlapping Schwarz block).
    pub interior: Vec<bool>,
    /// Local connectivity; owned elements first, then the overlap layer.
    pub elements: Vec<u32>,
    pub element_materials: Vec<u32>,
    pub global_elements: Vec<u64>,
    pub owned_element_count: u32,
    /// Prescribed values on local dofs.
    pub prescribed: Vec<(u32, f64)>,
    /// Assembled external loads on owned dofs.
    pub loads: Vec<(u32, f64)>,
    /// Neighbor plans in ascending neighbor subdomain order.
    pub neighbors: Vec<NeighborPlan>,
    /// Centroid and length scale of every subdomain's owned nodes (rigid-body coarse modes).
    pub centroids: Vec<[f64; 3]>,
    pub scales: Vec<f64>,
}

/// Placement of subdomains on worker processes.
#[derive(Clone, Debug, PartialEq)]
pub struct ProcessPlan {
    pub process: u32,
    pub processes: u32,
    /// Process of each subdomain.
    pub subdomain_process: Vec<u32>,
    /// Threads this process may use for its own subdomains and factorizations.
    pub threads: u32,
}

impl ProcessPlan {
    /// Contiguous blocks of subdomains per process.
    pub fn blocked(process: usize, processes: usize, subdomains: usize, threads: u32) -> Self {
        Self {
            process: process as u32,
            processes: processes as u32,
            subdomain_process: (0..subdomains)
                .map(|subdomain| (subdomain * processes / subdomains) as u32)
                .collect(),
            threads: threads.max(1),
        }
    }

    pub fn local_subdomains(&self) -> Vec<u32> {
        self.subdomain_process
            .iter()
            .enumerate()
            .filter(|(_, &process)| process == self.process)
            .map(|(subdomain, _)| subdomain as u32)
            .collect()
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct RankTimings {
    pub assembly_seconds: f64,
    pub factorization_seconds: f64,
    pub coarse_seconds: f64,
    pub iteration_seconds: f64,
    pub recovery_seconds: f64,
}

/// Element results recovered by the subdomain that owns the elements.
#[derive(Clone, Debug, PartialEq)]
pub struct SubdomainElementResults {
    pub elements: Vec<u64>,
    pub width: u32,
    pub values: Vec<f64>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct SubdomainOutput {
    pub subdomain: u32,
    pub owned_nodes: Vec<u64>,
    /// Solution on owned nodes, node-major.
    pub solution: Vec<f64>,
    pub results: SubdomainElementResults,
    pub local_dofs: u64,
    pub schwarz_dofs: u64,
}

#[derive(Clone, Debug, PartialEq)]
pub struct RankOutput {
    pub process: u32,
    pub subdomains: Vec<SubdomainOutput>,
    pub summary: KrylovSummary,
    /// Relative residual of the returned solution recomputed from the operator.
    pub true_relative_residual: f64,
    pub timings: RankTimings,
    pub coarse_dimension: u64,
}

struct Csr {
    offsets: Vec<usize>,
    columns: Vec<u32>,
    values: Vec<f64>,
}

struct Subdomain<'a> {
    input: &'a SubdomainInput,
    matrix: Csr,
    rhs: Vec<f64>,
    owned: Vec<bool>,
    owned_dofs: Vec<usize>,
    interior_dofs: Vec<usize>,
    prescribed: Vec<bool>,
    factor: Option<Box<dyn SubdomainFactorization>>,
    /// Coarse basis on owned dofs, `owned_dofs.len() * modes`.
    basis: Vec<f64>,
}

struct Coarse {
    modes: usize,
    factor: faer::linalg::solvers::Llt<f64>,
    active: Vec<bool>,
}

/// Run every subdomain of one process to completion.
pub fn run_rank(
    inputs: &[SubdomainInput],
    plan: &ProcessPlan,
    kernel: &dyn ElementKernel,
    factory: &dyn SubdomainSolverFactory,
    comm: &mut dyn Communicator,
) -> DistributedResult<RankOutput> {
    validate(inputs, plan, kernel, comm)?;
    // Parallelism comes from processes and from factorizing local subdomains side by side;
    // each factorization itself is sequential.
    faer::set_global_parallelism(faer::Par::Seq);
    let mut timings = RankTimings::default();
    let settings = inputs[0].settings;
    let field = inputs[0].field;
    let d = field.dofs_per_node();

    let started = Instant::now();
    let mut subdomains = parallel_map(plan.threads, inputs, |input| {
        let (matrix, rhs) = assemble(input, kernel)?;
        let dofs = input.coordinates.len() * d;
        let owned = (0..dofs)
            .map(|dof| input.node_owner[dof / d] == input.subdomain)
            .collect::<Vec<_>>();
        let mut prescribed = vec![false; dofs];
        for &(dof, _) in &input.prescribed {
            prescribed[dof as usize] = true;
        }
        Ok(Subdomain {
            input,
            matrix,
            rhs,
            owned_dofs: (0..dofs).filter(|&dof| owned[dof]).collect(),
            interior_dofs: (0..dofs).filter(|&dof| input.interior[dof / d]).collect(),
            owned,
            prescribed,
            factor: None,
            basis: Vec::new(),
        })
    })?;
    timings.assembly_seconds = started.elapsed().as_secs_f64();

    let started = Instant::now();
    let factors = parallel_map(plan.threads, &subdomains, |subdomain| {
        factorize_interior(&subdomain.matrix, &subdomain.interior_dofs, factory)
    })?;
    for (subdomain, factor) in subdomains.iter_mut().zip(factors) {
        subdomain.factor = Some(factor);
    }
    timings.factorization_seconds = started.elapsed().as_secs_f64();

    let mut process = Process {
        plan,
        field,
        subdomains,
        coarse: None,
        position: BTreeMap::new(),
    };
    process.position = process
        .subdomains
        .iter()
        .enumerate()
        .map(|(index, subdomain)| (subdomain.input.subdomain, index))
        .collect();

    let started = Instant::now();
    if settings.coarse_space {
        process.coarse = Some(process.build_coarse(comm)?);
    }
    timings.coarse_seconds = started.elapsed().as_secs_f64();

    let started = Instant::now();
    let (mut solution, summary) = process.solve(settings, comm)?;
    let true_relative_residual = process.true_residual(&mut solution, comm)?;
    timings.iteration_seconds = started.elapsed().as_secs_f64();

    let started = Instant::now();
    process.exchange_ghosts(&mut solution, comm)?;
    let mut outputs = Vec::with_capacity(process.subdomains.len());
    for (subdomain, vector) in process.subdomains.iter().zip(&solution) {
        let results = subdomain.recover(vector, kernel, d)?;
        let owned_nodes = (0..subdomain.input.coordinates.len())
            .filter(|&node| subdomain.input.node_owner[node] == subdomain.input.subdomain)
            .collect::<Vec<_>>();
        let mut owned_solution = Vec::with_capacity(owned_nodes.len() * d);
        for &node in &owned_nodes {
            owned_solution.extend_from_slice(&vector[node * d..(node + 1) * d]);
        }
        outputs.push(SubdomainOutput {
            subdomain: subdomain.input.subdomain,
            owned_nodes: owned_nodes
                .iter()
                .map(|&node| subdomain.input.global_nodes[node])
                .collect(),
            solution: owned_solution,
            results,
            local_dofs: subdomain.owned.len() as u64,
            schwarz_dofs: subdomain.interior_dofs.len() as u64,
        });
    }
    timings.recovery_seconds = started.elapsed().as_secs_f64();
    Ok(RankOutput {
        process: plan.process,
        subdomains: outputs,
        summary,
        true_relative_residual,
        timings,
        coarse_dimension: process.coarse.as_ref().map_or(0, |coarse| {
            (coarse.modes * inputs[0].subdomains as usize) as u64
        }),
    })
}

/// Map `items` with up to `threads` scoped threads, preserving order.
fn parallel_map<'a, T: Sync, R: Send>(
    threads: u32,
    items: &'a [T],
    map: impl Fn(&'a T) -> DistributedResult<R> + Sync,
) -> DistributedResult<Vec<R>> {
    let threads = (threads as usize).clamp(1, items.len().max(1));
    if threads == 1 {
        return items.iter().map(&map).collect();
    }
    let next = std::sync::atomic::AtomicUsize::new(0);
    let mut slots = (0..items.len())
        .map(|_| None)
        .collect::<Vec<Option<DistributedResult<R>>>>();
    let results = std::sync::Mutex::new(&mut slots);
    std::thread::scope(|scope| {
        for _ in 0..threads {
            scope.spawn(|| loop {
                let index = next.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                if index >= items.len() {
                    break;
                }
                let value = map(&items[index]);
                results.lock().expect("result slots")[index] = Some(value);
            });
        }
    });
    slots
        .into_iter()
        .map(|slot| slot.expect("every item mapped"))
        .collect()
}

fn validate(
    inputs: &[SubdomainInput],
    plan: &ProcessPlan,
    kernel: &dyn ElementKernel,
    comm: &dyn Communicator,
) -> DistributedResult<()> {
    let invalid = |message: &str| Err(DistributedError::InvalidProblem(message.into()));
    let Some(first) = inputs.first() else {
        return invalid("a process needs at least one subdomain");
    };
    if comm.rank() != plan.process as usize
        || comm.size() != plan.processes as usize
        || plan.subdomain_process.len() != first.subdomains as usize
        || plan
            .subdomain_process
            .iter()
            .any(|&process| process >= plan.processes)
        || plan.local_subdomains()
            != inputs
                .iter()
                .map(|input| input.subdomain)
                .collect::<Vec<_>>()
    {
        return invalid("process plan differs from its communicator or subdomains");
    }
    for input in inputs {
        let nodes = input.coordinates.len();
        let per_element = input.nodes_per_element as usize;
        if input.field != first.field
            || input.subdomains != first.subdomains
            || input.settings != first.settings
            || kernel.field() != input.field
            || kernel.nodes_per_element() != per_element
        {
            return invalid("subdomains disagree on field, element type or settings");
        }
        if input.global_nodes.len() != nodes
            || input.node_owner.len() != nodes
            || input.interior.len() != nodes
            || input.elements.len() != input.element_materials.len() * per_element
            || input.global_elements.len() != input.element_materials.len()
            || input.owned_element_count as usize > input.element_materials.len()
            || input.centroids.len() != input.subdomains as usize
            || input.scales.len() != input.subdomains as usize
            || input.elements.iter().any(|&node| node as usize >= nodes)
            || input
                .node_owner
                .iter()
                .any(|&owner| owner >= input.subdomains)
            || input
                .neighbors
                .windows(2)
                .any(|pair| pair[0].subdomain >= pair[1].subdomain)
        {
            return invalid("subdomain input is internally inconsistent");
        }
        let dofs = nodes * input.field.dofs_per_node();
        if input
            .prescribed
            .iter()
            .chain(&input.loads)
            .any(|&(dof, value)| dof as usize >= dofs || !value.is_finite())
        {
            return invalid("subdomain input has an invalid prescribed value or load");
        }
    }
    Ok(())
}

/// Assemble the local operator over every local element with prescribed dofs eliminated
/// symmetrically (identity rows and columns) and the right-hand side of owned rows.
fn assemble(
    input: &SubdomainInput,
    kernel: &dyn ElementKernel,
) -> DistributedResult<(Csr, Vec<f64>)> {
    let d = input.field.dofs_per_node();
    let n = input.nodes_per_element as usize;
    let nodes = input.coordinates.len();
    let dofs = nodes * d;
    let mut adjacency: Vec<Vec<u32>> = vec![Vec::new(); nodes];
    for element in input.elements.chunks_exact(n) {
        for &a in element {
            adjacency[a as usize].extend_from_slice(element);
        }
    }
    for list in &mut adjacency {
        list.sort_unstable();
        list.dedup();
    }
    let mut offsets = Vec::with_capacity(dofs + 1);
    offsets.push(0usize);
    for list in &adjacency {
        for _ in 0..d {
            offsets.push(offsets.last().copied().unwrap_or(0) + list.len() * d);
        }
    }
    let mut columns = vec![0u32; offsets[dofs]];
    for (a, list) in adjacency.iter().enumerate() {
        for i in 0..d {
            let start = offsets[a * d + i];
            for (position, &b) in list.iter().enumerate() {
                for j in 0..d {
                    columns[start + position * d + j] = b * d as u32 + j as u32;
                }
            }
        }
    }
    let mut values = vec![0.0f64; offsets[dofs]];
    let width = n * d;
    let mut element_matrix = vec![0.0f64; width * width];
    let mut coordinates = vec![[0.0f64; 3]; n];
    let mut positions = vec![0usize; n * n];
    for (element, connectivity) in input.elements.chunks_exact(n).enumerate() {
        for (slot, &node) in connectivity.iter().enumerate() {
            coordinates[slot] = input.coordinates[node as usize];
        }
        element_matrix.iter_mut().for_each(|value| *value = 0.0);
        kernel
            .element_matrix(
                input.element_materials[element],
                &coordinates,
                &mut element_matrix,
            )
            .map_err(DistributedError::from)?;
        for (s, &a) in connectivity.iter().enumerate() {
            for (t, &b) in connectivity.iter().enumerate() {
                positions[s * n + t] = adjacency[a as usize]
                    .binary_search(&b)
                    .expect("element nodes are adjacent");
            }
        }
        for (s, &a) in connectivity.iter().enumerate() {
            for i in 0..d {
                let row_start = offsets[a as usize * d + i];
                let element_row = (s * d + i) * width;
                for t in 0..n {
                    let base = row_start + positions[s * n + t] * d;
                    for j in 0..d {
                        values[base + j] += element_matrix[element_row + t * d + j];
                    }
                }
            }
        }
    }
    let mut prescribed = vec![None; dofs];
    for &(dof, value) in &input.prescribed {
        prescribed[dof as usize] = Some(value);
    }
    let mut rhs = vec![0.0f64; dofs];
    for &(dof, value) in &input.loads {
        rhs[dof as usize] += value;
    }
    let mut compact_offsets = Vec::with_capacity(dofs + 1);
    compact_offsets.push(0usize);
    let mut compact_columns = Vec::with_capacity(columns.len());
    let mut compact_values = Vec::with_capacity(values.len());
    for row in 0..dofs {
        if let Some(value) = prescribed[row] {
            compact_columns.push(row as u32);
            compact_values.push(1.0);
            rhs[row] = value;
        } else {
            for entry in offsets[row]..offsets[row + 1] {
                let column = columns[entry] as usize;
                match prescribed[column] {
                    Some(value) => rhs[row] -= values[entry] * value,
                    None => {
                        compact_columns.push(columns[entry]);
                        compact_values.push(values[entry]);
                    }
                }
            }
        }
        compact_offsets.push(compact_columns.len());
    }
    Ok((
        Csr {
            offsets: compact_offsets,
            columns: compact_columns,
            values: compact_values,
        },
        rhs,
    ))
}

fn factorize_interior(
    matrix: &Csr,
    interior_dofs: &[usize],
    factory: &dyn SubdomainSolverFactory,
) -> DistributedResult<Box<dyn SubdomainFactorization>> {
    let mut map = vec![usize::MAX; matrix.offsets.len() - 1];
    for (index, &dof) in interior_dofs.iter().enumerate() {
        map[dof] = index;
    }
    let mut column_offsets = Vec::with_capacity(interior_dofs.len() + 1);
    column_offsets.push(0);
    let mut row_indices = Vec::new();
    let mut values = Vec::new();
    for (column, &dof) in interior_dofs.iter().enumerate() {
        // Symmetric operator: row `dof` of the CSR is column `dof`; keep the lower triangle.
        for entry in matrix.offsets[dof]..matrix.offsets[dof + 1] {
            let row = map[matrix.columns[entry] as usize];
            if row != usize::MAX && row >= column {
                row_indices.push(row);
                values.push(matrix.values[entry]);
            }
        }
        column_offsets.push(row_indices.len());
    }
    factory.factorize(interior_dofs.len(), column_offsets, row_indices, values)
}

impl Subdomain<'_> {
    fn dot_owned(&self, left: &[f64], right: &[f64]) -> f64 {
        self.owned_dofs
            .iter()
            .map(|&dof| left[dof] * right[dof])
            .sum()
    }

    fn apply_owned_rows(&self, vector: &[f64], result: &mut [f64]) {
        result.iter_mut().for_each(|value| *value = 0.0);
        for &row in &self.owned_dofs {
            let mut sum = 0.0;
            for entry in self.matrix.offsets[row]..self.matrix.offsets[row + 1] {
                sum += self.matrix.values[entry] * vector[self.matrix.columns[entry] as usize];
            }
            result[row] = sum;
        }
    }

    fn mode_value(&self, mode: usize, node: usize, component: usize, owner: usize) -> f64 {
        match self.input.field {
            FieldKind::Scalar => 1.0,
            FieldKind::Elastic => {
                if mode < 3 {
                    return if component == mode { 1.0 } else { 0.0 };
                }
                let centroid = self.input.centroids[owner];
                let scale = self.input.scales[owner];
                let x = self.input.coordinates[node];
                let (dx, dy, dz) = (
                    (x[0] - centroid[0]) / scale,
                    (x[1] - centroid[1]) / scale,
                    (x[2] - centroid[2]) / scale,
                );
                let rotation = match mode {
                    3 => [0.0, -dz, dy],
                    4 => [dz, 0.0, -dx],
                    _ => [-dy, dx, 0.0],
                };
                rotation[component]
            }
        }
    }

    /// Coarse basis value of subdomain `owner`'s mode at a local dof (zero off its owned dofs).
    fn basis_at(&self, dof: usize, mode: usize, owner: usize, d: usize) -> f64 {
        if self.prescribed[dof] || self.input.node_owner[dof / d] as usize != owner {
            0.0
        } else {
            self.mode_value(mode, dof / d, dof % d, owner)
        }
    }

    fn recover(
        &self,
        solution: &[f64],
        kernel: &dyn ElementKernel,
        d: usize,
    ) -> DistributedResult<SubdomainElementResults> {
        let n = self.input.nodes_per_element as usize;
        let width = kernel.result_width();
        let owned_elements = self.input.owned_element_count as usize;
        let mut values = vec![0.0; owned_elements * width];
        let mut coordinates = vec![[0.0f64; 3]; n];
        let mut nodal = vec![0.0f64; n * d];
        for element in 0..owned_elements {
            let connectivity = &self.input.elements[element * n..(element + 1) * n];
            for (slot, &node) in connectivity.iter().enumerate() {
                coordinates[slot] = self.input.coordinates[node as usize];
                for component in 0..d {
                    nodal[slot * d + component] = solution[node as usize * d + component];
                }
            }
            kernel
                .element_result(
                    self.input.element_materials[element],
                    &coordinates,
                    &nodal,
                    &mut values[element * width..(element + 1) * width],
                )
                .map_err(DistributedError::from)?;
        }
        Ok(SubdomainElementResults {
            elements: self.input.global_elements[..owned_elements].to_vec(),
            width: width as u32,
            values,
        })
    }
}

struct Process<'a> {
    plan: &'a ProcessPlan,
    field: FieldKind,
    subdomains: Vec<Subdomain<'a>>,
    coarse: Option<Coarse>,
    /// Global subdomain id to its index in `subdomains`.
    position: BTreeMap<u32, usize>,
}

/// Which plan lists a subdomain-to-subdomain message carries.
#[derive(Clone, Copy)]
enum Message {
    Ghost,
    Correction,
}

impl Process<'_> {
    fn d(&self) -> usize {
        self.field.dofs_per_node()
    }

    /// Exchange one message per (sender, neighbor) subdomain pair. Local pairs are copied;
    /// remote pairs are batched per process in ascending (sender, receiver) order, which both
    /// sides derive from their plans without headers. Returns, per local subdomain, the
    /// received buffer of each neighbor (in neighbor order).
    fn exchange(
        &self,
        vectors: &[Vec<f64>],
        message: Message,
        comm: &mut dyn Communicator,
    ) -> DistributedResult<Vec<Vec<Vec<f64>>>> {
        let d = self.d();
        let process_of = |subdomain: u32| self.plan.subdomain_process[subdomain as usize];
        let mut outgoing: BTreeMap<u32, Vec<f64>> = BTreeMap::new();
        let mut received = self
            .subdomains
            .iter()
            .map(|subdomain| vec![Vec::new(); subdomain.input.neighbors.len()])
            .collect::<Vec<_>>();
        for (index, subdomain) in self.subdomains.iter().enumerate() {
            for neighbor in &subdomain.input.neighbors {
                let nodes = match message {
                    Message::Ghost => &neighbor.ghost_send,
                    Message::Correction => &neighbor.accum_send,
                };
                let payload = gather(&vectors[index], nodes, d);
                let destination = process_of(neighbor.subdomain);
                if destination == self.plan.process {
                    let target = self.position[&neighbor.subdomain];
                    let slot = self.subdomains[target]
                        .input
                        .neighbors
                        .binary_search_by_key(&subdomain.input.subdomain, |plan| plan.subdomain)
                        .map_err(|_| {
                            DistributedError::Communication("neighbor plans are asymmetric".into())
                        })?;
                    received[target][slot] = payload;
                } else {
                    outgoing.entry(destination).or_default().extend(payload);
                }
            }
        }
        // Every process with a remote neighbor pair takes part, in both directions.
        let mut peers = Vec::new();
        for subdomain in &self.subdomains {
            for neighbor in &subdomain.input.neighbors {
                let process = process_of(neighbor.subdomain);
                if process != self.plan.process && !peers.contains(&(process as usize)) {
                    peers.push(process as usize);
                }
            }
        }
        peers.sort_unstable();
        let sends = peers
            .iter()
            .map(|peer| outgoing.remove(&(*peer as u32)).unwrap_or_default())
            .collect();
        let buffers = comm.exchange(&peers, sends)?;
        for (peer, buffer) in peers.iter().zip(buffers) {
            // Expected order: sender subdomains ascending, then receivers ascending.
            let mut pairs = Vec::new();
            for (index, subdomain) in self.subdomains.iter().enumerate() {
                for (slot, neighbor) in subdomain.input.neighbors.iter().enumerate() {
                    if process_of(neighbor.subdomain) as usize == *peer {
                        let count = match message {
                            Message::Ghost => neighbor.ghost_recv.len(),
                            Message::Correction => neighbor.accum_recv.len(),
                        } * d;
                        pairs.push((
                            neighbor.subdomain,
                            subdomain.input.subdomain,
                            index,
                            slot,
                            count,
                        ));
                    }
                }
            }
            pairs.sort_unstable_by_key(|pair| (pair.0, pair.1));
            let mut cursor = 0usize;
            for (_, _, index, slot, count) in pairs {
                let end = cursor + count;
                if end > buffer.len() {
                    return Err(DistributedError::Communication(
                        "process exchange is shorter than its plans".into(),
                    ));
                }
                received[index][slot] = buffer[cursor..end].to_vec();
                cursor = end;
            }
            if cursor != buffer.len() {
                return Err(DistributedError::Communication(
                    "process exchange is longer than its plans".into(),
                ));
            }
        }
        Ok(received)
    }

    fn exchange_ghosts(
        &self,
        vectors: &mut [Vec<f64>],
        comm: &mut dyn Communicator,
    ) -> DistributedResult<()> {
        let d = self.d();
        let received = self.exchange(vectors, Message::Ghost, comm)?;
        for ((subdomain, vector), buffers) in self.subdomains.iter().zip(vectors).zip(received) {
            for (neighbor, values) in subdomain.input.neighbors.iter().zip(buffers) {
                if values.len() != neighbor.ghost_recv.len() * d {
                    return Err(DistributedError::Communication(
                        "ghost exchange size differs from its plan".into(),
                    ));
                }
                for (slot, &node) in neighbor.ghost_recv.iter().enumerate() {
                    vector[node as usize * d..(node as usize + 1) * d]
                        .copy_from_slice(&values[slot * d..(slot + 1) * d]);
                }
            }
        }
        Ok(())
    }

    /// Sum per-subdomain contributions over every process, in global subdomain order.
    fn reduce(
        &self,
        contributions: impl Fn(&Subdomain<'_>, usize) -> Vec<f64>,
        width: usize,
        comm: &mut dyn Communicator,
    ) -> DistributedResult<Vec<f64>> {
        let mut local = Vec::with_capacity(self.subdomains.len() * width);
        for (index, subdomain) in self.subdomains.iter().enumerate() {
            let values = contributions(subdomain, index);
            if values.len() != width {
                return Err(DistributedError::Communication(
                    "reduction contribution width differs".into(),
                ));
            }
            local.extend(values);
        }
        let gathered = comm.allgather(&local)?;
        let subdomains = self.plan.subdomain_process.len();
        let mut per_subdomain = vec![None; subdomains];
        for (process, values) in gathered.iter().enumerate() {
            let owned = self
                .plan
                .subdomain_process
                .iter()
                .enumerate()
                .filter(|(_, &owner)| owner as usize == process)
                .map(|(subdomain, _)| subdomain)
                .collect::<Vec<_>>();
            if values.len() != owned.len() * width {
                return Err(DistributedError::Communication(
                    "reduction contributions differ from the placement".into(),
                ));
            }
            for (slot, subdomain) in owned.into_iter().enumerate() {
                per_subdomain[subdomain] = Some(&values[slot * width..(slot + 1) * width]);
            }
        }
        let mut total = vec![0.0; width];
        for values in per_subdomain {
            let values = values.ok_or_else(|| {
                DistributedError::Communication("a subdomain contributed nothing".into())
            })?;
            for (sum, value) in total.iter_mut().zip(values) {
                *sum += value;
            }
        }
        Ok(total)
    }

    fn dot(
        &self,
        left: &[Vec<f64>],
        right: &[Vec<f64>],
        comm: &mut dyn Communicator,
    ) -> DistributedResult<f64> {
        Ok(self.reduce(
            |subdomain, index| vec![subdomain.dot_owned(&left[index], &right[index])],
            1,
            comm,
        )?[0])
    }

    fn apply_operator(
        &self,
        vectors: &mut [Vec<f64>],
        results: &mut [Vec<f64>],
        comm: &mut dyn Communicator,
    ) -> DistributedResult<()> {
        self.exchange_ghosts(vectors, comm)?;
        for ((subdomain, vector), result) in self.subdomains.iter().zip(vectors.iter()).zip(results)
        {
            subdomain.apply_owned_rows(vector, result);
        }
        Ok(())
    }

    /// Two-level additive Schwarz: overlapping subdomain solves plus the coarse correction.
    fn precondition(
        &self,
        residuals: &mut [Vec<f64>],
        results: &mut [Vec<f64>],
        comm: &mut dyn Communicator,
    ) -> DistributedResult<()> {
        let d = self.d();
        self.exchange_ghosts(residuals, comm)?;
        let local = parallel_map(
            self.plan.threads,
            &self
                .subdomains
                .iter()
                .zip(residuals.iter())
                .collect::<Vec<_>>(),
            |(subdomain, residual)| {
                let mut values = subdomain
                    .interior_dofs
                    .iter()
                    .map(|&dof| residual[dof])
                    .collect::<Vec<_>>();
                subdomain
                    .factor
                    .as_ref()
                    .expect("factorized subdomain")
                    .solve_in_place(&mut values);
                Ok(values)
            },
        )?;
        for ((subdomain, result), values) in
            self.subdomains.iter().zip(results.iter_mut()).zip(local)
        {
            result.iter_mut().for_each(|value| *value = 0.0);
            for (&dof, value) in subdomain.interior_dofs.iter().zip(values) {
                result[dof] = value;
            }
        }
        let received = self.exchange(results, Message::Correction, comm)?;
        for ((subdomain, result), buffers) in
            self.subdomains.iter().zip(results.iter_mut()).zip(received)
        {
            for (dof, value) in result.iter_mut().enumerate() {
                if !subdomain.owned[dof] {
                    *value = 0.0;
                }
            }
            for (neighbor, values) in subdomain.input.neighbors.iter().zip(buffers) {
                if values.len() != neighbor.accum_recv.len() * d {
                    return Err(DistributedError::Communication(
                        "Schwarz correction size differs from its plan".into(),
                    ));
                }
                for (slot, &node) in neighbor.accum_recv.iter().enumerate() {
                    for component in 0..d {
                        result[node as usize * d + component] += values[slot * d + component];
                    }
                }
            }
        }
        if let Some(coarse) = &self.coarse {
            let modes = coarse.modes;
            let subdomains = self.plan.subdomain_process.len();
            // Every subdomain projects its owned residual onto its own modes.
            let mut local = Vec::with_capacity(self.subdomains.len() * modes);
            for (subdomain, residual) in self.subdomains.iter().zip(residuals.iter()) {
                let mut projection = vec![0.0; modes];
                for (slot, &dof) in subdomain.owned_dofs.iter().enumerate() {
                    for (mode, value) in projection.iter_mut().enumerate() {
                        *value += subdomain.basis[slot * modes + mode] * residual[dof];
                    }
                }
                local.extend(projection);
            }
            let gathered = comm.allgather(&local)?;
            let mut rhs = Mat::<f64>::zeros(modes * subdomains, 1);
            for (process, values) in gathered.iter().enumerate() {
                let owned = self
                    .plan
                    .subdomain_process
                    .iter()
                    .enumerate()
                    .filter(|(_, &owner)| owner as usize == process)
                    .map(|(subdomain, _)| subdomain);
                for (slot, subdomain) in owned.enumerate() {
                    for mode in 0..modes {
                        if coarse.active[subdomain * modes + mode] {
                            rhs[(subdomain * modes + mode, 0)] = values[slot * modes + mode];
                        }
                    }
                }
            }
            coarse.factor.solve_in_place(rhs.as_mut());
            for (subdomain, result) in self.subdomains.iter().zip(results.iter_mut()) {
                let offset = subdomain.input.subdomain as usize * modes;
                for (slot, &dof) in subdomain.owned_dofs.iter().enumerate() {
                    let mut correction = 0.0;
                    for mode in 0..modes {
                        if coarse.active[offset + mode] {
                            correction +=
                                subdomain.basis[slot * modes + mode] * rhs[(offset + mode, 0)];
                        }
                    }
                    result[dof] += correction;
                }
            }
        }
        Ok(())
    }

    fn build_coarse(&mut self, comm: &mut dyn Communicator) -> DistributedResult<Coarse> {
        let d = self.d();
        let modes = self.field.coarse_modes();
        let subdomains = self.plan.subdomain_process.len();
        for subdomain in &mut self.subdomains {
            let own = subdomain.input.subdomain as usize;
            let mut basis = Vec::with_capacity(subdomain.owned_dofs.len() * modes);
            for &dof in &subdomain.owned_dofs {
                for mode in 0..modes {
                    basis.push(subdomain.basis_at(dof, mode, own, d));
                }
            }
            subdomain.basis = basis;
        }
        // Each subdomain forms Z_s^T A Z_t for t in {s} and its neighbors; owned rows are exact.
        // Contributions are fixed-size per subdomain: one block per possible target.
        let mut local = Vec::new();
        let mut layouts = Vec::new();
        for subdomain in &self.subdomains {
            let own = subdomain.input.subdomain as usize;
            let mut targets = vec![own];
            targets.extend(
                subdomain
                    .input
                    .neighbors
                    .iter()
                    .map(|plan| plan.subdomain as usize),
            );
            targets.sort_unstable();
            let mut blocks = Vec::with_capacity(targets.len() * (1 + modes * modes));
            for &target in &targets {
                let mut block = vec![0.0; modes * modes];
                for (slot, &row) in subdomain.owned_dofs.iter().enumerate() {
                    let mut product = vec![0.0; modes];
                    for entry in subdomain.matrix.offsets[row]..subdomain.matrix.offsets[row + 1] {
                        let column = subdomain.matrix.columns[entry] as usize;
                        if subdomain.input.node_owner[column / d] as usize != target {
                            continue;
                        }
                        let value = subdomain.matrix.values[entry];
                        for (mode, output) in product.iter_mut().enumerate() {
                            *output += value * subdomain.basis_at(column, mode, target, d);
                        }
                    }
                    for left in 0..modes {
                        let weight = subdomain.basis[slot * modes + left];
                        if weight != 0.0 {
                            for right in 0..modes {
                                block[left * modes + right] += weight * product[right];
                            }
                        }
                    }
                }
                blocks.push(target as f64);
                blocks.extend(block);
            }
            layouts.push(blocks.len());
            local.push(blocks);
        }
        // Lengths first (variable per subdomain), then the padded blocks.
        let lengths = self.reduce_lengths(&layouts, comm)?;
        let maximum = lengths.iter().copied().max().unwrap_or(0);
        let mut flat = Vec::with_capacity(self.subdomains.len() * maximum);
        for mut blocks in local {
            blocks.resize(maximum, f64::NAN);
            flat.extend(blocks);
        }
        let gathered = comm.allgather(&flat)?;
        let dimension = subdomains * modes;
        let mut operator = vec![0.0f64; dimension * dimension];
        for (process, values) in gathered.iter().enumerate() {
            let owned = self
                .plan
                .subdomain_process
                .iter()
                .enumerate()
                .filter(|(_, &owner)| owner as usize == process)
                .map(|(subdomain, _)| subdomain);
            for (slot, source) in owned.enumerate() {
                let used = lengths[source];
                let chunk = &values[slot * maximum..slot * maximum + used];
                for block in chunk.chunks_exact(1 + modes * modes) {
                    let target = block[0] as usize;
                    for left in 0..modes {
                        for right in 0..modes {
                            operator
                                [(source * modes + left) * dimension + target * modes + right] +=
                                block[1 + left * modes + right];
                        }
                    }
                }
            }
        }
        // Symmetrize (blocks were formed by different subdomains) and retire null modes.
        for i in 0..dimension {
            for j in i + 1..dimension {
                let mean = 0.5 * (operator[i * dimension + j] + operator[j * dimension + i]);
                operator[i * dimension + j] = mean;
                operator[j * dimension + i] = mean;
            }
        }
        let largest = (0..dimension)
            .map(|i| operator[i * dimension + i].abs())
            .fold(0.0, f64::max);
        let mut active = vec![true; dimension];
        for i in 0..dimension {
            if operator[i * dimension + i] <= 1e-12 * largest.max(f64::MIN_POSITIVE) {
                active[i] = false;
                for j in 0..dimension {
                    operator[i * dimension + j] = 0.0;
                    operator[j * dimension + i] = 0.0;
                }
                operator[i * dimension + i] = 1.0;
            }
        }
        let matrix = Mat::<f64>::from_fn(dimension, dimension, |i, j| operator[i * dimension + j]);
        let factor = matrix.llt(Side::Lower).map_err(|error| {
            DistributedError::Factorization(format!("coarse operator: {error:?}"))
        })?;
        Ok(Coarse {
            modes,
            factor,
            active,
        })
    }

    /// Per-subdomain lengths of every subdomain, indexed by global subdomain id.
    fn reduce_lengths(
        &self,
        local: &[usize],
        comm: &mut dyn Communicator,
    ) -> DistributedResult<Vec<usize>> {
        let values = local
            .iter()
            .map(|&length| length as f64)
            .collect::<Vec<_>>();
        let gathered = comm.allgather(&values)?;
        let mut lengths = vec![0usize; self.plan.subdomain_process.len()];
        for (process, values) in gathered.iter().enumerate() {
            let owned = self
                .plan
                .subdomain_process
                .iter()
                .enumerate()
                .filter(|(_, &owner)| owner as usize == process)
                .map(|(subdomain, _)| subdomain);
            for (slot, subdomain) in owned.enumerate() {
                lengths[subdomain] = values[slot] as usize;
            }
        }
        Ok(lengths)
    }

    fn solve(
        &self,
        settings: KrylovSettings,
        comm: &mut dyn Communicator,
    ) -> DistributedResult<(Vec<Vec<f64>>, KrylovSummary)> {
        let zeros = || {
            self.subdomains
                .iter()
                .map(|subdomain| vec![0.0; subdomain.owned.len()])
                .collect::<Vec<_>>()
        };
        let mut b = zeros();
        for (subdomain, vector) in self.subdomains.iter().zip(&mut b) {
            for &dof in &subdomain.owned_dofs {
                vector[dof] = subdomain.rhs[dof];
            }
        }
        let rhs_norm = self.dot(&b, &b, comm)?.sqrt();
        let mut x = zeros();
        let mut history = Vec::new();
        if rhs_norm == 0.0 {
            return Ok((
                x,
                KrylovSummary {
                    iterations: 0,
                    relative_residual: 0.0,
                    converged: true,
                    history,
                },
            ));
        }
        let mut r = b;
        let mut z = zeros();
        self.precondition(&mut r, &mut z, comm)?;
        let mut p = z.clone();
        let mut q = zeros();
        let mut rz = self.dot(&r, &z, comm)?;
        let mut relative = 1.0;
        for iteration in 1..=settings.maximum_iterations {
            self.apply_operator(&mut p, &mut q, comm)?;
            let pq = self.dot(&p, &q, comm)?;
            if pq <= 0.0 || !pq.is_finite() {
                return Err(DistributedError::NotConverged(format!(
                    "operator lost positive definiteness at iteration {iteration}"
                )));
            }
            let alpha = rz / pq;
            for (index, subdomain) in self.subdomains.iter().enumerate() {
                for &dof in &subdomain.owned_dofs {
                    x[index][dof] += alpha * p[index][dof];
                    r[index][dof] -= alpha * q[index][dof];
                }
            }
            self.precondition(&mut r, &mut z, comm)?;
            let pair = self.reduce(
                |subdomain, index| {
                    vec![
                        subdomain.dot_owned(&r[index], &r[index]),
                        subdomain.dot_owned(&r[index], &z[index]),
                    ]
                },
                2,
                comm,
            )?;
            relative = pair[0].sqrt() / rhs_norm;
            history.push(IterationRecord {
                iteration,
                relative_residual: relative,
            });
            if relative <= settings.relative_tolerance {
                return Ok((
                    x,
                    KrylovSummary {
                        iterations: iteration,
                        relative_residual: relative,
                        converged: true,
                        history,
                    },
                ));
            }
            let beta = pair[1] / rz;
            rz = pair[1];
            for (index, subdomain) in self.subdomains.iter().enumerate() {
                for &dof in &subdomain.owned_dofs {
                    p[index][dof] = z[index][dof] + beta * p[index][dof];
                }
            }
        }
        Err(DistributedError::NotConverged(format!(
            "relative residual {relative:.3e} after {} iterations",
            settings.maximum_iterations
        )))
    }

    fn true_residual(
        &self,
        solution: &mut [Vec<f64>],
        comm: &mut dyn Communicator,
    ) -> DistributedResult<f64> {
        let mut product = self
            .subdomains
            .iter()
            .map(|subdomain| vec![0.0; subdomain.owned.len()])
            .collect::<Vec<_>>();
        self.apply_operator(solution, &mut product, comm)?;
        let sums = self.reduce(
            |subdomain, index| {
                let mut sums = vec![0.0, 0.0];
                for &dof in &subdomain.owned_dofs {
                    sums[0] += (subdomain.rhs[dof] - product[index][dof]).powi(2);
                    sums[1] += subdomain.rhs[dof].powi(2);
                }
                sums
            },
            2,
            comm,
        )?;
        Ok(if sums[1] == 0.0 {
            0.0
        } else {
            (sums[0] / sums[1]).sqrt()
        })
    }
}

fn gather(vector: &[f64], nodes: &[u32], d: usize) -> Vec<f64> {
    let mut values = Vec::with_capacity(nodes.len() * d);
    for &node in nodes {
        values.extend_from_slice(&vector[node as usize * d..(node as usize + 1) * d]);
    }
    values
}

impl SubdomainInput {
    pub fn encode(&self) -> Vec<u8> {
        let mut encoder = Encoder::default();
        encoder.u32(self.subdomain);
        encoder.u32(self.subdomains);
        encoder.u32(self.field.code());
        encoder.u32(self.nodes_per_element);
        encoder.f64(self.settings.relative_tolerance);
        encoder.u32(self.settings.maximum_iterations);
        encoder.u32(u32::from(self.settings.coarse_space));
        encoder.u64s(&self.global_nodes);
        encoder.points(&self.coordinates);
        encoder.u32s(&self.node_owner);
        encoder.bools(&self.interior);
        encoder.u32s(&self.elements);
        encoder.u32s(&self.element_materials);
        encoder.u64s(&self.global_elements);
        encoder.u32(self.owned_element_count);
        encoder.pairs(&self.prescribed);
        encoder.pairs(&self.loads);
        encoder.len(self.neighbors.len());
        for neighbor in &self.neighbors {
            encoder.u32(neighbor.subdomain);
            encoder.u32s(&neighbor.ghost_recv);
            encoder.u32s(&neighbor.ghost_send);
            encoder.u32s(&neighbor.accum_send);
            encoder.u32s(&neighbor.accum_recv);
        }
        encoder.points(&self.centroids);
        encoder.f64s(&self.scales);
        encoder.bytes
    }

    pub fn decode(bytes: &[u8]) -> DistributedResult<Self> {
        let mut decoder = Decoder::new(bytes);
        let subdomain = decoder.u32()?;
        let subdomains = decoder.u32()?;
        let field = FieldKind::from_code(decoder.u32()?)?;
        let nodes_per_element = decoder.u32()?;
        let settings = KrylovSettings {
            relative_tolerance: decoder.f64()?,
            maximum_iterations: decoder.u32()?,
            coarse_space: decoder.u32()? != 0,
        };
        let global_nodes = decoder.u64s()?;
        let coordinates = decoder.points()?;
        let node_owner = decoder.u32s()?;
        let interior = decoder.bools()?;
        let elements = decoder.u32s()?;
        let element_materials = decoder.u32s()?;
        let global_elements = decoder.u64s()?;
        let owned_element_count = decoder.u32()?;
        let prescribed = decoder.pairs()?;
        let loads = decoder.pairs()?;
        let neighbor_count = decoder.len(4)?;
        let mut neighbors = Vec::with_capacity(neighbor_count);
        for _ in 0..neighbor_count {
            neighbors.push(NeighborPlan {
                subdomain: decoder.u32()?,
                ghost_recv: decoder.u32s()?,
                ghost_send: decoder.u32s()?,
                accum_send: decoder.u32s()?,
                accum_recv: decoder.u32s()?,
            });
        }
        let centroids = decoder.points()?;
        let scales = decoder.f64s()?;
        decoder.finish()?;
        Ok(Self {
            subdomain,
            subdomains,
            field,
            nodes_per_element,
            settings,
            global_nodes,
            coordinates,
            node_owner,
            interior,
            elements,
            element_materials,
            global_elements,
            owned_element_count,
            prescribed,
            loads,
            neighbors,
            centroids,
            scales,
        })
    }
}

impl ProcessPlan {
    pub fn encode(&self) -> Vec<u8> {
        let mut encoder = Encoder::default();
        encoder.u32(self.process);
        encoder.u32(self.processes);
        encoder.u32s(&self.subdomain_process);
        encoder.u32(self.threads);
        encoder.bytes
    }

    pub fn decode(bytes: &[u8]) -> DistributedResult<Self> {
        let mut decoder = Decoder::new(bytes);
        let plan = Self {
            process: decoder.u32()?,
            processes: decoder.u32()?,
            subdomain_process: decoder.u32s()?,
            threads: decoder.u32()?,
        };
        decoder.finish()?;
        Ok(plan)
    }
}

impl RankOutput {
    pub fn encode(&self) -> Vec<u8> {
        let mut encoder = Encoder::default();
        encoder.u32(self.process);
        encoder.len(self.subdomains.len());
        for subdomain in &self.subdomains {
            encoder.u32(subdomain.subdomain);
            encoder.u64s(&subdomain.owned_nodes);
            encoder.f64s(&subdomain.solution);
            encoder.u64s(&subdomain.results.elements);
            encoder.u32(subdomain.results.width);
            encoder.f64s(&subdomain.results.values);
            encoder.u64(subdomain.local_dofs);
            encoder.u64(subdomain.schwarz_dofs);
        }
        encoder.u32(self.summary.iterations);
        encoder.f64(self.summary.relative_residual);
        encoder.u32(u32::from(self.summary.converged));
        encoder.len(self.summary.history.len());
        for record in &self.summary.history {
            encoder.u32(record.iteration);
            encoder.f64(record.relative_residual);
        }
        encoder.f64(self.true_relative_residual);
        for value in [
            self.timings.assembly_seconds,
            self.timings.factorization_seconds,
            self.timings.coarse_seconds,
            self.timings.iteration_seconds,
            self.timings.recovery_seconds,
        ] {
            encoder.f64(value);
        }
        encoder.u64(self.coarse_dimension);
        encoder.bytes
    }

    pub fn decode(bytes: &[u8]) -> DistributedResult<Self> {
        let mut decoder = Decoder::new(bytes);
        let process = decoder.u32()?;
        let count = decoder.len(4)?;
        let mut subdomains = Vec::with_capacity(count);
        for _ in 0..count {
            subdomains.push(SubdomainOutput {
                subdomain: decoder.u32()?,
                owned_nodes: decoder.u64s()?,
                solution: decoder.f64s()?,
                results: SubdomainElementResults {
                    elements: decoder.u64s()?,
                    width: decoder.u32()?,
                    values: decoder.f64s()?,
                },
                local_dofs: decoder.u64()?,
                schwarz_dofs: decoder.u64()?,
            });
        }
        let iterations = decoder.u32()?;
        let relative_residual = decoder.f64()?;
        let converged = decoder.u32()? != 0;
        let history_length = decoder.len(12)?;
        let mut history = Vec::with_capacity(history_length);
        for _ in 0..history_length {
            history.push(IterationRecord {
                iteration: decoder.u32()?,
                relative_residual: decoder.f64()?,
            });
        }
        let true_relative_residual = decoder.f64()?;
        let timings = RankTimings {
            assembly_seconds: decoder.f64()?,
            factorization_seconds: decoder.f64()?,
            coarse_seconds: decoder.f64()?,
            iteration_seconds: decoder.f64()?,
            recovery_seconds: decoder.f64()?,
        };
        let coarse_dimension = decoder.u64()?;
        decoder.finish()?;
        Ok(Self {
            process,
            subdomains,
            summary: KrylovSummary {
                iterations,
                relative_residual,
                converged,
                history,
            },
            true_relative_residual,
            timings,
            coarse_dimension,
        })
    }
}

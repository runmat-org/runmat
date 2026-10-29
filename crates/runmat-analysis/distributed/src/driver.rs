//! Coordinator helpers: partition, place ranks and reassemble their owned pieces.

use crate::comm::ThreadCommunicator;
use crate::error::{DistributedError, DistributedResult};
use crate::krylov::KrylovSettings;
use crate::local_solver::SubdomainSolverFactory;
use crate::partition::{partition_elements, PartitionOptions, PartitionQuality};
use crate::problem::{ElementKernel, PartitionedProblem};
use crate::rank::{run_rank, ProcessPlan, RankOutput, RankTimings};
use crate::subdomain::{extract_rank_inputs, SubdomainLayout};

/// Global solution reassembled from every rank's owned nodes and elements.
#[derive(Clone, Debug, PartialEq)]
pub struct DistributedSolution {
    pub solution: Vec<f64>,
    pub element_results: Vec<f64>,
    pub result_width: usize,
    pub iterations: u32,
    pub relative_residual: f64,
    pub true_relative_residual: f64,
    /// Slowest rank per phase.
    pub timings: RankTimings,
    pub coarse_dimension: u64,
}

/// Reassemble rank outputs into global node-major solution and element-ordered results.
pub fn gather_outputs(
    outputs: &[RankOutput],
    node_count: usize,
    dofs_per_node: usize,
    element_count: usize,
) -> DistributedResult<DistributedSolution> {
    let first = outputs
        .first()
        .ok_or_else(|| DistributedError::InvalidProblem("no process outputs".into()))?;
    let width = first
        .subdomains
        .first()
        .map(|subdomain| subdomain.results.width as usize)
        .ok_or_else(|| {
            DistributedError::InvalidProblem("a process returned no subdomain".into())
        })?;
    let mut solution = vec![f64::NAN; node_count * dofs_per_node];
    let mut results = vec![f64::NAN; element_count * width];
    let mut timings = RankTimings::default();
    for output in outputs {
        if output.summary.iterations != first.summary.iterations {
            return Err(DistributedError::InvalidProblem(
                "processes disagree on their iteration count".into(),
            ));
        }
        for subdomain in &output.subdomains {
            if subdomain.results.width as usize != width
                || subdomain.solution.len() != subdomain.owned_nodes.len() * dofs_per_node
                || subdomain.results.values.len() != subdomain.results.elements.len() * width
            {
                return Err(DistributedError::InvalidProblem(
                    "subdomain outputs disagree on their layout".into(),
                ));
            }
            for (slot, &node) in subdomain.owned_nodes.iter().enumerate() {
                let node = node as usize;
                solution[node * dofs_per_node..(node + 1) * dofs_per_node].copy_from_slice(
                    &subdomain.solution[slot * dofs_per_node..(slot + 1) * dofs_per_node],
                );
            }
            for (slot, &element) in subdomain.results.elements.iter().enumerate() {
                let element = element as usize;
                results[element * width..(element + 1) * width]
                    .copy_from_slice(&subdomain.results.values[slot * width..(slot + 1) * width]);
            }
        }
        timings.assembly_seconds = timings
            .assembly_seconds
            .max(output.timings.assembly_seconds);
        timings.factorization_seconds = timings
            .factorization_seconds
            .max(output.timings.factorization_seconds);
        timings.coarse_seconds = timings.coarse_seconds.max(output.timings.coarse_seconds);
        timings.iteration_seconds = timings
            .iteration_seconds
            .max(output.timings.iteration_seconds);
        timings.recovery_seconds = timings
            .recovery_seconds
            .max(output.timings.recovery_seconds);
    }
    if solution.iter().any(|value| value.is_nan()) || results.iter().any(|value| value.is_nan()) {
        return Err(DistributedError::InvalidProblem(
            "rank outputs do not cover every node and element".into(),
        ));
    }
    Ok(DistributedSolution {
        solution,
        element_results: results,
        result_width: width,
        iterations: first.summary.iterations,
        relative_residual: first.summary.relative_residual,
        true_relative_residual: first.true_relative_residual,
        timings,
        coarse_dimension: first.coarse_dimension,
    })
}

/// Partition into `options.parts` subdomains and solve with `processes` ranks, each a thread of
/// this process.
pub fn solve_in_threads(
    problem: &PartitionedProblem,
    options: PartitionOptions,
    settings: KrylovSettings,
    processes: usize,
    kernel: &dyn ElementKernel,
    factory: &dyn SubdomainSolverFactory,
) -> DistributedResult<(DistributedSolution, PartitionQuality, SubdomainLayout)> {
    let (parts, quality) = partition_elements(&problem.mesh, options)?;
    let (inputs, layout) = extract_rank_inputs(problem, &parts, options.parts, settings)?;
    let processes = processes.clamp(1, options.parts);
    let plans = (0..processes)
        .map(|process| ProcessPlan::blocked(process, processes, options.parts, 1))
        .collect::<Vec<_>>();
    let communicators = ThreadCommunicator::group(processes);
    let outputs = std::thread::scope(|scope| {
        let handles = plans
            .iter()
            .zip(communicators)
            .map(|(plan, mut comm)| {
                let local = plan
                    .local_subdomains()
                    .into_iter()
                    .map(|subdomain| inputs[subdomain as usize].clone())
                    .collect::<Vec<_>>();
                scope.spawn(move || run_rank(&local, plan, kernel, factory, &mut comm))
            })
            .collect::<Vec<_>>();
        handles
            .into_iter()
            .map(|handle| {
                handle
                    .join()
                    .map_err(|_| DistributedError::Communication("rank thread panicked".into()))?
            })
            .collect::<DistributedResult<Vec<_>>>()
    })?;
    let solution = gather_outputs(
        &outputs,
        problem.mesh.node_count(),
        problem.field.dofs_per_node(),
        problem.mesh.element_count(),
    )?;
    Ok((solution, quality, layout))
}

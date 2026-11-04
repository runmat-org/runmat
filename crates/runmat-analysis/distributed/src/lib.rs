//! Domain decomposition of one finite element problem across cooperating workers.
//!
//! A solver mesh is graph-partitioned into non-overlapping element subdomains. Each rank owns
//! the degrees of freedom of its subdomain's nodes, assembles only its own elements plus one
//! overlapping element layer, factorizes its overlapping subdomain operator with a sparse
//! direct method, and takes part in a two-level additive Schwarz preconditioned conjugate
//! gradient iteration. The coarse space carries each subdomain's null-space modes (constants
//! for scalar fields, rigid-body modes for elasticity), so iteration counts stay bounded as the
//! number of subdomains grows. Ranks exchange only interface values with their neighbors and
//! small reductions; no rank ever holds the global operator.
//!
//! The crate is placement neutral: a [`Communicator`] connects ranks that run as threads in one
//! process, as worker processes on one machine, or as processes on several nodes.

mod comm;
mod driver;
mod error;
mod frame;
mod krylov;
mod local_solver;
mod partition;
mod problem;
mod rank;
mod subdomain;

pub use comm::{
    read_bytes, write_bytes, Communicator, TcpCommunicator, TcpRendezvous, ThreadCommunicator,
};
pub use driver::{gather_outputs, solve_in_threads, DistributedSolution};
pub use error::{DistributedError, DistributedResult, ElementKernelError};
pub use krylov::{IterationRecord, KrylovSettings, KrylovSummary};
pub use local_solver::{FaerCholeskyFactory, SubdomainFactorization, SubdomainSolverFactory};
pub use partition::{partition_elements, PartitionOptions, PartitionQuality};
pub use problem::{
    ElementKernel, FieldKind, PartitionMesh, PartitionedProblem, PointLoad, PrescribedDof,
};
pub use rank::{
    run_rank, NeighborPlan, ProcessPlan, RankOutput, RankTimings, SubdomainElementResults,
    SubdomainInput, SubdomainOutput,
};

#[cfg(test)]
mod tests;
pub use subdomain::{extract_rank_inputs, SubdomainLayout};

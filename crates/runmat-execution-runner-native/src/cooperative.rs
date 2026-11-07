//! Cooperative solution of one finite element problem by several worker processes.
//!
//! The coordinator partitions the solver mesh, starts one rank process per subdomain from the
//! native worker executable, hands each rank only its overlapping subdomain, and reassembles
//! the owned solution and element results the ranks return. Ranks talk to each other directly
//! over a TCP mesh; the coordinator only brokers the rendezvous.

use std::ffi::OsString;
use std::net::{IpAddr, SocketAddr};
use std::path::PathBuf;
use std::process::{Child, Command, Stdio};
use std::time::{Duration, Instant};

use runmat_analysis_distributed::{
    extract_rank_inputs, gather_outputs, partition_elements, read_bytes, run_rank, write_bytes,
    DistributedSolution, ElementKernel, ElementKernelError, FaerCholeskyFactory, FieldKind,
    KrylovSettings, PartitionOptions, PartitionQuality, PartitionedProblem, ProcessPlan,
    RankOutput, SubdomainInput, SubdomainLayout, TcpCommunicator, TcpRendezvous,
};
use runmat_analysis_fea::assembly::elements::solid::{
    elasticity_matrix, global_stiffness_matrix, tetrahedron10_global_stiffness_matrix,
    SolidMaterial, Tetrahedron10ElementGeometry, Tetrahedron4ElementGeometry,
};
use runmat_meshing_core::ElementOrder;

use crate::{NativeExecutionError, NativeExecutionResult};

pub const COOPERATIVE_RANK_ENV: &str = "RUNMAT_COOPERATIVE_RANK";
const COORDINATOR_ENV: &str = "RUNMAT_COOPERATIVE_COORDINATOR";
const TOKEN_ENV: &str = "RUNMAT_COOPERATIVE_TOKEN";
const RANKS_ENV: &str = "RUNMAT_COOPERATIVE_RANKS";
const BIND_ENV: &str = "RUNMAT_COOPERATIVE_BIND";

/// Physics and materials of the element operator, transported to every rank.
#[derive(Clone, Debug, PartialEq)]
pub enum CooperativeKernelSpec {
    /// Small-strain isotropic elasticity: per-material Young's modulus (Pa) and Poisson ratio.
    Elastic {
        order: ElementOrder,
        materials: Vec<(f64, f64)>,
    },
    /// Steady isotropic conduction: per-material conductivity (W/m/K).
    Conduction {
        order: ElementOrder,
        conductivities: Vec<f64>,
    },
}

impl CooperativeKernelSpec {
    pub fn field(&self) -> FieldKind {
        match self {
            Self::Elastic { .. } => FieldKind::Elastic,
            Self::Conduction { .. } => FieldKind::Scalar,
        }
    }

    fn order(&self) -> ElementOrder {
        match self {
            Self::Elastic { order, .. } | Self::Conduction { order, .. } => *order,
        }
    }

    pub fn encode(&self) -> Vec<u8> {
        let mut bytes = Vec::new();
        let (physics, values): (u8, Vec<f64>) = match self {
            Self::Elastic { materials, .. } => (
                1,
                materials
                    .iter()
                    .flat_map(|(modulus, ratio)| [*modulus, *ratio])
                    .collect(),
            ),
            Self::Conduction { conductivities, .. } => (2, conductivities.clone()),
        };
        bytes.push(physics);
        bytes.push(match self.order() {
            ElementOrder::Tet4 => 4,
            ElementOrder::Tet10 => 10,
        });
        bytes.extend_from_slice(&(values.len() as u64).to_le_bytes());
        for value in values {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        bytes
    }

    pub fn decode(bytes: &[u8]) -> NativeExecutionResult<Self> {
        let invalid =
            || NativeExecutionError::Protocol("cooperative kernel frame is invalid".into());
        if bytes.len() < 10 {
            return Err(invalid());
        }
        let order = match bytes[1] {
            4 => ElementOrder::Tet4,
            10 => ElementOrder::Tet10,
            _ => return Err(invalid()),
        };
        let count = u64::from_le_bytes(bytes[2..10].try_into().map_err(|_| invalid())?) as usize;
        if bytes.len() != 10 + count * 8 {
            return Err(invalid());
        }
        let values = bytes[10..]
            .chunks_exact(8)
            .map(|chunk| f64::from_le_bytes(chunk.try_into().expect("8 bytes")))
            .collect::<Vec<_>>();
        match bytes[0] {
            1 if values.len() % 2 == 0 => Ok(Self::Elastic {
                order,
                materials: values
                    .chunks_exact(2)
                    .map(|pair| (pair[0], pair[1]))
                    .collect(),
            }),
            2 => Ok(Self::Conduction {
                order,
                conductivities: values,
            }),
            _ => Err(invalid()),
        }
    }

    /// The element operator built from the shared element and quadrature owners.
    pub fn kernel(&self) -> Box<dyn ElementKernel> {
        match self {
            Self::Elastic { order, materials } => Box::new(ElasticKernel {
                order: *order,
                materials: materials
                    .iter()
                    .map(|&(youngs_modulus_pa, poisson_ratio)| SolidMaterial {
                        youngs_modulus_pa: runmat_types::physical::YoungsModulus::new(
                            youngs_modulus_pa,
                        ),
                        poisson_ratio: runmat_types::physical::PoissonRatio::new(poisson_ratio),
                        density_kg_per_m3: runmat_types::physical::MassDensity::new(1.0),
                    })
                    .collect(),
            }),
            Self::Conduction {
                order,
                conductivities,
            } => Box::new(ConductionKernel {
                order: *order,
                conductivities: conductivities.clone(),
            }),
        }
    }
}

fn node_count(order: ElementOrder) -> usize {
    match order {
        ElementOrder::Tet4 => 4,
        ElementOrder::Tet10 => 10,
    }
}

struct ElasticKernel {
    order: ElementOrder,
    materials: Vec<SolidMaterial>,
}

impl ElasticKernel {
    fn material(&self, index: u32) -> Result<SolidMaterial, String> {
        self.materials
            .get(index as usize)
            .copied()
            .ok_or_else(|| format!("element references unknown material {index}"))
    }
}

impl ElementKernel for ElasticKernel {
    fn field(&self) -> FieldKind {
        FieldKind::Elastic
    }

    fn nodes_per_element(&self) -> usize {
        node_count(self.order)
    }

    fn element_matrix(
        &self,
        material: u32,
        coordinates: &[[f64; 3]],
        matrix: &mut [f64],
    ) -> Result<(), ElementKernelError> {
        let material = self.material(material)?;
        match self.order {
            ElementOrder::Tet4 => {
                let nodes_m = coordinates
                    .try_into()
                    .map_err(|_| "Tet4 needs four nodes")?;
                let stiffness =
                    global_stiffness_matrix(material, Tetrahedron4ElementGeometry { nodes_m })
                        .map_err(|error| error.to_string())?;
                for (row, values) in stiffness.iter().enumerate() {
                    matrix[row * 12..(row + 1) * 12].copy_from_slice(values);
                }
            }
            ElementOrder::Tet10 => {
                let nodes_m = coordinates
                    .try_into()
                    .map_err(|_| "Tet10 needs ten nodes")?;
                let stiffness = tetrahedron10_global_stiffness_matrix(
                    material,
                    Tetrahedron10ElementGeometry { nodes_m },
                )
                .map_err(|error| error.to_string())?;
                for (row, values) in stiffness.iter().enumerate() {
                    matrix[row * 30..(row + 1) * 30].copy_from_slice(values);
                }
            }
        }
        Ok(())
    }

    fn result_width(&self) -> usize {
        6
    }

    /// Volume-averaged Cauchy stress (xx, yy, zz, yz, xz, xy) over the element quadrature.
    fn element_result(
        &self,
        material: u32,
        coordinates: &[[f64; 3]],
        solution: &[f64],
        result: &mut [f64],
    ) -> Result<(), ElementKernelError> {
        let d = elasticity_matrix(self.material(material)?).map_err(|error| error.to_string())?;
        let mut strain = [0.0f64; 6];
        let mut volume = 0.0;
        runmat_analysis_fea::topology::visit_tetrahedron_volume_quadrature(
            self.order,
            coordinates,
            |point| {
                let gradients = point.physical_shape_gradients();
                let weight = point.volume_weight_m3;
                let mut local = [0.0f64; 6];
                for (node, gradient) in gradients.iter().enumerate() {
                    let u = &solution[node * 3..node * 3 + 3];
                    local[0] += gradient[0] * u[0];
                    local[1] += gradient[1] * u[1];
                    local[2] += gradient[2] * u[2];
                    local[3] += gradient[2] * u[1] + gradient[1] * u[2];
                    local[4] += gradient[2] * u[0] + gradient[0] * u[2];
                    local[5] += gradient[1] * u[0] + gradient[0] * u[1];
                }
                for (total, value) in strain.iter_mut().zip(local) {
                    *total += weight * value;
                }
                volume += weight;
            },
        )
        .map_err(ElementKernelError::from)?
        .ok_or("element is degenerate or inverted")?;
        for (row, value) in result.iter_mut().enumerate().take(6) {
            *value = (0..6)
                .map(|column| d[row][column] * strain[column])
                .sum::<f64>()
                / volume;
        }
        Ok(())
    }
}

struct ConductionKernel {
    order: ElementOrder,
    conductivities: Vec<f64>,
}

impl ConductionKernel {
    fn conductivity(&self, index: u32) -> Result<f64, String> {
        self.conductivities
            .get(index as usize)
            .copied()
            .filter(|value| value.is_finite() && *value > 0.0)
            .ok_or_else(|| format!("element references invalid conductivity {index}"))
    }
}

impl ElementKernel for ConductionKernel {
    fn field(&self) -> FieldKind {
        FieldKind::Scalar
    }

    fn nodes_per_element(&self) -> usize {
        node_count(self.order)
    }

    fn element_matrix(
        &self,
        material: u32,
        coordinates: &[[f64; 3]],
        matrix: &mut [f64],
    ) -> Result<(), ElementKernelError> {
        let conductivity = self.conductivity(material)?;
        let n = node_count(self.order);
        runmat_analysis_fea::topology::visit_tetrahedron_volume_quadrature(
            self.order,
            coordinates,
            |point| {
                let gradients = point.physical_shape_gradients();
                let scale = conductivity * point.volume_weight_m3;
                for row in 0..n {
                    for column in 0..n {
                        matrix[row * n + column] += scale
                            * (gradients[row][0] * gradients[column][0]
                                + gradients[row][1] * gradients[column][1]
                                + gradients[row][2] * gradients[column][2]);
                    }
                }
            },
        )
        .map_err(ElementKernelError::from)?
        .ok_or_else(|| ElementKernelError::Detail("element is degenerate or inverted".into()))
    }

    fn result_width(&self) -> usize {
        3
    }

    /// Volume-averaged heat flux `-k grad T`.
    fn element_result(
        &self,
        material: u32,
        coordinates: &[[f64; 3]],
        solution: &[f64],
        result: &mut [f64],
    ) -> Result<(), ElementKernelError> {
        let conductivity = self.conductivity(material)?;
        let mut flux = [0.0f64; 3];
        let mut volume = 0.0;
        runmat_analysis_fea::topology::visit_tetrahedron_volume_quadrature(
            self.order,
            coordinates,
            |point| {
                for (node, gradient) in point.physical_shape_gradients().iter().enumerate() {
                    for axis in 0..3 {
                        flux[axis] -=
                            conductivity * gradient[axis] * solution[node] * point.volume_weight_m3;
                    }
                }
                volume += point.volume_weight_m3;
            },
        )
        .map_err(ElementKernelError::from)?
        .ok_or("element is degenerate or inverted")?;
        for axis in 0..3 {
            result[axis] = flux[axis] / volume;
        }
        Ok(())
    }
}

/// How rank processes are started for one cooperative solve.
#[derive(Clone, Debug)]
pub struct CooperativeLaunch {
    pub executable: PathBuf,
    pub worker_arguments: Vec<OsString>,
    /// Worker processes; the partition's subdomains are placed on them in contiguous blocks.
    pub processes: usize,
    pub threads_per_rank: u32,
    /// Interface every rank and the coordinator bind; loopback for one machine.
    pub bind: IpAddr,
    pub timeout: Duration,
}

impl CooperativeLaunch {
    /// Ranks are the current executable in its hidden execution-worker mode.
    pub fn for_current_executable(
        processes: usize,
        threads_per_rank: u32,
    ) -> NativeExecutionResult<Self> {
        Ok(Self {
            processes: processes.max(1),
            executable: std::env::current_exe().map_err(|error| {
                NativeExecutionError::Configuration(format!("current executable: {error}"))
            })?,
            worker_arguments: vec![runmat_process_host::HiddenMode::ExecutionWorker
                .marker()
                .into()],
            threads_per_rank: threads_per_rank.max(1),
            bind: IpAddr::from([127, 0, 0, 1]),
            timeout: Duration::from_secs(4 * 3_600),
        })
    }
}

/// Phase wall times of one cooperative solve, measured by the coordinator.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct CooperativeTimings {
    pub partition_seconds: f64,
    pub extraction_seconds: f64,
    pub launch_seconds: f64,
    pub ranks_seconds: f64,
    pub gather_seconds: f64,
    pub total_seconds: f64,
}

#[derive(Clone, Debug)]
pub struct CooperativeSolveReport {
    pub subdomains: usize,
    pub processes: usize,
    pub solution: DistributedSolution,
    pub quality: PartitionQuality,
    pub layout: SubdomainLayout,
    pub timings: CooperativeTimings,
}

/// Partition `problem` into `subdomains`, solve it with the launch's cooperating worker
/// processes and reassemble the owned pieces.
pub fn solve_cooperatively(
    problem: &PartitionedProblem,
    kernel: &CooperativeKernelSpec,
    subdomains: usize,
    settings: KrylovSettings,
    launch: &CooperativeLaunch,
) -> NativeExecutionResult<CooperativeSolveReport> {
    // This experimental rank launcher has no authenticated FEA child grant or
    // shared ledger. Refuse before partitioning or spawning any rank under an
    // admitted run; the standalone benchmark remains ordinary offline work.
    if runmat_simulation_usage::UsageContext::capture().is_some() {
        return Err(NativeExecutionError::UnsupportedMeteredCooperative);
    }
    let processes = launch.processes.clamp(1, subdomains.max(1));
    let started = Instant::now();
    if kernel.field() != problem.field
        || node_count(kernel.order()) != problem.mesh.nodes_per_element
    {
        return Err(NativeExecutionError::Protocol(
            "cooperative kernel differs from the problem field or element order".into(),
        ));
    }
    let mut timings = CooperativeTimings::default();
    let (parts, quality) = partition_elements(&problem.mesh, PartitionOptions::new(subdomains))
        .map_err(distributed)?;
    timings.partition_seconds = started.elapsed().as_secs_f64();
    let phase = Instant::now();
    let (inputs, layout) =
        extract_rank_inputs(problem, &parts, subdomains, settings).map_err(distributed)?;
    drop(parts);
    timings.extraction_seconds = phase.elapsed().as_secs_f64();

    let phase = Instant::now();
    let token = u64::from_le_bytes(
        uuid::Uuid::new_v4().as_bytes()[..8]
            .try_into()
            .expect("eight bytes"),
    );
    let rendezvous =
        TcpRendezvous::bind(SocketAddr::new(launch.bind, 0), token).map_err(distributed)?;
    let address = rendezvous.address().map_err(distributed)?;
    let mut children = Vec::with_capacity(processes);
    for rank in 0..processes {
        children.push(spawn_rank(launch, address, token, rank, processes)?);
    }
    let guard = ChildGuard(children);
    let controls = rendezvous.accept_ranks(processes).map_err(distributed)?;
    timings.launch_seconds = phase.elapsed().as_secs_f64();

    let phase = Instant::now();
    let kernel_frame = kernel.encode();
    let outputs = std::thread::scope(|scope| {
        let handles = controls
            .into_iter()
            .enumerate()
            .map(|(process, mut control)| {
                let kernel_frame = &kernel_frame;
                let inputs = &inputs;
                scope.spawn(move || -> NativeExecutionResult<RankOutput> {
                    let plan = ProcessPlan::blocked(
                        process,
                        processes,
                        subdomains,
                        launch.threads_per_rank,
                    );
                    control.set_read_timeout(Some(launch.timeout)).map_err(io)?;
                    write_bytes(&mut control, kernel_frame).map_err(distributed)?;
                    write_bytes(&mut control, &plan.encode()).map_err(distributed)?;
                    let local = plan.local_subdomains();
                    write_bytes(&mut control, &(local.len() as u64).to_le_bytes())
                        .map_err(distributed)?;
                    for subdomain in local {
                        write_bytes(&mut control, &inputs[subdomain as usize].encode())
                            .map_err(distributed)?;
                    }
                    std::io::Write::flush(&mut control).map_err(io)?;
                    let reply = read_bytes(&mut control).map_err(distributed)?;
                    match reply.split_first() {
                        Some((0, output)) => RankOutput::decode(output).map_err(distributed),
                        Some((2, encoded)) => {
                            let cause = serde_json::from_slice::<runmat_simulation_usage::LedgerError>(encoded)
                                .map_err(|error| NativeExecutionError::Protocol(format!("cooperative process {process} returned invalid operation-accounting cause: {error}")))?;
                            Err(NativeExecutionError::Usage(cause))
                        }
                        Some((_, message)) => Err(NativeExecutionError::Protocol(format!(
                            "cooperative process {process} failed: {}",
                            String::from_utf8_lossy(message)
                        ))),
                        None => Err(NativeExecutionError::Protocol(
                            "cooperative process returned an empty reply".into(),
                        )),
                    }
                })
            })
            .collect::<Vec<_>>();
        handles
            .into_iter()
            .map(|handle| {
                handle.join().map_err(|_| {
                    NativeExecutionError::Protocol("cooperative coordinator thread panicked".into())
                })?
            })
            .collect::<NativeExecutionResult<Vec<_>>>()
    })?;
    timings.ranks_seconds = phase.elapsed().as_secs_f64();
    guard.wait();

    let phase = Instant::now();
    let solution = gather_outputs(
        &outputs,
        problem.mesh.node_count(),
        problem.field.dofs_per_node(),
        problem.mesh.element_count(),
    )
    .map_err(distributed)?;
    timings.gather_seconds = phase.elapsed().as_secs_f64();
    timings.total_seconds = started.elapsed().as_secs_f64();
    Ok(CooperativeSolveReport {
        subdomains,
        processes,
        solution,
        quality,
        layout,
        timings,
    })
}

fn spawn_rank(
    launch: &CooperativeLaunch,
    coordinator: SocketAddr,
    token: u64,
    rank: usize,
    ranks: usize,
) -> NativeExecutionResult<Child> {
    Command::new(&launch.executable)
        .args(&launch.worker_arguments)
        .env(COOPERATIVE_RANK_ENV, rank.to_string())
        .env(COORDINATOR_ENV, coordinator.to_string())
        .env(TOKEN_ENV, token.to_string())
        .env(RANKS_ENV, ranks.to_string())
        .env(BIND_ENV, launch.bind.to_string())
        .env("RAYON_NUM_THREADS", launch.threads_per_rank.to_string())
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::inherit())
        .spawn()
        .map_err(|error| {
            NativeExecutionError::Configuration(format!("start cooperative rank {rank}: {error}"))
        })
}

/// Kills rank processes that are still running when the coordinator gives up.
struct ChildGuard(Vec<Child>);

impl ChildGuard {
    fn wait(mut self) {
        for child in &mut self.0 {
            let _ = child.wait();
        }
        self.0.clear();
    }
}

impl Drop for ChildGuard {
    fn drop(&mut self) {
        for child in &mut self.0 {
            let _ = child.kill();
            let _ = child.wait();
        }
    }
}

/// True when this process was started as a cooperative rank.
pub fn is_cooperative_rank_process() -> bool {
    std::env::var_os(COOPERATIVE_RANK_ENV).is_some()
}

/// Entry point of a rank process: join the mesh, run the rank and return its owned results.
pub fn run_cooperative_rank_from_env() -> NativeExecutionResult<()> {
    let variable = |name: &str| {
        std::env::var(name).map_err(|_| {
            NativeExecutionError::Configuration(format!("cooperative rank requires {name}"))
        })
    };
    let parse = |name: &str| -> NativeExecutionResult<u64> {
        variable(name)?
            .parse()
            .map_err(|_| NativeExecutionError::Configuration(format!("{name} is invalid")))
    };
    let rank = parse(COOPERATIVE_RANK_ENV)? as usize;
    let ranks = parse(RANKS_ENV)? as usize;
    let token = parse(TOKEN_ENV)?;
    let coordinator: SocketAddr = variable(COORDINATOR_ENV)?.parse().map_err(|_| {
        NativeExecutionError::Configuration("coordinator address is invalid".into())
    })?;
    let bind: IpAddr = variable(BIND_ENV)?
        .parse()
        .map_err(|_| NativeExecutionError::Configuration("bind address is invalid".into()))?;
    let (mut comm, mut control) =
        TcpCommunicator::connect(coordinator, token, rank, ranks, bind).map_err(distributed)?;
    let kernel = CooperativeKernelSpec::decode(&read_bytes(&mut control).map_err(distributed)?)?;
    let plan = ProcessPlan::decode(&read_bytes(&mut control).map_err(distributed)?)
        .map_err(distributed)?;
    let count = read_bytes(&mut control).map_err(distributed)?;
    let count = u64::from_le_bytes(
        count
            .as_slice()
            .try_into()
            .map_err(|_| NativeExecutionError::Protocol("subdomain count frame".into()))?,
    ) as usize;
    let inputs = (0..count)
        .map(|_| {
            SubdomainInput::decode(&read_bytes(&mut control).map_err(distributed)?)
                .map_err(distributed)
        })
        .collect::<NativeExecutionResult<Vec<_>>>()?;
    let kernel = kernel.kernel();
    let reply = match run_rank(
        &inputs,
        &plan,
        kernel.as_ref(),
        &FaerCholeskyFactory,
        &mut comm,
    ) {
        Ok(output) => {
            let mut bytes = vec![0u8];
            bytes.extend(output.encode());
            bytes
        }
        Err(error) => {
            let mut bytes = Vec::new();
            match error {
                runmat_analysis_distributed::DistributedError::Usage(cause) => {
                    bytes.push(2);
                    bytes.extend(serde_json::to_vec(&cause).map_err(|error| {
                        NativeExecutionError::Protocol(format!(
                            "cannot encode rank operation-accounting cause: {error}"
                        ))
                    })?);
                }
                detail => {
                    bytes.push(1);
                    bytes.extend(detail.to_string().into_bytes());
                }
            }
            bytes
        }
    };
    write_bytes(&mut control, &reply).map_err(distributed)?;
    std::io::Write::flush(&mut control).map_err(io)?;
    Ok(())
}

fn distributed(error: runmat_analysis_distributed::DistributedError) -> NativeExecutionError {
    match error {
        runmat_analysis_distributed::DistributedError::Usage(cause) => {
            NativeExecutionError::Usage(cause)
        }
        detail => NativeExecutionError::Protocol(detail.to_string()),
    }
}

fn io(error: std::io::Error) -> NativeExecutionError {
    NativeExecutionError::Protocol(error.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use runmat_simulation_usage::{OperationCount, UsageContext, UsageLedger};
    use std::sync::Arc;

    #[test]
    fn admitted_cooperative_route_refuses_before_partition_or_rank_launch() {
        let problem = PartitionedProblem {
            mesh: runmat_analysis_distributed::PartitionMesh {
                coordinates: Vec::new(),
                nodes_per_element: 0,
                element_nodes: Vec::new(),
            },
            element_materials: Vec::new(),
            field: FieldKind::Scalar,
            prescribed: Vec::new(),
            loads: Vec::new(),
        };
        let kernel = CooperativeKernelSpec::Conduction {
            order: ElementOrder::Tet4,
            conductivities: vec![1.0],
        };
        let launch = CooperativeLaunch {
            executable: PathBuf::from("unused-cooperative-rank"),
            worker_arguments: Vec::new(),
            processes: 1,
            threads_per_rank: 1,
            bind: IpAddr::V4(std::net::Ipv4Addr::LOCALHOST),
            timeout: Duration::from_secs(1),
        };
        let capacity = OperationCount::from_operations(10);
        let usage = UsageContext::new(Arc::new(UsageLedger::new(capacity, capacity).unwrap()));
        let admitted = usage.run_sync(|| {
            solve_cooperatively(&problem, &kernel, 1, KrylovSettings::default(), &launch)
        });
        assert!(matches!(
            admitted,
            Err(NativeExecutionError::UnsupportedMeteredCooperative)
        ));
        assert!(matches!(
            solve_cooperatively(&problem, &kernel, 1, KrylovSettings::default(), &launch),
            Err(NativeExecutionError::Protocol(_))
        ));
    }
}

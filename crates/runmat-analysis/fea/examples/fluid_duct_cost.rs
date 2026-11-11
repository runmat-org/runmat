//! Wall time of one steady canonical fluid solve on a straight square duct.
//!
//! `cargo run --release -p runmat-analysis-fea --example fluid_duct_cost -- <axial> <lateral> [threads] [speed m/s]`
//!
//! The duct is 300 mm long with a 10 mm square bore, split into `axial × lateral²` hexahedra of
//! six tetrahedra each. ISO VG 46 oil enters uniformly at x = 0 (0.2122 m/s by default), the
//! four side walls are no-slip and x = 300 mm is a 0 Pa outlet: the hydraulic-line card's
//! setup on a structured mesh. Prints the cell count, outer and linear iterations and the
//! wall time of preparation and solve.

use std::{env, error::Error, time::Instant};

use runmat_analysis_core::*;
use runmat_analysis_fea::{
    physics::cfd::canonical::{CanonicalFluidInput, CanonicalFluidPolicy, CanonicalFluidProblem},
    topology::{AnalysisProjection, FiniteVolumeProjectionPolicy, FiniteVolumeTopology},
};
use runmat_meshing_core::{fixtures, PersistentEntityId, PersistentEntityKind};

const LENGTH_M: f64 = 0.300;
const BORE_M: f64 = 0.010;

fn group(x: [f64; 3]) -> &'static str {
    if x[0].abs() < 1e-12 {
        "inlet"
    } else if (x[0] - LENGTH_M).abs() < 1e-12 {
        "outlet"
    } else {
        "wall"
    }
}

fn target(name: &str) -> Result<RegionTarget, Box<dyn Error>> {
    let mut target = RegionTarget::boundary(name);
    target.bind_resolved_entities(vec![entity(name)])?;
    Ok(target)
}

fn entity(name: &str) -> PersistentEntityId {
    PersistentEntityId::new(
        PersistentEntityKind::Face,
        name.to_string(),
        vec!["root".into()],
    )
}

fn main() -> Result<(), Box<dyn Error>> {
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .with_writer(std::io::stderr)
        .init();
    let args: Vec<String> = env::args().collect();
    let axial: usize = args.get(1).map_or(Ok(120), |a| a.parse())?;
    let lateral: usize = args.get(2).map_or(Ok(8), |a| a.parse())?;
    let threads: usize = args
        .get(3)
        .map_or(Ok(std::thread::available_parallelism()?.get()), |a| {
            a.parse()
        })?;
    let speed: f64 = args.get(4).map_or(Ok(0.2122), |a| a.parse())?;

    let mut mesh = fixtures::slab_bar_mesh(axial, lateral);
    for node in &mut mesh.topology.nodes {
        let [x, y, z] = node.coordinates_m.canonical_value();
        node.coordinates_m =
            runmat_types::physical::Quantity::new([x * LENGTH_M, y * BORE_M, z * BORE_M]);
    }
    let coordinates = mesh
        .topology
        .nodes
        .iter()
        .map(|n| (n.node_id, n.coordinates_m.canonical_value()))
        .collect::<std::collections::BTreeMap<_, _>>();
    for face in &mut mesh.topology.boundary_faces {
        let mut centroid = [0.0; 3];
        for id in &face.node_ids {
            for (axis, value) in coordinates[id].iter().enumerate() {
                centroid[axis] += value / face.node_ids.len() as f64;
            }
        }
        face.provenance = vec![entity(group(centroid))];
    }
    mesh.seal_canonical_digest()?;
    let source = AnalysisProjection::derive(&mesh)?;
    let topology = FiniteVolumeTopology::derive(
        &source,
        FiniteVolumeProjectionPolicy {
            max_cells: None,
            max_face_incidences: None,
            max_boundary_faces: None,
            max_scratch_bytes: None,
            max_traversal_operations: None,
        },
    )?;
    let material = CfdDomain {
        density: MassDensity::new(861.0),
        dynamic_viscosity: DynamicViscosity::new(0.0396),
        turbulence_intensity: TurbulenceIntensity::new(0.0),
        rheology: None,
        porous_zones: Vec::new(),
    };
    let constraints = vec![
        FluidConstraint {
            constraint_id: "supply".into(),
            kind: FluidConstraintKind::PrescribedVelocity {
                target: target("inlet")?,
                value: PrescribedValue::Constant([
                    Velocity::new(speed),
                    Velocity::new(0.0),
                    Velocity::new(0.0),
                ]),
            },
        },
        FluidConstraint {
            constraint_id: "bore".into(),
            kind: FluidConstraintKind::NoSlipWall {
                target: target("wall")?,
            },
        },
    ];
    let operators = vec![AnalysisBoundary {
        boundary_id: "discharge".into(),
        target: target("outlet")?,
        operator: BoundaryOperator::FluidOutletPressure {
            value: PrescribedValue::Constant(Pressure::new(0.0)),
        },
    }];
    println!(
        "cells={} faces={} threads={threads}",
        topology.cells.len(),
        topology.faces.len()
    );
    let result = runmat_cpu_execution::with_cpu_pool(threads, || {
        let started = Instant::now();
        let problem = CanonicalFluidProblem::prepare(CanonicalFluidInput {
            source: &source,
            topology: &topology,
            material: &material,
            constraints: &constraints,
            boundary_operators: &operators,
            signals: &[],
            excitations: &[],
            policy: CanonicalFluidPolicy {
                pressure_velocity_coupling: if env::var_os("QFS_SEGREGATED").is_some() {
                    runmat_analysis_fea::physics::cfd::canonical::PressureVelocityCoupling::Segregated
                } else {
                    runmat_analysis_fea::physics::cfd::canonical::PressureVelocityCoupling::Coupled
                },
                ..CanonicalFluidPolicy::default()
            },
        })
        .map_err(|e| e.to_string())?;
        let prepared = started.elapsed().as_secs_f64();
        let result = problem.solve().map_err(|e| e.to_string())?;
        Ok::<_, String>((prepared, started.elapsed().as_secs_f64() - prepared, result))
    })?;
    let (prepared, solved, result) = result?;
    let velocity = result.velocity_m_per_s.canonical_values();
    let peak = velocity
        .iter()
        .map(|v| (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt())
        .fold(0.0, f64::max);
    println!(
        "prepare_s={prepared:.2} solve_s={solved:.2} outer={} linear={} momentum={:e} continuity={:e} peak_speed={peak:.4}",
        result.outer_iterations,
        result.linear_iterations,
        result.residual.momentum_relative,
        result.residual.continuity_relative,
    );
    Ok(())
}

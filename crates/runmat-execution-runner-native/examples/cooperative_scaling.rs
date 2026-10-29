//! Strong and weak scaling of one finite element solve over cooperating worker processes.
//!
//! Builds a structured beam of tetrahedra, partitions it into a fixed number of subdomains and
//! solves it with 1..N worker processes started from this executable. For a fixed partition the
//! result is bitwise identical for every process count; the table reports the wall time of each
//! phase, the iteration count and the difference from the first process count.
//!
//! ```text
//! cooperative_scaling --physics elastic --order tet4 --cells 176x44x44 \
//!     --subdomains 16 --processes 1,2,4,8,16
//! cooperative_scaling --physics conduction --weak --cells 40x40x40 --processes 1,2,4,8
//! ```

use std::collections::HashMap;

use runmat_analysis_distributed::{
    FieldKind, KrylovSettings, PartitionMesh, PartitionedProblem, PointLoad, PrescribedDof,
};
use runmat_execution_runner_native::cooperative::{
    is_cooperative_rank_process, run_cooperative_rank_from_env, solve_cooperatively,
    CooperativeKernelSpec, CooperativeLaunch,
};
use runmat_meshing_core::ElementOrder;

struct Options {
    elastic: bool,
    order: ElementOrder,
    cells: [usize; 3],
    length: [f64; 3],
    subdomains: Option<usize>,
    processes: Vec<usize>,
    threads: u32,
    tolerance: f64,
    weak: bool,
}

fn parse() -> Options {
    let mut options = Options {
        elastic: true,
        order: ElementOrder::Tet4,
        cells: [64, 16, 16],
        length: [4.0, 1.0, 1.0],
        subdomains: None,
        processes: vec![1, 2, 4],
        threads: 1,
        tolerance: 1e-8,
        weak: false,
    };
    let arguments = std::env::args().skip(1).collect::<Vec<_>>();
    let mut index = 0;
    while index < arguments.len() {
        let value = arguments.get(index + 1).cloned().unwrap_or_default();
        match arguments[index].as_str() {
            "--physics" => options.elastic = value == "elastic",
            "--order" => {
                options.order = if value == "tet10" {
                    ElementOrder::Tet10
                } else {
                    ElementOrder::Tet4
                }
            }
            "--cells" => {
                let cells = value
                    .split('x')
                    .map(|part| part.parse().expect("cells"))
                    .collect::<Vec<usize>>();
                options.cells = [cells[0], cells[1], cells[2]];
                options.length = [
                    cells[0] as f64 / cells[1] as f64,
                    1.0,
                    cells[2] as f64 / cells[1] as f64,
                ];
            }
            "--subdomains" => options.subdomains = Some(value.parse().expect("subdomains")),
            "--processes" => {
                options.processes = value
                    .split(',')
                    .map(|part| part.parse().expect("processes"))
                    .collect()
            }
            "--threads" => options.threads = value.parse().expect("threads"),
            "--tolerance" => options.tolerance = value.parse().expect("tolerance"),
            "--weak" => {
                options.weak = true;
                index += 1;
                continue;
            }
            other => panic!("unknown argument {other}"),
        }
        index += 2;
    }
    options
}

/// Structured beam of `cells` hexahedra, each split into six tetrahedra; Tet10 adds one
/// midside node per unique edge in the shared corner-then-edge order (01, 12, 20, 03, 13, 23).
fn beam(cells: [usize; 3], length: [f64; 3], order: ElementOrder) -> PartitionMesh {
    let [nx, ny, nz] = cells;
    let index = |i: usize, j: usize, k: usize| (i + (nx + 1) * (j + (ny + 1) * k)) as u32;
    let mut coordinates = Vec::with_capacity((nx + 1) * (ny + 1) * (nz + 1));
    for k in 0..=nz {
        for j in 0..=ny {
            for i in 0..=nx {
                coordinates.push([
                    length[0] * i as f64 / nx as f64,
                    length[1] * j as f64 / ny as f64,
                    length[2] * k as f64 / nz as f64,
                ]);
            }
        }
    }
    let mut corners = Vec::with_capacity(nx * ny * nz * 24);
    for k in 0..nz {
        for j in 0..ny {
            for i in 0..nx {
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
                    corners.extend(tet.map(|corner| v[corner]));
                }
            }
        }
    }
    if order == ElementOrder::Tet4 {
        return PartitionMesh {
            coordinates,
            nodes_per_element: 4,
            element_nodes: corners,
        };
    }
    let mut midside = HashMap::<(u32, u32), u32>::new();
    let mut element_nodes = Vec::with_capacity(corners.len() / 4 * 10);
    for tet in corners.chunks_exact(4) {
        element_nodes.extend_from_slice(tet);
        for (a, b) in [(0, 1), (1, 2), (2, 0), (0, 3), (1, 3), (2, 3)] {
            let key = (tet[a].min(tet[b]), tet[a].max(tet[b]));
            let node = *midside.entry(key).or_insert_with(|| {
                let (p, q) = (coordinates[key.0 as usize], coordinates[key.1 as usize]);
                coordinates.push([
                    0.5 * (p[0] + q[0]),
                    0.5 * (p[1] + q[1]),
                    0.5 * (p[2] + q[2]),
                ]);
                (coordinates.len() - 1) as u32
            });
            element_nodes.push(node);
        }
    }
    PartitionMesh {
        coordinates,
        nodes_per_element: 10,
        element_nodes,
    }
}

/// Clamped cantilever with a unit tip shear, or a bar held at 300 K and 350 K at its ends.
fn problem(options: &Options, cells: [usize; 3]) -> PartitionedProblem {
    let mesh = beam(cells, options.length, options.order);
    let end = options.length[0];
    let mut prescribed = Vec::new();
    let mut loads = Vec::new();
    if options.elastic {
        let tip = mesh
            .coordinates
            .iter()
            .filter(|point| (point[0] - end).abs() < 1e-12)
            .count() as f64;
        for (node, point) in mesh.coordinates.iter().enumerate() {
            if point[0] == 0.0 {
                for component in 0..3 {
                    prescribed.push(PrescribedDof {
                        dof: node as u64 * 3 + component,
                        value: 0.0,
                    });
                }
            } else if (point[0] - end).abs() < 1e-12 {
                loads.push(PointLoad {
                    dof: node as u64 * 3 + 2,
                    value: -1.0e3 / tip,
                });
            }
        }
    } else {
        for (node, point) in mesh.coordinates.iter().enumerate() {
            if point[0] == 0.0 {
                prescribed.push(PrescribedDof {
                    dof: node as u64,
                    value: 300.0,
                });
            } else if (point[0] - end).abs() < 1e-12 {
                prescribed.push(PrescribedDof {
                    dof: node as u64,
                    value: 350.0,
                });
            }
        }
    }
    PartitionedProblem {
        element_materials: vec![0; mesh.element_count()],
        mesh,
        field: if options.elastic {
            FieldKind::Elastic
        } else {
            FieldKind::Scalar
        },
        prescribed,
        loads,
    }
}

fn main() {
    if is_cooperative_rank_process() {
        if let Err(error) = run_cooperative_rank_from_env() {
            eprintln!("cooperative rank failed: {error}");
            std::process::exit(2);
        }
        return;
    }
    let options = parse();
    let kernel = if options.elastic {
        CooperativeKernelSpec::Elastic {
            order: options.order,
            materials: vec![(200.0e9, 0.3)],
        }
    } else {
        CooperativeKernelSpec::Conduction {
            order: options.order,
            conductivities: vec![45.0],
        }
    };
    let settings = KrylovSettings {
        relative_tolerance: options.tolerance,
        ..KrylovSettings::default()
    };
    println!(
        "| mode | processes | subdomains | dofs | partition s | launch s | assembly s | factor s | coarse s | iterations | iterate s | total s | speedup | max diff | true residual |"
    );
    println!("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|");
    let mut reference: Option<(Vec<f64>, f64)> = None;
    let mut fixed = None;
    for &processes in &options.processes {
        let cells = if options.weak {
            let scale = (processes as f64).cbrt();
            [
                ((options.cells[0] as f64) * scale).round() as usize,
                ((options.cells[1] as f64) * scale).round() as usize,
                ((options.cells[2] as f64) * scale).round() as usize,
            ]
        } else {
            options.cells
        };
        let problem = fixed
            .take()
            .filter(|_| !options.weak)
            .unwrap_or_else(|| problem(&options, cells));
        let subdomains = options
            .subdomains
            .map(|count| {
                if options.weak {
                    count * processes
                } else {
                    count
                }
            })
            .unwrap_or(processes);
        let launch = CooperativeLaunch::for_current_executable(processes, options.threads)
            .expect("launch configuration");
        let report = solve_cooperatively(&problem, &kernel, subdomains, settings, &launch)
            .unwrap_or_else(|error| panic!("{processes} processes: {error}"));
        let solution = &report.solution;
        let (difference, speedup) = match &reference {
            Some((values, seconds)) if !options.weak => (
                values
                    .iter()
                    .zip(&solution.solution)
                    .map(|(a, b)| (a - b).abs())
                    .fold(0.0, f64::max),
                seconds / report.timings.total_seconds,
            ),
            Some((_, seconds)) => (f64::NAN, seconds / report.timings.total_seconds),
            None => {
                reference = Some((solution.solution.clone(), report.timings.total_seconds));
                (0.0, 1.0)
            }
        };
        if !options.elastic {
            // Exact linear temperature for the Dirichlet bar.
            let end = options.length[0];
            let exact = problem
                .mesh
                .coordinates
                .iter()
                .zip(&solution.solution)
                .map(|(point, value)| (value - (300.0 + 50.0 * point[0] / end)).abs())
                .fold(0.0, f64::max);
            eprintln!("{processes} processes: maximum error against the exact field {exact:.3e} K");
        }
        println!(
            "| {} | {} | {} | {} | {:.2} | {:.2} | {:.2} | {:.2} | {:.2} | {} | {:.2} | {:.2} | {:.2} | {:.1e} | {:.1e} |",
            if options.weak { "weak" } else { "strong" },
            report.processes,
            report.subdomains,
            problem.dof_count(),
            report.timings.partition_seconds + report.timings.extraction_seconds,
            report.timings.launch_seconds,
            solution.timings.assembly_seconds,
            solution.timings.factorization_seconds,
            solution.timings.coarse_seconds,
            solution.iterations,
            solution.timings.iteration_seconds,
            report.timings.total_seconds,
            speedup,
            difference,
            solution.true_relative_residual,
        );
        if !options.weak {
            fixed = Some(problem);
        }
    }
}

//! Reproducible Tet10 material-stiffness timing, not full assembly or study qualification.
//!
//! Run with `cargo run --release -p runmat-analysis-fea --example tet10_stiffness_cost -- 64 3`.
//! Peak RSS includes process startup and is not an isolated per-element allocation measure.

use std::{env, error::Error, hint::black_box, mem::MaybeUninit, time::Instant};

use runmat_analysis_fea::assembly::elements::solid::{
    tetrahedron10_global_stiffness_matrix, SolidMaterial, Tetrahedron10ElementGeometry,
};
use runmat_meshing_core::{fixtures::canonical_tetrahedron_solver_mesh, ElementOrder};

fn geometry(
    mut nodes_m: [[f64; 3]; 10],
    warp_m: f64,
    sample: usize,
) -> Tetrahedron10ElementGeometry {
    nodes_m[4][0] += warp_m + (sample % 16) as f64 * warp_m * 0.000_1;
    Tetrahedron10ElementGeometry { nodes_m }
}

fn run_case(
    name: &str,
    warp_m: f64,
    count: usize,
    repetitions: usize,
) -> Result<(), Box<dyn Error>> {
    let mesh = canonical_tetrahedron_solver_mesh(ElementOrder::Tet10);
    let base_nodes_m = std::array::from_fn(|index| mesh.topology.nodes[index].coordinates_m);
    let geometries = (0..count)
        .map(|sample| geometry(base_nodes_m, warp_m, sample))
        .collect::<Vec<_>>();
    let material = SolidMaterial {
        youngs_modulus_pa: 100.0,
        poisson_ratio: 0.25,
        density_kg_per_m3: 1.0,
    };
    let mut nanos_per_element = Vec::with_capacity(repetitions);
    let mut expected_checksum = None;
    for _ in 0..repetitions {
        let start = Instant::now();
        let mut checksum = 0.0;
        for &element in &geometries {
            let stiffness = tetrahedron10_global_stiffness_matrix(material, black_box(element))?;
            checksum += black_box(stiffness[0][0]);
        }
        let elapsed = start.elapsed();
        if !checksum.is_finite() || checksum <= 0.0 {
            return Err(format!("{name} produced an invalid stiffness checksum").into());
        }
        if let Some(expected) = expected_checksum {
            if checksum != expected {
                return Err(format!("{name} is not repeatable").into());
            }
        } else {
            expected_checksum = Some(checksum);
        }
        nanos_per_element.push(elapsed.as_nanos() as f64 / count as f64);
    }
    nanos_per_element.sort_by(f64::total_cmp);
    println!(
        "{name},{count},{repetitions},{:.0},{:.0},{:.12}",
        nanos_per_element[0],
        nanos_per_element[repetitions / 2],
        expected_checksum.expect("positive repetition count")
    );
    Ok(())
}

#[cfg(unix)]
fn process_peak_rss_bytes() -> Option<u64> {
    let mut usage = MaybeUninit::<libc::rusage>::uninit();
    // getrusage writes the complete structure before it is read below.
    if unsafe { libc::getrusage(libc::RUSAGE_SELF, usage.as_mut_ptr()) } != 0 {
        return None;
    }
    let raw = u64::try_from(unsafe { usage.assume_init() }.ru_maxrss).ok()?;
    if cfg!(target_os = "macos") {
        Some(raw)
    } else {
        raw.checked_mul(1024)
    }
}

#[cfg(not(unix))]
fn process_peak_rss_bytes() -> Option<u64> {
    None
}

fn main() -> Result<(), Box<dyn Error>> {
    let mut args = env::args().skip(1);
    let count: usize = args.next().ok_or("expected element count")?.parse()?;
    let repetitions: usize = args.next().ok_or("expected repetition count")?.parse()?;
    if args.next().is_some() || count == 0 || repetitions == 0 {
        return Err("expected two positive arguments: element count and repetitions".into());
    }
    println!("os={},arch={}", env::consts::OS, env::consts::ARCH);
    println!(
        "case,elements_per_trial,repetitions,min_ns_per_element,median_ns_per_element,checksum"
    );
    for (name, warp_m) in [("affine", 0.0), ("curved_0.1m", 0.1), ("curved_0.2m", 0.2)] {
        run_case(name, warp_m, count, repetitions)?;
    }
    if let Some(bytes) = process_peak_rss_bytes() {
        println!("process_peak_rss_bytes={bytes}");
    }
    Ok(())
}

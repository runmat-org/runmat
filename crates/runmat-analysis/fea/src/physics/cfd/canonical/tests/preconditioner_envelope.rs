//! The per-system preconditioner envelope bounds the AMG hierarchies the canonical solve
//! builds on a tetrahedral finite-volume cell operator: the pressure Laplacian with the default
//! options and a velocity component with the momentum options.
use super::*;
use crate::algebra::{MatrixProperties, OperatorRole, RealSparseAssembler};
use crate::solve::shared_sparse::{control, CsrArrays};
use runmat_sparse_solvers::precond::{Amg, AmgOptions};

fn cell_laplacian(
    topology: &crate::topology::FiniteVolumeTopology,
) -> crate::algebra::RealSparseMatrix {
    let n = topology.cells.len();
    let mut a = RealSparseAssembler::new(
        n,
        n,
        OperatorRole::EffectiveTangent,
        MatrixProperties::GENERAL,
    );
    for face in &topology.faces {
        match face.neighbor {
            Some(other) => {
                a.add(face.owner, face.owner, 1.0);
                a.add(other, other, 1.0);
                a.add(face.owner, other, -1.0);
                a.add(other, face.owner, -1.0);
            }
            None => a.add(face.owner, face.owner, 1.0),
        }
    }
    a.finish().unwrap()
}

#[test]
fn amg_hierarchies_stay_inside_the_preconditioner_envelope() {
    for n in [10, 24] {
        let f = case(n, ReferenceCase::Convective, false);
        let cells = f.topology.cells.len() as u64;
        // The reservation's tetrahedral bound: four face neighbours plus the diagonal.
        let reserved = cells * 5;
        let a = cell_laplacian(&f.topology);
        let arrays = CsrArrays::of(&a);
        let envelope = linear::PreconditionerEnvelope::for_counts(cells, reserved).unwrap();
        for options in [AmgOptions::default(), linear::momentum_amg_options()] {
            let amg = Amg::new(arrays.view(), options, control()).unwrap();
            let actual = amg.resident_bytes();
            eprintln!(
                "cells={cells} nnz={} reserved_nnz={} amg={} B ({:.0} B/cell) envelope={} B ({:.0} B/cell) ratio={:.2}",
                a.values().len(),
                reserved,
                actual,
                actual as f64 / cells as f64,
                envelope.resident_bytes_upper_bound,
                envelope.resident_bytes_upper_bound as f64 / cells as f64,
                envelope.resident_bytes_upper_bound as f64 / actual as f64,
            );
            assert!(actual <= envelope.resident_bytes_upper_bound);
        }
    }
}

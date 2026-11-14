//! The browser's numerical worker awaits capacity before each batch and quotes only while no
//! batch is outstanding. A curved Tet10 static solve through that contract matches the native
//! solve and leaves no reservation behind.

use super::*;
use crate::contracts::FeaRunError;
use runmat_meshing_core::CanonicalMeshingContract;
use runmat_simulation_usage::{OperationCount, UsageContext, UsageLedger};
use std::{future::Future, sync::Arc};

#[test]
fn curved_tet10_static_solve_runs_under_an_exclusive_capacity_owner() {
    let model = fixture_model(FixtureId::CantileverLinearStatic);
    let mut mesh = single_tetrahedron_solver_mesh_with_order_for_model(&model, ElementOrder::Tet10);
    let bent_node_id = mesh.topology.boundary_faces[1].node_ids[4];
    let bent_node = mesh
        .topology
        .nodes
        .iter_mut()
        .find(|node| node.node_id == bent_node_id)
        .unwrap();
    let mut coordinates = bent_node.coordinates_m.canonical_value();
    coordinates[1] = -0.05;
    bent_node.coordinates_m =
        runmat_types::physical::Quantity::<runmat_types::physical::kind::Position, _>::new(
            coordinates,
        );
    mesh.canonical_digest = runmat_meshing_core::StableDigest::ZERO;
    mesh.seal_canonical_digest().unwrap();
    mesh.validate_canonical().unwrap();
    let options = || LinearStaticSolveOptions {
        solver_mesh: Some(mesh.clone()),
        ..baseline_linear_static_options()
    };
    let native = crate::run_linear_static_with_options(&model, ComputeBackend::Cpu, options())
        .expect("curved Tet10 solves natively");

    let limit = OperationCount::from_operations(1 << 50);
    let ledger = Arc::new(UsageLedger::new(limit, limit).unwrap());
    let context = UsageContext::new(Arc::clone(&ledger));
    let owned = context.run_sync(|| {
        let ledger = Arc::clone(&ledger);
        let mut future = std::pin::pin!(crate::run_linear_static_with_options_tet4_owned(
            &model,
            ComputeBackend::Cpu,
            options(),
            usize::MAX,
            move |_phase, work| {
                let quoted = ledger
                    .exclusive_work_requirement(work)
                    .map(|_| ())
                    .map_err(FeaRunError::Usage);
                async move { quoted }
            },
        ));
        let mut waker = std::task::Context::from_waker(std::task::Waker::noop());
        loop {
            if let std::task::Poll::Ready(result) = future.as_mut().poll(&mut waker) {
                break result;
            }
        }
    });
    let owned: FeaRunResult = owned.expect("curved Tet10 solves under the exclusive owner");
    assert_vectors_within_tolerance(
        host_field(&owned, FEA_FIELD_STRUCTURAL_DISPLACEMENT),
        host_field(&native, FEA_FIELD_STRUCTURAL_DISPLACEMENT),
        ParityTolerance {
            abs: 1.0e-14,
            rel: 1.0e-12,
        },
    );
    let snapshot = ledger.snapshot().unwrap();
    assert!(snapshot.completed.assembly > OperationCount::ZERO);
    assert_eq!(snapshot.reserved, OperationCount::ZERO);
}

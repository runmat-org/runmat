//! Native nonzero physics and retained lifecycle evidence; not a production accuracy gate.
use super::*;
use crate::NativeAnalysisKernel;
use runmat_analysis_execution::{
    prepare_magnetostatic_task, AnalysisKernel, AnalysisOutputPolicy, MagnetostaticTaskPreparation,
    PreparedMagnetostaticTask,
};
use runmat_analysis_fea::execution::{AnalysisWorkKind, MagnetostaticSolverResult};
use runmat_meshing_core::ElementOrder;

mod accuracy;
mod fixture;
mod observations;
mod publication;
mod reference;

fn solve(task: &PreparedMagnetostaticTask) -> Vec<AnalysisKernelOutput> {
    NativeAnalysisKernel
        .execute(AnalysisKernelInvocation {
            kind: AnalysisWorkKind::SolveMagnetostatic,
            host: &task.host,
            inputs: &task.inputs,
            output_policy: task.output_policy,
        })
        .unwrap()
}
fn result(task: &PreparedMagnetostaticTask) -> MagnetostaticSolverResult {
    let outputs = solve(task);
    assert_eq!(outputs.len(), 3);
    let object = outputs
        .into_iter()
        .next()
        .unwrap()
        .into_prepared(
            task.output_policy.chunk_bytes(),
            task.output_policy.inventory_limits(),
        )
        .unwrap();
    MagnetostaticSolverResult::from_canonical_bytes(&object.payload()).unwrap()
}

#[test]
fn native_magnetic_nonzero_reference_converges_without_reference_solver_outputs() {
    let mu = runmat_analysis_fea::physics::magnetic::VACUUM_PERMEABILITY_H_PER_M.canonical_value();
    let exact_energy = reference::energy(mu, 255);
    let mut errors = Vec::new();
    for cells in [1, 2, 4] {
        let (_, mesh, task) = fixture::task(ElementOrder::Tet4, cells, 1.0, 1.0, 8192);
        let retained = result(&task);
        let response = retained.response();
        let projection = runmat_analysis_fea::topology::AnalysisProjection::derive(&mesh).unwrap();
        let mut potential = 0.0;
        let mut flux = 0.0;
        for sample in &response.samples(&projection).unwrap() {
            let [x, y, _] = sample.position_m;
            let (a, b) = reference::fields(x, y, mu, 1023);
            for axis in 0..3 {
                potential += sample.volume_weight_m3
                    * (sample.vector_potential_wb_per_m.canonical_value()[axis] - a[axis]).powi(2);
                flux += sample.volume_weight_m3
                    * (sample.flux_density_t.canonical_value()[axis] - b[axis]).powi(2);
            }
        }
        let error = (
            potential.sqrt() / mu,
            flux.sqrt() / mu,
            (response.magnetic_energy_j.canonical_value() - exact_energy).abs() / exact_energy,
        );
        eprintln!("native magnetic cells={cells} normalized A/B/energy errors={error:?}");
        errors.push(error);
    }
    for pair in errors.windows(2) {
        assert!(pair[1].0 < pair[0].0 / 1.4);
        assert!(pair[1].1 < pair[0].1 / 1.4);
        assert!(pair[1].2 < pair[0].2 / 1.4);
    }
    assert!(errors[2].0 < 0.02 && errors[2].1 < 0.10 && errors[2].2 < 0.25);
}

#[test]
fn native_magnetic_signed_material_scaling_and_affine_orders_are_transport_neutral() {
    let (_, _, base_task) = fixture::task(ElementOrder::Tet4, 2, 1.0, 1.0, 1024);
    let base = result(&base_task);
    for order in [ElementOrder::Tet4, ElementOrder::Tet10] {
        for (current, permeability) in [(-2.0, 3.0), (0.0, 1.0), (1.0, 1.0)] {
            let mut identity = None;
            for chunk in [1024, 8192] {
                let (workload, mesh, task) = fixture::task(order, 2, current, permeability, chunk);
                let cold = result(&task);
                cold.validate_for(&workload, &mesh).unwrap();
                let digest = cold.source_result_identity().unwrap();
                if let Some(previous) = identity {
                    assert_eq!(previous, digest);
                }
                identity = Some(digest);
                let scale = current * permeability;
                for (a, b) in cold
                    .response()
                    .edge_circulations_wb
                    .canonical_values()
                    .iter()
                    .zip(base.response().edge_circulations_wb.canonical_values())
                {
                    assert!((a - scale * b).abs() <= 1e-13 * scale.abs().max(1.0));
                }
                let expected = base.response().magnetic_energy_j * current * current * permeability;
                assert!(
                    (cold.response().magnetic_energy_j - expected)
                        .abs()
                        .canonical_value()
                        <= 1e-8 * expected.abs().canonical_value().max(f64::MIN_POSITIVE)
                );
                let outputs = solve(&task);
                let receipt = outputs.into_iter().nth(1).unwrap();
                let receipt = MagneticNumericalReceiptArtifact::from_canonical_bytes(
                    receipt.whole_payload().unwrap(),
                )
                .unwrap();
                receipt.validate_against(&workload, &cold, &mesh).unwrap();
            }
        }
    }
}

#[test]
fn native_magnetic_rejects_foreign_input_policy_and_missing_closure_before_solve() {
    let (mut workload, _, task) = fixture::task(ElementOrder::Tet4, 2, 1.0, 1.0, 1024);
    let invoke = |inputs: &[runmat_analysis_execution::PreparedAnalysisObjects], policy| {
        NativeAnalysisKernel.execute(AnalysisKernelInvocation {
            kind: AnalysisWorkKind::SolveMagnetostatic,
            host: &task.host,
            inputs,
            output_policy: policy,
        })
    };
    assert!(invoke(&task.inputs[..2], task.output_policy).is_err());
    assert!(invoke(
        &task.inputs,
        AnalysisOutputPolicy::new(8192, task.output_policy.inventory_limits()).unwrap()
    )
    .is_err());
    workload.relative_residual_target *= 0.1;
    let mut foreign = task.inputs.clone();
    foreign[0] = prepare_analysis_objects(
        AnalysisNumericalObjectKind::MagnetostaticWorkload,
        workload.identity().unwrap().digest(),
        &workload.canonical_bytes().unwrap(),
        1024,
        ObjectNamespace::InputValue,
        task.output_policy.inventory_limits(),
    )
    .unwrap();
    assert!(invoke(&foreign, task.output_policy).is_err());
}

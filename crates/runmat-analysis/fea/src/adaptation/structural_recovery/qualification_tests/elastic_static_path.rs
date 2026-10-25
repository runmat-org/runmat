//! The large static path (elastic rigidity proof + AMG-PCG, no global factor) agrees with the
//! proving factorization on a cantilever both can solve.
use super::*;
use crate::{contracts::StaticOperatorProof, solve::preconditioner::SpdPreconditionerKind};

fn solve(
    model: &runmat_analysis_core::AnalysisModel,
    mesh: &SolverMeshArtifact,
    preconditioner: SpdPreconditionerKind,
) -> crate::FeaRunResult {
    crate::run_linear_static_with_options(
        model,
        ComputeBackend::Cpu,
        LinearStaticSolveOptions {
            solver_mesh: Some(mesh.clone()),
            require_solver_mesh_for_solid: true,
            preconditioner_kind: preconditioner,
            ..LinearStaticSolveOptions::default()
        },
    )
    .expect("the cantilever must solve")
}

pub(super) fn qualify() {
    let model = fixture_model(FixtureId::CantileverLinearStatic);
    for order in [ElementOrder::Tet4, ElementOrder::Tet10] {
        let mesh = structured_cantilever(3, order);
        let direct = solve(&model, &mesh, SpdPreconditionerKind::Auto);
        let elastic = solve(&model, &mesh, SpdPreconditionerKind::Amg);
        let direct_convergence = direct.structural_static_convergence.as_ref().unwrap();
        let elastic_convergence = elastic.structural_static_convergence.as_ref().unwrap();
        assert_eq!(direct_convergence.operator_proof, StaticOperatorProof::SparseCholesky);
        assert_eq!(elastic_convergence.operator_proof, StaticOperatorProof::ElasticRigidity);
        assert_eq!(elastic.solver_method, crate::SolverMethod::AssembledPcg);
        assert_eq!(elastic.preconditioner, crate::SolverPreconditioner::Amg);
        let relative = elastic_convergence.relative_residual.unwrap();
        assert!(elastic_convergence.converged && relative <= 1.0e-8, "{relative}");
        let field = |run: &crate::FeaRunResult| {
            run.field(FEA_FIELD_STRUCTURAL_DISPLACEMENT)
                .unwrap()
                .as_host_f64()
                .unwrap()
                .to_vec()
        };
        let (a, b) = (field(&direct), field(&elastic));
        let scale = a.iter().map(|v| v.abs()).fold(0.0_f64, f64::max);
        let difference = a
            .iter()
            .zip(&b)
            .map(|(x, y)| (x - y).abs())
            .fold(0.0_f64, f64::max);
        eprintln!(
            "elastic static path {order:?}: nodes {} iterations {} relative residual {relative:.3e} max |u| {scale:.6e} max |du| {difference:.3e}",
            mesh.topology.nodes.len(),
            elastic_convergence.iterations
        );
        assert!(difference <= 1.0e-6 * scale, "{difference} vs {scale}");
    }
}

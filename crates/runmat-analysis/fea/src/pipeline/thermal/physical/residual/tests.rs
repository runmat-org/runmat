use super::*;
use crate::{
    algebra::{
        AssembledRealSystem, CanonicalRealOperators, MatrixProperties, OperatorRole,
        RealSparseAssembler, RealSparseMatrix,
    },
    solve::{
        backend::{cpu_reference::CpuReferenceBackend, kind::LinearAlgebraBackendKind},
        linear::solve_operator_system,
        preconditioner::SpdPreconditionerKind,
    },
};

fn solve(scale: f64, homogeneous: bool) -> LinearSolveResult {
    let mut k = RealSparseAssembler::square(
        2,
        OperatorRole::Stiffness,
        MatrixProperties::SYMMETRIC_POSITIVE_DEFINITE,
    );
    for (i, j, v) in [(0, 0, 4.0), (0, 1, -1.0), (1, 0, -1.0), (1, 1, 3.0)] {
        k.add(i, j, v);
    }
    let mass = RealSparseMatrix::diagonal(
        vec![1.0; 2],
        OperatorRole::Mass,
        MatrixProperties::SYMMETRIC_POSITIVE_DEFINITE,
    )
    .unwrap();
    let damping = RealSparseMatrix::diagonal(
        vec![0.0; 2],
        OperatorRole::Damping,
        MatrixProperties::SYMMETRIC_POSITIVE_SEMIDEFINITE,
    )
    .unwrap();
    let system = AssembledRealSystem::try_new(
        CanonicalRealOperators::try_new(k.finish().unwrap(), mass, damping).unwrap(),
        vec![false; 2],
        if homogeneous {
            vec![0.0; 2]
        } else {
            vec![scale, 2.0 * scale]
        },
    )
    .unwrap();
    let guess = if homogeneous {
        [scale, -scale]
    } else {
        [scale * (5.0 / 11.0 + 1e-10), scale * (9.0 / 11.0 - 1e-10)]
    };
    solve_operator_system(
        &system,
        SpdPreconditionerKind::Jacobi,
        LinearAlgebraBackendKind::CpuReference,
        &CpuReferenceBackend,
        Some(&guess),
    )
    .unwrap()
}

#[test]
fn small_power_cannot_make_the_same_equation_defect_look_more_accurate() {
    let mut values = Vec::new();
    for exponent in [-40, 0, 40] {
        let scale = 2.0_f64.powi(exponent);
        let result = solve(scale, false);
        assert!(result.converged);
        assert_eq!(result.iterations, 0); // Deliberate nonzero accepted warm defect.
        let residual = linear_residual(&result).unwrap();
        assert!(residual > 1e-11 && residual < 1e-8);
        let actual_equation_residual = ((scale - 4.0 * result.solution[0] + result.solution[1])
            .powi(2)
            + (2.0 * scale + result.solution[0] - 3.0 * result.solution[1]).powi(2))
        .sqrt();
        let expected = actual_equation_residual / (5.0_f64.sqrt() * scale);
        assert!((residual - expected).abs() < 1e-15);
        values.push(residual);
    }
    assert_eq!(values[0], values[1]);
    assert_eq!(values[1], values[2]);
}

#[test]
fn homogeneous_warm_equation_uses_original_solver_scale_and_missing_evidence_rejects() {
    let mut result = solve(2.0_f64.powi(-40), true);
    assert!(result.converged);
    assert!(linear_residual(&result).unwrap() <= 1e-8);
    for bad in [None, Some(f64::NAN), Some(-1.0), Some(f64::INFINITY)] {
        result.relative_residual = bad;
        assert!(linear_residual(&result).is_err());
    }
}

#[test]
fn public_small_power_fields_and_receipts_keep_original_relative_evidence() {
    use crate::execution::ThermalNumericalReceiptArtifact;
    use runmat_analysis_core::{
        AnalysisBoundary, BoundaryOperator, HeatFlux, PrescribedValue, RegionTarget, Temperature,
    };
    for quadratic in [false, true] {
        let mesh = crate::physics::thermal::qualification::mesh(quadratic);
        let mut records = Vec::new();
        for exponent in [-30, 0, 30] {
            let scale = 2.0_f64.powi(exponent);
            let mut workload = crate::physics::thermal::qualification::workload();
            workload.model.boundaries.clear();
            workload.model.materials[0]
                .thermal
                .as_mut()
                .unwrap()
                .conductivity_w_per_mk =
                runmat_types::physical::ThermalConductivity::new(2.0 * scale);
            crate::physics::thermal::qualification::left_temperature(
                &mut workload,
                PrescribedValue::Constant(Temperature::new(400.0)),
            );
            workload.model.boundaries.push(AnalysisBoundary {
                boundary_id: "scaled-incoming-power".into(),
                target: RegionTarget::boundary(runmat_meshing_core::fixtures::fixture_face(
                    "slab_right",
                )),
                operator: BoundaryOperator::ThermalHeatFlux {
                    value: PrescribedValue::Constant(HeatFlux::new(0.37 * scale)),
                },
            });
            let result = workload.execute(&mesh).unwrap();
            result.validate_complete().unwrap();
            for (node, &value) in mesh
                .topology
                .nodes
                .iter()
                .zip(result.temperature_snapshots[0].as_host_f64().unwrap())
            {
                // Independent 1D conduction reference; scaling k and q leaves T unchanged.
                assert!(
                    (value - (400.0 + 0.185 * node.coordinates_m.canonical_value()[0])).abs()
                        < 1e-7
                );
            }
            let receipt = ThermalNumericalReceiptArtifact::new(&workload, &result).unwrap();
            let cold = ThermalNumericalReceiptArtifact::from_canonical_bytes(
                &receipt.canonical_bytes().unwrap(),
            )
            .unwrap();
            assert_eq!(cold, receipt);
            assert_eq!(receipt.schema_version, 2);
            let mut legacy = receipt.canonical_bytes().unwrap();
            let prefix = b"runmat-thermal-numerical-receipt/v1\0";
            assert!(legacy.starts_with(b"runmat-thermal-numerical-receipt/v2\0"));
            legacy[..prefix.len()].copy_from_slice(prefix);
            assert!(ThermalNumericalReceiptArtifact::from_canonical_bytes(&legacy).is_err());
            let mut old_schema = serde_json::to_value(&receipt).unwrap();
            old_schema["schema_version"] = serde_json::json!(1);
            assert!(serde_json::from_value::<ThermalNumericalReceiptArtifact>(old_schema).is_err());

            let residual = result.residual_norms[0];
            assert_eq!(receipt.receipt().maximum_relative_linear_residual, residual);
            assert!(residual <= workload.residual_target);
            records.push(residual);
        }
        assert_eq!(records[0], records[1]);
        assert_eq!(records[1], records[2]);
    }
}

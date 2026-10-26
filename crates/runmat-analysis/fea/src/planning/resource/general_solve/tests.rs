use super::*;
use crate::{
    algebra::{MatrixProperties, OperatorRole, RealSparseAssembler},
    solve::general::solve_general_system,
};
fn policy() -> GeneralSolvePolicy {
    GeneralSolvePolicy::bounded(1e-11)
}
fn matrix() -> RealSparseMatrix {
    let mut a =
        RealSparseAssembler::square(3, OperatorRole::EffectiveTangent, MatrixProperties::GENERAL);
    for (i, row) in [[4., 2., -1.], [-3., 2., 1.], [1., -2., 5.]]
        .iter()
        .enumerate()
    {
        for (j, &v) in row.iter().enumerate() {
            a.add(i, j, v);
        }
    }
    a.finish().unwrap()
}
fn demand() -> GeneralSparseSolveResourceDemand {
    GeneralSparseSolveResourceDemand::derive(
        SparsePatternStatistics {
            rows: 3,
            columns: 3,
            nonzeros: 9,
            maximum_row_nonzeros: 3,
        },
        policy(),
        Count(7),
        GeneralSolveEnvelope::derive(&matrix(), policy()).unwrap(),
    )
    .unwrap()
}
#[test]
fn cold_demand_rederives_full_controls_and_solves_independent_nonsymmetric_reference() {
    let d = demand();
    let bytes = serde_json::to_vec(&d).unwrap();
    let cold: GeneralSparseSolveResourceDemand = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(d, cold);
    cold.validate().unwrap();
    let a = matrix();
    let admission = cold.admit_operator(&a, policy()).unwrap();
    // Independent dense arithmetic: A*[1.25,-0.5,2] = [2,-2.75,12.25].
    let solved =
        solve_general_system(&a, &[2., -2.75, 12.25], cold.controls().policy(), admission).unwrap();
    for (actual, expected) in solved.solution.iter().zip([1.25, -0.5, 2.]) {
        assert!((actual - expected).abs() < 1e-10);
    }
    assert_eq!(
        cold.scalar_operations_upper_bound().unwrap(),
        7 * admission.scalar_operations
    );
}
#[test]
fn changed_controls_and_shape_reject_even_when_workspace_matches() {
    let d = demand();
    let mut changed = policy();
    changed.relative_tolerance *= 0.5;
    assert_eq!(
        GeneralSolveEnvelope::derive(&matrix(), changed).unwrap(),
        GeneralSolveEnvelope::derive(&matrix(), policy()).unwrap()
    );
    assert_eq!(
        d.admit_operator(&matrix(), changed).unwrap_err(),
        GeneralSolveError::Admission
    );
    let mut encoded = serde_json::to_value(d).unwrap();
    encoded["dimension"] = serde_json::json!(2);
    let foreign: GeneralSparseSolveResourceDemand = serde_json::from_value(encoded).unwrap();
    assert!(foreign.admit_operator(&matrix(), policy()).is_err());
    let mut encoded = serde_json::to_value(d).unwrap();
    encoded["controls"]["contraction_limit"] = serde_json::json!(1.0);
    assert!(serde_json::from_value::<GeneralSparseSolveResourceDemand>(encoded).is_err());
    for field in ["relative_tolerance", "contraction_limit"] {
        let mut encoded = serde_json::to_value(d).unwrap();
        encoded["controls"].as_object_mut().unwrap().remove(field);
        assert!(serde_json::from_value::<GeneralSparseSolveResourceDemand>(encoded).is_err());
    }
}
#[test]
fn underreservation_and_cumulative_work_overflow_reject_before_execution() {
    for memory in [true, false] {
        let mut encoded = serde_json::to_value(demand()).unwrap();
        let field = if memory {
            "resident_bytes"
        } else {
            "scalar_operations"
        };
        let value = encoded["admission"][field].as_u64().unwrap();
        encoded["admission"][field] = serde_json::json!(value - 1);
        let short: GeneralSparseSolveResourceDemand = serde_json::from_value(encoded).unwrap();
        assert_eq!(short.validate().unwrap_err(), GeneralSolveError::Admission);
        assert_eq!(
            short.admit_operator(&matrix(), policy()).unwrap_err(),
            GeneralSolveError::Admission
        );
    }
    let mut encoded = serde_json::to_value(demand()).unwrap();
    encoded["maximum_solves"] = serde_json::json!(u64::MAX);
    let overflow: GeneralSparseSolveResourceDemand = serde_json::from_value(encoded).unwrap();
    assert_eq!(
        overflow.validate().unwrap_err(),
        GeneralSolveError::Overflow
    );
    let mut invalid = policy();
    invalid.relative_tolerance = f64::NAN;
    assert!(GeneralSolveResourceControls::try_from(invalid).is_err());
    invalid = policy();
    invalid.contraction_limit = f64::INFINITY;
    assert!(GeneralSolveResourceControls::try_from(invalid).is_err());
}

#[test]
fn actual_support_and_zero_rhs_singularity_are_not_authorized_by_portable_demand() {
    for (field, value) in [("nonzero_upper_bound", 8), ("maximum_row_nonzeros", 2)] {
        let mut encoded = serde_json::to_value(demand()).unwrap();
        encoded[field] = serde_json::json!(value);
        let actual = if field == "maximum_row_nonzeros" {
            encoded["nonzero_upper_bound"] = serde_json::json!(6);
            let mut builder = RealSparseAssembler::square(
                3,
                OperatorRole::EffectiveTangent,
                MatrixProperties::GENERAL,
            );
            for (i, row) in [[1., 2., 3.], [0., 1., 0.], [0., 0., 1.]]
                .iter()
                .enumerate()
            {
                for (j, &v) in row.iter().enumerate() {
                    builder.add(i, j, v);
                }
            }
            builder.finish().unwrap()
        } else {
            matrix()
        };
        let undersupported: GeneralSparseSolveResourceDemand =
            serde_json::from_value(encoded).unwrap();
        undersupported.validate().unwrap();
        assert_eq!(
            undersupported
                .admit_operator(&actual, policy())
                .unwrap_err(),
            GeneralSolveError::Admission
        );
    }
    let mut builder =
        RealSparseAssembler::square(3, OperatorRole::EffectiveTangent, MatrixProperties::GENERAL);
    for (i, row) in [[1., 2., 3.], [2., 4., 6.], [0., 0., 0.]]
        .iter()
        .enumerate()
    {
        for (j, &v) in row.iter().enumerate() {
            builder.add(i, j, v);
        }
    }
    let singular = builder.finish().unwrap();
    let d = demand();
    let admission = d.admit_operator(&singular, policy()).unwrap();
    assert!(matches!(
        solve_general_system(&singular, &[0.; 3], policy(), admission),
        Err(GeneralSolveError::NonsingularityNotEstablished(_))
    ));
}

use super::*;

pub(crate) fn closed_authored() -> (FluidSteadyWorkload, SolverMeshArtifact) {
    let (mut workload, _) = authored();
    let mesh = runmat_meshing_core::fixtures::slab_mesh(2);
    let outlet = workload.model.boundaries.pop().unwrap();
    workload
        .model
        .constraints
        .push(AnalysisConstraint::Fluid(FluidConstraint {
            constraint_id: "closed-wall".into(),
            kind: FluidConstraintKind::NoSlipWall {
                target: outlet.target.clone(),
            },
        }));
    workload
        .model
        .constraints
        .push(AnalysisConstraint::Fluid(FluidConstraint {
            constraint_id: "pressure-gauge".into(),
            kind: FluidConstraintKind::PressureReference {
                target: outlet.target,
                value: PrescribedValue::Constant(Pressure::new(101325.0)),
            },
        }));
    (workload, mesh)
}

#[test]
fn original_closed_wall_fields_receipt_and_gauge_survive_cold_workload_admission() {
    let (w, m) = closed_authored();
    let result = w.execute(&m).unwrap();
    assert!(result
        .original()
        .pressure_pa
        .canonical_values()
        .iter()
        .all(|p| *p == 101325.0));
    assert!(result
        .original()
        .pressure_fluctuation_pa
        .canonical_values()
        .iter()
        .all(|p| *p == 0.0));
    let bytes = result.canonical_bytes().unwrap();
    let cold =
        FluidSteadySolverResult::from_canonical_bytes_for(&bytes, Digest::sha256(&bytes), &w, &m)
            .unwrap();
    assert_eq!(cold, result);
    let receipt = crate::execution::FluidNumericalReceiptArtifact::new(&w, &cold, &m).unwrap();
    receipt.validate_against(&w, &cold, &m).unwrap();
    let mut foreign = w.clone();
    let AnalysisConstraint::Fluid(c) = foreign.model.constraints.last_mut().unwrap() else {
        unreachable!()
    };
    let FluidConstraintKind::PressureReference { value, .. } = &mut c.kind else {
        unreachable!()
    };
    *value = PrescribedValue::Constant(Pressure::new(101326.0));
    assert!(result.validate_for(&foreign, &m).is_err());
}

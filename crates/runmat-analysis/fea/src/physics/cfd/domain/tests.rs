use super::*;
use crate::fixtures::{fixture_model, FixtureId};
use runmat_analysis_core::{FluidConstraint, PrescribedValue, RegionTarget, Velocity};

#[test]
fn reduced_cfd_rejects_every_signed_vector_constraint() {
    let mut model = fixture_model(FixtureId::CantileverLinearStatic);
    for kind in [
        FluidConstraintKind::PrescribedVelocity {
            target: RegionTarget::authored(runmat_meshing_core::fixtures::fixture_face("inlet")),
            value: PrescribedValue::Constant([-2.0, 3.0, -4.0].map(Velocity::new)),
        },
        FluidConstraintKind::MovingWall {
            target: RegionTarget::authored(runmat_meshing_core::fixtures::fixture_face("wall")),
            velocity: PrescribedValue::Constant([0.0, -2.0, 0.0].map(Velocity::new)),
        },
    ] {
        model.constraints = vec![AnalysisConstraint::Fluid(FluidConstraint {
            constraint_id: "exact-boundary".into(),
            kind,
        })];
        let error = validate_authored_boundary_conditions(&model).unwrap_err();
        assert!(error.contains("canonical finite-volume execution is required"));
        assert!(realize_velocity_schedule(
            &model,
            PhysicsFamily::Fluid,
            AnalysisProcedure::Static,
            1
        )
        .is_err());
    }
}

#[test]
fn reduced_cfd_never_reduces_vector_excitation_to_unsigned_maximum_speed() {
    let mut model = fixture_model(FixtureId::CantileverLinearStatic);
    model.constraints.clear();
    let action = model.excitations[0].excitation().clone();
    for excitation in [
        ExcitationAction::InletVelocity(action.clone()),
        ExcitationAction::WallMotion(action),
    ] {
        model.excitations = vec![excitation];
        let error =
            realize_velocity_schedule(&model, PhysicsFamily::Fluid, AnalysisProcedure::Static, 1)
                .unwrap_err();
        assert!(error.contains("directional, framed, per-region"));
    }
}

#[test]
fn reduced_cfd_does_not_alias_a_gauge_to_an_outlet_or_ignore_exact_outlet_authority() {
    use runmat_analysis_core::{AnalysisBoundary, BoundaryOperator, Pressure};
    let mut model = fixture_model(FixtureId::CantileverLinearStatic);
    model.constraints = vec![AnalysisConstraint::Fluid(FluidConstraint {
        constraint_id: "datum".into(),
        kind: FluidConstraintKind::PressureReference {
            target: RegionTarget::authored(runmat_meshing_core::fixtures::fixture_face("datum")),
            value: PrescribedValue::Constant(Pressure::new(0.0)),
        },
    })];
    assert!(validate_authored_boundary_conditions(&model)
        .unwrap_err()
        .contains("not a physical outlet"));
    model.constraints.clear();
    model.boundaries = vec![AnalysisBoundary {
        boundary_id: "outlet".into(),
        target: RegionTarget::authored(runmat_meshing_core::fixtures::fixture_face("outlet")),
        operator: BoundaryOperator::FluidOutletPressure {
            value: PrescribedValue::Constant(Pressure::new(0.0)),
        },
    }];
    assert!(validate_authored_boundary_conditions(&model)
        .unwrap_err()
        .contains("exact physical outlet"));
}

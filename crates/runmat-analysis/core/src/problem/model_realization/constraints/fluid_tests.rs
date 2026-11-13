use super::*;
use crate::{AuthoredRegionSource, AuthoredTargetCardinality, PrescribedValue, SignalId, Velocity};
use runmat_geometry_core::{
    GeometryDefinitionRef, GeometryRegionRef, GeometryRegionRole, PersistentEntityId,
    PersistentEntityKind, ResolvedGeometryRegion,
};
use std::collections::BTreeMap;

fn authored_target(region: &str) -> AuthoredConstraintTarget {
    AuthoredConstraintTarget {
        region: AuthoredRegionSource::Geometry(GeometryRegionRef {
            module_path: "flow.fea".into(),
            geometry_id: "fluid".into(),
            region_id: region.into(),
            role: GeometryRegionRole::Face,
        }),
        cardinality: AuthoredTargetCardinality::One,
    }
}

#[test]
fn realization_keeps_exact_per_region_velocity_components_and_original_signal_reference() {
    let geometry = GeometryDefinitionRef {
        module_path: "flow.fea".into(),
        geometry_id: "fluid".into(),
    };
    let entity =
        |id: &str| PersistentEntityId::new(PersistentEntityKind::Face, id.into(), Vec::new());
    let regions = BTreeMap::from([
        (
            "inlet".into(),
            ResolvedGeometryRegion {
                role: GeometryRegionRole::Face,
                entity_ids: vec![entity("face-inlet")],
            },
        ),
        (
            "wall".into(),
            ResolvedGeometryRegion {
                role: GeometryRegionRole::Face,
                entity_ids: vec![entity("face-wall")],
            },
        ),
    ]);
    let targets = TargetResolver::new(&geometry, &regions);
    let components = [-2.0, 3.0, -4.0].map(Velocity::new);
    let source = AuthoredConstraint {
        constraint_id: "inlet-drive".into(),
        kind: AuthoredConstraintKind::FluidVelocity {
            target: authored_target("inlet"),
            value: PrescribedValue::Constant(components),
        },
    };
    let AnalysisConstraint::Fluid(realized) = realize_constraint(&source, &targets).unwrap() else {
        panic!("fluid");
    };
    let FluidConstraintKind::PrescribedVelocity { target, value } = realized.kind else {
        panic!("vector");
    };
    assert_eq!(target.region_id, "inlet");
    assert_eq!(target.resolved_entities(), &[entity("face-inlet")]);
    assert_eq!(value, PrescribedValue::Constant(components));

    let reference = PrescribedValue::Signal(SignalId("rotated-wall-history".into()));
    let source = AuthoredConstraint {
        constraint_id: "wall-drive".into(),
        kind: AuthoredConstraintKind::FluidMovingWall {
            target: authored_target("wall"),
            velocity: reference.clone(),
        },
    };
    let AnalysisConstraint::Fluid(realized) = realize_constraint(&source, &targets).unwrap() else {
        panic!("fluid");
    };
    let FluidConstraintKind::MovingWall { target, velocity } = realized.kind else {
        panic!("wall");
    };
    assert_eq!(target.region_id, "wall");
    assert_eq!(target.resolved_entities(), &[entity("face-wall")]);
    assert_eq!(velocity, reference);
}

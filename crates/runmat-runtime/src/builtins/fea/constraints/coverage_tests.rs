//! Focused coverage for typed FEA model-object constructors whose behavior is
//! not exercised by the family composition tests: every constraint family
//! method, support values, frames, step activation references and lifecycles,
//! the uniform geometry-revision class, and method-attributed case errors.

use super::acoustic::acoustic_prescribed_pressure_builtin;
use super::electrical::{
    electrical_equipotential_builtin, electrical_floating_conductor_builtin,
    electrical_integral_current_builtin, electrical_prescribed_potential_builtin,
};
use super::fluid::{
    fluid_moving_wall_builtin, fluid_no_slip_wall_builtin, fluid_prescribed_velocity_builtin,
    fluid_pressure_reference_builtin, fluid_slip_wall_builtin, fluid_symmetry_builtin,
};
use super::magnetic::{
    magnetic_coulomb_gauge_builtin, magnetic_prescribed_vector_potential_builtin,
    magnetic_vector_potential_ground_builtin,
};
use super::paired::{acoustic_periodic_builtin, thermal_tie_builtin};
use super::structural::{
    structural_antisymmetry_builtin, structural_normal_displacement_builtin,
    structural_rigid_coupling_builtin, structural_symmetry_builtin,
};
use super::support::dof::{
    structural_dof_rx_builtin, structural_dof_ry_builtin, structural_dof_rz_builtin,
    structural_dof_ux_builtin, structural_dof_uy_builtin, structural_dof_uz_builtin,
};
use super::support::frame::{
    coordinate_frame_boundary_builtin, coordinate_frame_cartesian_builtin,
    coordinate_frame_cylindrical_builtin, coordinate_frame_element_builtin,
    coordinate_frame_material_builtin, coordinate_frame_spherical_builtin,
};
use super::support::pairing::periodic_pairing_matching_builtin;
use super::support::periodic_transform::{
    periodic_transform_identity_builtin, periodic_transform_rotation_builtin,
};
use super::support::reference_point::reference_point_from_coordinates_builtin;
use super::*;
use crate::builtins::fea::{activations, excitations, initial_conditions, interfaces};
use crate::builtins::fea::{material_assignments, signal_policies, signals, values};
use crate::builtins::geometry::geometry_load_builtin;
use crate::builtins::geometry::region::geometry_region_builtin;
use futures::executor::block_on;
use runmat_value::{CellArray, Tensor};

const BLOCK_OBJ: &str = "v 0 0 0\nv 0.2 0 0\nv 0.2 0.1 0\nv 0 0.1 0\nv 0 0 0.1\nv 0.2 0 0.1\nv 0.2 0.1 0.1\nv 0 0.1 0.1\n\
g x_min\nf 1 5 8\nf 1 8 4\ng x_max\nf 2 3 7\nf 2 7 6\ng z_min\nf 1 4 3\nf 1 3 2\n";

struct Block {
    _dir: tempfile::TempDir,
    asset: Value,
}

impl Block {
    fn load() -> Self {
        let dir = tempfile::TempDir::new().unwrap();
        let path = dir.path().join("block.obj");
        std::fs::write(&path, BLOCK_OBJ).unwrap();
        let asset = block_on(geometry_load_builtin(
            Value::String(path.to_string_lossy().to_string().into()),
            crate::builtins::common::test_support::builtin_arguments(Vec::new()),
        ))
        .expect("block geometry should load");
        Self { _dir: dir, asset }
    }

    fn region(&self, name: &str) -> Value {
        block_on(geometry_region_builtin(self.asset.clone(), name.into())).expect("region")
    }
}

fn class_of(value: &Value) -> String {
    match value {
        Value::Object(object) => object.class_name().display_name().to_string(),
        other => panic!("expected an object, got {other:?}"),
    }
}

fn kind_key(value: &Value) -> String {
    let Value::Object(object) = value else {
        panic!("expected an object")
    };
    let Some(Value::Struct(constraint)) = object.properties().get("constraint") else {
        panic!("expected a constraint payload")
    };
    let Some(Value::Struct(kind)) = constraint.fields.get("kind") else {
        panic!("expected a constraint kind")
    };
    kind.fields
        .keys()
        .next()
        .map(str::to_owned)
        .expect("one kind")
}

fn direction(components: [f64; 3]) -> Value {
    block_on(super::support::direction::direction_vector_builtin(
        None,
        Value::Tensor(Tensor::new_2d(components.to_vec(), 1, 3).unwrap()),
    ))
    .unwrap()
}

fn origin() -> Value {
    block_on(values::position_meters_builtin(Value::Tensor(
        Tensor::new_2d(vec![0.0, 0.0, 0.0], 1, 3).unwrap(),
    )))
    .unwrap()
}

#[test]
fn every_region_constraint_method_builds_its_family_kind() {
    let block = Block::load();
    let face = || block.region("x_min");
    let pressure = block_on(values::pressure_pascals_builtin(Value::Num(0.0))).unwrap();
    let volts = block_on(values::electric_potential_volts_builtin(Value::Num(12.0))).unwrap();
    let amperes = block_on(values::electric_current_amperes_builtin(Value::Num(2.5))).unwrap();
    let potential = block_on(values::magnetic_vector_potential_builtin(Value::Num(1e-6))).unwrap();
    let displacement = block_on(values::displacement_millimeters_builtin(Value::Num(0.1))).unwrap();
    let speed = |v| block_on(values::velocity_meters_per_second_builtin(Value::Num(v))).unwrap();
    let velocity =
        Value::Cell(CellArray::new(vec![speed(1.5), speed(0.0), speed(0.0)], 1, 3).unwrap());
    let hub = block_on(reference_point_from_coordinates_builtin(
        None,
        "hub".into(),
        origin(),
    ))
    .unwrap();
    let cases = [
        (
            block_on(structural_symmetry_builtin("a".into(), face())),
            "fea.StructuralConstraint",
            "symmetry",
        ),
        (
            block_on(structural_antisymmetry_builtin("a".into(), face())),
            "fea.StructuralConstraint",
            "antisymmetry",
        ),
        (
            block_on(structural_normal_displacement_builtin(
                "a".into(),
                face(),
                displacement,
            )),
            "fea.StructuralConstraint",
            "normal_displacement",
        ),
        (
            block_on(structural_rigid_coupling_builtin("a".into(), face(), hub)),
            "fea.StructuralConstraint",
            "rigid_coupling",
        ),
        (
            block_on(electrical_equipotential_builtin("a".into(), face())),
            "fea.ElectricalConstraint",
            "equipotential",
        ),
        (
            block_on(electrical_floating_conductor_builtin("a".into(), face())),
            "fea.ElectricalConstraint",
            "floating_conductor",
        ),
        (
            block_on(electrical_prescribed_potential_builtin(
                "a".into(),
                face(),
                volts,
            )),
            "fea.ElectricalConstraint",
            "prescribed_potential",
        ),
        (
            block_on(electrical_integral_current_builtin(
                "a".into(),
                face(),
                amperes,
            )),
            "fea.ElectricalConstraint",
            "integral_current",
        ),
        (
            block_on(magnetic_vector_potential_ground_builtin("a".into(), face())),
            "fea.MagneticConstraint",
            "vector_potential_ground",
        ),
        (
            block_on(magnetic_coulomb_gauge_builtin("a".into(), face())),
            "fea.MagneticConstraint",
            "coulomb_gauge",
        ),
        (
            block_on(magnetic_prescribed_vector_potential_builtin(
                "a".into(),
                face(),
                potential,
            )),
            "fea.MagneticConstraint",
            "prescribed_vector_potential",
        ),
        (
            block_on(acoustic_prescribed_pressure_builtin(
                "a".into(),
                face(),
                pressure.clone(),
            )),
            "fea.AcousticConstraint",
            "prescribed_pressure",
        ),
        (
            block_on(fluid_no_slip_wall_builtin("a".into(), face())),
            "fea.FluidConstraint",
            "no_slip_wall",
        ),
        (
            block_on(fluid_slip_wall_builtin("a".into(), face())),
            "fea.FluidConstraint",
            "slip_wall",
        ),
        (
            block_on(fluid_symmetry_builtin("a".into(), face())),
            "fea.FluidConstraint",
            "symmetry",
        ),
        (
            block_on(fluid_pressure_reference_builtin(
                "a".into(),
                face(),
                pressure,
            )),
            "fea.FluidConstraint",
            "pressure_reference",
        ),
        (
            block_on(fluid_prescribed_velocity_builtin(
                "a".into(),
                face(),
                velocity.clone(),
            )),
            "fea.FluidConstraint",
            "prescribed_velocity",
        ),
        (
            block_on(fluid_moving_wall_builtin("a".into(), face(), velocity)),
            "fea.FluidConstraint",
            "moving_wall",
        ),
    ];
    for (value, class_name, kind) in cases {
        let value = value.unwrap_or_else(|error| panic!("{class_name}.{kind}: {error}"));
        assert_eq!(class_of(&value), class_name);
        assert_eq!(kind_key(&value), kind);
    }
    assert!(block_on(fluid_slip_wall_builtin(
        String::new(),
        block.region("x_min")
    ))
    .is_err());
}

#[test]
fn paired_tie_and_periodic_methods_cover_thermal_and_acoustic_families() {
    let block = Block::load();
    let pairing = || block_on(periodic_pairing_matching_builtin(None)).unwrap();
    let tie = block_on(thermal_tie_builtin(
        "bond".into(),
        block.region("x_min"),
        block.region("x_max"),
        pairing(),
    ))
    .unwrap();
    assert_eq!(kind_key(&tie), "tie");
    let periodic = block_on(acoustic_periodic_builtin(
        "cell".into(),
        block.region("x_min"),
        block.region("x_max"),
        block_on(periodic_transform_identity_builtin(None)).unwrap(),
        pairing(),
    ))
    .unwrap();
    assert_eq!(class_of(&periodic), "fea.AcousticConstraint");
    assert_eq!(kind_key(&periodic), "periodic");
}

#[test]
fn rotation_transform_and_every_structural_dof_are_typed() {
    let angle = block_on(values::rotation_degrees_builtin(Value::Num(45.0))).unwrap();
    let transform = block_on(periodic_transform_rotation_builtin(
        None,
        origin(),
        direction([0.0, 0.0, 2.0]),
        angle,
    ))
    .unwrap();
    assert_eq!(class_of(&transform), "fea.RotationPeriodicTransform");
    let resolved = super::support::periodic_transform_from_value("test", &transform).unwrap();
    let PeriodicTransform::Rotation { axis, angle, .. } = resolved else {
        panic!("expected a rotation")
    };
    assert_eq!(axis, [0.0, 0.0, 1.0]);
    assert!((angle.canonical_value() - std::f64::consts::FRAC_PI_4).abs() < 1e-15);
    assert!(block_on(periodic_transform_rotation_builtin(
        None,
        origin(),
        origin(),
        origin()
    ))
    .is_err());

    let dofs = [
        (
            block_on(structural_dof_ux_builtin(None)),
            "fea.StructuralTranslationDof",
            0,
        ),
        (
            block_on(structural_dof_uy_builtin(None)),
            "fea.StructuralTranslationDof",
            1,
        ),
        (
            block_on(structural_dof_uz_builtin(None)),
            "fea.StructuralTranslationDof",
            2,
        ),
        (
            block_on(structural_dof_rx_builtin(None)),
            "fea.StructuralRotationDof",
            0,
        ),
        (
            block_on(structural_dof_ry_builtin(None)),
            "fea.StructuralRotationDof",
            1,
        ),
        (
            block_on(structural_dof_rz_builtin(None)),
            "fea.StructuralRotationDof",
            2,
        ),
    ];
    for (value, class_name, axis) in dofs {
        let value = value.unwrap();
        assert_eq!(class_of(&value), class_name);
        let Value::Object(object) = &value else {
            unreachable!()
        };
        assert_eq!(
            object.properties().get("axis"),
            Some(&Value::Int(IntValue::U8(axis)))
        );
    }
}

#[test]
fn frames_build_every_class_and_reject_non_orthogonal_axes_before_construction() {
    for (value, class_name) in [
        (
            block_on(coordinate_frame_material_builtin(None)),
            "fea.MaterialCoordinateFrame",
        ),
        (
            block_on(coordinate_frame_element_builtin(None)),
            "fea.ElementCoordinateFrame",
        ),
        (
            block_on(coordinate_frame_boundary_builtin(None)),
            "fea.BoundaryNormalTangentFrame",
        ),
        (
            block_on(coordinate_frame_cylindrical_builtin(
                None,
                "shaft".into(),
                origin(),
                direction([1.0, 0.0, 0.0]),
                direction([1.0, 1.0, 0.0]),
            )),
            "fea.CylindricalCoordinateFrame",
        ),
        (
            block_on(coordinate_frame_spherical_builtin(
                None,
                "dome".into(),
                origin(),
                direction([0.0, 0.0, 1.0]),
                direction([1.0, 0.0, 0.0]),
            )),
            "fea.SphericalCoordinateFrame",
        ),
    ] {
        assert_eq!(class_of(&value.unwrap()), class_name);
    }
    let skewed = block_on(coordinate_frame_cartesian_builtin(
        None,
        "skew".into(),
        origin(),
        direction([1.0, 0.0, 0.0]),
        direction([1.0, 1.0, 0.0]),
    ))
    .expect_err("non-orthogonal Cartesian axes must be rejected");
    assert!(
        skewed.message().contains("orthogonal"),
        "{}",
        skewed.message()
    );
    let tilted = block_on(coordinate_frame_spherical_builtin(
        None,
        "tilted".into(),
        origin(),
        direction([0.0, 0.0, 1.0]),
        direction([1.0, 0.0, 1.0]),
    ))
    .expect_err("non-orthogonal spherical axes must be rejected");
    assert!(
        tilted.message().contains("orthogonal"),
        "{}",
        tilted.message()
    );
}

fn revision(value: &Value) -> Option<Value> {
    let Value::Object(object) = value else {
        panic!("expected an object")
    };
    object.properties().get("geometry_revision").cloned()
}

#[test]
fn geometry_bound_model_objects_share_the_uint32_revision_class() {
    let block = Block::load();
    let face = || block.region("x_min");
    let force = block_on(signals::force_constant(
        "pull".into(),
        Value::Tensor(Tensor::new_2d(vec![1.0, 0.0, 0.0], 1, 3).unwrap()),
    ))
    .unwrap();
    let temperature = block_on(signals::temperature_constant(
        "t0".into(),
        Value::Num(293.15),
    ))
    .unwrap();
    let bound = [
        block_on(structural_fixed_builtin("root".into(), face())).unwrap(),
        block_on(material_assignments::fea_property_region_target_solid_builtin(face())).unwrap(),
        block_on(
            material_assignments::fea_property_region_target_all_solids_builtin(
                block.asset.clone(),
            ),
        )
        .unwrap(),
        block_on(interfaces::contact_interface_frictionless_builtin(
            Value::String("seat".into()),
            face(),
            block.region("x_max"),
        ))
        .unwrap(),
        block_on(excitations::structural_force("pull".into(), face(), force)).unwrap(),
        block_on(initial_conditions::thermal_temperature(
            "t0".into(),
            block.region("z_min"),
            temperature,
            Value::Num(0.0),
            block_on(signal_policies::sample_approach_at()).unwrap(),
            "cooldown".into(),
        ))
        .unwrap(),
    ];
    for value in &bound {
        assert!(
            matches!(revision(value), Some(Value::Int(IntValue::U32(_)))),
            "{}: {:?}",
            class_of(value),
            revision(value)
        );
    }
}

#[test]
fn excitation_case_errors_name_the_failing_method() {
    let empty = Value::Cell(CellArray::new(Vec::new(), 1, 0).unwrap());
    for (error, builtin) in [
        (
            block_on(excitations::excitation_case_always(
                String::new(),
                empty.clone(),
            )),
            "fea.ExcitationCase.always",
        ),
        (
            block_on(excitations::excitation_case_during_steps(
                "case".into(),
                empty.clone(),
                Value::String("step".into()),
            )),
            "fea.ExcitationCase.duringSteps",
        ),
        (
            block_on(excitations::excitation_case_mutually_exclusive(
                "case".into(),
                empty,
                "group".into(),
            )),
            "fea.ExcitationCase.mutuallyExclusive",
        ),
    ] {
        let error = error.expect_err("invalid case must be rejected");
        assert_eq!(error.context.builtin.as_deref(), Some(builtin));
    }
}

#[test]
fn every_step_reference_and_lifecycle_round_trips_through_an_activation() {
    let references = [
        block_on(activations::constraint("root".into())),
        block_on(activations::boundary("b".into())),
        block_on(activations::connector("bolt".into())),
        block_on(activations::interface("i".into())),
        block_on(activations::excitation("e".into())),
        block_on(activations::initial_condition("ic".into())),
        block_on(activations::material_parameter("E".into())),
    ];
    let lifecycles = [
        block_on(activations::activate()).unwrap(),
        block_on(activations::continue_from("preload".into())).unwrap(),
        block_on(activations::deactivate("preload".into())).unwrap(),
        block_on(activations::modify(
            "preload".into(),
            block_on(activations::excitation("bolt_load".into())).unwrap(),
        ))
        .unwrap(),
    ];
    for reference in references {
        let reference = reference.unwrap();
        assert_eq!(class_of(&reference), "fea.StepObjectRef");
        for lifecycle in &lifecycles {
            assert_eq!(class_of(lifecycle), "fea.ActivationLifecycle");
            let activation = block_on(activations::activation(
                reference.clone(),
                lifecycle.clone(),
            ))
            .unwrap();
            assert_eq!(class_of(&activation), "fea.StepObjectActivation");
            assert_eq!(
                activations::activations("test", &activation).unwrap().len(),
                1
            );
        }
    }
    assert!(block_on(activations::continue_from(String::new())).is_err());
    assert!(block_on(activations::modify("preload".into(), Value::Num(1.0))).is_err());
}

use super::*;
use crate::builtins::fea::values::{quantity_scalar, TEMPERATURE_CLASS};
use crate::builtins::geometry::region::{region_ref_from_value, GeometryRegionRef};

pub(super) const STRUCTURAL_FIXED_NAME: &str = "fea.StructuralConstraint.fixed";
pub(super) const THERMAL_PRESCRIBED_NAME: &str = "fea.ThermalConstraint.prescribed";

/// Reject a request for more outputs than the catalog signature declares.
pub(crate) fn fixed_outputs(
    descriptor: &'static BuiltinDescriptor,
    builtin: &str,
) -> BuiltinResult<()> {
    crate::output_count::validate_fixed_outputs(descriptor, &ERROR_INPUT, builtin)
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct GeometryBoundConstraint<T> {
    pub(super) geometry_id: String,
    pub(super) geometry_revision: u32,
    pub(super) constraint: T,
}

pub(super) type BoundStructuralConstraint = GeometryBoundConstraint<StructuralConstraint>;
pub(super) type BoundThermalConstraint = GeometryBoundConstraint<ThermalConstraint>;
pub(super) type BoundElectricalConstraint = GeometryBoundConstraint<ElectricalConstraint>;
pub(super) type BoundMagneticConstraint = GeometryBoundConstraint<MagneticConstraint>;
pub(super) type BoundAcousticConstraint = GeometryBoundConstraint<AcousticConstraint>;
pub(super) type BoundFluidConstraint = GeometryBoundConstraint<FluidConstraint>;

impl<T> GeometryBoundConstraint<T> {
    pub(super) fn require_geometry(
        &self,
        builtin: &'static str,
        geometry: &GeometryAsset,
    ) -> BuiltinResult<()> {
        require_geometry_binding(
            builtin,
            "constraint",
            &self.geometry_id,
            self.geometry_revision,
            geometry,
        )
    }
}

pub(super) fn require_geometry_binding(
    builtin: &'static str,
    label: &str,
    geometry_id: &str,
    geometry_revision: u32,
    geometry: &GeometryAsset,
) -> BuiltinResult<()> {
    if geometry_id != geometry.geometry_id || geometry_revision != geometry.revision {
        return Err(builtin_error(
            builtin,
            &ERROR_INPUT,
            format!(
                "{label} belongs to geometry `{geometry_id}` revision {geometry_revision}, but the model uses `{}` revision {}",
                geometry.geometry_id, geometry.revision
            ),
        ));
    }
    Ok(())
}

pub(super) fn bound_region(
    builtin: &'static str,
    id: String,
    region: &Value,
) -> BuiltinResult<(String, String, u32, RegionTarget)> {
    if id.is_empty() {
        return Err(builtin_error(
            builtin,
            &ERROR_INPUT,
            "constraint id must not be empty",
        ));
    }
    let (geometry_id, geometry_revision, target) = bound_region_target(builtin, region)?;
    Ok((id, geometry_id, geometry_revision, target))
}

pub(super) fn bound_region_target(
    builtin: &'static str,
    region: &Value,
) -> BuiltinResult<(String, u32, RegionTarget)> {
    let GeometryRegionRef {
        geometry_id,
        geometry_revision,
        region_id,
        ..
    } = region_ref_from_value(builtin, region)?;
    Ok((
        geometry_id,
        geometry_revision,
        RegionTarget::boundary(region_id),
    ))
}

fn bound_region_pair(
    builtin: &'static str,
    id: String,
    primary: &Value,
    secondary: &Value,
    pairing: &Value,
) -> BuiltinResult<(String, String, u32, PairedConstraint)> {
    if id.is_empty() {
        return Err(builtin_error(
            builtin,
            &ERROR_INPUT,
            "constraint id must not be empty",
        ));
    }
    let (geometry_id, geometry_revision, primary_target) = bound_region_target(builtin, primary)?;
    let (secondary_geometry_id, secondary_revision, secondary_target) =
        bound_region_target(builtin, secondary)?;
    if geometry_id != secondary_geometry_id || geometry_revision != secondary_revision {
        return Err(builtin_error(
            builtin,
            &ERROR_INPUT,
            "paired regions must belong to the same geometry revision",
        ));
    }
    if primary_target.region_id == secondary_target.region_id {
        return Err(builtin_error(
            builtin,
            &ERROR_INPUT,
            "paired constraints require distinct primary and secondary regions",
        ));
    }
    Ok((
        id,
        geometry_id,
        geometry_revision,
        PairedConstraint {
            primary: primary_target,
            secondary: secondary_target,
            pairing: support::pairing_policy_from_value(builtin, pairing)?,
        },
    ))
}

pub(super) fn bound_constraint_to_object<T: Serialize>(
    builtin: &'static str,
    class_name: &'static str,
    geometry_id: String,
    geometry_revision: u32,
    constraint: T,
) -> BuiltinResult<Value> {
    let mut object = serializable_to_object_value(
        builtin,
        &ERROR_INTERNAL,
        class_name,
        &GeometryBoundConstraint {
            geometry_id,
            geometry_revision,
            constraint,
        },
    )?;
    crate::builtins::fea::object_api::replace(
        builtin,
        &mut object,
        "geometry_revision",
        Value::Int(IntValue::U32(geometry_revision)),
    )?;
    Ok(Value::Object(object))
}

#[runtime_builtin(
    name = "fea.StructuralConstraint.fixed",
    binding_variant = "default",
    builtin_path = "crate::builtins::fea::constraints"
)]
pub async fn structural_fixed_builtin(id: String, region: Value) -> BuiltinResult<Value> {
    fixed_outputs(
        &runmat_builtins::FEA_STRUCTURAL_CONSTRAINT_FIXED_DESCRIPTOR,
        STRUCTURAL_FIXED_NAME,
    )?;
    if id.is_empty() {
        return Err(builtin_error(
            STRUCTURAL_FIXED_NAME,
            &ERROR_INPUT,
            "constraint id must not be empty",
        ));
    }
    let GeometryRegionRef {
        geometry_id,
        geometry_revision,
        region_id,
        ..
    } = region_ref_from_value(STRUCTURAL_FIXED_NAME, &region)?;
    bound_constraint_to_object(
        STRUCTURAL_FIXED_NAME,
        FEA_STRUCTURAL_CONSTRAINT_CLASS,
        geometry_id,
        geometry_revision,
        StructuralConstraint {
            constraint_id: id,
            frame: CoordinateFrame::Global,
            kind: StructuralConstraintKind::Fixed {
                target: RegionTarget::boundary(region_id),
            },
        },
    )
}

#[runtime_builtin(
    name = "fea.ThermalConstraint.prescribed",
    binding_variant = "default",
    builtin_path = "crate::builtins::fea::constraints"
)]
pub async fn thermal_prescribed_builtin(
    id: String,
    region: Value,
    temperature: Value,
) -> BuiltinResult<Value> {
    fixed_outputs(
        &runmat_builtins::FEA_THERMAL_CONSTRAINT_PRESCRIBED_DESCRIPTOR,
        THERMAL_PRESCRIBED_NAME,
    )?;
    if id.is_empty() {
        return Err(builtin_error(
            THERMAL_PRESCRIBED_NAME,
            &ERROR_INPUT,
            "constraint id must not be empty",
        ));
    }
    let GeometryRegionRef {
        geometry_id,
        geometry_revision,
        region_id,
        ..
    } = region_ref_from_value(THERMAL_PRESCRIBED_NAME, &region)?;
    let kelvin = quantity_scalar(
        THERMAL_PRESCRIBED_NAME,
        &temperature,
        TEMPERATURE_CLASS,
        "kelvin",
    )?;
    bound_constraint_to_object(
        THERMAL_PRESCRIBED_NAME,
        FEA_THERMAL_CONSTRAINT_CLASS,
        geometry_id,
        geometry_revision,
        ThermalConstraint {
            constraint_id: id,
            kind: ThermalConstraintKind::Prescribed {
                target: RegionTarget::boundary(region_id),
                value: PrescribedValue::Constant(Temperature::new(kelvin)),
            },
        },
    )
}

macro_rules! define_simple_constraint {
    (
        $function:ident, $descriptor:ident, $builtin_name:literal, $class:expr,
        $builtin_path:literal, $build:expr
    ) => {
        #[runtime_builtin(name = $builtin_name, binding_variant = "default", builtin_path = $builtin_path)]
        pub async fn $function(id: String, region: Value) -> BuiltinResult<Value> {
            fixed_outputs(&runmat_builtins::$descriptor, $builtin_name)?;
            let (id, geometry_id, geometry_revision, target) =
                bound_region($builtin_name, id, &region)?;
            let constraint = ($build)(id, target);
            bound_constraint_to_object(
                $builtin_name,
                $class,
                geometry_id,
                geometry_revision,
                constraint,
            )
        }
    };
}

macro_rules! define_quantity_constraint {
    (
        $function:ident, $descriptor:ident, $builtin_name:literal, $class:expr,
        $builtin_path:literal, $value_class:expr, $value_field:literal, $build:expr
    ) => {
        #[runtime_builtin(name = $builtin_name, binding_variant = "default", builtin_path = $builtin_path)]
        pub async fn $function(id: String, region: Value, value: Value) -> BuiltinResult<Value> {
            fixed_outputs(&runmat_builtins::$descriptor, $builtin_name)?;
            let (id, geometry_id, geometry_revision, target) =
                bound_region($builtin_name, id, &region)?;
            let value = quantity_scalar($builtin_name, &value, $value_class, $value_field)?;
            let constraint = ($build)(id, target, value);
            bound_constraint_to_object(
                $builtin_name,
                $class,
                geometry_id,
                geometry_revision,
                constraint,
            )
        }
    };
}

pub(crate) mod acoustic;
pub(crate) mod electrical;
pub(crate) mod fluid;
pub(crate) mod magnetic;
pub(crate) mod paired;
pub(crate) mod structural;
pub(crate) mod support;

pub(super) fn ensure_constraint_support_classes_registered() {
    support::ensure_constraint_support_classes_registered();
}

#[cfg(test)]
use self::{
    acoustic::acoustic_prescribed_pressure_builtin,
    electrical::electrical_potential_reference_builtin, fluid::fluid_no_slip_wall_builtin,
    magnetic::magnetic_vector_potential_ground_builtin,
};

#[cfg(test)]
mod coverage_tests;

#[cfg(test)]
mod tests {
    use super::*;
    use super::{
        paired::{
            acoustic_phase_periodic_builtin, electrical_periodic_builtin, fluid_periodic_builtin,
            magnetic_phase_periodic_builtin, structural_periodic_builtin, structural_tie_builtin,
            thermal_affine_jump_builtin, thermal_periodic_builtin,
        },
        support::{
            pairing::{periodic_pairing_geometric_builtin, periodic_pairing_matching_builtin},
            periodic_transform::{
                periodic_transform_identity_builtin, periodic_transform_translation_builtin,
            },
        },
    };
    use crate::builtins::fea::test_support::load_stl_fixture;
    use crate::builtins::geometry::region::geometry_region_builtin;
    use crate::builtins::geometry::{
        geometry_asset_from_value_with_builtin, geometry_load_builtin,
    };
    use futures::executor::block_on;
    use runmat_analysis_core::{
        AccelerationSignalId, ReferencePointCoupling, SignalId, StructuralKinematicHistory,
        StructuralMotionReconstruction,
    };

    fn distinct_region(region: &Value) -> Value {
        let Value::Object(mut object) = region.clone() else {
            panic!("expected geometry.Region")
        };
        crate::builtins::fea::object_api::replace(
            "fea.fixture",
            &mut object,
            "region_id",
            Value::String("face:secondary".into()),
        )
        .expect("ordinary fixture property edit");
        crate::builtins::fea::object_api::replace(
            "fea.fixture",
            &mut object,
            "name",
            Value::String("Secondary".into()),
        )
        .expect("ordinary fixture property edit");
        Value::Object(object)
    }

    #[test]
    fn fixed_constraint_is_bound_to_the_selected_geometry_revision() {
        let tmp = tempfile::TempDir::new().unwrap();
        let path = tmp.path().join("part.stl");
        std::fs::write(
            &path,
            "solid base\nfacet normal 0 0 1\nouter loop\nvertex 0 0 0\nvertex 1 0 0\nvertex 0 1 0\nendloop\nendfacet\nendsolid base\n",
        )
        .unwrap();
        let asset = block_on(geometry_load_builtin(
            Value::String(path.to_string_lossy().to_string().into()),
            crate::builtins::common::test_support::builtin_arguments(vec![]),
        ))
        .expect("geometry should load");
        let decoded = geometry_asset_from_value_with_builtin(&asset, STRUCTURAL_FIXED_NAME)
            .expect("asset should decode");
        let region_id = decoded.regions[0].region_id.clone();
        let region = block_on(geometry_region_builtin(asset, region_id)).expect("region");
        let constraint =
            block_on(structural_fixed_builtin("base".into(), region)).expect("fixed constraint");
        let bound: BoundStructuralConstraint = object_payload(
            STRUCTURAL_FIXED_NAME,
            &constraint,
            FEA_STRUCTURAL_CONSTRAINT_CLASS,
        )
        .expect("constraint should decode");
        assert_eq!(bound.geometry_id, decoded.geometry_id);
        assert_eq!(bound.geometry_revision, decoded.revision);
        assert!(matches!(
            bound.constraint.kind,
            StructuralConstraintKind::Fixed { ref target } if target.region_id == decoded.regions[0].region_id
        ));

        let mut changed = decoded;
        changed.revision += 1;
        assert!(bound
            .require_geometry(STRUCTURAL_FIXED_NAME, &changed)
            .unwrap_err()
            .message()
            .contains("belongs to geometry"));
    }

    #[test]
    fn model_composition_accepts_matching_geometry_and_rejects_a_foreign_region() {
        let tmp = tempfile::TempDir::new().unwrap();
        let asset = load_stl_fixture(&tmp.path().join("part-a.stl"), 1.0);
        let decoded = geometry_asset_from_value_with_builtin(&asset, STRUCTURAL_FIXED_NAME)
            .expect("asset should decode");
        let region = block_on(geometry_region_builtin(
            asset.clone(),
            decoded.regions[0].region_id.clone(),
        ))
        .expect("region");
        let constraint =
            block_on(structural_fixed_builtin("base".into(), region)).expect("fixed constraint");
        let model = create_model_object_from_args(
            crate::builtins::common::test_support::builtin_arguments(vec![
                Value::String("matching".into()),
                asset,
                Value::String("Steps".into()),
                super::super::modeling::static_test_steps_value(),
                Value::String("Constraints".into()),
                constraint.clone(),
            ]),
        )
        .expect("matching geometry should compose");
        let model: AnalysisModel =
            object_payload(MODEL_NAME, &model, FEA_MODEL_CLASS).expect("model should decode");
        assert_eq!(model.constraints.len(), 1);

        let foreign_asset = load_stl_fixture(&tmp.path().join("part-b.stl"), 2.0);
        let error = create_model_object_from_args(
            crate::builtins::common::test_support::builtin_arguments(vec![
                Value::String("foreign".into()),
                foreign_asset,
                Value::String("Steps".into()),
                super::super::modeling::static_test_steps_value(),
                Value::String("Constraints".into()),
                constraint,
            ]),
        )
        .unwrap_err();
        assert!(error.message().contains("belongs to geometry"));
    }

    #[test]
    fn thermal_constraint_requires_a_typed_temperature() {
        let tmp = tempfile::TempDir::new().unwrap();
        let asset = load_stl_fixture(&tmp.path().join("thermal.stl"), 1.0);
        let decoded = geometry_asset_from_value_with_builtin(&asset, THERMAL_PRESCRIBED_NAME)
            .expect("asset should decode");
        let region = block_on(geometry_region_builtin(
            asset,
            decoded.regions[0].region_id.clone(),
        ))
        .expect("region");
        let temperature = block_on(crate::builtins::fea::values::temperature_celsius_builtin(
            Value::Num(20.0),
        ))
        .expect("temperature");
        let constraint = block_on(thermal_prescribed_builtin(
            "wall_temperature".into(),
            region.clone(),
            temperature,
        ))
        .expect("thermal constraint");
        let bound: BoundThermalConstraint = object_payload(
            THERMAL_PRESCRIBED_NAME,
            &constraint,
            FEA_THERMAL_CONSTRAINT_CLASS,
        )
        .expect("constraint should decode");
        assert!(matches!(
            bound.constraint.kind,
            ThermalConstraintKind::Prescribed {
                value: PrescribedValue::Constant(temperature),
                ..
            } if temperature.canonical_value() == 293.15
        ));

        let displacement = block_on(crate::builtins::fea::values::displacement_meters_builtin(
            Value::Num(1.0),
        ))
        .expect("displacement");
        assert!(block_on(thermal_prescribed_builtin(
            "wrong_dimension".into(),
            region,
            displacement,
        ))
        .is_err());
    }

    #[test]
    fn paired_relations_are_typed_transformed_and_dimensionally_sound() {
        let tmp = tempfile::TempDir::new().unwrap();
        let asset = load_stl_fixture(&tmp.path().join("paired.stl"), 1.0);
        let decoded = geometry_asset_from_value_with_builtin(&asset, STRUCTURAL_FIXED_NAME)
            .expect("asset should decode");
        let primary = block_on(geometry_region_builtin(
            asset,
            decoded.regions[0].region_id.clone(),
        ))
        .expect("primary region");
        let secondary = distinct_region(&primary);
        let matching = block_on(periodic_pairing_matching_builtin(None)).expect("matching pairing");
        let identity =
            block_on(periodic_transform_identity_builtin(None)).expect("identity transform");
        let translation = block_on(
            crate::builtins::fea::values::translation_millimeters_builtin(Value::Tensor(
                Tensor::new_2d(vec![10.0, 0.0, 0.0], 1, 3).unwrap(),
            )),
        )
        .expect("translation");
        let translated = block_on(periodic_transform_translation_builtin(None, translation))
            .expect("translated transform");
        let tolerance = block_on(crate::builtins::fea::values::length_micrometers_builtin(
            Value::Num(5.0),
        ))
        .expect("tolerance");
        let geometric = block_on(periodic_pairing_geometric_builtin(None, tolerance))
            .expect("geometric pairing");

        let tie = block_on(structural_tie_builtin(
            "tie".into(),
            primary.clone(),
            secondary.clone(),
            matching.clone(),
        ))
        .expect("structural tie");
        let tie: BoundStructuralConstraint = object_payload(
            "fea.StructuralConstraint.tie",
            &tie,
            FEA_STRUCTURAL_CONSTRAINT_CLASS,
        )
        .unwrap();
        assert!(matches!(
            tie.constraint.kind,
            StructuralConstraintKind::Tie(PairedConstraint {
                pairing: PairingPolicy::Matching,
                ..
            })
        ));

        let periodic = block_on(structural_periodic_builtin(
            "periodic".into(),
            primary.clone(),
            secondary.clone(),
            translated,
            geometric,
        ))
        .expect("structural periodic relation");
        let periodic: BoundStructuralConstraint = object_payload(
            "fea.StructuralConstraint.periodic",
            &periodic,
            FEA_STRUCTURAL_CONSTRAINT_CLASS,
        )
        .unwrap();
        assert!(matches!(
            periodic.constraint.kind,
            StructuralConstraintKind::Periodic(PeriodicConstraint {
                transform: PeriodicTransform::Translation(translation),
                ..
            }) if translation.canonical_value() == [0.01, 0.0, 0.0]
        ));

        let jump = block_on(
            crate::builtins::fea::values::temperature_difference_celsius_builtin(Value::Num(-15.0)),
        )
        .expect("temperature difference");
        let affine = block_on(thermal_affine_jump_builtin(
            "jump".into(),
            primary.clone(),
            secondary.clone(),
            identity.clone(),
            matching.clone(),
            jump,
        ))
        .expect("affine thermal relation");
        let affine: BoundThermalConstraint = object_payload(
            "fea.ThermalConstraint.affineJump",
            &affine,
            FEA_THERMAL_CONSTRAINT_CLASS,
        )
        .unwrap();
        assert!(matches!(
            affine.constraint.kind,
            ThermalConstraintKind::AffineJump {
                temperature_jump: PrescribedValue::Constant(temperature_jump),
                ..
            } if temperature_jump.canonical_value() == -15.0
        ));
        let absolute_temperature = block_on(
            crate::builtins::fea::values::temperature_celsius_builtin(Value::Num(15.0)),
        )
        .unwrap();
        assert!(block_on(thermal_affine_jump_builtin(
            "wrong_dimension".into(),
            primary.clone(),
            secondary.clone(),
            identity.clone(),
            matching.clone(),
            absolute_temperature,
        ))
        .is_err());

        let phase = block_on(crate::builtins::fea::values::complex_phase_degrees_builtin(
            Value::Num(90.0),
        ))
        .expect("phase");
        for value in [
            block_on(magnetic_phase_periodic_builtin(
                "magnetic".into(),
                primary.clone(),
                secondary.clone(),
                identity.clone(),
                matching.clone(),
                phase.clone(),
            ))
            .expect("magnetic phase relation"),
            block_on(acoustic_phase_periodic_builtin(
                "acoustic".into(),
                primary.clone(),
                secondary.clone(),
                identity.clone(),
                matching.clone(),
                phase,
            ))
            .expect("acoustic phase relation"),
        ] {
            let encoded = crate::execution::value_codec::encode_inline_value(&value)
                .expect("paired relation should encode");
            assert_eq!(
                crate::execution::value_codec::decode_inline_value(&encoded)
                    .expect("paired relation should decode"),
                value
            );
        }

        for result in [
            block_on(thermal_periodic_builtin(
                "thermal".into(),
                primary.clone(),
                secondary.clone(),
                identity.clone(),
                matching.clone(),
            )),
            block_on(electrical_periodic_builtin(
                "electrical".into(),
                primary.clone(),
                secondary.clone(),
                identity.clone(),
                matching.clone(),
            )),
            block_on(fluid_periodic_builtin(
                "fluid".into(),
                primary.clone(),
                secondary.clone(),
                identity.clone(),
                matching.clone(),
            )),
        ] {
            assert!(result.is_ok(), "family periodic constructor should succeed");
        }
        assert!(block_on(structural_tie_builtin(
            "self_pair".into(),
            primary.clone(),
            primary.clone(),
            matching.clone(),
        ))
        .is_err());
        let mut foreign = distinct_region(&primary);
        let Value::Object(object) = &mut foreign else {
            unreachable!()
        };
        crate::builtins::fea::object_api::replace(
            "fea.fixture",
            object,
            "geometry_id",
            Value::String("foreign".into()),
        )
        .expect("ordinary fixture property edit");
        assert!(block_on(structural_tie_builtin(
            "foreign_pair".into(),
            primary,
            foreign,
            matching,
        ))
        .is_err());
    }

    #[test]
    fn every_simple_family_constraint_composes_through_typed_classes() {
        let tmp = tempfile::TempDir::new().unwrap();
        let asset = load_stl_fixture(&tmp.path().join("families.stl"), 1.0);
        let decoded = geometry_asset_from_value_with_builtin(&asset, STRUCTURAL_FIXED_NAME)
            .expect("asset should decode");
        let select = || {
            block_on(geometry_region_builtin(
                asset.clone(),
                decoded.regions[0].region_id.clone(),
            ))
            .expect("region")
        };
        let temperature = block_on(crate::builtins::fea::values::temperature_kelvin_builtin(
            Value::Num(300.0),
        ))
        .unwrap();
        let pressure = block_on(crate::builtins::fea::values::pressure_pascals_builtin(
            Value::Num(1.0),
        ))
        .unwrap();
        let constraints = vec![
            block_on(structural_fixed_builtin("structural".into(), select())).unwrap(),
            block_on(thermal_prescribed_builtin(
                "thermal".into(),
                select(),
                temperature,
            ))
            .unwrap(),
            block_on(electrical_potential_reference_builtin(
                "electrical".into(),
                select(),
            ))
            .unwrap(),
            block_on(magnetic_vector_potential_ground_builtin(
                "magnetic".into(),
                select(),
            ))
            .unwrap(),
            block_on(acoustic_prescribed_pressure_builtin(
                "acoustic".into(),
                select(),
                pressure,
            ))
            .unwrap(),
            block_on(fluid_no_slip_wall_builtin("fluid".into(), select())).unwrap(),
        ];
        let conditions = crate::make_cell_with_shape(constraints, vec![1, 6]).unwrap();
        let model = create_model_object_from_args(
            crate::builtins::common::test_support::builtin_arguments(vec![
                Value::String("families".into()),
                asset,
                Value::String("Steps".into()),
                super::super::modeling::static_test_steps_value(),
                Value::String("Constraints".into()),
                conditions,
            ]),
        )
        .expect("typed family constraints should compose");
        let model: AnalysisModel = object_payload(MODEL_NAME, &model, FEA_MODEL_CLASS).unwrap();
        assert_eq!(model.constraints.len(), 6);
        assert!(matches!(
            model.constraints[0],
            AnalysisConstraint::Structural(_)
        ));
        assert!(matches!(
            model.constraints[1],
            AnalysisConstraint::Thermal(_)
        ));
        assert!(matches!(
            model.constraints[2],
            AnalysisConstraint::Electrical(_)
        ));
        assert!(matches!(
            model.constraints[3],
            AnalysisConstraint::Magnetic(_)
        ));
        assert!(matches!(
            model.constraints[4],
            AnalysisConstraint::Acoustic(_)
        ));
        assert!(matches!(model.constraints[5], AnalysisConstraint::Fluid(_)));

        ensure_fea_classes_registered();
        for (class_name, method) in [
            (FEA_STRUCTURAL_CONSTRAINT_CLASS, "normalDisplacement"),
            (FEA_THERMAL_CONSTRAINT_CLASS, "prescribed"),
            (FEA_ELECTRICAL_CONSTRAINT_CLASS, "integralCurrent"),
            (FEA_MAGNETIC_CONSTRAINT_CLASS, "coulombGauge"),
            (FEA_ACOUSTIC_CONSTRAINT_CLASS, "prescribedPressure"),
            (FEA_FLUID_CONSTRAINT_CLASS, "movingWall"),
        ] {
            let (definition, _) = crate::class_registry::lookup_method(
                &runmat_types::ClassIdentity::from(class_name),
                &runmat_types::MethodName::from(method),
            )
            .expect("typed class method should be registered");
            assert!(definition.is_static);
        }
    }

    #[test]
    fn prescribed_structural_dofs_and_reference_couplings_are_fully_typed() {
        let tmp = tempfile::TempDir::new().unwrap();
        let asset = load_stl_fixture(&tmp.path().join("typed-structural.stl"), 1.0);
        let decoded = geometry_asset_from_value_with_builtin(&asset, STRUCTURAL_FIXED_NAME)
            .expect("asset should decode");
        let select = || {
            block_on(geometry_region_builtin(
                asset.clone(),
                decoded.regions[0].region_id.clone(),
            ))
            .expect("region")
        };

        let ux = block_on(support::dof::structural_dof_ux_builtin(None)).unwrap();
        let rz = block_on(support::dof::structural_dof_rz_builtin(None)).unwrap();
        let displacement = block_on(
            crate::builtins::fea::values::displacement_millimeters_builtin(Value::Num(2.5)),
        )
        .unwrap();
        let rotation = block_on(crate::builtins::fea::values::rotation_degrees_builtin(
            Value::Num(90.0),
        ))
        .unwrap();
        let ux_value = block_on(support::dof::structural_translation_value_builtin(
            None,
            ux.clone(),
            displacement.clone(),
        ))
        .unwrap();
        let rz_value = block_on(support::dof::structural_rotation_value_builtin(
            None, rz, rotation,
        ))
        .unwrap();
        assert!(block_on(support::dof::structural_rotation_value_builtin(
            None,
            ux,
            displacement,
        ))
        .is_err());

        let values = crate::make_cell_with_shape(vec![ux_value.clone(), rz_value], vec![1, 2])
            .expect("DOF values cell");
        let frame = block_on(support::frame::coordinate_frame_global_builtin(None)).unwrap();
        let prescribed = block_on(structural::structural_prescribed_builtin(
            "guided".into(),
            select(),
            values,
            frame,
        ))
        .expect("typed prescribed constraint");
        let bound: BoundStructuralConstraint = object_payload(
            "fea.StructuralConstraint.prescribed",
            &prescribed,
            FEA_STRUCTURAL_CONSTRAINT_CLASS,
        )
        .unwrap();
        let StructuralConstraintKind::Prescribed(values) = bound.constraint.kind else {
            panic!("expected prescribed DOFs")
        };
        assert_eq!(values.translations.len(), 1);
        assert_eq!(values.rotations.len(), 1);
        assert_eq!(
            values.translations[0]
                .value
                .study_constant(&[])
                .map(|value| value.canonical_value()),
            Some(0.0025)
        );
        assert_eq!(
            values.rotations[0]
                .value
                .study_constant(&[])
                .map(|value| value.canonical_value()),
            Some(std::f64::consts::FRAC_PI_2)
        );

        let duplicate_values =
            crate::make_cell_with_shape(vec![ux_value.clone(), ux_value], vec![1, 2]).unwrap();
        let frame = block_on(support::frame::coordinate_frame_global_builtin(None)).unwrap();
        assert!(block_on(structural::structural_prescribed_builtin(
            "duplicate".into(),
            select(),
            duplicate_values,
            frame,
        ))
        .is_err());

        let position = block_on(crate::builtins::fea::values::position_millimeters_builtin(
            Value::Tensor(Tensor::new(vec![100.0, 0.0, 25.0], vec![1, 3]).unwrap()),
        ))
        .unwrap();
        let point = block_on(
            support::reference_point::reference_point_from_coordinates_builtin(
                None,
                "load_point".into(),
                position,
            ),
        )
        .unwrap();
        let coupling = block_on(structural::structural_distributing_coupling_builtin(
            "remote_load".into(),
            select(),
            point,
        ))
        .unwrap();
        let bound: BoundStructuralConstraint = object_payload(
            "fea.StructuralConstraint.distributingCoupling",
            &coupling,
            FEA_STRUCTURAL_CONSTRAINT_CLASS,
        )
        .unwrap();
        assert!(matches!(
            bound.constraint.kind,
            StructuralConstraintKind::DistributingCoupling(ReferencePointCoupling {
                reference_point: runmat_analysis_core::ReferencePoint { position, .. },
                ..
            }) if position.canonical_value() == [0.1, 0.0, 0.025]
        ));

        let conditions = crate::make_cell_with_shape(vec![prescribed, coupling], vec![1, 2])
            .expect("typed structural condition cell");
        let model = create_model_object_from_args(
            crate::builtins::common::test_support::builtin_arguments(vec![
                Value::String("typed_structural".into()),
                asset,
                Value::String("Steps".into()),
                super::super::modeling::static_test_steps_value(),
                Value::String("Constraints".into()),
                conditions,
            ]),
        )
        .expect("typed prescribed DOFs and coupling should compose into a model");
        let model: AnalysisModel = object_payload(MODEL_NAME, &model, FEA_MODEL_CLASS).unwrap();
        assert_eq!(model.constraints.len(), 2);

        ensure_value_classes_registered();
        for (class_name, method) in [
            ("fea.StructuralDof", "ux"),
            ("fea.StructuralDofValue", "translation"),
            ("fea.CoordinateFrame", "cartesian"),
            ("fea.ReferencePoint", "fromCoordinates"),
        ] {
            let (definition, _) = crate::class_registry::lookup_method(
                &runmat_types::ClassIdentity::from(class_name),
                &runmat_types::MethodName::from(method),
            )
            .unwrap_or_else(|| panic!("{class_name}.{method} should be registered"));
            assert!(definition.is_static);
        }
    }

    #[test]
    fn prescribed_support_motion_uses_typed_vector_acceleration_and_has_no_legacy_actions() {
        let tmp = tempfile::TempDir::new().unwrap();
        let asset = load_stl_fixture(&tmp.path().join("typed-motion.stl"), 1.0);
        let decoded = geometry_asset_from_value_with_builtin(&asset, STRUCTURAL_FIXED_NAME)
            .expect("asset should decode");
        let region = block_on(geometry_region_builtin(
            asset,
            decoded.regions[0].region_id.clone(),
        ))
        .expect("region");
        let acceleration = block_on(
            crate::builtins::fea::signals::time_samples::acceleration_time_samples(
                "support-acceleration".into(),
                Value::Tensor(Tensor::new_2d(vec![0.0, 1.0], 2, 1).unwrap()),
                Value::Tensor(Tensor::new_2d(vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0], 2, 3).unwrap()),
                block_on(crate::builtins::fea::signal_policies::interpolation_linear()).unwrap(),
                block_on(crate::builtins::fea::signal_policies::extrapolation_reject()).unwrap(),
            ),
        )
        .expect("typed acceleration history");
        let displacement = block_on(crate::builtins::fea::values::displacement_meters_builtin(
            Value::Num(0.25),
        ))
        .unwrap();
        let velocity = block_on(
            crate::builtins::fea::values::velocity_meters_per_second_builtin(Value::Num(-0.5)),
        )
        .unwrap();
        let ux = block_on(support::dof::structural_dof_ux_builtin(None)).unwrap();
        let motion = block_on(support::motion::structural_acceleration_motion_builtin(
            ux,
            acceleration,
            displacement,
            velocity,
        ))
        .expect("typed support motion");
        let constraint = block_on(structural::structural_prescribed_motion_builtin(
            "moving-base".into(),
            region,
            motion.clone(),
        ))
        .expect("typed prescribed-motion constraint");
        let bound: BoundStructuralConstraint = object_payload(
            "fea.StructuralConstraint.prescribedMotion",
            &constraint,
            FEA_STRUCTURAL_CONSTRAINT_CLASS,
        )
        .unwrap();
        let StructuralConstraintKind::PrescribedMotion(prescribed) = bound.constraint.kind else {
            panic!("expected prescribed support motion")
        };
        assert_eq!(prescribed.translations.len(), 1);
        assert!(matches!(
            &prescribed.translations[0].history,
            StructuralKinematicHistory::AccelerationDriven {
                acceleration: AccelerationSignalId(SignalId(id)),
                initial_displacement,
                initial_velocity,
                reconstruction: StructuralMotionReconstruction::PiecewiseLinearAccelerationExact,
            } if id == "support-acceleration"
                && initial_displacement.canonical_value() == 0.25
                && initial_velocity.canonical_value() == -0.5
        ));
        let duplicate = crate::make_cell_with_shape(vec![motion.clone(), motion], vec![1, 2])
            .expect("motion cell");
        assert!(support::motion::structural_motions_from_value("test", &duplicate).is_err());

        ensure_fea_classes_registered();
        assert!(crate::class_registry::lookup_method(
            &runmat_types::ClassIdentity::from("fea.StructuralMotion"),
            &runmat_types::MethodName::from("acceleration")
        )
        .is_some());
        assert!(crate::class_registry::lookup_method(
            &runmat_types::ClassIdentity::from("fea.StructuralConstraint"),
            &runmat_types::MethodName::from("prescribedMotion")
        )
        .is_some());
        for removed in ["baseDisplacement", "baseVelocity", "baseAcceleration"] {
            assert!(crate::class_registry::lookup_method(
                &runmat_types::ClassIdentity::from("fea.StructuralExcitation"),
                &runmat_types::MethodName::from(removed)
            )
            .is_none());
        }
    }

    #[test]
    fn coordinate_frames_validate_and_round_trip_without_string_dispatch() {
        let vector = |components: [f64; 3]| {
            block_on(support::direction::direction_vector_builtin(
                None,
                Value::Tensor(Tensor::new(components.to_vec(), vec![1, 3]).unwrap()),
            ))
            .unwrap()
        };
        let origin = block_on(crate::builtins::fea::values::position_meters_builtin(
            Value::Tensor(Tensor::new(vec![0.0, 0.0, 0.0], vec![1, 3]).unwrap()),
        ))
        .unwrap();
        let frame = block_on(support::frame::coordinate_frame_cartesian_builtin(
            None,
            "fixture".into(),
            origin.clone(),
            vector([1.0, 0.0, 0.0]),
            vector([0.0, 1.0, 0.0]),
        ))
        .expect("orthogonal axes should form a frame");
        assert!(matches!(
            support::coordinate_frame_from_value("test", &frame).unwrap(),
            CoordinateFrame::Cartesian(runmat_analysis_core::CartesianFrame {
                frame_id,
                x_axis: [1.0, 0.0, 0.0],
                y_axis: [0.0, 1.0, 0.0],
                ..
            }) if frame_id == "fixture"
        ));
        assert!(block_on(support::frame::coordinate_frame_cartesian_builtin(
            None,
            "parallel".into(),
            origin,
            vector([1.0, 0.0, 0.0]),
            vector([2.0, 0.0, 0.0]),
        ))
        .is_err());

        let payload = crate::execution::value_codec::encode_inline_value(&frame)
            .expect("typed frame should encode");
        let decoded = crate::execution::value_codec::decode_inline_value(&payload)
            .expect("typed frame should decode in a portable worker");
        assert!(matches!(
            support::coordinate_frame_from_value("test", &decoded).unwrap(),
            CoordinateFrame::Cartesian(_)
        ));
    }
}

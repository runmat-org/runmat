use super::*;
use crate::builtins::fea::constraints::{bound_region, fixed_outputs, require_geometry_binding};
use crate::builtins::fea::values::{
    quantity_scalar, ACOUSTIC_IMPEDANCE_CLASS, HEAT_FLUX_CLASS, HEAT_TRANSFER_COEFFICIENT_CLASS,
    PRESSURE_CLASS, TEMPERATURE_CLASS,
};

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct GeometryBoundBoundary {
    pub(super) geometry_id: String,
    pub(super) geometry_revision: u32,
    pub(super) boundary: AnalysisBoundary,
}

impl GeometryBoundBoundary {
    pub(super) fn require_geometry(
        &self,
        builtin: &'static str,
        geometry: &GeometryAsset,
    ) -> BuiltinResult<()> {
        require_geometry_binding(
            builtin,
            "boundary",
            &self.geometry_id,
            self.geometry_revision,
            geometry,
        )
    }

    pub(super) fn require_class(
        &self,
        builtin: &'static str,
        class_name: &'static str,
    ) -> BuiltinResult<()> {
        let expected = match &self.boundary.operator {
            BoundaryOperator::StructuralPressure { .. } => FEA_STRUCTURAL_BOUNDARY_CLASS,
            BoundaryOperator::ThermalHeatFlux { .. }
            | BoundaryOperator::ThermalConvection { .. }
            | BoundaryOperator::ThermalRadiation { .. } => FEA_THERMAL_BOUNDARY_CLASS,
            BoundaryOperator::MagneticInsulation => FEA_MAGNETIC_BOUNDARY_CLASS,
            BoundaryOperator::AcousticRigidWall
            | BoundaryOperator::AcousticRadiation
            | BoundaryOperator::AcousticSphericalRadiation { .. }
            | BoundaryOperator::AcousticImpedance { .. }
            | BoundaryOperator::AcousticImpedanceLaw { .. } => FEA_ACOUSTIC_BOUNDARY_CLASS,
            BoundaryOperator::FluidOutletPressure { .. } => FEA_FLUID_BOUNDARY_CLASS,
        };
        if class_name != expected {
            return Err(builtin_error(
                builtin,
                &ERROR_INPUT,
                format!("boundary operator belongs to `{expected}`, not `{class_name}`"),
            ));
        }
        Ok(())
    }
}

fn boundary_to_object(
    builtin: &'static str,
    class_name: &'static str,
    geometry_id: String,
    geometry_revision: u32,
    boundary: AnalysisBoundary,
) -> BuiltinResult<Value> {
    let mut object = serializable_to_object_value(
        builtin,
        &ERROR_INTERNAL,
        class_name,
        &GeometryBoundBoundary {
            geometry_id,
            geometry_revision,
            boundary,
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

fn bound_boundary(
    builtin: &'static str,
    class_name: &'static str,
    id: String,
    region: &Value,
    operator: BoundaryOperator,
) -> BuiltinResult<Value> {
    if id.is_empty() {
        return Err(builtin_error(
            builtin,
            &ERROR_INPUT,
            "boundary id must not be empty",
        ));
    }
    let (id, geometry_id, geometry_revision, target) = bound_region(builtin, id, region)?;
    boundary_to_object(
        builtin,
        class_name,
        geometry_id,
        geometry_revision,
        AnalysisBoundary {
            boundary_id: id,
            target,
            operator,
        },
    )
}

macro_rules! region_boundary {
    ($function:ident, $descriptor:ident, $name:literal, $class:expr, $operator:expr) => {
        #[runtime_builtin(
                                            name = $name,
                                            binding_variant = "default",
                                            builtin_path = "crate::builtins::fea::boundaries"
                                        )]
        pub async fn $function(id: String, region: Value) -> BuiltinResult<Value> {
            fixed_outputs(&runmat_builtins::$descriptor, $name)?;
            bound_boundary($name, $class, id, &region, $operator)
        }
    };
}

macro_rules! scalar_boundary {
    ($function:ident, $descriptor:ident, $name:literal, $class:expr, $quantity_class:expr, $field:literal, $build:expr) => {
        #[runtime_builtin(
                                            name = $name,
                                            binding_variant = "default",
                                            builtin_path = "crate::builtins::fea::boundaries"
                                        )]
        pub async fn $function(id: String, region: Value, value: Value) -> BuiltinResult<Value> {
            fixed_outputs(&runmat_builtins::$descriptor, $name)?;
            let value = quantity_scalar($name, &value, $quantity_class, $field)?;
            bound_boundary($name, $class, id, &region, ($build)(value))
        }
    };
}

scalar_boundary!(
    structural_pressure_builtin,
    FEA_STRUCTURAL_BOUNDARY_PRESSURE_DESCRIPTOR,
    "fea.StructuralBoundary.pressure",
    FEA_STRUCTURAL_BOUNDARY_CLASS,
    PRESSURE_CLASS,
    "pascals",
    |value| BoundaryOperator::StructuralPressure {
        value: PrescribedValue::Constant(Pressure::new(value))
    }
);
scalar_boundary!(
    thermal_heat_flux_builtin,
    FEA_THERMAL_BOUNDARY_HEAT_FLUX_DESCRIPTOR,
    "fea.ThermalBoundary.heatFlux",
    FEA_THERMAL_BOUNDARY_CLASS,
    HEAT_FLUX_CLASS,
    "watts_per_square_meter",
    |value| BoundaryOperator::ThermalHeatFlux {
        value: PrescribedValue::Constant(HeatFlux::new(value))
    }
);
region_boundary!(
    magnetic_insulation_builtin,
    FEA_MAGNETIC_BOUNDARY_INSULATION_DESCRIPTOR,
    "fea.MagneticBoundary.insulation",
    FEA_MAGNETIC_BOUNDARY_CLASS,
    BoundaryOperator::MagneticInsulation
);
region_boundary!(
    acoustic_rigid_wall_builtin,
    FEA_ACOUSTIC_BOUNDARY_RIGID_WALL_DESCRIPTOR,
    "fea.AcousticBoundary.rigidWall",
    FEA_ACOUSTIC_BOUNDARY_CLASS,
    BoundaryOperator::AcousticRigidWall
);
region_boundary!(
    acoustic_radiation_builtin,
    FEA_ACOUSTIC_BOUNDARY_RADIATION_DESCRIPTOR,
    "fea.AcousticBoundary.radiation",
    FEA_ACOUSTIC_BOUNDARY_CLASS,
    BoundaryOperator::AcousticRadiation
);
scalar_boundary!(
    acoustic_impedance_builtin,
    FEA_ACOUSTIC_BOUNDARY_IMPEDANCE_DESCRIPTOR,
    "fea.AcousticBoundary.impedance",
    FEA_ACOUSTIC_BOUNDARY_CLASS,
    ACOUSTIC_IMPEDANCE_CLASS,
    "pascal_seconds_per_meter",
    |value| BoundaryOperator::AcousticImpedance {
        impedance: SpecificAcousticImpedance::new(value)
    }
);
scalar_boundary!(
    fluid_outlet_pressure_builtin,
    FEA_FLUID_BOUNDARY_OUTLET_PRESSURE_DESCRIPTOR,
    "fea.FluidBoundary.outletPressure",
    FEA_FLUID_BOUNDARY_CLASS,
    PRESSURE_CLASS,
    "pascals",
    |value| BoundaryOperator::FluidOutletPressure {
        value: PrescribedValue::Constant(Pressure::new(value))
    }
);

#[runtime_builtin(
    name = "fea.ThermalBoundary.convection",
    binding_variant = "default",
    builtin_path = "crate::builtins::fea::boundaries"
)]
pub async fn thermal_convection_builtin(
    id: String,
    region: Value,
    ambient_temperature: Value,
    coefficient: Value,
) -> BuiltinResult<Value> {
    const NAME: &str = "fea.ThermalBoundary.convection";
    fixed_outputs(
        &runmat_builtins::FEA_THERMAL_BOUNDARY_CONVECTION_DESCRIPTOR,
        NAME,
    )?;
    let ambient_temperature = Temperature::new(quantity_scalar(
        NAME,
        &ambient_temperature,
        TEMPERATURE_CLASS,
        "kelvin",
    )?);
    let coefficient = HeatTransferCoefficient::new(quantity_scalar(
        NAME,
        &coefficient,
        HEAT_TRANSFER_COEFFICIENT_CLASS,
        "watts_per_square_meter_kelvin",
    )?);
    bound_boundary(
        NAME,
        FEA_THERMAL_BOUNDARY_CLASS,
        id,
        &region,
        BoundaryOperator::ThermalConvection {
            ambient_temperature: PrescribedValue::Constant(ambient_temperature),
            coefficient,
        },
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::builtins::fea::test_support::load_stl_fixture;
    use crate::builtins::fea::values::{
        heat_flux_watts_per_square_meter_builtin, heat_transfer_coefficient_builtin,
        pressure_pascals_builtin, specific_acoustic_impedance_builtin, temperature_kelvin_builtin,
    };
    use crate::builtins::geometry::geometry_asset_from_value_with_builtin;
    use crate::builtins::geometry::region::geometry_region_builtin;
    use futures::executor::block_on;

    fn first_region(asset: Value) -> Value {
        let decoded = geometry_asset_from_value_with_builtin(&asset, "fea.boundary.test")
            .expect("geometry should decode");
        block_on(geometry_region_builtin(
            asset,
            decoded.regions[0].region_id.clone(),
        ))
        .expect("region should resolve")
    }

    fn decode(value: &Value, class_name: &'static str) -> GeometryBoundBoundary {
        object_payload("fea.boundary.test", value, class_name).expect("boundary should decode")
    }

    #[test]
    fn family_boundary_constructors_produce_typed_geometry_bound_values() {
        let tmp = tempfile::TempDir::new().unwrap();
        let asset = load_stl_fixture(&tmp.path().join("part.stl"), 1.0);
        let select = || first_region(asset.clone());
        let pressure = block_on(pressure_pascals_builtin(Value::Num(12.0))).unwrap();
        let heat_flux =
            block_on(heat_flux_watts_per_square_meter_builtin(Value::Num(20.0))).unwrap();
        let temperature = block_on(temperature_kelvin_builtin(Value::Num(300.0))).unwrap();
        let coefficient = block_on(heat_transfer_coefficient_builtin(Value::Num(8.0))).unwrap();
        let impedance = block_on(specific_acoustic_impedance_builtin(Value::Num(400.0))).unwrap();

        let boundaries = [
            (
                block_on(structural_pressure_builtin(
                    "pressure".into(),
                    select(),
                    pressure.clone(),
                ))
                .unwrap(),
                FEA_STRUCTURAL_BOUNDARY_CLASS,
            ),
            (
                block_on(thermal_heat_flux_builtin(
                    "heat_flux".into(),
                    select(),
                    heat_flux,
                ))
                .unwrap(),
                FEA_THERMAL_BOUNDARY_CLASS,
            ),
            (
                block_on(thermal_convection_builtin(
                    "convection".into(),
                    select(),
                    temperature,
                    coefficient,
                ))
                .unwrap(),
                FEA_THERMAL_BOUNDARY_CLASS,
            ),
            (
                block_on(magnetic_insulation_builtin("insulation".into(), select())).unwrap(),
                FEA_MAGNETIC_BOUNDARY_CLASS,
            ),
            (
                block_on(acoustic_rigid_wall_builtin("rigid".into(), select())).unwrap(),
                FEA_ACOUSTIC_BOUNDARY_CLASS,
            ),
            (
                block_on(acoustic_radiation_builtin("radiation".into(), select())).unwrap(),
                FEA_ACOUSTIC_BOUNDARY_CLASS,
            ),
            (
                block_on(acoustic_impedance_builtin(
                    "impedance".into(),
                    select(),
                    impedance,
                ))
                .unwrap(),
                FEA_ACOUSTIC_BOUNDARY_CLASS,
            ),
            (
                block_on(fluid_outlet_pressure_builtin(
                    "outlet".into(),
                    select(),
                    pressure,
                ))
                .unwrap(),
                FEA_FLUID_BOUNDARY_CLASS,
            ),
        ];

        for (value, class_name) in boundaries {
            assert!(
                matches!(&value, Value::Object(object) if object.class_name().display_name() == class_name)
            );
            let boundary = decode(&value, class_name);
            let geometry = geometry_asset_from_value_with_builtin(&asset, "fea.boundary.test")
                .expect("geometry should decode");
            boundary
                .require_geometry("fea.boundary.test", &geometry)
                .expect("boundary should retain geometry identity");
            let encoded = crate::execution::value_codec::encode_inline_value(&value)
                .expect("typed boundary should cross an execution boundary");
            assert_eq!(
                crate::execution::value_codec::decode_inline_value(&encoded)
                    .expect("typed boundary should decode"),
                value
            );
        }
    }

    #[test]
    fn model_composition_separates_boundaries_and_rejects_foreign_geometry() {
        let tmp = tempfile::TempDir::new().unwrap();
        let asset = load_stl_fixture(&tmp.path().join("part-a.stl"), 1.0);
        let pressure = block_on(pressure_pascals_builtin(Value::Num(1.0))).unwrap();
        let boundary = block_on(structural_pressure_builtin(
            "pressure".into(),
            first_region(asset.clone()),
            pressure,
        ))
        .unwrap();
        let model = create_model_object_from_args(
            crate::builtins::common::test_support::builtin_arguments(vec![
                Value::String("model".into()),
                asset,
                Value::String("Steps".into()),
                super::super::modeling::static_test_steps_value(),
                Value::String("Boundaries".into()),
                boundary.clone(),
            ]),
        )
        .expect("matching boundary should compose");
        let model: AnalysisModel = object_payload(MODEL_NAME, &model, FEA_MODEL_CLASS).unwrap();
        assert_eq!(model.boundaries.len(), 1);

        let foreign = load_stl_fixture(&tmp.path().join("part-b.stl"), 2.0);
        let error = create_model_object_from_args(
            crate::builtins::common::test_support::builtin_arguments(vec![
                Value::String("foreign".into()),
                foreign,
                Value::String("Steps".into()),
                super::super::modeling::static_test_steps_value(),
                Value::String("Boundaries".into()),
                boundary.clone(),
            ]),
        )
        .expect_err("foreign boundary must be rejected");
        assert!(error.message().contains("belongs to geometry"));

        let Value::Object(mut mislabeled) = boundary else {
            unreachable!("boundary constructor returned an object")
        };
        mislabeled = crate::builtins::common::test_support::reclassify_object_unbounded(
            &mislabeled,
            FEA_THERMAL_BOUNDARY_CLASS,
        );
        let error = create_model_object_from_args(
            crate::builtins::common::test_support::builtin_arguments(vec![
                Value::String("mislabeled".into()),
                load_stl_fixture(&tmp.path().join("part-a-copy.stl"), 1.0),
                Value::String("Steps".into()),
                super::super::modeling::static_test_steps_value(),
                Value::String("Boundaries".into()),
                Value::Object(mislabeled),
            ]),
        )
        .expect_err("a boundary operator must match its family class");
        assert!(error.message().contains("boundary operator belongs to"));
    }

    #[test]
    fn boundary_constructors_reject_untyped_quantities_and_empty_ids() {
        let tmp = tempfile::TempDir::new().unwrap();
        let asset = load_stl_fixture(&tmp.path().join("part.stl"), 1.0);
        let region = first_region(asset);
        assert!(block_on(structural_pressure_builtin(
            "pressure".into(),
            region.clone(),
            Value::Num(1.0),
        ))
        .is_err());
        let pressure = block_on(pressure_pascals_builtin(Value::Num(1.0))).unwrap();
        assert!(block_on(structural_pressure_builtin("".into(), region, pressure)).is_err());
    }

    #[test]
    fn boundary_classes_expose_only_their_family_methods() {
        ensure_fea_classes_registered();
        for (class_name, method) in [
            (FEA_STRUCTURAL_BOUNDARY_CLASS, "pressure"),
            (FEA_THERMAL_BOUNDARY_CLASS, "heatFlux"),
            (FEA_THERMAL_BOUNDARY_CLASS, "convection"),
            (FEA_MAGNETIC_BOUNDARY_CLASS, "insulation"),
            (FEA_ACOUSTIC_BOUNDARY_CLASS, "rigidWall"),
            (FEA_ACOUSTIC_BOUNDARY_CLASS, "radiation"),
            (FEA_ACOUSTIC_BOUNDARY_CLASS, "impedance"),
            (FEA_FLUID_BOUNDARY_CLASS, "outletPressure"),
        ] {
            let (registered, owner) = crate::class_registry::lookup_method(
                &runmat_types::ClassIdentity::from(class_name),
                &runmat_types::MethodName::from(method),
            )
            .unwrap_or_else(|| panic!("missing {class_name}.{method}"));
            assert_eq!(owner.display_name(), class_name);
            assert!(registered.is_static);
        }
        assert!(crate::class_registry::lookup_method(
            &runmat_types::ClassIdentity::from(FEA_FLUID_BOUNDARY_CLASS),
            &runmat_types::MethodName::from("prescribedVelocity")
        )
        .is_none());
    }
}

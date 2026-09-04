use super::*;
use crate::builtins::fea::constraints::{bound_region, require_geometry_binding};
use crate::builtins::fea::values::{
    quantity_scalar, ACOUSTIC_IMPEDANCE_CLASS, HEAT_FLUX_CLASS, HEAT_TRANSFER_COEFFICIENT_CLASS,
    PRESSURE_CLASS, TEMPERATURE_CLASS,
};

const TYPED_BOUNDARY_INTEGER_AUDIT: BuiltinIntegerAuditDescriptor =
    BuiltinIntegerAuditDescriptor {
        kind: BuiltinIntegerAuditKind::NotApplicable,
        canonical_builtin: None,
        notes: "Typed FEA boundary constructors accept identifiers, immutable geometry.Region values, and typed physical quantities; they have no integer input role.",
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
            | BoundaryOperator::ThermalConvection { .. } => FEA_THERMAL_BOUNDARY_CLASS,
            BoundaryOperator::MagneticInsulation => FEA_MAGNETIC_BOUNDARY_CLASS,
            BoundaryOperator::AcousticRigidWall
            | BoundaryOperator::AcousticRadiation
            | BoundaryOperator::AcousticImpedance { .. } => FEA_ACOUSTIC_BOUNDARY_CLASS,
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

const REGION_INPUTS: [BuiltinParamDescriptor; 2] = [
    required("id", BuiltinParamType::StringScalar, "Boundary identifier."),
    required("region", BuiltinParamType::Any, "Typed geometry.Region."),
];
const VALUE_INPUTS: [BuiltinParamDescriptor; 3] = [
    required("id", BuiltinParamType::StringScalar, "Boundary identifier."),
    required("region", BuiltinParamType::Any, "Typed geometry.Region."),
    required(
        "value",
        BuiltinParamType::Any,
        "Typed dimensioned boundary value.",
    ),
];
const CONVECTION_INPUTS: [BuiltinParamDescriptor; 4] = [
    required("id", BuiltinParamType::StringScalar, "Boundary identifier."),
    required("region", BuiltinParamType::Any, "Typed geometry.Region."),
    required(
        "ambient_temperature",
        BuiltinParamType::Any,
        "Typed fea.Temperature.",
    ),
    required(
        "coefficient",
        BuiltinParamType::Any,
        "Typed fea.HeatTransferCoefficient.",
    ),
];

const fn required(
    name: &'static str,
    ty: BuiltinParamType,
    description: &'static str,
) -> BuiltinParamDescriptor {
    BuiltinParamDescriptor {
        name,
        ty,
        arity: BuiltinParamArity::Required,
        default: None,
        description,
    }
}

macro_rules! descriptor {
    ($descriptor:ident, $signatures:ident, $label:literal, $inputs:ident) => {
        const $signatures: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
            label: $label,
            inputs: &$inputs,
            outputs: &OUT_ANY,
        }];
        pub(super) const $descriptor: BuiltinDescriptor = BuiltinDescriptor {
            signatures: &$signatures,
            output_mode: BuiltinOutputMode::Fixed,
            completion_policy: BuiltinCompletionPolicy::Public,
            errors: &ERRORS,
        };
    };
}

descriptor!(
    STRUCTURAL_PRESSURE_DESCRIPTOR,
    STRUCTURAL_PRESSURE_SIGNATURES,
    "boundary = fea.StructuralBoundary.pressure(id, region, pressure)",
    VALUE_INPUTS
);
descriptor!(
    THERMAL_HEAT_FLUX_DESCRIPTOR,
    THERMAL_HEAT_FLUX_SIGNATURES,
    "boundary = fea.ThermalBoundary.heatFlux(id, region, heatFlux)",
    VALUE_INPUTS
);
descriptor!(
    THERMAL_CONVECTION_DESCRIPTOR,
    THERMAL_CONVECTION_SIGNATURES,
    "boundary = fea.ThermalBoundary.convection(id, region, ambientTemperature, coefficient)",
    CONVECTION_INPUTS
);
descriptor!(
    MAGNETIC_INSULATION_DESCRIPTOR,
    MAGNETIC_INSULATION_SIGNATURES,
    "boundary = fea.MagneticBoundary.insulation(id, region)",
    REGION_INPUTS
);
descriptor!(
    ACOUSTIC_RIGID_WALL_DESCRIPTOR,
    ACOUSTIC_RIGID_WALL_SIGNATURES,
    "boundary = fea.AcousticBoundary.rigidWall(id, region)",
    REGION_INPUTS
);
descriptor!(
    ACOUSTIC_RADIATION_DESCRIPTOR,
    ACOUSTIC_RADIATION_SIGNATURES,
    "boundary = fea.AcousticBoundary.radiation(id, region)",
    REGION_INPUTS
);
descriptor!(
    ACOUSTIC_IMPEDANCE_DESCRIPTOR,
    ACOUSTIC_IMPEDANCE_SIGNATURES,
    "boundary = fea.AcousticBoundary.impedance(id, region, impedance)",
    VALUE_INPUTS
);
descriptor!(
    FLUID_OUTLET_PRESSURE_DESCRIPTOR,
    FLUID_OUTLET_PRESSURE_SIGNATURES,
    "boundary = fea.FluidBoundary.outletPressure(id, region, pressure)",
    VALUE_INPUTS
);

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
    object.properties.insert(
        "geometry_revision".into(),
        Value::Int(IntValue::U32(geometry_revision)),
    );
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
    ($function:ident, $descriptor:ident, $name:literal, $class:expr, $summary:literal, $operator:expr) => {
        #[runtime_builtin(
                                    name = $name,
                                    category = "fea",
                                    summary = $summary,
                                    keywords = "fea,boundary,region,typed",
                                    descriptor(crate::builtins::fea::boundaries::$descriptor),
                                    integer_audit(TYPED_BOUNDARY_INTEGER_AUDIT),
                                    builtin_path = "crate::builtins::fea::boundaries"
                                )]
        pub async fn $function(id: String, region: Value) -> BuiltinResult<Value> {
            bound_boundary($name, $class, id, &region, $operator)
        }
    };
}

macro_rules! scalar_boundary {
    ($function:ident, $descriptor:ident, $name:literal, $class:expr, $summary:literal, $quantity_class:expr, $field:literal, $build:expr) => {
        #[runtime_builtin(
                                    name = $name,
                                    category = "fea",
                                    summary = $summary,
                                    keywords = "fea,boundary,region,quantity,typed",
                                    descriptor(crate::builtins::fea::boundaries::$descriptor),
                                    integer_audit(TYPED_BOUNDARY_INTEGER_AUDIT),
                                    builtin_path = "crate::builtins::fea::boundaries"
                                )]
        pub async fn $function(id: String, region: Value, value: Value) -> BuiltinResult<Value> {
            let value = quantity_scalar($name, &value, $quantity_class, $field)?;
            bound_boundary($name, $class, id, &region, ($build)(value))
        }
    };
}

scalar_boundary!(
    structural_pressure_builtin,
    STRUCTURAL_PRESSURE_DESCRIPTOR,
    "fea.StructuralBoundary.pressure",
    FEA_STRUCTURAL_BOUNDARY_CLASS,
    "Create a structural pressure boundary.",
    PRESSURE_CLASS,
    "pascals",
    |value| BoundaryOperator::StructuralPressure {
        value: PrescribedValue::Constant(Pressure::new(value))
    }
);
scalar_boundary!(
    thermal_heat_flux_builtin,
    THERMAL_HEAT_FLUX_DESCRIPTOR,
    "fea.ThermalBoundary.heatFlux",
    FEA_THERMAL_BOUNDARY_CLASS,
    "Create a prescribed thermal heat-flux boundary.",
    HEAT_FLUX_CLASS,
    "watts_per_square_meter",
    |value| BoundaryOperator::ThermalHeatFlux {
        value: PrescribedValue::Constant(HeatFlux::new(value))
    }
);
region_boundary!(
    magnetic_insulation_builtin,
    MAGNETIC_INSULATION_DESCRIPTOR,
    "fea.MagneticBoundary.insulation",
    FEA_MAGNETIC_BOUNDARY_CLASS,
    "Create a magnetic-insulation boundary.",
    BoundaryOperator::MagneticInsulation
);
region_boundary!(
    acoustic_rigid_wall_builtin,
    ACOUSTIC_RIGID_WALL_DESCRIPTOR,
    "fea.AcousticBoundary.rigidWall",
    FEA_ACOUSTIC_BOUNDARY_CLASS,
    "Create an acoustic rigid-wall boundary.",
    BoundaryOperator::AcousticRigidWall
);
region_boundary!(
    acoustic_radiation_builtin,
    ACOUSTIC_RADIATION_DESCRIPTOR,
    "fea.AcousticBoundary.radiation",
    FEA_ACOUSTIC_BOUNDARY_CLASS,
    "Create an acoustic radiation boundary.",
    BoundaryOperator::AcousticRadiation
);
scalar_boundary!(
    acoustic_impedance_builtin,
    ACOUSTIC_IMPEDANCE_DESCRIPTOR,
    "fea.AcousticBoundary.impedance",
    FEA_ACOUSTIC_BOUNDARY_CLASS,
    "Create a specific acoustic-impedance boundary.",
    ACOUSTIC_IMPEDANCE_CLASS,
    "pascal_seconds_per_meter",
    |value| BoundaryOperator::AcousticImpedance {
        impedance: SpecificAcousticImpedance::new(value)
    }
);
scalar_boundary!(
    fluid_outlet_pressure_builtin,
    FLUID_OUTLET_PRESSURE_DESCRIPTOR,
    "fea.FluidBoundary.outletPressure",
    FEA_FLUID_BOUNDARY_CLASS,
    "Create a fluid outlet-pressure boundary.",
    PRESSURE_CLASS,
    "pascals",
    |value| BoundaryOperator::FluidOutletPressure {
        value: PrescribedValue::Constant(Pressure::new(value))
    }
);

#[runtime_builtin(
    name = "fea.ThermalBoundary.convection",
    category = "fea",
    summary = "Create a thermal convection boundary.",
    keywords = "fea,thermal,boundary,convection,temperature,coefficient,typed",
    descriptor(crate::builtins::fea::boundaries::THERMAL_CONVECTION_DESCRIPTOR),
    integer_audit(TYPED_BOUNDARY_INTEGER_AUDIT),
    builtin_path = "crate::builtins::fea::boundaries"
)]
pub async fn thermal_convection_builtin(
    id: String,
    region: Value,
    ambient_temperature: Value,
    coefficient: Value,
) -> BuiltinResult<Value> {
    let ambient_temperature = Temperature::new(quantity_scalar(
        "fea.ThermalBoundary.convection",
        &ambient_temperature,
        TEMPERATURE_CLASS,
        "kelvin",
    )?);
    let coefficient = HeatTransferCoefficient::new(quantity_scalar(
        "fea.ThermalBoundary.convection",
        &coefficient,
        HEAT_TRANSFER_COEFFICIENT_CLASS,
        "watts_per_square_meter_kelvin",
    )?);
    bound_boundary(
        "fea.ThermalBoundary.convection",
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
            assert!(matches!(&value, Value::Object(object) if object.class_name == class_name));
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
        let model = create_model_object_from_args(vec![
            Value::String("model".into()),
            asset,
            Value::String("Template".into()),
            Value::String("linear_static_structural".into()),
            Value::String("Boundaries".into()),
            boundary.clone(),
        ])
        .expect("matching boundary should compose");
        let model: AnalysisModel = object_payload(MODEL_NAME, &model, FEA_MODEL_CLASS).unwrap();
        assert_eq!(model.boundaries.len(), 1);

        let foreign = load_stl_fixture(&tmp.path().join("part-b.stl"), 2.0);
        let error = create_model_object_from_args(vec![
            Value::String("foreign".into()),
            foreign,
            Value::String("Template".into()),
            Value::String("linear_static_structural".into()),
            Value::String("Boundaries".into()),
            boundary.clone(),
        ])
        .expect_err("foreign boundary must be rejected");
        assert!(error.message().contains("belongs to geometry"));

        let Value::Object(mut mislabeled) = boundary else {
            unreachable!("boundary constructor returned an object")
        };
        mislabeled.class_name = FEA_THERMAL_BOUNDARY_CLASS.into();
        let error = create_model_object_from_args(vec![
            Value::String("mislabeled".into()),
            load_stl_fixture(&tmp.path().join("part-a-copy.stl"), 1.0),
            Value::String("Template".into()),
            Value::String("linear_static_structural".into()),
            Value::String("Boundaries".into()),
            Value::Object(mislabeled),
        ])
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
            let (registered, owner) = crate::class_registry::lookup_method(class_name, method)
                .unwrap_or_else(|| panic!("missing {class_name}.{method}"));
            assert_eq!(owner, class_name);
            assert!(registered.is_static);
        }
        assert!(crate::class_registry::lookup_method(
            FEA_FLUID_BOUNDARY_CLASS,
            "prescribedVelocity"
        )
        .is_none());
    }
}

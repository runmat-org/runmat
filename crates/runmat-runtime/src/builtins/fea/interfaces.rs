use super::*;
use crate::builtins::fea::constraints::{bound_region_target, require_geometry_binding};
use crate::builtins::fea::values::{
    quantity_scalar, FRICTION_COEFFICIENT_CLASS, HEAT_TRANSFER_COEFFICIENT_CLASS,
    THERMAL_CONTACT_RESISTANCE_CLASS,
};

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct GeometryBoundInterface {
    pub(super) geometry_id: String,
    pub(super) geometry_revision: u32,
    pub(super) interface: AnalysisInterface,
}

impl GeometryBoundInterface {
    pub(super) fn require_geometry(
        &self,
        builtin: &'static str,
        geometry: &GeometryAsset,
    ) -> BuiltinResult<()> {
        require_geometry_binding(
            builtin,
            "interface",
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
        let expected = match self.interface.kind {
            AnalysisInterfaceKind::Contact(_) => FEA_CONTACT_INTERFACE_CLASS,
            AnalysisInterfaceKind::FluidStructure(_) => FEA_FLUID_STRUCTURE_INTERFACE_CLASS,
            AnalysisInterfaceKind::ConjugateHeatTransfer(_) => {
                FEA_CONJUGATE_HEAT_TRANSFER_INTERFACE_CLASS
            }
        };
        if class_name != expected {
            return Err(builtin_error(
                builtin,
                &ERROR_INPUT,
                format!("interface kind belongs to `{expected}`, not `{class_name}`"),
            ));
        }
        Ok(())
    }
}

fn bound_interface(
    builtin: &'static str,
    class_name: &'static str,
    id: Value,
    primary: &Value,
    secondary: &Value,
    kind: AnalysisInterfaceKind,
) -> BuiltinResult<Value> {
    let id = scalar_string(&id, builtin, &ERROR_INPUT)?;
    if id.is_empty() {
        return Err(builtin_error(
            builtin,
            &ERROR_INPUT,
            "interface id must not be empty",
        ));
    }
    let (geometry_id, geometry_revision, primary) = bound_region_target(builtin, primary)?;
    let (secondary_geometry_id, secondary_revision, secondary) =
        bound_region_target(builtin, secondary)?;
    if geometry_id != secondary_geometry_id || geometry_revision != secondary_revision {
        return Err(builtin_error(
            builtin,
            &ERROR_INPUT,
            "interface regions must belong to the same geometry revision",
        ));
    }
    if primary.region_id == secondary.region_id {
        return Err(builtin_error(
            builtin,
            &ERROR_INPUT,
            "interfaces require distinct primary and secondary regions",
        ));
    }
    let value = GeometryBoundInterface {
        geometry_id,
        geometry_revision,
        interface: AnalysisInterface {
            interface_id: id,
            primary,
            secondary,
            kind,
        },
    };
    let mut object = serializable_to_object_value(builtin, &ERROR_INTERNAL, class_name, &value)?;
    crate::builtins::fea::object_api::replace(
        builtin,
        &mut object,
        "geometry_revision",
        Value::Int(IntValue::U32(geometry_revision)),
    )?;
    Ok(Value::Object(object))
}

fn positive_quantity_scalar(
    builtin: &'static str,
    value: &Value,
    class_name: &'static str,
    property: &'static str,
) -> BuiltinResult<f64> {
    let scalar = quantity_scalar(builtin, value, class_name, property)?;
    if !scalar.is_finite() || scalar <= 0.0 {
        return Err(builtin_error(
            builtin,
            &ERROR_INPUT,
            format!("{class_name} must contain a finite positive `{property}` value"),
        ));
    }
    Ok(scalar)
}

#[runtime_builtin(
    name = "fea.ContactInterface.frictionless",
    binding_variant = "default",
    builtin_path = "crate::builtins::fea::interfaces"
)]
pub async fn contact_interface_frictionless_builtin(
    id: Value,
    primary: Value,
    secondary: Value,
) -> BuiltinResult<Value> {
    crate::builtins::fea::constraints::fixed_outputs(
        &runmat_builtins::FEA_CONTACT_INTERFACE_FRICTIONLESS_DESCRIPTOR,
        "fea.ContactInterface.frictionless",
    )?;
    bound_interface(
        "fea.ContactInterface.frictionless",
        FEA_CONTACT_INTERFACE_CLASS,
        id,
        &primary,
        &secondary,
        AnalysisInterfaceKind::Contact(ContactInterfaceModel {
            friction: ContactFriction::Frictionless,
        }),
    )
}

#[runtime_builtin(
    name = "fea.ContactInterface.coulomb",
    binding_variant = "default",
    builtin_path = "crate::builtins::fea::interfaces"
)]
pub async fn contact_interface_coulomb_builtin(
    id: Value,
    primary: Value,
    secondary: Value,
    friction: Value,
) -> BuiltinResult<Value> {
    crate::builtins::fea::constraints::fixed_outputs(
        &runmat_builtins::FEA_CONTACT_INTERFACE_COULOMB_DESCRIPTOR,
        "fea.ContactInterface.coulomb",
    )?;
    let friction = FrictionCoefficient::new(positive_quantity_scalar(
        "fea.ContactInterface.coulomb",
        &friction,
        FRICTION_COEFFICIENT_CLASS,
        "ratio",
    )?);
    bound_interface(
        "fea.ContactInterface.coulomb",
        FEA_CONTACT_INTERFACE_CLASS,
        id,
        &primary,
        &secondary,
        AnalysisInterfaceKind::Contact(ContactInterfaceModel {
            friction: ContactFriction::Coulomb {
                coefficient: friction,
            },
        }),
    )
}

#[runtime_builtin(
    name = "fea.FluidStructureInterface.noSlip",
    binding_variant = "default",
    builtin_path = "crate::builtins::fea::interfaces"
)]
pub async fn fluid_structure_interface_builtin(
    id: Value,
    fluid_region: Value,
    structure_region: Value,
) -> BuiltinResult<Value> {
    crate::builtins::fea::constraints::fixed_outputs(
        &runmat_builtins::FEA_FLUID_STRUCTURE_INTERFACE_NO_SLIP_DESCRIPTOR,
        "fea.FluidStructureInterface.noSlip",
    )?;
    bound_interface(
        "fea.FluidStructureInterface.noSlip",
        FEA_FLUID_STRUCTURE_INTERFACE_CLASS,
        id,
        &fluid_region,
        &structure_region,
        AnalysisInterfaceKind::FluidStructure(FluidStructureInterfaceModel {
            kinematics: FluidStructureKinematics::NoSlip,
        }),
    )
}

#[runtime_builtin(
    name = "fea.ConjugateHeatTransferInterface.perfect",
    binding_variant = "default",
    builtin_path = "crate::builtins::fea::interfaces"
)]
pub async fn conjugate_heat_transfer_interface_perfect_builtin(
    id: Value,
    fluid_region: Value,
    solid_region: Value,
) -> BuiltinResult<Value> {
    crate::builtins::fea::constraints::fixed_outputs(
        &runmat_builtins::FEA_CONJUGATE_HEAT_TRANSFER_INTERFACE_PERFECT_DESCRIPTOR,
        "fea.ConjugateHeatTransferInterface.perfect",
    )?;
    bound_interface(
        "fea.ConjugateHeatTransferInterface.perfect",
        FEA_CONJUGATE_HEAT_TRANSFER_INTERFACE_CLASS,
        id,
        &fluid_region,
        &solid_region,
        AnalysisInterfaceKind::ConjugateHeatTransfer(ConjugateHeatTransferInterfaceModel {
            transfer: ThermalInterfaceTransfer::Perfect,
        }),
    )
}

#[runtime_builtin(
    name = "fea.ConjugateHeatTransferInterface.conductance",
    binding_variant = "default",
    builtin_path = "crate::builtins::fea::interfaces"
)]
pub async fn conjugate_heat_transfer_interface_conductance_builtin(
    id: Value,
    fluid_region: Value,
    solid_region: Value,
    conductance: Value,
) -> BuiltinResult<Value> {
    crate::builtins::fea::constraints::fixed_outputs(
        &runmat_builtins::FEA_CONJUGATE_HEAT_TRANSFER_INTERFACE_CONDUCTANCE_DESCRIPTOR,
        "fea.ConjugateHeatTransferInterface.conductance",
    )?;
    let conductance = HeatTransferCoefficient::new(positive_quantity_scalar(
        "fea.ConjugateHeatTransferInterface.conductance",
        &conductance,
        HEAT_TRANSFER_COEFFICIENT_CLASS,
        "watts_per_square_meter_kelvin",
    )?);
    bound_interface(
        "fea.ConjugateHeatTransferInterface.conductance",
        FEA_CONJUGATE_HEAT_TRANSFER_INTERFACE_CLASS,
        id,
        &fluid_region,
        &solid_region,
        AnalysisInterfaceKind::ConjugateHeatTransfer(ConjugateHeatTransferInterfaceModel {
            transfer: ThermalInterfaceTransfer::Conductance(conductance),
        }),
    )
}

#[runtime_builtin(
    name = "fea.ConjugateHeatTransferInterface.resistance",
    binding_variant = "default",
    builtin_path = "crate::builtins::fea::interfaces"
)]
pub async fn conjugate_heat_transfer_interface_resistance_builtin(
    id: Value,
    fluid_region: Value,
    solid_region: Value,
    resistance: Value,
) -> BuiltinResult<Value> {
    crate::builtins::fea::constraints::fixed_outputs(
        &runmat_builtins::FEA_CONJUGATE_HEAT_TRANSFER_INTERFACE_RESISTANCE_DESCRIPTOR,
        "fea.ConjugateHeatTransferInterface.resistance",
    )?;
    let resistance = ThermalContactResistance::new(positive_quantity_scalar(
        "fea.ConjugateHeatTransferInterface.resistance",
        &resistance,
        THERMAL_CONTACT_RESISTANCE_CLASS,
        "square_meter_kelvin_per_watt",
    )?);
    bound_interface(
        "fea.ConjugateHeatTransferInterface.resistance",
        FEA_CONJUGATE_HEAT_TRANSFER_INTERFACE_CLASS,
        id,
        &fluid_region,
        &solid_region,
        AnalysisInterfaceKind::ConjugateHeatTransfer(ConjugateHeatTransferInterfaceModel {
            transfer: ThermalInterfaceTransfer::Resistance(resistance),
        }),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::builtins::fea::test_support::load_stl_fixture;
    use crate::builtins::fea::values::{
        friction_coefficient_builtin, heat_transfer_coefficient_builtin,
        thermal_contact_resistance_builtin,
    };
    use crate::builtins::geometry::geometry_asset_from_value_with_builtin;
    use crate::builtins::geometry::region::geometry_region_builtin;
    use futures::executor::block_on;

    fn regions(asset: &Value) -> (Value, Value) {
        let decoded = geometry_asset_from_value_with_builtin(asset, "fea.typed-interface.test")
            .expect("geometry should decode");
        let primary = block_on(geometry_region_builtin(
            asset.clone(),
            decoded.regions[0].region_id.clone(),
        ))
        .expect("region should resolve");
        let mut secondary = primary.clone();
        let Value::Object(object) = &mut secondary else {
            panic!("geometry.region should be an object")
        };
        crate::builtins::fea::object_api::replace(
            "fea.fixture",
            object,
            "region_id",
            Value::String("secondary".into()),
        )
        .expect("ordinary fixture property edit");
        (primary, secondary)
    }

    #[test]
    fn concrete_interfaces_are_geometry_bound_portable_and_class_checked() {
        let tmp = tempfile::TempDir::new().unwrap();
        let asset = load_stl_fixture(&tmp.path().join("part.stl"), 1.0);
        let geometry = geometry_asset_from_value_with_builtin(&asset, "fea.typed-interface.test")
            .expect("geometry should decode");
        let (primary, secondary) = regions(&asset);
        let friction = block_on(friction_coefficient_builtin(Value::Num(0.2))).unwrap();
        let conductance = block_on(heat_transfer_coefficient_builtin(Value::Num(500.0))).unwrap();
        let resistance = block_on(thermal_contact_resistance_builtin(Value::Num(0.002))).unwrap();
        let values = [
            (
                block_on(contact_interface_frictionless_builtin(
                    Value::String("frictionless".into()),
                    primary.clone(),
                    secondary.clone(),
                ))
                .unwrap(),
                FEA_CONTACT_INTERFACE_CLASS,
            ),
            (
                block_on(contact_interface_coulomb_builtin(
                    Value::String("contact".into()),
                    primary.clone(),
                    secondary.clone(),
                    friction,
                ))
                .unwrap(),
                FEA_CONTACT_INTERFACE_CLASS,
            ),
            (
                block_on(fluid_structure_interface_builtin(
                    Value::String("fsi".into()),
                    primary.clone(),
                    secondary.clone(),
                ))
                .unwrap(),
                FEA_FLUID_STRUCTURE_INTERFACE_CLASS,
            ),
            (
                block_on(conjugate_heat_transfer_interface_perfect_builtin(
                    Value::String("perfect".into()),
                    primary.clone(),
                    secondary.clone(),
                ))
                .unwrap(),
                FEA_CONJUGATE_HEAT_TRANSFER_INTERFACE_CLASS,
            ),
            (
                block_on(conjugate_heat_transfer_interface_conductance_builtin(
                    Value::String("cht".into()),
                    primary.clone(),
                    secondary.clone(),
                    conductance,
                ))
                .unwrap(),
                FEA_CONJUGATE_HEAT_TRANSFER_INTERFACE_CLASS,
            ),
            (
                block_on(conjugate_heat_transfer_interface_resistance_builtin(
                    Value::String("resistance".into()),
                    primary,
                    secondary,
                    resistance,
                ))
                .unwrap(),
                FEA_CONJUGATE_HEAT_TRANSFER_INTERFACE_CLASS,
            ),
        ];

        for (value, class_name) in values {
            assert!(
                matches!(&value, Value::Object(object) if object.class_name().display_name() == class_name)
            );
            let interface: GeometryBoundInterface =
                object_payload("fea.typed-interface.test", &value, class_name).unwrap();
            interface
                .require_geometry("fea.typed-interface.test", &geometry)
                .unwrap();
            interface
                .require_class("fea.typed-interface.test", class_name)
                .unwrap();
            let encoded = crate::execution::value_codec::encode_inline_value(&value).unwrap();
            assert_eq!(
                crate::execution::value_codec::decode_inline_value(&encoded).unwrap(),
                value
            );
        }
    }

    #[test]
    fn interface_construction_rejects_invalid_identity_and_pairing() {
        let tmp = tempfile::TempDir::new().unwrap();
        let asset = load_stl_fixture(&tmp.path().join("part.stl"), 1.0);
        let (primary, secondary) = regions(&asset);

        let numeric_id = block_on(contact_interface_frictionless_builtin(
            Value::Int(IntValue::U8(7)),
            primary.clone(),
            secondary,
        ))
        .expect_err("numeric ids must not stringify");
        assert_eq!(numeric_id.identifier(), Some("RunMat:fea:InvalidInput"));

        let same_region = block_on(contact_interface_frictionless_builtin(
            Value::String("contact".into()),
            primary.clone(),
            primary,
        ))
        .expect_err("an interface must join distinct regions");
        assert!(same_region.message().contains("distinct"));
    }

    #[test]
    fn interface_parameter_classes_enforce_semantic_ranges() {
        assert!(block_on(friction_coefficient_builtin(Value::Num(-0.1))).is_err());
        assert!(block_on(thermal_contact_resistance_builtin(Value::Num(-0.1))).is_err());

        let tmp = tempfile::TempDir::new().unwrap();
        let asset = load_stl_fixture(&tmp.path().join("part.stl"), 1.0);
        let (primary, secondary) = regions(&asset);
        let zero_friction = block_on(friction_coefficient_builtin(Value::Num(0.0))).unwrap();
        assert!(block_on(contact_interface_coulomb_builtin(
            Value::String("contact".into()),
            primary.clone(),
            secondary.clone(),
            zero_friction,
        ))
        .is_err());
        let zero_resistance =
            block_on(thermal_contact_resistance_builtin(Value::Num(0.0))).unwrap();
        assert!(
            block_on(conjugate_heat_transfer_interface_resistance_builtin(
                Value::String("cht".into()),
                primary,
                secondary,
                zero_resistance,
            ))
            .is_err()
        );
    }

    #[test]
    fn model_composition_accepts_matching_interfaces_and_rejects_foreign_geometry() {
        let tmp = tempfile::TempDir::new().unwrap();
        let asset = load_stl_fixture(&tmp.path().join("part-a.stl"), 1.0);
        let (primary, secondary) = regions(&asset);
        let interface = block_on(contact_interface_frictionless_builtin(
            Value::String("contact".into()),
            primary,
            secondary,
        ))
        .unwrap();

        let model = create_model_object_from_args(
            crate::builtins::common::test_support::builtin_arguments(vec![
                Value::String("model".into()),
                asset,
                Value::String("Steps".into()),
                super::super::modeling::static_test_steps_value(),
                Value::String("Interfaces".into()),
                interface.clone(),
            ]),
        )
        .expect("matching interface should compose");
        let model: AnalysisModel = object_payload(MODEL_NAME, &model, FEA_MODEL_CLASS).unwrap();
        assert_eq!(model.interfaces.len(), 1);
        assert!(!model.interfaces[0].primary.region_id.is_empty());
        assert_eq!(model.interfaces[0].secondary.region_id, "secondary");

        let foreign = load_stl_fixture(&tmp.path().join("part-b.stl"), 2.0);
        let error = create_model_object_from_args(
            crate::builtins::common::test_support::builtin_arguments(vec![
                Value::String("foreign".into()),
                foreign,
                Value::String("Steps".into()),
                super::super::modeling::static_test_steps_value(),
                Value::String("Interfaces".into()),
                interface,
            ]),
        )
        .expect_err("foreign interface must be rejected");
        assert!(error.message().contains("belongs to geometry"));
    }
}

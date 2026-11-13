use super::*;
use crate::PhysicalDimension;
#[test]
fn shared_kinds_preserve_same_dimension_semantics() {
    assert_eq!(
        QuantityKind::Moment.dimension(),
        QuantityKind::Energy.dimension()
    );
    assert_ne!(QuantityKind::Moment, QuantityKind::Energy);
    assert_ne!(
        QuantityKind::Temperature,
        QuantityKind::TemperatureDifference
    );
}
#[test]
fn electrical_and_thermal_products_are_typed() {
    let p: Power = ElectricPotential::new(12.0) * ElectricCurrent::new(2.0);
    assert_eq!(p.canonical_value(), 24.0);
    let h: HeatFlux = HeatTransferCoefficient::new(5.0) * TemperatureDifference::new(10.0);
    let p: Power = h * Area::new(2.0);
    assert_eq!(p.canonical_value(), 100.0);
    let difference: TemperatureDifference = Temperature::new(300.0) - Temperature::new(280.0);
    assert_eq!(difference.canonical_value(), 20.0);
}
#[test]
fn authored_units_share_one_conversion_authority() {
    let mm =
        Unit::<kind::Length>::try_from_definition(parse_unit_expression("mm").unwrap()).unwrap();
    assert_eq!(mm.quantity(40.0).unwrap().canonical_value(), 0.04);
    let inch = LengthUnit::Inch;
    assert_eq!(inch.quantity(1.0).unwrap().canonical_value(), 0.0254);
    assert_eq!(
        parse_unit_expression("W/A").unwrap().dimension(),
        QuantityKind::ElectricPotential.dimension()
    );
    assert_eq!(
        parse_unit_expression("W/m^2").unwrap().dimension(),
        QuantityKind::HeatFlux.dimension()
    );
}
#[test]
fn affine_units_require_absolute_temperature() {
    let definition = parse_unit_expression("degC").unwrap();
    let unit = Unit::<kind::Temperature>::try_from_definition(definition).unwrap();
    assert_eq!(unit.quantity(0.0).unwrap().canonical_value(), 273.15);
    assert!(Unit::<kind::TemperatureDifference>::try_from_definition(definition).is_err());
    assert!(Unit::<kind::ElectricCurrent>::try_from_definition(definition).is_err());
}
#[test]
fn quantity_wire_names_remain_canonical_and_wrong_dimension_is_rejected() {
    let current = ElectricCurrent::new(2.0);
    assert_eq!(
        serde_json::to_value(current).unwrap(),
        serde_json::json!({"amperes":2.0})
    );
    assert!(serde_json::from_value::<ElectricCurrent>(serde_json::json!({"volts":2.0})).is_err());
    let position = Position::new([1.0, 2.0, 3.0]);
    assert_eq!(
        serde_json::to_value(position).unwrap(),
        serde_json::json!({"meters":[1.0,2.0,3.0]})
    );
    assert!(ElectricCurrent::try_new(f64::NAN).is_err());
}
#[test]
fn invalid_unit_scale_and_offsets_are_rejected_at_deserialization() {
    let value = serde_json::to_value(LengthUnit::Meter).unwrap();
    let mut bad = value.clone();
    bad["scale"] = (-1.0).into();
    assert!(serde_json::from_value::<LengthUnit>(bad).is_err());
    let mut bad = value;
    bad["offset"] = 10.0.into();
    assert!(serde_json::from_value::<LengthUnit>(bad).is_err());
}
#[test]
fn source_evidence_retains_mixed_contexts_and_does_not_manufacture_a_default() {
    let evidence = LengthUnitEvidence::Declared {
        contexts: vec![
            LengthUnitDeclaration {
                context: "#1".into(),
                source_symbol: "mm".into(),
                unit: LengthUnit::Millimeter,
            },
            LengthUnitDeclaration {
                context: "#2".into(),
                source_symbol: "inch".into(),
                unit: LengthUnit::Inch,
            },
        ],
    };
    assert!(evidence.validate().is_ok());
    assert!(evidence.physical_scale_resolved());
    assert_eq!(evidence.uniform_unit(), None);
    let decoded: LengthUnitEvidence =
        serde_json::from_value(serde_json::to_value(&evidence).unwrap()).unwrap();
    assert_eq!(decoded, evidence);
    let missing = LengthUnitEvidence::default();
    assert_eq!(missing.uniform_unit(), None);
    assert!(matches!(
        missing.resolve_fallback(Some(LengthUnit::Meter), "authored units"),
        LengthUnitEvidence::ExplicitFallback { .. }
    ));
}
#[test]
fn source_measure_retains_original_value_symbol_and_attribution() {
    let unit =
        Unit::<kind::Length>::try_from_definition(parse_unit_expression("mm").unwrap()).unwrap();
    let source = SourceMeasure::try_new(
        40.0,
        "mm",
        unit,
        UnitAttribution::FileDeclaration {
            context: "#42".into(),
        },
    )
    .unwrap();
    assert_eq!(source.source_value(), 40.0);
    assert_eq!(source.source_unit_symbol(), "mm");
    assert_eq!(source.canonical().unwrap().canonical_value(), 0.04);
    let decoded: SourceMeasure<kind::Length> =
        serde_json::from_value(serde_json::to_value(&source).unwrap()).unwrap();
    assert_eq!(decoded, source);
}

#[test]
fn dynamic_dimension_never_guesses_energy_or_moment() {
    let value =
        DimensionedScalar::try_from_dimension(QuantityKind::Moment.dimension(), None, 2.0).unwrap();
    assert_eq!(value.semantic_kind(), None);
    assert!(value.resolve_kind::<kind::Moment>().is_ok());
    assert!(value.resolve_kind::<kind::Energy>().is_ok());
    let moment = DimensionedScalar::try_from_dimension(
        QuantityKind::Moment.dimension(),
        Some(QuantityKind::Moment),
        2.0,
    )
    .unwrap();
    assert_eq!(
        moment.resolve_kind::<kind::Energy>(),
        Err(UnitError::KindMismatch)
    );
}

#[test]
fn dynamic_external_descriptor_checks_known_kind_and_finite_storage() {
    let malformed = serde_json::json!({"dimension": QuantityKind::ElectricCurrent.dimension(), "semantic_kind":"electric_potential", "value_si":2.0});
    assert!(serde_json::from_value::<DimensionedScalar>(malformed).is_err());
    assert!(DimensionedScalar::try_from_dimension(
        QuantityKind::Power.dimension(),
        None,
        f64::INFINITY
    )
    .is_err());
    let value =
        DimensionedScalar::try_from_dimension(QuantityKind::Power.dimension(), None, 12.0).unwrap();
    assert_eq!(
        serde_json::from_value::<DimensionedScalar>(serde_json::to_value(value).unwrap()).unwrap(),
        value
    );
}

#[test]
fn bulk_points_and_borrowed_views_keep_the_quantity_descriptor() {
    let points = vec![[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]];
    let view = Quantity::<DynamicDimension, _>::try_from_dimension(
        QuantityKind::Position.dimension(),
        Some(QuantityKind::Position),
        points.as_slice(),
    )
    .unwrap();
    let typed = view.resolve_kind::<kind::Position>().unwrap();
    assert_eq!(typed.kind(), QuantityKind::Position);
    assert_eq!(typed.canonical_values().len(), 2);
    assert!(Quantity::<DynamicDimension, _>::try_from_dimension(
        QuantityKind::Position.dimension(),
        None,
        vec![[0.0, f64::NAN, 0.0]]
    )
    .is_err());
}

#[test]
fn provenance_rejects_unbounded_or_blank_context_labels() {
    let malformed = serde_json::json!({"status":"conflicting", "contexts":[{"context":"STEP #1","source_symbol":"m","unit":LengthUnit::Meter}], "reason":" "});
    assert!(serde_json::from_value::<LengthUnitEvidence>(malformed).is_err());
    let malformed = serde_json::json!({"status":"unsupported", "contexts":["x".repeat(257)], "reason":"unit unavailable"});
    assert!(serde_json::from_value::<LengthUnitEvidence>(malformed).is_err());
}

#[test]
fn nominal_angle_units_do_not_collapse_to_ratio_or_frequency() {
    let radians = parse_unit_expression("rad").unwrap();
    let ratio = parse_unit_expression("rad/rad").unwrap();
    assert_eq!(radians.dimension(), ratio.dimension());
    assert_eq!(radians.angle_power(), 1);
    assert_eq!(ratio.angle_power(), 0);
    assert!(Unit::<kind::Angle>::try_from_definition(radians).is_ok());
    assert_eq!(
        Unit::<kind::Dimensionless>::try_from_definition(radians),
        Err(UnitError::AngleMismatch)
    );
    let angular = parse_unit_expression("rad/s").unwrap();
    let cyclic = parse_unit_expression("Hz").unwrap();
    assert_eq!(angular.dimension(), cyclic.dimension());
    assert!(Unit::<kind::AngularVelocity>::try_from_definition(angular).is_ok());
    assert!(Unit::<kind::Frequency>::try_from_definition(cyclic).is_ok());
    assert_eq!(
        Unit::<kind::Frequency>::try_from_definition(angular),
        Err(UnitError::AngleMismatch)
    );
    assert_eq!(
        Unit::<kind::AngularVelocity>::try_from_definition(cyclic),
        Err(UnitError::AngleMismatch)
    );
}

#[test]
fn dynamic_descriptors_retain_nominal_angle_algebra_without_guessing_a_kind() {
    let angle = DynamicDimension::from_unit(parse_unit_expression("deg").unwrap());
    let time = DynamicDimension::from(PhysicalDimension::TIME);
    let rate = angle.divide(time).unwrap();
    assert_eq!(rate.dimension(), PhysicalDimension::FREQUENCY);
    assert_eq!(rate.angle_power(), 1);
    assert_eq!(rate.kind(), None);
    let value = DimensionedScalar::try_from_descriptor(rate, 2.0).unwrap();
    assert!(value.resolve_kind::<kind::AngularVelocity>().is_ok());
    assert_eq!(
        value.resolve_kind::<kind::Frequency>(),
        Err(UnitError::AngleMismatch)
    );
    let replay =
        serde_json::from_value::<DimensionedScalar>(serde_json::to_value(value).unwrap()).unwrap();
    assert_eq!(replay.descriptor(), rate);
}

#[test]
fn nominal_angle_zero_preserves_existing_length_wire_shape() {
    let wire = serde_json::to_value(LengthUnit::Millimeter).unwrap();
    assert!(wire.get("angle_power").is_none());
    let unit: LengthUnit = serde_json::from_value(wire).unwrap();
    assert_eq!(unit, LengthUnit::Millimeter);
    let bad = serde_json::json!({"dimension":PhysicalDimension::LENGTH,"scale":1.0,"offset":0.0,"angle_power":1});
    assert!(serde_json::from_value::<LengthUnit>(bad).is_err());
}

#[test]
fn heterogeneous_fields_keep_nominal_kind_and_validate_the_layout() {
    #[derive(Debug)]
    struct Layout;
    impl QuantityLayout for Layout {
        fn component_count(&self) -> usize {
            2
        }
        fn component_descriptor(&self, i: usize) -> Option<DynamicDimension> {
            let kind = match i {
                0 => QuantityKind::Displacement,
                1 => QuantityKind::Rotation,
                _ => return None,
            };
            DynamicDimension::quantity_type(kind)
                .with_semantic_kind(Some(kind))
                .ok()
        }
    }
    let values = [0.01, 0.2];
    let field = QuantityField::try_new(Layout, &values[..]).unwrap();
    assert_eq!(
        field.component(0).unwrap().semantic_kind(),
        Some(QuantityKind::Displacement)
    );
    assert_eq!(field.component(1).unwrap().angle_power(), 1);
    assert!(field
        .component(1)
        .unwrap()
        .resolve_kind::<kind::Dimensionless>()
        .is_err());
    assert!(matches!(
        QuantityField::try_new(Layout, &[0.01][..]),
        Err(QuantityFieldError::ShapeMismatch)
    ));
    assert!(matches!(
        QuantityField::try_new(Layout, &[0.0, f64::NAN][..]),
        Err(QuantityFieldError::InvalidValue(_))
    ));
}

#[test]
fn canonical_optional_values_preserve_null_and_reject_nonfinite_serialization() {
    #[derive(Debug, serde::Serialize, serde::Deserialize)]
    struct Owner {
        #[serde(with = "super::canonical_scalar::optional")]
        density: Option<MassDensity>,
    }
    assert_eq!(
        serde_json::to_value(Owner { density: None }).unwrap(),
        serde_json::json!({"density":null})
    );
    let owner: Owner = serde_json::from_value(serde_json::json!({"density":997.0})).unwrap();
    assert_eq!(owner.density.unwrap().canonical_value(), 997.0);
    assert!(serde_json::to_value(Owner {
        density: Some(MassDensity::new(f64::NAN))
    })
    .is_err());
    assert!(serde_json::to_value(Length::new(f64::INFINITY)).is_err());
}

#[test]
fn cycles_and_angular_frequency_convert_explicitly_and_retain_nominal_radians() {
    let cycles = Frequency::new(2.0);
    let angular = cycles.angular_frequency();
    assert_eq!(angular.angle_power(), 1);
    assert_eq!(angular.frequency(), cycles);
    let phase = angular * Time::new(0.125);
    assert!((phase.canonical_value() - std::f64::consts::FRAC_PI_2).abs() < 1.0e-15);
    assert!(Unit::<kind::AngularFrequency>::try_from_definition(
        parse_unit_expression("Hz").unwrap()
    )
    .is_err());
    assert!(Unit::<kind::AngularFrequency>::try_from_definition(
        parse_unit_expression("rad/s").unwrap()
    )
    .is_ok());
    assert_eq!(
        VACUUM_ELECTRIC_PERMITTIVITY.semantic_kind(),
        Some(QuantityKind::ElectricPermittivity)
    );
    assert_eq!(
        VACUUM_MAGNETIC_PERMEABILITY.semantic_kind(),
        Some(QuantityKind::MagneticPermeability)
    );
}

#[test]
fn phasor_time_derivative_cancels_nominal_angle_without_relabelling_hertz() {
    let angular = Frequency::new(2.0).angular_frequency();
    let rate = angular.phasor_derivative_rate();
    assert_eq!(angular.angle_power(), 1);
    assert_eq!(rate.angle_power(), 0);
    assert_eq!(rate.kind(), QuantityKind::TimeDerivativeRate);
    assert_eq!(rate.canonical_value(), 2.0 * std::f64::consts::TAU);
    let erased = rate.try_into_dynamic().unwrap();
    assert!(erased.resolve_kind::<kind::Frequency>().is_err());
    assert!(erased.resolve_kind::<kind::AngularFrequency>().is_err());
}

#[test]
fn typed_history_append_checks_only_the_new_value_and_retains_kind() {
    let mut history = Quantity::<kind::Energy, Vec<f64>>::try_new(vec![1.0]).unwrap();
    history.try_push(Energy::new(2.0)).unwrap();
    assert!(history.try_push(Energy::new(f64::NAN)).is_err());
    assert_eq!(history.canonical_values(), &[1.0, 2.0]);
    assert!(history.try_resize(3, Energy::new(f64::NAN)).is_err());
    assert_eq!(history.canonical_values(), &[1.0, 2.0]);
    let erased = history.try_into_dynamic().unwrap();
    assert_eq!(erased.semantic_kind(), Some(QuantityKind::Energy));
    assert_eq!(erased.canonical_values(), &[1.0, 2.0]);
}

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
enum TestParameterAxis {
    Physical(UnitDefinition),
    Native(String),
}
impl ParameterAxis for TestParameterAxis {
    fn validate_parameter_axis(&self) -> Result<(), String> {
        match self {
            Self::Native(name) if name.is_empty() => Err("missing evaluator identity".into()),
            _ => Ok(()),
        }
    }
    fn parameter_unit(&self) -> Option<UnitDefinition> {
        match self {
            Self::Physical(unit) => Some(*unit),
            Self::Native(_) => None,
        }
    }
}

#[test]
fn derivative_denominator_scale_is_normalized_per_order() {
    let axis = TestParameterAxis::Physical(parse_unit_expression("mm").unwrap());
    let tangent = ParametricDerivative::try_new(
        Quantity::<kind::Length, [f64; 3]>::new([0.001, 0.0, 0.0]),
        [DerivativeAxis::new(axis.clone(), 1)],
    )
    .unwrap();
    let resolved = tangent.into_canonical_quantity().unwrap();
    assert_eq!(
        resolved.dimension(),
        crate::PhysicalDimension::DIMENSIONLESS
    );
    assert_eq!(resolved.canonical_value(), [1.0, 0.0, 0.0]);

    // d²(3*x²/m)/d(x_mm)² = 6e-6 m/mm²; canonical result is 6/m.
    let second = ParametricDerivative::try_new(Length::new(6e-6), [DerivativeAxis::new(axis, 2)])
        .unwrap()
        .into_canonical_quantity()
        .unwrap();
    assert_eq!(
        second.dimension(),
        crate::PhysicalDimension::LENGTH.power(-1).unwrap()
    );
    assert!((second.canonical_value() - 6.0).abs() < 1e-14);
}

#[test]
fn derivative_native_axis_stays_unresolved_and_angle_denominator_stays_nominal() {
    let native = ParametricDerivative::try_new(
        Length::new(3.0),
        [DerivativeAxis::new(
            TestParameterAxis::Native("spline:revision42:u".into()),
            1,
        )],
    )
    .unwrap();
    assert_eq!(native.resolved_descriptor().unwrap(), None);
    native.with_canonical_storage(|value, axes| {
        assert_eq!(*value, 3.0);
        assert_eq!(axes[0].order, 1);
    });
    assert_eq!(
        native.into_canonical_quantity(),
        Err(DerivativeError::UnresolvedAxis)
    );
    let angular = ParametricDerivative::try_new(
        Length::new(2.0),
        [DerivativeAxis::new(
            TestParameterAxis::Physical(parse_unit_expression("rad").unwrap()),
            1,
        )],
    )
    .unwrap()
    .into_canonical_quantity()
    .unwrap();
    assert_eq!(angular.dimension(), crate::PhysicalDimension::LENGTH);
    assert_eq!(angular.angle_power(), -1);
    assert!(angular.resolve_kind::<kind::Length>().is_err());
}

#[test]
fn derivative_checked_serde_rejects_invalid_frames_orders_and_storage() {
    type D = ParametricDerivative<kind::Length, [f64; 3], TestParameterAxis>;
    let value = D::try_new(
        Quantity::new([1.0, 2.0, 3.0]),
        [DerivativeAxis::new(
            TestParameterAxis::Native("portable:curve7".into()),
            1,
        )],
    )
    .unwrap();
    let wire = serde_json::to_value(&value).unwrap();
    assert_eq!(serde_json::from_value::<D>(wire.clone()).unwrap(), value);
    let mut invalid = wire;
    invalid["axes"][0]["order"] = serde_json::json!(0);
    assert!(serde_json::from_value::<D>(invalid).is_err());
    assert!(D::try_new(
        Quantity::new([f64::NAN, 0.0, 0.0]),
        [DerivativeAxis::new(
            TestParameterAxis::Native("curve".into()),
            1
        )]
    )
    .is_err());
    assert!(D::try_new(
        Quantity::new([1.0, 0.0, 0.0]),
        [DerivativeAxis::new(
            TestParameterAxis::Native(String::new()),
            1
        )]
    )
    .is_err());
}

#[test]
fn mixed_derivative_retains_each_denominator_and_ignores_affine_offsets() {
    let mixed = ParametricDerivative::try_new(
        Length::new(0.001),
        [
            DerivativeAxis::new(
                TestParameterAxis::Physical(parse_unit_expression("mm").unwrap()),
                1,
            ),
            DerivativeAxis::new(
                TestParameterAxis::Physical(parse_unit_expression("rad").unwrap()),
                1,
            ),
        ],
    )
    .unwrap()
    .into_canonical_quantity()
    .unwrap();
    assert_eq!(mixed.dimension(), crate::PhysicalDimension::DIMENSIONLESS);
    assert_eq!(mixed.angle_power(), -1);
    assert_eq!(mixed.canonical_value(), 1.0);
    let temperature_axis =
        UnitDefinition::try_new(crate::PhysicalDimension::TEMPERATURE, 1.0, 273.15).unwrap();
    let derivative = ParametricDerivative::try_new(
        Length::new(2.0),
        [DerivativeAxis::new(
            TestParameterAxis::Physical(temperature_axis),
            1,
        )],
    )
    .unwrap()
    .into_canonical_quantity()
    .unwrap();
    assert_eq!(derivative.canonical_value(), 2.0);
}

#[test]
fn physical_vector_replacement_is_checked_without_a_mutable_raw_escape() {
    let pressure = DynamicDimension::try_new(
        QuantityKind::Pressure.dimension(),
        Some(QuantityKind::Pressure),
    )
    .unwrap();
    let temperature = DynamicDimension::try_new(
        QuantityKind::Temperature.dimension(),
        Some(QuantityKind::Temperature),
    )
    .unwrap();
    let mut values = Quantity::try_from_descriptor(pressure, vec![1.0, 2.0]).unwrap();
    values
        .try_replace(1, Quantity::try_from_descriptor(pressure, 3.0).unwrap())
        .unwrap();
    assert_eq!(values.canonical_values(), &[1.0, 3.0]);
    assert_eq!(
        values.try_replace(
            0,
            Quantity::try_from_descriptor(temperature, 300.0).unwrap()
        ),
        Err(QuantityVectorEditError::DescriptorMismatch)
    );
    assert_eq!(
        values.try_replace(2, Quantity::try_from_descriptor(pressure, 4.0).unwrap()),
        Err(QuantityVectorEditError::IndexOutOfBounds)
    );
    let mut static_values = Quantity::<kind::Pressure, _>::new(vec![1.0, 2.0]);
    assert_eq!(
        static_values.try_replace(0, Pressure::new(f64::NAN)),
        Err(QuantityVectorEditError::NonFinite)
    );
    assert_eq!(static_values.canonical_values(), &[1.0, 2.0]);
}

#[test]
fn formal_coordinates_preserve_native_identity_and_convert_only_declared_axes() {
    let native = TestParameterAxis::Native("curve-parameter".into());
    let value = ParameterCoordinate::try_new(2.5, native.clone()).unwrap();
    assert!(value.resolved_quantity().unwrap().is_none());
    assert_eq!(
        value.with_native_coordinate(|x, axis| (x, axis.clone())),
        (2.5, native.clone())
    );
    let millimeters = TestParameterAxis::Physical(
        UnitDefinition::try_for_kind(QuantityKind::Length, 0.001, 0.0).unwrap(),
    );
    let physical = ParameterCoordinate::try_new(25.0, millimeters.clone()).unwrap();
    assert_eq!(
        physical
            .resolved_quantity()
            .unwrap()
            .unwrap()
            .canonical_value(),
        0.025
    );
    let encoded = serde_json::to_value(&value).unwrap();
    let restored: ParameterCoordinate<TestParameterAxis> = serde_json::from_value(encoded).unwrap();
    assert_eq!(restored, value);
    assert!(ParameterCoordinate::try_new(f64::NAN, native).is_err());
}

#[test]
fn mixed_parameter_derivative_keeps_uv_numerator_and_denominator_frames() {
    let millimeters = TestParameterAxis::Physical(
        UnitDefinition::try_for_kind(QuantityKind::Length, 0.001, 0.0).unwrap(),
    );
    let degrees = TestParameterAxis::Physical(
        UnitDefinition::try_for_kind(QuantityKind::Angle, std::f64::consts::PI / 180.0, 0.0)
            .unwrap(),
    );
    let derivative = ParameterCoordinateDerivative::try_new(
        2.0,
        millimeters.clone(),
        [DerivativeAxis::new(degrees, 1)],
    )
    .unwrap();
    let resolved = derivative.resolved_quantity().unwrap().unwrap();
    assert!((resolved.canonical_value() - 0.002 * 180.0 / std::f64::consts::PI).abs() < 1e-15);
    assert_eq!(resolved.dimension(), PhysicalDimension::LENGTH);
    assert_eq!(resolved.angle_power(), -1);
    let native = TestParameterAxis::Native("surface-u".into());
    let unresolved =
        ParameterCoordinateDerivative::try_new(3.0, native, [DerivativeAxis::new(millimeters, 2)])
            .unwrap();
    assert!(unresolved.resolved_quantity().unwrap().is_none());
    let wire = serde_json::to_value(&unresolved).unwrap();
    let restored: ParameterCoordinateDerivative<TestParameterAxis> =
        serde_json::from_value(wire).unwrap();
    assert_eq!(restored, unresolved);
    assert!(ParameterCoordinateDerivative::try_new(
        1.0,
        TestParameterAxis::Native("u".into()),
        [DerivativeAxis::new(
            TestParameterAxis::Native("t".into()),
            0
        )]
    )
    .is_err());
}

#[test]
fn borrowed_quantity_slices_preserve_descriptor_and_storage() {
    let quantity = Quantity::<kind::Temperature, _>::new(vec![293.15, 310.0]);
    let borrowed = quantity.as_quantity_slice();
    assert_eq!(borrowed.descriptor(), quantity.descriptor());
    assert_eq!(
        borrowed.canonical_values().as_ptr(),
        quantity.canonical_values().as_ptr()
    );
    let dynamic = quantity.try_into_dynamic().unwrap();
    let borrowed = dynamic.as_quantity_slice();
    assert_eq!(borrowed.descriptor(), dynamic.descriptor());
    assert_eq!(
        borrowed.canonical_values().as_ptr(),
        dynamic.canonical_values().as_ptr()
    );
}

#[test]
fn checked_storage_map_preserves_spec_and_rejects_nonfinite_output() {
    let quantity = Quantity::<kind::Force, _>::new(vec![1.0, 2.0])
        .try_into_dynamic()
        .unwrap();
    let descriptor = quantity.descriptor();
    let mapped = quantity
        .clone()
        .try_map_canonical_storage(|mut values| {
            values[1] += 3.0;
            values
        })
        .unwrap();
    assert_eq!(mapped.descriptor(), descriptor);
    assert_eq!(mapped.canonical_values(), &vec![1.0, 5.0]);
    assert!(quantity
        .try_map_canonical_storage(|mut values| {
            values[0] = f64::INFINITY;
            values
        })
        .is_err());
}

#[test]
fn vorticity_is_its_own_per_second_kind_distinct_from_rates_and_angular_velocity() {
    let omega = Vorticity::new(3.5);
    assert_eq!(omega.kind(), QuantityKind::Vorticity);
    assert_ne!(QuantityKind::Vorticity, QuantityKind::TimeDerivativeRate);
    assert_ne!(QuantityKind::Vorticity, QuantityKind::AngularVelocity);
    assert_eq!(omega.angle_power(), 0);
    assert_eq!(omega.canonical_value(), 3.5);
    let per_second = parse_unit_expression("s^-1").unwrap();
    assert!(Unit::<kind::Vorticity>::try_from_definition(per_second).is_ok());
    assert_eq!(
        Unit::<kind::Vorticity>::try_from_definition(parse_unit_expression("rad/s").unwrap()),
        Err(UnitError::AngleMismatch)
    );
    let field = Quantity::<kind::Vorticity, Vec<f64>>::try_new(vec![0.0, -1.25, 4.0]).unwrap();
    assert_eq!(field.canonical_values(), &[0.0, -1.25, 4.0]);
    assert!(Quantity::<kind::Vorticity, Vec<f64>>::try_new(vec![f64::NAN]).is_err());
}

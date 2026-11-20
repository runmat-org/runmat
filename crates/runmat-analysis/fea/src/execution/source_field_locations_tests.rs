//! The source-language field location table (runmat-analysis-core `SOURCE_QUANTITIES`) against
//! what the solvers publish. Every (physics, quantity, location) row must be published by a
//! producer, and every field a producer publishes for a quantity with a `.fea` name must have a
//! row, so the checker admits exactly the fields a run can return.
//!
//! The producers are read two ways:
//! - the field inventories, called directly (structural recovery with connectors, thermal
//!   history, electrical for both equations, Maxwell, magnetic, fluid for every coupling, and the
//!   acoustic native plan);
//! - the layouts a pipeline builds inline, listed in `INLINE` with the file that builds them. The
//!   test reads each file and checks that it names the quantity and the location.
use std::collections::HashSet;

use runmat_analysis_core::{
    published_locations, CoupledPhysicsFamily as C, FieldLocation as L, ObservationQuantity as Q,
    PhysicsFamily as P, StaticElectricalEquation, SOURCE_QUANTITIES,
};

use super::{
    acoustic::native_acoustic_field_plan, coupled_fluid_field_definition_inventory,
    electrical_native_field_definition_inventory, magnetic_native_field_definition_inventory,
    maxwell_native_field_definition_inventory, FluidCoupling,
};

type Row = (P, Q, L);

/// Layouts built inline by a pipeline rather than declared in an inventory, with the file
/// (under runmat-analysis-fea/src) that builds each one.
const INLINE: &[(P, Q, L, &str)] = &[
    (
        P::Structural,
        Q::Velocity,
        L::Node,
        "pipeline/transient/finalization.rs",
    ),
    (
        P::Structural,
        Q::Acceleration,
        L::Node,
        "pipeline/transient/finalization.rs",
    ),
    (
        P::Structural,
        Q::ReactionForce,
        L::Node,
        "pipeline/transient/finalization.rs",
    ),
    (
        P::Structural,
        Q::ReactionMoment,
        L::Node,
        "pipeline/transient/recovery.rs",
    ),
    (
        P::Structural,
        Q::StrainEnergy,
        L::Element,
        "pipeline/transient/finalization.rs",
    ),
    (
        P::Structural,
        Q::StrainEnergy,
        L::Global,
        "pipeline/transient/finalization.rs",
    ),
    (
        P::Structural,
        Q::StrainEnergy,
        L::Connector,
        "pipeline/transient/finalization.rs",
    ),
    (
        P::Structural,
        Q::ModalFrequency,
        L::Mode,
        "pipeline/modal.rs",
    ),
    (
        P::Structural,
        Q::ModalParticipation,
        L::Mode,
        "pipeline/modal.rs",
    ),
    (P::Structural, Q::Eigenvalue, L::Mode, "pipeline/modal.rs"),
    (P::Structural, Q::Residual, L::Global, "pipeline/modal.rs"),
    (P::Structural, Q::Residual, L::Mode, "pipeline/modal.rs"),
    (
        P::Structural,
        Q::ContactGap,
        L::Interface,
        "pipeline/nonlinear/recovery.rs",
    ),
    (
        P::Structural,
        Q::ContactPressure,
        L::Interface,
        "pipeline/nonlinear/recovery.rs",
    ),
    (
        P::Thermal,
        Q::HeatFlux,
        L::Interface,
        "physics/cht/recovery.rs",
    ),
    (
        P::Thermal,
        Q::Residual,
        L::Global,
        "pipeline/thermo_mechanical.rs",
    ),
    (
        P::Coupled(C::Thermomechanical),
        Q::Temperature,
        L::Element,
        "pipeline/thermo_mechanical.rs",
    ),
    (
        P::Coupled(C::Thermomechanical),
        Q::Temperature,
        L::IntegrationPoint,
        "pipeline/thermo_mechanical/solved.rs",
    ),
    (
        P::Acoustic,
        Q::AcousticPressure,
        L::Mode,
        "execution/acoustic_response/fields.rs",
    ),
    (
        P::Acoustic,
        Q::AcousticParticleVelocity,
        L::Node,
        "physics/acoustic/reduced.rs",
    ),
    (
        P::Fluid,
        Q::FluidPressure,
        L::Node,
        "physics/fsi/recovery.rs",
    ),
    (
        P::Fluid,
        Q::FluidVelocity,
        L::Node,
        "physics/fsi/recovery.rs",
    ),
    (
        P::Fluid,
        Q::FluidPressure,
        L::Interface,
        "physics/fsi/recovery.rs",
    ),
    (P::Fluid, Q::Residual, L::Global, "physics/fsi/recovery.rs"),
];

/// The single physics the table gives `quantity`, for inventories that mix physics (a conjugate
/// fluid inventory publishes the solid's temperature). Rows keyed to a coupled family name
/// fields only its coupled pipeline builds, so they don't count here.
fn table_physics(quantity: Q, fallback: P) -> P {
    let mut physics = SOURCE_QUANTITIES
        .iter()
        .filter(|row| row.quantity == quantity)
        .flat_map(|row| row.fields.iter().map(|(physics, _)| *physics))
        .filter(|physics| !matches!(physics, P::Coupled(_)));
    match (physics.next(), physics.next()) {
        (Some(only), None) => only,
        _ => fallback,
    }
}

fn inventories() -> HashSet<Row> {
    let mut rows = HashSet::new();
    let mut add = |fallback: P, quantity: Q, location: L| {
        rows.insert((table_physics(quantity, fallback), quantity, location));
    };
    for definition in crate::post::fields::structural_static_observable_field_definitions().unwrap()
    {
        add(P::Structural, definition.quantity(), definition.location());
    }
    for definition in crate::physics::thermal::history_fields::definitions().unwrap() {
        add(P::Thermal, definition.quantity(), definition.location());
    }
    for equation in [
        StaticElectricalEquation::Electrostatics,
        StaticElectricalEquation::SteadyConduction,
    ] {
        for definition in electrical_native_field_definition_inventory(equation).unwrap() {
            add(
                P::Electromagnetic,
                definition.quantity(),
                definition.location(),
            );
        }
    }
    for definition in maxwell_native_field_definition_inventory()
        .unwrap()
        .into_iter()
        .chain(magnetic_native_field_definition_inventory().unwrap())
    {
        add(
            P::Electromagnetic,
            definition.quantity(),
            definition.location(),
        );
    }
    for coupling in [
        FluidCoupling::Standalone,
        FluidCoupling::ConjugateHeat,
        FluidCoupling::FluidStructure,
    ] {
        for definition in coupled_fluid_field_definition_inventory(coupling).unwrap() {
            add(P::Fluid, definition.quantity(), definition.location());
        }
    }
    for requirement in native_acoustic_field_plan(1, 1).unwrap().requirements {
        add(
            P::Acoustic,
            requirement.quantity,
            requirement.output_location,
        );
    }
    rows
}

fn inline() -> HashSet<Row> {
    let root = concat!(env!("CARGO_MANIFEST_DIR"), "/src/");
    INLINE
        .iter()
        .map(|&(physics, quantity, location, file)| {
            let text = std::fs::read_to_string(format!("{root}{file}"))
                .unwrap_or_else(|error| panic!("{file}: {error}"));
            for name in [
                format!("ObservationQuantity::{quantity:?}"),
                format!("FieldLocation::{location:?}"),
            ] {
                assert!(text.contains(&name), "{file} does not build {name}");
            }
            (physics, quantity, location)
        })
        .collect()
}

fn table() -> HashSet<Row> {
    SOURCE_QUANTITIES
        .iter()
        .flat_map(|row| {
            row.fields.iter().flat_map(move |(physics, locations)| {
                locations
                    .iter()
                    .map(move |location| (*physics, row.quantity, *location))
            })
        })
        .collect()
}

#[test]
fn every_location_row_is_published_by_a_solver() {
    let published: HashSet<Row> = inventories().union(&inline()).copied().collect();
    let mut unpublished: Vec<_> = table()
        .difference(&published)
        .map(|row| format!("{row:?}"))
        .collect();
    unpublished.sort();
    assert!(
        unpublished.is_empty(),
        "rows no solver publishes: {unpublished:?}"
    );
}

#[test]
fn every_published_field_with_a_source_name_has_a_row() {
    let named: HashSet<Q> = SOURCE_QUANTITIES.iter().map(|row| row.quantity).collect();
    let table = table();
    let mut missing: Vec<_> = inventories()
        .union(&inline())
        .filter(|(_, quantity, _)| named.contains(quantity))
        .filter(|row| !table.contains(row))
        .map(|row| format!("{row:?}"))
        .collect();
    missing.sort();
    assert!(
        missing.is_empty(),
        "published fields with no row: {missing:?}"
    );
}

/// Per family: a thermal step (steady or transient, both published through the thermal history
/// inventory) publishes temperature at nodes only, and the thermo-mechanical step adds the
/// element and integration-point temperatures its prestress snapshots carry.
#[test]
fn temperature_locations_follow_the_family() {
    let thermal: HashSet<L> = crate::physics::thermal::history_fields::definitions()
        .unwrap()
        .into_iter()
        .filter(|definition| definition.quantity() == Q::Temperature)
        .map(|definition| definition.location())
        .collect();
    assert_eq!(thermal, HashSet::from([L::Node]));
    assert_eq!(published_locations(P::Thermal, Q::Temperature), [L::Node]);

    let coupled = published_locations(P::Coupled(C::Thermomechanical), Q::Temperature);
    assert_eq!(coupled, [L::Node, L::Element, L::IntegrationPoint]);
    let inline = inline();
    for location in [L::Element, L::IntegrationPoint] {
        assert!(inline.contains(&(P::Coupled(C::Thermomechanical), Q::Temperature, location)));
        assert!(!published_locations(P::Thermal, Q::Temperature).contains(&location));
    }
}

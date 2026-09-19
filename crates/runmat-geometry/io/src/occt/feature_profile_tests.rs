use runmat_geometry_core::{
    ExactMassPropertiesImplementation, GeometryFeatureRef, GeometryFeatureRegionSelection,
    GeometryRegionCardinality, GeometryRegionRole, GeometryRegionSelector,
};
use runmat_geometry_ops::{
    GeometryFeatureProvider, GeometryFeatureProviderSelection, ResolvedGeometryFeatureOperation,
};

use super::feature::OcctExactFeatureProvider;
use crate::{ExactCadImportOptions, GeometryImportContext, ImportedExactCad};

const SHORT_EDGE_FACE: &[u8] = include_bytes!("../../tests/fixtures/short_edge_face.brep");

#[test]
fn exact_profile_sweeps_are_qualified_with_complete_lineage() {
    let provider = provider();
    let profile = profile();
    assert_eq!(profile.topology.faces.len(), 1);
    let properties = profile
        .evaluators
        .mass_properties
        .iter()
        .find_map(|record| match &record.implementation {
            ExactMassPropertiesImplementation::KernelValidated { properties, .. } => {
                Some(properties)
            }
            ExactMassPropertiesImplementation::Kernel { .. } => None,
        })
        .unwrap();
    let extrusion = execute(
        &provider,
        "extrusion",
        ResolvedGeometryFeatureOperation::Extrude {
            profile: GeometryFeatureRef("profile".into()),
            distance_m: 0.5,
            direction: [0.0, 0.0, 1.0],
        },
        &[("profile", &profile)],
    );
    assert_eq!(extrusion.topology.solids.len(), 1);
    assert_close(volume(&extrusion), properties.surface_area_m2 * 0.5, 1.0e-9);
    extrusion.build_closure().unwrap();

    let revolution = execute(
        &provider,
        "revolution",
        ResolvedGeometryFeatureOperation::Revolve {
            profile: GeometryFeatureRef("profile".into()),
            axis: [0.0, 1.0, 0.0],
            origin_m: [
                properties.centroid_m[0] - 2.0,
                properties.centroid_m[1],
                properties.centroid_m[2],
            ],
            angle_rad: 2.0 * std::f64::consts::PI,
        },
        &[("profile", &profile)],
    );
    assert_eq!(revolution.topology.solids.len(), 1);
    let pappus_volume = properties.surface_area_m2 * 2.0 * std::f64::consts::PI * 2.0;
    assert_close(volume(&revolution), pappus_volume, 5.0e-7);
    revolution.build_closure().unwrap();
}

#[test]
fn exact_removal_and_edge_treatment_features_are_qualified() {
    let provider = provider();
    let stock = execute(
        &provider,
        "stock",
        ResolvedGeometryFeatureOperation::Box {
            width_m: 1.0,
            depth_m: 1.0,
            height_m: 1.0,
        },
        &[],
    );
    let hole = execute(
        &provider,
        "hole",
        ResolvedGeometryFeatureOperation::Hole {
            target: GeometryFeatureRef("stock".into()),
            radius_m: 0.1,
            depth_m: 1.0,
            axis: [0.0, 0.0, 1.0],
            origin_m: [0.5, 0.5, 0.0],
        },
        &[("stock", &stock)],
    );
    assert_close(volume(&hole), 1.0 - std::f64::consts::PI * 0.01, 1.0e-9);
    assert_eq!(hole.topology.solids.len(), 1);
    hole.build_closure().unwrap();
    let outside_hole = provider.execute(
        "outside_hole",
        &ResolvedGeometryFeatureOperation::Hole {
            target: GeometryFeatureRef("stock".into()),
            radius_m: 0.1,
            depth_m: 1.0,
            axis: [0.0, 0.0, 1.0],
            origin_m: [2.0, 2.0, 0.0],
        },
        &GeometryFeatureProviderSelection::None,
        &[("stock", &stock)],
        None,
    );
    assert!(outside_hole.is_err());

    let profile = profile();
    let profile_area = profile
        .evaluators
        .mass_properties
        .iter()
        .find_map(|record| match &record.implementation {
            ExactMassPropertiesImplementation::KernelValidated { properties, .. } => {
                Some(properties.surface_area_m2)
            }
            ExactMassPropertiesImplementation::Kernel { .. } => None,
        })
        .unwrap();
    let offset_profile = execute(
        &provider,
        "offset_profile",
        ResolvedGeometryFeatureOperation::Transform {
            source: GeometryFeatureRef("profile".into()),
            matrix: [
                0.5, 0.0, 0.0, 0.25, 0.0, 0.5, 0.0, 0.25, 0.0, 0.0, 1.0, -0.1, 0.0, 0.0, 0.0, 1.0,
            ],
        },
        &[("profile", &profile)],
    );
    let pocket = execute(
        &provider,
        "pocket",
        ResolvedGeometryFeatureOperation::Pocket {
            target: GeometryFeatureRef("stock".into()),
            profile: GeometryFeatureRef("profile".into()),
            depth_m: 0.35,
            direction: [0.0, 0.0, 1.0],
        },
        &[("stock", &stock), ("profile", &offset_profile)],
    );
    assert_close(
        volume(&pocket),
        1.0 - profile_area * 0.5_f64.powi(2) * 0.25,
        5.0e-7,
    );
    assert_eq!(pocket.topology.solids.len(), 1);
    pocket.build_closure().unwrap();

    for (feature_id, operation) in [
        (
            "fillet",
            ResolvedGeometryFeatureOperation::Fillet {
                target: GeometryFeatureRef("stock".into()),
                edges: edge_selection(),
                radius_m: 0.1,
            },
        ),
        (
            "chamfer",
            ResolvedGeometryFeatureOperation::Chamfer {
                target: GeometryFeatureRef("stock".into()),
                edges: edge_selection(),
                distance_m: 0.1,
            },
        ),
    ] {
        let edge = stock.topology.edges.first().unwrap().id.clone();
        let treated = provider
            .execute(
                feature_id,
                &operation,
                &GeometryFeatureProviderSelection::Edges(vec![edge]),
                &[("stock", &stock)],
                None,
            )
            .unwrap()
            .geometry;
        assert!(volume(&treated) > 0.0 && volume(&treated) < 1.0);
        assert_eq!(treated.topology.solids.len(), 1);
        treated.build_closure().unwrap();
    }
}

fn profile() -> ImportedExactCad {
    crate::import_exact_cad(
        "short_edge_face.brep",
        SHORT_EDGE_FACE,
        crate::GeometryFormat::Brep,
        &ExactCadImportOptions::default(),
        &GeometryImportContext::new(),
    )
    .unwrap()
}

fn provider() -> OcctExactFeatureProvider {
    OcctExactFeatureProvider::new(
        ExactCadImportOptions::default(),
        GeometryImportContext::new(),
    )
}

fn execute(
    provider: &OcctExactFeatureProvider,
    id: &str,
    operation: ResolvedGeometryFeatureOperation,
    dependencies: &[(&str, &ImportedExactCad)],
) -> ImportedExactCad {
    provider
        .execute(
            id,
            &operation,
            &GeometryFeatureProviderSelection::None,
            dependencies,
            None,
        )
        .unwrap()
        .geometry
}

fn edge_selection() -> GeometryFeatureRegionSelection {
    GeometryFeatureRegionSelection {
        role: GeometryRegionRole::Edge,
        selector: GeometryRegionSelector::Result,
        cardinality: GeometryRegionCardinality::OneOrMore,
    }
}

fn volume(geometry: &ImportedExactCad) -> f64 {
    geometry
        .evaluators
        .mass_properties
        .iter()
        .map(|record| match &record.implementation {
            ExactMassPropertiesImplementation::KernelValidated { properties, .. } => {
                properties.volume_m3
            }
            ExactMassPropertiesImplementation::Kernel { .. } => 0.0,
        })
        .sum()
}

fn assert_close(actual: f64, expected: f64, relative_tolerance: f64) {
    let tolerance = relative_tolerance * expected.abs().max(1.0);
    assert!(
        (actual - expected).abs() <= tolerance,
        "actual {actual:.16e} differs from expected {expected:.16e} by more than {tolerance:.16e}"
    );
}

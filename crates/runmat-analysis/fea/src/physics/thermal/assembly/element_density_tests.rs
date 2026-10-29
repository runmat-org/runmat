use super::{tests::assemble, *};
use crate::{
    excitation::{realize_step_excitations, ComplexComponent},
    fixtures::{fixture_model, FixtureId},
};
use runmat_analysis_core::{
    CoordinateFrame, ExcitationAction, ExcitationDistribution, ExcitationKinematics,
    ExcitationTarget, FieldLocation, HeatGenerationDensity, PhysicalSignal, PropertyRegionTarget,
    Signal, SignalApproximationEvidence, SignalExtrapolation, SignalInterpolation, SignalKnot,
    SignalOrdinate, SignalPrecision, SignalProvenance, SignalRepresentation, SignalShape,
    SignalValue, TabulatedSignal, TabulatedSignalDomain,
};
use runmat_execution::Digest;
use runmat_meshing_core::{ElementOrder, SolverMeshArtifact};

fn realize_excitations(
    model: &AnalysisModel,
    _: Option<&str>,
    sample: ExcitationSamplePoint<'_>,
) -> Result<RealizedExcitations, crate::excitation::RealizationError> {
    realize_step_excitations(model, &model.sequence.steps[0], sample)
}

fn fixture(order: ElementOrder) -> (AnalysisModel, SolverMeshArtifact) {
    let mesh = match order {
        ElementOrder::Tet4 => crate::topology::test_meshes::slab_mesh(2),
        ElementOrder::Tet10 => crate::topology::test_meshes::quadratic_slab_mesh(2),
    };
    let mut model = fixture_model(FixtureId::CantileverLinearStatic);
    model.constraints.clear();
    model.boundaries.clear();
    model.material_assignments[0].target =
        PropertyRegionTarget::solid(runmat_meshing_core::fixtures::fixture_solid("slab"));
    let mut action = model.excitations[0].excitation().clone();
    action.target = ExcitationTarget::WholeModel;
    action.direction = None;
    action.frame = CoordinateFrame::Global;
    action.kinematics = ExcitationKinematics::FixedFrame;
    action.distribution = ExcitationDistribution::TopologyField;
    action.projection.preserve_resultant = true;
    action.projection.preserve_moment = false;
    action.projection.relative_tolerance = 1e-12;
    let values = mesh
        .topology
        .volume_elements
        .iter()
        .enumerate()
        .map(|(row, _)| HeatGenerationDensity::new((row + 1) as f64))
        .collect();
    model.signals = vec![PhysicalSignal::HeatGenerationDensity(Signal {
        signal_id: action.signal_id.clone(),
        shape: SignalShape::TopologyField {
            topology_id: Digest::from_bytes(*mesh.canonical_digest.bytes()).to_string(),
            location: FieldLocation::Element,
            component_count: 1,
        },
        frame: Some(CoordinateFrame::Global),
        representation: SignalRepresentation::Constant(SignalValue::Tensor {
            shape: vec![mesh.topology.volume_elements.len(), 1],
            values,
        }),
        precision: SignalPrecision::Real64,
        evidence: SignalApproximationEvidence {
            max_absolute_error: Some(0.0),
            max_relative_error: Some(0.0),
            derivative_max_absolute_error: None,
            sample_count: 1,
        },
        provenance: SignalProvenance::Authored,
    })];
    model.excitations = vec![ExcitationAction::HeatGeneration(action)];
    (model, mesh)
}

fn signal(model: &mut AnalysisModel) -> &mut Signal<HeatGenerationDensity> {
    let PhysicalSignal::HeatGenerationDensity(signal) = &mut model.signals[0] else {
        unreachable!()
    };
    signal
}

fn run(
    model: &AnalysisModel,
    mesh: &SolverMeshArtifact,
    sources: &RealizedExcitations,
) -> Result<ThermalOperatorAssembly, ThermalAssemblyError> {
    assemble(
        model,
        mesh,
        crate::planning::resource::ThermalOperatorResourceFacts::derive(mesh, &[])
            .unwrap()
            .support(),
        sources,
        ExcitationSamplePoint::Static,
    )
}

#[test]
fn element_density_both_orders_conserves_power_and_matches_scalar_shape_integration() {
    for order in [ElementOrder::Tet4, ElementOrder::Tet10] {
        let (mut model, mesh) = fixture(order);
        let sources = realize_excitations(&model, None, ExcitationSamplePoint::Static).unwrap();
        let actual = run(&model, &mesh, &sources).unwrap();
        // Independent unit-cube partition: 48 straight tetrahedra, each volume 1/48 m³.
        let count = mesh.topology.volume_elements.len();
        let expected = (count * (count + 1) / 2) as f64 / count as f64;
        assert!(
            (actual.applied_source.canonical_values().iter().sum::<f64>() - expected).abs() < 1e-10
        );
        assert!(actual.source_evidence[0].accepted);
        // Each field row must produce exactly the same consistent load as a scalar source
        // on that single canonical region; this catches accidental row reindexing.
        let mut expected_nodes = vec![0.0; actual.applied_source.scalar_count()];
        for (row, element) in actual.projection.elements.iter().enumerate() {
            actual
                .projection
                .visit_element_quadrature(element, |point| {
                    for (&node, &shape) in element.node_indices.iter().zip(point.shape_values()) {
                        expected_nodes[node] += (row + 1) as f64 * shape * point.volume_weight_m3;
                    }
                })
                .unwrap();
        }
        for (a, b) in actual
            .applied_source
            .canonical_values()
            .iter()
            .zip(expected_nodes)
        {
            assert!((a - b).abs() < 1e-12);
        }
        // A constant field and the established scalar source must agree node-by-node.
        let SignalRepresentation::Constant(SignalValue::Tensor { values, .. }) =
            &mut signal(&mut model).representation
        else {
            unreachable!()
        };
        values.fill(HeatGenerationDensity::new(7.0));
        let field = run(
            &model,
            &mesh,
            &realize_excitations(&model, None, ExcitationSamplePoint::Static).unwrap(),
        )
        .unwrap();
        signal(&mut model).shape = SignalShape::Scalar;
        signal(&mut model).representation =
            SignalRepresentation::Constant(SignalValue::Scalar(HeatGenerationDensity::new(7.0)));
        let ExcitationAction::HeatGeneration(action) = &mut model.excitations[0] else {
            unreachable!()
        };
        action.distribution = ExcitationDistribution::IntensityPerVolume;
        let scalar = run(
            &model,
            &mesh,
            &realize_excitations(&model, None, ExcitationSamplePoint::Static).unwrap(),
        )
        .unwrap();
        assert_eq!(
            field.applied_source.canonical_values(),
            scalar.applied_source.canonical_values()
        );
    }
}

#[test]
fn element_density_rejects_foreign_topology_location_cardinality_and_policy() {
    for order in [ElementOrder::Tet4, ElementOrder::Tet10] {
        let (model, mesh) = fixture(order);
        for location in [FieldLocation::Node, FieldLocation::IntegrationPoint] {
            let mut bad = model.clone();
            let SignalShape::TopologyField {
                location: declared, ..
            } = &mut signal(&mut bad).shape
            else {
                unreachable!()
            };
            *declared = location;
            assert!(realize_excitations(&bad, None, ExcitationSamplePoint::Static).is_err());
        }
        let mut bad = model.clone();
        let SignalShape::TopologyField { topology_id, .. } = &mut signal(&mut bad).shape else {
            unreachable!()
        };
        *topology_id = Digest::from_bytes([7; 32]).to_string();
        let sources = realize_excitations(&bad, None, ExcitationSamplePoint::Static).unwrap();
        assert!(matches!(
            run(&bad, &mesh, &sources),
            Err(ThermalAssemblyError::InvalidProjection(_))
        ));
        let mut bad = model.clone();
        let SignalRepresentation::Constant(SignalValue::Tensor { shape, values }) =
            &mut signal(&mut bad).representation
        else {
            unreachable!()
        };
        shape[0] -= 1;
        values.pop();
        let sources = realize_excitations(&bad, None, ExcitationSamplePoint::Static).unwrap();
        assert!(matches!(
            run(&bad, &mesh, &sources),
            Err(ThermalAssemblyError::InvalidProjection(_))
        ));
        let mut bad = model.clone();
        let ExcitationAction::HeatGeneration(action) = &mut bad.excitations[0] else {
            unreachable!()
        };
        action.distribution = ExcitationDistribution::IntensityPerVolume;
        assert!(realize_excitations(&bad, None, ExcitationSamplePoint::Static).is_err());
        for wrong_shape in [
            vec![mesh.topology.volume_elements.len()],
            vec![mesh.topology.volume_elements.len() / 2, 2],
        ] {
            let mut bad = model.clone();
            let SignalRepresentation::Constant(SignalValue::Tensor { shape, .. }) =
                &mut signal(&mut bad).representation
            else {
                unreachable!()
            };
            *shape = wrong_shape;
            assert!(realize_excitations(&bad, None, ExcitationSamplePoint::Static).is_err());
        }
    }
}

#[test]
fn selected_region_keeps_full_canonical_rows_and_integrates_only_that_region() {
    for order in [ElementOrder::Tet4, ElementOrder::Tet10] {
        let (mut model, mut mesh) = fixture(order);
        let mut selected = mesh.topology.volume_elements[0].region_id.clone();
        selected.source_topology_id = "selected".into();
        let mut other = selected.clone();
        other.source_topology_id = "other".into();
        let mut selected_ids = Vec::new();
        let mut other_ids = Vec::new();
        let mut expected = 0.0;
        let count = mesh.topology.volume_elements.len();
        for (row, element) in mesh.topology.volume_elements.iter_mut().enumerate() {
            // Canonical row values remain global, not compacted to the selected region.
            if row % 2 == 0 {
                element.region_id = selected.clone();
                selected_ids.push(element.element_id);
                expected += (row + 1) as f64 / count as f64;
            } else {
                element.region_id = other.clone();
                other_ids.push(element.element_id);
            }
        }
        mesh.topology.regions = vec![
            runmat_meshing_core::MeshRegion {
                region_id: selected,
                element_ids: selected_ids,
            },
            runmat_meshing_core::MeshRegion {
                region_id: other,
                element_ids: other_ids,
            },
        ];
        mesh.topology
            .regions
            .sort_by(|a, b| a.region_id.cmp(&b.region_id));
        mesh.seal_canonical_digest().unwrap();
        let SignalShape::TopologyField { topology_id, .. } = &mut signal(&mut model).shape else {
            unreachable!()
        };
        *topology_id = Digest::from_bytes(*mesh.canonical_digest.bytes()).to_string();
        let ExcitationAction::HeatGeneration(action) = &mut model.excitations[0] else {
            unreachable!()
        };
        action.target = ExcitationTarget::Region(runmat_analysis_core::RegionTarget::volume(
            runmat_meshing_core::fixtures::fixture_region("selected"),
        ));
        let sources = realize_excitations(&model, None, ExcitationSamplePoint::Static).unwrap();
        let output = run(&model, &mesh, &sources).unwrap();
        assert!(
            (output.applied_source.canonical_values().iter().sum::<f64>() - expected).abs() < 1e-10
        );
        assert!(output.source_evidence[0].accepted);
    }
}

#[test]
fn element_density_complex_selection_is_explicit_and_static_imaginary_is_rejected() {
    for order in [ElementOrder::Tet4, ElementOrder::Tet10] {
        let (mut model, mesh) = fixture(order);
        let real = match signal(&mut model).representation.clone() {
            SignalRepresentation::Constant(value) => value,
            _ => unreachable!(),
        };
        signal(&mut model).representation = SignalRepresentation::Tabulated(TabulatedSignal {
            domain: TabulatedSignalDomain::TimeSeconds,
            knots: vec![SignalKnot::continuous(
                0.0,
                SignalOrdinate::Complex {
                    real: real.clone(),
                    imaginary: real,
                },
            )],
            interpolation: SignalInterpolation::Linear,
            extrapolation: SignalExtrapolation::Reject,
        });
        let sources = realize_excitations(&model, None, ExcitationSamplePoint::Static).unwrap();
        assert!(matches!(
            run(&model, &mesh, &sources),
            Err(ThermalAssemblyError::ComplexSource { .. })
        ));
        let real = run(&model, &mesh, &sources.component(ComplexComponent::Real)).unwrap();
        let imaginary = run(
            &model,
            &mesh,
            &sources.component(ComplexComponent::Imaginary),
        )
        .unwrap();
        assert_eq!(
            real.applied_source.canonical_values(),
            imaginary.applied_source.canonical_values()
        );
    }
}

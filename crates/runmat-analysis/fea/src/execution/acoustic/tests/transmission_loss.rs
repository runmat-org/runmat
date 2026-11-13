//! Transmission loss through the shared paired-port observation, against the closed-form
//! plane-wave result for a layer of another medium between an incident and an anechoic port.
use super::*;
use crate::results::{
    field::RecoveredFieldCatalog,
    observation::{evaluate_observations, resolve_observation_plan, ObservationLimits},
};
use crate::topology::test_meshes::entity;
use runmat_analysis_core::*;
use runmat_meshing_core::{MeshRegion, PersistentEntityKind};

/// Ambient medium (ρ 3, c 2: Z₁ = 6) on both sides of a layer (ρ 5, c 4: Z₂ = 20) of
/// thickness 1/3 in the unit slab; 1 Pa incident on the left face, plane-wave radiation on the
/// right face, rigid side walls. `m = Z₂/Z₁`; the layer is the expansion-chamber equation's
/// impedance ratio in place of the area ratio.
fn layered(cells: usize, frequency: f64) -> (AcousticHarmonicWorkload, SolverMeshArtifact) {
    let mut mesh = quadratic_slab_mesh(cells);
    let regions = [
        entity(PersistentEntityKind::Region, "ambient"),
        entity(PersistentEntityKind::Region, "layer"),
    ];
    let mut elements = [Vec::new(), Vec::new()];
    let topology = &mut *mesh.topology;
    for element in &mut topology.volume_elements {
        let x = element.node_ids[..4]
            .iter()
            .map(|id| {
                topology.nodes[*id as usize - 1]
                    .coordinates_m
                    .canonical_value()[0]
            })
            .sum::<f64>()
            / 4.0;
        let side = usize::from(x > 1.0 / 3.0 && x < 2.0 / 3.0);
        element.region_id = regions[side].clone();
        elements[side].push(element.element_id);
    }
    mesh.topology.regions = regions
        .into_iter()
        .zip(elements)
        .map(|(region_id, element_ids)| MeshRegion {
            region_id,
            element_ids,
        })
        .collect();
    mesh.seal_canonical_digest().unwrap();
    mesh.validate().unwrap();

    let mut authored = workload();
    let model = &mut authored.model;
    model.constraints.clear();
    model.boundaries.clear();
    model.material_assignments[0].target =
        PropertyRegionTarget::solid(runmat_meshing_core::fixtures::fixture_region("ambient"));
    let mut layer = model.materials[0].clone();
    layer.material_id = "layer_fluid".into();
    layer.mass_density_kg_per_m3 = Some(runmat_types::physical::MassDensity::new(5.0));
    layer.acoustic.as_mut().unwrap().speed_of_sound_m_per_s =
        runmat_types::physical::SpeedOfSound::new(4.0);
    let mut assignment = model.material_assignments[0].clone();
    assignment.target =
        PropertyRegionTarget::solid(runmat_meshing_core::fixtures::fixture_region("layer"));
    assignment.assigned_material_id = layer.material_id.clone();
    assignment.expected_material_id = layer.material_id.clone();
    model.materials.push(layer);
    model.material_assignments.push(assignment);
    model.signals.push(PhysicalSignal::Pressure(Signal {
        signal_id: SignalId("incident".into()),
        shape: SignalShape::Scalar,
        frame: None,
        representation: SignalRepresentation::Constant(SignalValue::Scalar(Pressure::new(1.0))),
        precision: SignalPrecision::Real64,
        evidence: SignalApproximationEvidence {
            max_absolute_error: Some(0.0),
            max_relative_error: Some(0.0),
            derivative_max_absolute_error: None,
            sample_count: 1,
        },
        provenance: SignalProvenance::Authored,
    }));
    // The incident wave replaces the fixture's loads; its signal was pushed above.
    model.excitations.clear();
    model.excitation_cases = vec![ExcitationCase {
        case_id: ExcitationCaseId("incident-case".into()),
        action_ids: vec![ExcitationId("engine".into())],
        activation: ExcitationCaseActivation::Always,
    }];
    model
        .excitations
        .extend([ExcitationAction::AcousticIncidentPressure(
            AnalysisExcitation {
                excitation_id: ExcitationId("engine".into()),
                target: ExcitationTarget::Region(RegionTarget::boundary(
                    runmat_meshing_core::fixtures::fixture_face("slab_left"),
                )),
                frame: CoordinateFrame::Global,
                direction: None,
                distribution: ExcitationDistribution::UniformKinematic,
                kinematics: ExcitationKinematics::FixedFrame,
                signal_id: SignalId("incident".into()),
                active_step_ids: Vec::new(),
                combination_group: None,
                projection: ProjectionRequirement {
                    preserve_resultant: true,
                    preserve_moment: false,
                    relative_tolerance: 1e-10,
                },
            },
        )]);
    model.boundaries.push(AnalysisBoundary {
        boundary_id: "tailpipe".into(),
        target: RegionTarget::boundary(runmat_meshing_core::fixtures::fixture_face("slab_right")),
        operator: BoundaryOperator::AcousticRadiation,
    });
    model.sequence.steps[0].independent_domain = IndependentDomain::FrequencyHertz(vec![frequency]);
    model.sequence.steps[0]
        .observations
        .push(ObservationRequest::Reduction(ObservationSpec {
            observation_id: ObservationId("transmission_loss".into()),
            quantity: ObservationQuantity::AcousticPortIntensity,
            value_shape: ObservationValueShape::Scalar,
            target: ObservationTarget::RegionPair {
                first: RegionTarget::boundary(runmat_meshing_core::fixtures::fixture_face(
                    "slab_left",
                )),
                second: RegionTarget::boundary(runmat_meshing_core::fixtures::fixture_face(
                    "slab_right",
                )),
            },
            source_location: FieldLocation::Element,
            output_location: FieldLocation::Global,
            recovery: RecoveryPolicy::Native,
            component: ObservationComponent::All,
            frame: CoordinateFrame::Global,
            sign_convention: SignConvention::Native,
            complex_representation: ComplexRepresentation::Real,
            sampling: SamplingPolicy::Final,
            reduction: Some(ReductionPolicy {
                spatial: SpatialReduction::TransmissionLoss,
                series: SeriesReduction::None,
                extrema_ties: ExtremaTiePolicy::FirstCanonical,
            }),
            storage: ObservationStoragePolicy {
                precision: SignalPrecision::Real64,
                compression: CompressionPolicy::Lossless,
                retention: RetentionPolicy::Canonical,
                partial_access: PartialAccessPolicy::ManifestAndChunks,
            },
            purpose: ObservationPurpose::Publication,
            cost: ObservationCostPolicy::UpperBound(ObservationCostEstimate {
                operation_count: 100_000_000,
                peak_resident_bytes: 1_000_000_000,
                stored_bytes: 1_000_000,
                transfer_bytes: 1_000_000,
                collective_count: 0,
            }),
        }));
    authored.frequency = Frequency::new(frequency);
    (authored, mesh)
}

fn transmission_loss(cells: usize, frequency: f64) -> f64 {
    let (authored, mesh) = layered(cells, frequency);
    let result = authored.execute(&mesh).unwrap();
    let request = result.native_field_request(&authored, &mesh).unwrap();
    let artifact = result
        .recover_native_fields(&authored, &mesh, &request)
        .unwrap();
    let definitions = artifact
        .fields()
        .iter()
        .map(|field| field.definition().clone())
        .collect::<Vec<_>>();
    let projection = crate::topology::AnalysisProjection::derive(&mesh).unwrap();
    let catalog = RecoveredFieldCatalog::new_for_topology(
        projection.projection_digest,
        artifact.entity_counts(),
        artifact.into_fields(),
    )
    .unwrap();
    let plan = resolve_observation_plan(
        &authored.model.sequence.steps[0],
        &projection,
        &definitions,
        ObservationLimits::default(),
    )
    .unwrap();
    let observed = evaluate_observations(&plan, &projection, &catalog).unwrap();
    assert_eq!(
        observed[0].quantity,
        ObservationQuantity::AcousticPortIntensity
    );
    observed[0].field.as_host_f64().unwrap()[0]
}

/// TL = 10·log10[1 + ¼(m − 1/m)² sin²(k₂L)], m = Z₂/Z₁, k₂ = ω/c₂, L = 1/3.
fn reference(frequency: f64) -> f64 {
    let m: f64 = 20.0 / 6.0;
    let k2l = std::f64::consts::TAU * frequency / 4.0 / 3.0;
    10.0 * (1.0 + 0.25 * (m - 1.0 / m).powi(2) * k2l.sin().powi(2)).log10()
}

// witness: FEA-ACOUSTIC-OUTPUTS-001/intensity_and_sound_power @ acoustic_harmonic/tet10/numerical_kernel;
//   coverage: partial; check: layer-transmission-loss; boundary: transmission loss through a plane layer between an incident and
//   an anechoic port matches the closed-form plane-wave result per frequency and converges under
//   refinement; three-dimensional chamber modes are not exercised.
#[test]
fn transmission_loss_of_a_layer_matches_the_plane_wave_reference() {
    // k₂L from π/6 to π/2: 1.97, 4.35 and 5.19 dB (the peak). Cells are multiples of three so
    // the layer faces are element faces.
    for frequency in [1.0, 2.0, 3.0] {
        let expected = reference(frequency);
        let coarse = (transmission_loss(6, frequency) - expected).abs();
        let fine = (transmission_loss(9, frequency) - expected).abs();
        eprintln!(
            "TL f={frequency}: reference {expected:.5} dB, error {coarse:.2e} (6) {fine:.2e} (9)"
        );
        assert!(fine < 0.05, "f={frequency}: |TL - reference| = {fine} dB");
        assert!(fine <= coarse + 1e-9, "f={frequency}: no refinement gain");
    }
    // No layer contrast: the whole incident power leaves through the anechoic end.
    assert!(reference(0.0).abs() < 1e-15);
}

//! `pressure_drop` through the shared paired-port observation on plane Poiseuille flow: the
//! drop between the inlet and outlet faces is (−dp/dx)·L with −dp/dx = 12 µ Ū / h².
use super::*;
use crate::results::{
    field::RecoveredFieldCatalog,
    observation::{evaluate_observations, resolve_observation_plan, ObservationLimits},
};
use runmat_meshing_core::{PersistentEntityId, PersistentEntityKind};

fn port(name: &str) -> PersistentEntityId {
    PersistentEntityId::new(
        PersistentEntityKind::Face,
        name.to_string(),
        vec!["root".into()],
    )
}

/// Unit channel, walls at y = 0 and 1, symmetry planes at z = 0 and 1 (plane flow), the
/// parabolic profile u = 6 Ū y (1 − y) with Ū = 1 m/s prescribed face by face at x = 0 and
/// 0 Pa at x = 1. µ = 1 Pa·s and ρ = 1 kg/m³ (Re = 1). Returns the workload, its mesh and
/// the discrete mean inlet speed.
fn channel(n: usize) -> (FluidSteadyWorkload, SolverMeshArtifact, f64) {
    let mut mesh = fixtures::slab_mesh(n);
    let coordinates = |id: u64, mesh: &SolverMeshArtifact| {
        mesh.topology.nodes[id as usize - 1]
            .coordinates_m
            .canonical_value()
    };
    let mut faces = Vec::new();
    for face in &mesh.topology.boundary_faces {
        let corners = face.node_ids[..3]
            .iter()
            .map(|id| coordinates(*id, &mesh))
            .collect::<Vec<_>>();
        let centroid: [f64; 3] =
            std::array::from_fn(|k| corners.iter().map(|x| x[k]).sum::<f64>() / 3.0);
        let edge = |a: usize, b: usize| -> [f64; 3] {
            std::array::from_fn(|k| corners[b][k] - corners[a][k])
        };
        let (u, v) = (edge(0, 1), edge(0, 2));
        let cross = [
            u[1] * v[2] - u[2] * v[1],
            u[2] * v[0] - u[0] * v[2],
            u[0] * v[1] - u[1] * v[0],
        ];
        let area = 0.5 * cross.iter().map(|x| x * x).sum::<f64>().sqrt();
        faces.push((face.face_id, centroid, area));
    }
    let mut constraints = Vec::new();
    let mut boundaries = Vec::new();
    let (mut flow, mut inlet_area) = (0.0, 0.0);
    for (face, (face_id, centroid, area)) in mesh.topology.boundary_faces.iter_mut().zip(&faces) {
        let own = port(&format!("face-{face_id}"));
        let mut target = RegionTarget::boundary(format!("face-{face_id}"));
        target.bind_resolved_entities(vec![own.clone()]).unwrap();
        let [x, y, z] = *centroid;
        face.provenance = vec![own];
        if x.abs() < 1e-12 {
            face.provenance.push(port("inlet"));
            let speed = 6.0 * y * (1.0 - y);
            flow += speed * area;
            inlet_area += area;
            constraints.push(AnalysisConstraint::Fluid(FluidConstraint {
                constraint_id: format!("inlet-{face_id}"),
                kind: FluidConstraintKind::PrescribedVelocity {
                    target,
                    value: PrescribedValue::Constant([speed, 0.0, 0.0].map(Velocity::new)),
                },
            }));
        } else if (x - 1.0).abs() < 1e-12 {
            face.provenance.push(port("outlet"));
            boundaries.push(AnalysisBoundary {
                boundary_id: format!("outlet-{face_id}"),
                target,
                operator: BoundaryOperator::FluidOutletPressure {
                    value: PrescribedValue::Constant(Pressure::new(0.0)),
                },
            });
        } else if y.abs() < 1e-12 || (y - 1.0).abs() < 1e-12 {
            constraints.push(AnalysisConstraint::Fluid(FluidConstraint {
                constraint_id: format!("wall-{face_id}"),
                kind: FluidConstraintKind::NoSlipWall { target },
            }));
        } else {
            assert!(z.abs() < 1e-12 || (z - 1.0).abs() < 1e-12);
            constraints.push(AnalysisConstraint::Fluid(FluidConstraint {
                constraint_id: format!("span-{face_id}"),
                kind: FluidConstraintKind::Symmetry { target },
            }));
        }
    }
    mesh.seal_canonical_digest().unwrap();
    let (mut workload, _) = authored();
    workload.model.constraints = constraints;
    workload.model.boundaries = boundaries;
    workload.policy = crate::physics::cfd::canonical::CanonicalFluidPolicy::default();
    workload.geometry_policy = crate::topology::FiniteVolumeProjectionPolicy::default();
    let mut drop = RegionTarget::boundary("inlet".to_string());
    drop.bind_resolved_entities(vec![port("inlet")]).unwrap();
    let mut outlet = RegionTarget::boundary("outlet".to_string());
    outlet.bind_resolved_entities(vec![port("outlet")]).unwrap();
    workload.model.sequence.steps[0]
        .observations
        .push(ObservationRequest::Reduction(ObservationSpec {
            observation_id: ObservationId("pressure_drop".into()),
            quantity: ObservationQuantity::FluidPressure,
            value_shape: ObservationValueShape::Scalar,
            target: ObservationTarget::RegionPair {
                first: drop,
                second: outlet,
            },
            source_location: FieldLocation::Face,
            output_location: FieldLocation::Global,
            recovery: RecoveryPolicy::Native,
            component: ObservationComponent::All,
            frame: CoordinateFrame::Global,
            sign_convention: SignConvention::Native,
            complex_representation: ComplexRepresentation::Real,
            sampling: SamplingPolicy::Final,
            reduction: Some(ReductionPolicy {
                spatial: SpatialReduction::MeanDifference,
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
    (workload, mesh, flow / inlet_area)
}

fn pressure_drop(n: usize) -> (f64, f64) {
    let (workload, mesh, mean_speed) = channel(n);
    let result = workload.execute(&mesh).unwrap();
    let source = workload.prepare(&mesh).unwrap();
    let request = result.native_field_request(&workload, &mesh).unwrap();
    let artifact = result
        .recover_native_fields(&workload, &mesh, &request)
        .unwrap();
    let definitions = artifact
        .fields()
        .iter()
        .map(|field| field.definition().clone())
        .collect::<Vec<_>>();
    let catalog = RecoveredFieldCatalog::new_for_topology(
        source.source.projection_digest,
        artifact.entity_counts(),
        artifact.into_fields(),
    )
    .unwrap();
    let plan = resolve_observation_plan(
        &workload.model.sequence.steps[0],
        &source.source,
        &definitions,
        ObservationLimits::default(),
    )
    .unwrap();
    let observed = evaluate_observations(&plan, &source.source, &catalog).unwrap();
    let drop = observed[0].field.as_host_f64().unwrap()[0];
    assert_eq!(observed[0].quantity, ObservationQuantity::FluidPressure);
    (drop, mean_speed)
}

// witness: FEA-FLUID-OUTPUTS-001/pressure_drop_and_flow_rate @ incompressible_laminar_steady/canonical_tetrahedra/numerical_kernel;
//   coverage: partial; boundary: the face-pressure mean difference between the inlet and outlet
//   of plane Poiseuille flow approaches (−dp/dx)·L = 12 µ Ū L / h² under refinement; developing
//   and three-dimensional flows are not referenced.
#[test]
fn pressure_drop_of_plane_poiseuille_flow_is_twelve_mu_u_l_over_h_squared() {
    let mut errors = Vec::new();
    for n in [3, 6] {
        let (drop, mean_speed) = pressure_drop(n);
        // Reference from the discrete inlet flow: −dp/dx = 12 µ Ū / h², L = h = 1.
        let reference = 12.0 * mean_speed;
        let error = (drop - reference).abs() / reference;
        eprintln!("pressure_drop n={n}: {drop:.5} Pa, reference {reference:.5} Pa, relative error {error:.3e}");
        errors.push(error);
    }
    assert!(errors[1] < errors[0], "no refinement gain: {errors:?}");
    assert!(errors[1] < 0.05, "relative error {:?}", errors[1]);
}

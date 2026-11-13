use super::{super::*, reference};
use crate::topology::{AnalysisProjection, FiniteVolumeProjectionPolicy, FiniteVolumeTopology};
use runmat_analysis_core::*;
use runmat_meshing_core::{fixtures, PersistentEntityId, PersistentEntityKind};

#[derive(Clone, Copy)]
pub enum ReferenceCase {
    Couette,
    Convective,
    Duct,
}
pub struct Fixture {
    pub mesh: runmat_meshing_core::SolverMeshArtifact,
    pub source: AnalysisProjection,
    pub topology: FiniteVolumeTopology,
    pub constraints: Vec<FluidConstraint>,
    pub operators: Vec<AnalysisBoundary>,
    pub signals: Vec<PhysicalSignal>,
    pub actions: Vec<ExcitationAction>,
    pub material: CfdDomain,
    pub policy: CanonicalFluidPolicy,
    pub reference: ReferenceCase,
    pub rotated: bool,
}
impl Fixture {
    pub fn input(&self) -> CanonicalFluidInput<'_> {
        CanonicalFluidInput {
            source: &self.source,
            topology: &self.topology,
            material: &self.material,
            constraints: &self.constraints,
            boundary_operators: &self.operators,
            signals: &self.signals,
            excitations: &self.actions,
            policy: self.policy,
        }
    }
}
pub fn material() -> CfdDomain {
    CfdDomain {
        density: MassDensity::new(1.0),
        dynamic_viscosity: DynamicViscosity::new(1.0),
        turbulence_intensity: TurbulenceIntensity::new(0.0),
        rheology: None,
        porous_zones: Vec::new(),
    }
}
pub fn policy() -> CanonicalFluidPolicy {
    CanonicalFluidPolicy {
        pressure_velocity_coupling: PressureVelocityCoupling::Coupled,
        fixed_point_acceleration: None,
        maximum_outer_iterations: 2000,
        maximum_nonorthogonal_iterations: 8,
        maximum_linear_iterations: 1000,
        gmres_restart: 48,
        momentum_relaxation: 0.5,
        pressure_relaxation: 0.25,
        linear_relative_tolerance: 1e-11,
        physical_relative_tolerance: 1e-7,
        maximum_resident_bytes: Some(2_000_000_000),
        maximum_operator_work: Some(100_000_000_000_000),
        maximum_authority_items: 10000,
    }
}

pub fn geometry_policy() -> FiniteVolumeProjectionPolicy {
    FiniteVolumeProjectionPolicy {
        max_cells: None,
        max_face_incidences: None,
        max_boundary_faces: None,
        max_scratch_bytes: None,
        max_traversal_operations: None,
    }
}

pub fn case(n: usize, reference: ReferenceCase, rotated: bool) -> Fixture {
    case_mapped(n, reference, rotated, |x| x)
}

/// The reference case on a slab whose nodes are moved by `map` (for example graded toward a
/// plane), keeping its topology and boundary faces.
pub fn case_mapped(
    n: usize,
    reference: ReferenceCase,
    rotated: bool,
    map: fn([f64; 3]) -> [f64; 3],
) -> Fixture {
    let mut mesh = fixtures::slab_mesh(n);
    for node in &mut mesh.topology.nodes {
        node.coordinates_m =
            runmat_types::physical::Quantity::new(map(node.coordinates_m.canonical_value()));
    }
    if rotated {
        for node in &mut mesh.topology.nodes {
            node.coordinates_m =
                runmat_types::physical::Quantity::<runmat_types::physical::kind::Position, _>::new(
                    reference::rotate(node.coordinates_m.canonical_value()),
                );
        }
    }
    for face in &mut mesh.topology.boundary_faces {
        face.provenance = vec![PersistentEntityId::new(
            PersistentEntityKind::Face,
            format!("original-reference-face-{}", face.face_id),
            vec!["root".into()],
        )];
    }
    mesh.seal_canonical_digest().unwrap();
    let source = AnalysisProjection::derive(&mesh).unwrap();
    let topology = FiniteVolumeTopology::derive(&source, geometry_policy()).unwrap();
    let mut constraints = Vec::new();
    let mut operators = Vec::new();
    for f in &source.boundary_faces {
        let mut target = RegionTarget::boundary(format!("original-reference-face-{}", f.face_id));
        target.bind_resolved_entities(f.provenance.clone()).unwrap();
        if reference::outlet(f.centroid_m, rotated) {
            operators.push(AnalysisBoundary {
                boundary_id: format!("outlet-{}", f.face_id),
                target,
                operator: BoundaryOperator::FluidOutletPressure {
                    value: PrescribedValue::Constant(Pressure::new(reference::pressure(
                        reference,
                        f.centroid_m,
                        rotated,
                    ))),
                },
            });
        } else {
            let v = reference::velocity(reference, f.centroid_m, rotated).map(Velocity::new);
            constraints.push(FluidConstraint {
                constraint_id: format!("original-vector-face-{}", f.face_id),
                kind: FluidConstraintKind::PrescribedVelocity {
                    target,
                    value: PrescribedValue::Constant(v),
                },
            });
        }
    }
    let mut signals = Vec::new();
    let mut actions = Vec::new();
    if matches!(reference, ReferenceCase::Duct) {
        let force = if rotated {
            reference::rotate([1.0, 0.0, 0.0])
        } else {
            [1.0, 0.0, 0.0]
        };
        signals.push(PhysicalSignal::BodyForceDensity(Signal {
            signal_id: SignalId("original-duct-force".into()),
            shape: SignalShape::Vector3,
            frame: Some(CoordinateFrame::Global),
            representation: SignalRepresentation::Constant(SignalValue::Vector3(
                force.map(BodyForceDensity::new),
            )),
            precision: SignalPrecision::Real64,
            evidence: SignalApproximationEvidence {
                max_absolute_error: Some(0.0),
                max_relative_error: Some(0.0),
                derivative_max_absolute_error: None,
                sample_count: 1,
            },
            provenance: SignalProvenance::Authored,
        }));
        actions.push(ExcitationAction::BodyForce(AnalysisExcitation {
            excitation_id: ExcitationId("original-duct-volume-force".into()),
            target: ExcitationTarget::WholeModel,
            frame: CoordinateFrame::Global,
            direction: None,
            distribution: ExcitationDistribution::IntensityPerVolume,
            kinematics: ExcitationKinematics::FixedFrame,
            signal_id: SignalId("original-duct-force".into()),
            active_step_ids: Vec::new(),
            combination_group: None,
            projection: ProjectionRequirement {
                preserve_resultant: true,
                preserve_moment: true,
                relative_tolerance: 1e-12,
            },
        }));
    }
    Fixture {
        mesh,
        source,
        topology,
        constraints,
        operators,
        signals,
        actions,
        material: material(),
        policy: policy(),
        reference,
        rotated,
    }
}

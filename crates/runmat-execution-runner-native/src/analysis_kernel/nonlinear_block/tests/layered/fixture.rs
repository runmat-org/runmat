//! Two material layers, homogeneous zero transverse displacement, axial traction.
use super::*;
use runmat_meshing_core::{fixtures::quadratic_slab_mesh, MeshRegion};

pub(super) fn prepare(quadratic: bool) -> (NonlinearSolidWorkload, SolverMeshArtifact) {
    let (mut workload, _) = super::super::fixture(quadratic);
    let mut mesh = if quadratic {
        quadratic_slab_mesh(2)
    } else {
        slab_mesh(2)
    };
    let original_region = mesh.topology.regions[0].region_id.clone();
    mesh.topology.regions.clear();
    for layer in 0..2 {
        let mut region = original_region.clone();
        region.source_topology_id = format!("layer_{layer}");
        let mut elements = Vec::new();
        let topo = &mut *mesh.topology;
        for element in &mut topo.volume_elements {
            let x = element.node_ids[..4]
                .iter()
                .map(|id| {
                    topo.nodes
                        .iter()
                        .find(|node| node.node_id == *id)
                        .unwrap()
                        .coordinates_m
                        .canonical_value()[0]
                })
                .sum::<f64>()
                / 4.0;
            if usize::from(x > 0.5) == layer {
                element.region_id = region.clone();
                element.provenance = vec![region.clone()];
                elements.push(element.element_id);
            }
        }
        assert!(!elements.is_empty());
        mesh.topology.regions.push(MeshRegion {
            region_id: region,
            element_ids: elements,
        });
    }
    mesh.seal_canonical_digest().unwrap();
    mesh.validate_canonical().unwrap();
    let model = &mut workload.model;
    model.sequence.steps[0].accuracy.criteria = vec![
        AccuracyCriterion {
            criterion_id: "newton".into(),
            target: AccuracyTarget::GlobalEquilibrium,
            contribution: ErrorContributionKind::Nonlinear,
            tolerance: AccuracyTolerance::Relative(1e-9),
            evidence: RequiredAccuracyEvidence::ResidualBound,
            mandatory: true,
        },
        AccuracyCriterion {
            criterion_id: "mesh".into(),
            target: AccuracyTarget::NumericalResult,
            contribution: ErrorContributionKind::SpatialDiscretization,
            tolerance: AccuracyTolerance::Relative(1e-7),
            evidence: RequiredAccuracyEvidence::ConvergenceStudy,
            mandatory: true,
        },
    ];
    let original_material = model.materials[0].clone();
    model.materials.clear();
    model.material_assignments.clear();
    for (layer, yield_stress) in [2.0, 3.5].into_iter().enumerate() {
        let mut material = original_material.clone();
        material.material_id = format!("layer_material_{layer}");
        material.plastic = Some(MaterialPlasticModel::J2IsotropicLinearHardening {
            yield_stress_pa: runmat_types::physical::YieldStress::new(yield_stress),
            isotropic_hardening_modulus_pa: runmat_types::physical::HardeningModulus::new(30.0),
        });
        model.material_assignments.push(MaterialAssignment {
            target: PropertyRegionTarget::solid({
                let mut region = original_region.clone();
                region.source_topology_id = format!("layer_{layer}");
                region.analysis_selection_id().unwrap()
            }),
            expected_material_id: material.material_id.clone(),
            assigned_material_id: material.material_id.clone(),
            confidence: EvidenceConfidence::Verified,
        });
        model.materials.push(material);
    }
    model.constraints.clear();
    for node in &mesh.topology.nodes {
        let mut translations = vec![
            TranslationDofValue {
                dof: StructuralTranslationDof::Uy,
                value: PrescribedValue::Constant(Displacement::new(0.0)),
            },
            TranslationDofValue {
                dof: StructuralTranslationDof::Uz,
                value: PrescribedValue::Constant(Displacement::new(0.0)),
            },
        ];
        if node.coordinates_m.canonical_value()[0] == 0.0 {
            translations.push(TranslationDofValue {
                dof: StructuralTranslationDof::Ux,
                value: PrescribedValue::Constant(Displacement::new(0.0)),
            });
        }
        model
            .constraints
            .push(AnalysisConstraint::Structural(StructuralConstraint {
                constraint_id: format!("layered_pin_{}", node.node_id),
                frame: CoordinateFrame::Global,
                kind: StructuralConstraintKind::Prescribed(PrescribedStructuralDofs {
                    target: RegionTarget::point(format!("node:{}", node.node_id)),
                    translations,
                    rotations: Vec::new(),
                }),
            }));
    }
    workload.validate().unwrap();
    (workload, mesh)
}

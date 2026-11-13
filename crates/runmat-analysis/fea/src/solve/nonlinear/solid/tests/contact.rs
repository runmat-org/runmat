//! Independent constrained deformable two-body penalty/Coulomb reference.
use super::*;
use crate::{
    assembly::{assemble_structural_system_borrowed, StructuralFormulationRequirement},
    excitation::RealizedExcitations,
    fixtures::{fixture_model, replace_fixture_excitations, FixtureId},
    physics::structural::{
        contact::{CanonicalContactOperator, ContactOperatorTestFixture},
        topology::node_space::CanonicalStructuralNodeSpace,
    },
    solve::general::{GeneralSolveEnvelope, GeneralSolvePolicy},
    topology::AnalysisProjection,
};
use runmat_analysis_core::{MaterialElasticityModel, MaterialMechanicalModel};
mod blocks;
mod cold;
mod energy;
mod reference;
mod resources;
mod rollback;
mod tangent;
struct CoupledFixture {
    surface: ContactOperatorTestFixture,
    model: runmat_analysis_core::AnalysisModel,
    summary: AssemblySummary,
}
impl CoupledFixture {
    fn new(order: ElementOrder, friction: bool) -> Self {
        let mut surface = ContactOperatorTestFixture::new(order, friction);
        for node in &mut surface.mesh.topology.nodes {
            if (100..200).contains(&node.node_id) {
                {
                    let quantity_change = 0.0499;
                    let quantity_index = 2;
                    let quantity_slot = &mut node.coordinates_m;
                    let mut quantity_coordinates = quantity_slot.canonical_value();
                    quantity_coordinates[quantity_index] += quantity_change;
                    *quantity_slot = runmat_types::physical::Quantity::<
                        runmat_types::physical::kind::Position,
                        _,
                    >::new(quantity_coordinates);
                };
            }
        }
        surface.mesh.seal_canonical_digest().unwrap();
        surface.mesh.validate().unwrap();
        surface.projection = AnalysisProjection::derive(&surface.mesh).unwrap();
        surface.positions = surface
            .projection
            .nodes
            .iter()
            .map(|n| n.coordinates_m.canonical_value())
            .collect();
        let mut model = fixture_model(FixtureId::MultiMaterialAssembly);
        replace_fixture_excitations(&mut model, Vec::new());
        model.constraints.clear();
        for material in &mut model.materials {
            material.elasticity = Some(MaterialElasticityModel::IsotropicLinear(
                MaterialMechanicalModel {
                    youngs_modulus_pa: runmat_types::physical::YoungsModulus::new(1200.),
                    poisson_ratio: runmat_types::physical::PoissonRatio::new(0.),
                },
            ));
            material.plastic = None;
        }
        for assignment in &mut model.material_assignments {
            assignment.assigned_material_id = {
                let region = assignment.selected_region_id().unwrap();
                let fixture = runmat_meshing_core::fixtures::fixture_region;
                if region == fixture("tip_steel") {
                    "mat_steel"
                } else if region == fixture("mid_aluminum") {
                    "mat_aluminum"
                } else if region == fixture("polymer_segment") {
                    "mat_polymer"
                } else {
                    panic!("region {region}")
                }
            }
            .into();
        }
        let mut summary = assemble_structural_system_borrowed(
            &model,
            &RealizedExcitations::default(),
            Some(&surface.mesh),
            StructuralFormulationRequirement::PhysicalDiscretization,
        )
        .unwrap();
        let nodes = CanonicalStructuralNodeSpace::derive_solid(&surface.mesh).unwrap();
        surface.rows = surface
            .projection
            .nodes
            .iter()
            .map(|node| {
                let index = nodes
                    .nodes
                    .iter()
                    .position(|n| n.identity == node.stable_identity)
                    .unwrap();
                [
                    StructuralDofKind::Ux,
                    StructuralDofKind::Uy,
                    StructuralDofKind::Uz,
                ]
                .map(|kind| summary.structural_dof_layout.index(index, kind).unwrap())
            })
            .collect();
        let master = surface
            .projection
            .nodes
            .iter()
            .position(|n| {
                (100..200).contains(&n.node_id)
                    && (n.coordinates_m.canonical_value()[2] + 0.0001).abs() < 1e-12
            })
            .unwrap();
        let free = if friction {
            vec![surface.rows[master][0], surface.rows[master][2]]
        } else {
            vec![surface.rows[master][2]]
        };
        let mut map = vec![None; summary.dof_count];
        for (n, rows) in surface.projection.nodes.iter().zip(&surface.rows) {
            if (100..200).contains(&n.node_id) {
                let q = (n.coordinates_m.canonical_value()[2] + 0.2001) / 0.2;
                if q > 1e-12 {
                    map[rows[2]] = Some((free.len() - 1, q));
                    if friction {
                        map[rows[0]] = Some((0, q));
                    }
                }
            }
        }
        for (j, &row) in free.iter().enumerate() {
            map[row] = Some((j, 1.));
        }
        let mut offsets = vec![0];
        let mut columns = Vec::new();
        let mut coefficients = Vec::new();
        for item in map {
            if let Some((column, value)) = item {
                columns.push(column);
                coefficients.push(value);
            }
            offsets.push(columns.len());
        }
        summary.constraint_matrix = None;
        summary.affine_constraint_reduction = Some(AffineConstraintReduction {
            full_dof_count: summary.dof_count,
            reduced_dof_count: free.len(),
            free_dof_indices: free.clone(),
            pivot_dof_indices: (0..summary.dof_count)
                .filter(|row| !free.contains(row))
                .collect(),
            recovery_row_offsets: offsets,
            recovery_column_indices: columns,
            recovery_coefficients: coefficients,
            offset: vec![0.; summary.dof_count],
        });
        summary.physical_load = vec![0.; summary.dof_count];
        summary.physical_load[*free.last().unwrap()] = 0.016;
        if friction {
            summary.physical_load[free[0]] = 0.01;
        }
        Self {
            surface,
            model,
            summary,
        }
    }
    fn operator(&self, friction: bool) -> CanonicalContactOperator<'_> {
        CanonicalContactOperator::prepare(
            &self.surface.projection,
            &self.surface.interfaces,
            &self.surface.rows,
            self.summary.dof_count,
            &[ContactOperatorTestFixture::control(friction)],
            ContactOperatorTestFixture::budget(),
        )
        .unwrap()
    }
    fn coupling<'a>(
        operator: &'a CanonicalContactOperator<'a>,
    ) -> super::super::contact::SolidContactCoupling<'a> {
        super::super::contact::SolidContactCoupling {
            operator,
            solve_policy: GeneralSolvePolicy::bounded(policy().linear_relative_tolerance),
            maximum_additional_resident_bytes: 100_000_000,
            maximum_geometry_trials_per_advance: 10000,
            maximum_symbolic_scalar_operations: 100_000_000,
            solve_admission: GeneralSolveEnvelope {
                resident_bytes: 100_000_000,
                scalar_operations: 100_000_000,
            },
        }
    }
}
fn near(actual: f64, expected: f64, tolerance: f64) {
    assert!(
        (actual - expected).abs() < tolerance,
        "actual={actual:e} expected={expected:e}"
    );
}

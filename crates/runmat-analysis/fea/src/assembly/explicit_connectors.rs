use runmat_analysis_core::{AnalysisConnector, ConnectorKind, StructuralModel};

use super::{
    dofs::{StructuralDofKind, StructuralDofLayout},
    structural_target_nodes, StructuralAssemblyError,
};
use crate::algebra::{
    ContributionDomain, ContributionId, ContributionProvenance, RealSparseAssembler,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct ExplicitConnectorSummary {
    pub stiffness_count: usize,
    pub damping_count: usize,
}

pub(super) fn assemble_explicit_connectors(
    connectors: &[AnalysisConnector],
    structural: &StructuralModel,
    layout: &StructuralDofLayout,
    stiffness: &mut RealSparseAssembler,
    damping: &mut RealSparseAssembler,
) -> Result<ExplicitConnectorSummary, StructuralAssemblyError> {
    let mut stiffness_count = 0;
    let mut damping_count = 0;
    for connector in connectors {
        let primary = resolved_nodes(connector, structural, true)?;
        let secondary = connector
            .secondary
            .as_ref()
            .map(|_| resolved_nodes(connector, structural, false))
            .transpose()?;
        match connector.kind {
            ConnectorKind::TranslationalSpring { stiffness: value } => {
                add_generalized_translation_operator(
                    stiffness,
                    layout,
                    &primary,
                    secondary.as_deref(),
                    value.newtons_per_meter,
                );
                stiffness.add_contribution(provenance(connector));
                stiffness_count += 1;
            }
            ConnectorKind::TranslationalDashpot { damping: value } => {
                add_generalized_translation_operator(
                    damping,
                    layout,
                    &primary,
                    secondary.as_deref(),
                    value.newton_seconds_per_meter,
                );
                damping.add_contribution(provenance(connector));
                damping_count += 1;
            }
            ConnectorKind::ElasticFoundation { stiffness: value } => {
                if connector.secondary.is_some() {
                    return Err(StructuralAssemblyError::InvalidExplicitConnector {
                        connector_id: connector.connector_id.clone(),
                        reason: "an elastic foundation is attached to ground and cannot have a secondary target",
                    });
                }
                add_distributed_foundation(stiffness, layout, &primary, value.newtons_per_meter);
                stiffness.add_contribution(provenance(connector));
                stiffness_count += 1;
            }
        }
    }
    Ok(ExplicitConnectorSummary {
        stiffness_count,
        damping_count,
    })
}

fn resolved_nodes(
    connector: &AnalysisConnector,
    structural: &StructuralModel,
    primary: bool,
) -> Result<Vec<usize>, StructuralAssemblyError> {
    let target = if primary {
        &connector.primary
    } else {
        connector.secondary.as_ref().expect("secondary requested")
    };
    let nodes = structural_target_nodes(structural, &target.region_id);
    if nodes.is_empty() {
        return Err(StructuralAssemblyError::UnmappedExplicitConnector {
            connector_id: connector.connector_id.clone(),
            target: if primary { "primary" } else { "secondary" },
            region_id: target.region_id.clone(),
        });
    }
    Ok(nodes)
}

fn add_generalized_translation_operator(
    assembler: &mut RealSparseAssembler,
    layout: &StructuralDofLayout,
    primary: &[usize],
    secondary: Option<&[usize]>,
    coefficient: f64,
) {
    for dof in [
        StructuralDofKind::Ux,
        StructuralDofKind::Uy,
        StructuralDofKind::Uz,
    ] {
        let mut terms = generalized_terms(layout, primary, dof, 1.0);
        if let Some(secondary) = secondary {
            terms.extend(generalized_terms(layout, secondary, dof, -1.0));
        }
        for &(row, row_weight) in &terms {
            for &(column, column_weight) in &terms {
                assembler.add(row, column, coefficient * row_weight * column_weight);
            }
        }
    }
}

fn generalized_terms(
    layout: &StructuralDofLayout,
    nodes: &[usize],
    dof: StructuralDofKind,
    sign: f64,
) -> Vec<(usize, f64)> {
    let weight = sign / nodes.len() as f64;
    nodes
        .iter()
        .map(|&node| {
            (
                layout
                    .index(node, dof)
                    .expect("explicit structural nodes own translational DOFs"),
                weight,
            )
        })
        .collect()
}

fn add_distributed_foundation(
    assembler: &mut RealSparseAssembler,
    layout: &StructuralDofLayout,
    nodes: &[usize],
    stiffness: f64,
) {
    let nodal_stiffness = stiffness / nodes.len() as f64;
    for &node in nodes {
        for dof in [
            StructuralDofKind::Ux,
            StructuralDofKind::Uy,
            StructuralDofKind::Uz,
        ] {
            let row = layout
                .index(node, dof)
                .expect("explicit structural nodes own translational DOFs");
            assembler.add(row, row, nodal_stiffness);
        }
    }
}

fn provenance(connector: &AnalysisConnector) -> ContributionProvenance {
    ContributionProvenance {
        domain: ContributionDomain::Structural,
        contribution_id: ContributionId(format!("connector:{}", connector.connector_id)),
    }
}

#[cfg(test)]
mod tests {
    use runmat_analysis_core::{
        AnalysisConnector, AnalysisConstraint, ConnectorKind, PrescribedStructuralDofs,
        PrescribedValue, RegionTarget, Rotation, RotationDofValue, Stiffness,
        StructuralConstraintKind, StructuralRotationDof,
    };

    use crate::{
        assembly::{assemble_structural_system, StructuralFormulationRequirement},
        excitation::{realize_step_excitations, ExcitationSamplePoint},
        fixtures::{fixture_model, FixtureId},
    };

    #[test]
    fn connector_stiffness_and_constraints_jointly_eliminate_rigid_modes() {
        let mut model = fixture_model(FixtureId::StructuralBeamCantileverEndMomentReference);
        let constraint = match &mut model.constraints[0] {
            AnalysisConstraint::Structural(constraint) => constraint,
            _ => unreachable!(),
        };
        constraint.kind = StructuralConstraintKind::Prescribed(PrescribedStructuralDofs {
            target: RegionTarget::point("node:1"),
            translations: Vec::new(),
            rotations: [
                StructuralRotationDof::Rx,
                StructuralRotationDof::Ry,
                StructuralRotationDof::Rz,
            ]
            .into_iter()
            .map(|dof| RotationDofValue {
                dof,
                value: PrescribedValue::Constant(Rotation::new(0.0)),
            })
            .collect(),
        });

        let rotation_only = assemble(&model);
        assert!(rotation_only.operator.nullspace_evidence().is_none());

        model.connectors = vec![AnalysisConnector {
            connector_id: "root_translation_anchor".into(),
            primary: RegionTarget::point("node:1"),
            secondary: None,
            kind: ConnectorKind::TranslationalSpring {
                stiffness: Stiffness::new(4_000.0),
            },
        }];
        let anchored = assemble(&model);
        assert!(anchored.operator.nullspace_evidence().is_some());
        anchored
            .operator
            .prepare_linear()
            .expect("the combined physical anchoring must prove coercive");

        model.constraints.clear();
        let partly_anchored = assemble(&model);
        assert!(partly_anchored.operator.nullspace_evidence().is_none());
        partly_anchored
            .operator
            .prepare_linear()
            .expect_err("a translational spring alone must not conceal rotational mechanisms");
    }

    fn assemble(model: &runmat_analysis_core::AnalysisModel) -> super::super::AssemblySummary {
        let excitations = realize_step_excitations(
            model,
            &model.sequence.steps[0],
            ExcitationSamplePoint::Static,
        )
        .expect("fixture excitations");
        assemble_structural_system(
            model,
            &excitations,
            None,
            StructuralFormulationRequirement::PhysicalDiscretization,
        )
        .expect("valid explicit structural assembly")
    }
}

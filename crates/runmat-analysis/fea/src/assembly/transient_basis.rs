//! Borrowed immutable basis for uncoupled canonical-solid linear transient assembly.
//!
//! This is an execution-local view of a freshly assembled, admitted summary, not portable
//! authority. It retains no operator/mesh/profile copies. Mixed, explicit, synthetic,
//! connector and coupled assemblies deliberately keep the original full-assembly path.
//! The one-time property recompile authenticates its topology/material identity. The
//! constraint plan, node space and analysis projection are pure functions of the borrowed
//! model and mesh, so they are compiled and validated once here; every time sample lowers its
//! realized sources on them and still checks that the lowered constraints equal the initial
//! assembly's. Existing prescribed-motion lifting stays in the transient pipeline. This
//! changes assembly work, not integration or physics gates.

use super::*;
use crate::excitation::{realize_step_excitations, ExcitationSamplePoint};
use runmat_analysis_core::ProcedureStep;

pub(super) mod forcing;
#[cfg(test)]
pub(super) mod tests;

pub(crate) struct PreparedStructuralTransientBasis<'a> {
    model: &'a AnalysisModel,
    step: &'a ProcedureStep,
    mesh: &'a SolverMeshArtifact,
    initial: &'a AssemblySummary,
    /// Compiled and validated once: a pure function of the borrowed model and mesh.
    constraint_plan: CanonicalStructuralConstraintPlan,
    /// Node space and analysis projection of the borrowed mesh, derived once.
    domain: forcing::SolidLoadDomain,
}

impl<'a> PreparedStructuralTransientBasis<'a> {
    /// Only call with the initial assembly admitted for these immutable inputs. Neither this
    /// view nor its summary can be rebound to a different authored model or topology.
    pub(crate) fn from_admitted(
        model: &'a AnalysisModel,
        step: &'a ProcedureStep,
        mesh: Option<&'a SolverMeshArtifact>,
        initial: &'a AssemblySummary,
    ) -> Result<Option<Self>, LinearAssemblyError> {
        let Some(mesh) = mesh else {
            return Ok(None);
        };
        if model.thermo_mechanical.is_some()
            || model.electro_thermal.is_some()
            || model
                .structural
                .as_ref()
                .is_some_and(|s| !s.elements.is_empty())
            || !model.connectors.is_empty()
            || initial.thermo_mechanical.is_some()
            || initial.electro_thermal.is_some()
            || initial.structural_solid_element_count == 0
        {
            return Ok(None);
        }
        if !model.sequence.steps.contains(step) {
            return Err(invalid(
                "prepared transient step is not authored by its model".into(),
            ));
        }
        // Reauthenticate geometry, topology, material ownership and constitutive semantics once.
        // Drop the temporary compiled inventory; retain only borrowed immutable authority.
        let properties = solid_properties::compile_for_assembly(
            model,
            Some(mesh),
            Some(StructuralFormulationRequirement::PhysicalDiscretization),
            false,
        )?
        .ok_or_else(|| invalid("prepared transient solid properties missing".into()))?;
        if Some(&properties.identity) != initial.structural_property_map_identity.as_ref() {
            return Err(invalid(
                "prepared transient properties or mesh differ from initial assembly".into(),
            ));
        }
        let constraint_plan = CanonicalStructuralConstraintPlan::compile(model, mesh)
            .map_err(LinearAssemblyError::ConstraintPlan)?;
        constraint_plan
            .validate_against(model, mesh)
            .map_err(LinearAssemblyError::ConstraintPlan)?;
        Ok(Some(Self {
            model,
            step,
            mesh,
            initial,
            constraint_plan,
            domain: forcing::SolidLoadDomain::derive(mesh)?,
        }))
    }

    pub(crate) fn sample(
        &self,
        point: ExcitationSamplePoint,
    ) -> Result<AssemblySummary, LinearAssemblyError> {
        let excitations = realize_step_excitations(self.model, self.step, point)
            .map_err(|error| invalid(error.to_string()))?;
        validate_structural_inputs(
            self.model,
            &excitations,
            StructuralProjectionBasis::CanonicalSolid,
        )
        .map_err(LinearAssemblyError::Structural)?;
        // Lower the sampled sources with the original lowering on the constraint plan and
        // spatial domain compiled once for these immutable inputs. This does not compile
        // properties, recover elements, or assemble any K/M/C contributions.
        let forcing = forcing::lower(
            &excitations,
            self.mesh,
            &self.constraint_plan,
            &self.initial.structural_dof_layout,
            &self.domain,
        )?;
        if forcing.constrained != self.initial.operator.constrained_dofs()
            || forcing.constraint_matrix != self.initial.constraint_matrix
            || forcing.affine_constraint_reduction != self.initial.affine_constraint_reduction
        {
            return Err(invalid("prepared transient constraints changed".into()));
        }
        let operator = self
            .initial
            .operator
            .with_load(forcing.rhs)
            .map_err(|error| LinearAssemblyError::SparseOperator(error.to_string()))?;
        let moment_count = excitations
            .iter()
            .filter(|load| matches!(load.kind, RealizedExcitationKind::Moment(_)))
            .count();
        // Explicit metadata construction avoids cloning an entire summary and then replacing
        // its operator, which would transiently retain a third full operator allocation.
        Ok(AssemblySummary {
            dof_count: self.initial.dof_count,
            structural_node_count: self.initial.structural_node_count,
            structural_translational_dof_count: self.initial.structural_translational_dof_count,
            structural_rotational_dof_count: self.initial.structural_rotational_dof_count,
            structural_rotation_node_count: self.initial.structural_rotation_node_count,
            structural_explicit_node_indices: self.initial.structural_explicit_node_indices.clone(),
            structural_load_plan_identity: forcing.plan_identity,
            structural_moment_load_count: moment_count,
            structural_direct_rotational_moment_load_count: if self
                .initial
                .structural_dof_layout
                .has_rotational_dofs()
            {
                moment_count
            } else {
                0
            },
            structural_requested_moment_norm_n_m: requested_moment_norm(&excitations),
            structural_moment_projection: forcing.moment_projection,
            spatial_projection_evidence: forcing.projection_evidence,
            structural_rotational_constraint_count: self
                .initial
                .structural_rotational_constraint_count,
            structural_beam_element_count: self.initial.structural_beam_element_count,
            structural_shell_element_count: self.initial.structural_shell_element_count,
            structural_connector_stiffness_count: self.initial.structural_connector_stiffness_count,
            structural_connector_damping_count: self.initial.structural_connector_damping_count,
            structural_connector_plan: self.initial.structural_connector_plan.clone(),
            structural_solid_element_count: self.initial.structural_solid_element_count,
            structural_solid_total_mass_kg: self.initial.structural_solid_total_mass_kg,
            structural_solid_recovery: self.initial.structural_solid_recovery.clone(),
            structural_dof_layout: self.initial.structural_dof_layout.clone(),
            structural_reduced_basis: self.initial.structural_reduced_basis.clone(),
            structural_beam_recovery: self.initial.structural_beam_recovery.clone(),
            structural_shell_recovery: self.initial.structural_shell_recovery.clone(),
            structural_formulations: self.initial.structural_formulations.clone(),
            structural_property_map_identity: self.initial.structural_property_map_identity.clone(),
            structural_sources: structural_source_implementations(
                &excitations,
                StructuralProjectionBasis::CanonicalSolid,
            ),
            structural_boundary_transfers: self.initial.structural_boundary_transfers.clone(),
            constrained_dof_count: self.initial.constrained_dof_count,
            constraint_matrix: forcing.constraint_matrix,
            affine_constraint_reduction: forcing.affine_constraint_reduction,
            physical_load: forcing.physical_load,
            load_count: excitations.len(),
            structural_material: self.initial.structural_material,
            thermo_mechanical: self.initial.thermo_mechanical.clone(),
            electro_thermal: self.initial.electro_thermal.clone(),
            operator,
        })
    }
}

fn invalid(message: String) -> LinearAssemblyError {
    LinearAssemblyError::Structural(StructuralAssemblyError::InvalidLoadPlan(message))
}

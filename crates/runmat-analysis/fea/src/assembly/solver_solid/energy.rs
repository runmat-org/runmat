//! Original local stiffness capture; it neither repeats quadrature nor constructs a solver.
use super::*;
use crate::{
    algebra::{anchored_energy::AnchoredEnergySource, LinearPreparationReuse},
    constraints::{AffineConstraintReduction, HomogeneousTranslationSelection},
};

#[derive(Default)]
pub(crate) struct SolidEnergyPreparation {
    capture: Option<(AnchoredEnergySource, HomogeneousTranslationSelection)>,
}

impl SolidEnergyPreparation {
    /// Elastic solid source only. Added connector/coupling operators need their
    /// own construction evidence; the ordinary shared proof still handles them.
    pub fn supports_model(model: &runmat_analysis_core::AnalysisModel) -> bool {
        model.connectors.is_empty()
            && model.thermo_mechanical.is_none()
            && model.electro_thermal.is_none()
            && model
                .structural
                .as_ref()
                .is_none_or(|domain| domain.elements.is_empty())
    }

    /// Includes original blocks, selection construction and retained-map headers.
    /// This demand is additional to ordinary physical assembly/operator storage.
    pub fn resident_bytes(mesh: &SolverMeshArtifact) -> Result<u64, SolverSolidAssemblyError> {
        let source = AnchoredEnergySource::resident_bytes_for_widths(
            mesh.topology
                .volume_elements
                .iter()
                .map(|element| element.node_ids.len()),
        )
        .map_err(energy_error)?;
        source
            .checked_add(
                (mesh.topology.nodes.len() as u64)
                    .checked_mul(128)
                    .ok_or_else(|| energy_error(crate::algebra::EnergyError::Arithmetic))?,
            )
            .and_then(|bytes| bytes.checked_add(512))
            .ok_or_else(|| energy_error(crate::algebra::EnergyError::Arithmetic))
    }

    pub(in crate::assembly) fn begin(
        &mut self,
        mesh: &SolverMeshArtifact,
        fixed: &[bool],
        reduction: Option<&AffineConstraintReduction>,
    ) -> Result<(), SolverSolidAssemblyError> {
        if crate::progress::is_cancelled() {
            return Err(SolverSolidAssemblyError::Cancelled);
        }
        // Derive the exact inventory before allocating the source payload.
        let bytes = Self::resident_bytes(mesh)?;
        let selection = match reduction {
            Some(reduction) => HomogeneousTranslationSelection::from_reduction(
                reduction,
                mesh.topology.nodes.len(),
            )
            .map_err(|e| SolverSolidAssemblyError::InvalidMaterialCoverage(e.to_string()))?,
            None => {
                HomogeneousTranslationSelection::from_fixed_rows(fixed, mesh.topology.nodes.len())
            }
        };
        self.capture = selection
            .map(|selection| {
                AnchoredEnergySource::new(
                    mesh.topology
                        .volume_elements
                        .iter()
                        .map(|element| element.node_ids.len()),
                    bytes,
                )
                .map(|source| (source, selection))
            })
            .transpose()
            .map_err(energy_error)?;
        Ok(())
    }

    pub(super) fn capture<const N: usize, const D: usize>(
        &mut self,
        offsets: &[usize; N],
        stiffness: &[[f64; D]; D],
    ) -> Result<(), SolverSolidAssemblyError> {
        if let Some((source, selection)) = self.capture.as_mut() {
            let nodes = offsets.map(|offset| offset / 3);
            let values = source.capture_next(&nodes).map_err(energy_error)?;
            let axis = selection.axis();
            for i in 0..N {
                for j in 0..N {
                    values[i * N + j] = stiffness[3 * i + axis][3 * j + axis];
                }
            }
        }
        Ok(())
    }

    pub fn into_reuse(self) -> LinearPreparationReuse {
        match self.capture {
            Some((source, selection)) => {
                LinearPreparationReuse::with_selected_energy(source, selection.into_rows())
            }
            None => LinearPreparationReuse::default(),
        }
    }
}

fn energy_error(error: crate::algebra::EnergyError) -> SolverSolidAssemblyError {
    match error {
        crate::algebra::EnergyError::Cancelled => SolverSolidAssemblyError::Cancelled,
        _ => SolverSolidAssemblyError::SparseOperator(error.to_string()),
    }
}

pub(in crate::assembly) fn assemble_preparing(
    mesh: &SolverMeshArtifact,
    properties: &CompiledPropertyMap,
    execution: LocalExecutionPlan,
    mut capture: Option<&mut SolidEnergyPreparation>,
    constraints: (&[bool], Option<&AffineConstraintReduction>),
) -> Result<SolverSolidOperators, SolverSolidAssemblyError> {
    if let Some(capture) = capture.as_deref_mut() {
        capture.begin(mesh, constraints.0, constraints.1)?;
    }
    #[cfg(test)]
    crate::assembly::transient_basis::tests::record_operator_assembly();
    assemble_solver_solid_operators_capturing(mesh, properties, execution, capture)
}

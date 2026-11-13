//! Original local stiffness capture; it neither repeats quadrature nor constructs a solver.
use super::*;
use crate::{
    algebra::{anchored_energy::AnchoredEnergySource, LinearPreparationReuse},
    constraints::{AffineConstraintReduction, HomogeneousTranslationSelection},
};

#[derive(Default)]
pub(crate) struct SolidEnergyPreparation {
    capture: Option<(AnchoredEnergySource, HomogeneousTranslationSelection)>,
    /// Elastic rigidity mode: every element block is checked as it is assembled.
    elastic: Option<ElasticCapture>,
}

enum ElasticCapture {
    Requested,
    Capturing(crate::algebra::elastic_rigidity::ElasticRigidityCapture),
    Refused(crate::algebra::ElasticRigidityError),
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

    /// Proves the elastic operator by element rigidity, gluing and anchors instead of the scalar
    /// selected energy (the large static path that never forms a global factor).
    pub fn elastic() -> Self {
        Self {
            capture: None,
            elastic: Some(ElasticCapture::Requested),
        }
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
        if self.elastic.is_some() {
            let fixed = match reduction {
                // Only a pure selection (each DOF fixed or kept with coefficient one) admits the
                // elastic proof; any other affine relation is refused with the capture.
                Some(reduction) => selection_fixed_rows(reduction),
                None => Some(fixed.to_vec()),
            };
            self.elastic = Some(match fixed {
                None => ElasticCapture::Refused(crate::algebra::ElasticRigidityError::Shape),
                Some(fixed) => {
                    let index = mesh
                        .topology
                        .nodes
                        .iter()
                        .enumerate()
                        .map(|(index, node)| (node.node_id, index as u32))
                        .collect::<std::collections::HashMap<_, _>>();
                    let elements = mesh
                        .topology
                        .volume_elements
                        .iter()
                        .map(|element| {
                            element
                                .node_ids
                                .iter()
                                .map(|id| index.get(id).copied().ok_or(()))
                                .collect::<Result<Vec<_>, _>>()
                                .map(|nodes| (element.element_id, nodes))
                        })
                        .collect::<Result<Vec<_>, _>>();
                    match elements {
                        Err(()) => {
                            ElasticCapture::Refused(crate::algebra::ElasticRigidityError::Shape)
                        }
                        Ok(elements) => {
                            match crate::algebra::elastic_rigidity::ElasticRigidityCapture::new(
                                mesh.topology.nodes.iter().map(|node| {
                                    (node.node_id, node.coordinates_m.canonical_value())
                                }),
                                elements.into_iter(),
                                fixed,
                            ) {
                                Ok(capture) => ElasticCapture::Capturing(capture),
                                Err(error) => ElasticCapture::Refused(error),
                            }
                        }
                    }
                }
            });
            return Ok(());
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

    /// `dofs` are the element's global DOFs (three per node, node-major); `stiffness` is the
    /// row-major element matrix over those DOFs.
    pub(super) fn capture(
        &mut self,
        dofs: &[u32],
        stiffness: &[f64],
    ) -> Result<(), SolverSolidAssemblyError> {
        if let Some(ElasticCapture::Capturing(capture)) = self.elastic.as_mut() {
            if let Err(error) = capture.capture(dofs, stiffness) {
                self.elastic = Some(ElasticCapture::Refused(error));
            }
            return Ok(());
        }
        if let Some((source, selection)) = self.capture.as_mut() {
            let n = dofs.len() / 3;
            let d = dofs.len();
            let nodes = dofs
                .iter()
                .step_by(3)
                .map(|&dof| dof as usize / 3)
                .collect::<Vec<_>>();
            let values = source.capture_next(&nodes).map_err(energy_error)?;
            let axis = selection.axis();
            for i in 0..n {
                for j in 0..n {
                    values[i * n + j] = stiffness[(3 * i + axis) * d + 3 * j + axis];
                }
            }
        }
        Ok(())
    }

    /// The completed elastic proof, or the typed reason the operator was refused.
    pub fn into_elastic_reuse(
        self,
    ) -> Result<LinearPreparationReuse, crate::algebra::ElasticRigidityError> {
        match self.elastic {
            Some(ElasticCapture::Capturing(capture)) => capture
                .finish()
                .map(LinearPreparationReuse::with_elastic_rigidity),
            Some(ElasticCapture::Refused(error)) => Err(error),
            Some(ElasticCapture::Requested) | None => {
                Err(crate::algebra::ElasticRigidityError::Shape)
            }
        }
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

/// Full-DOF fixed flags of a reduction that only selects DOFs (each recovery row empty, or a
/// single unit coefficient). Prescribed offsets on fixed rows are allowed.
fn selection_fixed_rows(reduction: &AffineConstraintReduction) -> Option<Vec<bool>> {
    (0..reduction.full_dof_count)
        .map(|full| {
            let range =
                reduction.recovery_row_offsets[full]..reduction.recovery_row_offsets[full + 1];
            match range.len() {
                0 => Some(true),
                1 if reduction.recovery_coefficients[range.start] == 1.0
                    && reduction.offset[full] == 0.0 =>
                {
                    Some(false)
                }
                _ => None,
            }
        })
        .collect()
}

fn energy_error(error: crate::algebra::EnergyError) -> SolverSolidAssemblyError {
    match error {
        crate::algebra::EnergyError::Cancelled => SolverSolidAssemblyError::Cancelled,
        _ => SolverSolidAssemblyError::SparseOperator(error.to_string()),
    }
}

pub(in crate::assembly) async fn assemble_preparing_with_owner<O, F, A>(
    mesh: &SolverMeshArtifact,
    properties: &CompiledPropertyMap,
    mut execution: SolidAssemblyExecution<'_>,
    constraints: (&[bool], Option<&AffineConstraintReduction>),
    before_block: O,
) -> Result<SolverSolidOperators, A>
where
    O: FnMut(super::SolidCapacityPhase, runmat_simulation_usage::OperationCount) -> F,
    F: std::future::Future<Output = Result<(), A>>,
    A: From<SolverSolidAssemblyError> + Send,
{
    if let Some(capture) = execution.capture.as_deref_mut() {
        capture.begin(mesh, constraints.0, constraints.1)?;
    }
    #[cfg(test)]
    crate::assembly::transient_basis::tests::record_operator_assembly();
    assemble_solver_solid_operators_capturing_with_owner(mesh, properties, execution, before_block)
        .await
}

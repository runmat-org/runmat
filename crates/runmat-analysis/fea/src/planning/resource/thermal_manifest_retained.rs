//! Typed metadata retained by a manifest alongside its borrowed source result and catalog.
use super::{
    ByteCount, ResourcePlanningError, ThermalMetadataResourceFacts,
    ThermalSourceEvidenceResourceFacts,
};
use crate::{
    admission::AnalysisRunIdentity,
    diagnostics::FeaDiagnostic,
    execution::{ThermalSolveSchedule, ThermalSolverResultManifest},
    solve::accelerator::AcceleratorSolveEvidence,
    ThermalEnergyBalance,
};
use runmat_analysis_core::{AnalysisModel, ProcedureStep};
use runmat_meshing_core::SolverMeshArtifact;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ThermalManifestRetainedResourceFacts {
    pub inline_bytes: ByteCount,
    pub scalar_inventory_bytes: ByteCount,
    pub identity_heap_bytes: ByteCount,
    pub source_bytes: ByteCount,
    pub diagnostic_bytes: ByteCount,
    pub backend_bytes: ByteCount,
    pub initial_state_heap_bytes: ByteCount,
}

impl ThermalManifestRetainedResourceFacts {
    pub fn derive(
        model: &AnalysisModel,
        step: &ProcedureStep,
        mesh: &SolverMeshArtifact,
        schedule: &ThermalSolveSchedule,
        identity: &AnalysisRunIdentity,
        metadata: &ThermalMetadataResourceFacts,
        sources: &ThermalSourceEvidenceResourceFacts,
        interface_nodes: usize,
    ) -> Result<Self, ResourcePlanningError> {
        let scalar_inventory_bytes = sum(&[
            slots::<u64>(mesh.topology.nodes.len())?,
            slots::<u64>(interface_nodes)?,
            slots::<u64>(mesh.topology.volume_elements.len())?,
            slots::<f64>(schedule.sample_count())?,
            slots::<f64>(schedule.solve_count())?,
            slots::<ThermalEnergyBalance>(schedule.solve_count())?,
        ])?;
        let diagnostic_slots = sum(&[
            slots::<FeaDiagnostic>(crate::diagnostics::thermal::thermal_diagnostic_record_count())?,
            slots::<FeaDiagnostic>(crate::diagnostics::material_assignment::record_count(
                &model.material_assignments,
            ))?,
            slots::<FeaDiagnostic>(crate::solve::linear::successful_diagnostic_record_count())?,
        ])?;
        // Encoded byte bounds include every code/message/ID UTF8 byte. They are used only for
        // string payloads: typed record and String slots are explicitly sized separately.
        let diagnostic_bytes = sum(&[
            diagnostic_slots,
            metadata.physical_diagnostics.encoded_bytes_upper_bound,
            metadata.assignment_diagnostics.encoded_bytes_upper_bound,
            metadata.solver_diagnostics.encoded_bytes_upper_bound,
        ])?;
        let initial_state_heap_bytes = sum(&[
            slots::<String>(
                crate::physics::thermal::initial_state::selected_condition_count(model, step),
            )?,
            metadata.initial_state.encoded_bytes_upper_bound,
        ])?;
        let source_bytes = sources
            .record_bytes_per_sample
            .0
            .checked_mul(schedule.sample_count() as u64)
            .ok_or(ResourcePlanningError::Overflow(
                "thermal manifest source clone",
            ))?;
        Ok(Self {
            // Includes all Vec/String headers nested inline in run and initial-state records.
            inline_bytes: ByteCount(std::mem::size_of::<ThermalSolverResultManifest>() as u64),
            scalar_inventory_bytes: ByteCount(scalar_inventory_bytes),
            identity_heap_bytes: ByteCount(identity_heap(identity)?),
            source_bytes: ByteCount(source_bytes),
            diagnostic_bytes: ByteCount(diagnostic_bytes),
            // Successful Completed records contain only inline provider data, no loss strings.
            backend_bytes: ByteCount(slots::<AcceleratorSolveEvidence>(
                metadata.backend_evidence.maximum_records,
            )?),
            initial_state_heap_bytes: ByteCount(initial_state_heap_bytes),
        })
    }

    pub fn reservation_bytes(&self) -> Result<ByteCount, ResourcePlanningError> {
        sum(&[
            self.inline_bytes.0,
            self.scalar_inventory_bytes.0,
            self.identity_heap_bytes.0,
            self.source_bytes.0,
            self.diagnostic_bytes.0,
            self.backend_bytes.0,
            self.initial_state_heap_bytes.0,
        ])
        .map(ByteCount)
    }
}

fn slots<T>(count: usize) -> Result<u64, ResourcePlanningError> {
    count
        .checked_mul(std::mem::size_of::<T>())
        .and_then(|bytes| u64::try_from(bytes).ok())
        .ok_or(ResourcePlanningError::Overflow(
            "thermal manifest typed slots",
        ))
}
fn sum(parts: &[u64]) -> Result<u64, ResourcePlanningError> {
    parts.iter().try_fold(0_u64, |sum, part| {
        sum.checked_add(*part)
            .ok_or(ResourcePlanningError::Overflow(
                "thermal manifest retained metadata",
            ))
    })
}

fn identity_heap(identity: &AnalysisRunIdentity) -> Result<u64, ResourcePlanningError> {
    use runmat_analysis_core::{
        BoundaryTransferImplementation, ExactFormulationComponent, ExactFormulationRealization,
        SourceImplementation, TemporalImplementation, TopologyImplementation,
    };
    let ExactFormulationRealization {
        components,
        temporals,
        coupling: _,
        boundary_transfers,
    } = identity.realization();
    let mut bytes = sum(&[
        slots::<ExactFormulationComponent>(components.len())?,
        slots::<TemporalImplementation>(temporals.len())?,
        slots::<BoundaryTransferImplementation>(boundary_transfers.len())?,
    ])?;
    for ExactFormulationComponent {
        spatial_operator: _,
        constitutive: _,
        recovery: _,
        topologies,
        sources,
    } in components
    {
        bytes = sum(&[
            bytes,
            slots::<TopologyImplementation>(topologies.len())?,
            slots::<SourceImplementation>(sources.len())?,
        ])?;
    }
    Ok(bytes)
}

#[cfg(test)]
mod tests;

//! Original producing host closure bound to each typed numerical receipt.
use super::*;

pub(super) fn receipt_matches_solve(
    receipt: &AccuracySourceReceipt,
    host: &AnalysisHostWorkload,
) -> bool {
    match receipt {
        AccuracySourceReceipt::NonlinearSolid(receipt) => {
            let r = receipt.receipt();
            r.producer == host.workload.step
                && r.complete
                && host
                    .workload
                    .inputs
                    .iter()
                    .filter(|i| {
                        i.kind == AnalysisNumericalObjectKind::NonlinearSolidWorkload
                            && i.logical_identity == receipt.source_workload().digest()
                    })
                    .count()
                    == 1
                && host
                    .workload
                    .inputs
                    .iter()
                    .filter(|i| {
                        i.kind == AnalysisNumericalObjectKind::NonlinearSolidBlockRequest
                            && i.logical_identity == r.request
                    })
                    .count()
                    == 1
                && host
                    .workload
                    .inputs
                    .iter()
                    .filter(|i| {
                        i.kind == AnalysisNumericalObjectKind::Mesh
                            && i.logical_identity == r.topology
                    })
                    .count()
                    == 1
        }
        AccuracySourceReceipt::FluidSteady(fluid) => {
            let mut workloads = host
                .workload
                .inputs
                .iter()
                .filter(|input| input.kind == AnalysisNumericalObjectKind::FluidSteadyWorkload);
            let mut meshes = host
                .workload
                .inputs
                .iter()
                .filter(|input| input.kind == AnalysisNumericalObjectKind::Mesh);
            matches!(workloads.next(),Some(input) if input.logical_identity==fluid.source_workload().digest())
                && workloads.next().is_none()
                && matches!(meshes.next(),Some(input) if input.logical_identity==fluid.receipt().mesh_digest)
                && meshes.next().is_none()
                && fluid.receipt().producer == host.workload.step
        }
        AccuracySourceReceipt::MaxwellHarmonic(maxwell) => {
            let mut workloads =
                host.workload.inputs.iter().filter(|input| {
                    input.kind == AnalysisNumericalObjectKind::MaxwellHarmonicWorkload
                });
            let mut meshes = host
                .workload
                .inputs
                .iter()
                .filter(|input| input.kind == AnalysisNumericalObjectKind::Mesh);
            matches!(workloads.next(), Some(input) if input.logical_identity == maxwell.source_workload().digest())
                && workloads.next().is_none()
                && matches!(meshes.next(), Some(input) if input.logical_identity == maxwell.receipt().topology_digest)
                && meshes.next().is_none()
                && maxwell.receipt().producer == host.workload.step
        }
        AccuracySourceReceipt::Magnetostatic(magnetic) => {
            let mut workloads =
                host.workload.inputs.iter().filter(|input| {
                    input.kind == AnalysisNumericalObjectKind::MagnetostaticWorkload
                });
            let mut meshes = host
                .workload
                .inputs
                .iter()
                .filter(|input| input.kind == AnalysisNumericalObjectKind::Mesh);
            matches!(workloads.next(), Some(input) if input.logical_identity == magnetic.source_workload().digest())
                && workloads.next().is_none()
                && matches!(meshes.next(), Some(input) if input.logical_identity == magnetic.receipt().topology_digest)
                && meshes.next().is_none()
                && magnetic.receipt().producer == host.workload.step
        }
        AccuracySourceReceipt::AcousticResponse(response) => {
            let mut workloads = host.workload.inputs.iter().filter(|input| {
                input.kind == AnalysisNumericalObjectKind::AcousticResponseWorkload
            });
            let mut meshes = host
                .workload
                .inputs
                .iter()
                .filter(|input| input.kind == AnalysisNumericalObjectKind::Mesh);
            matches!(workloads.next(), Some(input) if input.logical_identity == response.source_workload().digest())
                && workloads.next().is_none()
                && matches!(meshes.next(), Some(input) if input.logical_identity == response.receipt().topology_digest)
                && meshes.next().is_none()
                && response.receipt().producer == host.workload.step
        }
        AccuracySourceReceipt::ElectricalStatic(electrical) => {
            let mut workloads = host.workload.inputs.iter().filter(|input| {
                input.kind == AnalysisNumericalObjectKind::ElectricalStaticWorkload
            });
            let mut meshes = host
                .workload
                .inputs
                .iter()
                .filter(|input| input.kind == AnalysisNumericalObjectKind::Mesh);
            matches!(workloads.next(), Some(input) if input.logical_identity == electrical.source_workload().digest())
                && workloads.next().is_none()
                && matches!(meshes.next(), Some(input) if input.logical_identity == electrical.receipt().topology_digest)
                && meshes.next().is_none()
                && electrical.receipt().producer == host.workload.step
        }
        AccuracySourceReceipt::AcousticHarmonic(acoustic) => {
            let mut workloads = host.workload.inputs.iter().filter(|input| {
                input.kind == AnalysisNumericalObjectKind::AcousticHarmonicWorkload
            });
            let mut meshes = host
                .workload
                .inputs
                .iter()
                .filter(|input| input.kind == AnalysisNumericalObjectKind::Mesh);
            matches!(workloads.next(), Some(input) if input.logical_identity == acoustic.source_workload().digest())
                && workloads.next().is_none()
                && matches!(meshes.next(), Some(input) if input.logical_identity == acoustic.receipt().topology_digest)
                && meshes.next().is_none()
                && acoustic.receipt().producer == host.workload.step
        }
        AccuracySourceReceipt::Thermal(thermal) => {
            let inputs_of_kind = |kind| {
                host.workload
                    .inputs
                    .iter()
                    .filter(move |input| input.kind == kind)
            };
            let mut workloads = inputs_of_kind(AnalysisNumericalObjectKind::ThermalWorkload);
            let mut meshes = inputs_of_kind(AnalysisNumericalObjectKind::Mesh);
            // Thermal receipts bind numerical intent, not the transport/placement host identity.
            matches!(workloads.next(), Some(input) if input.logical_identity == thermal.source_workload.digest())
                && workloads.next().is_none()
                && matches!(meshes.next(), Some(input) if input.logical_identity == thermal.receipt().topology_digest)
                && meshes.next().is_none()
                && thermal.receipt().producer == host.workload.step
        }
        AccuracySourceReceipt::StructuralLinear(_)
        | AccuracySourceReceipt::StructuralModalResponseSpectrum(_)
        | AccuracySourceReceipt::StructuralRandomResponse(_) => {
            receipt.source_workload() == host.workload.identity
        }
    }
}

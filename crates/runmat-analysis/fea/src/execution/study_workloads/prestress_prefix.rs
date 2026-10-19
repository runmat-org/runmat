//! Shared producer-prefix selection and exact thermal-to-preload lineage validation.
use super::*;
use runmat_analysis_core::{StateKind, StateSource, StateUse};

pub(super) fn validate_consumer_chain(
    mechanical: &[&StructuralStepWorkload],
) -> AnalysisExecutionResult<()> {
    let supported = match mechanical {
        [StructuralStepWorkload::ModalExtraction { .. }, StructuralStepWorkload::ModalTransient { .. }]
        | [StructuralStepWorkload::ModalExtraction { .. }, StructuralStepWorkload::ModalFrequency { .. }]
        | [StructuralStepWorkload::ModalExtraction { .. }, StructuralStepWorkload::ModalResponseSpectrum { .. }]
        | [StructuralStepWorkload::ModalExtraction { .. }, StructuralStepWorkload::RandomResponse { .. }]
        | [StructuralStepWorkload::Linear { .. }, StructuralStepWorkload::Buckling { .. }]
        | [StructuralStepWorkload::Linear { .. }, StructuralStepWorkload::PrestressedModal { .. }] => {
            true
        }
        [StructuralStepWorkload::Linear { .. }, StructuralStepWorkload::PrestressedFrequency { template }] =>
        {
            matches!(
                &template.solver,
                StructuralPrestressedFrequencySolverTemplate::Direct { .. }
            )
        }
        [StructuralStepWorkload::Linear { .. }, StructuralStepWorkload::PrestressedModal { .. }, StructuralStepWorkload::PrestressedFrequency { template }] =>
        {
            matches!(
                &template.solver,
                StructuralPrestressedFrequencySolverTemplate::ModalReduced
            )
        }
        _ => false,
    };
    if !supported {
        return Err(invalid(
            "heterogeneous structural execution does not support this authored procedure chain",
        ));
    }
    Ok(())
}

/// Validated shared producer prefix and its ordered prestress consumers.
pub struct StructuralPrestressPrefix<'a> {
    pub thermal: Option<&'a ThermalWorkload>,
    pub preload: &'a StructuralLinearWorkload,
    pub consumer_step_ids: Vec<ProcedureStepId>,
}

impl StructuralStudyWorkloads {
    pub fn prestress_prefix(&self) -> AnalysisExecutionResult<StructuralPrestressPrefix<'_>> {
        self.validate()?;
        let Self::HeterogeneousSequence { workloads } = self else {
            return Err(invalid(
                "prestress prefix requires a heterogeneous sequence",
            ));
        };
        let order = self.step_ids()?;
        let (thermal, offset) = match &workloads[&order[0]] {
            StructuralStepWorkload::Thermal { workload } => (Some(workload.as_ref()), 1),
            _ => (None, 0),
        };
        let StructuralStepWorkload::Linear { workload } = &workloads[&order[offset]] else {
            return Err(invalid("prestress prefix requires a static preload"));
        };
        Ok(StructuralPrestressPrefix {
            thermal,
            preload: workload,
            consumer_step_ids: order[offset + 1..].to_vec(),
        })
    }
}

pub(super) fn mechanical_suffix<'a, 'b>(
    model: &AnalysisModel,
    ordered: &'a [&'b StructuralStepWorkload],
) -> AnalysisExecutionResult<&'a [&'b StructuralStepWorkload]> {
    match ordered {
        [StructuralStepWorkload::Thermal { workload: thermal }, StructuralStepWorkload::Linear { workload: preload }, rest @ ..] =>
        {
            let [input] = preload.procedure_step()?.state_inputs.as_slice() else {
                return Err(invalid(
                    "thermal prestress prefix requires exactly one thermal state input",
                ));
            };
            if input.required_kind != StateKind::Thermal
                || input.use_as != StateUse::Prestress
                || !matches!(&input.source, StateSource::Upstream { producer_step, .. } if producer_step == &thermal.step_id)
                || model.sequence.transitions.iter().any(|transition| {
                    transition.producer_step == thermal.step_id
                        && transition.consumer_step != preload.step_id
                })
                || model.sequence.dependencies.iter().any(|dependency| {
                    dependency.predecessor == thermal.step_id
                        && dependency.successor != preload.step_id
                })
                || !model.sequence.branches.is_empty()
                || rest.is_empty()
            {
                return Err(invalid(
                    "thermal producer must feed the exact thermomechanical prestress preload",
                ));
            }
            Ok(&ordered[1..])
        }
        _ => Ok(ordered),
    }
}

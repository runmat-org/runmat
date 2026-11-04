use runmat_analysis_core::AnalysisField;

use super::{
    mesh_result_figures_from_fields, render_topology_for_persisted_run,
    AnalysisFigureGenerationOptions, AnalysisGeneratedFigure,
};
use crate::analysis::{
    analysis_field_page_by_run_id_op, AnalysisFieldDescriptor, AnalysisFieldPageResult,
    AnalysisFieldPageValues, AnalysisFieldRequestOptions, AnalysisFieldSelection,
    AnalysisStudySpec,
};
use crate::operations::OperationContext;

pub(super) fn generate(
    study: &AnalysisStudySpec,
    run_id: &str,
    selection: &AnalysisFieldSelection,
    options: AnalysisFigureGenerationOptions,
) -> Result<Vec<AnalysisGeneratedFigure>, String> {
    let materialized = materialize_visualization_field(run_id, selection)?;
    let render_topology = render_topology_for_persisted_run(study, run_id)?;
    let mut figures = mesh_result_figures_from_fields(
        &study.geometry,
        Some(&study.model),
        render_topology.as_ref(),
        std::slice::from_ref(&materialized.field),
        run_id,
        options,
    );
    if materialized.complex_magnitude {
        for figure in &mut figures {
            figure.warnings.push(
                "Complex frequency-response phasors are rendered as component magnitudes; the persisted result field retains exact real and imaginary values."
                    .to_string(),
            );
        }
    }
    Ok(figures)
}

struct MaterializedVisualizationField {
    field: AnalysisField,
    complex_magnitude: bool,
}

fn materialize_visualization_field(
    run_id: &str,
    selection: &AnalysisFieldSelection,
) -> Result<MaterializedVisualizationField, String> {
    let mut offset = 0usize;
    let mut descriptor: Option<AnalysisFieldDescriptor> = None;
    let mut values = Vec::new();
    let mut complex_magnitude: Option<bool> = None;
    loop {
        let page = load_page(run_id, selection, offset)?;
        if page.offset != offset || page.count == 0 && page.total_count != offset {
            return Err("persisted FEA field paging did not make exact forward progress".into());
        }
        if let Some(expected) = descriptor.as_ref() {
            if expected != &page.descriptor || expected.value_count != page.total_count {
                return Err("persisted FEA field descriptor changed between pages".into());
            }
        } else {
            descriptor = Some(page.descriptor.clone());
        }

        match page.values {
            AnalysisFieldPageValues::Real(page_values) => {
                ensure_representation(&mut complex_magnitude, false)?;
                if page_values.len() != page.count {
                    return Err("real FEA field page has an invalid logical value count".into());
                }
                values.extend(page_values);
            }
            AnalysisFieldPageValues::ComplexInterleaved(page_values) => {
                ensure_representation(&mut complex_magnitude, true)?;
                if page_values.len() != page.count.saturating_mul(2) {
                    return Err("complex FEA field page does not contain exact phasor pairs".into());
                }
                values.extend(
                    page_values
                        .chunks_exact(2)
                        .map(|pair| pair[0].hypot(pair[1])),
                );
            }
        }
        offset = offset
            .checked_add(page.count)
            .ok_or_else(|| "persisted FEA field page offset overflow".to_string())?;
        if offset == page.total_count {
            break;
        }
    }

    let descriptor =
        descriptor.ok_or_else(|| "persisted FEA field has no descriptor".to_string())?;
    if values.len() != descriptor.value_count {
        return Err("persisted FEA field visualization is incomplete".into());
    }
    // Complex pages materialize as magnitudes, so the field is the quantity's magnitude.
    let layout = if complex_magnitude == Some(true) {
        descriptor
            .layout
            .clone()
            .representation(runmat_analysis_core::ComplexRepresentation::Magnitude)
    } else {
        descriptor.layout.clone()
    };
    let field = AnalysisField::host_f64(descriptor.field_id, layout, descriptor.shape, values);
    Ok(MaterializedVisualizationField {
        field,
        complex_magnitude: complex_magnitude.unwrap_or(false),
    })
}

fn load_page(
    run_id: &str,
    selection: &AnalysisFieldSelection,
    offset: usize,
) -> Result<AnalysisFieldPageResult, String> {
    analysis_field_page_by_run_id_op(
        run_id,
        selection.clone(),
        AnalysisFieldRequestOptions {
            offset: Some(offset),
            limit: None,
        },
        OperationContext::new(None, None),
    )
    .map(|envelope| envelope.data)
    .map_err(|error| format!("{}: {}", error.error_code, error.message))
}

fn ensure_representation(state: &mut Option<bool>, complex: bool) -> Result<(), String> {
    match state {
        Some(expected) if *expected != complex => {
            Err("persisted FEA field numeric representation changed between pages".into())
        }
        Some(_) => Ok(()),
        None => {
            *state = Some(complex);
            Ok(())
        }
    }
}

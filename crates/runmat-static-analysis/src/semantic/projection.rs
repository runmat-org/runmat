use std::collections::BTreeMap;

use runmat_hir::{BindingId, ReferenceKind};
use runmat_mir::analysis::AnalysisStore;
use runmat_types::{ProgramFunctionId, ProgramSpan, RegionValueId};

use super::{
    SemanticBindingFacts, SemanticBindingRegion, SemanticDocumentFacts, SemanticFactObservation,
    SemanticFunctionFacts,
};

pub fn project_document_facts(
    lowering: &runmat_hir::LoweringResult,
    mir: &runmat_mir::MirAssembly,
    store: &AnalysisStore,
) -> SemanticDocumentFacts {
    let mut references: BTreeMap<BindingId, Vec<ProgramSpan>> = BTreeMap::new();
    for reference in &lowering.hir_index.references {
        let ReferenceKind::Binding(binding) = &reference.kind else {
            continue;
        };
        references
            .entry(*binding)
            .or_default()
            .push(portable_span(reference.span));
    }
    for spans in references.values_mut() {
        spans.sort_by_key(|span| (span.start, span.end));
        spans.dedup();
    }

    let mut regions: BTreeMap<BindingId, Vec<SemanticBindingRegion>> = BTreeMap::new();
    for (function, body) in &mir.bodies {
        let Ok(function_ordinal) = u32::try_from(function.0) else {
            continue;
        };
        let program_function = ProgramFunctionId(function_ordinal);
        let function_span = mir
            .functions
            .get(function)
            .map_or(ProgramSpan { start: 0, end: 0 }, |metadata| {
                portable_span(metadata.span)
            });
        for local in &body.locals {
            let Some(binding) = local.binding else {
                continue;
            };
            let Ok(local_ordinal) = u32::try_from(local.id.0) else {
                continue;
            };
            let value = RegionValueId {
                function: program_function,
                local: local_ordinal,
            };
            let observations = store
                .program_points
                .iter()
                .filter(|point| point.point.function == program_function)
                .filter_map(|point| {
                    let local = point.local(value)?;
                    Some(SemanticFactObservation {
                        point: point.point,
                        span: point.span,
                        assignment: local.assignment,
                        fact: local.fact.clone(),
                    })
                })
                .collect();
            regions
                .entry(binding)
                .or_default()
                .push(SemanticBindingRegion {
                    value,
                    function_span,
                    observations,
                });
        }
    }
    for binding_regions in regions.values_mut() {
        binding_regions.sort_by_key(|region| region.value);
    }

    let mut bindings = lowering
        .assembly
        .bindings
        .iter()
        .map(|binding| SemanticBindingFacts {
            binding: binding.id,
            name: binding.name.0.clone(),
            role: binding.role.clone(),
            storage: binding.storage.clone(),
            workspace_visibility: binding.workspace_visibility.clone(),
            declaration: portable_span(binding.declared_span),
            references: references.remove(&binding.id).unwrap_or_default(),
            regions: regions.remove(&binding.id).unwrap_or_default(),
        })
        .collect::<Vec<_>>();
    bindings.sort_by_key(|binding| binding.binding.0);

    let mut functions = lowering
        .assembly
        .functions
        .iter()
        .filter_map(|function| {
            let ordinal = u32::try_from(function.id.0).ok()?;
            let function_id = ProgramFunctionId(ordinal);
            Some(SemanticFunctionFacts {
                function: function_id,
                name: function.name.0.clone(),
                span: portable_span(function.span),
                parameters: function.params.clone(),
                outputs: function.outputs.clone(),
                analysis: store.function(function_id).cloned(),
            })
        })
        .collect::<Vec<_>>();
    functions.sort_by_key(|function| function.function);

    SemanticDocumentFacts {
        revision: store.revision.clone(),
        bindings,
        functions,
        classes: store.classes.clone(),
    }
}

fn portable_span(span: runmat_hir::Span) -> ProgramSpan {
    ProgramSpan {
        start: span.start as u64,
        end: span.end as u64,
    }
}

//! Publication projection from selected bindings, exact peeled paths and
//! canonical request facts. This does not certify provider arithmetic.
use std::collections::{HashMap, HashSet};

use runmat_accelerate_api::{
    moment_precision, ProviderImageMomentConversion, ProviderImageMomentProgram,
    ProviderImageMomentPublication, ProviderPrecision,
};
use runmat_builtins::{infer_catalog_call, DOUBLE_CATALOG_ENTRY, SINGLE_CATALOG_ENTRY};
use runmat_runtime::builtins::common::spec::{
    canonical_fusion_declaration, FusionAdmissionWork, FusionProofStage, FusionReductionContext,
};
use runmat_types::{CallRequest, OutputSelection, StorageFact, ValueFact};
use runmat_value::Value;

use super::{
    ImageMeanForm, ImageMeanRoute, ImageSourceAdmissionFailure, ImageSourceInvocation,
    ImageSourceStep,
};
use crate::fusion::{FusionGroupPlan, FusionPattern};
use crate::graph::{AccelNodeLabel, NodeId, ValueId};
use crate::placement::PlacementCorrelationId;

type ProjectedMoments = (
    ProviderImageMomentProgram,
    Vec<(ValueId, FusionAdmissionWork)>,
);

pub(super) fn conversion_entry(
    step: &ImageSourceStep,
) -> Option<&'static runmat_builtins::BuiltinCatalogEntry> {
    let binding = step.selected_binding?;
    if !matches!(step.label, AccelNodeLabel::Builtin { .. }) || step.inputs.len() != 1 {
        return None;
    }
    let entry = if binding.builtin == SINGLE_CATALOG_ENTRY.identity {
        &SINGLE_CATALOG_ENTRY
    } else if binding.builtin == DOUBLE_CATALOG_ENTRY.identity {
        &DOUBLE_CATALOG_ENTRY
    } else {
        return None;
    };
    entry
        .bindings
        .iter()
        .any(|variant| entry.binding_identity(variant) == binding)
        .then_some(entry)
}

struct Projection<'a> {
    source: &'a ImageSourceInvocation,
    facts: HashMap<ValueId, ValueFact>,
    nodes: HashMap<NodeId, &'a ImageSourceStep>,
    leaves: HashMap<ValueId, &'a Value>,
    precision: ProviderPrecision,
    placement: Option<PlacementCorrelationId>,
    work: Vec<(ValueId, FusionAdmissionWork)>,
}

impl Projection<'_> {
    async fn publication(
        &mut self,
        raw: ValueId,
        nodes: &[NodeId],
        reduction: bool,
    ) -> Result<Option<ProviderImageMomentPublication>, ImageSourceAdmissionFailure> {
        let Some(step) = self
            .source
            .source
            .steps
            .iter()
            .find(|step| step.output == raw)
        else {
            return Ok(None);
        };
        let Some(&input) = step.inputs.first() else {
            return Ok(None);
        };
        let Some(mut raw_fact) = self.facts.get(&raw).cloned() else {
            return Ok(None);
        };
        if reduction && raw_fact.storage == StorageFact::Unknown {
            let Some(binding) = step.selected_binding else {
                return Ok(None);
            };
            let Some(declaration) = canonical_fusion_declaration(binding) else {
                return Ok(None);
            };
            let Some(owner) = declaration.reduction else {
                return Ok(None);
            };
            let Some(input_fact) = self.facts.get(&input) else {
                return Ok(None);
            };
            let Some(controls) = step.inputs[1..]
                .iter()
                .map(|id| self.leaves.get(id).copied())
                .collect::<Option<Vec<_>>>()
            else {
                return Ok(None);
            };
            // Control preparation is observed before starting, including failure
            // and cancellation. Cloning COW controls is not an element scan.
            let mut observation =
                crate::fusion::admission::AdmissionObservation::new(self.placement);
            let mut shape_dimensions = 0_u64;
            let mut string_bytes = 0_u64;
            let mut char_elements = 0_u64;
            for value in &controls {
                match value {
                    Value::Tensor(tensor) => shape_dimensions += tensor.shape().len() as u64,
                    Value::String(value) => string_bytes += value.len() as u64,
                    Value::StringArray(value) => {
                        shape_dimensions += value.shape.len() as u64;
                        string_bytes +=
                            value.data.iter().map(|text| text.len() as u64).sum::<u64>();
                    }
                    Value::CharArray(value) => {
                        shape_dimensions += value.shape.len() as u64;
                        char_elements += value.data.len() as u64;
                    }
                    Value::Num(_) | Value::Int(_) => {}
                    _ => return Ok(None),
                }
            }
            observation.work = FusionAdmissionWork::DomainProof {
                binding,
                provider_precision: self.precision,
                stages: vec![FusionProofStage::ControlMaterialization {
                    values: controls.len() as u64,
                    shape_dimensions,
                    string_bytes,
                    char_elements,
                }],
            };
            let controls = controls.into_iter().cloned().collect::<Vec<_>>();
            let storage = owner
                .certify_publication_storage(
                    FusionReductionContext {
                        input: input_fact,
                        controls: &controls,
                        requested_outputs: 1,
                    },
                    &raw_fact,
                )
                .await
                .map_err(|error| {
                    ImageSourceAdmissionFailure::CanonicalExecution(
                        runmat_accelerate_api::ProviderExecutionError::from_source(error),
                    )
                })?;
            self.work.push((raw, observation.completed()));
            let Some(storage) = storage else {
                return Ok(None);
            };
            raw_fact.storage = storage;
        }
        if moment_precision(&raw_fact).is_none() {
            return Ok(None);
        }
        self.facts.insert(raw, raw_fact.clone());
        let mut current = raw;
        let mut current_fact = raw_fact.clone();
        let mut conversions = Vec::new();
        for node in nodes {
            let Some(step) = self.nodes.get(node).copied() else {
                return Ok(None);
            };
            if step.inputs.as_slice() != [current] {
                return Ok(None);
            }
            let Some(entry) = conversion_entry(step) else {
                return Ok(None);
            };
            let inference = infer_catalog_call(
                entry,
                &CallRequest {
                    arguments: vec![current_fact],
                    outputs: OutputSelection::default(),
                    literals: Default::default(),
                },
            );
            if !inference.diagnostics.is_empty() || inference.outputs.len() != 1 {
                return Ok(None);
            }
            let output = inference.outputs.into_iter().next().expect("one output");
            let Some(canonical) = self.facts.get(&step.output) else {
                return Ok(None);
            };
            if output.numeric() != canonical.numeric()
                || output.shape != canonical.shape
                || output.shape != raw_fact.shape
                || moment_precision(&output).is_none()
            {
                return Ok(None);
            }
            conversions.push(ProviderImageMomentConversion {
                source_input: current.into(),
                source_output: step.output.into(),
                output_fact: output.clone(),
            });
            self.facts.insert(step.output, output.clone());
            current_fact = output;
            current = step.output;
        }
        Ok(Some(ProviderImageMomentPublication {
            source_input: input.into(),
            raw_output: raw.into(),
            raw_fact,
            conversions,
            public_output: current.into(),
        }))
    }

    async fn route(
        &mut self,
        route: &ImageMeanRoute,
    ) -> Result<Option<Vec<ProviderImageMomentPublication>>, ImageSourceAdmissionFailure> {
        let calls = match &route.form {
            ImageMeanForm::VecDims { call, .. } => vec![call],
            ImageMeanForm::Nested { inner, outer, .. } => vec![inner, outer],
        };
        let mut publications = Vec::new();
        for call in calls {
            // Peeled paths are outer-to-inner. Deduplicate exact selected
            // edges, never equal classes. Shared path prefixes remain once.
            let mut selected = HashSet::new();
            for path in &route.cast_paths {
                if path.peeled_value != call.output {
                    continue;
                }
                let mut current = call.output;
                for node in path.nodes.iter().rev() {
                    let Some(step) = self.nodes.get(node) else {
                        return Ok(None);
                    };
                    if step.inputs.as_slice() != [current] || conversion_entry(step).is_none() {
                        return Ok(None);
                    }
                    selected.insert(*node);
                    current = step.output;
                }
                if current != path.consumer_value {
                    return Ok(None);
                }
            }
            let nodes = self
                .source
                .source
                .steps
                .iter()
                .filter(|step| selected.contains(&step.node))
                .map(|step| step.node)
                .collect::<Vec<_>>();
            let Some(publication) = self.publication(call.output, &nodes, true).await? else {
                return Ok(None);
            };
            publications.push(publication);
        }
        Ok(Some(publications))
    }
}

impl ImageSourceInvocation {
    pub(super) async fn project_moment_program(
        &self,
        plan: &FusionGroupPlan,
        inputs: &[Value],
        precision: ProviderPrecision,
        placement: Option<PlacementCorrelationId>,
    ) -> Result<Option<ProjectedMoments>, ImageSourceAdmissionFailure> {
        let Some(FusionPattern::ImageNormalize(pattern)) = plan.group.pattern.as_ref() else {
            return Ok(None);
        };
        let Some(routes) = pattern.reduction_routes.as_ref() else {
            return Ok(None);
        };
        let Some(producers) = routes.stage_producers() else {
            return Ok(None);
        };
        let mut leaves = plan
            .const_values
            .iter()
            .map(|(id, value)| (*id, value))
            .collect::<HashMap<_, _>>();
        leaves.extend(plan.inputs.iter().copied().zip(inputs));
        let mut projection = Projection {
            source: self,
            facts: self.facts.iter().cloned().collect(),
            nodes: self
                .source
                .steps
                .iter()
                .map(|step| (step.node, step))
                .collect(),
            leaves,
            precision,
            placement,
            work: Vec::new(),
        };
        let Some(mu) = projection.route(&routes.mu).await? else {
            return Ok(None);
        };
        let Some(variance) = projection.route(&routes.variance).await? else {
            return Ok(None);
        };
        let Some(position) = self
            .source
            .steps
            .iter()
            .position(|step| step.output == producers.sigma)
        else {
            return Ok(None);
        };
        let mut sigma_nodes = Vec::new();
        let mut current = producers.sigma;
        for step in &self.source.steps[position + 1..] {
            if step.inputs.as_slice() != [current] || conversion_entry(step).is_none() {
                break;
            }
            sigma_nodes.push(step.node);
            current = step.output;
        }
        let Some(sigma) = projection
            .publication(producers.sigma, &sigma_nodes, false)
            .await?
        else {
            return Ok(None);
        };
        let program = ProviderImageMomentProgram {
            mu,
            variance,
            sigma,
        };
        Ok(program.validate().then_some((program, projection.work)))
    }
}

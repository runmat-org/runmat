//! Publication projection from selected bindings, exact peeled paths and
//! canonical request facts. This does not certify provider arithmetic.
use runmat_accelerate_api::{
    moment_precision, ProviderImageMomentConversion, ProviderImageMomentProgram,
    ProviderImageMomentPublication, ProviderPrecision,
};
use runmat_builtins::{infer_fusion_numeric_fact, DOUBLE_CATALOG_ENTRY, SINGLE_CATALOG_ENTRY};
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
    facts: Vec<(ValueId, ValueFact)>,
    nodes: Vec<(NodeId, &'a ImageSourceStep)>,
    leaves: Vec<(ValueId, &'a Value)>,
    precision: ProviderPrecision,
    placement: Option<PlacementCorrelationId>,
    work: Vec<(ValueId, FusionAdmissionWork)>,
}

fn lookup<'a, T>(entries: &'a [(ValueId, T)], id: &ValueId) -> Option<&'a T> {
    entries
        .binary_search_by_key(id, |(key, _)| *key)
        .ok()
        .map(|index| &entries[index].1)
}

fn replace_fact(entries: &mut [(ValueId, ValueFact)], id: ValueId, fact: ValueFact) -> Option<()> {
    let index = entries.binary_search_by_key(&id, |(key, _)| *key).ok()?;
    entries[index].1 = fact;
    Some(())
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
        let Some(mut raw_fact) = lookup(&self.facts, &raw).cloned() else {
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
            let Some(input_fact) = lookup(&self.facts, &input) else {
                return Ok(None);
            };
            let Some(controls) = step.inputs[1..]
                .iter()
                .map(|id| lookup(&self.leaves, id).copied())
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
                    Value::String(value) => {
                        string_bytes += value.as_str().map_or(0, str::len) as u64
                    }
                    Value::StringArray(value) => {
                        shape_dimensions += value.shape.len() as u64;
                        string_bytes += value
                            .data()
                            .iter()
                            .map(|text| text.as_str().map_or(0, str::len) as u64)
                            .sum::<u64>();
                    }
                    Value::CharArray(value) => {
                        shape_dimensions += value.shape.len() as u64;
                        char_elements += value.data().len() as u64;
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
        if replace_fact(&mut self.facts, raw, raw_fact.clone()).is_none() {
            return Ok(None);
        }
        let mut current = raw;
        let mut current_fact = raw_fact.clone();
        let mut conversions = Vec::new();
        for node in nodes {
            let Some(step) = lookup(&self.nodes, node).copied() else {
                return Ok(None);
            };
            if step.inputs.as_slice() != [current] {
                return Ok(None);
            }
            let Some(entry) = conversion_entry(step) else {
                return Ok(None);
            };
            let request = CallRequest {
                arguments: vec![current_fact],
                outputs: OutputSelection::default(),
                literals: Default::default(),
            };
            let Some(Ok(output)) = infer_fusion_numeric_fact(entry, &request) else {
                return Ok(None);
            };
            let Some(canonical) = lookup(&self.facts, &step.output) else {
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
            if replace_fact(&mut self.facts, step.output, output.clone()).is_none() {
                return Ok(None);
            }
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
            let mut selected = Vec::new();
            for path in &route.cast_paths {
                if path.peeled_value != call.output {
                    continue;
                }
                let mut current = call.output;
                for node in path.nodes.iter().rev() {
                    let Some(step) = lookup(&self.nodes, node) else {
                        return Ok(None);
                    };
                    if step.inputs.as_slice() != [current] || conversion_entry(step).is_none() {
                        return Ok(None);
                    }
                    selected.push(*node);
                    current = step.output;
                }
                if current != path.consumer_value {
                    return Ok(None);
                }
            }
            selected.sort_unstable();
            selected.dedup();
            let nodes = self
                .source
                .source
                .steps
                .iter()
                .filter(|step| selected.binary_search(&step.node).is_ok())
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
        let mut ordered = Vec::new();
        ordered
            .try_reserve_exact(
                plan.const_values
                    .len()
                    .checked_add(plan.inputs.len())
                    .ok_or(ImageSourceAdmissionFailure::Inputs)?,
            )
            .map_err(|error| ImageSourceAdmissionFailure::CanonicalUnproven(error.into()))?;
        // Explicit priority retains the live value on an ID collision. The
        // unstable in-place sort needs no hidden scratch allocation.
        ordered.extend(
            plan.inputs
                .iter()
                .copied()
                .zip(inputs)
                .map(|(id, value)| (id, value, true)),
        );
        ordered.extend(
            plan.const_values
                .iter()
                .map(|(id, value)| (*id, value, false)),
        );
        ordered.sort_unstable_by_key(|(id, _, live)| (*id, !*live));
        ordered.dedup_by(|left, right| left.0 == right.0);
        let leaves: Vec<_> = ordered
            .into_iter()
            .map(|(id, value, _)| (id, value))
            .collect();
        let mut nodes: Vec<_> = self
            .source
            .steps
            .iter()
            .map(|step| (step.node, step))
            .collect();
        nodes.sort_unstable_by_key(|(id, _)| *id);
        let mut projection = Projection {
            source: self,
            facts: self.facts.to_vec(),
            nodes,
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

//! A graph-backed generator adds argument roots to its original closed callable
//! snapshot. The original activation graph is never rebuilt from flat captures.
use super::*;
use runmat_execution::value::{GraphInvocationRoots, GraphNode, GraphNodeRef, ValuePayload};
use runmat_execution_artifact::GraphExecutionRequest;

fn arguments<'a>(
    admission: GeneratorProgramAdmissionRef<'_>,
    plan: &'a GeneratorChunk,
) -> AnalysisBridgeResult<&'a [ValuePayload]> {
    if !plan
        .arguments
        .starts_with(admission.frozen_inputs.direct_captures())
    {
        return Err(AnalysisBridgeError::Identity(
            "generator chunk changed original capture values",
        ));
    }
    Ok(&plan.arguments[admission.frozen_inputs.direct_captures().len()..])
}

pub(super) fn prepare(
    admission: GeneratorProgramAdmissionRef<'_>,
    plan: &GeneratorChunk,
    objects: &[ProgramExecutionObject],
) -> AnalysisBridgeResult<GraphExecutionRequest> {
    let template = admission
        .frozen_inputs
        .graph()
        .ok_or(AnalysisBridgeError::Identity(
            "graph callable admission is absent",
        ))?;
    let GraphInvocationRoots::Callable { target, .. } = &template.roots else {
        return Err(AnalysisBridgeError::Identity(
            "graph callable target is absent",
        ));
    };
    let target = *target;
    let mut request = template.clone();
    let mut roots = Vec::new();
    for argument in arguments(admission, plan)? {
        let value = if let ValuePayload::Object(reference) = argument {
            let object = objects
                .iter()
                .find(|object| object.reference == **reference)
                .ok_or(AnalysisBridgeError::Identity(
                    "graph argument has no original immutable object",
                ))?;
            object
                .decode_externalized_value()?
                .ok_or(AnalysisBridgeError::Identity(
                    "graph argument object is not a portable value",
                ))?
        } else {
            argument.clone()
        };
        if value.logical_digest()? != argument.logical_digest()? {
            return Err(AnalysisBridgeError::Identity(
                "graph argument object changed logical value",
            ));
        }
        let ordinal = u32::try_from(request.graph.nodes.len())
            .map_err(|_| AnalysisBridgeError::Invalid("graph node count overflows".into()))?;
        roots.push(GraphNodeRef(ordinal));
        request.graph.nodes.push(GraphNode::Leaf { value });
    }
    request.graph.roots = std::iter::once(target)
        .chain(roots.iter().copied())
        .collect();
    request.roots = GraphInvocationRoots::Callable {
        target,
        arguments: roots,
    };
    validate(admission, plan, &request)?;
    Ok(request)
}

pub(super) fn validate(
    admission: GeneratorProgramAdmissionRef<'_>,
    plan: &GeneratorChunk,
    request: &GraphExecutionRequest,
) -> AnalysisBridgeResult<()> {
    let template = admission
        .frozen_inputs
        .graph()
        .ok_or(AnalysisBridgeError::Identity(
            "graph request has no original graph admission",
        ))?;
    request.validate()?;
    // Compare all non-invocation fields by borrow; validation does not copy the
    // executable or graph solely to manufacture an expected request.
    let same = request.schema_version == template.schema_version
        && request.recipe == template.recipe
        && request.artifact == template.artifact
        && request.retained_units == template.retained_units
        && request.class_dependencies == template.class_dependencies
        && request.class_method_dependencies == template.class_method_dependencies
        && request.static_dependencies == template.static_dependencies
        && request.selected_bundle_archives == template.selected_bundle_archives
        && request.selected_foreign_policies == template.selected_foreign_policies
        && request.context == template.context
        && request.assignment == template.assignment
        && request.job_id == template.job_id
        && request.requested_outputs == template.requested_outputs;
    let GraphInvocationRoots::Callable {
        target,
        arguments: actual,
    } = &request.roots
    else {
        return Err(AnalysisBridgeError::Identity(
            "generator graph changed invocation role",
        ));
    };
    let GraphInvocationRoots::Callable {
        target: original, ..
    } = &template.roots
    else {
        return Err(AnalysisBridgeError::Identity(
            "admitted generator graph has no target",
        ));
    };
    let arguments = arguments(admission, plan)?;
    let count = template
        .graph
        .nodes
        .len()
        .checked_add(arguments.len())
        .ok_or_else(|| AnalysisBridgeError::Invalid("graph extent overflow".into()))?;
    if !same
        || target != original
        || actual.len() != arguments.len()
        || request.graph.nodes.len() != count
        || request.graph.nodes[..template.graph.nodes.len()] != template.graph.nodes
    {
        return Err(AnalysisBridgeError::Identity(
            "generator graph differs from its original selected closure",
        ));
    }
    for (index, (reference, expected)) in actual.iter().zip(arguments).enumerate() {
        if usize::try_from(reference.0).ok() != Some(template.graph.nodes.len() + index) {
            return Err(AnalysisBridgeError::Identity(
                "graph argument root order changed",
            ));
        }
        let Some(GraphNode::Leaf { value }) = request.graph.nodes.get(reference.0 as usize) else {
            return Err(AnalysisBridgeError::Identity(
                "graph argument is not a portable numeric leaf",
            ));
        };
        if value.logical_digest()? != expected.logical_digest()? {
            return Err(AnalysisBridgeError::Identity(
                "graph argument differs from frozen chunk value",
            ));
        }
    }
    Ok(())
}

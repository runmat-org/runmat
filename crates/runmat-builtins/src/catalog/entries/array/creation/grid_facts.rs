use crate::catalog::inference::{argument_error, finish_fixed_outputs, literal_text};
use crate::BuiltinCatalogEntry;
use runmat_types::{
    infer_call, CallContract, CallInference, CallRequest, DynamicReason, ResidencyFact, ShapeFact,
    StorageFact, ValueFact,
};

#[derive(Clone, Copy)]
pub(super) enum GridKind {
    Mesh,
    Nd,
}

pub(super) fn infer(
    request: &CallRequest,
    entry: &BuiltinCatalogEntry,
    grid: GridKind,
) -> CallInference {
    let mut diagnostics = Vec::new();
    let like = if matches!(grid, GridKind::Mesh) {
        request.literals.literal_args.iter().position(|literal| {
            literal_text(literal).is_some_and(|text| text.eq_ignore_ascii_case("like"))
        })
    } else {
        None
    };
    let axes = &request.arguments[..like.unwrap_or(request.arguments.len())];
    if axes.is_empty() {
        diagnostics.push(argument_error(
            "RM-GRID-AXES",
            "at least one coordinate axis is required",
            0,
        ));
    }
    if matches!(grid, GridKind::Mesh) && axes.len() > 3 {
        diagnostics.push(argument_error(
            "RM-MESHGRID-AXES",
            "meshgrid accepts at most three coordinate axes",
            3,
        ));
    }
    let shape = match grid {
        GridKind::Mesh => {
            let x = axes.first().and_then(axis_len);
            let y = axes.get(1).and_then(axis_len).or(x);
            let z = axes.get(2).and_then(axis_len);
            match (axes.len() > 2, z) {
                (false, _) | (true, Some(1)) => ShapeFact::from(vec![y, x]),
                (true, Some(z)) => ShapeFact::from(vec![y, x, Some(z)]),
                (true, None) => ShapeFact::Unknown,
            }
        }
        GridKind::Nd => {
            if axes.len() == 1 {
                let n = request.outputs.requested.known_count().unwrap_or(1);
                if n == 1 {
                    ShapeFact::from(vec![axes.first().and_then(axis_len), Some(1)])
                } else {
                    ShapeFact::from(vec![axes.first().and_then(axis_len); n])
                }
            } else {
                ShapeFact::from(axes.iter().map(axis_len).collect::<Vec<_>>())
            }
        }
    };
    let outputs = match grid {
        GridKind::Mesh => axes.len().clamp(2, 3),
        GridKind::Nd if axes.len() == 1 => {
            request.outputs.requested.known_count().unwrap_or(1).max(1)
        }
        GridKind::Nd => axes.len(),
    };
    let output_facts = (0..outputs)
        .map(|index| {
            let source = like
                .and_then(|like| request.arguments.get(like + 1))
                .or_else(|| axes.get(index).or_else(|| axes.first()));
            let mut output = source
                .cloned()
                .unwrap_or_else(|| ValueFact::unknown(DynamicReason::RuntimeValue));
            output.shape = shape.clone();
            output.storage = StorageFact::Dense;
            if !matches!(output.residency, ResidencyFact::Host) {
                output.residency = ResidencyFact::Unknown;
            }
            output
        })
        .collect::<Vec<_>>();
    if matches!(grid, GridKind::Nd)
        && axes.len() == 1
        && request.outputs.requested.known_count().is_none()
    {
        let mut contract = CallContract::fixed(output_facts.clone());
        contract.maximum_outputs = None;
        contract.variadic_output = output_facts.first().cloned().map(Box::new);
        contract.effects = entry.contract.effect_set();
        contract.capabilities = entry.contract.capability_set();
        let mut inference = infer_call(&contract, request);
        diagnostics.append(&mut inference.diagnostics);
        inference.diagnostics = diagnostics;
        inference
    } else {
        finish_fixed_outputs(entry, request, output_facts, diagnostics)
    }
}

fn axis_len(value: &ValueFact) -> Option<usize> {
    if value.shape.is_proven_equivalent(&ShapeFact::Scalar) {
        return Some(1);
    }
    let dims = value.shape.known_dims()?;
    if dims.iter().filter(|dim| dim.is_some_and(|n| n > 1)).count() > 1 {
        return None;
    }
    value.shape.element_count()
}

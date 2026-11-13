use super::*;

pub(crate) fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let mut diagnostics = Vec::new();
    let Some(input) = request.arguments.first() else {
        diagnostics.push(invalid(entry, "sample data is required"));
        return finish(
            request,
            entry,
            vec![ValueFact::unknown(DynamicReason::RuntimeValue)],
            diagnostics,
        );
    };
    if !dense_samples(input) && input.kind != ValueKindFact::Unknown {
        diagnostics.push(invalid_input(
            crate::NumericalReductionKind::Gradient,
            "expected dense real or floating complex samples",
        ));
    }
    let dimensions = input.shape.known_dims().map(|mut dimensions| {
        if dimensions.is_empty() {
            dimensions = vec![Some(0), Some(0)];
        } else if dimensions.len() == 1 {
            dimensions.insert(0, Some(1));
        }
        dimensions
    });
    let shape = dimensions
        .clone()
        .map_or_else(|| input.shape.clone(), ShapeFact::from);
    let axes = dimensions.as_ref().and_then(|dimensions| {
        if dimensions.len() > 2 {
            let mut axes = vec![2, 1];
            axes.extend(3..=dimensions.len());
            return Some(axes);
        }
        let first = dimensions.first().copied().flatten()?;
        let second = dimensions.get(1).copied().flatten()?;
        Some(if first == 1 || second == 1 {
            vec![if first == 1 && second != 1 { 2 } else { 1 }]
        } else {
            vec![2, 1]
        })
    });
    let output = numeric_output(input, shape);
    if let Some(axes) = axes {
        let spacing_count = request.arguments.len().saturating_sub(1);
        if spacing_count != 0 && spacing_count != 1 && spacing_count != axes.len() {
            diagnostics.push(invalid(
                entry,
                "provide one scalar spacing or one spacing per gradient dimension",
            ));
        }
        for (offset, spacing) in request.arguments.iter().enumerate().skip(1) {
            let real = matches!(
                spacing.kind,
                ValueKindFact::Numeric(NumericFact {
                    domain: NumericDomain::Real,
                    ..
                }) | ValueKindFact::Logical
                    | ValueKindFact::Unknown
            );
            if !real || spacing.storage == StorageFact::Sparse {
                diagnostics.push(invalid(
                    entry,
                    "spacing must be dense real numeric or logical data",
                ));
            }
            if let Some(count) = spacing.shape.element_count() {
                if count == 0 {
                    diagnostics.push(invalid(entry, "spacing cannot be empty"));
                }
                if count != 1 {
                    if spacing_count == 1 && axes.len() != 1 {
                        diagnostics.push(invalid(entry, "coordinate vectors require one spacing for every array gradient dimension"));
                    }
                    let axis = axes.get(offset - 1).or_else(|| axes.first());
                    if let Some(expected) =
                        axis.and_then(|axis| dimensions.as_ref()?.get(axis - 1).copied().flatten())
                    {
                        if count != expected {
                            diagnostics.push(invalid(
                                entry,
                                "coordinate count must match its gradient dimension",
                            ));
                        }
                    }
                }
            }
            if let Some(value) =
                literal(request, offset).and_then(crate::ResolveContext::numeric_from_literal)
            {
                if !value.is_finite() || value == 0.0 {
                    diagnostics.push(invalid(entry, "scalar spacing must be finite and nonzero"));
                }
            }
        }
        finish(request, entry, vec![output; axes.len()], diagnostics)
    } else {
        // Shape-dependent nargout cannot be replaced by a fixed two-output
        // claim when the input rank/vector classification is unknown.
        let mut contract = CallContract::dynamic(DynamicReason::RuntimeValue);
        contract.variadic_output = Some(Box::new(output));
        contract.effects = entry.contract.effect_set();
        contract.capabilities = entry.contract.capability_set();
        let mut result = infer_call(&contract, request);
        result.diagnostics.extend(diagnostics);
        result
    }
}

//! Editor-only descriptor selection; no runtime overload dispatch.
use lsp_types::{ParameterInformation, ParameterLabel, SignatureHelp, SignatureInformation};
use runmat_builtins::{BuiltinParamArity, BuiltinParamType, BuiltinSignatureDescriptor};

mod call_site;
use call_site::ArgumentKind;
pub(super) use call_site::CallSite;

#[cfg(test)]
mod tests;

impl CallSite {
    pub fn best(&self, signatures: &[BuiltinSignatureDescriptor]) -> Vec<usize> {
        let mut best = None;
        let mut selected = Vec::new();
        for (index, signature) in signatures.iter().enumerate() {
            let Some(score) = self.score(signature) else {
                continue;
            };
            if best.is_none_or(|prior| score < prior) {
                best = Some(score);
                selected.clear();
            }
            if best == Some(score) {
                selected.push(index);
            }
        }
        selected
    }

    fn score(
        &self,
        signature: &BuiltinSignatureDescriptor,
    ) -> Option<(usize, usize, usize, usize)> {
        let required = signature
            .inputs
            .iter()
            .filter(|parameter| parameter.arity == BuiltinParamArity::Required)
            .count();
        let variadic = signature
            .inputs
            .last()
            .is_some_and(|parameter| parameter.arity == BuiltinParamArity::Variadic);
        if (!variadic && self.arguments.len() > signature.inputs.len())
            || (self.complete && self.arguments.len() < required)
        {
            return None;
        }
        let output_variadic = signature
            .outputs
            .last()
            .is_some_and(|parameter| parameter.arity == BuiltinParamArity::Variadic);
        let output_difference = match self.outputs {
            Some(requested) if !output_variadic && requested > signature.outputs.len() => {
                return None
            }
            Some(requested) if !output_variadic => {
                signature.outputs.len().saturating_sub(requested)
            }
            _ => 0,
        };
        let mut kind_penalty = 0;
        for (index, kind) in self.arguments.iter().enumerate() {
            let parameter = signature
                .inputs
                .get(index)
                .or_else(|| variadic.then(|| signature.inputs.last()).flatten())?;
            kind_penalty += match_quality(*kind, parameter.ty)?;
        }
        Some((
            output_difference,
            required.saturating_sub(self.arguments.len()),
            signature.inputs.len().saturating_sub(self.arguments.len()),
            kind_penalty,
        ))
    }
}

fn match_quality(kind: ArgumentKind, expected: BuiltinParamType) -> Option<usize> {
    use ArgumentKind as Kind;
    use BuiltinParamType as Type;
    if kind == Kind::Unknown {
        return Some(1);
    }
    if matches!(
        expected,
        Type::Any | Type::PropertyValue | Type::LikePrototype
    ) {
        return Some(2);
    }
    match (kind, expected) {
        (Kind::Integer, Type::IntegerScalar | Type::NumericScalar | Type::SizeArg)
        | (Kind::Number, Type::NumericScalar | Type::SizeArg)
        | (Kind::Text, Type::StringScalar | Type::StyleSpec | Type::PropertyName)
        | (Kind::Logical, Type::LogicalArray)
        | (Kind::NumericArray, Type::NumericArray | Type::SizeArg | Type::StyleSpec)
        | (Kind::Callable, Type::Callable) => Some(0),
        (Kind::Integer | Kind::Number | Kind::Logical, Type::NumericArray | Type::AxesHandle)
        | (Kind::Text, Type::Callable) => Some(1),
        _ => None,
    }
}

pub(super) fn help(
    signatures: &[BuiltinSignatureDescriptor],
    site: Option<&CallSite>,
) -> SignatureHelp {
    let active_signature = site
        .and_then(|site| site.best(signatures).first().copied())
        .unwrap_or(0);
    let active_argument = site.map(|site| site.active_argument as u32);
    SignatureHelp {
        signatures: signatures
            .iter()
            .map(|signature| SignatureInformation {
                label: signature.label.into(),
                documentation: None,
                parameters: Some(parameters(signature)),
                active_parameter: active_argument
                    .filter(|_| !signature.inputs.is_empty())
                    .map(|index| index.min(signature.inputs.len().saturating_sub(1) as u32)),
            })
            .collect(),
        active_signature: Some(active_signature as u32),
        active_parameter: active_argument,
    }
}

fn parameters(signature: &BuiltinSignatureDescriptor) -> Vec<ParameterInformation> {
    // Search input labels in order, after the callee, so repeated placeholders
    // refer to distinct inputs rather than an output or an earlier argument.
    let mut cursor = signature.label.find('(').map(|offset| offset + 1);
    signature
        .inputs
        .iter()
        .map(|parameter| {
            let range = cursor.and_then(|start| {
                signature.label[start..].find(parameter.name).map(|offset| {
                    let begin = start + offset;
                    begin..begin + parameter.name.len()
                })
            });
            cursor = range.as_ref().map(|range| range.end);
            let label = range.map_or_else(
                || ParameterLabel::Simple(parameter.name.into()),
                |range| {
                    ParameterLabel::LabelOffsets([
                        signature.label[..range.start].encode_utf16().count() as u32,
                        signature.label[..range.end].encode_utf16().count() as u32,
                    ])
                },
            );
            ParameterInformation {
                label,
                documentation: (!parameter.description.is_empty())
                    .then(|| lsp_types::Documentation::String(parameter.description.into())),
            }
        })
        .collect()
}

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
    // These are display-label offsets, not a second parameter-type authority.
    // Positional ranges also cover placeholders and names such as dim/vecdim.
    let ranges =
        parameter_ranges(signature.label).filter(|ranges| ranges.len() == signature.inputs.len());
    signature
        .inputs
        .iter()
        .enumerate()
        .map(|(index, parameter)| {
            let range = ranges.as_ref().and_then(|ranges| ranges.get(index));
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

fn parameter_ranges(label: &str) -> Option<Vec<std::ops::Range<usize>>> {
    use runmat_lexer::Token;
    let tokens = runmat_lexer::tokenize_detailed(label);
    let open = tokens
        .iter()
        .position(|token| token.token == Token::LParen)?;
    let mut expected = vec![Token::RParen];
    let mut begin = tokens[open].end;
    let mut ranges = Vec::new();
    for token in &tokens[open + 1..] {
        let separator = token.token == Token::Comma && expected.len() == 1;
        let end = token.token == Token::RParen && expected.len() == 1;
        if separator || end {
            let raw = &label[begin..token.start];
            let content = raw.trim();
            if !content.is_empty() {
                let start = begin + raw.len() - raw.trim_start().len();
                ranges.push(start..start + content.len());
            } else if separator {
                return None;
            }
            if end {
                return Some(ranges);
            }
            begin = token.end;
        }
        match token.token {
            Token::LParen => expected.push(Token::RParen),
            Token::LBracket => expected.push(Token::RBracket),
            Token::LBrace => expected.push(Token::RBrace),
            Token::RParen | Token::RBracket | Token::RBrace => {
                if expected.pop().as_ref() != Some(&token.token) {
                    return None;
                }
            }
            _ => {}
        }
    }
    None // Informal/variadic display grammar retains the named-label fallback.
}

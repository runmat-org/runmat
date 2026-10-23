use super::*;

pub(super) enum FitlmAction {
    Intercept(bool),
    ResponseVar(String),
    PredictorVars(Vec<String>),
    VarNames(Vec<String>),
    Exclude(ExcludeSpec),
    Weights(WeightSpec),
}

pub(super) fn decode_fit_option(named: &NamedOption, value: &Value) -> BuiltinResult<FitlmAction> {
    Ok(match named.identity {
        FitlmOption::Intercept => FitlmAction::Intercept(bool_scalar(value, "Intercept")?),
        FitlmOption::ResponseVar => FitlmAction::ResponseVar(
            scalar_text(value)?
                .ok_or_else(|| fitlm_invalid("fitlm: ResponseVar must be text"))?
                .into_owned(),
        ),
        FitlmOption::PredictorVars => {
            FitlmAction::PredictorVars(text_list(value, "PredictorVars")?)
        }
        FitlmOption::VarNames => FitlmAction::VarNames(text_list(value, "VarNames")?),
        FitlmOption::Exclude => FitlmAction::Exclude(exclude_spec(value)?),
        FitlmOption::Weights => FitlmAction::Weights(if let Some(name) = scalar_text(value)? {
            WeightSpec::Variable(name.into_owned())
        } else {
            WeightSpec::Values(numeric_vector(value, "Weights")?)
        }),
        FitlmOption::RobustOpts | FitlmOption::CategoricalVars => {
            return Err(fitlm_invalid(format!(
                "fitlm: option '{}' is not supported yet",
                named.spelling
            )));
        }
    })
}

pub(super) fn parse_fit_name_values(
    values: crate::call::FunctionArgumentView<'_>,
    names: Vec<PreparedOption>,
    index_offset: usize,
    options: &mut FitOptions,
) -> BuiltinResult<()> {
    for named in names {
        let action = match named.predecoded {
            Some(action) => action,
            None => decode_fit_option(&named.name, &values[named.name.index - index_offset + 1])?,
        };
        match action {
            FitlmAction::Intercept(value) => options.intercept = value,
            FitlmAction::ResponseVar(value) => options.response_var = Some(value),
            FitlmAction::PredictorVars(value) => options.predictor_vars = Some(value),
            FitlmAction::VarNames(value) => options.var_names = Some(value),
            FitlmAction::Exclude(value) => options.exclude = Some(value),
            FitlmAction::Weights(value) => options.weights = Some(value),
        }
    }
    Ok(())
}

pub(super) fn parse_predict_options(
    values: crate::call::FunctionArgumentView<'_>,
) -> BuiltinResult<PredictOptions> {
    if !values.len().is_multiple_of(2) {
        return Err(predict_invalid(
            "predict: name-value options must be paired",
        ));
    }
    let mut options = PredictOptions::default();
    let mut index = 0;
    while index < values.len() {
        let name = scalar_text(&values[index])?
            .ok_or_else(|| predict_invalid("predict: option name must be text"))?;
        let value = &values[index + 1];
        if name.eq_ignore_ascii_case("Alpha") {
            let alpha =
                numeric_scalar(value, "Alpha").map_err(|err| predict_invalid(err.message))?;
            if !(0.0..1.0).contains(&alpha) {
                return Err(predict_invalid("predict: Alpha must be between 0 and 1"));
            }
            options.alpha = alpha;
        } else if name.eq_ignore_ascii_case("Prediction") {
            let text = scalar_text(value)?
                .ok_or_else(|| predict_invalid("predict: Prediction must be text"))?;
            if text.eq_ignore_ascii_case("curve") {
                options.prediction = PredictionKind::Curve;
            } else if text.eq_ignore_ascii_case("observation") {
                options.prediction = PredictionKind::Observation;
            } else {
                return Err(predict_invalid(
                    "predict: Prediction must be 'curve' or 'observation'",
                ));
            }
        } else if name.eq_ignore_ascii_case("Simultaneous") {
            let simultaneous =
                bool_scalar(value, "Simultaneous").map_err(|err| predict_invalid(err.message))?;
            if simultaneous {
                return Err(predict_invalid(
                    "predict: Simultaneous=true intervals are not supported yet",
                ));
            }
        } else {
            return Err(predict_invalid(format!("predict: unknown option '{name}'")));
        }
        index += 2;
    }
    Ok(options)
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum ModelSpec {
    Constant,
    Linear,
    Interactions,
    PureQuadratic,
    Quadratic,
}

pub(super) fn parse_model_spec(value: &Value, predictors: usize) -> BuiltinResult<ModelSpec> {
    if let Some(text) = scalar_text(value)? {
        parse_model_spec_text(&text, predictors)
    } else {
        Err(fitlm_invalid(
            "fitlm: numeric terms matrices are not supported yet",
        ))
    }
}

pub(super) fn parse_model_spec_text(text: &str, _predictors: usize) -> BuiltinResult<ModelSpec> {
    if text.eq_ignore_ascii_case("constant") {
        Ok(ModelSpec::Constant)
    } else if text.eq_ignore_ascii_case("linear") {
        Ok(ModelSpec::Linear)
    } else if text.eq_ignore_ascii_case("interactions") {
        Ok(ModelSpec::Interactions)
    } else if text.eq_ignore_ascii_case("purequadratic") {
        Ok(ModelSpec::PureQuadratic)
    } else if text.eq_ignore_ascii_case("quadratic") {
        Ok(ModelSpec::Quadratic)
    } else {
        Err(fitlm_invalid(format!(
            "fitlm: unsupported model specification '{text}'"
        )))
    }
}

pub(super) fn is_known_model_spec(text: &str) -> bool {
    matches!(
        text.to_ascii_lowercase().as_str(),
        "constant" | "linear" | "interactions" | "purequadratic" | "quadratic"
    )
}

pub(super) fn model_terms(spec: ModelSpec, predictors: usize, intercept: bool) -> Vec<Vec<u32>> {
    let mut terms = Vec::new();
    if intercept {
        terms.push(vec![0; predictors]);
    }
    if matches!(
        spec,
        ModelSpec::Linear
            | ModelSpec::Interactions
            | ModelSpec::PureQuadratic
            | ModelSpec::Quadratic
    ) {
        for col in 0..predictors {
            let mut term = vec![0; predictors];
            term[col] = 1;
            terms.push(term);
        }
    }
    if matches!(spec, ModelSpec::Interactions | ModelSpec::Quadratic) {
        for left in 0..predictors {
            for right in left + 1..predictors {
                let mut term = vec![0; predictors];
                term[left] = 1;
                term[right] = 1;
                terms.push(term);
            }
        }
    }
    if matches!(spec, ModelSpec::PureQuadratic | ModelSpec::Quadratic) {
        for col in 0..predictors {
            let mut term = vec![0; predictors];
            term[col] = 2;
            terms.push(term);
        }
    }
    terms
}

pub(super) fn parse_formula_predictors(
    formula: &str,
    table_names: &[String],
    response_name: &str,
    options: &mut FitOptions,
) -> BuiltinResult<Vec<String>> {
    let (_, rhs) = formula
        .split_once('~')
        .ok_or_else(|| fitlm_invalid("fitlm: formula must contain '~'"))?;
    let mut out = Vec::new();
    for raw in formula_rhs_tokens(rhs) {
        let term = raw.trim();
        if term.is_empty() || term == "1" {
            continue;
        }
        if term == "0" || term == "-1" {
            options.intercept = false;
            continue;
        }
        if term.contains(':') || term.contains('*') || term.contains('^') {
            return Err(fitlm_invalid(
                "fitlm: formula interactions and powers are not supported yet",
            ));
        }
        if term == response_name {
            return Err(fitlm_invalid("fitlm: response cannot also be a predictor"));
        }
        if !table_names.iter().any(|name| name == term) {
            return Err(fitlm_invalid(format!(
                "fitlm: formula predictor '{term}' is not in the table"
            )));
        }
        if !out.iter().any(|name| name == term) {
            out.push(term.to_string());
        }
    }
    Ok(out)
}

pub(super) fn parse_formula_terms(
    formula: &str,
    predictor_names: &[String],
    intercept: bool,
) -> BuiltinResult<Vec<Vec<u32>>> {
    let (_, rhs) = formula
        .split_once('~')
        .ok_or_else(|| fitlm_invalid("fitlm: formula must contain '~'"))?;
    let mut terms = Vec::new();
    if intercept {
        terms.push(vec![0; predictor_names.len()]);
    }
    for raw in formula_rhs_tokens(rhs) {
        let term = raw.trim();
        if term.is_empty() || term == "1" || term == "0" || term == "-1" {
            continue;
        }
        let position = predictor_names
            .iter()
            .position(|name| name == term)
            .ok_or_else(|| fitlm_invalid(format!("fitlm: unknown formula term '{term}'")))?;
        let mut exponents = vec![0; predictor_names.len()];
        exponents[position] = 1;
        terms.push(exponents);
    }
    Ok(terms)
}

pub(super) fn formula_rhs_tokens(rhs: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut current = String::new();
    let mut sign = 1_i8;
    let mut saw_token = false;
    for ch in rhs.chars() {
        match ch {
            '+' | '-' => {
                if saw_token || !current.trim().is_empty() {
                    let token = current.trim();
                    if !token.is_empty() {
                        out.push(if sign < 0 {
                            format!("-{token}")
                        } else {
                            token.to_string()
                        });
                    }
                    current.clear();
                    saw_token = false;
                }
                sign = if ch == '-' { -1 } else { 1 };
            }
            _ => {
                if !ch.is_whitespace() {
                    saw_token = true;
                }
                current.push(ch);
            }
        }
    }
    let token = current.trim();
    if !token.is_empty() {
        out.push(if sign < 0 {
            format!("-{token}")
        } else {
            token.to_string()
        });
    }
    out
}

pub(super) fn coefficient_names(terms: &[Vec<u32>], predictor_names: &[String]) -> Vec<String> {
    terms
        .iter()
        .map(|term| {
            if term.iter().all(|power| *power == 0) {
                "(Intercept)".to_string()
            } else {
                let mut factors = Vec::new();
                for (name, power) in predictor_names.iter().zip(term) {
                    match *power {
                        0 => {}
                        1 => factors.push(name.clone()),
                        n => factors.push(format!("{name}^{n}")),
                    }
                }
                factors.join(":")
            }
        })
        .collect()
}

pub(super) fn formula_text(response_name: &str, coefficient_names: &[String]) -> String {
    let rhs = if coefficient_names.is_empty() {
        "0".to_string()
    } else {
        coefficient_names
            .iter()
            .map(|name| {
                if name == "(Intercept)" {
                    "1".to_string()
                } else {
                    name.clone()
                }
            })
            .collect::<Vec<_>>()
            .join(" + ")
    };
    format!("{response_name} ~ {rhs}")
}

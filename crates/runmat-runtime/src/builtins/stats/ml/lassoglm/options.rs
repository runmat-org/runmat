use super::*;

pub(super) fn parse_options<'a>(
    rest: impl IntoIterator<Item = &'a OptionInput>,
) -> BuiltinResult<Options> {
    let mut options = Options::default();
    for input in rest {
        let value = &input.value;
        match input.option {
            LassoglmOption::Alpha => {
                options.alpha = scalar_f64(value, "Alpha")?;
                if !(options.alpha > 0.0 && options.alpha <= 1.0) {
                    return Err(invalid(
                        "lassoglm: Alpha must be a scalar in the interval (0, 1]",
                    ));
                }
            }
            LassoglmOption::Lambda => options.lambda = Some(nonnegative_vector(value, "Lambda")?),
            LassoglmOption::LambdaRatio => {
                options.lambda_ratio = scalar_f64(value, "LambdaRatio")?;
                if !(options.lambda_ratio >= 0.0 && options.lambda_ratio <= 1.0) {
                    return Err(invalid(
                        "lassoglm: LambdaRatio must be a scalar in the interval [0, 1]",
                    ));
                }
            }
            LassoglmOption::NumLambda => {
                options.num_lambda = bounded_usize(value, "NumLambda", MAX_NUM_LAMBDA)?
            }
            LassoglmOption::Standardize => options.standardize = scalar_bool(value, "Standardize")?,
            LassoglmOption::Intercept => options.intercept = scalar_bool(value, "Intercept")?,
            LassoglmOption::Weights => {
                options.weights = Some(nonnegative_weight_vector(value, "Weights")?)
            }
            LassoglmOption::Offset => options.offset = Some(numeric_vector(value, "Offset")?),
            LassoglmOption::BinomialSize => {
                options.binomial_size = Some(positive_vector(value, "BinomialSize")?)
            }
            LassoglmOption::RelTol => {
                options.rel_tol = scalar_f64(value, &input.name)?;
                if !(options.rel_tol > 0.0 && options.rel_tol.is_finite()) {
                    return Err(invalid("lassoglm: RelTol must be positive and finite"));
                }
            }
            LassoglmOption::MaxIter => {
                options.max_iter = bounded_usize(value, "MaxIter", MAX_ITERATIONS)?
            }
            LassoglmOption::Cv => options.cv = parse_cv(value)?,
            LassoglmOption::PredictorNames => {
                options.predictor_names = Some(string_list(value, "PredictorNames")?)
            }
            LassoglmOption::Options => apply_stat_options(value, &mut options)?,
            LassoglmOption::Link => {
                let text = scalar_text(value, "Link")?;
                options.link =
                    Some(LassoglmLink::from_public_name(&text).ok_or_else(|| {
                        invalid(format!("lassoglm: Link '{text}' is not supported"))
                    })?);
            }
            LassoglmOption::EstDisp => {
                let text = scalar_text(value, "EstDisp")?;
                LassoglmEstimateDisp::from_public_name(&text).ok_or_else(|| {
                    invalid("lassoglm: EstDisp values other than 'off' are not supported")
                })?;
            }
            LassoglmOption::McReps => {
                if positive_usize(value, "MCReps")? != 1 {
                    return Err(invalid(
                        "lassoglm: MCReps values greater than 1 are not supported",
                    ));
                }
            }
        }
    }
    Ok(options)
}

pub(super) fn apply_stat_options(value: &Value, options: &mut Options) -> BuiltinResult<()> {
    let Value::Struct(st) = value else {
        if is_empty_numeric(value) {
            return Ok(());
        }
        return Err(invalid("lassoglm: Options must be [] or a statset struct"));
    };
    if let Some(value) = nonempty_field(st, StatOptionField::MaxIter) {
        options.max_iter = bounded_usize(value, "Options.MaxIter", MAX_ITERATIONS)?;
    }
    if let Some(value) = nonempty_field(st, StatOptionField::TolX)
        .or_else(|| nonempty_field(st, StatOptionField::TolFun))
    {
        options.rel_tol = scalar_f64(value, "Options.TolX")?;
        if !(options.rel_tol > 0.0 && options.rel_tol.is_finite()) {
            return Err(invalid(
                "lassoglm: Options tolerance must be positive and finite",
            ));
        }
    }
    if let Some(value) = nonempty_field(st, StatOptionField::UseParallel) {
        if scalar_bool(value, "Options.UseParallel")? {
            return Err(invalid(
                "lassoglm: parallel statset Options are not supported by RunMat's CPU solver",
            ));
        }
    }
    Ok(())
}

pub(super) fn validate_distribution_options(
    distribution: Distribution,
    options: &Options,
) -> BuiltinResult<()> {
    if let Some(link) = options.link {
        let supported = matches!(
            (distribution, link),
            (Distribution::Normal, LassoglmLink::Identity)
                | (Distribution::Binomial, LassoglmLink::Logit)
                | (Distribution::Poisson, LassoglmLink::Log)
        );
        if !supported {
            return Err(invalid(format!(
                "lassoglm: Link '{}' is not supported for {} distribution",
                link.public_name(),
                distribution.public_name()
            )));
        }
    }
    if options.binomial_size.is_some() && distribution != Distribution::Binomial {
        return Err(invalid(
            "lassoglm: BinomialSize is only valid for binomial distribution",
        ));
    }
    Ok(())
}

pub(super) fn nonempty_field(st: &StructValue, field: StatOptionField) -> Option<&Value> {
    st.fields
        .get(field.name())
        .filter(|value| !is_empty_numeric(value))
}

pub(super) fn is_empty_numeric(value: &Value) -> bool {
    matches!(value, Value::Tensor(tensor) if tensor::tensor_element_len(tensor) == 0)
}

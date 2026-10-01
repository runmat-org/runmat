use super::*;

pub(super) fn parse_options<'a>(
    rest: impl IntoIterator<Item = &'a OptionInput>,
) -> BuiltinResult<LassoOptions> {
    let mut options = LassoOptions::default();
    for input in rest {
        let value = &input.value;
        match input.option {
            LassoOption::Alpha => {
                options.alpha = scalar_f64(value, "Alpha")?;
                if !(options.alpha > 0.0 && options.alpha <= 1.0) {
                    return Err(invalid_argument(
                        "lasso: Alpha must be a scalar in the interval (0, 1]",
                    ));
                }
            }
            LassoOption::Lambda => options.lambda = Some(nonnegative_vector(value, "Lambda")?),
            LassoOption::LambdaRatio => {
                options.lambda_ratio = scalar_f64(value, "LambdaRatio")?;
                if !(options.lambda_ratio >= 0.0 && options.lambda_ratio <= 1.0) {
                    return Err(invalid_argument(
                        "lasso: LambdaRatio must be a scalar in the interval [0, 1]",
                    ));
                }
            }
            LassoOption::NumLambda => {
                options.num_lambda = bounded_usize(value, "NumLambda", MAX_NUM_LAMBDA)?
            }
            LassoOption::Standardize => options.standardize = scalar_bool(value, "Standardize")?,
            LassoOption::Intercept => options.intercept = scalar_bool(value, "Intercept")?,
            LassoOption::Weights => {
                options.weights = Some(positive_weight_vector(value, "Weights")?)
            }
            LassoOption::RelTol => {
                options.rel_tol = scalar_f64(value, "RelTol")?;
                if !(options.rel_tol > 0.0 && options.rel_tol.is_finite()) {
                    return Err(invalid_argument(
                        "lasso: RelTol must be positive and finite",
                    ));
                }
            }
            LassoOption::MaxIter => {
                options.max_iter = bounded_usize(value, "MaxIter", MAX_ITERATIONS)?
            }
            LassoOption::Cv => options.cv = parse_cv(value)?,
            LassoOption::PredictorNames => {
                options.predictor_names = Some(string_list(value, "PredictorNames")?)
            }
            LassoOption::DfMax => options.df_max = Some(positive_usize(value, "DFmax")?),
            LassoOption::UseCovariance => {
                scalar_bool(value, "UseCovariance")?;
            }
            LassoOption::McReps => {
                if positive_usize(value, "MCReps")? != 1 {
                    return Err(invalid_argument(
                        "lasso: MCReps values greater than 1 are not supported",
                    ));
                }
            }
        }
    }
    Ok(options)
}

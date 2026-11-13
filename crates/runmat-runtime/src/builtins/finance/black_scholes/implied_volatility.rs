//! Host implied-volatility options, class admission, and root finding.

use super::*;
use runmat_builtins::BLSIMPV_NONDOUBLE_EXTENSION;
use runmat_value::{CellArray, CharArray, StringArray};

#[runmat_macros::runtime_builtin(
    name = "blsimpv",
    context = true,
    binding_variant = "default",
    builtin_path = "crate::builtins::finance::black_scholes::implied_volatility"
)]
pub(super) async fn blsimpv_builtin(
    runtime: Option<crate::context::RuntimeContext>,
    price: Value,
    strike: Value,
    rate: Value,
    time: Value,
    value: Value,
    rest: crate::call::BuiltinArguments,
) -> BuiltinResult<Value> {
    let requested_outputs = crate::output_count::current_output_count();
    if requested_outputs == Some(0) {
        return Ok(Value::output_list(Vec::new()));
    }
    if requested_outputs.is_some_and(|count| count > 1) {
        return Err(error(
            BLSIMPV,
            "blsimpv: too many output arguments",
            &ERROR_INVALID_ARGUMENT,
        ));
    }
    admit_nondouble_pricing_inputs(
        BLSIMPV,
        &BLSIMPV_NONDOUBLE_EXTENSION,
        [&price, &strike, &rate, &time, &value],
    )?;
    if !rest.len().is_multiple_of(2) {
        return Err(error(
            BLSIMPV,
            "blsimpv: name-value options must appear in pairs",
            &ERROR_INVALID_ARGUMENT,
        ));
    }
    {
        let mut pricing_values = vec![&price, &strike, &rate, &time, &value];
        for option in rest.chunks_exact(2) {
            let name = text_scalar(BLSIMPV, &option[0])?;
            let name = name.trim().to_ascii_lowercase();
            if !matches!(
                name.as_str(),
                "yield" | "limit" | "tolerance" | "class" | "method"
            ) {
                return Err(error(
                    BLSIMPV,
                    format!("blsimpv: unsupported option '{name}'"),
                    &ERROR_INVALID_ARGUMENT,
                ));
            }
            if matches!(name.as_str(), "yield" | "limit" | "tolerance") {
                admit_nondouble_pricing_inputs(
                    BLSIMPV,
                    &BLSIMPV_NONDOUBLE_EXTENSION,
                    [&option[1]],
                )?;
            }
            if name == "yield" && !is_empty_value(&option[1]) {
                pricing_values.push(option.get(1).expect("validated option pair"));
            }
            if matches!(name.as_str(), "limit" | "tolerance")
                && !is_empty_value(&option[1])
                && !matches!(&option[1], Value::GpuTensor(_))
            {
                let number = scalar_numeric(BLSIMPV, &option[1], &name)?;
                if !number.is_finite() || number <= 0.0 {
                    return Err(error(
                        BLSIMPV,
                        format!("blsimpv: {name} must be a positive scalar"),
                        &ERROR_INVALID_ARGUMENT,
                    ));
                }
            }
        }
        admission::preflight_numeric_shapes(BLSIMPV, pricing_values)?;
    }
    let gathered_rest = rest
        .try_map_preserving_roots_async(rest.len(), runtime.as_ref(), async |index| {
            let gather = gather_if_needed_async(&rest[index]);
            match runtime.as_ref() {
                Some(runtime) => runtime.scope(gather).await,
                None => gather.await,
            }
            .map_err(|err| with_context(BLSIMPV, err))
        })
        .await?;
    let options = ImpvOptions::parse(gathered_rest.view())?;
    let fixed = [
        price,
        strike,
        rate,
        time,
        value,
        options.yield_value.clone(),
        options.class_value.clone(),
    ];
    let fixed = crate::call::BuiltinArguments::from_borrowed(&fixed, runtime.as_ref())?;
    let mut numeric = fixed
        .try_map_preserving_roots_async(fixed.len(), runtime.as_ref(), async |index| {
            let gather = gather_if_needed_async(&fixed[index]);
            match runtime.as_ref() {
                Some(runtime) => runtime.scope(gather).await,
                None => gather.await,
            }
            .map_err(|err| with_context(BLSIMPV, err))
        })
        .await?;
    let class_value = numeric.pop().expect("class value");
    let inputs = BroadcastInputs::from_values(BLSIMPV, numeric, precision::PricingClass::Double)?;
    let classes = ClassInput::from_value(class_value, &inputs.shape, inputs.len)?;
    let mut out = Vec::with_capacity(inputs.len);
    for idx in 0..inputs.len {
        let params = ImpvParams {
            price: inputs.arg(0, idx),
            strike: inputs.arg(1, idx),
            rate: inputs.arg(2, idx),
            time: inputs.arg(3, idx),
            option_value: inputs.arg(4, idx),
            yield_rate: inputs.arg(5, idx),
            class: classes.value(idx),
            limit: options.effective_limit(),
            tolerance: options.effective_tolerance(),
        };
        out.push(implied_volatility(params));
    }
    output_value(out, inputs.shape, precision::PricingClass::Double, BLSIMPV)
}

#[derive(Clone, Copy)]
enum OptionClass {
    Call,
    Put,
}

struct ClassInput {
    values: Vec<OptionClass>,
    is_scalar: bool,
}

impl ClassInput {
    fn from_value(value: Value, shape: &[usize], len: usize) -> BuiltinResult<Self> {
        match value {
            Value::Bool(flag) => Ok(Self {
                values: vec![class_from_bool(flag)],
                is_scalar: true,
            }),
            Value::LogicalArray(logical) => {
                if logical.len() == 1 {
                    Ok(Self {
                        values: vec![class_from_bool(logical.data[0] != 0)],
                        is_scalar: true,
                    })
                } else if logical_shape(&logical.shape) == shape {
                    Ok(Self {
                        values: logical
                            .data
                            .into_iter()
                            .map(|bit| class_from_bool(bit != 0))
                            .collect(),
                        is_scalar: false,
                    })
                } else {
                    Err(error(
                        BLSIMPV,
                        "blsimpv: Class dimensions must match nonscalar inputs",
                        &ERROR_INVALID_INPUT,
                    ))
                }
            }
            Value::String(runmat_value::StringElement::Text(text)) => Ok(Self {
                values: vec![parse_class_text(&text)?],
                is_scalar: true,
            }),
            Value::StringArray(array) => {
                let text_shape = string_array_shape(&array);
                Self::from_texts(
                    &array
                        .data()
                        .iter()
                        .map(|text| {
                            text.as_str().ok_or_else(|| {
                                error(
                                    BLSIMPV,
                                    "blsimpv: Class must contain nonmissing text",
                                    &ERROR_INVALID_INPUT,
                                )
                            })
                        })
                        .collect::<BuiltinResult<Vec<_>>>()?,
                    text_shape,
                    shape,
                    len,
                )
            }
            Value::CharArray(chars) => {
                if chars.rows == 1 {
                    Ok(Self {
                        values: vec![parse_class_text(
                            &runmat_value::decode_utf16_code_units(chars.data()).map_err(
                                |source| {
                                    let mut error =
                                        error(BLSIMPV, source.to_string(), &ERROR_INVALID_ARGUMENT);
                                    error.source = Some(Box::new(source));
                                    error
                                },
                            )?,
                        )?],
                        is_scalar: true,
                    })
                } else {
                    let texts = char_rows(chars)?;
                    Self::from_texts(&texts, vec![len, 1], shape, len)
                }
            }
            Value::Cell(cell) => Self::from_cell(cell, shape, len),
            other => Err(error(
                BLSIMPV,
                format!("blsimpv: unsupported Class value {other:?}"),
                &ERROR_INVALID_ARGUMENT,
            )),
        }
    }

    fn from_cell(cell: CellArray, shape: &[usize], len: usize) -> BuiltinResult<Self> {
        if cell.data().len() == 1 {
            return Ok(Self {
                values: vec![class_from_cell_value(&cell.data()[0])?],
                is_scalar: true,
            });
        }
        if cell.shape.as_slice() != shape
            && tensor_shape(&cell.shape, cell.rows, cell.cols) != shape
        {
            return Err(error(
                BLSIMPV,
                "blsimpv: Class cell dimensions must match nonscalar inputs",
                &ERROR_INVALID_INPUT,
            ));
        }
        if cell.data().len() != len {
            return Err(error(
                BLSIMPV,
                "blsimpv: Class cell length must match nonscalar inputs",
                &ERROR_INVALID_INPUT,
            ));
        }
        let values = cell
            .data()
            .iter()
            .map(class_from_cell_value)
            .collect::<BuiltinResult<Vec<_>>>()?;
        Ok(Self {
            values,
            is_scalar: false,
        })
    }

    fn from_texts<T: AsRef<str>>(
        texts: &[T],
        text_shape: Vec<usize>,
        shape: &[usize],
        len: usize,
    ) -> BuiltinResult<Self> {
        if texts.len() == 1 {
            return Ok(Self {
                values: vec![parse_class_text(texts[0].as_ref())?],
                is_scalar: true,
            });
        }
        if text_shape != shape || texts.len() != len {
            return Err(error(
                BLSIMPV,
                "blsimpv: Class string dimensions must match nonscalar inputs",
                &ERROR_INVALID_INPUT,
            ));
        }
        let values = texts
            .iter()
            .map(|text| parse_class_text(text.as_ref()))
            .collect::<BuiltinResult<Vec<_>>>()?;
        Ok(Self {
            values,
            is_scalar: false,
        })
    }

    fn value(&self, index: usize) -> OptionClass {
        if self.is_scalar {
            self.values[0]
        } else {
            self.values[index]
        }
    }
}

fn class_from_bool(flag: bool) -> OptionClass {
    if flag {
        OptionClass::Call
    } else {
        OptionClass::Put
    }
}

fn class_from_cell_value(value: &Value) -> BuiltinResult<OptionClass> {
    match value {
        Value::String(runmat_value::StringElement::Text(text)) => parse_class_text(text),
        Value::CharArray(chars) if chars.rows == 1 => parse_class_text(
            &runmat_value::decode_utf16_code_units(chars.data()).map_err(|source| {
                let mut error = error(BLSIMPV, source.to_string(), &ERROR_INVALID_ARGUMENT);
                error.source = Some(Box::new(source));
                error
            })?,
        ),
        Value::Bool(flag) => Ok(class_from_bool(*flag)),
        Value::LogicalArray(logical) if logical.len() == 1 => {
            Ok(class_from_bool(logical.data[0] != 0))
        }
        other => Err(error(
            BLSIMPV,
            format!("blsimpv: invalid Class cell value {other:?}"),
            &ERROR_INVALID_ARGUMENT,
        )),
    }
}

fn parse_class_text(text: &str) -> BuiltinResult<OptionClass> {
    match text.trim().to_ascii_lowercase().as_str() {
        "call" => Ok(OptionClass::Call),
        "put" => Ok(OptionClass::Put),
        other => Err(error(
            BLSIMPV,
            format!("blsimpv: unsupported Class '{other}'"),
            &ERROR_INVALID_ARGUMENT,
        )),
    }
}

fn string_array_shape(array: &StringArray) -> Vec<usize> {
    tensor_shape(&array.shape, array.rows, array.cols)
}

fn char_rows(chars: CharArray) -> BuiltinResult<Vec<String>> {
    let mut rows = Vec::with_capacity(chars.rows);
    for row in 0..chars.rows {
        rows.push(
            runmat_value::decode_utf16_code_units(
                &chars.data()[row * chars.cols..(row + 1) * chars.cols],
            )
            .map_err(|source| {
                let mut error = error(BLSIMPV, source.to_string(), &ERROR_INVALID_ARGUMENT);
                error.source = Some(Box::new(source));
                error
            })?,
        );
    }
    Ok(rows)
}

struct ImpvOptions {
    limit: f64,
    tolerance: f64,
    yield_value: Value,
    class_value: Value,
    method: ImpvMethod,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum ImpvMethod {
    Jackel2016,
    Search,
}

impl ImpvOptions {
    fn parse(args: crate::call::FunctionArgumentView<'_>) -> BuiltinResult<Self> {
        if !args.len().is_multiple_of(2) {
            return Err(error(
                BLSIMPV,
                "blsimpv: name-value options must appear in pairs",
                &ERROR_INVALID_ARGUMENT,
            ));
        }
        let mut options = Self {
            limit: DEFAULT_LIMIT,
            tolerance: DEFAULT_TOLERANCE,
            yield_value: Value::Num(DEFAULT_YIELD),
            class_value: Value::Bool(true),
            method: ImpvMethod::Jackel2016,
        };
        let mut idx = 0usize;
        while idx < args.len() {
            let name = text_scalar(BLSIMPV, &args[idx])?;
            let value = args[idx + 1].clone();
            match name.trim().to_ascii_lowercase().as_str() {
                "limit" => {
                    if is_empty_value(&value) {
                        options.limit = DEFAULT_LIMIT;
                        idx += 2;
                        continue;
                    }
                    options.limit = scalar_numeric(BLSIMPV, &value, "Limit")?;
                    if !options.limit.is_finite() || options.limit <= 0.0 {
                        return Err(error(
                            BLSIMPV,
                            "blsimpv: Limit must be a positive scalar",
                            &ERROR_INVALID_ARGUMENT,
                        ));
                    }
                }
                "tolerance" => {
                    if is_empty_value(&value) {
                        options.tolerance = DEFAULT_TOLERANCE;
                        idx += 2;
                        continue;
                    }
                    options.tolerance = scalar_numeric(BLSIMPV, &value, "Tolerance")?;
                    if !options.tolerance.is_finite() || options.tolerance <= 0.0 {
                        return Err(error(
                            BLSIMPV,
                            "blsimpv: Tolerance must be a positive scalar",
                            &ERROR_INVALID_ARGUMENT,
                        ));
                    }
                }
                "yield" => {
                    options.yield_value = if is_empty_value(&value) {
                        Value::Num(DEFAULT_YIELD)
                    } else {
                        value
                    };
                }
                "class" => {
                    options.class_value = if is_empty_value(&value) {
                        Value::Bool(true)
                    } else {
                        value
                    };
                }
                "method" => {
                    if is_empty_value(&value) {
                        options.method = ImpvMethod::Jackel2016;
                        idx += 2;
                        continue;
                    }
                    let method = text_scalar(BLSIMPV, &value)?;
                    match method.trim().to_ascii_lowercase().as_str() {
                        "search" => options.method = ImpvMethod::Search,
                        "jackel2016" => options.method = ImpvMethod::Jackel2016,
                        other => {
                            return Err(error(
                                BLSIMPV,
                                format!("blsimpv: unsupported Method '{other}'"),
                                &ERROR_INVALID_ARGUMENT,
                            ));
                        }
                    }
                }
                other => {
                    return Err(error(
                        BLSIMPV,
                        format!("blsimpv: unsupported option '{other}'"),
                        &ERROR_INVALID_ARGUMENT,
                    ));
                }
            }
            idx += 2;
        }
        Ok(options)
    }

    fn effective_limit(&self) -> f64 {
        match self.method {
            ImpvMethod::Jackel2016 => DEFAULT_LIMIT,
            ImpvMethod::Search => self.limit,
        }
    }

    fn effective_tolerance(&self) -> f64 {
        match self.method {
            ImpvMethod::Jackel2016 => DEFAULT_TOLERANCE,
            ImpvMethod::Search => self.tolerance,
        }
    }
}

#[derive(Clone, Copy)]
struct ImpvParams {
    price: f64,
    strike: f64,
    rate: f64,
    time: f64,
    option_value: f64,
    yield_rate: f64,
    class: OptionClass,
    limit: f64,
    tolerance: f64,
}

fn implied_volatility(params: ImpvParams) -> f64 {
    if !params.price.is_finite()
        || !params.strike.is_finite()
        || !params.rate.is_finite()
        || !params.time.is_finite()
        || !params.option_value.is_finite()
        || !params.yield_rate.is_finite()
        || params.price < 0.0
        || params.strike <= 0.0
        || params.time < 0.0
        || params.option_value < 0.0
    {
        return f64::NAN;
    }
    let target = params.option_value;
    let value_at_zero = option_value(params, 0.0);
    let value_at_limit = option_value(params, params.limit);
    if (value_at_zero - target).abs() <= params.tolerance {
        return 0.0;
    }
    if !value_at_limit.is_finite() || target < value_at_zero || target > value_at_limit {
        return f64::NAN;
    }
    let mut low = 0.0;
    let mut high = params.limit;
    for _ in 0..MAX_IMPLIED_VOL_ITERATIONS {
        let mid = 0.5 * (low + high);
        let value = option_value(params, mid);
        if !value.is_finite() {
            return f64::NAN;
        }
        let diff = value - target;
        if diff.abs() <= params.tolerance || (high - low).abs() <= params.tolerance {
            return mid;
        }
        if diff < 0.0 {
            low = mid;
        } else {
            high = mid;
        }
    }
    0.5 * (low + high)
}

fn option_value(params: ImpvParams, volatility: f64) -> f64 {
    let result = price_pair(PriceParams {
        price: params.price,
        strike: params.strike,
        rate: params.rate,
        time: params.time,
        volatility,
        yield_rate: params.yield_rate,
    });
    match params.class {
        OptionClass::Call => result.call,
        OptionClass::Put => result.put,
    }
}
#[cfg(test)]
pub(crate) async fn blsimpv_builtin_source(
    price: Value,
    strike: Value,
    rate: Value,
    time: Value,
    value: Value,
    rest: Vec<Value>,
) -> BuiltinResult<Value> {
    blsimpv_builtin(
        None,
        price,
        strike,
        rate,
        time,
        value,
        crate::call::BuiltinArguments::from_borrowed(&rest, None)?,
    )
    .await
}

//! Executor-neutral function-entry and function-output semantics.

use crate::builtins::common::validation;
use crate::object::cell::expand_all_cell_values;
use crate::runtime_error::semantic_error;
use crate::RuntimeError;
use runmat_types::{
    FunctionArgDefaultValue, FunctionArgDim, FunctionArgSizeSpec, FunctionArgValidationLiteral,
    FunctionArgValidator,
};
use runmat_value::{CellArray, IntValue, Tensor, Value};

/// Borrowed semantic contract for one fixed function input.
///
/// Executors project their local/binding identity to a fixed-input index and
/// borrow the shared constraint vocabulary; Runtime owns the behavior.
#[derive(Clone, Copy, Debug)]
pub struct FunctionInputSpec<'a> {
    pub input_index: usize,
    pub size: Option<&'a FunctionArgSizeSpec>,
    pub class_name: Option<&'a str>,
    pub validators: &'a [FunctionArgValidator],
    pub default_value: Option<&'a FunctionArgDefaultValue>,
}

#[derive(Clone, Debug)]
pub struct PreparedFunctionInputs {
    /// One value per fixed input. `None` preserves MATLAB's unassigned state
    /// for an omitted argument that has no default.
    pub fixed: Vec<Option<Value>>,
    pub varargin: Option<CellArray>,
    pub nargin: usize,
}

pub fn prepare_function_inputs(
    function_name: &str,
    supplied: &[Value],
    fixed_input_count: usize,
    accepts_varargin: bool,
    specs: &[FunctionInputSpec<'_>],
) -> Result<PreparedFunctionInputs, RuntimeError> {
    let preparation = prepare_function_inputs_async(
        function_name,
        supplied,
        fixed_input_count,
        accepts_varargin,
        specs,
    );
    #[cfg(not(target_arch = "wasm32"))]
    {
        pollster::block_on(preparation)
    }
    #[cfg(target_arch = "wasm32")]
    {
        futures::executor::block_on(preparation)
    }
}

pub async fn prepare_function_inputs_async(
    function_name: &str,
    supplied: &[Value],
    fixed_input_count: usize,
    accepts_varargin: bool,
    specs: &[FunctionInputSpec<'_>],
) -> Result<PreparedFunctionInputs, RuntimeError> {
    if supplied.len() > fixed_input_count && !accepts_varargin {
        return Err(semantic_error(
            "TooManyInputs",
            format!(
                "semantic function {function_name} expected {fixed_input_count} inputs, got {}",
                supplied.len()
            ),
        ));
    }
    if specs
        .iter()
        .any(|spec| spec.input_index >= fixed_input_count)
    {
        return Err(semantic_error(
            "InvalidInputSlot",
            "function argument slot out of bounds",
        ));
    }

    let mut fixed = vec![None; fixed_input_count];
    for (target, value) in fixed.iter_mut().zip(supplied) {
        *target = Some(value.clone());
    }
    for spec in specs {
        if fixed[spec.input_index].is_none() {
            if let Some(default) = spec.default_value {
                fixed[spec.input_index] = Some(default_value(default));
            }
        }
        if let Some(value) = &fixed[spec.input_index] {
            validate_input(function_name, spec.input_index, value, spec).await?;
        }
    }

    let varargin = if accepts_varargin {
        let values = supplied
            .get(fixed_input_count..)
            .unwrap_or_default()
            .to_vec();
        let columns = values.len();
        Some(
            CellArray::new(values, 1, columns)
                .map_err(|error| semantic_error("VararginPack", format!("varargin: {error}")))?,
        )
    } else {
        None
    };
    Ok(PreparedFunctionInputs {
        fixed,
        varargin,
        nargin: supplied.len(),
    })
}

pub fn collect_function_outputs(
    function_name: &str,
    fixed_outputs: &[Value],
    varargout: Option<&Value>,
    requested_outputs: usize,
) -> Result<Vec<Value>, RuntimeError> {
    if requested_outputs > fixed_outputs.len() && varargout.is_none() {
        return Err(semantic_error(
            "TooManyOutputs",
            format!(
                "semantic function {function_name} expected {} outputs, got {requested_outputs}",
                fixed_outputs.len()
            ),
        ));
    }
    let mut values = fixed_outputs
        .iter()
        .take(requested_outputs)
        .cloned()
        .collect::<Vec<_>>();
    if values.len() < requested_outputs {
        if let Some(varargout) = varargout {
            let expanded = match varargout {
                Value::Cell(cell) => expand_all_cell_values(cell)?,
                _ => Vec::new(),
            };
            let available = expanded.len();
            values.extend(expanded.into_iter().take(requested_outputs - values.len()));
            if values.len() < requested_outputs {
                let needed = requested_outputs - fixed_outputs.len();
                return Err(semantic_error(
                    "VarargoutMismatch",
                    format!(
                        "Function '{function_name}' returned {available} varargout values, {needed} requested"
                    ),
                ));
            }
        }
    }
    values.resize(requested_outputs, Value::Num(0.0));
    Ok(values)
}

fn default_value(default: &FunctionArgDefaultValue) -> Value {
    match default {
        FunctionArgDefaultValue::Number(value) => Value::Num(*value),
        FunctionArgDefaultValue::Integer(value) => Value::Int(IntValue::from(value)),
        FunctionArgDefaultValue::Bool(value) => Value::Bool(*value),
        FunctionArgDefaultValue::String(value) => Value::String(value.clone()),
        FunctionArgDefaultValue::EmptyArray => Value::Tensor(
            Tensor::new(Vec::new(), vec![0, 0]).expect("empty default tensor is always valid"),
        ),
    }
}

async fn validate_input(
    function_name: &str,
    input_index: usize,
    value: &Value,
    spec: &FunctionInputSpec<'_>,
) -> Result<(), RuntimeError> {
    if let Some(size) = spec.size {
        let (rows, columns) = validation::value_shape_2d(value);
        if !dim_matches(&size.rows, rows) || !dim_matches(&size.cols, columns) {
            return Err(semantic_error(
                "ArgumentValidationSize",
                format!(
                    "Function '{function_name}' argument #{} failed size validation",
                    input_index + 1
                ),
            ));
        }
    }
    if let Some(class_name) = spec.class_name {
        if !validation::value_matches_class(value, class_name) {
            return Err(semantic_error(
                "ArgumentValidationClass",
                format!(
                    "Function '{function_name}' argument #{} failed class validation (expected {class_name})",
                    input_index + 1
                ),
            ));
        }
    }
    for validator in spec.validators {
        if !validator_passes(value, validator).await? {
            return Err(semantic_error(
                "ArgumentValidationFunction",
                format!(
                    "Function '{function_name}' argument #{} failed {} validation",
                    input_index + 1,
                    validator_name(validator)
                ),
            ));
        }
    }
    Ok(())
}

fn dim_matches(dim: &FunctionArgDim, actual: usize) -> bool {
    matches!(dim, FunctionArgDim::Any)
        || matches!(dim, FunctionArgDim::Exact(expected) if *expected == actual)
}

async fn validator_passes(
    value: &Value,
    validator: &FunctionArgValidator,
) -> Result<bool, RuntimeError> {
    use FunctionArgValidator as V;
    Ok(match validator {
        V::A(names) => validation::must_be_a(value, names.clone())?,
        V::Column => validation::value_is_column(value),
        V::Finite => {
            validation::ensure_resident_extension(value, "mustBeFinite")?;
            validation::value_is_finite_async(value).await?
        }
        V::Float => validation::value_is_float(value),
        V::Folder => validation::dispatch_validator_async("mustBeFolder", vec![value.clone()])
            .await
            .is_ok(),
        V::File => validation::dispatch_validator_async("mustBeFile", vec![value.clone()])
            .await
            .is_ok(),
        V::NumericOrLogical => validation::value_is_numeric_or_logical(value),
        V::Numeric => validation::value_is_numeric(value),
        V::Text => validation::value_is_text(value),
        V::TextScalar => validation::value_is_text_scalar(value),
        V::NonzeroLengthText => validation::value_is_nonzero_length_text(value),
        V::Nonempty => !validation::value_is_empty(value),
        V::ScalarOrEmpty => validation::value_is_scalar_or_empty(value),
        V::Real => validation::value_is_real_async(value).await?,
        V::Integer => {
            validation::ensure_resident_extension(value, "mustBeInteger")?;
            validation::value_is_integer_async(value).await?
        }
        V::Vector { allow_all_empties } => {
            validation::value_satisfies_vector_validator(value, *allow_all_empties)?
        }
        V::Positive => validation::value_is_positive_async(value).await?,
        V::Negative => validation::value_is_negative_async(value).await?,
        V::Nonnegative => validation::value_is_nonnegative_async(value).await?,
        V::Nonmissing => validation::value_is_nonmissing_async(value).await?,
        V::NonNan => {
            validation::ensure_resident_extension(value, "mustBeNonNan")?;
            validation::value_is_non_nan_async(value).await?
        }
        V::Nonzero => {
            validation::ensure_resident_extension(value, "mustBeNonzero")?;
            validation::value_is_nonzero_async(value).await?
        }
        V::Nonpositive => validation::value_is_nonpositive_async(value).await?,
        V::Nonsparse => {
            validation::dispatch_validator_async("mustBeNonsparse", vec![value.clone()])
                .await
                .is_ok()
        }
        V::Sparse => validation::dispatch_validator_async("mustBeSparse", vec![value.clone()])
            .await
            .is_ok(),
        V::ValidVariableName => validation::isvarname_value(value),
        V::UnderlyingType(names) => {
            validation::value_underlying_type_matches(value, names.clone())?
        }
        V::Member(literals) => {
            let allowed = literals.iter().map(literal_atom).collect::<Vec<_>>();
            validation::value_is_member_atoms_async(value, &allowed).await?
        }
        V::InRange(lower, upper, inclusivity) => {
            validation::value_is_in_range_documented_async(
                value,
                &Value::Num(*lower),
                &Value::Num(*upper),
                validation::RangeInclusivity {
                    lower: inclusivity.lower,
                    upper: inclusivity.upper,
                },
            )
            .await?
        }
        V::GreaterThanOrEqual(threshold) => {
            validation::value_is_greater_than_or_equal_values_async(value, &Value::Num(*threshold))
                .await?
        }
        V::LessThanOrEqual(threshold) => {
            validation::value_is_less_than_or_equal_values_async(value, &Value::Num(*threshold))
                .await?
        }
        V::GreaterThan(threshold) => {
            validation::value_is_greater_than_values_async(value, &Value::Num(*threshold)).await?
        }
        V::LessThan(threshold) => {
            validation::value_is_less_than_values_async(value, &Value::Num(*threshold)).await?
        }
    })
}

fn literal_atom(literal: &FunctionArgValidationLiteral) -> validation::ValidationAtom {
    match literal {
        FunctionArgValidationLiteral::Number(value) => validation::ValidationAtom::Number(*value),
        FunctionArgValidationLiteral::Integer(value) => {
            validation::ValidationAtom::Integer(IntValue::from(value))
        }
        FunctionArgValidationLiteral::Text(value) => {
            validation::ValidationAtom::Text(value.clone())
        }
        FunctionArgValidationLiteral::Bool(value) => validation::ValidationAtom::Bool(*value),
    }
}

fn validator_name(validator: &FunctionArgValidator) -> &'static str {
    use FunctionArgValidator as V;
    match validator {
        V::A(_) => "mustBeA",
        V::Column => "mustBeColumn",
        V::Finite => "mustBeFinite",
        V::Float => "mustBeFloat",
        V::Folder => "mustBeFolder",
        V::File => "mustBeFile",
        V::NumericOrLogical => "mustBeNumericOrLogical",
        V::Numeric => "mustBeNumeric",
        V::Text => "mustBeText",
        V::TextScalar => "mustBeTextScalar",
        V::NonzeroLengthText => "mustBeNonzeroLengthText",
        V::Nonempty => "mustBeNonempty",
        V::ScalarOrEmpty => "mustBeScalarOrEmpty",
        V::Real => "mustBeReal",
        V::Integer => "mustBeInteger",
        V::Vector { .. } => "mustBeVector",
        V::Positive => "mustBePositive",
        V::Negative => "mustBeNegative",
        V::Nonnegative => "mustBeNonnegative",
        V::Nonmissing => "mustBeNonmissing",
        V::NonNan => "mustBeNonNan",
        V::Nonzero => "mustBeNonzero",
        V::Nonpositive => "mustBeNonpositive",
        V::Nonsparse => "mustBeNonsparse",
        V::Sparse => "mustBeSparse",
        V::ValidVariableName => "mustBeValidVariableName",
        V::UnderlyingType(_) => "mustBeUnderlyingType",
        V::Member(_) => "mustBeMember",
        V::InRange(_, _, _) => "mustBeInRange",
        V::GreaterThanOrEqual(_) => "mustBeGreaterThanOrEqual",
        V::LessThanOrEqual(_) => "mustBeLessThanOrEqual",
        V::GreaterThan(_) => "mustBeGreaterThan",
        V::LessThan(_) => "mustBeLessThan",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn prepares_defaults_varargin_and_nargin() {
        let positive = [FunctionArgValidator::Positive];
        let default = FunctionArgDefaultValue::Number(3.0);
        let specs = [FunctionInputSpec {
            input_index: 1,
            size: None,
            class_name: None,
            validators: &positive,
            default_value: Some(&default),
        }];
        let prepared = prepare_function_inputs("sample", &[Value::Num(1.0)], 2, true, &specs)
            .expect("prepare inputs");
        assert_eq!(
            prepared.fixed,
            vec![Some(Value::Num(1.0)), Some(Value::Num(3.0))]
        );
        assert_eq!(prepared.nargin, 1);
        assert_eq!(prepared.varargin.unwrap().data.len(), 0);
    }

    #[test]
    fn preserves_omitted_unassigned_inputs_and_rejects_validation_failures() {
        let positive = [FunctionArgValidator::Positive];
        let specs = [FunctionInputSpec {
            input_index: 0,
            size: None,
            class_name: None,
            validators: &positive,
            default_value: None,
        }];
        assert_eq!(
            prepare_function_inputs("sample", &[], 1, false, &specs)
                .unwrap()
                .fixed,
            vec![None]
        );
        let error =
            prepare_function_inputs("sample", &[Value::Num(-1.0)], 1, false, &specs).unwrap_err();
        assert_eq!(
            error.identifier(),
            Some("RunMat:ArgumentValidationFunction")
        );
    }

    #[test]
    fn collects_fixed_and_variadic_outputs_exactly() {
        let cell = CellArray::new(vec![Value::Num(2.0), Value::Num(3.0)], 1, 2).unwrap();
        assert_eq!(
            collect_function_outputs("sample", &[Value::Num(1.0)], Some(&Value::Cell(cell)), 3,)
                .unwrap(),
            vec![Value::Num(1.0), Value::Num(2.0), Value::Num(3.0)]
        );
    }
}

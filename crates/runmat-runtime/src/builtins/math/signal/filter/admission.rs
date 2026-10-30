//! Admitted filter geometry and optional-argument normalization.
use super::initial_state::InitialState;
use super::inputs::{CoeffInput, CoefficientRole, SignalInput};
use super::*;

pub(super) struct FilterArgs {
    pub(super) coeffs_b: CoeffInput,
    pub(super) coeffs_a: CoeffInput,
    pub(super) signal: SignalInput,
    pub(super) dim_idx: usize,
    pub(super) shape_ext: Vec<usize>,
    pub(super) state_shape: Vec<usize>,
    pub(super) order: usize,
    pub(super) state_len: usize,
    pub(super) leading: usize,
    pub(super) trailing: usize,
    pub(super) channel_count: usize,
    pub(super) initial: InitialState,
    pub(super) is_complex: bool,
}

impl FilterArgs {
    pub(super) async fn parse(
        b: Value,
        a: Value,
        x: Value,
        rest: crate::call::FunctionArgumentView<'_>,
        cooperation: &mut SignalCooperation,
    ) -> BuiltinResult<Self> {
        let (zi_raw, dim_raw) = parse_optional_arguments(rest)?;
        let zi_value = match zi_raw {
            Some(val) if is_empty_placeholder(&val) => None,
            other => other,
        };
        let dim_value = match dim_raw {
            Some(val) if is_empty_placeholder(&val) => None,
            other => other,
        };

        let admitted = preflight::operands(&b, &a, &x, zi_value.as_ref())?;
        let host_dim = match dim_value.as_ref() {
            Some(Value::GpuTensor(_)) | None => None,
            Some(value) => Some(parse_dimension_arg(value).await?),
        };
        preflight::resident_owners(&b, &a, &x, zi_value.as_ref(), dim_value.as_ref())?;

        let signal = SignalInput::from_value(x, cooperation).await?;

        let dim = if let Some(dim_val) = dim_value {
            match host_dim {
                Some(dim) => dim,
                None => parse_dimension_arg(&dim_val).await?,
            }
        } else {
            default_dimension_from_shape(&signal.shape)
        };
        let dim_idx = dim - 1;

        let mut shape_ext = signal.shape.clone();
        if dim > shape_ext.len() {
            shape_ext
                .try_reserve_exact(dim - shape_ext.len())
                .map_err(|source| {
                    filter_error_with_source(
                        &FILTER_ERROR_INVALID_DIMENSION,
                        "dimension extent cannot be represented",
                        build_runtime_error(source.to_string())
                            .with_builtin(BUILTIN_NAME)
                            .with_source(source)
                            .build(),
                    )
                })?;
            shape_ext.extend(std::iter::repeat_n(1, dim - shape_ext.len()));
        }

        let order = admitted.coefficient_len;
        let state_len = order.saturating_sub(1);

        let state_shape =
            runmat_accelerate_api::provider_iir_state_shape(&shape_ext, dim_idx, state_len)
                .map_err(|source| {
                    crate::runtime_error::with_provider_execution_source(
                        filter_error(&FILTER_ERROR_INTERNAL),
                        source.into(),
                    )
                })?;

        let checked_product = |extents: &[usize]| {
            extents
                .iter()
                .copied()
                .try_fold(1usize, usize::checked_mul)
                .ok_or_else(|| filter_error(&FILTER_ERROR_INTERNAL))
        };
        let leading = checked_product(&shape_ext[..dim_idx])?;
        let trailing = checked_product(&shape_ext[dim_idx + 1..])?;
        let channel_count = leading
            .checked_mul(trailing)
            .ok_or_else(|| filter_error(&FILTER_ERROR_INTERNAL))?;
        let expected_states = state_len
            .checked_mul(channel_count)
            .ok_or_else(|| filter_error(&FILTER_ERROR_INVALID_INITIAL_STATE))?;

        let initial = if let Some(zi_val) = zi_value {
            InitialState::from_value(
                zi_val,
                state_len,
                &state_shape,
                expected_states,
                cooperation,
            )
            .await?
        } else {
            InitialState::empty(state_shape.clone())
        };

        let coeffs_b = CoeffInput::from_value(CoefficientRole::Numerator, b, cooperation).await?;
        let coeffs_a = CoeffInput::from_value(CoefficientRole::Denominator, a, cooperation).await?;

        if coeffs_a.len == 0 {
            return Err(filter_error(&FILTER_ERROR_EMPTY_DENOMINATOR));
        }
        if coeffs_b.len == 0 {
            return Err(filter_error(&FILTER_ERROR_EMPTY_NUMERATOR));
        }

        let is_complex =
            coeffs_b.is_complex || coeffs_a.is_complex || signal.is_complex || initial.is_complex;

        Ok(Self {
            coeffs_b,
            coeffs_a,
            signal,
            dim_idx,
            shape_ext,
            state_shape,
            order,
            state_len,
            leading,
            trailing,
            channel_count,
            initial,
            is_complex,
        })
    }
}

fn parse_optional_arguments(
    rest: crate::call::FunctionArgumentView<'_>,
) -> BuiltinResult<(Option<Value>, Option<Value>)> {
    match rest.len() {
        0 => Ok((None, None)),
        1 => Ok((Some(rest[0].clone()), None)),
        2 => Ok((Some(rest[0].clone()), Some(rest[1].clone()))),
        _ => Err(filter_error_with_detail(
            &FILTER_ERROR_ARG_COUNT,
            "(b, a, x [, zi [, dim]])",
        )),
    }
}

fn is_empty_placeholder(value: &Value) -> bool {
    match value {
        Value::Tensor(t) => tensor::tensor_element_len(t) == 0,
        Value::ComplexTensor(t) => tensor::complex_tensor_element_len(t) == 0,
        Value::LogicalArray(l) => l.data.is_empty(),
        Value::StringArray(sa) => sa.data().is_empty(),
        Value::CharArray(ca) => ca.data().is_empty(),
        Value::Cell(cell) => cell.data().is_empty(),
        Value::String(runmat_value::StringElement::Text(s)) => s.is_empty(),
        Value::Struct(st) => st.fields.is_empty(),
        _ => false,
    }
}

pub(super) fn ensure_vector_shape(role: CoefficientRole, shape: &[usize]) -> BuiltinResult<()> {
    let non_singleton = shape.iter().copied().filter(|&d| d > 1).count();
    if non_singleton > 1 {
        Err(filter_error_with_detail(
            &FILTER_ERROR_INVALID_COEFFICIENTS,
            format!(
                "{BUILTIN_NAME}: {} coefficients must be a row or column vector",
                role.label()
            ),
        ))
    } else {
        Ok(())
    }
}

fn default_dimension_from_shape(shape: &[usize]) -> usize {
    if shape.is_empty() {
        1
    } else {
        shape
            .iter()
            .position(|&d| d > 1)
            .map(|idx| idx + 1)
            .unwrap_or(1)
    }
}

pub(super) fn shapes_compatible(expected: &[usize], actual: &[usize]) -> bool {
    let max_len = expected.len().max(actual.len());
    for i in 0..max_len {
        let e = expected.get(i).copied().unwrap_or(1);
        let a = actual.get(i).copied().unwrap_or(1);
        if e != a {
            return false;
        }
    }
    true
}

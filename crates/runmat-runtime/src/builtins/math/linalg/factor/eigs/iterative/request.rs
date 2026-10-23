//! Retained views of iterative controls: validated options affect execution.
use super::super::controls::{ImaginaryAlias, Key, Text};
use super::super::{Sigma, DEFAULT_K};
use crate::builtins::math::linalg::ops::linear_operator::{LinearOperator, OperatorWork};
use crate::{call::BuiltinArguments, context::RuntimeContext, BuiltinResult};
use runmat_value::Value;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum Failure {
    ReplaceNan,
    Keep,
    Drop,
}

pub(super) struct Request<'a> {
    pub n: usize,
    pub b: Option<&'a Value>,
    pub k: usize,
    pub sigma: Sigma,
    pub tolerance: Option<f64>,
    pub iterations: usize,
    pub subspace: usize,
    pub start: Option<&'a Value>,
    pub failure: Failure,
    pub symmetric: bool,
    pub display: bool,
    pub cholesky: bool,
    pub permutation: Option<&'a Value>,
    pub definite: bool,
    pub matrix: Option<LinearOperator<'a>>,
    pub mass: Option<LinearOperator<'a>>,
    pub imaginary_alias: ImaginaryAlias,
    pub cholesky_owner: Option<super::operator::cholesky::Cholesky<'a>>,
}

impl<'a> Request<'a> {
    pub async fn parse(
        a: &'a Value,
        args: &'a BuiltinArguments,
        runtime: Option<&RuntimeContext>,
    ) -> BuiltinResult<Self> {
        let mut work = runtime
            .map(OperatorWork::new)
            .transpose()
            .map_err(|e| super::operator_error(runtime, e))?;
        use super::super::{
            is_empty_matrix, is_generalized_b, matrix_order, parse_k, parse_sigma_in_context,
        };
        let callable = crate::builtins::math::linalg::factor::iterative::controls::is_callable(a);
        let mut cursor = 0;
        let n = if callable {
            let n = parse_k(
                args.first()
                    .ok_or_else(|| super::invalid("function form requires matrix order"))?,
            )?
            .ok_or_else(|| super::invalid("function form requires a positive matrix order"))?;
            cursor += 1;
            n
        } else {
            matrix_order(a)?
        };
        let mut b = None;
        if let Some(value) = args.get(cursor) {
            if is_empty_matrix(value) {
                cursor += 1;
            } else if is_generalized_b(value) {
                if matrix_order(value)? != n {
                    return Err(super::invalid("A and B must have equal order"));
                }
                super::super::ensure_eigs_coefficient_class_enabled(value)?;
                b = Some(value);
                cursor += 1;
            }
        }
        let mut k = DEFAULT_K.min(n);
        if let Some(value) = args.get(cursor) {
            if let Some(value) = parse_k(value)? {
                k = value.min(n);
                cursor += 1;
            }
        }
        let mut sigma = Sigma::LargestAbs;
        let mut imaginary_alias = ImaginaryAlias::Explicit;
        if let Some(value) = args.get(cursor) {
            let parsed = if let Some(text) = Text::of(value) {
                let text = text
                    .trim(&mut work)
                    .await
                    .map_err(|e| super::operator_error(runtime, e))?;
                if super::super::controls::option(text).is_some() {
                    None
                } else {
                    Some(
                        super::super::controls::selector(text)
                            .ok_or_else(|| super::invalid("unsupported sigma selector"))?,
                    )
                }
            } else {
                parse_sigma_in_context(runtime, value)?
                    .map(|sigma| (sigma, ImaginaryAlias::Explicit))
            };
            if let Some((parsed, alias)) = parsed {
                if super::super::is_typed_integer_value(value) {
                    crate::compatibility::ensure_builtin_extension_enabled(
                        &super::super::EIGS_INTEGER_SIGMA_EXTENSION,
                        "eigs",
                    )?;
                }
                sigma = parsed;
                imaginary_alias = alias;
                cursor += 1;
            }
        }
        let matrix = if callable {
            None
        } else {
            Some(match runtime {
                Some(runtime) => LinearOperator::borrow_cooperatively(a, runtime)
                    .await
                    .map_err(|e| super::operator_error(Some(runtime), e))?,
                None => {
                    LinearOperator::borrow_unbounded(a).map_err(|e| super::matrix_error(None, e))?
                }
            })
        };
        let mut request = Self {
            n,
            b,
            k,
            sigma,
            tolerance: None,
            iterations: 300,
            subspace: 0,
            start: None,
            failure: Failure::ReplaceNan,
            symmetric: false,
            display: false,
            cholesky: false,
            permutation: None,
            definite: false,
            matrix,
            mass: None,
            imaginary_alias,
            cholesky_owner: None,
        };
        let options = args.view().range(cursor..).expect("bounded option range");
        if options.len() == 1 && matches!(options.get(0), Some(Value::Struct(_))) {
            let Some(Value::Struct(options)) = options.get(0) else {
                unreachable!()
            };
            for (name, value) in &options.fields {
                let text = Text::Utf8(name)
                    .trim(&mut work)
                    .await
                    .map_err(|e| super::operator_error(runtime, e))?;
                let key = super::super::controls::option(text)
                    .ok_or_else(|| super::invalid("unsupported eigs option"))?;
                request.option(key, value, &mut work, runtime).await?;
            }
        } else {
            if !options.len().is_multiple_of(2) {
                return Err(super::invalid("options must occur in name/value pairs"));
            }
            for index in (0..options.len()).step_by(2) {
                let text = Text::of(options.get(index).expect("bounded option index"))
                    .ok_or_else(|| super::invalid("option names must be text scalars"))?
                    .trim(&mut work)
                    .await
                    .map_err(|e| super::operator_error(runtime, e))?;
                let key = super::super::controls::option(text)
                    .ok_or_else(|| super::invalid("unsupported eigs option"))?;
                request
                    .option(
                        key,
                        options.get(index + 1).expect("bounded option value"),
                        &mut work,
                        runtime,
                    )
                    .await?;
            }
        }
        if let Some(matrix) = request.matrix.as_ref() {
            request.symmetric = matrix
                .is_hermitian(runtime)
                .await
                .map_err(|e| super::operator_error(runtime, e))?;
        }
        if request.subspace == 0 {
            request.subspace = n.min(20.max(k.saturating_mul(2).saturating_add(1)));
        }
        let minimum =
            if request.symmetric && (request.b.is_none() || request.cholesky || request.definite) {
                k.saturating_add(1)
            } else {
                k.saturating_add(2)
            }
            .min(n);
        if request.subspace > n || request.subspace < minimum {
            return Err(super::invalid(
                "SubspaceDimension must be at least k+1 for symmetric problems or k+2 otherwise, bounded by the matrix order",
            ));
        }
        if let Some(start) = request.start {
            if super::super::numeric_vector_len(start, "StartVector")? != n {
                return Err(super::invalid("StartVector length must equal matrix order"));
            }
            let shape = match start {
                Value::Tensor(t) => Some(t.shape()),
                Value::ComplexTensor(t) => Some(t.shape()),
                _ => None,
            };
            if shape.is_some_and(|shape| shape.iter().skip(2).any(|&n| n != 1)) {
                return Err(super::invalid(
                    "StartVector must be a vector, not a paged array",
                ));
            }
        }
        if matches!(request.sigma,Sigma::Near(value) if !value.re.is_finite()||!value.im.is_finite())
        {
            return Err(super::invalid("numeric sigma must be finite"));
        }
        let inverse = matches!(request.sigma, Sigma::Near(_) | Sigma::SmallestAbs);
        if (request.cholesky || request.permutation.is_some()) && (request.b.is_none() || inverse) {
            return Err(super::invalid(
                "Cholesky options require B and a non-inverse text selector",
            ));
        }
        if request.permutation.is_some() && !request.cholesky {
            return Err(super::invalid(
                "CholeskyPermutation requires IsCholesky=true",
            ));
        }
        if request.definite && (!inverse || request.b.is_none()) {
            return Err(super::invalid(
                "IsSymmetricDefinite requires B and numeric sigma or smallestabs",
            ));
        }
        if request.cholesky {
            request.cholesky_owner = Some(
                super::operator::cholesky::Cholesky::preflight(
                    request
                        .b
                        .ok_or_else(|| super::invalid("Cholesky options require B"))?,
                    request.permutation,
                    n,
                    runtime,
                )
                .await?,
            );
        } else if let Some(b) = request.b {
            request.mass = Some(super::operator::borrow(b, runtime).await?);
        }
        if let Some(work) = work.as_mut() {
            work.flush()
                .await
                .map_err(|e| super::operator_error(runtime, e))?;
        }
        Ok(request)
    }
    async fn option(
        &mut self,
        key: Key,
        value: &'a Value,
        work: &mut Option<OperatorWork<'_>>,
        runtime: Option<&RuntimeContext>,
    ) -> BuiltinResult<()> {
        let name = key.name();
        use super::super::{numeric_scalar, parse_bool, positive_integer_option};
        match key {
            Key::Tolerance => {
                let tol = numeric_scalar(value, "Tolerance")?;
                if !tol.is_finite() || tol <= 0.0 {
                    return Err(super::invalid("Tolerance must be positive and finite"));
                }
                self.tolerance = Some(tol);
            }
            Key::Iterations => self.iterations = positive_integer_option(value, "MaxIterations")?,
            Key::Subspace => self.subspace = positive_integer_option(value, "SubspaceDimension")?,
            Key::Start => {
                if super::super::is_typed_integer_value(value) {
                    crate::compatibility::ensure_builtin_extension_enabled(
                        &super::super::EIGS_INTEGER_START_VECTOR_EXTENSION,
                        "eigs",
                    )?;
                }
                self.start = Some(value);
            }
            Key::Failure => {
                let text = Text::of(value)
                    .ok_or_else(|| super::invalid("FailureTreatment must be text"))?
                    .trim(work)
                    .await
                    .map_err(|e| super::operator_error(runtime, e))?;
                self.failure = if text.eq("keep") {
                    Failure::Keep
                } else if text.eq("drop") {
                    Failure::Drop
                } else if text.eq("replacenan") {
                    Failure::ReplaceNan
                } else {
                    return Err(super::invalid("unsupported FailureTreatment"));
                };
            }
            Key::Symmetric => self.symmetric = parse_bool(value, name)?,
            Key::Display => self.display = parse_bool(value, name)?,
            Key::Cholesky => self.cholesky = parse_bool(value, name)?,
            Key::Permutation => self.permutation = Some(value),
            Key::Definite => self.definite = parse_bool(value, name)?,
        }
        Ok(())
    }
}

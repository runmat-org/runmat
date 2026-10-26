//! Scalar input and explicit-endpoint preparation for the shared root solver.
use super::*;
#[derive(Clone, Copy, Debug)]
pub(super) struct Bracket {
    pub(super) a: f64,
    pub(super) b: f64,
    pub(super) fa: f64,
    pub(super) fb: f64,
    pub(super) evals: usize,
    pub(super) function_single: bool,
}

pub(super) async fn initial_bracket(
    function: &Value,
    x: Value,
    options: &FzeroOptions,
    single: bool,
    cooperation: &mut ScalarCooperation,
) -> BuiltinResult<Bracket> {
    match x {
        Value::Tensor(tensor) => match tensor.len() {
            2 => {
                bracket_from_endpoints(
                    function,
                    tensor::tensor_value_f64(&tensor, 0),
                    tensor::tensor_value_f64(&tensor, 1),
                    single,
                    cooperation,
                )
                .await
            }
            1 => {
                expand_bracket(
                    function,
                    tensor::tensor_value_f64(&tensor, 0),
                    options,
                    single,
                    cooperation,
                )
                .await
            }
            _ => Err(fzero_error_with_detail(
                &FZERO_ERROR_INVALID_INPUT,
                "initial point must be a scalar or two-element bracket",
            )),
        },
        Value::Num(n) => expand_bracket(function, n, options, single, cooperation).await,
        Value::Int(i) => expand_bracket(function, i.to_f64(), options, single, cooperation).await,
        Value::Bool(b) => {
            expand_bracket(
                function,
                if b { 1.0 } else { 0.0 },
                options,
                single,
                cooperation,
            )
            .await
        }
        Value::LogicalArray(array) => match array.data.as_slice() {
            [guess] => {
                expand_bracket(
                    function,
                    if *guess != 0 { 1.0 } else { 0.0 },
                    options,
                    single,
                    cooperation,
                )
                .await
            }
            [a, b] => {
                bracket_from_endpoints(
                    function,
                    if *a != 0 { 1.0 } else { 0.0 },
                    if *b != 0 { 1.0 } else { 0.0 },
                    single,
                    cooperation,
                )
                .await
            }
            _ => Err(fzero_error_with_detail(
                &FZERO_ERROR_INVALID_INPUT,
                "initial point must be a scalar or two-element bracket",
            )),
        },
        _ => Err(fzero_error_with_detail(
            &FZERO_ERROR_INVALID_INPUT,
            "initial point must be a scalar or two-element bracket",
        )),
    }
}

pub(super) async fn bracket_from_endpoints(
    function: &Value,
    a: f64,
    b: f64,
    single: bool,
    cooperation: &mut ScalarCooperation,
) -> BuiltinResult<Bracket> {
    cooperation.completed(0).await?;
    if !a.is_finite() || !b.is_finite() || a == b {
        return Err(fzero_error_with_detail(
            &FZERO_ERROR_INVALID_INPUT,
            "bracket endpoints must be finite and distinct",
        ));
    }
    let (fa, fa_single) = call_scalar_function_with_precision_info(
        NAME,
        OptimSolver::Fzero,
        function,
        a,
        single,
        cooperation,
    )
    .await?;
    cooperation.completed(1).await?;
    if fa == 0.0 {
        return Ok(Bracket {
            a,
            b: a,
            fa,
            fb: fa,
            evals: 1,
            function_single: fa_single,
        });
    }
    let (fb, fb_single) = call_scalar_function_with_precision_info(
        NAME,
        OptimSolver::Fzero,
        function,
        b,
        single,
        cooperation,
    )
    .await?;
    cooperation.completed(1).await?;
    if fb == 0.0 || fa.signum() != fb.signum() {
        Ok(Bracket {
            a,
            b,
            fa,
            fb,
            evals: 2,
            function_single: fa_single || fb_single,
        })
    } else {
        Err(fzero_error_with_detail(
            &FZERO_ERROR_INVALID_INPUT,
            "function values at bracket endpoints must differ in sign",
        ))
    }
}

pub(super) async fn expand_bracket(
    function: &Value,
    x0: f64,
    options: &FzeroOptions,
    single: bool,
    cooperation: &mut ScalarCooperation,
) -> BuiltinResult<Bracket> {
    let mut function = InitialFunction {
        function,
        single,
        first_single: None,
    };
    let selected = bracket::expand(&mut function, x0, options.max_fun_evals, cooperation).await?;
    let bracket = selected.numeric;
    Ok(Bracket {
        a: bracket.a,
        b: bracket.b,
        fa: bracket.fa,
        fb: bracket.fb,
        evals: bracket.evals,
        function_single: function.first_single.unwrap_or(false),
    })
}
struct InitialFunction<'a> {
    function: &'a Value,
    single: bool,
    first_single: Option<bool>,
}
impl crate::builtins::math::optim::brent::ZeroFunction for InitialFunction<'_> {
    type Receipt = ();
    async fn evaluate(
        &mut self,
        x: f64,
        cooperation: &mut ScalarCooperation,
    ) -> BuiltinResult<(f64, ())> {
        let value = if self.first_single.is_none() {
            let (value, single) = call_scalar_function_with_precision_info(
                NAME,
                OptimSolver::Fzero,
                self.function,
                x,
                self.single,
                cooperation,
            )
            .await?;
            self.first_single = Some(single);
            value
        } else {
            call_scalar_function_with_precision(
                NAME,
                OptimSolver::Fzero,
                self.function,
                x,
                self.single,
                cooperation,
            )
            .await?
        };
        Ok((value, ()))
    }
}

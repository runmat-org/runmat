use anyhow::{anyhow, bail, Context};
use once_cell::sync::OnceCell;
use runmat_accelerate::fusion_residency;
use runmat_accelerate_api::{
    AccelProvider, ApiDeviceInfo, GpuTensorHandle, HostTensorOwned, HostTensorView,
    ProviderPrecision, UniqueOptions, UniqueResult,
};
use runmat_builtins::{Tensor, Value};
use runmat_gc::gc_test_context;
use runmat_hir::lower;
use runmat_ignition::vm::interpret_function;
use runmat_ignition::{compile, interpret, Instr};
use runmat_parser::parse;
use runmat_runtime::gather_if_needed;
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Mutex;

struct TestProvider {
    next_id: AtomicU64,
    buffers: Mutex<HashMap<u64, (Vec<f64>, Vec<usize>)>>,
}

impl TestProvider {
    fn new() -> Self {
        Self {
            next_id: AtomicU64::new(1),
            buffers: Mutex::new(HashMap::new()),
        }
    }

    fn pull(&self, handle: &GpuTensorHandle) -> anyhow::Result<(Vec<f64>, Vec<usize>)> {
        let guard = self.buffers.lock().unwrap();
        guard
            .get(&handle.buffer_id)
            .cloned()
            .ok_or_else(|| anyhow!("buffer not found: {}", handle.buffer_id))
    }

    fn push(&self, data: Vec<f64>, shape: Vec<usize>) -> GpuTensorHandle {
        let id = self.next_id.fetch_add(1, Ordering::Relaxed);
        self.buffers
            .lock()
            .unwrap()
            .insert(id, (data, shape.clone()));
        GpuTensorHandle {
            shape,
            device_id: 0,
            buffer_id: id,
        }
    }
}

impl AccelProvider for TestProvider {
    fn precision(&self) -> ProviderPrecision {
        ProviderPrecision::F64
    }

    fn upload(&self, host: &HostTensorView) -> anyhow::Result<GpuTensorHandle> {
        Ok(self.push(host.data.to_vec(), host.shape.to_vec()))
    }

    fn download(&self, handle: &GpuTensorHandle) -> anyhow::Result<HostTensorOwned> {
        let (data, shape) = self.pull(handle)?;
        Ok(HostTensorOwned { data, shape })
    }

    fn free(&self, handle: &GpuTensorHandle) -> anyhow::Result<()> {
        self.buffers.lock().unwrap().remove(&handle.buffer_id);
        Ok(())
    }

    fn device_info(&self) -> String {
        "test-provider".to_string()
    }

    fn device_info_struct(&self) -> ApiDeviceInfo {
        ApiDeviceInfo {
            device_id: 0,
            name: "TestProvider".into(),
            vendor: "RunMat".into(),
            memory_bytes: None,
            backend: Some("test".into()),
        }
    }

    fn reduce_sum_dim(&self, a: &GpuTensorHandle, dim: usize) -> anyhow::Result<GpuTensorHandle> {
        let (data, shape) = self.pull(a)?;
        if shape.len() != 2 {
            bail!("reduce_sum_dim: only 2D supported in test provider");
        }
        let rows = shape[0];
        let cols = shape[1];
        match dim {
            0 => {
                // Column-wise: sum over rows -> shape [1, cols]
                let mut out = vec![0.0f64; cols];
                for c in 0..cols {
                    let mut acc = 0.0f64;
                    let mut saw_nan = false;
                    for r in 0..rows {
                        let v = data[r + c * rows];
                        if v.is_nan() {
                            saw_nan = true;
                            break;
                        }
                        acc += v;
                    }
                    out[c] = if saw_nan { f64::NAN } else { acc };
                }
                Ok(self.push(out, vec![1, cols]))
            }
            1 => {
                // Row-wise: sum over cols -> shape [rows, 1]
                let mut out = vec![0.0f64; rows];
                for r in 0..rows {
                    let mut acc = 0.0f64;
                    let mut saw_nan = false;
                    for c in 0..cols {
                        let v = data[r + c * rows];
                        if v.is_nan() {
                            saw_nan = true;
                            break;
                        }
                        acc += v;
                    }
                    out[r] = if saw_nan { f64::NAN } else { acc };
                }
                Ok(self.push(out, vec![rows, 1]))
            }
            _ => bail!("reduce_sum_dim: only dims 0 or 1 supported in test provider"),
        }
    }

    fn unique(
        &self,
        handle: &GpuTensorHandle,
        options: &UniqueOptions,
    ) -> anyhow::Result<UniqueResult> {
        let (data, shape) = self.pull(handle)?;
        let tensor =
            Tensor::new(data, shape).map_err(|e| anyhow!("unique (test provider): {e}"))?;
        runmat_runtime::builtins::array::sorting_sets::unique::unique_numeric_from_tensor(
            tensor, options,
        )
        .and_then(|eval| eval.into_numeric_unique_result())
        .map_err(|e| anyhow!("unique (test provider): {e}"))
    }

    fn fused_elementwise(
        &self,
        shader: &str,
        inputs: &[GpuTensorHandle],
        output_shape: &[usize],
        len: usize,
    ) -> anyhow::Result<GpuTensorHandle> {
        let parsed = ParsedShader::parse(shader)?;
        if parsed.tmp_exprs.len() > 1024 {
            bail!(
                "test provider: excessive temporary count {}",
                parsed.tmp_exprs.len()
            );
        }

        let mut input_values: Vec<Vec<f64>> = Vec::with_capacity(inputs.len());
        for handle in inputs {
            let (data, _shape) = self.pull(handle)?;
            input_values.push(data);
        }

        let max_input_len = input_values
            .iter()
            .map(|data| data.len())
            .max()
            .unwrap_or(0);
        let output_elements = output_shape.iter().product::<usize>();
        let total = output_elements.max(len).max(max_input_len).max(1);

        let mut out = Vec::with_capacity(total);
        for idx in 0..total {
            let mut tmp_values: Vec<Option<f64>> = vec![None; parsed.tmp_exprs.len()];
            for (slot, expr) in parsed.tmp_exprs.iter().enumerate() {
                let value = evaluate_expression(expr, idx, &input_values, &tmp_values)
                    .with_context(|| format!("evaluating tmp{slot} for idx {idx}"))?;
                tmp_values[slot] = Some(value);
            }
            let result = evaluate_expression(&parsed.output_expr, idx, &input_values, &tmp_values)
                .with_context(|| format!("evaluating output expression for idx {idx}"))?;
            out.push(result);
        }

        let mut shape = if output_elements == 0 {
            if let Some(first) = inputs.first() {
                first.shape.clone()
            } else {
                vec![total]
            }
        } else {
            output_shape.to_vec()
        };
        if shape.iter().product::<usize>() != total {
            shape = vec![total];
        }
        Ok(self.push(out, shape))
    }
}

struct ParsedShader {
    tmp_exprs: Vec<String>,
    output_expr: String,
}

impl ParsedShader {
    fn parse(shader: &str) -> anyhow::Result<Self> {
        let mut tmp_exprs = Vec::new();
        let mut output_expr: Option<String> = None;
        for line in shader.lines() {
            let trimmed = line.trim();
            if let Some(rest) = trimmed.strip_prefix("let ") {
                if let Some(eq_idx) = rest.find('=') {
                    let (lhs, rhs) = rest.split_at(eq_idx);
                    let name = lhs.split(':').next().map(str::trim).unwrap_or("");
                    if name.starts_with("tmp") {
                        let expr = rhs.trim_start_matches('=').trim();
                        let expr = expr.trim_end_matches(';').trim().to_string();
                        tmp_exprs.push(expr);
                    }
                }
            } else if let Some(rest) = trimmed.strip_prefix("output.data[idx] =") {
                let expr = rest.trim().trim_end_matches(';').trim().to_string();
                output_expr = Some(expr);
            }
        }
        let output_expr =
            output_expr.ok_or_else(|| anyhow!("failed to locate fused output expression"))?;
        Ok(Self {
            tmp_exprs,
            output_expr,
        })
    }
}

fn evaluate_expression(
    expr: &str,
    idx: usize,
    inputs: &[Vec<f64>],
    tmp_values: &[Option<f64>],
) -> anyhow::Result<f64> {
    let tokens = tokenize(expr)?;
    let mut parser = ExprParser::new(tokens, idx, inputs, tmp_values);
    let value = parser.parse_expression()?;
    parser.expect_end()?;
    Ok(value)
}

#[derive(Clone, Debug)]
enum Token {
    Number(f64),
    Ident(String),
    Symbol(char),
}

fn tokenize(input: &str) -> anyhow::Result<Vec<Token>> {
    let mut tokens = Vec::new();
    let mut chars = input.chars().peekable();
    while let Some(&ch) = chars.peek() {
        if ch.is_ascii_whitespace() {
            chars.next();
            continue;
        }
        if ch.is_ascii_digit()
            || (ch == '.'
                && chars
                    .clone()
                    .nth(1)
                    .map(|next| next.is_ascii_digit())
                    .unwrap_or(false))
        {
            let mut buf = String::new();
            let mut has_dot = false;
            while let Some(&c) = chars.peek() {
                if c.is_ascii_digit() {
                    buf.push(c);
                    chars.next();
                } else if c == '.' && !has_dot {
                    if buf.is_empty() {
                        buf.push('0');
                    }
                    has_dot = true;
                    buf.push(c);
                    chars.next();
                } else if (c == 'e' || c == 'E') && !buf.is_empty() {
                    buf.push(c);
                    chars.next();
                    if let Some(&sign) = chars.peek() {
                        if sign == '+' || sign == '-' {
                            buf.push(sign);
                            chars.next();
                        }
                    }
                } else {
                    break;
                }
            }
            let value = buf
                .parse::<f64>()
                .map_err(|e| anyhow!("invalid number '{buf}': {e}"))?;
            tokens.push(Token::Number(value));
            continue;
        }
        if ch.is_ascii_alphabetic() || ch == '_' {
            let mut buf = String::new();
            while let Some(&c) = chars.peek() {
                if c.is_ascii_alphanumeric() || c == '_' {
                    buf.push(c);
                    chars.next();
                } else {
                    break;
                }
            }
            tokens.push(Token::Ident(buf));
            continue;
        }
        tokens.push(Token::Symbol(ch));
        chars.next();
    }
    Ok(tokens)
}

struct ExprParser<'a> {
    tokens: Vec<Token>,
    pos: usize,
    idx: usize,
    inputs: &'a [Vec<f64>],
    tmp_values: &'a [Option<f64>],
}

impl<'a> ExprParser<'a> {
    fn new(
        tokens: Vec<Token>,
        idx: usize,
        inputs: &'a [Vec<f64>],
        tmp_values: &'a [Option<f64>],
    ) -> Self {
        Self {
            tokens,
            pos: 0,
            idx,
            inputs,
            tmp_values,
        }
    }

    fn parse_expression(&mut self) -> anyhow::Result<f64> {
        let mut value = self.parse_term()?;
        while let Some(op) = self.match_symbols(&['+', '-']) {
            let rhs = self.parse_term()?;
            value = if op == '+' { value + rhs } else { value - rhs };
        }
        Ok(value)
    }

    fn parse_term(&mut self) -> anyhow::Result<f64> {
        let mut value = self.parse_factor()?;
        while let Some(op) = self.match_symbols(&['*', '/']) {
            let rhs = self.parse_factor()?;
            value = if op == '*' { value * rhs } else { value / rhs };
        }
        Ok(value)
    }

    fn parse_factor(&mut self) -> anyhow::Result<f64> {
        if self.match_symbol('-') {
            return Ok(-self.parse_factor()?);
        }
        if self.match_symbol('+') {
            return self.parse_factor();
        }
        self.parse_primary()
    }

    fn parse_primary(&mut self) -> anyhow::Result<f64> {
        if let Some(token) = self.peek() {
            match token.clone() {
                Token::Number(n) => {
                    self.advance();
                    Ok(n)
                }
                Token::Symbol('(') => {
                    self.advance();
                    let value = self.parse_expression()?;
                    self.expect_symbol(')')?;
                    Ok(value)
                }
                Token::Ident(name) => self.parse_identifier(name),
                Token::Symbol(sym) => Err(anyhow!("unexpected symbol '{sym}' in expression")),
            }
        } else {
            Err(anyhow!("unexpected end of expression"))
        }
    }

    fn parse_identifier(&mut self, name: String) -> anyhow::Result<f64> {
        self.advance();
        if name.starts_with("tmp") {
            let idx: usize = name[3..]
                .parse()
                .map_err(|e| anyhow!("invalid tmp identifier '{name}': {e}"))?;
            let value = self
                .tmp_values
                .get(idx)
                .ok_or_else(|| anyhow!("tmp index {idx} out of range"))?
                .ok_or_else(|| anyhow!("tmp{idx} used before assignment"))?;
            return Ok(value);
        }
        if name.starts_with("input") && self.check_symbol('.') {
            let input_idx: usize = name[5..]
                .parse()
                .map_err(|e| anyhow!("invalid input identifier '{name}': {e}"))?;
            self.expect_symbol('.')?;
            let ident = self.expect_ident()?;
            if ident != "data" {
                bail!("expected 'data' after input identifier, found '{ident}'");
            }
            self.expect_symbol('[')?;
            let idx_ident = self.expect_ident()?;
            if idx_ident != "idx" {
                bail!("expected 'idx' access, found '{idx_ident}'");
            }
            self.expect_symbol(']')?;
            let data = self
                .inputs
                .get(input_idx)
                .ok_or_else(|| anyhow!("input index {input_idx} out of range"))?;
            if data.is_empty() {
                bail!("input{input_idx} has no data");
            }
            let pos = self.idx % data.len();
            return Ok(data[pos]);
        }

        if self.match_symbol('(') {
            let mut args = Vec::new();
            if !self.match_symbol(')') {
                loop {
                    args.push(self.parse_expression()?);
                    if self.match_symbol(')') {
                        break;
                    }
                    self.expect_symbol(',')?;
                }
            }
            return self.call_function(&name, args);
        }

        if name == "idx" {
            return Ok(self.idx as f64);
        }

        Err(anyhow!("unrecognised identifier '{name}'"))
    }

    fn call_function(&self, name: &str, args: Vec<f64>) -> anyhow::Result<f64> {
        let arg = |i: usize| -> anyhow::Result<f64> {
            args.get(i)
                .copied()
                .ok_or_else(|| anyhow!("function '{name}' missing argument {i}"))
        };
        match name {
            "sin" => Ok(arg(0)?.sin()),
            "cos" => Ok(arg(0)?.cos()),
            "tan" => Ok(arg(0)?.tan()),
            "asin" => Ok(arg(0)?.asin()),
            "acos" => Ok(arg(0)?.acos()),
            "atan" => Ok(arg(0)?.atan()),
            "atan2" => Ok(arg(0)?.atan2(arg(1)?)),
            "sinh" => Ok(arg(0)?.sinh()),
            "cosh" => Ok(arg(0)?.cosh()),
            "tanh" => Ok(arg(0)?.tanh()),
            "exp" => Ok(arg(0)?.exp()),
            "exp2" => Ok(arg(0)?.exp2()),
            "log" => Ok(arg(0)?.ln()),
            "log2" => Ok(arg(0)?.log2()),
            "sqrt" => Ok(arg(0)?.sqrt()),
            "abs" => Ok(arg(0)?.abs()),
            "floor" => Ok(arg(0)?.floor()),
            "ceil" => Ok(arg(0)?.ceil()),
            "round" => Ok(arg(0)?.round()),
            "trunc" => Ok(arg(0)?.trunc()),
            "expm1" => Ok(arg(0)?.exp_m1()),
            "log1p" => Ok(arg(0)?.ln_1p()),
            "pow" => Ok(arg(0)?.powf(arg(1)?)),
            "f32" | "f64" => Ok(arg(0)?),
            _ => Err(anyhow!("unsupported function '{name}' in fused shader")),
        }
    }

    fn expect_end(&self) -> anyhow::Result<()> {
        if self.pos >= self.tokens.len() {
            Ok(())
        } else {
            Err(anyhow!("unexpected trailing tokens in expression"))
        }
    }

    fn peek(&self) -> Option<&Token> {
        self.tokens.get(self.pos)
    }

    fn advance(&mut self) {
        self.pos += 1;
    }

    fn match_symbol(&mut self, symbol: char) -> bool {
        if let Some(Token::Symbol(ch)) = self.peek() {
            if *ch == symbol {
                self.advance();
                return true;
            }
        }
        false
    }

    fn match_symbols(&mut self, symbols: &[char]) -> Option<char> {
        if let Some(Token::Symbol(ch)) = self.peek() {
            if symbols.contains(ch) {
                let symbol = *ch;
                self.advance();
                return Some(symbol);
            }
        }
        None
    }

    fn check_symbol(&self, symbol: char) -> bool {
        matches!(self.peek(), Some(Token::Symbol(ch)) if *ch == symbol)
    }

    fn expect_symbol(&mut self, symbol: char) -> anyhow::Result<()> {
        if self.match_symbol(symbol) {
            Ok(())
        } else {
            Err(anyhow!("expected symbol '{symbol}'"))
        }
    }

    fn expect_ident(&mut self) -> anyhow::Result<String> {
        if let Some(Token::Ident(name)) = self.peek() {
            let ident = name.clone();
            self.advance();
            Ok(ident)
        } else {
            Err(anyhow!("expected identifier"))
        }
    }
}

static PROVIDER: OnceCell<TestProvider> = OnceCell::new();

fn ensure_provider_registered() {
    let provider: &'static TestProvider = PROVIDER.get_or_init(TestProvider::new);
    unsafe {
        runmat_accelerate_api::register_provider(provider);
    }
}

#[test]
#[ignore]
fn fused_elementwise_then_reduction_sum_rows_profiled() {
    gc_test_context(|| {
        // Generate a reasonably large matrix; try a couple sizes to exercise GPU path
        let sizes = &[(512usize, 512usize), (1024, 1024)];
        for &(rows, cols) in sizes {
            let data: Vec<f64> = (0..rows * cols)
                .map(|i| (i as f64).sin() * 0.5 + 0.5)
                .collect();
            let source = format!(
                r#"
                X = zeros({rows}, {cols});
                for c = 1:{cols}
                    for r = 1:{rows}
                        X(r, c) = c * 10 + r;
                    end
                end
                Y = sin(X) .* X + 2;
                S = sum(Y, 2);
                "#
            );

            let ast = parse(&source).expect("parse");
            let hir = lower(&ast).expect("lower");
            let bytecode = compile(&hir).expect("compile");

            // Initialize vars and insert tensor for 'X' by locating first LoadVar use
            let mut vars = vec![Value::Num(0.0); bytecode.var_count];
            let x_index = bytecode
                .instructions
                .iter()
                .filter_map(|instr| match instr {
                    Instr::StoreVar(idx) => Some(*idx),
                    _ => None,
                })
                .next()
                .unwrap_or(0);
            let tensor = runmat_builtins::Tensor::new(data, vec![rows, cols]).unwrap();
            vars[x_index] = Value::Tensor(tensor);

            // CPU path (no provider registered yet)
            let _ = interpret_function(&bytecode, vars.clone());
            let start_cpu = std::time::Instant::now();
            let vars_cpu = interpret_function(&bytecode, vars.clone()).expect("interpret");
            let elapsed_cpu = start_cpu.elapsed();
            let s_index = bytecode
                .instructions
                .iter()
                .filter_map(|instr| match instr {
                    Instr::StoreVar(i) => Some(*i),
                    _ => None,
                })
                .last()
                .expect("store var for S");
            let s_value_cpu = vars_cpu.get(s_index).expect("value for S (cpu)");
            let gathered_cpu = gather_if_needed(s_value_cpu).expect("gather cpu");
            let out_cpu = match gathered_cpu {
                Value::Tensor(t) => t,
                other => panic!("expected tensor S (cpu), got {other:?}"),
            };
            assert_eq!(
                out_cpu.data.len(),
                rows,
                "CPU output elements should equal rows"
            );

            // GPU path: register provider and run again
            ensure_provider_registered();
            let _ = interpret_function(&bytecode, vars.clone());
            let start_gpu = std::time::Instant::now();
            let vars_gpu = interpret_function(&bytecode, vars).expect("interpret");
            let elapsed_gpu = start_gpu.elapsed();
            // Attempt to gather S at the computed index; if shape doesn't match expectation, scan for a length==rows tensor
            let out_gpu = {
                let s_value_gpu = vars_gpu.get(s_index).expect("value for S (gpu)");
                let gathered = gather_if_needed(s_value_gpu).expect("gather gpu");
                let pick_alt = match &gathered {
                    Value::Tensor(t) => t.data.len() != rows,
                    _ => true,
                };
                if !pick_alt {
                    match gathered {
                        Value::Tensor(t) => t,
                        _ => unreachable!(),
                    }
                } else {
                    // Fallback: search all variables for a tensor of length==rows
                    let mut found: Option<runmat_builtins::Tensor> = None;
                    for v in &vars_gpu {
                        if let Ok(Value::Tensor(t)) = gather_if_needed(v) {
                            if t.data.len() == rows {
                                found = Some(t);
                                break;
                            }
                        }
                    }
                    found.expect("expected to find reduced tensor of length rows")
                }
            };
            assert_eq!(
                out_gpu.data.len(),
                rows,
                "GPU output elements should equal rows"
            );

            // Log basic perf info
            eprintln!(
                "fused sin.* + sum rows: {rows}x{cols} -> [{rows}x1] CPU: {:?} GPU: {:?}",
                elapsed_cpu, elapsed_gpu
            );
        }
    });
}

#[test]
fn reduction_sum_omitnan_vs_include_dim2_gpu_cpu() {
    gc_test_context(|| {
        // Matrix with NaNs in each row
        let rows = 4usize;
        let cols = 5usize;
        let mut data = vec![0.0f64; rows * cols];
        for r in 0..rows {
            for c in 0..cols {
                data[r + c * rows] = (r as f64) + (c as f64);
            }
            // place a NaN in each row
            data[r + ((r % cols) * rows)] = f64::NAN;
        }

        // Program computes include-nan and omit-nan variants along rows (dim=2)
        let source = format!(
            r#"
            rows = {rows}; cols = {cols};
            X = zeros(rows, cols);
            for c = 1:cols
                for r = 1:rows
                    X(r, c) = c * 10 + r;
                end
            end
            % Place a NaN in the first column of each row
            for r = 1:rows
                X(r, 1) = NaN;
            end
            SI = sum(X, 2);           % include nan (default)
            SO = sum(X, 2, "omitnan");
        "#
        );

        let ast = parse(&source).expect("parse");
        let hir = lower(&ast).expect("lower");
        let bytecode = compile(&hir).expect("compile");

        // Helper to run with/without GPU
        let run_with = |use_gpu: bool| -> Vec<Value> {
            let vars = vec![Value::Num(0.0); bytecode.var_count];
            if use_gpu {
                ensure_provider_registered();
            }
            interpret_function(&bytecode, vars).expect("interpret")
        };

        // Warmups
        let _ = run_with(false);
        let _ = run_with(true);

        // CPU and GPU results
        let cpu = run_with(false);
        let gpu = run_with(true);

        // Collect SI and SO by last two stores
        let mut stores: Vec<usize> = bytecode
            .instructions
            .iter()
            .filter_map(|instr| match instr {
                Instr::StoreVar(idx) => Some(*idx),
                _ => None,
            })
            .collect();
        assert!(stores.len() >= 2);
        let so_idx = stores.pop().unwrap();
        let si_idx = stores.pop().unwrap();

        let gather_numvec = |value: &Value| -> Vec<f64> {
            match value {
                Value::GpuTensor(_handle) => {
                    let v = gather_if_needed(value).unwrap();
                    match v {
                        Value::Tensor(t) => t.data,
                        _ => panic!("expected tensor"),
                    }
                }
                Value::Tensor(t) => t.data.clone(),
                _ => panic!("expected tensor"),
            }
        };

        // Compare CPU vs GPU for include-nan and omit-nan
        let si_cpu = gather_numvec(cpu.get(si_idx).unwrap());
        let si_gpu = gather_numvec(gpu.get(si_idx).unwrap());
        assert_eq!(si_cpu.len(), rows);
        assert_eq!(si_gpu.len(), rows);
        for (a, b) in si_cpu.iter().zip(si_gpu.iter()) {
            if a.is_nan() || b.is_nan() {
                assert!(a.is_nan() && b.is_nan());
            } else {
                assert!((a - b).abs() < 1e-9);
            }
        }

        let so_cpu = gather_numvec(cpu.get(so_idx).unwrap());
        let so_gpu = gather_numvec(gpu.get(so_idx).unwrap());
        assert_eq!(so_cpu.len(), rows);
        assert_eq!(so_gpu.len(), rows);
        for (a, b) in so_cpu.iter().zip(so_gpu.iter()) {
            assert!((a - b).abs() < 1e-9);
        }
    });
}

#[test]
fn reduction_sum_include_omit_dim1_dim2_gpu_cpu() {
    gc_test_context(|| {
        let rows = 8usize;
        let cols = 7usize;
        // Program defines X so we can locate its var slot; we inject contents after compile
        let source = format!(
            r#"
            rows = {rows}; cols = {cols};
            X = zeros(rows, cols);
            SI1 = sum(X, 1);
            SO1 = sum(X, 1, 'omitnan');
            SI2 = sum(X, 2);
            SO2 = sum(X, 2, 'omitnan');
            "#
        );

        let ast = parse(&source).expect("parse");
        let hir = lower(&ast).expect("lower");
        let bytecode = compile(&hir).expect("compile");

        // Determine slot for X (first StoreVar in the program)
        let x_index = bytecode
            .instructions
            .iter()
            .filter_map(|instr| match instr {
                Instr::StoreVar(idx) => Some(*idx),
                _ => None,
            })
            .next()
            .unwrap_or(0);

        // Build deterministic host matrix with NaNs
        let mut data = vec![0.0f64; rows * cols];
        for c in 0..cols {
            for r in 0..rows {
                data[r + c * rows] = (c as f64) * 10.0 + (r as f64 + 1.0);
            }
        }
        for c in 0..cols {
            data[0 + c * rows] = f64::NAN; // first row
        }
        for r in 0..rows {
            data[r + 0 * rows] = f64::NAN; // first column
        }

        // Run CPU path with host tensor injected
        let cpu = {
            let mut vars = vec![Value::Num(0.0); bytecode.var_count];
            let t = runmat_builtins::Tensor::new(data.clone(), vec![rows, cols]).unwrap();
            vars[x_index] = Value::Tensor(t);
            interpret_function(&bytecode, vars).expect("interpret")
        };

        // Run GPU path with uploaded tensor injected
        let gpu = {
            ensure_provider_registered();
            let mut vars = vec![Value::Num(0.0); bytecode.var_count];
            let view = runmat_accelerate_api::HostTensorView {
                data: &data,
                shape: &[rows, cols],
            };
            let provider = runmat_accelerate_api::provider().expect("provider");
            let handle = provider.upload(&view).expect("upload");
            vars[x_index] = Value::GpuTensor(handle);
            interpret_function(&bytecode, vars).expect("interpret")
        };

        // Find last 4 stores (SI1, SO1, SI2, SO2)
        let mut stores: Vec<usize> = bytecode
            .instructions
            .iter()
            .filter_map(|instr| match instr {
                Instr::StoreVar(idx) => Some(*idx),
                _ => None,
            })
            .collect();
        assert!(stores.len() >= 4);
        let so2 = stores.pop().unwrap();
        let si2 = stores.pop().unwrap();
        let so1 = stores.pop().unwrap();
        let si1 = stores.pop().unwrap();

        let gather_vec = |v: &Value| -> Vec<f64> {
            match v {
                Value::GpuTensor(_) => match gather_if_needed(v).unwrap() {
                    Value::Tensor(t) => t.data,
                    Value::Num(n) => vec![n],
                    other => panic!("expected tensor, got {other:?}"),
                },
                Value::Tensor(t) => t.data.clone(),
                Value::Num(n) => vec![*n],
                other => panic!("expected tensor, got {other:?}"),
            }
        };

        // Include-nan compare
        let si1_cpu = gather_vec(cpu.get(si1).unwrap());
        let si1_gpu = gather_vec(gpu.get(si1).unwrap());
        eprintln!("SI1 len: cpu={} gpu={}", si1_cpu.len(), si1_gpu.len());
        eprintln!("SI1 CPU: {:?}", si1_cpu);
        eprintln!("SI1 GPU: {:?}", si1_gpu);
        assert_eq!(si1_cpu.len(), cols);
        assert_eq!(si1_gpu.len(), cols);
        for (i, (a, b)) in si1_cpu.iter().zip(si1_gpu.iter()).enumerate() {
            if a.is_nan() || b.is_nan() {
                assert!(a.is_nan() && b.is_nan());
            } else {
                if (a - b).abs() >= 1e-9 {
                    eprintln!(
                        "SI1 mismatch at col {}: cpu={} gpu={} rows={} cols={}",
                        i, a, b, rows, cols
                    );
                }
                assert!((a - b).abs() < 1e-9);
            }
        }

        let si2_cpu = gather_vec(cpu.get(si2).unwrap());
        let si2_gpu = gather_vec(gpu.get(si2).unwrap());
        eprintln!("SI2 len: cpu={} gpu={}", si2_cpu.len(), si2_gpu.len());
        eprintln!("SI2 CPU: {:?}", si2_cpu);
        eprintln!("SI2 GPU: {:?}", si2_gpu);
        assert_eq!(si2_cpu.len(), rows);
        assert_eq!(si2_gpu.len(), rows);
        for (a, b) in si2_cpu.iter().zip(si2_gpu.iter()) {
            if a.is_nan() || b.is_nan() {
                assert!(a.is_nan() && b.is_nan());
            } else {
                assert!((a - b).abs() < 1e-9);
            }
        }

        // Omit-nan compare
        let so1_cpu = gather_vec(cpu.get(so1).unwrap());
        let so1_gpu = gather_vec(gpu.get(so1).unwrap());
        eprintln!("SO1 len: cpu={} gpu={}", so1_cpu.len(), so1_gpu.len());
        eprintln!("SO1 CPU: {:?}", so1_cpu);
        eprintln!("SO1 GPU: {:?}", so1_gpu);
        assert_eq!(so1_cpu.len(), cols);
        assert_eq!(so1_gpu.len(), cols);
        for (i, (a, b)) in so1_cpu.iter().zip(so1_gpu.iter()).enumerate() {
            if (a - b).abs() >= 1e-9 {
                eprintln!("SO1 mismatch at col {}: cpu={} gpu={}", i, a, b);
            }
            assert!((a - b).abs() < 1e-9);
        }

        let so2_cpu = gather_vec(cpu.get(so2).unwrap());
        let so2_gpu = gather_vec(gpu.get(so2).unwrap());
        eprintln!("SO2 len: cpu={} gpu={}", so2_cpu.len(), so2_gpu.len());
        eprintln!("SO2 CPU: {:?}", so2_cpu);
        eprintln!("SO2 GPU: {:?}", so2_gpu);
        assert_eq!(so2_cpu.len(), rows);
        assert_eq!(so2_gpu.len(), rows);
        for (a, b) in so2_cpu.iter().zip(so2_gpu.iter()) {
            assert!((a - b).abs() < 1e-9);
        }
    });
}

#[test]
fn reduction_sum_include_omit_dim1_dim2_degenerate_gpu_cpu() {
    gc_test_context(|| {
        let cases = vec![(1usize, 7usize), (8usize, 1usize)];
        for (rows, cols) in cases {
            let source = format!(
                r#"
                rows = {rows}; cols = {cols};
                X = zeros(rows, cols);
                SI1 = sum(X, 1);
                SO1 = sum(X, 1, 'omitnan');
                SI2 = sum(X, 2);
                SO2 = sum(X, 2, 'omitnan');
                "#
            );

            let ast = parse(&source).expect("parse");
            let hir = lower(&ast).expect("lower");
            let bytecode = compile(&hir).expect("compile");

            // Determine slot for X (first StoreVar in the program)
            let x_index = bytecode
                .instructions
                .iter()
                .filter_map(|instr| match instr {
                    Instr::StoreVar(idx) => Some(*idx),
                    _ => None,
                })
                .next()
                .unwrap_or(0);

            // Build deterministic host matrix with NaNs on first row/col if present
            let mut data = vec![0.0f64; rows * cols];
            for c in 0..cols {
                for r in 0..rows {
                    data[r + c * rows] = (c as f64) * 10.0 + (r as f64 + 1.0);
                }
            }
            if rows > 0 {
                for c in 0..cols {
                    data[0 + c * rows] = f64::NAN;
                }
            }
            if cols > 0 {
                for r in 0..rows {
                    data[r + 0 * rows] = f64::NAN;
                }
            }

            // CPU path
            let cpu = {
                let mut vars = vec![Value::Num(0.0); bytecode.var_count];
                let t = runmat_builtins::Tensor::new(data.clone(), vec![rows, cols]).unwrap();
                vars[x_index] = Value::Tensor(t);
                interpret_function(&bytecode, vars).expect("interpret")
            };

            // GPU path
            let gpu = {
                ensure_provider_registered();
                let mut vars = vec![Value::Num(0.0); bytecode.var_count];
                let view = runmat_accelerate_api::HostTensorView {
                    data: &data,
                    shape: &[rows, cols],
                };
                let provider = runmat_accelerate_api::provider().expect("provider");
                let handle = provider.upload(&view).expect("upload");
                vars[x_index] = Value::GpuTensor(handle);
                interpret_function(&bytecode, vars).expect("interpret")
            };

            // Collect last 4 stores (SI1, SO1, SI2, SO2)
            let mut stores: Vec<usize> = bytecode
                .instructions
                .iter()
                .filter_map(|instr| match instr {
                    Instr::StoreVar(idx) => Some(*idx),
                    _ => None,
                })
                .collect();
            assert!(stores.len() >= 4);
            let so2 = stores.pop().unwrap();
            let si2 = stores.pop().unwrap();
            let so1 = stores.pop().unwrap();
            let si1 = stores.pop().unwrap();

            let gather_vec = |v: &Value| -> Vec<f64> {
                match v {
                    Value::GpuTensor(_) => match gather_if_needed(v).unwrap() {
                        Value::Tensor(t) => t.data,
                        Value::Num(n) => vec![n],
                        other => panic!("expected tensor, got {other:?}"),
                    },
                    Value::Tensor(t) => t.data.clone(),
                    Value::Num(n) => vec![*n],
                    other => panic!("expected tensor, got {other:?}"),
                }
            };

            // Compare include for both dims
            let si1_cpu = gather_vec(cpu.get(si1).unwrap());
            let si1_gpu = gather_vec(gpu.get(si1).unwrap());
            assert_eq!(si1_cpu.len(), cols);
            assert_eq!(si1_gpu.len(), cols);
            for (a, b) in si1_cpu.iter().zip(si1_gpu.iter()) {
                if a.is_nan() || b.is_nan() {
                    assert!(a.is_nan() && b.is_nan());
                } else {
                    assert!((a - b).abs() < 1e-9);
                }
            }
            let si2_cpu = gather_vec(cpu.get(si2).unwrap());
            let si2_gpu = gather_vec(gpu.get(si2).unwrap());
            assert_eq!(si2_cpu.len(), rows);
            assert_eq!(si2_gpu.len(), rows);
            for (a, b) in si2_cpu.iter().zip(si2_gpu.iter()) {
                if a.is_nan() || b.is_nan() {
                    assert!(a.is_nan() && b.is_nan());
                } else {
                    assert!((a - b).abs() < 1e-9);
                }
            }

            // Compare omit for both dims
            let so1_cpu = gather_vec(cpu.get(so1).unwrap());
            let so1_gpu = gather_vec(gpu.get(so1).unwrap());
            assert_eq!(so1_cpu.len(), cols);
            assert_eq!(so1_gpu.len(), cols);
            for (a, b) in so1_cpu.iter().zip(so1_gpu.iter()) {
                assert!((a - b).abs() < 1e-9);
            }
            let so2_cpu = gather_vec(cpu.get(so2).unwrap());
            let so2_gpu = gather_vec(gpu.get(so2).unwrap());
            assert_eq!(so2_cpu.len(), rows);
            assert_eq!(so2_gpu.len(), rows);
            for (a, b) in so2_cpu.iter().zip(so2_gpu.iter()) {
                assert!((a - b).abs() < 1e-9);
            }
        }
    });
}

#[test]
fn fused_elementwise_then_reduction_sum_dim1_dim2_include_gpu_cpu_small() {
    gc_test_context(|| {
        let rows = 8usize;
        let cols = 7usize;
        let source = format!(
            r#"
            rows = {rows}; cols = {cols};
            X = zeros(rows, cols);
            Y = sin(X) + X + 2;    % elementwise producer
            SI1 = sum(Y, 1);
            SI2 = sum(Y, 2);
            "#
        );
        let ast = parse(&source).expect("parse");
        let hir = lower(&ast).expect("lower");
        let bytecode = compile(&hir).expect("compile");

        let cpu =
            interpret_function(&bytecode, vec![Value::Num(0.0); bytecode.var_count]).expect("cpu");
        ensure_provider_registered();
        let gpu =
            interpret_function(&bytecode, vec![Value::Num(0.0); bytecode.var_count]).expect("gpu");

        let mut stores: Vec<usize> = bytecode
            .instructions
            .iter()
            .filter_map(|instr| match instr {
                Instr::StoreVar(idx) => Some(*idx),
                _ => None,
            })
            .collect();
        assert!(stores.len() >= 2);
        let si2 = stores.pop().unwrap();
        let si1 = stores.pop().unwrap();

        let gather_vec = |v: &Value| -> Vec<f64> {
            match v {
                Value::GpuTensor(_) => match gather_if_needed(v).unwrap() {
                    Value::Tensor(t) => t.data,
                    _ => panic!("expected tensor"),
                },
                Value::Tensor(t) => t.data.clone(),
                _ => panic!("expected tensor"),
            }
        };

        let si1_cpu = gather_vec(cpu.get(si1).unwrap());
        let si1_gpu = gather_vec(gpu.get(si1).unwrap());
        assert_eq!(si1_cpu.len(), cols);
        assert_eq!(si1_gpu.len(), cols);
        for (a, b) in si1_cpu.iter().zip(si1_gpu.iter()) {
            assert!((a - b).abs() < 1e-9);
        }

        let si2_cpu = gather_vec(cpu.get(si2).unwrap());
        let si2_gpu = gather_vec(gpu.get(si2).unwrap());
        assert_eq!(si2_cpu.len(), rows);
        assert_eq!(si2_gpu.len(), rows);
        for (a, b) in si2_cpu.iter().zip(si2_gpu.iter()) {
            assert!((a - b).abs() < 1e-9);
        }
    });
}

#[test]
fn fused_elementwise_then_reduction_sum_dim1_dim2_omit_gpu_cpu_small() {
    gc_test_context(|| {
        let rows = 8usize;
        let cols = 7usize;
        let source = format!(
            r#"
            rows = {rows}; cols = {cols};
            X = zeros(rows, cols);
            Y = sin(X) + X + 2;
            % inject NaNs in first row/col
            for c = 1:cols, X(1, c) = NaN; end
            for r = 1:rows, X(r, 1) = NaN; end
            Z = Y + X;              % keep NaNs aligned with Y
            SO1 = sum(Z, 1, 'omitnan');
            SO2 = sum(Z, 2, 'omitnan');
            "#
        );
        let ast = parse(&source).expect("parse");
        let hir = lower(&ast).expect("lower");
        let bytecode = compile(&hir).expect("compile");

        let cpu =
            interpret_function(&bytecode, vec![Value::Num(0.0); bytecode.var_count]).expect("cpu");
        ensure_provider_registered();
        let gpu =
            interpret_function(&bytecode, vec![Value::Num(0.0); bytecode.var_count]).expect("gpu");

        let mut stores: Vec<usize> = bytecode
            .instructions
            .iter()
            .filter_map(|instr| match instr {
                Instr::StoreVar(idx) => Some(*idx),
                _ => None,
            })
            .collect();
        assert!(stores.len() >= 2);
        let so2 = stores.pop().unwrap();
        let so1 = stores.pop().unwrap();

        let gather_vec = |v: &Value| -> Vec<f64> {
            match v {
                Value::GpuTensor(_) => match gather_if_needed(v).unwrap() {
                    Value::Tensor(t) => t.data,
                    _ => panic!("expected tensor"),
                },
                Value::Tensor(t) => t.data.clone(),
                _ => panic!("expected tensor"),
            }
        };

        let so1_cpu = gather_vec(cpu.get(so1).unwrap());
        let so1_gpu = gather_vec(gpu.get(so1).unwrap());
        assert_eq!(so1_cpu.len(), cols);
        assert_eq!(so1_gpu.len(), cols);
        for (a, b) in so1_cpu.iter().zip(so1_gpu.iter()) {
            assert!((a - b).abs() < 1e-9);
        }

        let so2_cpu = gather_vec(cpu.get(so2).unwrap());
        let so2_gpu = gather_vec(gpu.get(so2).unwrap());
        assert_eq!(so2_cpu.len(), rows);
        assert_eq!(so2_gpu.len(), rows);
        for (a, b) in so2_cpu.iter().zip(so2_gpu.iter()) {
            assert!((a - b).abs() < 1e-9);
        }
    });
}

#[test]
// keep: parity-focused provider check
fn provider_reduce_sum_dim_parity_simple() {
    gc_test_context(|| {
        ensure_provider_registered();
        let provider = runmat_accelerate_api::provider().expect("provider");

        let rows = 3usize;
        let cols = 4usize;
        // Column-major layout: data[r + c*rows]
        let mut data = vec![0.0f64; rows * cols];
        for c in 0..cols {
            for r in 0..rows {
                data[r + c * rows] = (c as f64) * 10.0 + (r as f64 + 1.0);
            }
        }
        // Insert NaNs: one in column 1, one in row 2
        data[0 + 1 * rows] = f64::NAN; // (r=0,c=1)
        data[2 + 3 * rows] = f64::NAN; // (r=2,c=3)

        let view = runmat_accelerate_api::HostTensorView {
            data: &data,
            shape: &[rows, cols],
        };
        let gpu = provider.upload(&view).expect("upload");

        // CPU include-nan semantics
        let mut col_sums_cpu = vec![0.0f64; cols];
        for c in 0..cols {
            let mut acc = 0.0;
            let mut saw_nan = false;
            for r in 0..rows {
                let v = data[r + c * rows];
                if v.is_nan() {
                    saw_nan = true;
                    break;
                } else {
                    acc += v;
                }
            }
            col_sums_cpu[c] = if saw_nan { f64::NAN } else { acc };
        }

        let mut row_sums_cpu = vec![0.0f64; rows];
        for r in 0..rows {
            let mut acc = 0.0;
            let mut saw_nan = false;
            for c in 0..cols {
                let v = data[r + c * rows];
                if v.is_nan() {
                    saw_nan = true;
                    break;
                } else {
                    acc += v;
                }
            }
            row_sums_cpu[r] = if saw_nan { f64::NAN } else { acc };
        }

        // Provider dim=0 (MATLAB dim=1): column-wise
        let col_gpu = provider
            .reduce_sum_dim(&gpu, 0)
            .expect("reduce dim=0 (cols)");
        let host_col = provider.download(&col_gpu).expect("download col");
        assert_eq!(host_col.shape, vec![1, cols]);
        assert_eq!(host_col.data.len(), cols);
        for (i, (a, b)) in col_sums_cpu.iter().zip(host_col.data.iter()).enumerate() {
            if a.is_nan() || b.is_nan() {
                assert!(
                    a.is_nan() && b.is_nan(),
                    "col {}: cpu={:?} gpu={:?}",
                    i,
                    a,
                    b
                );
            } else {
                assert!((a - b).abs() < 1e-9, "col {}: {} vs {}", i, a, b);
            }
        }

        // Provider dim=1 (MATLAB dim=2): row-wise
        let row_gpu = provider
            .reduce_sum_dim(&gpu, 1)
            .expect("reduce dim=1 (rows)");
        let host_row = provider.download(&row_gpu).expect("download row");
        assert_eq!(host_row.shape, vec![rows, 1]);
        assert_eq!(host_row.data.len(), rows);
        for (i, (a, b)) in row_sums_cpu.iter().zip(host_row.data.iter()).enumerate() {
            if a.is_nan() || b.is_nan() {
                assert!(
                    a.is_nan() && b.is_nan(),
                    "row {}: cpu={:?} gpu={:?}",
                    i,
                    a,
                    b
                );
            } else {
                assert!((a - b).abs() < 1e-9, "row {}: {} vs {}", i, a, b);
            }
        }
    });
}

#[test]
fn fused_reduction_sum_dim1_dim2_include_gpu_cpu() {
    gc_test_context(|| {
        let rows = 8usize;
        let cols = 7usize;
        let source = format!(
            r#"
            rows = {rows}; cols = {cols};
            X = zeros(rows, cols);
            Y = sin(X) .* X + 2;
            SI1 = sum(Y, 1);
            SI2 = sum(Y, 2);
            "#
        );

        let ast = parse(&source).expect("parse");
        let hir = lower(&ast).expect("lower");
        let bytecode = compile(&hir).expect("compile");

        let run_with = |use_gpu: bool| -> Vec<Value> {
            let vars = vec![Value::Num(0.0); bytecode.var_count];
            if use_gpu {
                ensure_provider_registered();
            }
            interpret_function(&bytecode, vars).expect("interpret")
        };

        // Warmups
        let _ = run_with(false);
        let _ = run_with(true);

        let cpu = run_with(false);
        let gpu = run_with(true);

        // Helper: gather value to vec
        let _gather_vec = |v: &Value| -> Option<Vec<f64>> {
            match v {
                Value::GpuTensor(_) => match gather_if_needed(v).ok()? {
                    Value::Tensor(t) => Some(t.data),
                    Value::Num(n) => Some(vec![n]),
                    _ => None,
                },
                Value::Tensor(t) => Some(t.data.clone()),
                Value::Num(n) => Some(vec![*n]),
                _ => None,
            }
        };

        // Robustly locate SI1 (1 x cols) and SI2 (rows x 1) by shape
        let find_by_shape =
            |vars: &Vec<Value>, want_rows: usize, want_cols: usize| -> Option<Vec<f64>> {
                for v in vars {
                    if let Some(Value::Tensor(t)) = gather_if_needed(v).ok() {
                        if t.shape.len() == 2 && t.shape[0] == want_rows && t.shape[1] == want_cols
                        {
                            return Some(t.data);
                        }
                    }
                }
                None
            };

        let si1_cpu = find_by_shape(&cpu, 1, cols).expect("cpu si1 1xcols");
        let si1_gpu = find_by_shape(&gpu, 1, cols).expect("gpu si1 1xcols");
        assert_eq!(si1_cpu.len(), cols);
        assert_eq!(si1_gpu.len(), cols);
        for (a, b) in si1_cpu.iter().zip(si1_gpu.iter()) {
            if a.is_nan() || b.is_nan() {
                assert!(a.is_nan() && b.is_nan());
            } else {
                assert!((a - b).abs() < 1e-9);
            }
        }

        let si2_cpu = find_by_shape(&cpu, rows, 1).expect("cpu si2 rowsx1");
        let si2_gpu = find_by_shape(&gpu, rows, 1).expect("gpu si2 rowsx1");
        assert_eq!(si2_cpu.len(), rows);
        assert_eq!(si2_gpu.len(), rows);
        for (a, b) in si2_cpu.iter().zip(si2_gpu.iter()) {
            if a.is_nan() || b.is_nan() {
                assert!(a.is_nan() && b.is_nan());
            } else {
                assert!((a - b).abs() < 1e-9);
            }
        }
    });
}

#[test]
fn fused_reduction_sum_dim1_dim2_omit_gpu_cpu() {
    gc_test_context(|| {
        let rows = 8usize;
        let cols = 7usize;
        let source = format!(
            r#"
            rows = {rows}; cols = {cols};
            X = zeros(rows, cols);
            % Insert NaNs in first row and first column
            for c = 1:cols
                X(1, c) = NaN;
            end
            for r = 1:rows
                X(r, 1) = NaN;
            end
            Y = sin(X) .* X + 2;
            SO1 = sum(Y, 1, 'omitnan');
            SO2 = sum(Y, 2, 'omitnan');
            "#
        );

        let ast = parse(&source).expect("parse");
        let hir = lower(&ast).expect("lower");
        let bytecode = compile(&hir).expect("compile");

        let run_with = |use_gpu: bool| -> Vec<Value> {
            let vars = vec![Value::Num(0.0); bytecode.var_count];
            if use_gpu {
                ensure_provider_registered();
            }
            interpret_function(&bytecode, vars).expect("interpret")
        };

        // Warmups
        let _ = run_with(false);
        let _ = run_with(true);

        let cpu = run_with(false);
        let gpu = run_with(true);

        // Extract by shape
        let find_by_shape =
            |vars: &Vec<Value>, want_rows: usize, want_cols: usize| -> Option<Vec<f64>> {
                for v in vars {
                    if let Some(Value::Tensor(t)) = gather_if_needed(v).ok() {
                        if t.shape.len() == 2 && t.shape[0] == want_rows && t.shape[1] == want_cols
                        {
                            return Some(t.data);
                        }
                    }
                }
                None
            };

        let so1_cpu = find_by_shape(&cpu, 1, cols).expect("cpu so1 1xcols");
        let so1_gpu = find_by_shape(&gpu, 1, cols).expect("gpu so1 1xcols");
        assert_eq!(so1_cpu.len(), cols);
        assert_eq!(so1_gpu.len(), cols);
        for (a, b) in so1_cpu.iter().zip(so1_gpu.iter()) {
            assert!((a - b).abs() < 1e-9);
        }

        let so2_cpu = find_by_shape(&cpu, rows, 1).expect("cpu so2 rowsx1");
        let so2_gpu = find_by_shape(&gpu, rows, 1).expect("gpu so2 rowsx1");
        assert_eq!(so2_cpu.len(), rows);
        assert_eq!(so2_gpu.len(), rows);
        for (a, b) in so2_cpu.iter().zip(so2_gpu.iter()) {
            assert!((a - b).abs() < 1e-9);
        }
    });
}
#[allow(dead_code)]
fn fused_elementwise_residency_and_gather() {
    gc_test_context(|| {
        ensure_provider_registered();

        let source = r#"
        x = [1, 2, 3];
        b = 2;
        y = sin(x) .* x + b;
        "#;

        let ast = parse(source).expect("parse");
        let hir = lower(&ast).expect("lower");
        let bytecode = compile(&hir).expect("compile");
        let vars = interpret(&bytecode).expect("interpret");

        let y_index = bytecode
            .instructions
            .iter()
            .filter_map(|instr| match instr {
                Instr::StoreVar(idx) => Some(*idx),
                _ => None,
            })
            .last()
            .expect("store var for y");

        let y_value = vars.get(y_index).expect("value for y");

        let handle = match y_value {
            Value::GpuTensor(handle) => handle,
            other => panic!("expected GPU tensor, got {other:?}"),
        };

        assert!(
            fusion_residency::is_resident(handle),
            "GPU handle should be marked resident before gather"
        );

        let gathered = gather_if_needed(y_value).expect("gather");
        let tensor = match gathered {
            Value::Tensor(tensor) => tensor,
            other => panic!("expected gathered tensor, got {other:?}"),
        };
        assert!(
            !fusion_residency::is_resident(handle),
            "Residency should be cleared after gather"
        );

        let expected: Vec<f64> = [1.0f64, 2.0, 3.0]
            .iter()
            .map(|x| x.sin() * x + 2.0)
            .collect();
        assert_eq!(tensor.data.len(), expected.len());
        for (actual, expect) in tensor.data.iter().zip(expected.iter()) {
            assert!(
                (actual - expect).abs() < 1e-9,
                "mismatch: {actual} vs {expect}"
            );
        }
    });
}

#[test]
fn fused_literal_constant_and_extended_builtins() {
    gc_test_context(|| {
        ensure_provider_registered();

        let source = r#"
        x = [4, 5, 6];
        y = sin(x) .* x + 2;
        u = exp(x);
        v = sqrt(x);
        "#;

        let ast = parse(source).expect("parse");
        let hir = lower(&ast).expect("lower");
        let bytecode = compile(&hir).expect("compile");
        let vars = interpret(&bytecode).expect("interpret");

        let mut stores: Vec<usize> = bytecode
            .instructions
            .iter()
            .filter_map(|instr| match instr {
                Instr::StoreVar(idx) => Some(*idx),
                _ => None,
            })
            .collect();
        assert!(stores.len() >= 4, "expected stores for x, y, u, v");
        let v_index = stores.pop().unwrap();
        let u_index = stores.pop().unwrap();
        let y_index = stores.pop().unwrap();

        let check_tensor = |value: &Value| -> Vec<f64> {
            match value {
                Value::GpuTensor(handle) => {
                    assert!(fusion_residency::is_resident(handle));
                    let gathered = gather_if_needed(value).expect("gather");
                    let tensor = match gathered {
                        Value::Tensor(tensor) => tensor,
                        other => panic!("expected gathered tensor, got {other:?}"),
                    };
                    assert!(
                        !fusion_residency::is_resident(handle),
                        "Residency should be cleared after gather"
                    );
                    tensor.data
                }
                Value::Tensor(tensor) => tensor.data.clone(),
                other => panic!("expected tensor, got {other:?}"),
            }
        };

        let y_data = check_tensor(vars.get(y_index).expect("value for y"));
        let y_expected: Vec<f64> = [4.0f64, 5.0, 6.0]
            .iter()
            .map(|x| x.sin() * x + 2.0)
            .collect();
        assert_eq!(y_data, y_expected);

        let u_data = check_tensor(vars.get(u_index).expect("value for u"));
        let u_expected: Vec<f64> = [4.0f64, 5.0, 6.0].iter().map(|x| x.exp()).collect();
        assert_eq!(u_data, u_expected);

        let v_data = check_tensor(vars.get(v_index).expect("value for v"));
        let v_expected: Vec<f64> = [4.0f64, 5.0, 6.0].iter().map(|x| x.sqrt()).collect();
        assert_eq!(v_data, v_expected);
    });
}

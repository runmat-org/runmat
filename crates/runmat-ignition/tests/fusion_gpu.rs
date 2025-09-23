use anyhow::{anyhow, bail, Context};
use once_cell::sync::OnceCell;
use runmat_accelerate::fusion_residency;
use runmat_accelerate_api::{
    AccelProvider, ApiDeviceInfo, GpuTensorHandle, HostTensorOwned, HostTensorView,
    ProviderPrecision,
};
use runmat_builtins::Value;
use runmat_gc::gc_test_context;
use runmat_hir::lower;
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

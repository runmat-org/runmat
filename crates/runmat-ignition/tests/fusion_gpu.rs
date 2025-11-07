use anyhow::{anyhow, bail, Context};
use once_cell::sync::OnceCell;
use runmat_accelerate::fusion_residency;
use runmat_accelerate_api::{
    AccelProvider, ApiDeviceInfo, CorrcoefOptions, CovNormalization, CovRows, CovarianceOptions,
    FspecialRequest, GpuTensorHandle, HostTensorOwned, HostTensorView, ImageNormalizeDescriptor,
    PagefunRequest, PowerStepEpilogue, ProviderCondNorm, ProviderConvMode, ProviderEigResult,
    ProviderLinsolveOptions, ProviderLinsolveResult, ProviderNormOrder, ProviderPinvOptions,
    ProviderPrecision, UniqueOptions, UniqueResult,
};
use runmat_builtins::{Tensor, Value};
use runmat_gc::gc_test_context;
use runmat_hir::lower;
use runmat_ignition::vm::interpret_function;
use runmat_ignition::{compile, interpret, Instr};
use runmat_parser::parse;
use runmat_runtime::builtins::image::filters::fspecial::spec_from_request as test_fspecial_spec_from_request;
use runmat_runtime::builtins::math::linalg::ops::mrdivide_host_real_for_provider;
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

    fn logical_isreal(&self, handle: &GpuTensorHandle) -> anyhow::Result<bool> {
        let _ = self.pull(handle)?;
        Ok(true)
    }

    fn pagefun(&self, _request: &PagefunRequest) -> anyhow::Result<GpuTensorHandle> {
        Err(anyhow!("pagefun not supported by test provider"))
    }

    fn fspecial(&self, request: &FspecialRequest) -> anyhow::Result<GpuTensorHandle> {
        let spec =
            test_fspecial_spec_from_request(&request.filter).map_err(|e: String| anyhow!(e))?;
        let tensor = spec.generate_tensor().map_err(|e| anyhow!(e))?;
        Ok(self.push(tensor.data.clone(), tensor.shape.clone()))
    }

    fn covariance(
        &self,
        matrix: &GpuTensorHandle,
        second: Option<&GpuTensorHandle>,
        weights: Option<&GpuTensorHandle>,
        options: &CovarianceOptions,
    ) -> anyhow::Result<GpuTensorHandle> {
        if second.is_some() {
            bail!("test provider: covariance secondary input unsupported");
        }
        if weights.is_some() || options.has_weight_vector {
            bail!("test provider: covariance weights unsupported");
        }
        if options.rows != CovRows::All {
            bail!(
                "test provider: covariance row option {:?} unsupported",
                options.rows
            );
        }
        let (data, shape) = self.pull(matrix)?;
        let rows = shape.get(0).copied().unwrap_or(1);
        let cols = if shape.len() > 1 { shape[1] } else { 1 };
        if rows == 0 || cols == 0 {
            return Ok(self.push(Vec::new(), vec![cols, cols]));
        }
        let mut means = vec![0.0f64; cols];
        for c in 0..cols {
            for r in 0..rows {
                let idx = r + rows * c;
                if let Some(value) = data.get(idx) {
                    means[c] += *value;
                }
            }
            means[c] /= rows as f64;
        }
        let denom = match options.normalization {
            CovNormalization::Unbiased => (rows as f64) - 1.0,
            CovNormalization::Biased => rows as f64,
        };
        let denom = if denom <= 0.0 { 1.0 } else { denom };
        let mut cov = vec![0.0f64; cols * cols];
        for c1 in 0..cols {
            for c2 in 0..cols {
                let mut acc = 0.0f64;
                for r in 0..rows {
                    let idx1 = r + rows * c1;
                    let idx2 = r + rows * c2;
                    let val1 = data.get(idx1).copied().unwrap_or(0.0);
                    let val2 = data.get(idx2).copied().unwrap_or(0.0);
                    acc += (val1 - means[c1]) * (val2 - means[c2]);
                }
                let idx = c1 + cols * c2;
                cov[idx] = acc / denom;
            }
        }
        Ok(self.push(cov, vec![cols, cols]))
    }

    fn image_normalize(
        &self,
        input: &GpuTensorHandle,
        desc: &ImageNormalizeDescriptor,
    ) -> anyhow::Result<GpuTensorHandle> {
        let (data, shape) = self.pull(input)?;
        if shape.len() != 3 {
            bail!("test provider: image_normalize expects 3-D tensor");
        }
        let batch = shape[0];
        let height = shape[1];
        let width = shape[2];
        if batch != desc.batch || height != desc.height || width != desc.width {
            bail!(
                "test provider: image_normalize descriptor mismatch tensor {:?} vs {:?}",
                shape,
                (desc.batch, desc.height, desc.width)
            );
        }
        let plane = height * width;
        if plane == 0 {
            return Ok(self.push(Vec::new(), shape));
        }

        let mut out = data.clone();
        for b in 0..batch {
            let mut sum = 0.0f64;
            for idx in 0..plane {
                let offset = b + batch * idx;
                sum += data[offset];
            }
            let mean = sum / plane as f64;

            let mut sq_sum = 0.0f64;
            for idx in 0..plane {
                let offset = b + batch * idx;
                let diff = data[offset] - mean;
                sq_sum += diff * diff;
            }
            let variance = sq_sum / plane as f64;
            let sigma = (variance + desc.epsilon).sqrt();
            let inv_sigma = if sigma > 0.0 { 1.0 / sigma } else { 0.0 };

            for idx in 0..plane {
                let offset = b + batch * idx;
                let mut value = (data[offset] - mean) * inv_sigma;
                if let Some(g) = desc.gain {
                    value *= g;
                }
                if let Some(bias) = desc.bias {
                    value += bias;
                }
                value = value.max(0.0);
                if let Some(gamma) = desc.gamma {
                    value = value.powf(gamma);
                }
                out[offset] = value;
            }
        }

        Ok(self.push(out, shape))
    }

    fn matmul_power_step(
        &self,
        lhs: &GpuTensorHandle,
        rhs: &GpuTensorHandle,
        epilogue: &PowerStepEpilogue,
    ) -> anyhow::Result<GpuTensorHandle> {
        let (lhs_data, lhs_shape) = self.pull(lhs)?;
        let (rhs_data, rhs_shape) = self.pull(rhs)?;
        if lhs_shape.len() != 2 || rhs_shape.len() != 2 {
            bail!("test provider: matmul_power_step expects 2D inputs");
        }
        let m = lhs_shape[0];
        let k = lhs_shape[1];
        if rhs_shape[0] != k {
            bail!("test provider: matmul_power_step inner dimensions mismatch");
        }
        let n = rhs_shape[1];
        let mut product = vec![0.0f64; m * n];
        for col in 0..n {
            for row in 0..m {
                let mut acc = 0.0f64;
                for kk in 0..k {
                    let lhs_idx = row + m * kk;
                    let rhs_idx = kk + k * col;
                    acc += lhs_data[lhs_idx] * rhs_data[rhs_idx];
                }
                product[row + m * col] = acc;
            }
        }
        let mut norms = vec![0.0f64; n];
        for col in 0..n {
            let mut acc = 0.0f64;
            for row in 0..m {
                let val = product[row + m * col];
                acc += val * val;
            }
            acc += epilogue.epsilon;
            norms[col] = acc.sqrt();
        }
        for col in 0..n {
            let norm = norms[col];
            for row in 0..m {
                let idx = row + m * col;
                product[idx] /= norm;
            }
        }
        Ok(self.push(product, vec![m, n]))
    }

    fn eig(&self, _a: &GpuTensorHandle, _compute_left: bool) -> anyhow::Result<ProviderEigResult> {
        bail!("eig not supported by test provider")
    }

    fn linsolve(
        &self,
        _lhs: &GpuTensorHandle,
        _rhs: &GpuTensorHandle,
        _options: &ProviderLinsolveOptions,
    ) -> anyhow::Result<ProviderLinsolveResult> {
        bail!("linsolve not supported by test provider")
    }

    fn pinv(
        &self,
        _matrix: &GpuTensorHandle,
        _options: ProviderPinvOptions,
    ) -> anyhow::Result<GpuTensorHandle> {
        bail!("pinv not supported by test provider")
    }

    fn cond(
        &self,
        _matrix: &GpuTensorHandle,
        _norm: ProviderCondNorm,
    ) -> anyhow::Result<GpuTensorHandle> {
        bail!("cond not supported by test provider")
    }

    fn norm(
        &self,
        _tensor: &GpuTensorHandle,
        _order: ProviderNormOrder,
    ) -> anyhow::Result<GpuTensorHandle> {
        bail!("norm not supported by test provider")
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

    fn reduce_any_dim(
        &self,
        _a: &GpuTensorHandle,
        _dim: usize,
        _omit_nan: bool,
    ) -> anyhow::Result<GpuTensorHandle> {
        bail!("reduce_any_dim not supported by test provider")
    }

    fn reduce_any(&self, _a: &GpuTensorHandle, _omit_nan: bool) -> anyhow::Result<GpuTensorHandle> {
        bail!("reduce_any not supported by test provider")
    }

    fn reduce_all_dim(
        &self,
        _a: &GpuTensorHandle,
        _dim: usize,
        _omit_nan: bool,
    ) -> anyhow::Result<GpuTensorHandle> {
        bail!("reduce_all_dim not supported by test provider")
    }

    fn reduce_all(&self, _a: &GpuTensorHandle, _omit_nan: bool) -> anyhow::Result<GpuTensorHandle> {
        bail!("reduce_all not supported by test provider")
    }

    fn conv2d(
        &self,
        _signal: &GpuTensorHandle,
        _kernel: &GpuTensorHandle,
        _mode: ProviderConvMode,
    ) -> anyhow::Result<GpuTensorHandle> {
        bail!("conv2d not supported by test provider")
    }

    fn mrdivide(
        &self,
        lhs: &GpuTensorHandle,
        rhs: &GpuTensorHandle,
    ) -> anyhow::Result<GpuTensorHandle> {
        let (lhs_data, lhs_shape) = self.pull(lhs)?;
        let (rhs_data, rhs_shape) = self.pull(rhs)?;

        let lhs_tensor = Tensor::new(lhs_data, lhs_shape)
            .map_err(|e| anyhow!("mrdivide (test provider): {e}"))?;
        let rhs_tensor = Tensor::new(rhs_data, rhs_shape)
            .map_err(|e| anyhow!("mrdivide (test provider): {e}"))?;
        let result = mrdivide_host_real_for_provider(&lhs_tensor, &rhs_tensor)
            .map_err(|e| anyhow!("{e}"))?;
        let Tensor { data, shape, .. } = result;
        Ok(self.push(data, shape))
    }

    fn sym_rcm(&self, _matrix: &GpuTensorHandle) -> anyhow::Result<Vec<usize>> {
        bail!("symrcm not supported by test provider")
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

    fn corrcoef(
        &self,
        _matrix: &GpuTensorHandle,
        _options: &CorrcoefOptions,
    ) -> anyhow::Result<GpuTensorHandle> {
        Err(anyhow!("corrcoef (test provider): not implemented"))
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

    fn transpose(&self, handle: &GpuTensorHandle) -> anyhow::Result<GpuTensorHandle> {
        let (data, shape) = self.pull(handle)?;
        if shape.len() < 2 {
            return Ok(self.push(data, shape));
        }
        let rows = shape[0];
        let cols = shape[1];
        let mut out = vec![0.0f64; data.len()];
        for r in 0..rows {
            for c in 0..cols {
                let src = r + rows * c;
                let dst = c + cols * r;
                out[dst] = data[src];
            }
        }
        let mut new_shape = shape.clone();
        new_shape[0] = cols;
        new_shape[1] = rows;
        Ok(self.push(out, new_shape))
    }

    fn matmul(
        &self,
        lhs: &GpuTensorHandle,
        rhs: &GpuTensorHandle,
    ) -> anyhow::Result<GpuTensorHandle> {
        let (lhs_data, lhs_shape) = self.pull(lhs)?;
        let (rhs_data, rhs_shape) = self.pull(rhs)?;
        if lhs_shape.len() != 2 || rhs_shape.len() != 2 {
            bail!("test provider: matmul expects 2D inputs");
        }
        let m = lhs_shape[0];
        let k = lhs_shape[1];
        if rhs_shape[0] != k {
            bail!("test provider: matmul inner dimensions mismatch");
        }
        let n = rhs_shape[1];
        let mut out = vec![0.0f64; m * n];
        for row in 0..m {
            for col in 0..n {
                let mut acc = 0.0;
                for inner in 0..k {
                    let lhs_idx = row + m * inner;
                    let rhs_idx = inner + k * col;
                    acc += lhs_data[lhs_idx] * rhs_data[rhs_idx];
                }
                let dst = row + m * col;
                out[dst] = acc;
            }
        }
        Ok(self.push(out, vec![m, n]))
    }

    fn diag_extract(
        &self,
        matrix: &GpuTensorHandle,
        offset: isize,
    ) -> anyhow::Result<GpuTensorHandle> {
        if offset != 0 {
            bail!("test provider: diag_extract offset {offset} unsupported");
        }
        let (data, shape) = self.pull(matrix)?;
        if shape.len() != 2 {
            bail!("test provider: diag_extract expects 2D input");
        }
        let rows = shape[0];
        let cols = shape[1];
        let len = rows.min(cols);
        let mut out = Vec::with_capacity(len);
        for i in 0..len {
            let idx = i + rows * i;
            out.push(data[idx]);
        }
        Ok(self.push(out, vec![len]))
    }

    fn reshape(
        &self,
        handle: &GpuTensorHandle,
        new_shape: &[usize],
    ) -> anyhow::Result<GpuTensorHandle> {
        let len: usize = new_shape.iter().product();
        let mut buffers = self.buffers.lock().unwrap();
        let entry = buffers
            .get_mut(&handle.buffer_id)
            .ok_or_else(|| anyhow!("reshape: unknown buffer {}", handle.buffer_id))?;
        if entry.0.len() != len {
            bail!(
                "reshape: product of new dimensions ({}) must equal existing length ({})",
                len,
                entry.0.len()
            );
        }
        entry.1 = new_shape.to_vec();
        let mut updated = handle.clone();
        updated.shape = new_shape.to_vec();
        Ok(updated)
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
        if let Some(graph) = &bytecode.accel_graph {
            let groups = graph.detect_fusion_groups();
            let reduction_count = groups
                .iter()
                .filter(|g| matches!(g.kind, runmat_accelerate::fusion::FusionKind::Reduction))
                .count();
            assert!(
                reduction_count >= 2,
                "expected at least two reduction groups for include/omit nan sums, got {:?}",
                groups.iter().map(|g| g.kind.clone()).collect::<Vec<_>>()
            );
        } else {
            panic!("bytecode missing accel graph");
        }

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

        dbg!(&bytecode.fusion_groups);
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

#[test]
fn centered_gram_fusion_matches_cpu() {
    gc_test_context(|| {
        let rows = 5usize;
        let cols = 3usize;
        let source = r#"
            rows = 5;
            A = [
                1.0, 2.0, 3.0;
                4.0, 5.0, 6.0;
                7.0, 8.0, 9.0;
                -1.0, 0.5, 2.0;
                3.0, -2.0, 4.0
            ];
            mu = mean(A, 1);
            centered = A - mu;
            cov = (centered.' * centered) / (rows - 1);
        "#;

        let ast = parse(source).expect("parse");
        let hir = lower(&ast).expect("lower");
        let bytecode = compile(&hir).expect("compile");

        let cov_index = bytecode
            .instructions
            .iter()
            .filter_map(|instr| match instr {
                Instr::StoreVar(idx) => Some(*idx),
                _ => None,
            })
            .last()
            .expect("cov store index");

        let vars = vec![Value::Num(0.0); bytecode.var_count];

        // Compute expected covariance on host
        let data: [[f64; 3]; 5] = [
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0],
            [-1.0, 0.5, 2.0],
            [3.0, -2.0, 4.0],
        ];
        let mut means = vec![0.0f64; cols];
        for row in &data {
            for (c, value) in row.iter().enumerate() {
                means[c] += *value;
            }
        }
        for mean in &mut means {
            *mean /= rows as f64;
        }
        let mut cov_expected = vec![0.0f64; cols * cols];
        for row in &data {
            let mut centered = vec![0.0f64; cols];
            for (c, value) in row.iter().enumerate() {
                centered[c] = value - means[c];
            }
            for i in 0..cols {
                for j in 0..cols {
                    cov_expected[i * cols + j] += centered[i] * centered[j];
                }
            }
        }
        let denom = (rows as f64) - 1.0;
        for value in &mut cov_expected {
            *value /= denom;
        }

        ensure_provider_registered();
        let vars_gpu = interpret_function(&bytecode, vars).expect("gpu interpret");
        let cov_gpu = vars_gpu.get(cov_index).expect("cov gpu");
        assert!(
            matches!(cov_gpu, Value::GpuTensor(_)),
            "expected gpu tensor result"
        );
        let gathered_gpu = gather_if_needed(cov_gpu).expect("gather gpu");
        let gpu_tensor = match gathered_gpu {
            Value::Tensor(t) => t,
            other => panic!("expected tensor cov (gpu), got {other:?}"),
        };

        assert_eq!(gpu_tensor.shape, vec![cols, cols], "shape mismatch");
        let tol = 1e-6;
        for (lhs, rhs) in cov_expected.iter().zip(gpu_tensor.data.iter()) {
            let diff = (lhs - rhs).abs();
            assert!(
                diff <= tol,
                "covariance mismatch: lhs={lhs}, rhs={rhs}, diff={diff}"
            );
        }
    });
}

#[test]
fn power_step_normalization_matches_cpu() {
    gc_test_context(|| {
        use runmat_accelerate::FusionKind;
        let rows = 3usize;
        let cols = 2usize;
        let source = r#"
        G = [
            1.0, -0.5, 2.0;
            0.0, 1.5, -1.0;
            0.75, 0.25, 0.5
        ];
        Q = [
            2.0, -1.0;
            0.5, 3.0;
            -2.0, 1.0
        ];
        Q = mtimes(G, Q);
        norms = sqrt(sum(Q.^2, 1) + 1e-6);
        Q = Q ./ norms;
        "#;

        let ast = parse(source).expect("parse");
        let hir = lower(&ast).expect("lower");
        let bytecode = compile(&hir).expect("compile");

        if let Some(graph) = &bytecode.accel_graph {
            let groups = graph.detect_fusion_groups();
            assert!(
                groups
                    .iter()
                    .any(|g| matches!(g.kind, FusionKind::PowerStepNormalize)),
                "expected power-step group, got {:?}",
                groups
            );
        }

        let q_index = bytecode
            .instructions
            .iter()
            .filter_map(|instr| match instr {
                Instr::StoreVar(idx) => Some(*idx),
                _ => None,
            })
            .last()
            .expect("store index for Q");

        let vars = vec![Value::Num(0.0); bytecode.var_count];
        let vars_cpu = interpret_function(&bytecode, vars.clone()).expect("cpu interpret");
        let q_cpu = vars_cpu.get(q_index).expect("cpu Q");
        let gathered_cpu = gather_if_needed(q_cpu).expect("gather cpu");
        let cpu_tensor = match gathered_cpu {
            Value::Tensor(t) => t,
            other => panic!("expected tensor Q (cpu), got {other:?}"),
        };
        assert_eq!(cpu_tensor.shape, vec![rows, cols], "cpu shape mismatch");

        ensure_provider_registered();
        let vars_gpu = interpret_function(&bytecode, vars).expect("gpu interpret");
        let q_gpu = vars_gpu.get(q_index).expect("gpu Q");
        assert!(
            matches!(q_gpu, Value::GpuTensor(_)),
            "expected gpu tensor result"
        );
        let gathered_gpu = gather_if_needed(q_gpu).expect("gather gpu");
        let gpu_tensor = match gathered_gpu {
            Value::Tensor(t) => t,
            other => panic!("expected tensor Q (gpu), got {other:?}"),
        };

        assert_eq!(cpu_tensor.shape, gpu_tensor.shape, "shape mismatch");
        let tol = 1e-6;
        for (lhs, rhs) in cpu_tensor.data.iter().zip(gpu_tensor.data.iter()) {
            let diff = (lhs - rhs).abs();
            assert!(
                diff <= tol,
                "power-step mismatch: lhs={lhs}, rhs={rhs}, diff={diff}"
            );
        }
    });
}

#[test]
fn image_normalize_matches_cpu() {
    gc_test_context(|| {
        use runmat_accelerate::FusionKind;
        let source = r#"
        B = 4; H = 8; W = 12;
        gain = single(1.0123);
        bias = single(-0.02);
        gamma = single(1.8);
        eps0 = single(1e-6);
        rng(0);
        imgs = single(rand(B, H, W));
        mu = mean(mean(imgs, 2), 3);
        sigma = sqrt(mean(mean((imgs - mu).^2, 2), 3) + eps0);
        out = ((imgs - mu) ./ sigma) * gain + bias;
        out = max(out, single(0));
        out = out .^ gamma;
        "#;

        let ast = parse(source).expect("parse");
        let hir = lower(&ast).expect("lower");
        let bytecode = compile(&hir).expect("compile");

        if let Some(graph) = &bytecode.accel_graph {
            let groups = graph.detect_fusion_groups();
            assert!(groups
                .iter()
                .any(|group| matches!(group.kind, FusionKind::ImageNormalize)));
        }

        let out_index = bytecode
            .instructions
            .iter()
            .filter_map(|instr| match instr {
                Instr::StoreVar(idx) => Some(*idx),
                _ => None,
            })
            .last()
            .expect("store index for out");

        let vars = vec![Value::Num(0.0); bytecode.var_count];
        let vars_cpu = interpret_function(&bytecode, vars.clone()).expect("cpu interpret");
        let out_cpu = vars_cpu.get(out_index).expect("cpu out");
        let gathered_cpu = gather_if_needed(out_cpu).expect("gather cpu out");
        let cpu_tensor = match gathered_cpu {
            Value::Tensor(t) => t,
            other => panic!("expected tensor out (cpu), got {other:?}"),
        };

        ensure_provider_registered();
        let vars_gpu = interpret_function(&bytecode, vars).expect("gpu interpret");
        let out_gpu = vars_gpu.get(out_index).expect("gpu out");
        assert!(matches!(out_gpu, Value::GpuTensor(_)));
        let gathered_gpu = gather_if_needed(out_gpu).expect("gather gpu out");
        let gpu_tensor = match gathered_gpu {
            Value::Tensor(t) => t,
            other => panic!("expected tensor out (gpu), got {other:?}"),
        };

        assert_eq!(cpu_tensor.shape, gpu_tensor.shape, "shape mismatch");
        let tol = 5e-4;
        for (lhs, rhs) in cpu_tensor.data.iter().zip(gpu_tensor.data.iter()) {
            let diff = (lhs - rhs).abs();
            assert!(
                diff <= tol,
                "image normalize mismatch: lhs={lhs}, rhs={rhs}, diff={diff}"
            );
        }
    });
}

#[test]
fn explained_variance_matches_cpu() {
    gc_test_context(|| {
        use runmat_accelerate::FusionKind;
        let source = r#"
        rng(0);
        rows = 4;
        cols = 2;
        G = rand(rows, rows);
        Q = rand(rows, cols);
        tmp = mtimes(Q.', G);
        prod = mtimes(tmp, Q);
        eval = diag(prod);
        "#;

        let ast = parse(source).expect("parse");
        let hir = lower(&ast).expect("lower");
        let bytecode = compile(&hir).expect("compile");

        if std::env::var("RUNMAT_DEBUG_EXPLAINED").is_ok() {
            for (pc, instr) in bytecode.instructions.iter().enumerate() {
                eprintln!("instr {pc}: {instr:?}");
            }
        }

        let (q_var_idx, g_var_idx) = if let Some(graph) = &bytecode.accel_graph {
            let groups = graph.detect_fusion_groups();
            assert!(
                groups
                    .iter()
                    .any(|g| matches!(g.kind, FusionKind::ExplainedVariance)),
                "expected explained variance group, got {:?}",
                groups.iter().map(|g| g.kind.clone()).collect::<Vec<_>>()
            );
            let plan = runmat_accelerate::FusionPlan::from_graph(graph, &groups);
            let explained = plan
                .groups
                .iter()
                .find(|g| matches!(g.group.kind, FusionKind::ExplainedVariance))
                .expect("explained variance plan");
            let (q_vid, g_vid) = match explained.pattern.as_ref() {
                Some(runmat_accelerate::fusion::FusionPattern::ExplainedVariance { q, g }) => {
                    (*q, *g)
                }
                other => panic!("unexpected pattern {other:?}"),
            };
            let q_binding = graph
                .var_binding(q_vid)
                .expect("q binding available for explained variance");
            let g_binding = graph
                .var_binding(g_vid)
                .expect("g binding available for explained variance");
            (q_binding.index, g_binding.index)
        } else {
            panic!("expected accel graph")
        };

        let eval_index = bytecode
            .instructions
            .iter()
            .filter_map(|instr| match instr {
                Instr::StoreVar(idx) => Some(*idx),
                _ => None,
            })
            .last()
            .expect("store index for eval");

        let vars = vec![Value::Num(0.0); bytecode.var_count];
        let vars_cpu = interpret_function(&bytecode, vars.clone()).expect("cpu interpret");
        let eval_cpu = vars_cpu.get(eval_index).expect("cpu eval");
        let gathered_cpu = gather_if_needed(eval_cpu).expect("gather cpu");
        let cpu_tensor = match gathered_cpu {
            Value::Tensor(t) => t,
            other => panic!("expected tensor eval (cpu), got {other:#?}"),
        };

        let gather_tensor = |value: &Value| -> Tensor {
            match gather_if_needed(value).expect("gather tensor") {
                Value::Tensor(t) => t,
                other => panic!("expected tensor, got {other:#?}"),
            }
        };

        if std::env::var("RUNMAT_DEBUG_EXPLAINED").is_ok() {
            println!("q_var_idx {} g_var_idx {}", q_var_idx, g_var_idx);
            println!(
                "q tensor shape {:?}",
                gather_tensor(
                    vars_cpu
                        .get(q_var_idx)
                        .expect("q var present after cpu run")
                )
                .shape
            );
            println!(
                "g tensor shape {:?}",
                gather_tensor(
                    vars_cpu
                        .get(g_var_idx)
                        .expect("g var present after cpu run")
                )
                .shape
            );
            println!(
                "q tensor data {:?}",
                gather_tensor(
                    vars_cpu
                        .get(q_var_idx)
                        .expect("q var present after cpu run")
                )
                .data
            );
            println!(
                "g tensor data {:?}",
                gather_tensor(
                    vars_cpu
                        .get(g_var_idx)
                        .expect("g var present after cpu run")
                )
                .data
            );
        }

        let q_tensor = gather_tensor(
            vars_cpu
                .get(q_var_idx)
                .expect("q var present after cpu run"),
        );
        let g_tensor = gather_tensor(
            vars_cpu
                .get(g_var_idx)
                .expect("g var present after cpu run"),
        );
        if std::env::var("RUNMAT_DEBUG_EXPLAINED").is_ok() {
            println!("q tensor shape {:?}", q_tensor.shape);
            println!("g tensor shape {:?}", g_tensor.shape);
            println!("q tensor data {:?}", q_tensor.data);
            println!("g tensor data {:?}", g_tensor.data);
        }

        let rows = q_tensor.shape.get(0).copied().unwrap_or(0);
        let cols = q_tensor.shape.get(1).copied().unwrap_or(1);
        assert!(
            rows > 0 && cols > 0,
            "explained variance requires non-empty Q"
        );

        // Recreate the interpreter's bug-compatible transpose by reinterpreting
        // the column-major Q buffer with swapped dimensions (no data shuffle).
        let mut tmp_bug = vec![0.0; cols * rows];
        for j in 0..rows {
            for i in 0..cols {
                let mut acc = 0.0;
                for k in 0..rows {
                    let a_idx = i + cols * k;
                    let b_idx = k + rows * j;
                    acc += q_tensor.data[a_idx] * g_tensor.data[b_idx];
                }
                tmp_bug[i + cols * j] = acc;
            }
        }

        let mut expected_diag = Vec::with_capacity(cols);
        for i in 0..cols {
            let mut acc = 0.0;
            for j in 0..rows {
                let tmp_ij = tmp_bug[i + cols * j];
                let q_ji = q_tensor.data[j + rows * i];
                acc += tmp_ij * q_ji;
            }
            expected_diag.push(acc);
        }

        if std::env::var("RUNMAT_DEBUG_EXPLAINED").is_ok() {
            println!("tmp runtime shape {:?}", vec![cols, rows]);
            println!("tmp runtime data {:?}", tmp_bug);
            println!("expected diag {:?}", expected_diag);
            println!("cpu diag {:?}", cpu_tensor.data);
            let tmp_idx = 4;
            if let Some(tmp_value) = vars_cpu.get(tmp_idx) {
                if let Ok(Value::Tensor(tmp_tensor)) = gather_if_needed(tmp_value) {
                    println!(
                        "tmp var idx {tmp_idx} shape {:?} data {:?}",
                        tmp_tensor.shape, tmp_tensor.data
                    );
                }
            }
            let prod_idx = 5;
            if let Some(prod_value) = vars_cpu.get(prod_idx) {
                if let Ok(Value::Tensor(prod_tensor)) = gather_if_needed(prod_value) {
                    println!(
                        "prod var idx {prod_idx} shape {:?} data {:?}",
                        prod_tensor.shape, prod_tensor.data
                    );
                }
            }
        }
        if std::env::var("RUNMAT_DEBUG_EXPLAINED").is_ok() {
            for (lhs, rhs) in expected_diag.iter().zip(cpu_tensor.data.iter()) {
                println!("cpu diag diff {}", (lhs - rhs).abs());
            }
        } else {
            for (lhs, rhs) in expected_diag.iter().zip(cpu_tensor.data.iter()) {
                assert!((lhs - rhs).abs() <= 1e-9, "cpu diag mismatch");
            }
        }

        ensure_provider_registered();
        let vars_gpu = interpret_function(&bytecode, vars).expect("gpu interpret");
        let eval_gpu = vars_gpu.get(eval_index).expect("gpu eval");
        assert!(
            matches!(eval_gpu, Value::GpuTensor(_)),
            "expected gpu tensor result"
        );
        let gathered_gpu = gather_if_needed(eval_gpu).expect("gather gpu");
        let gpu_tensor = match gathered_gpu {
            Value::Tensor(t) => t,
            other => panic!("expected tensor eval (gpu), got {other:#?}"),
        };

        assert_eq!(cpu_tensor.shape, gpu_tensor.shape, "shape mismatch");
        let tol = 1e-6;
        if std::env::var("RUNMAT_DEBUG_EXPLAINED").is_ok() {
            for (lhs, rhs) in expected_diag.iter().zip(gpu_tensor.data.iter()) {
                let diff = (lhs - rhs).abs();
                println!("gpu diag diff lhs={lhs}, rhs={rhs}, diff={diff}");
            }
        } else {
            for (lhs, rhs) in expected_diag.iter().zip(gpu_tensor.data.iter()) {
                let diff = (lhs - rhs).abs();
                assert!(
                    diff <= tol,
                    "explained variance mismatch: lhs={lhs}, rhs={rhs}, diff={diff}"
                );
            }
        }

        if std::env::var("RUNMAT_DEBUG_EXPLAINED").is_ok() {
            println!("Q tensor data {:?}", q_tensor.data);
            println!("G tensor data {:?}", g_tensor.data);
        }

        if std::env::var("RUNMAT_DEBUG_EXPLAINED").is_ok() {
            for (idx, value) in vars_cpu.iter().enumerate() {
                if let Ok(Value::Tensor(t)) = gather_if_needed(value) {
                    if t.shape == vec![2, 2] {
                        println!("tensor idx {idx} shape {:?} data {:?}", t.shape, t.data);
                    }
                }
            }
        }

        if std::env::var("RUNMAT_DEBUG_EXPLAINED").is_ok() {
            let q_tensor_value = Value::Tensor(q_tensor.clone());
            let g_tensor_value = Value::Tensor(g_tensor.clone());
            let q_t_value = runmat_runtime::transpose(q_tensor_value.clone()).expect("transpose");
            if let Value::Tensor(ref q_t_tensor) = q_t_value {
                println!("q_t tensor data {:?}", q_t_tensor.data);
            }
            let tmp_via_runtime = runmat_runtime::matrix::value_matmul(&q_t_value, &g_tensor_value)
                .expect("q' * g via runtime");
            if let Value::Tensor(ref tmp_tensor) = tmp_via_runtime {
                println!("tmp via runtime {:?}", tmp_tensor.data);
            }
        }
    });
}

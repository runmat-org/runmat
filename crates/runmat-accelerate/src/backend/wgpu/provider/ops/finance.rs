use super::*;

impl WgpuProvider {
    pub(crate) fn black_scholes_price_exec(
        &self,
        request: &ProviderBlackScholesPriceRequest<'_>,
    ) -> Result<ProviderBlackScholesPriceResult> {
        ensure!(
            request.inputs.len() == 6,
            "black_scholes_price: expected six inputs"
        );
        ensure!(
            !request.output_shape.is_empty(),
            "black_scholes_price: missing output shape"
        );
        ensure!(
            request.output_shape.len() <= 8,
            "black_scholes_price: rank exceeds GPU kernel limit"
        );
        let total_len = request
            .output_shape
            .iter()
            .copied()
            .try_fold(1usize, |acc, extent| acc.checked_mul(extent))
            .ok_or_else(|| anyhow!("black_scholes_price: output size exceeds GPU limits"))?;
        ensure!(
            total_len == request.len,
            "black_scholes_price: output length does not match shape"
        );
        ensure!(
            total_len <= u32::MAX as usize,
            "black_scholes_price: tensor length exceeds GPU dispatch limits"
        );

        let entries = request
            .inputs
            .iter()
            .enumerate()
            .map(|(idx, input)| {
                ensure!(
                    input.shape.len() == request.output_shape.len()
                        && input.strides.len() == request.output_shape.len(),
                    "black_scholes_price: input {} broadcast metadata rank mismatch",
                    idx + 1
                );
                let input_len = input
                    .shape
                    .iter()
                    .copied()
                    .try_fold(1usize, |acc, extent| acc.checked_mul(extent))
                    .ok_or_else(|| {
                        anyhow!(
                            "black_scholes_price: input {} shape exceeds GPU limits",
                            idx + 1
                        )
                    })?;
                ensure!(
                    input_len <= u32::MAX as usize,
                    "black_scholes_price: input {} length exceeds GPU limits",
                    idx + 1
                );
                let entry = self.get_entry(input.handle)?;
                ensure!(
                    entry.storage != GpuTensorStorage::ComplexInterleaved,
                    "black_scholes_price: complex input {} is not supported",
                    idx + 1
                );
                ensure!(
                    entry.len == input_len,
                    "black_scholes_price: input {} shape does not match buffer length",
                    idx + 1
                );
                Ok(entry)
            })
            .collect::<Result<Vec<_>>>()?;

        let call_buffer =
            self.create_storage_buffer_checked(total_len, "runmat-black-scholes-call-out")?;
        let put_buffer =
            self.create_storage_buffer_checked(total_len, "runmat-black-scholes-put-out")?;

        if total_len > 0 {
            let chunk_capacity = (crate::backend::wgpu::config::MAX_DISPATCH_WORKGROUPS as usize)
                * crate::backend::wgpu::config::WORKGROUP_SIZE as usize;
            let mut offset = 0usize;
            while offset < total_len {
                let chunk_len = (total_len - offset).min(chunk_capacity);
                let shader = black_scholes_price_shader(
                    self.precision,
                    request,
                    total_len as u32,
                    offset as u32,
                    chunk_len as u32,
                );
                let shader_module =
                    self.device_ref()
                        .create_shader_module(wgpu::ShaderModuleDescriptor {
                            label: Some("runmat-black-scholes-price-shader"),
                            source: wgpu::ShaderSource::Wgsl(Cow::Owned(shader)),
                        });
                let bind_layout =
                    self.device_ref()
                        .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                            label: Some("runmat-black-scholes-price-layout"),
                            entries: &black_scholes_bind_layout_entries(),
                        });
                let pipeline_layout =
                    self.device_ref()
                        .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                            label: Some("runmat-black-scholes-price-pipeline-layout"),
                            bind_group_layouts: &[&bind_layout],
                            push_constant_ranges: &[],
                        });
                let pipeline = self.device_ref().create_compute_pipeline(
                    &crate::backend::wgpu::compat::wgpu_compute_pipeline_descriptor! {
                        label: Some("runmat-black-scholes-price-pipeline"),
                        layout: Some(&pipeline_layout),
                        module: &shader_module,
                        entry_point: "main",
                    },
                );
                let bind_group = self
                    .device_ref()
                    .create_bind_group(&wgpu::BindGroupDescriptor {
                        label: Some("runmat-black-scholes-price-bind"),
                        layout: &bind_layout,
                        entries: &[
                            wgpu::BindGroupEntry {
                                binding: 0,
                                resource: entries[0].buffer.as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 1,
                                resource: entries[1].buffer.as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 2,
                                resource: entries[2].buffer.as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 3,
                                resource: entries[3].buffer.as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 4,
                                resource: entries[4].buffer.as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 5,
                                resource: entries[5].buffer.as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 6,
                                resource: call_buffer.as_ref().as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 7,
                                resource: put_buffer.as_ref().as_entire_binding(),
                            },
                        ],
                    });
                let workgroups = crate::backend::wgpu::dispatch::common::dispatch_size(
                    chunk_len as u32,
                    crate::backend::wgpu::config::WORKGROUP_SIZE,
                );
                crate::backend::wgpu::dispatch::creation::run(
                    self.device_ref(),
                    self.queue_ref(),
                    &pipeline,
                    &bind_group,
                    workgroups,
                    "runmat-black-scholes-price-encoder",
                    "runmat-black-scholes-price-pass",
                );
                offset += chunk_len;
            }
        }

        Ok(ProviderBlackScholesPriceResult {
            call: self.register_existing_buffer(
                call_buffer,
                request.output_shape.to_vec(),
                total_len,
            ),
            put: self.register_existing_buffer(
                put_buffer,
                request.output_shape.to_vec(),
                total_len,
            ),
        })
    }
}

fn black_scholes_bind_layout_entries() -> [wgpu::BindGroupLayoutEntry; 8] {
    std::array::from_fn(|binding| {
        let read_only = binding <= 5;
        wgpu::BindGroupLayoutEntry {
            binding: binding as u32,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        }
    })
}

fn black_scholes_price_shader(
    precision: NumericPrecision,
    request: &ProviderBlackScholesPriceRequest<'_>,
    total: u32,
    offset: u32,
    chunk: u32,
) -> String {
    let ty = precision.as_str();
    let input_indices = request
        .inputs
        .iter()
        .map(|input| {
            wgsl_broadcast_index_expression(
                request.output_shape,
                input.shape,
                input.strides,
                "linear",
            )
        })
        .collect::<Vec<_>>();
    let workgroup = crate::backend::wgpu::config::WORKGROUP_SIZE;
    let max_finite = match precision {
        NumericPrecision::F64 => "1.7976931348623157e308",
        NumericPrecision::F32 => "3.4028234663852886e38",
    };

    format!(
        r#"
const MAX_FINITE_BLS: {ty} = {ty}({max_finite});

struct Tensor {{
  data: array<{ty}>,
}};

@group(0) @binding(0) var<storage, read> price_buf: Tensor;
@group(0) @binding(1) var<storage, read> strike_buf: Tensor;
@group(0) @binding(2) var<storage, read> rate_buf: Tensor;
@group(0) @binding(3) var<storage, read> time_buf: Tensor;
@group(0) @binding(4) var<storage, read> vol_buf: Tensor;
@group(0) @binding(5) var<storage, read> yield_buf: Tensor;
@group(0) @binding(6) var<storage, read_write> call_out: Tensor;
@group(0) @binding(7) var<storage, read_write> put_out: Tensor;

fn is_nan_bls(x: {ty}) -> bool {{
  return x != x;
}}

fn is_finite_bls(x: {ty}) -> bool {{
  return (x == x) && (abs(x) < MAX_FINITE_BLS);
}}

fn nan_bls() -> {ty} {{
  let zero = {ty}(0.0);
  return zero / zero;
}}

fn normcdf_bls(x: {ty}) -> {ty} {{
  let ax = abs(x);
  let t = {ty}(1.0) / ({ty}(1.0) + {ty}(0.2316419) * ax);
  let polynomial = t * ({ty}(0.319381530) + t * ({ty}(-0.356563782) + t * ({ty}(1.781477937) + t * ({ty}(-1.821255978) + t * {ty}(1.330274429)))));
  let upper = {ty}(1.0) - {ty}(0.3989422804014327) * exp({ty}(-0.5) * ax * ax) * polynomial;
  return select({ty}(1.0) - upper, upper, x >= {ty}(0.0));
}}

fn load_price(linear: u32) -> {ty} {{
  return price_buf.data[{price_index}];
}}

fn load_strike(linear: u32) -> {ty} {{
  return strike_buf.data[{strike_index}];
}}

fn load_rate(linear: u32) -> {ty} {{
  return rate_buf.data[{rate_index}];
}}

fn load_time(linear: u32) -> {ty} {{
  return time_buf.data[{time_index}];
}}

fn load_volatility(linear: u32) -> {ty} {{
  return vol_buf.data[{vol_index}];
}}

fn load_yield(linear: u32) -> {ty} {{
  return yield_buf.data[{yield_index}];
}}

@compute @workgroup_size({workgroup})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {{
  if (gid.x >= {chunk}u) {{
    return;
  }}
  let linear = gid.x + {offset}u;
  if (linear >= {total}u) {{
    return;
  }}

  let price = load_price(linear);
  let strike = load_strike(linear);
  let rate = load_rate(linear);
  let time = load_time(linear);
  let volatility = load_volatility(linear);
  let yield_rate = load_yield(linear);
  if (!(is_finite_bls(price) && is_finite_bls(strike) && is_finite_bls(rate) && is_finite_bls(time) && is_finite_bls(volatility) && is_finite_bls(yield_rate) && price >= {ty}(0.0) && strike > {ty}(0.0) && time >= {ty}(0.0) && volatility >= {ty}(0.0))) {{
    let nan = nan_bls();
    call_out.data[linear] = nan;
    put_out.data[linear] = nan;
    return;
  }}

  if ((time == {ty}(0.0)) || (volatility == {ty}(0.0))) {{
    let forward_price = price * exp(-yield_rate * time);
    let discounted_strike = strike * exp(-rate * time);
    call_out.data[linear] = max(forward_price - discounted_strike, {ty}(0.0));
    put_out.data[linear] = max(discounted_strike - forward_price, {ty}(0.0));
    return;
  }}

  let sqrt_time = sqrt(time);
  let d1 = (log(price / strike) + (rate - yield_rate + {ty}(0.5) * volatility * volatility) * time) / (volatility * sqrt_time);
  let d2 = d1 - volatility * sqrt_time;
  let discounted_price = price * exp(-yield_rate * time);
  let discounted_strike = strike * exp(-rate * time);
  call_out.data[linear] = discounted_price * normcdf_bls(d1) - discounted_strike * normcdf_bls(d2);
  put_out.data[linear] = discounted_strike * normcdf_bls(-d2) - discounted_price * normcdf_bls(-d1);
}}
"#,
        price_index = input_indices[0],
        strike_index = input_indices[1],
        rate_index = input_indices[2],
        time_index = input_indices[3],
        vol_index = input_indices[4],
        yield_index = input_indices[5],
        ty = ty,
        max_finite = max_finite,
        workgroup = workgroup,
        total = total,
        offset = offset,
        chunk = chunk,
    )
}

fn wgsl_broadcast_index_expression(
    output_shape: &[usize],
    input_shape: &[usize],
    input_strides: &[usize],
    linear: &str,
) -> String {
    let mut output_stride = 1usize;
    let mut terms = Vec::new();
    for ((&output_extent, &input_extent), &input_stride) in
        output_shape.iter().zip(input_shape).zip(input_strides)
    {
        if input_extent != 1 && output_extent != 0 {
            terms.push(format!(
                "((({linear} / {output_stride}u) % {output_extent}u) * {input_stride}u)"
            ));
        }
        output_stride = output_stride.saturating_mul(output_extent.max(1));
    }
    if terms.is_empty() {
        "0u".to_owned()
    } else {
        terms.join(" + ")
    }
}

#[cfg(test)]
mod tests {
    use crate::backend::wgpu::provider::{register_test_wgpu_provider, WgpuProviderOptions};
    use runmat_accelerate_api::{
        AccelProvider, HostTensorView, ProviderBlackScholesPriceInput,
        ProviderBlackScholesPriceRequest,
    };

    fn strides(shape: &[usize]) -> Vec<usize> {
        let mut out = Vec::with_capacity(shape.len());
        let mut stride = 1usize;
        for &extent in shape {
            out.push(stride);
            stride = stride.saturating_mul(extent.max(1));
        }
        out
    }

    #[test]
    fn black_scholes_price_wgpu_broadcasts_resident_inputs() {
        let Ok(provider) = register_test_wgpu_provider(WgpuProviderOptions::default()) else {
            return;
        };

        let price_shape = [2usize, 1usize];
        let strike_shape = [1usize, 3usize];
        let scalar_shape = [1usize, 1usize];
        let output_shape = [2usize, 3usize];
        let price = provider
            .upload(&HostTensorView {
                data: &[100.0, 110.0],
                shape: &price_shape,
            })
            .expect("upload price");
        let strike = provider
            .upload(&HostTensorView {
                data: &[90.0, 100.0, 110.0],
                shape: &strike_shape,
            })
            .expect("upload strike");
        let rate = provider
            .upload(&HostTensorView {
                data: &[0.05],
                shape: &scalar_shape,
            })
            .expect("upload rate");
        let time = provider
            .upload(&HostTensorView {
                data: &[0.5],
                shape: &scalar_shape,
            })
            .expect("upload time");
        let volatility = provider
            .upload(&HostTensorView {
                data: &[0.2],
                shape: &scalar_shape,
            })
            .expect("upload volatility");
        let yield_rate = provider
            .upload(&HostTensorView {
                data: &[0.0],
                shape: &scalar_shape,
            })
            .expect("upload yield");

        let price_strides = strides(&price_shape);
        let strike_strides = strides(&strike_shape);
        let scalar_strides = strides(&scalar_shape);
        let inputs = [
            ProviderBlackScholesPriceInput {
                handle: &price,
                shape: &price_shape,
                strides: &price_strides,
            },
            ProviderBlackScholesPriceInput {
                handle: &strike,
                shape: &strike_shape,
                strides: &strike_strides,
            },
            ProviderBlackScholesPriceInput {
                handle: &rate,
                shape: &scalar_shape,
                strides: &scalar_strides,
            },
            ProviderBlackScholesPriceInput {
                handle: &time,
                shape: &scalar_shape,
                strides: &scalar_strides,
            },
            ProviderBlackScholesPriceInput {
                handle: &volatility,
                shape: &scalar_shape,
                strides: &scalar_strides,
            },
            ProviderBlackScholesPriceInput {
                handle: &yield_rate,
                shape: &scalar_shape,
                strides: &scalar_strides,
            },
        ];

        let result = provider
            .black_scholes_price(&ProviderBlackScholesPriceRequest {
                inputs: &inputs,
                output_shape: &output_shape,
                len: 6,
            })
            .expect("black scholes price");
        let call = pollster::block_on(provider.download(&result.call)).expect("download call");
        let put = pollster::block_on(provider.download(&result.put)).expect("download put");
        assert_eq!(call.shape, output_shape);
        assert_eq!(put.shape, output_shape);

        let expected_call = [
            13.498517482637212,
            22.547751983647927,
            6.888728577680624,
            14.075384036381692,
            2.906471321592413,
            7.577601435448678,
        ];
        let expected_put = [
            1.276409565187162,
            0.32564406619787967,
            4.41971978051388,
            1.6063752392149624,
            10.190561644708993,
            4.8616917585652715,
        ];
        let tolerance = match provider.precision() {
            runmat_accelerate_api::ProviderPrecision::F64 => 2.0e-5,
            runmat_accelerate_api::ProviderPrecision::F32 => 3.0e-5,
        };
        for idx in 0..call.data.len() {
            assert!(
                (call.data[idx] - expected_call[idx]).abs() < tolerance,
                "call lane {idx}: got {}, expected {}",
                call.data[idx],
                expected_call[idx]
            );
            assert!(
                (put.data[idx] - expected_put[idx]).abs() < tolerance,
                "put lane {idx}: got {}, expected {}",
                put.data[idx],
                expected_put[idx]
            );
        }

        for handle in [
            &price,
            &strike,
            &rate,
            &time,
            &volatility,
            &yield_rate,
            &result.call,
            &result.put,
        ] {
            provider.free(handle).ok();
        }
    }
}

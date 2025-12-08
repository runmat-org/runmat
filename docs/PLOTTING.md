# Plotting in RunMat

RunMat’s plotting stack is designed around the same zero-copy GPU pipeline that powers Accelerate. Every builtin in `crates/runmat-runtime/src/builtins/plotting` produces `runmat-plot` primitives, which stream tensors directly into WebGPU compute shaders when a provider-backed device is available. Figures/axes/hold/subplot state lives in `state.rs`, so MATLAB calls such as `figure(2); subplot(2,1,1); hold on` all manipulate the shared registry before rendering.

## GPU-first data path

- Runtime tensors that already live on a GPU provider avoid host gathers entirely: we call `runmat_accelerate_api::export_wgpu_buffer`, wrap the buffer in a `HistogramGpuInputs`/`LineGpuInputs` struct, and dispatch compute kernels in `crates/runmat-plot/src/gpu`. Each shader ships both `f32` and `f64` variants, so adapters with `SHADER_F64` consume double-precision buffers directly (others fall back to the CPU path automatically).
- Figure lifecycle events (`FigureEvent::Created/Updated/Closed`) flow through the wasm bindings (`bindings/ts/src/index.ts`) so host shells can mount canvases per handle. Desktop shells use the same event stream to open/close winit windows without polling.
- Performance knobs such as `RUNMAT_PLOT_SCATTER_TARGET` and `RUNMAT_PLOT_SURFACE_VERTEX_BUDGET` cap GPU work per plot, and runtime/TS helpers mirror those setters so CLIs/web hosts can tune budgets programmatically.

## Bars and histograms with MATLAB semantics

- `bar` and `hist` share the `parse_bar_style_args` helper, so MATLAB-style options (`'FaceColor'`, `'EdgeColor'`, `'FaceAlpha'`, `'BarWidth'`, `'DisplayName'`, `'stacked'`) work uniformly across CPU and GPU paths. Grouped/stacked bars keep per-series offsets/strides on-device, and histogram bins reuse the same GPU packer so name/value styles stay zero-copy.
- `'Weights'` now flows end-to-end: `HistWeightsInput::to_gpu_weights` uploads host vectors (respecting `single`/`double`) or proxies provider buffers in place, the histogram compute shader performs float atomic adds per bin, and a follow-up pass scales by the requested normalization. Both `'probability'` and `'pdf'` modes divide by the true weighted total (and bin width for PDFs), so `hist(data,'Normalization','pdf','Weights',w)` matches MATLAB numerically even when everything stays on the GPU.
- CPU fallbacks receive the same behaviour: weighted samples run through `build_histogram_chart`, `apply_normalization` divides by the weighted sum, and degenerate bins (all samples identical) reuse the newly-plumbed totals so probability/PDF plots remain correct.
- `hist` mirrors MATLAB’s richer API: `[N, X] = hist(...)` now works without re-running the builtin, and the parser accepts `NumBins`, `BinWidth`, `BinLimits`, and the `'sqrt'`, `'sturges'`, or `'integers'` `BinMethod` tokens on top of the existing `'Normalization'`/`'Weights'` options. GPU execution stays zero-copy for any option set that produces uniform bin widths; non-uniform requests automatically fall back to the CPU path with the same semantics.

### Example

```matlab
% Weighted probability histogram with custom styling
data = gpuArray(randn(1, 200000));
w = linspace(0.5, 2.0, numel(data));
hist(data, 40, ...
    'Weights', w, ...
    'Normalization', 'probability', ...
    'FaceColor', [0.15 0.5 0.8], ...
    'EdgeColor', 'none', ...
    'DisplayName', 'Weighted pdf');
legend show;
```

The example calls the shared parser, keeps both the samples and weights on the provider GPU, computes weighted bin sums via `runmat-plot/src/gpu/histogram.rs`, and streams those vertices into the active figure without touching host memory. The same path also powers the multi-output `[counts, centers] = hist(...)` form and the additional `NumBins`/`BinWidth`/`BinLimits` controls when they describe uniform bins; other configurations transparently fall back to the CPU implementation.

## Scatter/line markers and edge colours

- `plot`, `scatter`, `scatter3`, and `stairs` now share the line/marker parser (`core::style`). Inline style strings (`plot(x1,y1,'r',x2,y2,'--')`), `'MarkerFaceColor'`, `'MarkerEdgeColor','flat'`, per-point size/colour arrays, `'LineStyleOrder'`, and `'filled'` all resolve to a `LineMarkerAppearance` structure that feeds both CPU + GPU packers.
- Marker metadata rides the zero-copy path: scatter/line shaders read per-point colours, sizes, and edge colours so dashed/marker-heavy plots no longer trigger CPU fallbacks.

### Example

```matlab
t = linspace(0, 8*pi, 2000);
plot(t, sin(t), 'LineWidth', 2, 'Color', [0 0.6 0.2], '--');
hold on;
scatter(t(1:20:end), sin(t(1:20:end)), 60, ...
    'Marker', 'o', ...
    'MarkerFaceColor', 'flat', ...
    'MarkerEdgeColor', 'flat', ...
    'DisplayName', 'Samples');
hold off;
```

## Where to look next

- Runtime builtins: `crates/runmat-runtime/src/builtins/plotting/**`
- GPU kernels + packers: `crates/runmat-plot/src/gpu`
- Wasm/TS bindings: `bindings/ts/src/index.ts` + `README.md`
- Live work tracker: `TEMP_WORK_CONTEXT.md`

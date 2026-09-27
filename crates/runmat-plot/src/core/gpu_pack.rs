use runmat_accelerate_api::WgpuEmission;

/// GPU context used by plot elements to build viewport-aware GPU geometry.
///
/// On wasm, some geometry generation is deferred until render-time so we can incorporate the
/// current viewport pixel size (e.g. to convert `LineWidth` pixels → data units).
pub struct GpuPackContext<'a> {
    pub emission: &'a WgpuEmission<'a>,
}

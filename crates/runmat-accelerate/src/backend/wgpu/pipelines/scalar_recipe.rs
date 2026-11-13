//! Immutable scalar shader ABI and launch selection, supplied by the provider
//! constructor. Lazy compilation must not re-read a different global selector.
use super::FrozenLinearLaunch;
use crate::backend::wgpu::{cache::key, types::NumericPrecision};

#[derive(Clone)]
pub(crate) struct ScalarLaunchRecipe {
    shader: String,
    launch: FrozenLinearLaunch,
    precision: NumericPrecision,
    layout: &'static str,
    valid_abi: bool,
}

impl ScalarLaunchRecipe {
    pub(super) fn new(
        precision: NumericPrecision,
        launch: FrozenLinearLaunch,
        layout: &'static str,
    ) -> Self {
        let source = match precision {
            NumericPrecision::F32 => super::SCALAR_SHADER_F32,
            NumericPrecision::F64 => super::SCALAR_SHADER_F64,
        };
        Self::from_source(precision, launch, layout, source)
    }

    pub(super) fn new_max(
        precision: NumericPrecision,
        launch: FrozenLinearLaunch,
        layout: &'static str,
        policy: crate::backend::wgpu::shaders::real_max::RealMaxPolicy,
    ) -> Self {
        let source = crate::backend::wgpu::shaders::real_max::scalar_shader(precision, policy);
        Self::from_source(precision, launch, layout, &source)
    }

    fn from_source(
        precision: NumericPrecision,
        launch: FrozenLinearLaunch,
        layout: &'static str,
        source: &str,
    ) -> Self {
        // These canonical scalar templates historically used a literal 512,
        // unlike the tokenized fusion templates. The old fixed launch was
        // consistent; this projection honors the constructor's device choice.
        let marker = "@workgroup_size(512)";
        let valid_abi = source.matches(marker).count() == 1;
        Self {
            shader: source.replace(marker, &format!("@workgroup_size({})", launch.workgroup())),
            launch,
            precision,
            layout,
            valid_abi,
        }
    }
    pub(super) fn shader(&self) -> &str {
        &self.shader
    }
    pub(crate) fn workgroup(&self) -> u32 {
        self.launch.workgroup()
    }
    pub(crate) fn chunk_capacity(&self) -> Option<u64> {
        if !self.valid_abi {
            return None;
        }
        u64::try_from(self.launch.chunk_capacity()?).ok()
    }
    pub(crate) fn uniform_requested_bytes(&self) -> u64 {
        let size = match self.precision {
            NumericPrecision::F32 => {
                std::mem::size_of::<crate::backend::wgpu::params::ScalarParamsF32>()
            }
            NumericPrecision::F64 => {
                std::mem::size_of::<crate::backend::wgpu::params::ScalarParamsF64>()
            }
        };
        (size as u64).max(1).div_ceil(256) * 256
    }
    pub(crate) fn key(&self) -> u64 {
        key::compute_pipeline_hash_bytes(
            self.shader.as_bytes(),
            self.layout,
            Some(self.launch.workgroup()),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn recipe(
        precision: NumericPrecision,
        workgroup: u32,
        layout: &'static str,
    ) -> ScalarLaunchRecipe {
        ScalarLaunchRecipe::new(
            precision,
            FrozenLinearLaunch::new(workgroup, 65_535).unwrap(),
            layout,
        )
    }

    #[cfg_attr(not(target_arch = "wasm32"), test)]
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    fn max_policy_freezes_distinct_shader_identity_with_same_scalar_geometry() {
        use crate::backend::wgpu::shaders::real_max::RealMaxPolicy;
        use crate::backend::wgpu::shaders::validation_tests::{validate_generated, RequiredCaps};
        for (precision, caps) in [
            (NumericPrecision::F32, RequiredCaps::None),
            (NumericPrecision::F64, RequiredCaps::Float64Int64),
        ] {
            let launch = FrozenLinearLaunch::new(64, 3).unwrap();
            let max = ScalarLaunchRecipe::new_max(
                precision,
                launch,
                "max-layout",
                RealMaxPolicy::IncludeNaNPreferPositiveZero,
            );
            let ordinary = ScalarLaunchRecipe::new(precision, launch, "max-layout");
            assert_ne!(max.key(), ordinary.key());
            assert_eq!(max.chunk_capacity(), ordinary.chunk_capacity());
            assert_eq!(
                max.uniform_requested_bytes(),
                ordinary.uniform_requested_bytes()
            );
            assert_eq!(max.chunk_capacity(), Some(192));
            let frozen_key = max.key();
            let later = ScalarLaunchRecipe::new_max(
                precision,
                FrozenLinearLaunch::new(128, 3).unwrap(),
                "max-layout",
                RealMaxPolicy::IncludeNaNPreferPositiveZero,
            );
            assert_ne!(max.key(), later.key());
            assert_eq!(max.key(), frozen_key);
            assert!(max.shader().contains("@workgroup_size(64)"));
            validate_generated("frozen scalar Max policy", max.shader(), 64, 8, caps);
        }
    }

    #[cfg_attr(not(target_arch = "wasm32"), test)]
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    fn frozen_nondefault_scalar_shader_and_tail_geometry_share_one_selection() {
        let first = recipe(NumericPrecision::F32, 64, "runmat-scalar-layout");
        let key = first.key();
        let later_selection = recipe(NumericPrecision::F32, 128, "runmat-scalar-layout");
        assert_ne!(key, later_selection.key());
        assert_ne!(
            key,
            recipe(NumericPrecision::F32, 64, "runmat-f32-real-scalar-layout").key()
        );
        assert_eq!(first.key(), key);
        assert!(first.shader().contains("@workgroup_size(64)"));
        assert!(!first.shader().contains("@WG@"));
        let chunk = first.chunk_capacity().unwrap();
        assert_eq!(chunk.div_ceil(chunk), 1);
        assert_eq!((chunk + 1).div_ceil(chunk), 2);
        assert_eq!(65_u64.div_ceil(u64::from(first.workgroup())), 2);
        assert_eq!(first.uniform_requested_bytes(), 256);
        assert_eq!(
            recipe(NumericPrecision::F64, 64, "runmat-scalar-layout").uniform_requested_bytes(),
            256
        );
        assert!(FrozenLinearLaunch::new(0, 65_535).is_none());
        let limited = ScalarLaunchRecipe::new(
            NumericPrecision::F32,
            FrozenLinearLaunch::new(64, 3).unwrap(),
            "runmat-scalar-layout",
        );
        assert_eq!(limited.chunk_capacity(), Some(192));
        assert_eq!(limited.workgroup(), 64);
    }

    #[cfg_attr(not(target_arch = "wasm32"), test)]
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    fn generated_double_scalar_abi_and_chunk_tail_are_valid() {
        let recipe = recipe(NumericPrecision::F64, 64, "runmat-scalar-layout");
        let shader = recipe.shader();
        crate::backend::wgpu::shaders::validation_tests::validate_generated(
            "frozen double scalar",
            shader,
            recipe.workgroup(),
            8,
            crate::backend::wgpu::shaders::validation_tests::RequiredCaps::Float64,
        );
        assert!(shader.contains("offset: u32"));
        assert!(shader.contains("total: u32"));
        assert!(shader.contains("let idx = params.offset + local"));
        assert!(shader.contains("A.data[idx]"));
        assert!(shader.contains("Out.data[idx]"));
        assert_eq!(
            std::mem::offset_of!(crate::backend::wgpu::params::ScalarParamsF64, offset),
            8
        );
        assert_eq!(
            std::mem::offset_of!(crate::backend::wgpu::params::ScalarParamsF64, total),
            12
        );
        assert_eq!(
            std::mem::offset_of!(crate::backend::wgpu::params::ScalarParamsF64, scalar),
            16
        );
        let capacity = recipe.chunk_capacity().unwrap();
        let total = capacity + 1;
        let chunks = [(0, capacity), (capacity, 1)];
        assert_eq!(total.div_ceil(capacity), chunks.len() as u64);
        for (offset, len) in chunks {
            assert!(offset + len <= total);
            assert_eq!(offset + (len - 1), (offset + len) - 1);
            assert!(
                len.div_ceil(u64::from(recipe.workgroup()))
                    <= u64::from(crate::backend::wgpu::config::MAX_DISPATCH_WORKGROUPS)
            );
        }
        assert_eq!(chunks[1].0, capacity);
        assert_eq!(chunks[1].1, 1);
    }
}

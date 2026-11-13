//! Initial selected RNG streams; ordinary `rng` mutation remains the same
//! context-owned language and GPU stream implementation.

impl super::RuntimeContext {
    pub fn install_selected_random_inputs(
        &self,
        inputs: &runmat_execution::host::NativeRandomInputs,
    ) -> Result<(), crate::RuntimeError> {
        self.service_ports()
            .require_selected_local_effect(runmat_types::EffectKind::Randomness)?;
        let mut language = self.state().random.try_borrow_mut().map_err(|source| {
            crate::build_runtime_error("selected language random stream is in use")
                .with_source(source)
                .build()
        })?;
        let mut gpu = self.state().gpu_random.try_borrow_mut().map_err(|source| {
            crate::build_runtime_error("selected GPU random stream is in use")
                .with_source(source)
                .build()
        })?;
        let language_state = crate::builtins::common::random::RngSnapshot::new(
            crate::builtins::common::random::RandomState::seeded(inputs.language_seed),
            Some(inputs.language_seed),
        );
        let gpu_state = crate::builtins::common::random::gpu::GpuRandomSnapshot {
            algorithm: runmat_accelerate_api::random::GpuRandomAlgorithm::Threefry4x64_20,
            seed: inputs.gpu_seed,
            position: 0,
            normal_transform: runmat_accelerate_api::random::GpuNormalTransform::BoxMuller,
        };
        *language = language_state.into();
        gpu.replace(gpu_state);
        Ok(())
    }
}

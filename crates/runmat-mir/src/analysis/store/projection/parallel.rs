use super::Projection;
use runmat_types::*;

impl Projection<'_> {
    pub(super) fn parallel(&self, manifest: &mut ParallelManifest) -> Result<(), String> {
        manifest
            .parfor_regions
            .retain(|region| self.contains(region.id.0.function));
        for region in &mut manifest.parfor_regions {
            self.function(&mut region.id.0.function)?;
            self.value(&mut region.loop_variable)?;
            self.fact(&mut region.iterable)?;
            for variable in &mut region.variables {
                self.variable(variable)?;
            }
            region.variables.sort_by_key(|variable| variable.value);
        }
        manifest
            .spmd_regions
            .retain(|region| self.contains(region.id.0.function));
        for region in &mut manifest.spmd_regions {
            self.function(&mut region.id.0.function)?;
            for variable in region.captures.iter_mut().chain(&mut region.outputs) {
                self.variable(variable)?;
            }
            region.captures.sort_by_key(|variable| variable.value);
            region.outputs.sort_by_key(|variable| variable.value);
        }
        manifest
            .distributed_values
            .retain(|value| self.contains(value.id.function));
        for value in &mut manifest.distributed_values {
            self.function(&mut value.id.function)?;
            self.owner(&mut value.owner)?;
            self.fact(&mut value.value)?;
            if let Some(coordination) = &mut value.coordination {
                self.function(&mut coordination.region.0.function)?;
            }
        }
        manifest
            .collectives
            .retain(|collective| self.contains(collective.id.region.0.function));
        for collective in &mut manifest.collectives {
            self.function(&mut collective.id.region.0.function)?;
            for value in [
                &mut collective.input,
                &mut collective.output,
                &mut collective.root,
                &mut collective.source,
                &mut collective.destination,
                &mut collective.tag,
                &mut collective.dimension,
                &mut collective.reducer,
            ]
            .into_iter()
            .flatten()
            {
                self.fact(value)?;
            }
        }
        manifest.parfor_regions.sort_by_key(|region| region.id);
        manifest.spmd_regions.sort_by_key(|region| region.id);
        manifest.distributed_values.sort_by_key(|value| value.id);
        manifest.collectives.sort_by_key(|collective| collective.id);
        manifest.validate().map_err(|error| {
            format!(
                "projected parallel manifest: {}: {}",
                error.path, error.message
            )
        })
    }

    fn variable(&self, variable: &mut ParallelVariableContract) -> Result<(), String> {
        self.value(&mut variable.value)?;
        self.fact(&mut variable.fact)
    }
}

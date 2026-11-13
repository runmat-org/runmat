//! Closed fixed numeric-egress operations. No device, queue, encoder or
//! privileged callback can escape this module.
use super::{same_authority, CanonicalWgpuEmissionAuthority, WgpuDeviceAccess};
use crate::{AccelProvider, GpuTensorHandle, WgpuBufferOwner};
use anyhow::{anyhow, ensure};
use runmat_execution::profiling::{CommandPhysicalSubmission, CommandProfilingReceipt};
use std::borrow::Borrow;

#[derive(Clone)]
pub struct WgpuNumericReadbackSource {
    provider: &'static dyn AccelProvider,
    source: GpuTensorHandle,
    buffer: WgpuBufferOwner,
    access: WgpuDeviceAccess,
}

impl WgpuNumericReadbackSource {
    /// The canonical provider must first match this exact source buffer with
    /// its allocation record. Only its retained fresh-device authority can
    /// construct this closed physical capability.
    pub fn new(
        authority: &CanonicalWgpuEmissionAuthority,
        provider: &dyn AccelProvider,
        source: &GpuTensorHandle,
        buffer: &WgpuBufferOwner,
    ) -> anyhow::Result<Self> {
        crate::authorize_numeric_readback(provider, source)?;
        let access = WgpuDeviceAccess {
            inner: authority.inner.clone(),
        };
        let owner = crate::numeric_readback::exact_source_owner(source.device_id())
            .ok_or_else(|| anyhow!("numeric readback owner disappeared"))?;
        ensure!(
            provider
                .wgpu_device_access()
                .is_some_and(|owned| owned.same_device(&access)),
            "numeric readback authority belongs to another provider"
        );
        buffer.validate_creation_scope()?;
        ensure!(
            buffer.usage().contains(wgpu::BufferUsages::COPY_SRC),
            "numeric readback source is not copyable"
        );
        Ok(Self {
            provider: owner,
            source: source.clone(),
            buffer: buffer.clone(),
            access,
        })
    }

    fn authorize(&self) -> anyhow::Result<()> {
        crate::authorize_numeric_readback(self.provider, &self.source)?;
        self.buffer.validate_creation_scope()?;
        Ok(())
    }

    pub fn ensure_device(&self, access: &WgpuDeviceAccess) -> anyhow::Result<()> {
        self.authorize()?;
        ensure!(
            self.access.same_device(access),
            "numeric readback belongs to another device authority"
        );
        Ok(())
    }

    fn wrap(&self, buffer: wgpu::Buffer) -> WgpuNumericReadbackBuffer {
        WgpuNumericReadbackBuffer {
            source: self.clone(),
            owner: WgpuBufferOwner::new(buffer),
        }
    }

    pub fn create_stage(&self, size: u64) -> anyhow::Result<WgpuNumericReadbackBuffer> {
        self.authorize()?;
        ensure!(
            size == self.buffer.size() && size > 0,
            "numeric readback stage must cover its exact source backing"
        );
        Ok(self.access.inner.gate.shared(|| {
            self.wrap(
                self.access
                    .inner
                    .device
                    .create_buffer(&wgpu::BufferDescriptor {
                        label: Some("runmat-numeric-readback-staging"),
                        size,
                        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                        mapped_at_creation: false,
                    }),
            )
        })?)
    }

    pub fn submit(
        &self,
        stage: &WgpuNumericReadbackBuffer,
        timestamps: bool,
        fence: bool,
        direct_fence: bool,
        command: Option<&CommandProfilingReceipt>,
    ) -> anyhow::Result<WgpuNumericReadbackSubmission> {
        self.authorize()?;
        stage.authorize()?;
        ensure!(
            stage
                .owner
                .usage()
                .contains(wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST),
            "numeric readback stage requires readable copy-destination storage"
        );
        ensure!(
            same_authority(&self.access.inner, &stage.source.access.inner)
                && self.source.view_identity() == stage.source.source.view_identity()
                && WgpuBufferOwner::ptr_eq(&self.buffer, &stage.source.buffer)
                && stage.owner.size() == self.buffer.size(),
            "numeric readback stage belongs to another source"
        );
        if timestamps {
            ensure!(
                self.access.enabled_features().contains(
                    wgpu::Features::TIMESTAMP_QUERY
                        | wgpu::Features::TIMESTAMP_QUERY_INSIDE_ENCODERS
                ),
                "numeric readback timestamp features unavailable"
            );
        }
        Ok(self.access.inner.gate.shared(|| {
            let device = &self.access.inner.device;
            let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("runmat-numeric-readback"),
            });
            let timing = timestamps.then(|| {
                let query_set = device.create_query_set(&wgpu::QuerySetDescriptor {
                    label: Some("runmat-numeric-readback-timestamps"),
                    ty: wgpu::QueryType::Timestamp,
                    count: 18,
                });
                let resolve = self.wrap(device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("runmat-numeric-readback-resolve"),
                    size: 16,
                    usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
                    mapped_at_creation: false,
                }));
                let staging = self.wrap(device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("runmat-numeric-readback-timing-stage"),
                    size: 16,
                    usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                    mapped_at_creation: false,
                }));
                WgpuNumericReadbackTimestampResources {
                    query_set,
                    resolve,
                    staging,
                }
            });
            if let Some(timing) = &timing {
                encoder.write_timestamp(&timing.query_set, 0);
            }
            encoder.copy_buffer_to_buffer(&self.buffer, 0, &stage.owner, 0, self.buffer.size());
            if let Some(timing) = &timing {
                encoder.write_timestamp(&timing.query_set, 1);
                encoder.resolve_query_set(&timing.query_set, 0..2, &timing.resolve.owner, 0);
                encoder.copy_buffer_to_buffer(
                    &timing.resolve.owner,
                    0,
                    &timing.staging.owner,
                    0,
                    16,
                );
            }
            let make_fence = |encoder: &mut wgpu::CommandEncoder| {
                let buffer = self.wrap(device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("runmat-numeric-readback-fence"),
                    size: 8,
                    usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                    mapped_at_creation: false,
                }));
                encoder.clear_buffer(&buffer.owner, 0, None);
                buffer
            };
            let fence = fence.then(|| make_fence(&mut encoder));
            let direct_fence = direct_fence.then(|| make_fence(&mut encoder));
            let encoded = encoder.finish();
            let physical = command.and_then(CommandProfilingReceipt::begin_physical_submission);
            crate::observation::record_source_region_wgpu_submission(&self.access);
            let submission = self.access.inner.queue.submit(Some(encoded));
            WgpuNumericReadbackSubmission {
                submission,
                physical,
                timestamps: timing,
                fence,
                direct_fence,
            }
        })?)
    }
}

#[derive(Clone)]
pub struct WgpuNumericReadbackBuffer {
    source: WgpuNumericReadbackSource,
    owner: WgpuBufferOwner,
}
impl WgpuNumericReadbackBuffer {
    fn authorize(&self) -> anyhow::Result<()> {
        self.source.authorize()?;
        self.owner.validate_creation_scope()?;
        Ok(())
    }
    pub fn creation_scope(&self) -> Option<crate::DeviceAllocationScope> {
        self.owner.creation_scope()
    }
    pub fn map_async(
        &self,
        callback: impl FnOnce(Result<(), wgpu::BufferAsyncError>) + wgpu::WasmNotSend + 'static,
    ) -> anyhow::Result<()> {
        self.authorize()?;
        ensure!(
            self.owner.usage().contains(wgpu::BufferUsages::MAP_READ),
            "numeric readback buffer is not host-readable"
        );
        Ok(self.source.access.inner.gate.shared(|| {
            self.owner
                .slice(..)
                .map_async(wgpu::MapMode::Read, callback)
        })?)
    }
    pub fn mapped_range(&self) -> anyhow::Result<wgpu::BufferView<'_>> {
        self.authorize()?;
        ensure!(
            self.owner.usage().contains(wgpu::BufferUsages::MAP_READ),
            "numeric readback buffer is not host-readable"
        );
        Ok(self
            .source
            .access
            .inner
            .gate
            .shared(|| self.owner.slice(..).get_mapped_range())?)
    }
}
impl Borrow<wgpu::Buffer> for WgpuNumericReadbackBuffer {
    fn borrow(&self) -> &wgpu::Buffer {
        self.owner.borrow()
    }
}

pub struct WgpuNumericReadbackTimestampResources {
    pub query_set: wgpu::QuerySet,
    pub resolve: WgpuNumericReadbackBuffer,
    pub staging: WgpuNumericReadbackBuffer,
}
pub struct WgpuNumericReadbackSubmission {
    pub submission: wgpu::SubmissionIndex,
    pub physical: Option<CommandPhysicalSubmission>,
    pub timestamps: Option<WgpuNumericReadbackTimestampResources>,
    pub fence: Option<WgpuNumericReadbackBuffer>,
    pub direct_fence: Option<WgpuNumericReadbackBuffer>,
}

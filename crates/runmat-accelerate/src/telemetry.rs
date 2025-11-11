use std::sync::atomic::{AtomicU64, Ordering};

use runmat_accelerate_api::{ProviderDispatchStats, ProviderTelemetry};

#[derive(Default)]
pub struct AccelTelemetry {
    fused_elementwise_count: AtomicU64,
    fused_elementwise_wall_ns: AtomicU64,
    fused_reduction_count: AtomicU64,
    fused_reduction_wall_ns: AtomicU64,
    matmul_count: AtomicU64,
    matmul_wall_ns: AtomicU64,
    upload_bytes: AtomicU64,
    download_bytes: AtomicU64,
}

impl AccelTelemetry {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn record_upload_bytes(&self, bytes: u64) {
        if bytes > 0 {
            self.upload_bytes.fetch_add(bytes, Ordering::Relaxed);
        }
    }

    pub fn record_download_bytes(&self, bytes: u64) {
        if bytes > 0 {
            self.download_bytes.fetch_add(bytes, Ordering::Relaxed);
        }
    }

    pub fn record_fused_elementwise(&self, wall_ns: u64) {
        self.fused_elementwise_count.fetch_add(1, Ordering::Relaxed);
        if wall_ns > 0 {
            self.fused_elementwise_wall_ns
                .fetch_add(wall_ns, Ordering::Relaxed);
        }
    }

    pub fn record_fused_reduction(&self, wall_ns: u64) {
        self.fused_reduction_count.fetch_add(1, Ordering::Relaxed);
        if wall_ns > 0 {
            self.fused_reduction_wall_ns
                .fetch_add(wall_ns, Ordering::Relaxed);
        }
    }

    pub fn record_matmul(&self, wall_ns: u64) {
        self.matmul_count.fetch_add(1, Ordering::Relaxed);
        if wall_ns > 0 {
            self.matmul_wall_ns.fetch_add(wall_ns, Ordering::Relaxed);
        }
    }

    pub fn reset(&self) {
        self.fused_elementwise_count.store(0, Ordering::Relaxed);
        self.fused_elementwise_wall_ns.store(0, Ordering::Relaxed);
        self.fused_reduction_count.store(0, Ordering::Relaxed);
        self.fused_reduction_wall_ns.store(0, Ordering::Relaxed);
        self.matmul_count.store(0, Ordering::Relaxed);
        self.matmul_wall_ns.store(0, Ordering::Relaxed);
        self.upload_bytes.store(0, Ordering::Relaxed);
        self.download_bytes.store(0, Ordering::Relaxed);
    }

    pub fn snapshot(
        &self,
        fusion_cache_hits: u64,
        fusion_cache_misses: u64,
        bind_group_cache_hits: u64,
        bind_group_cache_misses: u64,
        bind_group_cache_by_layout: Option<Vec<runmat_accelerate_api::BindGroupLayoutTelemetry>>,
    ) -> ProviderTelemetry {
        ProviderTelemetry {
            fused_elementwise: ProviderDispatchStats {
                count: self.fused_elementwise_count.load(Ordering::Relaxed),
                total_wall_time_ns: self.fused_elementwise_wall_ns.load(Ordering::Relaxed),
            },
            fused_reduction: ProviderDispatchStats {
                count: self.fused_reduction_count.load(Ordering::Relaxed),
                total_wall_time_ns: self.fused_reduction_wall_ns.load(Ordering::Relaxed),
            },
            matmul: ProviderDispatchStats {
                count: self.matmul_count.load(Ordering::Relaxed),
                total_wall_time_ns: self.matmul_wall_ns.load(Ordering::Relaxed),
            },
            upload_bytes: self.upload_bytes.load(Ordering::Relaxed),
            download_bytes: self.download_bytes.load(Ordering::Relaxed),
            fusion_cache_hits,
            fusion_cache_misses,
            bind_group_cache_hits,
            bind_group_cache_misses,
            bind_group_cache_by_layout,
        }
    }
}

fn saturating_duration_ns(duration: std::time::Duration) -> u64 {
    duration.as_nanos().min(u64::MAX as u128) as u64
}

impl AccelTelemetry {
    pub fn record_fused_elementwise_duration(&self, duration: std::time::Duration) {
        self.record_fused_elementwise(saturating_duration_ns(duration));
    }

    pub fn record_fused_reduction_duration(&self, duration: std::time::Duration) {
        self.record_fused_reduction(saturating_duration_ns(duration));
    }

    pub fn record_matmul_duration(&self, duration: std::time::Duration) {
        self.record_matmul(saturating_duration_ns(duration));
    }
}

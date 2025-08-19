#![cfg(feature = "wgpu")]

use crate::{AccelerateBackend, BufferHandle, DeviceInfo, DeviceKind, DeviceMatrix};
use anyhow::Result;
use once_cell::sync::OnceCell;
use runmat_builtins::Tensor;
use std::sync::Arc;

pub struct WgpuBuffer {
    pub len: usize,
}

impl BufferHandle for WgpuBuffer {
    fn len(&self) -> usize {
        self.len
    }
}

pub struct WgpuMatrix {
    rows: usize,
    cols: usize,
    buf: Arc<WgpuBuffer>,
}

impl DeviceMatrix for WgpuMatrix {
    fn rows(&self) -> usize {
        self.rows
    }
    fn cols(&self) -> usize {
        self.cols
    }
    fn as_buffer(&self) -> &dyn BufferHandle {
        &*self.buf
    }
}

pub struct WgpuBackend {
    info: DeviceInfo,
}

static INSTANCE: OnceCell<WgpuBackend> = OnceCell::new();

impl WgpuBackend {
    pub fn global() -> &'static WgpuBackend {
        INSTANCE.get_or_init(|| WgpuBackend {
            info: DeviceInfo {
                kind: DeviceKind::Wgpu,
                name: "wgpu-generic".to_string(),
                vendor: "unknown".to_string(),
                memory_bytes: None,
                compute_capability: None,
            },
        })
    }
}

impl AccelerateBackend for WgpuBackend {
    fn device_info(&self) -> DeviceInfo {
        self.info.clone()
    }

    fn upload_matrix(&self, host: &Tensor) -> Result<Box<dyn DeviceMatrix>> {
        // Placeholder: allocate logical handle only
        let buf = Arc::new(WgpuBuffer {
            len: host.data.len(),
        });
        Ok(Box::new(WgpuMatrix {
            rows: host.rows,
            cols: host.cols,
            buf,
        }))
    }

    fn download_matrix(&self, dev: &dyn DeviceMatrix) -> Result<Tensor> {
        // Placeholder: returns zeros with same shape
        Ok(Tensor::zeros(vec![dev.rows(), dev.cols()]))
    }

    fn elem_add(
        &self,
        a: &dyn DeviceMatrix,
        _b: &dyn DeviceMatrix,
    ) -> Result<Box<dyn DeviceMatrix>> {
        let buf = Arc::new(WgpuBuffer {
            len: a.as_buffer().len(),
        });
        Ok(Box::new(WgpuMatrix {
            rows: a.rows(),
            cols: a.cols(),
            buf,
        }))
    }
    fn elem_sub(
        &self,
        a: &dyn DeviceMatrix,
        _b: &dyn DeviceMatrix,
    ) -> Result<Box<dyn DeviceMatrix>> {
        let buf = Arc::new(WgpuBuffer {
            len: a.as_buffer().len(),
        });
        Ok(Box::new(WgpuMatrix {
            rows: a.rows(),
            cols: a.cols(),
            buf,
        }))
    }
    fn elem_mul(
        &self,
        a: &dyn DeviceMatrix,
        _b: &dyn DeviceMatrix,
    ) -> Result<Box<dyn DeviceMatrix>> {
        let buf = Arc::new(WgpuBuffer {
            len: a.as_buffer().len(),
        });
        Ok(Box::new(WgpuMatrix {
            rows: a.rows(),
            cols: a.cols(),
            buf,
        }))
    }
    fn elem_div(
        &self,
        a: &dyn DeviceMatrix,
        _b: &dyn DeviceMatrix,
    ) -> Result<Box<dyn DeviceMatrix>> {
        let buf = Arc::new(WgpuBuffer {
            len: a.as_buffer().len(),
        });
        Ok(Box::new(WgpuMatrix {
            rows: a.rows(),
            cols: a.cols(),
            buf,
        }))
    }
    fn elem_pow(
        &self,
        a: &dyn DeviceMatrix,
        _b: &dyn DeviceMatrix,
    ) -> Result<Box<dyn DeviceMatrix>> {
        let buf = Arc::new(WgpuBuffer {
            len: a.as_buffer().len(),
        });
        Ok(Box::new(WgpuMatrix {
            rows: a.rows(),
            cols: a.cols(),
            buf,
        }))
    }

    fn matmul(&self, a: &dyn DeviceMatrix, b: &dyn DeviceMatrix) -> Result<Box<dyn DeviceMatrix>> {
        let rows = a.rows();
        let cols = b.cols();
        let buf = Arc::new(WgpuBuffer { len: rows * cols });
        Ok(Box::new(WgpuMatrix { rows, cols, buf }))
    }

    fn transpose(&self, a: &dyn DeviceMatrix) -> Result<Box<dyn DeviceMatrix>> {
        let buf = Arc::new(WgpuBuffer {
            len: a.as_buffer().len(),
        });
        Ok(Box::new(WgpuMatrix {
            rows: a.cols(),
            cols: a.rows(),
            buf,
        }))
    }
}

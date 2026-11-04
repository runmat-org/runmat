//! The C++ boundary of K1's OCCT kernel (`exact_kernel*.cc`).
//!
//! A kernel operation crosses as one program of flat instructions over registry bodies; the
//! result is the output bodies (registry ids plus topology summaries) with each face's and each
//! boundary edge's origins. See `exact_kernel.hxx` for the instruction set and origin kinds.

#[cxx::bridge(namespace = "runmat_geometry_io::occt_kernel")]
pub(crate) mod bridge {
    #[derive(Debug, Clone, Default)]
    struct KInstr {
        op: u32,
        ints: Vec<i64>,
        reals: Vec<f64>,
        strs: Vec<String>,
    }

    #[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
    struct KOrigin {
        kind: u8,
        a: u32,
        b: u32,
        c: u32,
    }

    #[derive(Debug, Clone)]
    struct KFace {
        origins: Vec<KOrigin>,
        kept: bool,
        /// 0 plane, 1 cylinder, 2 other.
        surface: u8,
        /// normal(3) or axis point(3) + axis direction(3) + radius, then centroid(3), area,
        /// bbox min(3), bbox max(3): always 17 reals (unused entries zero).
        geom: Vec<f64>,
    }

    #[derive(Debug, Clone)]
    struct KEdge {
        f0: u32,
        f1: u32,
        laminar: bool,
        /// start(3), end(3), length.
        geom: Vec<f64>,
        origins: Vec<KOrigin>,
    }

    #[derive(Debug, Clone)]
    struct KVertex {
        edges: Vec<u32>,
        point: Vec<f64>,
    }

    #[derive(Debug, Clone)]
    struct KBody {
        id: u64,
        /// Which OUTPUT instruction of the program made it.
        output: u32,
        sheet: bool,
        faces: Vec<KFace>,
        edges: Vec<KEdge>,
        vertices: Vec<KVertex>,
    }

    #[derive(Debug, Clone)]
    struct KResult {
        bodies: Vec<KBody>,
        /// Names read from imported files, referenced by name origins.
        names: Vec<String>,
    }

    #[derive(Debug, Clone)]
    struct KProjected {
        body: u32,
        /// 0 edge, 1 silhouette of a face.
        kind: u8,
        entity: u32,
        visible: bool,
        /// x, y pairs in the view frame.
        points: Vec<f64>,
    }

    unsafe extern "C++" {
        include!("src/occt/exact_kernel_ffi.hxx");

        fn kernel_execute(program: &[KInstr], operands: &[u64]) -> Result<KResult>;
        fn kernel_release(id: u64);
        fn kernel_live_bodies() -> u64;
        fn kernel_ray_hits(bodies: &[u64], face_filter: &[i64], ray: &[f64]) -> Result<Vec<f64>>;
        fn kernel_extent(bodies: &[u64], axis: &[f64]) -> Result<Vec<f64>>;
        fn kernel_analyze(kind: u32, reals: &[f64], bodies: &[u64]) -> Result<Vec<f64>>;
        fn kernel_project(
            bodies: &[u64],
            transforms: &[f64],
            frame: &[f64],
            flags: u32,
            tolerance: f64,
        ) -> Result<Vec<KProjected>>;
        fn kernel_proximity(a: u64, ta: &[f64], b: u64, tb: &[f64]) -> Result<Vec<f64>>;
        fn kernel_export_step(document: &[KInstr], bodies: &[u64]) -> Result<Vec<u8>>;
        fn kernel_occt_version() -> String;
        fn kernel_step_products(path: &str) -> Result<Vec<String>>;
    }
}

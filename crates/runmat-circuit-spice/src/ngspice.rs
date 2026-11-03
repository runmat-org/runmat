//! The ngspice shared-library engine.
//!
//! ngspice (BSD-licensed) builds as a shared library with a small C interface, described in
//! the ngspice manual's "shared ngspice" chapter: `ngSpice_Init` registers callbacks,
//! `ngSpice_Circ` loads a deck from an array of lines, `ngSpice_Command` runs commands
//! (`run` blocks until the analysis finishes when not started in the background),
//! `ngSpice_CurPlot` / `ngSpice_AllVecs` list result vectors and `ngGet_Vec_Info` reads one.
//!
//! The library is opened at run time from, in order: `$RUNMAT_NGSPICE` (a file path), then
//! the platform's library names in the system search path and the usual install prefixes.
//! When none loads, [`Ngspice::load`] returns [`SpiceError::EngineUnavailable`] listing what
//! it tried and how to install ngspice.
//!
//! ngspice keeps one circuit state per process, so every call goes through a process-wide
//! lock; an [`Ngspice`] handle can be used from any thread.

use crate::netlist::Deck;
use crate::{AcResult, Analysis, OpPoint, SimResult, SpiceEngine, SpiceError, Sweep};
use libloading::{Library, Symbol};
use std::collections::BTreeMap;
use std::ffi::{c_char, c_int, c_void, CStr, CString};
use std::sync::{Mutex, OnceLock};

#[repr(C)]
struct NgComplex {
    re: f64,
    im: f64,
}

#[repr(C)]
struct VectorInfo {
    name: *const c_char,
    v_type: c_int,
    flags: i16,
    real: *const f64,
    complex: *const NgComplex,
    length: c_int,
}

type SendChar = unsafe extern "C" fn(*mut c_char, c_int, *mut c_void) -> c_int;
type SendStat = unsafe extern "C" fn(*mut c_char, c_int, *mut c_void) -> c_int;
type ControlledExit = unsafe extern "C" fn(c_int, bool, bool, c_int, *mut c_void) -> c_int;
type InitFn = unsafe extern "C" fn(
    Option<SendChar>,
    Option<SendStat>,
    Option<ControlledExit>,
    *mut c_void,
    *mut c_void,
    *mut c_void,
    *mut c_void,
) -> c_int;
type CommandFn = unsafe extern "C" fn(*const c_char) -> c_int;
type CircFn = unsafe extern "C" fn(*mut *mut c_char) -> c_int;
type CurPlotFn = unsafe extern "C" fn() -> *const c_char;
type AllVecsFn = unsafe extern "C" fn(*const c_char) -> *mut *mut c_char;
type VecInfoFn = unsafe extern "C" fn(*const c_char) -> *const VectorInfo;

/// Engine output lines, collected by the callbacks.
fn log() -> &'static Mutex<Vec<String>> {
    static LOG: OnceLock<Mutex<Vec<String>>> = OnceLock::new();
    LOG.get_or_init(|| Mutex::new(Vec::new()))
}

/// The process-wide lock and the loaded library.
fn library() -> &'static Mutex<Option<(String, Library)>> {
    static LIB: OnceLock<Mutex<Option<(String, Library)>>> = OnceLock::new();
    LIB.get_or_init(|| Mutex::new(None))
}

unsafe extern "C" fn on_char(s: *mut c_char, _id: c_int, _u: *mut c_void) -> c_int {
    if !s.is_null() {
        // SAFETY: ngspice passes a NUL-terminated string valid for the callback's duration.
        let t = unsafe { CStr::from_ptr(s) }.to_string_lossy().into_owned();
        if let Ok(mut l) = log().lock() {
            l.push(t);
        }
    }
    0
}

unsafe extern "C" fn on_stat(_s: *mut c_char, _id: c_int, _u: *mut c_void) -> c_int {
    0
}

unsafe extern "C" fn on_exit(
    status: c_int,
    _imm: bool,
    _quit: bool,
    _id: c_int,
    _u: *mut c_void,
) -> c_int {
    if let Ok(mut l) = log().lock() {
        l.push(format!("ngspice exited with status {status}"));
    }
    0
}

/// Candidate library locations for this platform.
pub fn candidates() -> Vec<String> {
    let mut v = Vec::new();
    if let Ok(p) = std::env::var("RUNMAT_NGSPICE") {
        v.push(p);
    }
    let names: &[&str] = if cfg!(target_os = "macos") {
        &["libngspice.dylib", "libngspice.0.dylib"]
    } else if cfg!(windows) {
        &["ngspice.dll", "libngspice-0.dll"]
    } else {
        &["libngspice.so.0", "libngspice.so"]
    };
    v.extend(names.iter().map(|s| s.to_string()));
    let prefixes: &[&str] = if cfg!(target_os = "macos") {
        &["/opt/homebrew/lib", "/usr/local/lib", "/opt/local/lib"]
    } else if cfg!(windows) {
        &["C:\\Spice64\\dll-vs", "C:\\Program Files\\ngspice\\bin"]
    } else {
        &[
            "/usr/lib",
            "/usr/local/lib",
            "/usr/lib/x86_64-linux-gnu",
            "/usr/lib/aarch64-linux-gnu",
        ]
    };
    for p in prefixes {
        for n in names {
            v.push(format!("{p}/{n}"));
        }
    }
    v
}

/// A handle on the loaded ngspice library.
#[derive(Clone, Debug)]
pub struct Ngspice {
    path: String,
}

impl Ngspice {
    /// Loads (once per process) and initialises ngspice.
    pub fn load() -> Result<Ngspice, SpiceError> {
        let mut guard = library().lock().map_err(|_| SpiceError::Simulation {
            message: "ngspice lock poisoned".into(),
            log: vec![],
        })?;
        if let Some((p, _)) = guard.as_ref() {
            return Ok(Ngspice { path: p.clone() });
        }
        let tried = candidates();
        for c in &tried {
            // SAFETY: loading a shared library runs its initialisers; ngspice's are benign.
            let Ok(lib) = (unsafe { Library::new(c) }) else {
                continue;
            };
            // SAFETY: the symbol has the documented `ngSpice_Init` signature.
            let ok = unsafe {
                match lib.get::<InitFn>(b"ngSpice_Init\0") {
                    Ok(init) => {
                        init(
                            Some(on_char),
                            Some(on_stat),
                            Some(on_exit),
                            std::ptr::null_mut(),
                            std::ptr::null_mut(),
                            std::ptr::null_mut(),
                            std::ptr::null_mut(),
                        );
                        true
                    }
                    Err(_) => false,
                }
            };
            if ok {
                *guard = Some((c.clone(), lib));
                return Ok(Ngspice { path: c.clone() });
            }
        }
        Err(SpiceError::EngineUnavailable {
            tried,
            hint: "Install ngspice built as a shared library (macOS: `brew install ngspice`; \
                   Debian/Ubuntu: `apt install libngspice0`; Windows: the ngspice DLL \
                   package) or set RUNMAT_NGSPICE to the library's path."
                .into(),
        })
    }

    pub fn path(&self) -> &str {
        &self.path
    }

    fn with<R>(&self, f: impl FnOnce(&Library) -> Result<R, SpiceError>) -> Result<R, SpiceError> {
        let guard = library().lock().map_err(|_| SpiceError::Simulation {
            message: "ngspice lock poisoned".into(),
            log: vec![],
        })?;
        let (_, lib) = guard
            .as_ref()
            .ok_or_else(|| SpiceError::EngineUnavailable {
                tried: vec![],
                hint: "ngspice was not loaded".into(),
            })?;
        f(lib)
    }
}

fn sym<'l, T>(lib: &'l Library, name: &[u8]) -> Result<Symbol<'l, T>, SpiceError> {
    // SAFETY: callers name ngspice entry points with their documented signatures.
    unsafe { lib.get::<T>(name) }.map_err(|e| SpiceError::Simulation {
        message: format!("ngspice entry point missing: {e}"),
        log: vec![],
    })
}

fn fail(message: String) -> SpiceError {
    let log = log().lock().map(|l| l.clone()).unwrap_or_default();
    SpiceError::Simulation { message, log }
}

/// Reads every vector of the current plot: name → (real data, complex data).
type Vectors = BTreeMap<String, (Vec<f64>, Vec<(f64, f64)>)>;

fn read_vectors(lib: &Library) -> Result<Vectors, SpiceError> {
    let cur: Symbol<CurPlotFn> = sym(lib, b"ngSpice_CurPlot\0")?;
    let all: Symbol<AllVecsFn> = sym(lib, b"ngSpice_AllVecs\0")?;
    let info: Symbol<VecInfoFn> = sym(lib, b"ngGet_Vec_Info\0")?;
    let mut out = Vectors::new();
    // SAFETY: ngspice returns NUL-terminated strings and a NULL-terminated array that stay
    // valid until the next command; we copy everything before releasing the lock.
    unsafe {
        let plot = cur();
        if plot.is_null() {
            return Err(fail("no current plot".into()));
        }
        let plot_name = CStr::from_ptr(plot).to_string_lossy().into_owned();
        let names = all(plot);
        if names.is_null() {
            return Ok(out);
        }
        let mut k = 0;
        loop {
            let p = *names.add(k);
            if p.is_null() {
                break;
            }
            k += 1;
            let name = CStr::from_ptr(p).to_string_lossy().into_owned();
            let full = CString::new(format!("{plot_name}.{name}")).unwrap();
            let vi = info(full.as_ptr());
            if vi.is_null() {
                continue;
            }
            let vi = &*vi;
            let n = vi.length.max(0) as usize;
            let mut re = Vec::new();
            let mut cx = Vec::new();
            if !vi.real.is_null() {
                re = std::slice::from_raw_parts(vi.real, n).to_vec();
            } else if !vi.complex.is_null() {
                cx = std::slice::from_raw_parts(vi.complex, n)
                    .iter()
                    .map(|c| (c.re, c.im))
                    .collect();
            }
            out.insert(name.to_lowercase(), (re, cx));
        }
    }
    Ok(out)
}

impl SpiceEngine for Ngspice {
    fn run(&mut self, deck: &Deck, analysis: &Analysis) -> Result<SimResult, SpiceError> {
        if let Ok(mut l) = log().lock() {
            l.clear();
        }
        self.with(|lib| {
            let circ: Symbol<CircFn> = sym(lib, b"ngSpice_Circ\0")?;
            let cmd: Symbol<CommandFn> = sym(lib, b"ngSpice_Command\0")?;
            let mut lines = deck.lines();
            let end = lines.pop();
            lines.push(format!(".{}", analysis.card()));
            lines.extend(end);
            let owned: Vec<CString> = lines
                .iter()
                .map(|l| CString::new(l.replace('\0', " ")).unwrap())
                .collect();
            let mut ptrs: Vec<*mut c_char> =
                owned.iter().map(|c| c.as_ptr() as *mut c_char).collect();
            ptrs.push(std::ptr::null_mut());
            // SAFETY: a NULL-terminated array of NUL-terminated lines, alive for the call.
            if unsafe { circ(ptrs.as_mut_ptr()) } != 0 {
                return Err(fail("ngspice rejected the circuit".into()));
            }
            let run = CString::new("run").unwrap();
            // SAFETY: a NUL-terminated command string.
            if unsafe { cmd(run.as_ptr()) } != 0 {
                return Err(fail(format!("{} analysis failed", analysis.kind())));
            }
            let vecs = read_vectors(lib)?;
            let r = convert(deck, analysis, vecs);
            let reset = CString::new("destroy all").unwrap();
            // SAFETY: as above.
            unsafe { cmd(reset.as_ptr()) };
            r
        })
    }
    fn describe(&self) -> String {
        format!("ngspice ({})", self.path)
    }
}

/// Maps ngspice vectors to results: `v(node)` or bare node names are voltages (renamed to
/// their nets), `<source>#branch` are source currents.
pub fn convert(deck: &Deck, analysis: &Analysis, vecs: Vectors) -> Result<SimResult, SpiceError> {
    let net_of: BTreeMap<String, String> = deck
        .node_of
        .iter()
        .map(|(net, node)| (node.to_lowercase(), net.clone()))
        .collect();
    let sources: BTreeMap<String, String> = deck
        .elements
        .iter()
        .map(|e| (e.name.to_lowercase(), e.name.clone()))
        .collect();
    let classify = |name: &str| -> Option<(bool, String)> {
        if let Some(s) = name.strip_suffix("#branch") {
            let s = s.strip_prefix("i(").unwrap_or(s);
            return sources.get(s).map(|n| (false, n.clone()));
        }
        let node = name
            .strip_prefix("v(")
            .and_then(|x| x.strip_suffix(')'))
            .unwrap_or(name);
        net_of.get(node).map(|n| (true, n.clone()))
    };
    let scale = match analysis {
        Analysis::Op => None,
        Analysis::Dc { .. } => Some(
            vecs.iter()
                .find(|(k, _)| k.ends_with("-sweep") || *k == "sweep")
                .map(|(_, v)| v.0.clone())
                .unwrap_or_default(),
        ),
        Analysis::Ac { .. } => Some(
            vecs.get("frequency")
                .map(|v| {
                    if v.0.is_empty() {
                        v.1.iter().map(|c| c.0).collect()
                    } else {
                        v.0.clone()
                    }
                })
                .unwrap_or_default(),
        ),
        Analysis::Tran { .. } => Some(vecs.get("time").map(|v| v.0.clone()).unwrap_or_default()),
    };
    match analysis {
        Analysis::Op => {
            let mut op = OpPoint::default();
            for (k, (re, _)) in &vecs {
                if let (Some((volt, n)), Some(v)) = (classify(k), re.first()) {
                    if volt {
                        op.voltages.insert(n, *v);
                    } else {
                        op.currents.insert(n, *v);
                    }
                }
            }
            for (net, node) in &deck.node_of {
                if node == "0" {
                    op.voltages.insert(net.clone(), 0.0);
                }
            }
            Ok(SimResult::Op(op))
        }
        Analysis::Ac { .. } => {
            let mut r = AcResult {
                freq: scale.unwrap_or_default(),
                ..Default::default()
            };
            for (k, (re, cx)) in vecs {
                let data = if cx.is_empty() {
                    re.iter().map(|v| (*v, 0.0)).collect()
                } else {
                    cx
                };
                match classify(&k) {
                    Some((true, n)) => {
                        r.voltages.insert(n, data);
                    }
                    Some((false, n)) => {
                        r.currents.insert(n, data);
                    }
                    None => {}
                }
            }
            Ok(SimResult::Ac(r))
        }
        Analysis::Dc { .. } | Analysis::Tran { .. } => {
            let mut s = Sweep {
                x: scale.unwrap_or_default(),
                ..Default::default()
            };
            for (k, (re, _)) in vecs {
                match classify(&k) {
                    Some((true, n)) => {
                        s.voltages.insert(n, re);
                    }
                    Some((false, n)) => {
                        s.currents.insert(n, re);
                    }
                    None => {}
                }
            }
            Ok(if matches!(analysis, Analysis::Dc { .. }) {
                SimResult::Dc(s)
            } else {
                SimResult::Tran(s)
            })
        }
    }
}

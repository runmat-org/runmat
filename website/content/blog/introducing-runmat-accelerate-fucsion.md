%%{init: {'theme': 'dark', 'flowchart': {'curve': 'linear'}}}%%
flowchart LR

  A["Your .m code"] --> B["Typed Graph (IR)"]
  B --> C["Planner (per step)"]
  B --> D["Profiler → Device Map"]
  D -. "size thresholds + transfer costs (refined at runtime)" .-> C

  subgraph CPU["CPU"]
    CJ["Ignition ➜ JIT (small arrays)<br/>Profiles hot loops; V8-style"]
    CB["CPU BLAS<br/>(Big CPU math or FP64)"]
  end

  subgraph GPU["GPU Fusion Engine (via WGPU)"]
    GF["Fuse back-to-back math into a bigger step<br/>(avoid re-scans)"]
    GR["Keep data resident on GPU until CPU needs it"]
  end

  classDef cpu fill:#e6e6e6,stroke:#999,color:#000
  classDef gpu fill:#e7f5e6,stroke:#7ab97a,color:#073b0b
  classDef mgr fill:#ffb000,stroke:#b37400,color:#2a1a00

  RM["Residency manager"]:::mgr
  C -. "partition / choose target / co-locate" .-> RM

  C -- "FP64 or big CPU-optimal" --> CB:::cpu
  C -- "small arrays" --> CJ:::cpu
  C -- "big or fuse-friendly" --> GF:::gpu

  RM <--> GF
  RM -. "avoid ping-pong" .- CJ
  RM -. "avoid ping-pong" .- CB

  GF --> GR:::gpu
  R(("Results (host-visible)"))
  GF --> R
  GR --> R
  CB --> R
  CJ --> R

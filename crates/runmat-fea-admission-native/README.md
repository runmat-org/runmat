# Native FEA admission

`NativeFeaAdmission` is the shared local lease adapter for the CLI and standalone native images. Constructing it captures only the typed organization selector. It reads credentials and starts its bounded transport owner only when an FEA root requests admission. Plain programs and untaken FEA branches do not contact Server.

The Runtime owns root CPU-wall accounting, cancellation and publication checks. The adapter implements start, cumulative usage and settlement with the existing Server client. It refuses dispatched requests: authenticated cluster admission remains with its original host.

Hosts may use `new(explicit_org)` to apply the normal explicit/ENV selector precedence, or `from_selection` for an already captured typed selector. Install the adapter in `RuntimeServicePorts::with_sync_fea_run_admission`; an explicit invocation host takes precedence over that session default. Do not replace an originating host with a newly granted local lease.

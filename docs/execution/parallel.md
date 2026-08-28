# Parallel Execution

RunMat supports `parfor`, process and browser worker pools, and asynchronous function execution through the Parallel Computing Toolbox-compatible APIs. The compiler analyzes each `parfor` before execution, assigns every referenced value a parallel role, and records the resulting contract in the executable artifact.

## Create and Inspect a Pool

`parpool` creates a pool or returns the compatible pool already owned by the current session:

```matlab
pool = parpool();             % use the host's automatic worker count
pool = parpool(4);            % request four workers
pool = parpool("processes", 4);
```

The available backend depends on the host. Native CLI and Desktop sessions use isolated local processes. The web runtime uses browser workers when the browser host provides them. Cluster execution installs a remote pool for work admitted to customer-managed or hosted nodes. A host rejects a pool kind or worker count that it cannot provide.

Use `gcp` to inspect the session pool:

```matlab
pool = gcp();                 % create an automatic pool when needed
pool = gcp("nocreate");      % return [] when no pool exists
```

Calling `delete(pool)` closes the pool and cancels unfinished child work. A later `parpool` call creates a new pool generation; handles from the closed generation are invalid.

## Run a `parfor` Loop

```matlab
pool = parpool(4);
values = zeros(1, 100);
total = 0;

parfor index = 1:100
  values(index) = expensiveCalculation(index);
  total = total + values(index);
end
```

The compiler classifies the loop variable and each value used by the body as loop, broadcast, sliced, reduction, temporary, or private state. Sliced assignments must use one consistent loop-indexed dimension. RunMat also recognizes constant affine offsets such as `values(index + offset)` when `offset` is a transferable broadcast scalar. Supported reductions use a single consistent reduction operator.

The compiled region contains its exact bytecode boundaries, frame slots, variable roles, effects, capabilities, and transfer requirements. Workers receive that region identity and its captured values; they do not reparse source or resolve functions against an ambient path.

RunMat divides the iteration space into deterministic bounded chunks and schedules those chunks across the admitted workers. Results are assembled by logical iteration order, so completion order does not change sliced outputs or reduction order. Deterministic random loops receive one reserved random stream per logical iteration and produce the same result when the worker count or completion order changes.

An optional worker limit applies to one loop:

```matlab
parfor (index = 1:100, 2)
  values(index) = expensiveCalculation(index);
end
```

Use a limit of zero to run the same analyzed loop serially in the coordinating process:

```matlab
parfor (index = 1:100, 0)
  values(index) = expensiveCalculation(index);
end
```

The serial path uses the same region contract and result semantics. It is useful when debugging or when a host cannot admit parallel workers.

Nested parallel regions, suspension, loop-control escape, ambiguous sliced writes, iteration-variable escape, and inconsistent reductions are rejected during analysis. A structurally valid loop can still be ineligible for worker placement when a captured value cannot cross the selected worker boundary or the body requires an effect that cannot run safely on a worker. In that case, RunMat executes the analyzed region through its serial correctness path instead of weakening its transfer or effect contract.

## Schedule Functions

`parfeval` schedules one function invocation and returns a future:

```matlab
future = parfeval(pool, @calculate, 1, input);
result = fetchOutputs(future);
```

Omit the pool to use the current or automatic pool:

```matlab
future = parfeval(@calculate, 1, input);
```

The numeric argument after the function handle is its requested output count. Multiple outputs remain distinct through scheduling and transfer:

```matlab
future = parfeval(pool, @minAndMax, 2, values);
[minimumValue, maximumValue] = fetchOutputs(future);
```

`parfevalOnAll` schedules the invocation once per worker and returns one aggregate future. `fetchNext` selects the next completed unread future and returns its one-based index followed by the function outputs:

```matlab
futures(1) = parfeval(pool, @calculate, 1, firstInput);
futures(2) = parfeval(pool, @calculate, 1, secondInput);

[index, result] = fetchNext(futures);
```

For an array of futures, `fetchOutputs(futures, "UniformOutput", false)` returns cell arrays instead of concatenating compatible outputs. `cancel(future)` requests cooperative cancellation. Work already completed remains completed; active worker work is cancelled at the next supported cancellation boundary.

## Inspect Worker Context

Code running on a scheduled worker can inspect its execution assignment:

```matlab
task = getCurrentTask();
worker = getCurrentWorker();
job = getCurrentJob();
```

`getCurrentTask` and `getCurrentWorker` return objects derived from the scheduler's typed assignment. `getCurrentJob` returns the durable job identity when the invocation belongs to a submitted job. Each function returns `[]` when its corresponding context is not active, including ordinary driver code.

## Errors, Retry, and Cancellation

A worker error retains its identifier, message, call stack, and source location when it returns to the coordinating process or browser. The coordinator cancels sibling chunks after a terminal loop failure and does not publish partially assembled loop outputs.

Compiler-approved `parfor` chunks use infrastructure-only retry. A worker or transport failure may be retried when the scheduler can prove that no successful result was committed. Program errors are not retried. Lost work that cannot be classified safely becomes indeterminate rather than being reported as success.

Cancellation is scoped. Cancelling a future, closing a pool, interrupting the parent execution, or cancelling a remote job propagates through the scheduler to its active worker attempts. Native calls and provider work remain cooperative and stop when they reach a cancellation boundary.

## Host Boundaries

Native, browser, and remote workers consume the same versioned program request, callable identity, parallel-region contract, value payload, assignment, and structured failure schemas. Platform-specific code owns process creation, browser worker launch, transport, and resource admission; it does not redefine the language or result assembly rules.

Remote execution uses the frozen package graph and exact executable bundle described in [Remote Execution](/docs/runtime/execution/remote). General distributed arrays, SPMD regions, ranks, and collectives are separate language capabilities and are not implied by a `parfor` pool.

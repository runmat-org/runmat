# Java Interoperability

RunMat can call Java classes from MATLAB-syntax source on native hosts. Java objects retain their JVM identity and remain owned by the RunMat session that created them.

## Configure Java

RunMat discovers Java from `[runtime.foreign.java].home`, `JAVA_HOME`, and supported platform locations, in that order. Version bounds and JVM options are checked before the process JVM starts:

```toml
[runtime.foreign.java]
minimum_version = 17
maximum_version = 21
classpath = ["lib/analysis.jar"]
options = ["-Xmx1g"]
```

The JVM starts on the first operation that needs it. Its installation and startup options cannot change afterward. `jenv` reports the selected installation and load state, while `usejava("jvm")` reports whether Java is available.

## Construct objects and call methods

Use `javaObject` and `javaMethod` with the same class and method spelling used by MATLAB code:

```matlab
builder = javaObject("java.lang.StringBuilder");
javaMethod("append", builder, "RunMat");
text = javaMethod("toString", builder);

largest = javaMethod("max", "java.lang.Math", 4, 9);
```

Fully qualified dotted calls are also resolved at the Java boundary after RunMat source functions and packages:

```matlab
value = java.lang.Integer.parseInt("42");
```

RunMat resolves constructors and overloads from Java reflection. Exact matches take precedence over widening conversions, assignable interfaces and base classes are supported, and unresolved ambiguities return an error instead of depending on reflection order.

## Values and objects

Boolean, numeric, character, string, and null scalars have reviewed Java mappings. Numeric vectors use matching primitive arrays where Java has the same signed type. Unsigned vectors widen to a signed primitive that can retain every value; `uint64` values use `java.math.BigInteger` when necessary. No fixed-width integer is routed through `double` solely to cross the Java boundary.

Java primitive arrays return typed RunMat row vectors. Java string arrays return RunMat string arrays, while reference arrays and collections preserve contained Java objects through session-owned foreign references. Multidimensional arrays created with `javaArray` remain Java objects so mutation and identity are not lost.

Java exceptions become RunMat errors with the Java class, message, cause chain, and Java stack frames.

## Classpaths

The configured `classpath` is the stable project layer. Use `javaaddpath`, `javarmpath`, and `javaclasspath` for the current session's dynamic layer:

```matlab
javaaddpath("lib/plugin.jar");
dynamicEntries = javaclasspath("-dynamic");
```

Classpath changes replace the session class loader without changing the process JVM's bootstrap path. Existing objects continue to use the loader that defined their classes. The session loader is also installed as the calling thread's context loader so libraries that use `ServiceLoader`, JDBC drivers, and similar discovery mechanisms can find project and dynamic artifacts.

## Callbacks and listeners

Pass a RunMat function handle where a Java method expects a functional interface or listener:

```matlab
operator = @(x) x + 2;
stream = javaMethod("iterate", "java.util.stream.IntStream", int32(1), operator);
```

Synchronous callbacks re-enter the originating RunMat session with its callable resolver, cancellation state, and foreign-runtime services. Java event objects remain Java objects and can be passed back to later Java calls. Reusing the same function handle for listener removal reuses the same Java proxy identity.

RunMat session values are thread-affine. A callback initiated on an unrelated Java worker thread returns an affinity error rather than accessing the session from that thread. Libraries that schedule callbacks on their own executors must marshal them to the RunMat host's supported callback thread.

## Event Dispatch Thread

`javaObjectEDT` and `javaMethodEDT` synchronously run their Java constructor or method on the AWT Event Dispatch Thread:

```matlab
label = javaObjectEDT("javax.swing.JLabel", "Ready");
javaMethodEDT("setText", label, "Running");
```

These functions require a Desktop host that advertises Java UI support. CLI, server, and browser sessions return `RunMat:Java:EdtUnavailable` instead of starting a Java UI thread implicitly. Calls made from the EDT execute directly; other Desktop calls use synchronous event-thread dispatch.

## Browser boundary

Browser and WebAssembly sessions do not embed a JVM. Java builtins remain present for source compatibility and return the stable `RunMat:Foreign:UnsupportedOnWasm` error when invoked. Projects that require Java artifacts must run on a native host with the matching capability.

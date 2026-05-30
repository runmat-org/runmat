---
title: "Hello World"
category: "Getting Started"
section: "1.7"
last_updated: "May 30, 2026"
---

# Run your first script

## Hello World in RunMat

Save the file as `hello.m` and add the following content:

```matlab
disp("Hello, World!");
```

Run the example with:

```bash
runmat hello.m
```

This should output:

```text
Hello, World!
```

You have now successfully run your first RunMat program!

## Running a Project

You can create multi-file projects by adding a `runmat.toml` file to the project root.

```toml
[package]
name = "my-project"

[sources]
roots = ["."]

[entrypoints.analysis]
path = "hello.m"
```

You can then run the project with:

```bash
runmat run analysis
```

This should output:

```text
Hello, World!
```

## Remote Projects

You can run RunMat projects hosted on a remote filesystem without copying the files to your local machine. This is useful when you want to run a project on a different machine, such as CI/CD pipelines, need to run scripts on a large GPU remote server, or if you need elastic scale for large datasets. 

RunMat server projects can scale to petabytes of data, with the virtual filesystem materializing the subset of data needed for the current script execution into the local filesystem temporarily as they are accessed, with an LRU cache to avoid unnecessary re-downloads.

To run a remote project, you need to authenticate with the remote server and select the project.

```bash
runmat login
runmat project select <project-id>
```

You can then run a script within the project with:

```bash
runmat remote run /scripts/analysis.m
```

This will run the `analysis.m` script on your local machine, with a virtual filesystem backed by the remote server mounted in the runtime's filesystem abstraction.

For a full project layout with source roots, packages, private functions, and class folders, see [Projects](/docs/runtime/getting-started/projects). For more details on remote filesystems, see the [Filesystem Abstraction](/docs/runtime/fs) documentation.

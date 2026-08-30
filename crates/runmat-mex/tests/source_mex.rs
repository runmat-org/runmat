#![cfg(not(target_family = "wasm"))]

use std::collections::BTreeMap;
use std::fs;
use std::rc::Rc;
use std::sync::Mutex;

use runmat_mex::{
    MexApi, MexBuild, MexDiagnostic, MexHostServices, MexLoadError, MexModule, MexSourceLanguage,
    MxApiMode,
};
use runmat_value::Value;

fn fortran_compiler_available() -> bool {
    std::process::Command::new("gfortran")
        .arg("--version")
        .output()
        .is_ok_and(|output| output.status.success())
}

#[test]
fn fortran_gateway_uses_native_handles_and_fortran_array_copy_routines() {
    if !fortran_compiler_available() {
        eprintln!("skipping Fortran MEX fixture because gfortran is unavailable");
        return;
    }
    let directory = tempfile::tempdir().unwrap();
    let source = directory.path().join("fortran_gateway.F");
    fs::write(
        &source,
        r#"
#include "fintrf.h"
      subroutine mexFunction(nlhs, plhs, nrhs, prhs)
      implicit none
      integer nlhs, nrhs
      mwPointer plhs(*), prhs(*)
      mwPointer input_data, output_data
      mwSize count
      mwPointer mxGetDoubles, mxCreateDoubleMatrix
      integer*4 mxIsDouble
      real*8 input_values(2), output_values(2)

      if (nrhs .ne. 1 .or. nlhs .ne. 1) then
         call mexErrMsgIdAndTxt('RunMat:test:arity',
     +        'expected one input and one output')
      endif
      if (mxIsDouble(prhs(1)) .eq. 0) then
         call mexErrMsgIdAndTxt('RunMat:test:type',
     +        'expected double input')
      endif
      count = 2
      input_data = mxGetDoubles(prhs(1))
      call mxCopyPtrToReal8(input_data, input_values, count)
      output_values(1) = input_values(1) * 2.0d0
      output_values(2) = input_values(2) * 2.0d0
      plhs(1) = mxCreateDoubleMatrix(1, 2, mxREAL)
      output_data = mxGetDoubles(plhs(1))
      call mxCopyReal8ToPtr(output_values, output_data, count)
      return
      end
"#,
    )
    .unwrap();

    let artifact = MexBuild::new(&source, directory.path()).compile().unwrap();
    assert_eq!(
        artifact.artifact.source_language,
        MexSourceLanguage::Fortran
    );
    let module = MexModule::load(&artifact.module).unwrap();
    assert_eq!(
        module.boundary_interface(),
        runmat_mex::MxBoundaryInterface::FortranMatrix
    );
    let error = module.invoke(&[], 1, module.api_mode()).unwrap_err();
    assert!(
        matches!(
            error,
            MexLoadError::Invocation { ref identifier, ref message }
                if identifier == " (RunMat:test:arity)"
                    && message == "expected one input and one output"
        ),
        "unexpected Fortran diagnostic: {error:?}"
    );
    let result = module
        .invoke(
            &[Value::Tensor(
                runmat_value::Tensor::new(vec![3.5, -2.0], vec![1, 2]).unwrap(),
            )],
            1,
            module.api_mode(),
        )
        .unwrap();
    let Value::Tensor(output) = &result.outputs[0] else {
        panic!("Fortran output must remain a tensor");
    };
    assert_eq!(output.materialize_f64(), vec![7.0, -4.0]);
}

#[test]
fn mixed_c_and_fortran_sources_use_their_own_compilers_and_one_fortran_link() {
    if !fortran_compiler_available() {
        eprintln!("skipping Fortran MEX fixture because gfortran is unavailable");
        return;
    }
    let directory = tempfile::tempdir().unwrap();
    let helper = directory.path().join("numeric_helper.c");
    let gateway = directory.path().join("mixed_gateway.F90");
    fs::write(
        &helper,
        "double scale_value_(const double *value) { return *value * 3.0; }\n",
    )
    .unwrap();
    fs::write(
        &gateway,
        r#"
#include "fintrf.h"
subroutine mexFunction(nlhs, plhs, nrhs, prhs)
  implicit none
  integer nlhs, nrhs
  mwPointer plhs(*), prhs(*)
  mwPointer mxCreateDoubleScalar
  real*8 scale_value
  plhs(1) = mxCreateDoubleScalar(scale_value(4.0d0))
end
"#,
    )
    .unwrap();

    let build = MexBuild::new(&helper, directory.path()).source(&gateway);
    let plan = build.plan().unwrap();
    assert!(plan.steps[0].arguments.contains(&"-std=c11".to_string()));
    assert!(plan.steps[1].arguments.contains(&"-std=legacy".to_string()));
    assert!(plan
        .steps
        .last()
        .unwrap()
        .compiler
        .file_name()
        .unwrap()
        .to_string_lossy()
        .starts_with("gfortran"));
    let artifact = build.compile().unwrap();
    assert_eq!(
        artifact.artifact.source_language,
        MexSourceLanguage::Fortran
    );
    let module = MexModule::load(&artifact.module).unwrap();
    let result = module.invoke(&[], 1, module.api_mode()).unwrap();
    assert_eq!(result.outputs, vec![Value::Num(12.0)]);
}

#[test]
fn fortran_gateway_preserves_interleaved_complex_aliases() {
    if !fortran_compiler_available() {
        eprintln!("skipping Fortran MEX fixture because gfortran is unavailable");
        return;
    }
    let directory = tempfile::tempdir().unwrap();
    let source = directory.path().join("fortran_complex_callback.F");
    fs::write(
        &source,
        r#"
#include "fintrf.h"
      subroutine mexFunction(nlhs, plhs, nrhs, prhs)
      implicit none
      integer nlhs, nrhs, status
      integer*4 mxCopyPtrToComplex16
      mwPointer plhs(*), prhs(*)
      mwSize count
      mwPointer complex_data
      mwPointer mxGetComplexDoubles
      complex*16 values(2)

      if (nrhs .ne. 1 .or. nlhs .ne. 1) then
         call mexErrMsgIdAndTxt('RunMat:test:arity',
     +        'expected one input and one output')
      endif
      count = 2
      complex_data = mxGetComplexDoubles(prhs(1))
      status = mxCopyPtrToComplex16(complex_data, values, count)
      if (status .ne. 1) then
         call mexErrMsgTxt('complex copy failed')
      endif
      plhs(1) = prhs(1)
      return
      end
"#,
    )
    .unwrap();

    let input =
        runmat_value::ComplexTensor::new(vec![(2.0, -3.0), (-0.0, 4.5)], vec![1, 2]).unwrap();
    let runmat_value::ComplexStorage::F64(input_buffer) = input.complex_storage() else {
        unreachable!("fixture creates complex double storage");
    };
    let input_address = unsafe { input_buffer.foreign_data_pointer() } as usize;
    let artifact = MexBuild::new(&source, directory.path())
        .api(MexApi::R2018a)
        .compile()
        .unwrap();
    let module = MexModule::load(&artifact.module).unwrap();
    let result = module
        .invoke_with_services(
            &[Value::ComplexTensor(input)],
            1,
            module.api_mode(),
            Rc::new(FixtureHost::default()),
        )
        .unwrap();
    let Value::ComplexTensor(output) = &result.outputs[0] else {
        panic!("aliased Fortran output must remain complex");
    };
    let runmat_value::ComplexStorage::F64(output_buffer) = output.complex_storage() else {
        panic!("Fortran output must retain complex double storage");
    };
    let output_address = unsafe { output_buffer.foreign_data_pointer() } as usize;
    assert_eq!(output_address, input_address);
    assert_eq!(output.materialize_f64(), vec![(2.0, -3.0), (-0.0, 4.5)]);
}

#[test]
fn fortran_separate_complex_copy_family_preserves_both_components() {
    if !fortran_compiler_available() {
        eprintln!("skipping Fortran MEX fixture because gfortran is unavailable");
        return;
    }
    let directory = tempfile::tempdir().unwrap();
    let source = directory.path().join("fortran_separate_complex.F");
    fs::write(
        &source,
        r#"
#include "fintrf.h"
      subroutine mexFunction(nlhs, plhs, nrhs, prhs)
      implicit none
      integer nlhs, nrhs
      mwPointer plhs(*), prhs(*)
      mwPointer real_input, imaginary_input
      mwPointer real_output, imaginary_output
      mwSize count
      mwPointer mxGetPr, mxGetPi, mxCreateDoubleMatrix
      complex*16 values(2)

      count = 2
      real_input = mxGetPr(prhs(1))
      imaginary_input = mxGetPi(prhs(1))
      call mxCopyPtrToComplex16(real_input, imaginary_input, values,
     +                          count)
      values(1) = values(1) * (2.0d0, 0.0d0)
      values(2) = values(2) * (2.0d0, 0.0d0)
      plhs(1) = mxCreateDoubleMatrix(1, 2, mxCOMPLEX)
      real_output = mxGetPr(plhs(1))
      imaginary_output = mxGetPi(plhs(1))
      call mxCopyComplex16ToPtr(values, real_output,
     +                          imaginary_output, count)
      return
      end
"#,
    )
    .unwrap();

    let artifact = MexBuild::new(&source, directory.path()).compile().unwrap();
    assert_eq!(artifact.artifact.api, MexApi::R2017b);
    let module = MexModule::load(&artifact.module).unwrap();
    let input =
        runmat_value::ComplexTensor::new(vec![(1.5, -2.0), (-4.0, 3.0)], vec![1, 2]).unwrap();
    let result = module
        .invoke(&[Value::ComplexTensor(input)], 1, module.api_mode())
        .unwrap();
    let Value::ComplexTensor(output) = &result.outputs[0] else {
        panic!("separate-complex Fortran output must remain complex");
    };
    assert_eq!(output.materialize_f64(), vec![(3.0, -4.0), (-8.0, 6.0)]);
}

#[test]
fn fortran_gateway_uses_character_arguments_for_callbacks_and_diagnostics() {
    if !fortran_compiler_available() {
        eprintln!("skipping Fortran MEX fixture because gfortran is unavailable");
        return;
    }
    let directory = tempfile::tempdir().unwrap();
    let source = directory.path().join("fortran_callback.F");
    fs::write(
        &source,
        r#"
#include "fintrf.h"
      subroutine mexFunction(nlhs, plhs, nrhs, prhs)
      implicit none
      integer nlhs, nrhs
      integer*4 status, mexCallMATLAB
      mwPointer plhs(*), prhs(*)
      mwPointer callback_inputs(1), callback_outputs(1)

      if (nrhs .ne. 1 .or. nlhs .ne. 1) then
         call mexErrMsgIdAndTxt('RunMat:test:arity',
     +        'expected one input and one output')
      endif
      callback_inputs(1) = prhs(1)
      status = mexCallMATLAB(1, callback_outputs, 1,
     +                       callback_inputs, 'plus_one')
      if (status .ne. 0) then
         call mexErrMsgTxt('callback failed')
      endif
      plhs(1) = callback_outputs(1)
      return
      end
"#,
    )
    .unwrap();

    let artifact = MexBuild::new(&source, directory.path()).compile().unwrap();
    let module = MexModule::load(&artifact.module).unwrap();
    let result = module
        .invoke_with_services(
            &[Value::Num(12.0)],
            1,
            module.api_mode(),
            Rc::new(FixtureHost::default()),
        )
        .unwrap();
    assert_eq!(result.outputs, vec![Value::Num(13.0)]);
}

#[test]
fn fortran_container_api_uses_fortran_index_and_field_number_conventions() {
    if !fortran_compiler_available() {
        eprintln!("skipping Fortran MEX fixture because gfortran is unavailable");
        return;
    }
    let directory = tempfile::tempdir().unwrap();
    let source = directory.path().join("fortran_containers.F");
    fs::write(
        &source,
        r#"
#include "fintrf.h"
      subroutine mexFunction(nlhs, plhs, nrhs, prhs)
      implicit none
      integer nlhs, nrhs
      integer*4 logical_value, mxIsLogicalScalarTrue
      integer*4 mxGetFieldNumber
      mwPointer plhs(*), prhs(*)
      mwPointer mxCreateCellMatrix, mxCreateStructMatrix
      mwPointer mxCreateDoubleScalar, mxCreateLogicalScalar
      mwPointer mxGetCell, mxGetFieldByNumber
      mwPointer number, flag
      character*8 field_names(1)

      field_names(1) = 'value'
      number = mxCreateDoubleScalar(7.0d0)
      logical_value = 1
      flag = mxCreateLogicalScalar(logical_value)
      plhs(1) = mxCreateCellMatrix(1, 1)
      call mxSetCell(plhs(1), 1, number)
      plhs(2) = mxCreateStructMatrix(1, 1, 1, field_names)
      call mxSetField(plhs(2), 1, 'value', flag)
      if (mxGetCell(plhs(1), 1) .ne. number) then
         call mexErrMsgTxt('one-based cell lookup failed')
      endif
      if (mxGetFieldNumber(plhs(2), 'value') .ne. 1) then
         call mexErrMsgTxt('one-based field number failed')
      endif
      if (mxIsLogicalScalarTrue(mxGetFieldByNumber(plhs(2), 1, 1))
     +    .eq. 0) then
         call mexErrMsgTxt('one-based field lookup failed')
      endif
      return
      end
"#,
    )
    .unwrap();

    let artifact = MexBuild::new(&source, directory.path()).compile().unwrap();
    let module = MexModule::load(&artifact.module).unwrap();
    let result = module.invoke(&[], 2, module.api_mode()).unwrap();
    let Value::Cell(cell) = &result.outputs[0] else {
        panic!("first Fortran output must be a cell array");
    };
    assert_eq!(cell.data, vec![Value::Num(7.0)]);
    let Value::Struct(structure) = &result.outputs[1] else {
        panic!("second Fortran output must be a structure");
    };
    assert_eq!(structure.fields.get("value"), Some(&Value::Bool(true)));
}

#[test]
fn fortran_compatible_array_dims_keeps_pointer_and_dimension_widths_distinct() {
    if !fortran_compiler_available() {
        eprintln!("skipping Fortran MEX fixture because gfortran is unavailable");
        return;
    }
    let directory = tempfile::tempdir().unwrap();
    let source = directory.path().join("fortran_compatible_dims.F");
    fs::write(
        &source,
        r#"
#include "fintrf.h"
      subroutine mexFunction(nlhs, plhs, nrhs, prhs)
      implicit none
      integer nlhs, nrhs
      mwPointer plhs(*), prhs(*)
      mwPointer mxCreateDoubleMatrix
      plhs(1) = mxCreateDoubleMatrix(2, 3, mxREAL)
      return
      end
"#,
    )
    .unwrap();
    let artifact = MexBuild::new(&source, directory.path())
        .api(MexApi::CompatibleArrayDims)
        .compile()
        .unwrap();
    let module = MexModule::load(&artifact.module).unwrap();
    let result = module.invoke(&[], 1, module.api_mode()).unwrap();
    let Value::Tensor(output) = &result.outputs[0] else {
        panic!("compatible-dimension Fortran output must be numeric");
    };
    assert_eq!(output.shape, vec![2, 3]);
    assert_eq!(output.materialize_f64(), vec![0.0; 6]);
}

#[test]
fn fortran_sparse_api_preserves_canonical_buffers_and_zero_based_csc_indices() {
    if !fortran_compiler_available() {
        eprintln!("skipping Fortran MEX fixture because gfortran is unavailable");
        return;
    }
    let directory = tempfile::tempdir().unwrap();
    let source = directory.path().join("fortran_sparse.F");
    fs::write(
        &source,
        r#"
#include "fintrf.h"
      subroutine mexFunction(nlhs, plhs, nrhs, prhs)
      implicit none
      integer nlhs, nrhs
      integer*4 mxIsSparse
      mwPointer plhs(*), prhs(*)
      mwPointer mxGetIr, mxGetJc, mxGetDoubles
      mwPointer rows_pointer, columns_pointer, values_pointer
      integer*8 rows(2), columns(3)
      real*8 values(2)

      if (nlhs .ne. 1 .or. nrhs .ne. 1) then
         call mexErrMsgTxt('expected one sparse input and output')
      endif
      if (mxIsSparse(prhs(1)) .eq. 0) then
         call mexErrMsgTxt('expected sparse input')
      endif
      rows_pointer = mxGetIr(prhs(1))
      columns_pointer = mxGetJc(prhs(1))
      values_pointer = mxGetDoubles(prhs(1))
      call mxCopyPtrToInteger8(rows_pointer, rows, 2)
      call mxCopyPtrToInteger8(columns_pointer, columns, 3)
      call mxCopyPtrToReal8(values_pointer, values, 2)
      if (rows(1) .ne. 1 .or. rows(2) .ne. 0) then
         call mexErrMsgTxt('sparse rows must remain zero based')
      endif
      if (columns(1) .ne. 0 .or. columns(2) .ne. 1 .or.
     +    columns(3) .ne. 2) then
         call mexErrMsgTxt('sparse column pointers changed')
      endif
      if (values(1) .ne. 3.0d0 .or. values(2) .ne. 4.0d0) then
         call mexErrMsgTxt('sparse values changed')
      endif
      plhs(1) = prhs(1)
      return
      end
"#,
    )
    .unwrap();

    let input =
        runmat_value::SparseTensor::new(2, 2, vec![0, 1, 2], vec![1, 0], vec![3.0, 4.0]).unwrap();
    let input_values =
        unsafe { input.numeric_host_buffer().unwrap().foreign_data_pointer() } as usize;
    let input_rows = unsafe { input.row_indices.foreign_data_pointer() } as usize;
    let input_columns = unsafe { input.col_ptrs.foreign_data_pointer() } as usize;
    let artifact = MexBuild::new(&source, directory.path()).compile().unwrap();
    let module = MexModule::load(&artifact.module).unwrap();
    let result = module
        .invoke(&[Value::SparseTensor(input)], 1, module.api_mode())
        .unwrap();
    let Value::SparseTensor(output) = &result.outputs[0] else {
        panic!("Fortran output must remain sparse");
    };
    assert_eq!(output.materialize_f64(), vec![3.0, 4.0]);
    assert_eq!(
        unsafe { output.numeric_host_buffer().unwrap().foreign_data_pointer() } as usize,
        input_values
    );
    assert_eq!(
        unsafe { output.row_indices.foreign_data_pointer() } as usize,
        input_rows
    );
    assert_eq!(
        unsafe { output.col_ptrs.foreign_data_pointer() } as usize,
        input_columns
    );
}

#[test]
fn fortran_lifecycle_callbacks_reuse_the_module_host_context() {
    if !fortran_compiler_available() {
        eprintln!("skipping Fortran MEX fixture because gfortran is unavailable");
        return;
    }
    let directory = tempfile::tempdir().unwrap();
    let source = directory.path().join("fortran_lifecycle.F");
    fs::write(
        &source,
        r#"
#include "fintrf.h"
      subroutine mexFunction(nlhs, plhs, nrhs, prhs)
      implicit none
      integer nlhs, nrhs
      integer*4 status, mexAtExit
      mwPointer plhs(*), prhs(*)
      external record_exit

      status = mexAtExit(record_exit)
      if (status .ne. 0) then
         call mexErrMsgTxt('could not register Fortran at-exit')
      endif
      call mexLock
      return
      end

      subroutine record_exit()
      implicit none
      integer*4 status, mexPutVariable
      mwPointer value, mxCreateDoubleScalar

      value = mxCreateDoubleScalar(73.0d0)
      status = mexPutVariable('base', 'fortran_exit', value)
      if (status .ne. 0) then
         call mexErrMsgTxt('Fortran at-exit workspace callback failed')
      endif
      call mxDestroyArray(value)
      return
      end
"#,
    )
    .unwrap();

    let artifact = MexBuild::new(&source, directory.path()).compile().unwrap();
    let host = Rc::new(FixtureHost::default());
    let module = MexModule::load(&artifact.module).unwrap();
    module
        .invoke_with_services(&[], 0, module.api_mode(), host.clone())
        .unwrap();
    assert!(module.is_locked());
    module.shutdown_with_services(host.clone()).unwrap();
    assert_eq!(
        host.workspace.lock().unwrap().get("fortran_exit"),
        Some(&Value::Num(73.0))
    );
}

#[test]
fn modern_cpp_data_api_shares_inputs_detaches_mutation_and_adopts_buffers() {
    let directory = tempfile::tempdir().unwrap();
    let source = directory.path().join("modern_data_api.cpp");
    fs::write(
        &source,
        r#"
#include "mex.hpp"
#include "mexAdapter.hpp"
#include <cstdint>
#include <vector>

class MexFunction : public matlab::mex::Function {
public:
    void operator()(matlab::mex::ArgumentList outputs,
                    matlab::mex::ArgumentList inputs) override {
        if (outputs.size() != 5 || inputs.size() != 1) {
            throw matlab::Exception("expected one input and five outputs");
        }
        matlab::data::ArrayFactory factory;

        outputs[0] = inputs[0];

        matlab::data::TypedArray<double> changed(inputs[0]);
        changed[0] = 19.0;
        outputs[1] = changed;

        auto buffer = factory.createBuffer<double>(2);
        double *allocation = buffer.get();
        buffer.get()[0] = 7.0;
        buffer.get()[1] = -4.0;
        outputs[2] = factory.createArrayFromBuffer<double>({1, 2}, std::move(buffer));
        outputs[3] = factory.createScalar<std::uint64_t>(
            static_cast<std::uint64_t>(reinterpret_cast<std::uintptr_t>(allocation)));

        std::vector<matlab::data::Array> callbackInputs{
            factory.createScalar<double>(5.0)};
        outputs[4] = getEngine()->feval(u"plus_one", callbackInputs);
    }
};
"#,
    )
    .unwrap();

    let input = runmat_value::Tensor::new(vec![3.0, 4.0], vec![1, 2]).unwrap();
    // SAFETY: the test observes the address only while the input or its shared
    // output owns the canonical host allocation.
    let input_address = unsafe { input.host_buffer().foreign_data_pointer() } as usize as u64;
    let artifact = MexBuild::new(&source, directory.path()).compile().unwrap();
    assert_eq!(artifact.artifact.source_language, MexSourceLanguage::Cxx);
    assert_eq!(artifact.artifact.api, MexApi::R2018a);
    let module = MexModule::load(&artifact.module).unwrap();
    let copies_before =
        runmat_value::host_copy_metrics(runmat_value::HostCopyReason::CopyOnWriteMutation);
    let result = module
        .invoke_with_services(
            &[Value::Tensor(input)],
            5,
            module.api_mode(),
            Rc::new(FixtureHost::default()),
        )
        .unwrap();
    let copies_after =
        runmat_value::host_copy_metrics(runmat_value::HostCopyReason::CopyOnWriteMutation);
    assert!(copies_after.operations > copies_before.operations);
    assert!(copies_after.bytes >= copies_before.bytes + 2 * std::mem::size_of::<f64>() as u64);

    let Value::Tensor(shared) = &result.outputs[0] else {
        panic!("shared C++ output must remain a tensor");
    };
    let shared_address = unsafe { shared.host_buffer().foreign_data_pointer() } as usize as u64;
    assert_eq!(shared_address, input_address);
    assert_eq!(shared.materialize_f64(), vec![3.0, 4.0]);

    let Value::Tensor(changed) = &result.outputs[1] else {
        panic!("mutated C++ copy must remain a tensor");
    };
    assert_eq!(changed.materialize_f64(), vec![19.0, 4.0]);
    let changed_address = unsafe { changed.host_buffer().foreign_data_pointer() } as usize as u64;
    assert_ne!(changed_address, input_address);

    let Value::Tensor(adopted) = &result.outputs[2] else {
        panic!("buffer-created C++ output must remain a tensor");
    };
    let adopted_address = unsafe { adopted.host_buffer().foreign_data_pointer() } as usize as u64;
    let Value::Int(recorded_address) = &result.outputs[3] else {
        panic!("recorded buffer address must remain uint64");
    };
    assert_eq!(recorded_address.try_to_u64(), Some(adopted_address));
    assert_eq!(adopted.materialize_f64(), vec![7.0, -4.0]);
    assert_eq!(result.outputs[4], Value::Num(6.0));
}

#[test]
fn modern_cpp_array_control_can_retain_an_input_across_gateway_calls() {
    let directory = tempfile::tempdir().unwrap();
    let source = directory.path().join("retained_data_api_input.cpp");
    fs::write(
        &source,
        r#"
#include "mex.hpp"
#include "mexAdapter.hpp"

class MexFunction : public matlab::mex::Function {
public:
    void operator()(matlab::mex::ArgumentList outputs,
                    matlab::mex::ArgumentList inputs) override {
        if (!inputs.empty()) {
            retained_ = inputs[0];
            return;
        }
        if (outputs.size() != 1 || !retained_) {
            throw matlab::Exception("expected one retained input");
        }
        outputs[0] = retained_;
    }

private:
    matlab::data::Array retained_;
};
"#,
    )
    .unwrap();

    let input = runmat_value::Tensor::new(vec![11.0, 12.0], vec![1, 2]).unwrap();
    let input_address = unsafe { input.host_buffer().foreign_data_pointer() } as usize;
    let artifact = MexBuild::new(&source, directory.path()).compile().unwrap();
    let module = MexModule::load(&artifact.module).unwrap();
    module
        .invoke(&[Value::Tensor(input)], 0, module.api_mode())
        .unwrap();
    let result = module.invoke(&[], 1, module.api_mode()).unwrap();
    let Value::Tensor(retained) = &result.outputs[0] else {
        panic!("retained C++ input must remain a tensor");
    };
    assert_eq!(retained.materialize_f64(), vec![11.0, 12.0]);
    assert_eq!(
        unsafe { retained.host_buffer().foreign_data_pointer() } as usize,
        input_address
    );
    drop(module);
}

#[test]
fn modern_cpp_data_api_preserves_aggregate_complex_and_layout_semantics() {
    let directory = tempfile::tempdir().unwrap();
    let source = directory.path().join("modern_aggregate_api.cpp");
    fs::write(
        &source,
        r#"
#include "mex.hpp"
#include "mexAdapter.hpp"
#include <complex>
#include <cstdint>

class MexFunction : public matlab::mex::Function {
public:
    void operator()(matlab::mex::ArgumentList outputs,
                    matlab::mex::ArgumentList inputs) override {
        (void)inputs;
        matlab::data::ArrayFactory factory;
        outputs[0] = factory.createCellArray(
            {1, 2}, factory.createScalar<std::uint64_t>(UINT64_MAX),
            factory.createCharArray("cell"));

        auto record = factory.createStructArray({1, 1}, {"value", "label"});
        record[0]["value"] = factory.createScalar<std::int32_t>(-17);
        record[0]["label"] = factory.createCharArray("record");
        matlab::data::Reference<matlab::data::Array> field = record[0]["value"];
        (void)field;
        outputs[1] = record;

        auto complexValues = factory.createArray<std::complex<double>>(
            {1, 2}, {{3.0, -4.0}, {5.0, 12.0}});
        complexValues[1] = std::complex<double>(8.0, -6.0);
        outputs[2] = complexValues;

        auto rowMajor = factory.createBuffer<double>(6);
        for (std::size_t index = 0; index < 6; ++index) {
            rowMajor.get()[index] = static_cast<double>(index + 1);
        }
        outputs[3] = factory.createArrayFromBuffer<double>(
            {2, 3}, std::move(rowMajor), matlab::data::MemoryLayout::ROW_MAJOR);

        auto complexBuffer = factory.createBuffer<std::complex<double>>(2);
        std::complex<double> *complexAllocation = complexBuffer.get();
        complexBuffer.get()[0] = {2.0, -3.0};
        complexBuffer.get()[1] = {5.0, 7.0};
        outputs[4] = factory.createArrayFromBuffer<std::complex<double>>(
            {1, 2}, std::move(complexBuffer));
        outputs[5] = factory.createScalar<std::uint64_t>(
            static_cast<std::uint64_t>(
                reinterpret_cast<std::uintptr_t>(complexAllocation)));
    }
};
"#,
    )
    .unwrap();

    let artifact = MexBuild::new(&source, directory.path()).compile().unwrap();
    let module = MexModule::load(&artifact.module).unwrap();
    let layout_copies_before =
        runmat_value::host_copy_metrics(runmat_value::HostCopyReason::MemoryLayoutConversion);
    let result = module.invoke(&[], 6, module.api_mode()).unwrap();
    let layout_copies_after =
        runmat_value::host_copy_metrics(runmat_value::HostCopyReason::MemoryLayoutConversion);
    assert_eq!(
        layout_copies_after.operations,
        layout_copies_before.operations + 1
    );
    assert_eq!(
        layout_copies_after.bytes,
        layout_copies_before.bytes + 6 * std::mem::size_of::<f64>() as u64
    );

    let Value::Cell(cell) = &result.outputs[0] else {
        panic!("C++ cell output must remain a cell");
    };
    assert_eq!(
        cell.get(0, 0).unwrap(),
        Value::Int(runmat_value::IntValue::U64(u64::MAX))
    );
    assert_eq!(
        cell.get(0, 1).unwrap(),
        Value::CharArray(runmat_value::CharArray::new_row("cell"))
    );

    let Value::Struct(record) = &result.outputs[1] else {
        panic!("C++ struct output must remain a struct");
    };
    assert_eq!(
        record.fields.get("value"),
        Some(&Value::Int(runmat_value::IntValue::I32(-17)))
    );
    assert_eq!(
        record.fields.get("label"),
        Some(&Value::CharArray(runmat_value::CharArray::new_row(
            "record"
        )))
    );

    let Value::ComplexTensor(complex) = &result.outputs[2] else {
        panic!("C++ complex output must remain complex");
    };
    assert_eq!(complex.materialize_f64(), vec![(3.0, -4.0), (8.0, -6.0)]);

    let Value::Tensor(row_major) = &result.outputs[3] else {
        panic!("row-major C++ buffer output must remain a tensor");
    };
    assert_eq!(row_major.shape, vec![2, 3]);
    assert_eq!(
        row_major.materialize_f64(),
        vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]
    );

    let Value::ComplexTensor(adopted_complex) = &result.outputs[4] else {
        panic!("buffer-created C++ complex output must remain complex");
    };
    assert_eq!(
        adopted_complex.materialize_f64(),
        vec![(2.0, -3.0), (5.0, 7.0)]
    );
    let Value::Int(recorded_address) = &result.outputs[5] else {
        panic!("recorded complex buffer address must remain uint64");
    };
    let adopted_address = adopted_complex
        .as_f64_slice()
        .expect("double-complex host buffer")
        .as_ptr() as usize as u64;
    assert_eq!(recorded_address.try_to_u64(), Some(adopted_address));
}

#[test]
fn modern_cpp_string_arrays_remain_distinct_from_character_arrays() {
    let directory = tempfile::tempdir().unwrap();
    let source = directory.path().join("modern_strings.cpp");
    fs::write(
        &source,
        r#"
#include "mex.hpp"
#include "mexAdapter.hpp"

class MexFunction : public matlab::mex::Function {
public:
    void operator()(matlab::mex::ArgumentList outputs,
                    matlab::mex::ArgumentList inputs) override {
        matlab::data::StringArray strings(inputs[0]);
        matlab::data::MATLABString first = strings[0];
        if (strings.getType() != matlab::data::ArrayType::MATLAB_STRING ||
            !first || first.value() != u"alpha") {
            throw matlab::Exception("string input did not retain its Data API type");
        }
        outputs[0] = inputs[0];
        strings[1] = matlab::data::String(u"changed");
        outputs[1] = strings;

        matlab::data::ArrayFactory factory;
        outputs[2] = factory.createArray<matlab::data::MATLABString>(
            {1, 2}, {matlab::data::MATLABString(u"βeta"),
                     matlab::data::MATLABString(u"雪")});
        outputs[3] = factory.createCharArray(u"chars");
        auto missing = factory.createArray<matlab::data::MATLABString>({1, 2});
        if (static_cast<matlab::data::MATLABString>(missing[0])) {
            throw matlab::Exception("default string element is not missing");
        }
        missing[1] = matlab::data::String(u"value");
        outputs[4] = missing;
        outputs[5] = factory.createScalar(matlab::data::MATLABString());
        outputs[6] = factory.createCellArray({1, 1}, std::string("text"));
    }
};
"#,
    )
    .unwrap();

    let input = Value::StringArray(
        runmat_value::StringArray::new(vec!["alpha".into(), "untouched".into()], vec![1, 2])
            .unwrap(),
    );
    let artifact = MexBuild::new(&source, directory.path()).compile().unwrap();
    let module = MexModule::load(&artifact.module).unwrap();
    let result = module
        .invoke(std::slice::from_ref(&input), 7, module.api_mode())
        .unwrap();
    assert_eq!(result.outputs[0], input);
    assert_eq!(
        result.outputs[1],
        Value::StringArray(
            runmat_value::StringArray::new(vec!["alpha".into(), "changed".into()], vec![1, 2],)
                .unwrap()
        )
    );
    assert_eq!(
        result.outputs[2],
        Value::StringArray(
            runmat_value::StringArray::new(vec!["βeta".into(), "雪".into()], vec![1, 2]).unwrap()
        )
    );
    assert_eq!(
        result.outputs[3],
        Value::CharArray(runmat_value::CharArray::new_row("chars"))
    );
    assert_eq!(
        result.outputs[4],
        Value::StringArray(
            runmat_value::StringArray::new(vec!["<missing>".into(), "value".into()], vec![1, 2])
                .unwrap()
        )
    );
    assert_eq!(result.outputs[5], Value::String("<missing>".into()));
    let Value::Cell(string_cell) = &result.outputs[6] else {
        panic!("C++ string cell output must remain a cell");
    };
    assert_eq!(string_cell.get(0, 0).unwrap(), Value::String("text".into()));
}

#[test]
fn modern_cpp_sparse_factory_adopts_ordered_buffers_and_classifies_reordering() {
    let directory = tempfile::tempdir().unwrap();
    let source = directory.path().join("modern_sparse.cpp");
    fs::write(
        &source,
        r#"
#include "mex.hpp"
#include "mexAdapter.hpp"
#include <cstdint>

class MexFunction : public matlab::mex::Function {
public:
    void operator()(matlab::mex::ArgumentList outputs,
                    matlab::mex::ArgumentList inputs) override {
        (void)inputs;
        matlab::data::ArrayFactory factory;
        auto data = factory.createBuffer<double>(3);
        auto rows = factory.createBuffer<std::size_t>(3);
        auto columns = factory.createBuffer<std::size_t>(3);
        double *dataAddress = data.get();
        std::size_t *rowAddress = rows.get();
        data.get()[0] = 2.0; data.get()[1] = 4.0; data.get()[2] = 6.0;
        rows.get()[0] = 0; rows.get()[1] = 2; rows.get()[2] = 1;
        columns.get()[0] = 0; columns.get()[1] = 0; columns.get()[2] = 2;
        auto sparse = factory.createSparseArray<double>(
            {3, 3}, 3, std::move(data), std::move(rows), std::move(columns));
        auto position = sparse.begin();
        if (sparse.getNumberOfNonZeroElements() != 3 ||
            sparse.getIndex(position) != matlab::data::SparseIndex(0, 0) ||
            static_cast<double>(*position) != 2.0) {
            throw matlab::Exception("ordered sparse data was not retained");
        }
        outputs[0] = sparse;
        outputs[1] = factory.createScalar<std::uint64_t>(
            reinterpret_cast<std::uintptr_t>(dataAddress));
        outputs[2] = factory.createScalar<std::uint64_t>(
            reinterpret_cast<std::uintptr_t>(rowAddress));

        auto unorderedData = factory.createBuffer<double>(3);
        auto unorderedRows = factory.createBuffer<std::size_t>(3);
        auto unorderedColumns = factory.createBuffer<std::size_t>(3);
        unorderedData.get()[0] = 6.0; unorderedData.get()[1] = 2.0;
        unorderedData.get()[2] = 4.0;
        unorderedRows.get()[0] = 1; unorderedRows.get()[1] = 0;
        unorderedRows.get()[2] = 2;
        unorderedColumns.get()[0] = 2; unorderedColumns.get()[1] = 0;
        unorderedColumns.get()[2] = 0;
        outputs[3] = factory.createSparseArray<double>(
            {3, 3}, 3, std::move(unorderedData), std::move(unorderedRows),
            std::move(unorderedColumns));
    }
};
"#,
    )
    .unwrap();

    let artifact = MexBuild::new(&source, directory.path()).compile().unwrap();
    let module = MexModule::load(&artifact.module).unwrap();
    let copies_before =
        runmat_value::host_copy_metrics(runmat_value::HostCopyReason::SparseLayoutConversion);
    let result = module.invoke(&[], 4, module.api_mode()).unwrap();
    let copies_after =
        runmat_value::host_copy_metrics(runmat_value::HostCopyReason::SparseLayoutConversion);
    assert!(copies_after.operations >= copies_before.operations + 3);
    assert!(
        copies_after.bytes
            >= copies_before.bytes
                + 3 * 3 * std::mem::size_of::<usize>() as u64
                + 3 * std::mem::size_of::<f64>() as u64
    );

    let Value::SparseTensor(ordered) = &result.outputs[0] else {
        panic!("ordered C++ sparse output must remain sparse");
    };
    assert_eq!(&ordered.col_ptrs[..], &[0, 2, 2, 3]);
    assert_eq!(&ordered.row_indices[..], &[0, 2, 1]);
    assert_eq!(
        ordered.to_dense().unwrap().materialize_f64(),
        vec![2.0, 0.0, 4.0, 0.0, 0.0, 0.0, 0.0, 6.0, 0.0]
    );
    let data_address = unsafe {
        ordered
            .numeric_host_buffer()
            .unwrap()
            .foreign_data_pointer()
    } as usize as u64;
    let row_address = ordered.row_indices.as_ptr() as usize as u64;
    let Value::Int(recorded_data) = &result.outputs[1] else {
        panic!("data address must remain uint64");
    };
    let Value::Int(recorded_rows) = &result.outputs[2] else {
        panic!("row address must remain uint64");
    };
    assert_eq!(recorded_data.try_to_u64(), Some(data_address));
    assert_eq!(recorded_rows.try_to_u64(), Some(row_address));

    let Value::SparseTensor(unordered) = &result.outputs[3] else {
        panic!("reordered C++ sparse output must remain sparse");
    };
    assert_eq!(unordered, ordered);
}

#[test]
fn modern_cpp_object_properties_and_enumerations_use_data_api_types() {
    let directory = tempfile::tempdir().unwrap();
    let source = directory.path().join("modern_objects.cpp");
    fs::write(
        &source,
        r#"
#include "mex.hpp"
#include "mexAdapter.hpp"

class MexFunction : public matlab::mex::Function {
public:
    void operator()(matlab::mex::ArgumentList outputs,
                    matlab::mex::ArgumentList inputs) override {
        matlab::data::Array object = inputs[0];
        if (object.getType() != matlab::data::ArrayType::VALUE_OBJECT) {
            throw matlab::Exception("input is not a Data API value object");
        }
        auto engine = getEngine();
        matlab::data::TypedArray<double> value =
            engine->getProperty(object, u"Value");
        if (static_cast<double>(value[0]) != 4.0) {
            throw matlab::Exception("object property value was not retained");
        }
        matlab::data::ArrayFactory factory;
        auto replacement = factory.createScalar<double>(9.0);
        engine->setProperty(object, u"Value", replacement);
        outputs[0] = object;

        auto state = factory.createEnumArray({1, 2}, "FixtureState",
                                             {u8"Réady", "Done"});
        if (state.getType() != matlab::data::ArrayType::ENUM ||
            state.getClassName() != "FixtureState" ||
            static_cast<std::string>(state[1]) != "Done" ||
            static_cast<std::string>(*state.cbegin()) != u8"Réady") {
            throw matlab::Exception("enumeration metadata was not retained");
        }
        outputs[1] = state;
    }
};
"#,
    )
    .unwrap();

    let mut object = runmat_value::ObjectInstance::new("FixtureObject");
    object.properties.insert("Value".into(), Value::Num(4.0));
    object
        .properties
        .insert("Untouched".into(), Value::String("shared".into()));
    let artifact = MexBuild::new(&source, directory.path()).compile().unwrap();
    let module = MexModule::load(&artifact.module).unwrap();
    let result = module
        .invoke(&[Value::Object(object.clone())], 2, module.api_mode())
        .unwrap();
    assert_eq!(object.properties.get("Value"), Some(&Value::Num(4.0)));
    let Value::Object(changed) = &result.outputs[0] else {
        panic!("C++ object output must remain an object");
    };
    assert_eq!(changed.properties.get("Value"), Some(&Value::Num(9.0)));
    assert_eq!(
        changed.properties.get("Untouched"),
        Some(&Value::String("shared".into()))
    );
    let Value::ObjectArray(states) = &result.outputs[1] else {
        panic!("C++ enum output must remain a homogeneous object array");
    };
    assert!(states
        .class_name()
        .is(runmat_types::StaticClassIdentity::new("FixtureState")));
    let members = states
        .data()
        .iter()
        .map(|value| {
            let Value::Object(value) = value else {
                panic!("enum element must remain an object");
            };
            value.properties.get("__enum_member__").cloned()
        })
        .collect::<Vec<_>>();
    assert_eq!(
        members,
        vec![
            Some(Value::String("Réady".into())),
            Some(Value::String("Done".into()))
        ]
    );
}

#[test]
fn modern_cpp_exceptions_unwind_before_the_host_reports_the_error() {
    let directory = tempfile::tempdir().unwrap();
    let source = directory.path().join("modern_exception.cpp");
    fs::write(
        &source,
        r#"
#include "mex.hpp"
#include "mexAdapter.hpp"

static bool destroyed = false;
struct Guard { ~Guard() { destroyed = true; } };

class MexFunction : public matlab::mex::Function {
public:
    void operator()(matlab::mex::ArgumentList outputs,
                    matlab::mex::ArgumentList inputs) override {
        matlab::data::ArrayFactory factory;
        if (inputs.empty()) {
            Guard guard;
            throw matlab::Exception("fixture failure");
        }
        outputs[0] = factory.createScalar<bool>(destroyed);
    }
};
"#,
    )
    .unwrap();

    let artifact = MexBuild::new(&source, directory.path()).compile().unwrap();
    let module = MexModule::load(&artifact.module).unwrap();
    let error = module.invoke(&[], 0, module.api_mode()).unwrap_err();
    assert!(error.to_string().contains("RunMat:mex:cppException"));
    assert!(error.to_string().contains("fixture failure"));
    let result = module
        .invoke(&[Value::Bool(true)], 1, module.api_mode())
        .unwrap();
    assert_eq!(result.outputs, vec![Value::Bool(true)]);
}

#[test]
fn modern_cpp_gateway_object_retains_module_state_between_calls() {
    let directory = tempfile::tempdir().unwrap();
    let source = directory.path().join("modern_state.cpp");
    fs::write(
        &source,
        r#"
#include "mex.hpp"
#include "mexAdapter.hpp"

class MexFunction : public matlab::mex::Function {
public:
    void operator()(matlab::mex::ArgumentList outputs,
                    matlab::mex::ArgumentList inputs) override {
        (void)inputs;
        matlab::data::ArrayFactory factory;
        outputs[0] = factory.createScalar<double>(++calls_);
    }

private:
    double calls_ = 0.0;
};
"#,
    )
    .unwrap();

    let artifact = MexBuild::new(&source, directory.path()).compile().unwrap();
    let module = MexModule::load(&artifact.module).unwrap();
    let first = module.invoke(&[], 1, module.api_mode()).unwrap();
    let second = module.invoke(&[], 1, module.api_mode()).unwrap();
    assert_eq!(first.outputs, vec![Value::Num(1.0)]);
    assert_eq!(second.outputs, vec![Value::Num(2.0)]);
}

#[test]
fn modern_cpp_builds_keep_c_support_translation_units_in_c_mode() {
    let directory = tempfile::tempdir().unwrap();
    let source = directory.path().join("mixed_gateway.cpp");
    let support = directory.path().join("support.c");
    fs::write(
        &source,
        r#"
#include "mex.hpp"
#include "mexAdapter.hpp"
extern "C" double c_support(double value);

class MexFunction : public matlab::mex::Function {
public:
    void operator()(matlab::mex::ArgumentList outputs,
                    matlab::mex::ArgumentList inputs) override {
        (void)inputs;
        matlab::data::ArrayFactory factory;
        outputs[0] = factory.createScalar<double>(c_support(5.0));
    }
};
"#,
    )
    .unwrap();
    fs::write(
        &support,
        r#"
double c_support(double value) {
    return _Generic(value, double: value + 2.0, default: 0.0);
}
"#,
    )
    .unwrap();

    let build = MexBuild::new(&source, directory.path()).source(&support);
    let plan = build.plan().unwrap();
    assert_eq!(plan.steps.len(), 4);
    assert!(plan.steps[0].arguments.contains(&"-std=c++17".to_string()));
    assert!(plan.steps[1].arguments.contains(&"-std=c11".to_string()));
    let artifact = build.compile().unwrap();
    let module = MexModule::load(&artifact.module).unwrap();
    let result = module.invoke(&[], 1, module.api_mode()).unwrap();
    assert_eq!(result.outputs, vec![Value::Num(7.0)]);
}

#[test]
fn dense_numeric_inputs_and_outputs_keep_their_host_allocation() {
    let directory = tempfile::tempdir().unwrap();
    let source = directory.path().join("allocation_identity.c");
    fs::write(
        &source,
        r#"
#include "mex.h"
#include <stdint.h>

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
    if (nlhs != 3 || nrhs != 1) mexErrMsgTxt("expected one input and three outputs");
    plhs[0] = mxCreateNumericMatrix(1, 2, mxINT32_CLASS, mxREAL);
    mxInt32 *values = mxGetInt32s(plhs[0]);
    values[0] = 17;
    values[1] = -9;
    plhs[1] = mxCreateNumericMatrix(1, 1, mxUINT64_CLASS, mxREAL);
    mxGetUint64s(plhs[1])[0] = (mxUint64)(uintptr_t)values;
    plhs[2] = mxCreateNumericMatrix(1, 1, mxUINT64_CLASS, mxREAL);
    mxGetUint64s(plhs[2])[0] = (mxUint64)(uintptr_t)mxGetData(prhs[0]);
}

"#,
    )
    .unwrap();

    let input = runmat_value::Tensor::new_integer(
        runmat_value::IntegerStorage::I32(vec![3, 4]),
        vec![1, 2],
    )
    .unwrap();
    // SAFETY: only pointer identity is observed, and the source remains alive
    // through the synchronous module invocation.
    let input_address = unsafe { input.host_buffer().foreign_data_pointer() } as usize as u64;
    let artifact = MexBuild::new(&source, directory.path()).compile().unwrap();
    let module = MexModule::load(&artifact.module).unwrap();
    let result = module
        .invoke(&[Value::Tensor(input)], 3, module.api_mode())
        .unwrap();

    let Value::Tensor(output) = &result.outputs[0] else {
        panic!("two-element output must remain a tensor");
    };
    // SAFETY: only pointer identity is observed while the output owns its
    // allocation.
    let output_address = unsafe { output.host_buffer().foreign_data_pointer() } as usize as u64;
    let Value::Int(created_address) = &result.outputs[1] else {
        panic!("created pointer address must remain uint64");
    };
    let Value::Int(observed_input_address) = &result.outputs[2] else {
        panic!("input pointer address must remain uint64");
    };
    assert_eq!(created_address.try_to_u64(), Some(output_address));
    assert_eq!(observed_input_address.try_to_u64(), Some(input_address));
}

#[test]
fn mxsetdata_adopts_a_proven_compatible_host_allocation() {
    let directory = tempfile::tempdir().unwrap();
    let source = directory.path().join("adopted_allocation.c");
    fs::write(
        &source,
        r#"
#include "mex.h"
#include <stdint.h>

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
    (void)nrhs; (void)prhs;
    if (nlhs != 2) mexErrMsgTxt("expected two outputs");
    mxArray *output = mxCreateDoubleMatrix(1, 2, mxREAL);
    double *owned = (double *)mxMalloc(2 * sizeof(double));
    if (owned == NULL) mexErrMsgTxt("allocation failed");
    owned[0] = 17.0;
    owned[1] = -9.0;
    plhs[1] = mxCreateNumericMatrix(1, 1, mxUINT64_CLASS, mxREAL);
    mxGetUint64s(plhs[1])[0] = (mxUint64)(uintptr_t)owned;
    mxSetDoubles(output, owned);
    if (mxGetDoubles(output) != owned) mexErrMsgTxt("allocation was not adopted");
    plhs[0] = output;
}
"#,
    )
    .unwrap();

    let artifact = MexBuild::new(&source, directory.path()).compile().unwrap();
    let module = MexModule::load(&artifact.module).unwrap();
    let result = module.invoke(&[], 2, module.api_mode()).unwrap();
    let Value::Tensor(output) = &result.outputs[0] else {
        panic!("adopted output must remain a tensor");
    };
    let Value::Int(observed_address) = &result.outputs[1] else {
        panic!("allocation address must remain uint64");
    };
    // SAFETY: only pointer identity is observed while `output` owns the block.
    let output_address = unsafe { output.host_buffer().foreign_data_pointer() } as usize as u64;
    assert_eq!(observed_address.try_to_u64(), Some(output_address));
    assert_eq!(output.materialize_f64(), vec![17.0, -9.0]);
}

#[test]
fn mxsetdata_normalizes_registered_logical_storage_during_conversion() {
    let directory = tempfile::tempdir().unwrap();
    let source = directory.path().join("logical_allocation.c");
    fs::write(
        &source,
        r#"
#include "mex.h"

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
    (void)nrhs; (void)prhs;
    if (nlhs != 1) mexErrMsgTxt("expected one output");
    mxArray *output = mxCreateLogicalMatrix(1, 2);
    mxLogical *owned = (mxLogical *)mxMalloc(2 * sizeof(mxLogical));
    if (owned == NULL) mexErrMsgTxt("allocation failed");
    owned[0] = 0;
    owned[1] = 7;
    mxSetData(output, owned);
    plhs[0] = output;
}
"#,
    )
    .unwrap();

    let artifact = MexBuild::new(&source, directory.path()).compile().unwrap();
    let module = MexModule::load(&artifact.module).unwrap();
    let result = module.invoke(&[], 1, module.api_mode()).unwrap();
    let Value::LogicalArray(output) = &result.outputs[0] else {
        panic!("logical output must remain an array");
    };
    assert_eq!(output.data, vec![0, 1]);
}

#[test]
fn interleaved_complex_inputs_and_outputs_keep_their_host_allocation() {
    let directory = tempfile::tempdir().unwrap();
    let source = directory.path().join("complex_allocation_identity.c");
    fs::write(
        &source,
        r#"
#include "mex.h"
#include <stdint.h>

static mxArray *address_of(const void *pointer) {
    mxArray *value = mxCreateNumericMatrix(1, 1, mxUINT64_CLASS, mxREAL);
    mxGetUint64s(value)[0] = (mxUint64)(uintptr_t)pointer;
    return value;
}

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
    if (nlhs != 3 || nrhs != 1 || !mxIsComplex(prhs[0])) {
        mexErrMsgTxt("expected one complex input and three outputs");
    }
    plhs[0] = mxCreateDoubleMatrix(1, 2, mxCOMPLEX);
    mxComplexDouble *values = mxGetComplexDoubles(plhs[0]);
    values[0].real = 17.0;
    values[0].imag = -9.0;
    values[1].real = 4.0;
    values[1].imag = 3.0;
    plhs[1] = address_of(values);
    plhs[2] = address_of(mxGetComplexDoubles(prhs[0]));
}
"#,
    )
    .unwrap();

    let input =
        runmat_value::ComplexTensor::new(vec![(3.0, 4.0), (5.0, 12.0)], vec![1, 2]).unwrap();
    let runmat_value::ComplexStorage::F64(input_values) = input.complex_storage() else {
        unreachable!("constructor creates double complex storage")
    };
    // SAFETY: only pointer identity is observed during the synchronous call.
    let input_address = unsafe { input_values.foreign_data_pointer() } as usize as u64;
    let artifact = MexBuild::new(&source, directory.path())
        .api(MexApi::R2018a)
        .compile()
        .unwrap();
    let module = MexModule::load(&artifact.module).unwrap();
    let result = module
        .invoke(&[Value::ComplexTensor(input)], 3, module.api_mode())
        .unwrap();

    let Value::ComplexTensor(output) = &result.outputs[0] else {
        panic!("two-element output must remain a complex tensor");
    };
    let runmat_value::ComplexStorage::F64(output_values) = output.complex_storage() else {
        panic!("double complex output must retain its class");
    };
    // SAFETY: only pointer identity is observed while the output owns its allocation.
    let output_address = unsafe { output_values.foreign_data_pointer() } as usize as u64;
    let Value::Int(created_address) = &result.outputs[1] else {
        panic!("created pointer address must remain uint64");
    };
    let Value::Int(observed_input_address) = &result.outputs[2] else {
        panic!("input pointer address must remain uint64");
    };
    assert_eq!(created_address.try_to_u64(), Some(output_address));
    assert_eq!(observed_input_address.try_to_u64(), Some(input_address));
}

#[test]
fn interleaved_complex_sparse_c_api_retains_input_and_output_allocations() {
    let directory = tempfile::tempdir().unwrap();
    let source = directory.path().join("complex_sparse_interleaved.c");
    fs::write(
        &source,
        r#"
#include "mex.h"
#include <stdint.h>

static mxArray *address_of(const void *pointer) {
    mxArray *value = mxCreateNumericMatrix(1, 1, mxUINT64_CLASS, mxREAL);
    mxGetUint64s(value)[0] = (mxUint64)(uintptr_t)pointer;
    return value;
}

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
    if (nlhs != 3 || nrhs != 1 || !mxIsSparse(prhs[0]) ||
        !mxIsComplex(prhs[0])) {
        mexErrMsgTxt("expected one complex sparse input and three outputs");
    }
    const mxComplexDouble *input = mxGetComplexDoubles(prhs[0]);
    if (input == NULL || input[0].real != 3.0 || input[0].imag != -4.0 ||
        input[1].real != 5.0 || input[1].imag != 6.0) {
        mexErrMsgTxt("complex sparse input was not interleaved correctly");
    }

    plhs[0] = mxCreateSparse(3, 2, 2, mxCOMPLEX);
    mxComplexDouble *values = mxGetComplexDoubles(plhs[0]);
    mwIndex *rows = mxGetIr(plhs[0]);
    mwIndex *columns = mxGetJc(plhs[0]);
    values[0].real = 7.0; values[0].imag = -8.0;
    values[1].real = -9.0; values[1].imag = 10.0;
    rows[0] = 2; rows[1] = 1;
    columns[0] = 0; columns[1] = 1; columns[2] = 2;
    plhs[1] = address_of(values);
    plhs[2] = address_of(input);
}
"#,
    )
    .unwrap();

    let input = runmat_value::SparseTensor::new_complex(
        3,
        2,
        vec![0, 1, 2],
        vec![0, 1],
        vec![(3.0, -4.0), (5.0, 6.0)],
    )
    .unwrap();
    let input_address =
        unsafe { input.complex_host_buffer().unwrap().foreign_data_pointer() } as usize as u64;
    let artifact = MexBuild::new(&source, directory.path())
        .api(MexApi::R2018a)
        .compile()
        .unwrap();
    let module = MexModule::load(&artifact.module).unwrap();
    let result = module
        .invoke(&[Value::SparseTensor(input)], 3, module.api_mode())
        .unwrap();
    let Value::SparseTensor(output) = &result.outputs[0] else {
        panic!("complex sparse output expected");
    };
    assert!(output.is_complex());
    assert_eq!(&output.col_ptrs[..], &[0, 1, 2]);
    assert_eq!(&output.row_indices[..], &[2, 1]);
    assert_eq!(
        output.materialize_complex_f64().unwrap(),
        vec![(7.0, -8.0), (-9.0, 10.0)]
    );
    let output_address =
        unsafe { output.complex_host_buffer().unwrap().foreign_data_pointer() } as usize as u64;
    let addresses = result.outputs[1..]
        .iter()
        .map(|value| match value {
            Value::Int(value) => value.try_to_u64().unwrap(),
            _ => panic!("pointer address must remain uint64"),
        })
        .collect::<Vec<_>>();
    assert_eq!(addresses, vec![output_address, input_address]);
}

#[test]
fn separate_complex_sparse_c_api_preserves_components_and_mutation() {
    let directory = tempfile::tempdir().unwrap();
    let source = directory.path().join("complex_sparse_separate.c");
    fs::write(
        &source,
        r#"
#include "mex.h"

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
    if (nlhs != 1 || nrhs != 1 || !mxIsSparse(prhs[0]) ||
        !mxIsComplex(prhs[0])) {
        mexErrMsgTxt("expected one complex sparse input and one output");
    }
    const double *inputReal = mxGetPr(prhs[0]);
    const double *inputImaginary = mxGetPi(prhs[0]);
    if (inputReal == NULL || inputImaginary == NULL ||
        inputReal[0] != 3.0 || inputImaginary[0] != -4.0 ||
        inputReal[1] != 5.0 || inputImaginary[1] != 6.0) {
        mexErrMsgTxt("complex sparse input was not split correctly");
    }

    plhs[0] = mxCreateSparse(3, 2, 2, mxCOMPLEX);
    double *real = mxGetPr(plhs[0]);
    double *imaginary = mxGetPi(plhs[0]);
    mwIndex *rows = mxGetIr(plhs[0]);
    mwIndex *columns = mxGetJc(plhs[0]);
    real[0] = 11.0; imaginary[0] = -12.0;
    real[1] = -13.0; imaginary[1] = 14.0;
    rows[0] = 1; rows[1] = 2;
    columns[0] = 0; columns[1] = 1; columns[2] = 2;
}
"#,
    )
    .unwrap();

    let input = Value::SparseTensor(
        runmat_value::SparseTensor::new_complex(
            3,
            2,
            vec![0, 1, 2],
            vec![0, 1],
            vec![(3.0, -4.0), (5.0, 6.0)],
        )
        .unwrap(),
    );
    let artifact = MexBuild::new(&source, directory.path())
        .api(MexApi::R2017b)
        .compile()
        .unwrap();
    let module = MexModule::load(&artifact.module).unwrap();
    assert_eq!(module.api_mode(), MxApiMode::SeparateComplex);
    let result = module.invoke(&[input], 1, module.api_mode()).unwrap();
    let Value::SparseTensor(output) = &result.outputs[0] else {
        panic!("complex sparse output expected");
    };
    assert_eq!(&output.col_ptrs[..], &[0, 1, 2]);
    assert_eq!(&output.row_indices[..], &[1, 2]);
    assert_eq!(
        output.materialize_complex_f64().unwrap(),
        vec![(11.0, -12.0), (-13.0, 14.0)]
    );
}

#[test]
fn interleaved_complex_sparse_duplicate_detaches_and_nzmax_growth_is_owned() {
    let directory = tempfile::tempdir().unwrap();
    let source = directory.path().join("complex_sparse_copy_on_write.c");
    fs::write(
        &source,
        r#"
#include "mex.h"

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
    if (nlhs != 2 || nrhs != 1 || !mxIsSparse(prhs[0]) ||
        !mxIsComplex(prhs[0])) {
        mexErrMsgTxt("expected one complex sparse input and two outputs");
    }
    mxArray *copy = mxDuplicateArray(prhs[0]);
    mxComplexDouble *original = mxGetComplexDoubles(prhs[0]);
    mxComplexDouble *values = mxGetComplexDoubles(copy);
    if (values == original) {
        mexErrMsgTxt("writable duplicate did not detach");
    }
    values[0].real = 11.0;
    values[0].imag = -12.0;
    if (original[0].real != 3.0 || original[0].imag != -4.0) {
        mexErrMsgTxt("duplicate mutation changed the input");
    }

    mxSetNzmax(copy, 3);
    values = mxGetComplexDoubles(copy);
    mwIndex *rows = mxGetIr(copy);
    mwIndex *columns = mxGetJc(copy);
    values[2].real = 7.0;
    values[2].imag = 8.0;
    rows[2] = 2;
    columns[0] = 0;
    columns[1] = 1;
    columns[2] = 3;
    plhs[0] = copy;
    plhs[1] = mxDuplicateArray(prhs[0]);
}
"#,
    )
    .unwrap();

    let input = Value::SparseTensor(
        runmat_value::SparseTensor::new_complex(
            3,
            2,
            vec![0, 1, 2],
            vec![0, 1],
            vec![(3.0, -4.0), (5.0, 6.0)],
        )
        .unwrap(),
    );
    let artifact = MexBuild::new(&source, directory.path())
        .api(MexApi::R2018a)
        .compile()
        .unwrap();
    let module = MexModule::load(&artifact.module).unwrap();
    let result = module.invoke(&[input], 2, module.api_mode()).unwrap();
    let Value::SparseTensor(updated) = &result.outputs[0] else {
        panic!("updated complex sparse output expected");
    };
    assert_eq!(&updated.col_ptrs[..], &[0, 1, 3]);
    assert_eq!(&updated.row_indices[..], &[0, 1, 2]);
    assert_eq!(
        updated.materialize_complex_f64().unwrap(),
        vec![(11.0, -12.0), (5.0, 6.0), (7.0, 8.0)]
    );
    let Value::SparseTensor(original) = &result.outputs[1] else {
        panic!("original complex sparse output expected");
    };
    assert_eq!(&original.col_ptrs[..], &[0, 1, 2]);
    assert_eq!(&original.row_indices[..], &[0, 1]);
    assert_eq!(
        original.materialize_complex_f64().unwrap(),
        vec![(3.0, -4.0), (5.0, 6.0)]
    );
}

#[test]
fn modern_cpp_complex_sparse_factory_adopts_ordered_storage() {
    let directory = tempfile::tempdir().unwrap();
    let source = directory.path().join("modern_complex_sparse.cpp");
    fs::write(
        &source,
        r#"
#include "mex.hpp"
#include "mexAdapter.hpp"
#include <complex>
#include <cstdint>

class MexFunction : public matlab::mex::Function {
public:
    void operator()(matlab::mex::ArgumentList outputs,
                    matlab::mex::ArgumentList inputs) override {
        (void)inputs;
        matlab::data::ArrayFactory factory;
        auto data = factory.createBuffer<std::complex<double>>(3);
        auto rows = factory.createBuffer<std::size_t>(3);
        auto columns = factory.createBuffer<std::size_t>(3);
        std::complex<double> *allocation = data.get();
        data.get()[0] = {2.0, -3.0};
        data.get()[1] = {4.0, 5.0};
        data.get()[2] = {-6.0, 7.0};
        rows.get()[0] = 0; rows.get()[1] = 2; rows.get()[2] = 1;
        columns.get()[0] = 0; columns.get()[1] = 0; columns.get()[2] = 2;
        auto sparse = factory.createSparseArray<std::complex<double>>(
            {3, 3}, 3, std::move(data), std::move(rows), std::move(columns));
        auto position = sparse.begin();
        if (sparse.getType() != matlab::data::ArrayType::SPARSE_COMPLEX_DOUBLE ||
            sparse.getNumberOfNonZeroElements() != 3 ||
            sparse.getIndex(position) != matlab::data::SparseIndex(0, 0) ||
            static_cast<std::complex<double>>(*position) !=
                std::complex<double>(2.0, -3.0)) {
            throw matlab::Exception("complex sparse storage was not retained");
        }
        outputs[0] = sparse;
        outputs[1] = factory.createScalar<std::uint64_t>(
            reinterpret_cast<std::uintptr_t>(allocation));
    }
};
"#,
    )
    .unwrap();

    let artifact = MexBuild::new(&source, directory.path()).compile().unwrap();
    let module = MexModule::load(&artifact.module).unwrap();
    let result = module.invoke(&[], 2, module.api_mode()).unwrap();
    let Value::SparseTensor(output) = &result.outputs[0] else {
        panic!("complex sparse output expected");
    };
    assert_eq!(&output.col_ptrs[..], &[0, 2, 2, 3]);
    assert_eq!(&output.row_indices[..], &[0, 2, 1]);
    assert_eq!(
        output.materialize_complex_f64().unwrap(),
        vec![(2.0, -3.0), (4.0, 5.0), (-6.0, 7.0)]
    );
    let output_address =
        unsafe { output.complex_host_buffer().unwrap().foreign_data_pointer() } as usize as u64;
    let Value::Int(recorded_address) = &result.outputs[1] else {
        panic!("recorded buffer address must remain uint64");
    };
    assert_eq!(recorded_address.try_to_u64(), Some(output_address));
}

#[test]
fn sparse_numeric_inputs_and_outputs_keep_compatible_host_allocations() {
    let directory = tempfile::tempdir().unwrap();
    let source = directory.path().join("sparse_allocation_identity.c");
    fs::write(
        &source,
        r#"
#include "mex.h"
#include <stdint.h>

static mxArray *address_of(const void *pointer) {
    mxArray *value = mxCreateNumericMatrix(1, 1, mxUINT64_CLASS, mxREAL);
    mxGetUint64s(value)[0] = (mxUint64)(uintptr_t)pointer;
    return value;
}

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
    if (nlhs != 7 || nrhs != 1 || !mxIsSparse(prhs[0])) {
        mexErrMsgTxt("expected one sparse input and seven outputs");
    }
    plhs[0] = mxCreateSparse(2, 2, 2, mxREAL);
    double *values = (double *)mxMalloc(2 * sizeof(double));
    mwIndex *rows = (mwIndex *)mxMalloc(2 * sizeof(mwIndex));
    mwIndex *columns = (mwIndex *)mxMalloc(3 * sizeof(mwIndex));
    if (values == NULL || rows == NULL || columns == NULL) {
        mexErrMsgTxt("allocation failed");
    }
    values[0] = 5.0;
    values[1] = -2.0;
    rows[0] = 0;
    rows[1] = 1;
    columns[0] = 0;
    columns[1] = 1;
    columns[2] = 2;
    mxSetDoubles(plhs[0], values);
    mxSetIr(plhs[0], rows);
    mxSetJc(plhs[0], columns);
    plhs[1] = address_of(mxGetData(prhs[0]));
    plhs[2] = address_of(mxGetIr(prhs[0]));
    plhs[3] = address_of(mxGetJc(prhs[0]));
    plhs[4] = address_of(values);
    plhs[5] = address_of(rows);
    plhs[6] = address_of(columns);
}
"#,
    )
    .unwrap();

    let input =
        runmat_value::SparseTensor::new(2, 2, vec![0, 1, 2], vec![1, 0], vec![3.0, 4.0]).unwrap();
    // SAFETY: the test observes pointer identity only while the owning sparse
    // value remains live through the synchronous invocation.
    let input_data =
        unsafe { input.numeric_host_buffer().unwrap().foreign_data_pointer() } as usize as u64;
    let input_rows = unsafe { input.row_indices.foreign_data_pointer() } as usize as u64;
    let input_columns = unsafe { input.col_ptrs.foreign_data_pointer() } as usize as u64;

    let artifact = MexBuild::new(&source, directory.path()).compile().unwrap();
    let module = MexModule::load(&artifact.module).unwrap();
    let result = module
        .invoke(&[Value::SparseTensor(input)], 7, module.api_mode())
        .unwrap();
    let Value::SparseTensor(output) = &result.outputs[0] else {
        panic!("first output must remain sparse");
    };
    let output_data =
        unsafe { output.numeric_host_buffer().unwrap().foreign_data_pointer() } as usize as u64;
    let output_rows = unsafe { output.row_indices.foreign_data_pointer() } as usize as u64;
    let output_columns = unsafe { output.col_ptrs.foreign_data_pointer() } as usize as u64;
    let addresses = result.outputs[1..]
        .iter()
        .map(|value| match value {
            Value::Int(value) => value.try_to_u64().unwrap(),
            _ => panic!("pointer address must remain uint64"),
        })
        .collect::<Vec<_>>();
    assert_eq!(
        addresses,
        vec![
            input_data,
            input_rows,
            input_columns,
            output_data,
            output_rows,
            output_columns
        ]
    );
}

#[test]
fn documented_matrix_api_helpers_preserve_types_objects_and_ownership() {
    let directory = tempfile::tempdir().unwrap();
    let source = directory.path().join("matrix_api.c");
    fs::write(
        &source,
        r#"
#include "mex.h"
#include <string.h>

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
    (void)nrhs; (void)prhs;
    if (nlhs != 4) mexErrMsgTxt("expected four outputs");

    mwSize dims[3] = {2, 3, 4};
    mxArray *indices = mxCreateUninitNumericArray(3, dims, mxUINT64_CLASS, mxREAL);
    mxUint64 *owned = (mxUint64 *)mxCalloc(24, sizeof(mxUint64));
    owned[23] = UINT64_MAX;
    mxSetUint64s(indices, owned);
    mwIndex subs[3] = {1, 2, 3};
    if (!mxIsScalar(mxCreateDoubleScalar(1.0)) ||
        mxCalcSingleSubscript(indices, 3, subs) != 23 ||
        mxGetUint64s(indices)[23] != UINT64_MAX) {
        mexErrMsgTxt("typed storage or subscript helper failed");
    }
    plhs[0] = indices;

    const char *rows[2] = {"wide", "µ"};
    plhs[1] = mxCreateCharMatrixFromStrings(2, rows);
    char *utf8 = mxArrayToUTF8String(mxCreateString("RunMat ✓"));
    if (utf8 == NULL || strcmp(utf8, "RunMat ✓") != 0) {
        mexErrMsgTxt("UTF-8 conversion failed");
    }
    mxFree(utf8);

    const char *properties[1] = {"value"};
    mxArray *object = mxCreateStructMatrix(1, 1, 1, properties);
    mxSetField(object, 0, "value", mxCreateDoubleScalar(7.0));
    if (mxSetClassName(object, "FixtureObject") != 0 ||
        !mxIsClass(object, "FixtureObject") ||
        mxGetScalar(mxGetProperty(object, 0, "value")) != 7.0) {
        mexErrMsgTxt("object property conversion failed");
    }
    mxSetProperty(object, 0, "value", mxCreateDoubleScalar(9.0));
    plhs[2] = object;

    mxArray *complex_value = mxCreateNumericMatrix(1, 1, mxDOUBLE_CLASS, mxREAL);
    mxGetDoubles(complex_value)[0] = 2.0;
    if (mxMakeArrayComplex(complex_value) == 0) mexErrMsgTxt("make complex failed");
#if defined(MX_HAS_INTERLEAVED_COMPLEX)
    mxGetComplexDoubles(complex_value)[0].imag = 5.0;
#else
    mxGetPi(complex_value)[0] = 5.0;
#endif
    plhs[3] = complex_value;
}
"#,
    )
    .unwrap();

    for api in [MexApi::R2017b, MexApi::R2018a] {
        let artifact = MexBuild::new(&source, directory.path())
            .api(api)
            .output_name(format!("matrix_api_{api:?}"))
            .compile()
            .unwrap();
        let module = MexModule::load(&artifact.module).unwrap();
        let result = module.invoke(&[], 4, module.api_mode()).unwrap();
        let Value::Tensor(indices) = &result.outputs[0] else {
            panic!("typed N-D result must remain a tensor");
        };
        assert_eq!(indices.shape, vec![2, 3, 4]);
        assert_eq!(
            indices.numeric_value_at(23),
            Some(runmat_value::NumericScalar::U64(u64::MAX))
        );
        let Value::CharArray(rows) = &result.outputs[1] else {
            panic!("character matrix must remain a character array");
        };
        assert_eq!(rows.shape(), &[2, 4]);
        let Value::Object(object) = &result.outputs[2] else {
            panic!("classed struct must become a RunMat object");
        };
        assert_eq!(object.class_name.display_name(), "FixtureObject");
        assert_eq!(object.properties.get("value"), Some(&Value::Num(9.0)));
        assert_eq!(result.outputs[3], Value::Complex(2.0, 5.0));
    }
}

#[test]
fn api_pins_control_dimension_width_and_complex_layout() {
    let directory = tempfile::tempdir().unwrap();
    let source = directory.path().join("api_pin.c");
    fs::write(
        &source,
        r#"
#include "mex.h"

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
    (void)nrhs; (void)prhs;
    if (nlhs != 1) mexErrMsgTxt("expected one output");
    mwSize dims[2] = {1, 2};
    plhs[0] = mxCreateDoubleMatrix(dims[0], dims[1], mxREAL);
    double *values = mxGetDoubles(plhs[0]);
    values[0] = (double)sizeof(mwSize);
    values[1] = (double)mxGetN(plhs[0]);
}
"#,
    )
    .unwrap();

    for (api, expected_width, expected_mode) in [
        (
            MexApi::R2017b,
            std::mem::size_of::<usize>(),
            MxApiMode::SeparateComplex,
        ),
        (
            MexApi::R2018a,
            std::mem::size_of::<usize>(),
            MxApiMode::InterleavedComplex,
        ),
        (
            MexApi::LargeArrayDims,
            std::mem::size_of::<usize>(),
            MxApiMode::SeparateComplex,
        ),
        (
            MexApi::CompatibleArrayDims,
            std::mem::size_of::<i32>(),
            MxApiMode::SeparateComplex,
        ),
    ] {
        let output_name = format!("api_{api:?}");
        let artifact = MexBuild::new(&source, directory.path())
            .api(api)
            .output_name(output_name)
            .compile()
            .unwrap();
        let module = MexModule::load(&artifact.module).unwrap();
        assert_eq!(module.api_mode(), expected_mode);
        let result = module.invoke(&[], 1, module.api_mode()).unwrap();
        let Value::Tensor(tensor) = &result.outputs[0] else {
            panic!("API fixture must return a matrix");
        };
        assert_eq!(tensor.materialize_f64(), vec![expected_width as f64, 2.0]);
    }
}

#[test]
fn c_gateway_compatibility_definitions_and_scalar_spellings_are_available() {
    let directory = tempfile::tempdir().unwrap();
    let source = directory.path().join("compatibility_surface.c");
    fs::write(
        &source,
        r#"
#include "mex.h"

#ifndef MATLAB_MEX_FILE
#error "MEX builds must define MATLAB_MEX_FILE"
#endif

#if MEX_INFORMATION_VERSION != 1
#error "unexpected MEX information version"
#endif

#if TARGET_API_VERSION != 700
#error "the default API pin must select the separate-complex compatibility surface"
#endif

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
    (void)nrhs; (void)prhs;
    if (nlhs != 1) mexErrMsgTxt("expected one output");

    bool enabled = true;
    int8_T *value = (int8_T *)malloc(sizeof(int8_T));
    if (value == NULL) mexErrMsgTxt("allocation failed");
    *value = 7;
    boolean_T flag = enabled ? 1 : 0;
    real32_T single_value = 0.5f;
    real64_T result = (real64_T)(*value + flag + abs(-3)) + single_value;
    printf("compatibility surface\n");
    plhs[0] = mxCreateDoubleScalar((real_T)result);
    free(value);
}
"#,
    )
    .unwrap();

    let artifact = MexBuild::new(&source, directory.path()).compile().unwrap();
    let module = MexModule::load(&artifact.module).unwrap();
    let result = module.invoke(&[], 1, module.api_mode()).unwrap();

    assert_eq!(result.outputs, vec![Value::Num(11.5)]);
    assert_eq!(result.console, "compatibility surface\n");
}

#[test]
fn independently_compiled_gateway_loads_and_preserves_typed_input() {
    let directory = tempfile::tempdir().unwrap();
    let source = directory.path().join("add_one.c");
    fs::write(
        &source,
        r#"
#include "mex.h"

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
    if (nrhs != 1 || nlhs != 1 || !mxIsUint64(prhs[0])) {
        mexErrMsgIdAndTxt("RunMat:Fixture:Arguments", "expected one uint64 input and output");
    }
    mexPrintf("typed fixture\n");
    plhs[0] = mxCreateNumericMatrix(1, 1, mxUINT64_CLASS, mxREAL);
    mxGetUint64s(plhs[0])[0] = mxGetUint64s(prhs[0])[0] + 1;
}
"#,
    )
    .unwrap();
    let artifact = MexBuild::new(&source, directory.path()).compile().unwrap();
    assert!(artifact.manifest.is_file());
    assert_eq!(
        runmat_mex::MexArtifactManifest::from_canonical_bytes(
            &fs::read(&artifact.manifest).unwrap()
        )
        .unwrap(),
        artifact.artifact
    );
    artifact
        .artifact
        .validate_module(&fs::read(&artifact.module).unwrap())
        .unwrap();
    let rebuilt = MexBuild::new(&source, directory.path()).compile().unwrap();
    assert_eq!(rebuilt.manifest, artifact.manifest);
    assert_eq!(
        runmat_mex::MexArtifactManifest::from_canonical_bytes(
            &fs::read(&rebuilt.manifest).unwrap()
        )
        .unwrap(),
        rebuilt.artifact
    );
    let module = MexModule::load(&artifact.module).unwrap();
    let result = module
        .invoke(
            &[Value::Int(runmat_value::IntValue::U64(
                9_007_199_254_740_993,
            ))],
            1,
            module.api_mode(),
        )
        .unwrap();
    assert_eq!(
        result.outputs,
        vec![Value::Int(runmat_value::IntValue::U64(
            9_007_199_254_740_994
        ))]
    );
    assert_eq!(result.console, "typed fixture\n");
}

#[test]
fn loader_rejects_a_module_that_no_longer_matches_its_artifact_identity() {
    use std::io::Write as _;

    let directory = tempfile::tempdir().unwrap();
    let source = directory.path().join("tamper.c");
    fs::write(
        &source,
        r#"
#include "mex.h"
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
    (void)nlhs; (void)plhs; (void)nrhs; (void)prhs;
}

"#,
    )
    .unwrap();
    let artifact = MexBuild::new(&source, directory.path()).compile().unwrap();
    fs::OpenOptions::new()
        .append(true)
        .open(&artifact.module)
        .unwrap()
        .write_all(b"tampered")
        .unwrap();

    let error = match MexModule::load(&artifact.module) {
        Ok(_) => panic!("loader admitted a module that did not match its manifest"),
        Err(error) => error,
    };
    assert!(matches!(error, MexLoadError::ArtifactManifest { .. }));
}

#[test]
fn compatible_isolated_tier_does_not_weaken_exact_manifest_admission() {
    let directory = tempfile::tempdir().unwrap();
    let source = directory.path().join("compatible.c");
    fs::write(
        &source,
        r#"
#include "mex.h"
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
    (void)nrhs; (void)prhs;
    if (nlhs == 1) plhs[0] = mxCreateDoubleScalar(42.0);
}
"#,
    )
    .unwrap();
    let artifact = MexBuild::new(&source, directory.path()).compile().unwrap();
    fs::remove_file(&artifact.manifest).unwrap();

    assert!(matches!(
        MexModule::load(&artifact.module),
        Err(MexLoadError::ArtifactManifestRead { .. })
    ));
    let module = MexModule::load_compatible_isolated(&artifact.module).unwrap();
    let result = module.invoke(&[], 1, module.api_mode()).unwrap();
    assert_eq!(result.outputs, vec![Value::Num(42.0)]);
}

#[test]
fn mex_error_stops_the_gateway_without_unwinding_through_rust() {
    let directory = tempfile::tempdir().unwrap();
    let source = directory.path().join("failure.c");
    fs::write(
        &source,
        r#"
#include "mex.h"
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
    (void)nlhs; (void)plhs; (void)nrhs; (void)prhs;
    mexErrMsgIdAndTxt("Fixture:Expected", "failure %d", 42);
}
"#,
    )
    .unwrap();
    let artifact = MexBuild::new(&source, directory.path()).compile().unwrap();
    let module = MexModule::load(&artifact.module).unwrap();
    let error = module.invoke(&[], 0, module.api_mode()).unwrap_err();
    assert!(matches!(
        error,
        MexLoadError::Invocation { identifier, message }
            if identifier.contains("Fixture:Expected") && message == "failure 42"
    ));
}

#[test]
fn cell_and_struct_ownership_crosses_the_gateway_once() {
    let directory = tempfile::tempdir().unwrap();
    let source = directory.path().join("containers.c");
    fs::write(
        &source,
        r#"
#include "mex.h"
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
    (void)nrhs; (void)prhs;
    if (nlhs != 2) mexErrMsgTxt("expected two outputs");
    plhs[0] = mxCreateCellMatrix(1, 1);
    mxSetCell(plhs[0], 0, mxCreateDoubleScalar(7.0));
    const char *fields[] = {"value"};
    plhs[1] = mxCreateStructMatrix(1, 1, 1, fields);
    mxSetField(plhs[1], 0, "value", mxCreateLogicalScalar(1));
    if (mxGetScalar(mxGetCell(plhs[0], 0)) != 7.0 ||
        !mxIsLogicalScalarTrue(mxGetFieldByNumber(plhs[1], 0, 0))) {
        mexErrMsgTxt("nested array lookup failed");
    }
}
"#,
    )
    .unwrap();
    let artifact = MexBuild::new(&source, directory.path()).compile().unwrap();
    let module = MexModule::load(&artifact.module).unwrap();
    let result = module.invoke(&[], 2, module.api_mode()).unwrap();
    let Value::Cell(cell) = &result.outputs[0] else {
        panic!("first output must be a cell array");
    };
    assert_eq!(cell.data, vec![Value::Num(7.0)]);
    let Value::Struct(structure) = &result.outputs[1] else {
        panic!("second output must be a struct");
    };
    assert_eq!(structure.fields.get("value"), Some(&Value::Bool(true)));
}

#[test]
fn sparse_capacity_uses_column_pointers_as_the_actual_nonzero_boundary() {
    let directory = tempfile::tempdir().unwrap();
    let source = directory.path().join("sparse_fixture.c");
    fs::write(
        &source,
        r#"
#include "mex.h"
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
    (void)nrhs; (void)prhs;
    if (nlhs != 1) mexErrMsgTxt("expected one output");
    plhs[0] = mxCreateSparse(3, 2, 4, mxREAL);
    mwIndex *ir = mxGetIr(plhs[0]);
    mwIndex *jc = mxGetJc(plhs[0]);
    double *values = mxGetDoubles(plhs[0]);
    ir[0] = 1; values[0] = 4.0;
    ir[1] = 0; values[1] = 8.0;
    jc[0] = 0; jc[1] = 1; jc[2] = 2;
}
"#,
    )
    .unwrap();
    for api in [MexApi::R2017b, MexApi::CompatibleArrayDims] {
        let artifact = MexBuild::new(&source, directory.path())
            .api(api)
            .output_name(format!("sparse_{api:?}"))
            .compile()
            .unwrap();
        let module = MexModule::load(&artifact.module).unwrap();
        let result = module.invoke(&[], 1, module.api_mode()).unwrap();
        let Value::SparseTensor(value) = &result.outputs[0] else {
            panic!("output must be sparse");
        };
        assert_eq!(value.col_ptrs, vec![0, 1, 2]);
        assert_eq!(value.row_indices, vec![1, 0]);
        assert_eq!(value.nnz(), 2);
    }
}

#[test]
fn persistent_arrays_and_locks_are_owned_by_the_loaded_module() {
    let directory = tempfile::tempdir().unwrap();
    let source = directory.path().join("persistent_fixture.c");
    fs::write(
        &source,
        r#"
#include "mex.h"
static mxArray *counter = NULL;
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
    (void)prhs;
    if (counter == NULL) {
        counter = mxCreateDoubleScalar(0.0);
        mexMakeArrayPersistent(counter);
        mexLock();
    }
    mxGetDoubles(counter)[0] += 1.0;
    if (nlhs == 1) plhs[0] = mxDuplicateArray(counter);
    if (nrhs == 1) mexUnlock();
}
"#,
    )
    .unwrap();
    let artifact = MexBuild::new(&source, directory.path()).compile().unwrap();
    let module = MexModule::load(&artifact.module).unwrap();
    let first = module.invoke(&[], 1, module.api_mode()).unwrap();
    let second = module.invoke(&[], 1, module.api_mode()).unwrap();
    assert_eq!(first.outputs, vec![Value::Num(1.0)]);
    assert_eq!(second.outputs, vec![Value::Num(2.0)]);
    assert!(module.is_locked());
    assert!(!module.clear().unwrap());
    module
        .invoke(&[Value::Num(0.0)], 0, module.api_mode())
        .unwrap();
    assert!(module.clear().unwrap());
}

#[derive(Default)]
struct FixtureHost {
    workspace: Mutex<BTreeMap<String, Value>>,
}

impl MexHostServices for FixtureHost {
    fn eval(&self, command: &str) -> Result<(), MexDiagnostic> {
        if command == "accepted" {
            Ok(())
        } else {
            Err(MexDiagnostic {
                identifier: Some("Fixture:Eval".into()),
                message: command.into(),
            })
        }
    }

    fn call(
        &self,
        function: &str,
        arguments: Vec<Value>,
        requested_outputs: usize,
    ) -> Result<Vec<Value>, MexDiagnostic> {
        if function == "plus_one" && requested_outputs == 1 {
            let Value::Num(value) = arguments[0] else {
                panic!("fixture expected numeric scalar");
            };
            return Ok(vec![Value::Num(value + 1.0)]);
        }
        Err(MexDiagnostic {
            identifier: Some("Fixture:Call".into()),
            message: function.into(),
        })
    }

    fn get_variable(&self, _workspace: &str, name: &str) -> Result<Option<Value>, MexDiagnostic> {
        Ok(self.workspace.lock().unwrap().get(name).cloned())
    }

    fn put_variable(
        &self,
        _workspace: &str,
        name: &str,
        value: Value,
    ) -> Result<(), MexDiagnostic> {
        self.workspace.lock().unwrap().insert(name.into(), value);
        Ok(())
    }
}

#[test]
fn callbacks_and_workspace_access_route_through_the_explicit_host_port() {
    let directory = tempfile::tempdir().unwrap();
    let source = directory.path().join("callback_fixture.c");
    fs::write(
        &source,
        r#"
#include "mex.h"
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
    if (nlhs != 2 || nrhs != 1) mexErrMsgTxt("expected two outputs and one input");
    mxArray *arguments[] = {(mxArray *)prhs[0]};
    mxArray *called[] = {NULL};
    mexCallMATLAB(1, called, 1, arguments, "plus_one");
    mexPutVariable("base", "saved", called[0]);
    plhs[0] = mexGetVariable("base", "saved");
    mexEvalString("accepted");
    plhs[1] = mexEvalStringWithTrap("trapped");
    if (plhs[1] == NULL || !mxIsStruct(plhs[1])) {
        mexErrMsgTxt("trap did not return an exception value");
    }
}
"#,
    )
    .unwrap();
    let artifact = MexBuild::new(&source, directory.path()).compile().unwrap();
    let module = MexModule::load(&artifact.module).unwrap();
    let result = module
        .invoke_with_services(
            &[Value::Num(8.0)],
            2,
            module.api_mode(),
            Rc::new(FixtureHost::default()),
        )
        .unwrap();
    assert_eq!(result.outputs[0], Value::Num(9.0));
    let Value::Struct(exception) = &result.outputs[1] else {
        panic!("trap output must be an exception structure");
    };
    assert_eq!(
        exception.fields.get("identifier"),
        Some(&Value::CharArray(runmat_value::CharArray::new_row(
            "Fixture:Eval"
        )))
    );
}

#[test]
fn local_cpp_engine_futures_complete_through_the_explicit_host_port() {
    let directory = tempfile::tempdir().unwrap();
    let source = directory.path().join("local_async_fixture.cpp");
    fs::write(
        &source,
        r#"
#include "mex.hpp"
#include "mexAdapter.hpp"

class MexFunction : public matlab::mex::Function {
public:
    void operator()(matlab::mex::ArgumentList outputs,
                    matlab::mex::ArgumentList inputs) override {
        (void)inputs;
        outputs[0] = matlab::data::ArrayFactory().createScalar<double>(
            getEngine()->fevalAsync<double>(u"plus_one", 8.0).get());
    }
};
"#,
    )
    .unwrap();
    let artifact = MexBuild::new(&source, directory.path()).compile().unwrap();
    let module = MexModule::load(&artifact.module).unwrap();
    let result = module
        .invoke_with_services(&[], 1, module.api_mode(), Rc::new(FixtureHost::default()))
        .unwrap();
    assert_eq!(result.outputs, vec![Value::Num(9.0)]);
}

#[test]
fn completed_gateway_does_not_retain_its_invocation_host() {
    let directory = tempfile::tempdir().unwrap();
    let source = directory.path().join("host_lifetime.c");
    fs::write(
        &source,
        r#"
#include "mex.h"
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
    (void)nlhs; (void)plhs; (void)nrhs; (void)prhs;
}
"#,
    )
    .unwrap();
    let artifact = MexBuild::new(&source, directory.path()).compile().unwrap();
    let module = MexModule::load(&artifact.module).unwrap();
    let host = Rc::new(FixtureHost::default());
    let weak = Rc::downgrade(&host);
    module
        .invoke_with_services(&[], 0, module.api_mode(), host.clone())
        .unwrap();
    drop(host);

    assert!(weak.upgrade().is_none());
}

#[test]
fn forced_shutdown_runs_at_exit_with_the_originating_host_services_alive() {
    let directory = tempfile::tempdir().unwrap();
    let source = directory.path().join("at_exit_fixture.c");
    fs::write(
        &source,
        r#"
#include "mex.h"
static void record_exit(void) {
    mxArray *value = mxCreateDoubleScalar(99.0);
    if (mexPutVariable("base", "exit_seen", value) != 0) {
        mexErrMsgTxt("at-exit workspace callback failed");
    }
    mxDestroyArray(value);
}

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
    (void)nlhs; (void)plhs; (void)nrhs; (void)prhs;
    if (mexAtExit(record_exit) != 0) mexErrMsgTxt("could not register at-exit callback");
    mexLock();
}
"#,
    )
    .unwrap();
    let artifact = MexBuild::new(&source, directory.path()).compile().unwrap();
    let host = Rc::new(FixtureHost::default());
    let module = MexModule::load(&artifact.module).unwrap();
    module
        .invoke_with_services(&[], 0, module.api_mode(), host.clone())
        .unwrap();
    assert!(module.is_locked());
    module.shutdown_with_services(host.clone()).unwrap();

    assert_eq!(
        host.workspace.lock().unwrap().get("exit_seen"),
        Some(&Value::Num(99.0))
    );
}

#[test]
fn at_exit_error_is_reported_without_crossing_the_c_abi() {
    let directory = tempfile::tempdir().unwrap();
    let source = directory.path().join("at_exit_error.c");
    fs::write(
        &source,
        r#"
#include "mex.h"
static void fail_exit(void) {
    mexErrMsgIdAndTxt("Fixture:AtExit", "expected teardown failure");
}
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
    (void)nlhs; (void)plhs; (void)nrhs; (void)prhs;
    mexAtExit(fail_exit);
}
"#,
    )
    .unwrap();
    let artifact = MexBuild::new(&source, directory.path()).compile().unwrap();
    let module = MexModule::load(&artifact.module).unwrap();
    let host = Rc::new(FixtureHost::default());
    module
        .invoke_with_services(&[], 0, module.api_mode(), host.clone())
        .unwrap();
    let error = module
        .shutdown_with_services(host)
        .expect_err("at-exit failure must reach the host");

    assert!(matches!(
        error,
        MexLoadError::Invocation { identifier, message }
            if identifier.contains("Fixture:AtExit") && message == "expected teardown failure"
    ));
}

#[derive(Default)]
struct ReentrantHost {
    module: std::cell::RefCell<Option<std::rc::Weak<MexModule>>>,
}

impl MexHostServices for ReentrantHost {
    fn eval(&self, _command: &str) -> Result<(), MexDiagnostic> {
        unreachable!("reentrancy fixture does not evaluate source")
    }

    fn call(
        &self,
        function: &str,
        _arguments: Vec<Value>,
        _requested_outputs: usize,
    ) -> Result<Vec<Value>, MexDiagnostic> {
        assert_eq!(function, "recursive_entry");
        let module = self
            .module
            .borrow()
            .as_ref()
            .unwrap()
            .upgrade()
            .expect("fixture module is alive");
        let error = module
            .invoke(&[], 0, module.api_mode())
            .expect_err("same-module recursive entry must fail");
        Err(MexDiagnostic {
            identifier: Some("RunMat:MEX:ReentrantInvocation".into()),
            message: error.to_string(),
        })
    }

    fn get_variable(&self, _workspace: &str, _name: &str) -> Result<Option<Value>, MexDiagnostic> {
        unreachable!("reentrancy fixture does not read workspace state")
    }

    fn put_variable(
        &self,
        _workspace: &str,
        _name: &str,
        _value: Value,
    ) -> Result<(), MexDiagnostic> {
        unreachable!("reentrancy fixture does not write workspace state")
    }
}

#[test]
fn same_module_callback_reentry_fails_without_deadlocking() {
    let directory = tempfile::tempdir().unwrap();
    let source = directory.path().join("reentrant_fixture.c");
    fs::write(
        &source,
        r#"
#include "mex.h"
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
    (void)nlhs; (void)plhs; (void)nrhs; (void)prhs;
    if (mexCallMATLAB(0, NULL, 0, NULL, "recursive_entry") != 0) {
        mexErrMsgTxt("recursive callback rejected");
    }
}
"#,
    )
    .unwrap();
    let artifact = MexBuild::new(&source, directory.path()).compile().unwrap();

    let module = Rc::new(MexModule::load(&artifact.module).unwrap());
    let host = Rc::new(ReentrantHost::default());
    *host.module.borrow_mut() = Some(Rc::downgrade(&module));
    let error = module
        .invoke_with_services(&[], 0, module.api_mode(), host)
        .expect_err("recursive gateway must fail");

    assert!(matches!(
        error,
        MexLoadError::Invocation { identifier, message }
            if identifier.contains("RunMat:MEX:ReentrantInvocation")
                && message.contains("recursive invocation of the same C MEX module")
    ));
}

#[test]
fn one_in_process_owner_is_admitted_per_module_image() {
    let directory = tempfile::tempdir().unwrap();
    let source = directory.path().join("ownership_fixture.c");
    fs::write(
        &source,
        r#"
#include "mex.h"
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
    (void)nlhs; (void)plhs; (void)nrhs; (void)prhs;
}
"#,
    )
    .unwrap();
    let artifact = MexBuild::new(&source, directory.path()).compile().unwrap();

    let owner = MexModule::load(&artifact.module).unwrap();
    let contender = match MexModule::load(&artifact.module) {
        Ok(_) => panic!("a second in-process owner must use an isolated host"),
        Err(error) => error,
    };
    assert!(contender.requires_isolated_host());

    drop(owner);
    let successor = MexModule::load(&artifact.module)
        .expect("ownership must be released when the loaded module is dropped");
    successor.invoke(&[], 0, successor.api_mode()).unwrap();
}

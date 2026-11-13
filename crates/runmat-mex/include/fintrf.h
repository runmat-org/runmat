#ifndef RUNMAT_FINTRF_H
#define RUNMAT_FINTRF_H

#if defined(RUNMAT_MX_COMPATIBLE_ARRAY_DIMS)
#define mwSize integer*4
#define mwIndex integer*4
#define RUNMAT_MW_ZERO 0_4
#else
#define mwSize integer*8
#define mwIndex integer*8
#define RUNMAT_MW_ZERO 0_8
#endif
#if defined(__LP64__) || defined(_WIN64)
#define mwPointer integer*8
#else
#define mwPointer integer*4
#endif
#define mwSignedIndex mwPointer

#define mxUNKNOWN_CLASS 0
#define mxCELL_CLASS 1
#define mxSTRUCT_CLASS 2
#define mxLOGICAL_CLASS 3
#define mxCHAR_CLASS 4
#define mxVOID_CLASS 5
#define mxDOUBLE_CLASS 6
#define mxSINGLE_CLASS 7
#define mxINT8_CLASS 8
#define mxUINT8_CLASS 9
#define mxINT16_CLASS 10
#define mxUINT16_CLASS 11
#define mxINT32_CLASS 12
#define mxUINT32_CLASS 13
#define mxINT64_CLASS 14
#define mxUINT64_CLASS 15
#define mxFUNCTION_CLASS 16
#define mxOPAQUE_CLASS 17
#define mxOBJECT_CLASS 18

#define mxREAL 0
#define mxCOMPLEX 1
#if defined(RUNMAT_MX_INTERLEAVED_COMPLEX)
#define MX_HAS_INTERLEAVED_COMPLEX 1
#endif

#if defined(RUNMAT_MEX_FORTRAN_GATEWAY)
#define mexFunction RUNMAT_MEX_FORTRAN_GATEWAY
#endif

/* Fixed-form Fortran has no implicit interface at these call sites. Promote
   documented dimension/count expressions to the selected mwSize width before
   the compiler lowers the call. The lowercase target avoids macro recursion
   and retains the conventional external symbol spelling. */
#define mxCreateDoubleMatrix(m,n,c) mxcreatedoublematrix((m)+RUNMAT_MW_ZERO,(n)+RUNMAT_MW_ZERO,c)
#define mxCreateNumericMatrix(m,n,t,c) mxcreatenumericmatrix((m)+RUNMAT_MW_ZERO,(n)+RUNMAT_MW_ZERO,t,c)
#define mxCreateNumericArray(d,s,t,c) mxcreatenumericarray((d)+RUNMAT_MW_ZERO,s,t,c)
#define mxCreateLogicalMatrix(m,n) mxcreatelogicalmatrix((m)+RUNMAT_MW_ZERO,(n)+RUNMAT_MW_ZERO)
#define mxCreateLogicalArray(d,s) mxcreatelogicalarray((d)+RUNMAT_MW_ZERO,s)
#define mxCreateCellMatrix(m,n) mxcreatecellmatrix((m)+RUNMAT_MW_ZERO,(n)+RUNMAT_MW_ZERO)
#define mxCreateCellArray(d,s) mxcreatecellarray((d)+RUNMAT_MW_ZERO,s)
#define mxCreateCharArray(d,s) mxcreatechararray((d)+RUNMAT_MW_ZERO,s)
#define mxCreateStructArray(d,s,f,n) mxcreatestructarray((d)+RUNMAT_MW_ZERO,s,f,n)
#define mxCreateStructMatrix(m,n,f,s) mxcreatestructmatrix((m)+RUNMAT_MW_ZERO,(n)+RUNMAT_MW_ZERO,f,s)
#define mxCreateSparse(m,n,z,c) mxcreatesparse((m)+RUNMAT_MW_ZERO,(n)+RUNMAT_MW_ZERO,(z)+RUNMAT_MW_ZERO,c)
#define mxCreateSparseLogicalMatrix(m,n,z) mxcreatesparselogicalmatrix((m)+RUNMAT_MW_ZERO,(n)+RUNMAT_MW_ZERO,(z)+RUNMAT_MW_ZERO)
#define mxGetCell(a,i) mxgetcell(a,(i)+RUNMAT_MW_ZERO)
#define mxSetCell(a,i,v) mxsetcell(a,(i)+RUNMAT_MW_ZERO,v)
#define mxGetField(a,i,n) mxgetfield(a,(i)+RUNMAT_MW_ZERO,n)
#define mxSetField(a,i,n,v) mxsetfield(a,(i)+RUNMAT_MW_ZERO,n,v)
#define mxGetFieldByNumber(a,i,f) mxgetfieldbynumber(a,(i)+RUNMAT_MW_ZERO,f)
#define mxSetFieldByNumber(a,i,f,v) mxsetfieldbynumber(a,(i)+RUNMAT_MW_ZERO,f,v)
#define mxGetProperty(a,i,n) mxgetproperty(a,(i)+RUNMAT_MW_ZERO,n)
#define mxSetProperty(a,i,n,v) mxsetproperty(a,(i)+RUNMAT_MW_ZERO,n,v)
#define mxSetM(a,m) mxsetm(a,(m)+RUNMAT_MW_ZERO)
#define mxSetN(a,n) mxsetn(a,(n)+RUNMAT_MW_ZERO)
#define mxSetNzmax(a,n) mxsetnzmax(a,(n)+RUNMAT_MW_ZERO)
#define mxMalloc(n) mxmalloc((n)+RUNMAT_MW_ZERO)
#define mxCalloc(n,s) mxcalloc((n)+RUNMAT_MW_ZERO,(s)+RUNMAT_MW_ZERO)
#define mxRealloc(p,n) mxrealloc(p,(n)+RUNMAT_MW_ZERO)
#define mxGetString(a,s,n) mxgetstring(a,s,(n)+RUNMAT_MW_ZERO)
#define mxCopyPtrToReal8(p,v,n) mxcopyptrtoreal8(p,v,(n)+RUNMAT_MW_ZERO)
#define mxCopyReal8ToPtr(v,p,n) mxcopyreal8toptr(v,p,(n)+RUNMAT_MW_ZERO)
#define mxCopyPtrToReal4(p,v,n) mxcopyptrtoreal4(p,v,(n)+RUNMAT_MW_ZERO)
#define mxCopyReal4ToPtr(v,p,n) mxcopyreal4toptr(v,p,(n)+RUNMAT_MW_ZERO)
#define mxCopyPtrToInteger1(p,v,n) mxcopyptrtointeger1(p,v,(n)+RUNMAT_MW_ZERO)
#define mxCopyInteger1ToPtr(v,p,n) mxcopyinteger1toptr(v,p,(n)+RUNMAT_MW_ZERO)
#define mxCopyPtrToInteger2(p,v,n) mxcopyptrtointeger2(p,v,(n)+RUNMAT_MW_ZERO)
#define mxCopyInteger2ToPtr(v,p,n) mxcopyinteger2toptr(v,p,(n)+RUNMAT_MW_ZERO)
#define mxCopyPtrToInteger4(p,v,n) mxcopyptrtointeger4(p,v,(n)+RUNMAT_MW_ZERO)
#define mxCopyInteger4ToPtr(v,p,n) mxcopyinteger4toptr(v,p,(n)+RUNMAT_MW_ZERO)
#define mxCopyPtrToInteger8(p,v,n) mxcopyptrtointeger8(p,v,(n)+RUNMAT_MW_ZERO)
#define mxCopyInteger8ToPtr(v,p,n) mxcopyinteger8toptr(v,p,(n)+RUNMAT_MW_ZERO)
#if defined(RUNMAT_MX_INTERLEAVED_COMPLEX)
#define mxCopyPtrToComplex16(p,v,n) mxcopyptrtocomplex16(p,v,(n)+RUNMAT_MW_ZERO)
#define mxCopyComplex16ToPtr(v,p,n) mxcopycomplex16toptr(v,p,(n)+RUNMAT_MW_ZERO)
#define mxCopyPtrToComplex8(p,v,n) mxcopyptrtocomplex8(p,v,(n)+RUNMAT_MW_ZERO)
#define mxCopyComplex8ToPtr(v,p,n) mxcopycomplex8toptr(v,p,(n)+RUNMAT_MW_ZERO)
#else
#define mxCopyPtrToComplex16(r,i,v,n) mxcopyptrtocomplex16(r,i,v,(n)+RUNMAT_MW_ZERO)
#define mxCopyComplex16ToPtr(v,r,i,n) mxcopycomplex16toptr(v,r,i,(n)+RUNMAT_MW_ZERO)
#define mxCopyPtrToComplex8(r,i,v,n) mxcopyptrtocomplex8(r,i,v,(n)+RUNMAT_MW_ZERO)
#define mxCopyComplex8ToPtr(v,r,i,n) mxcopycomplex8toptr(v,r,i,(n)+RUNMAT_MW_ZERO)
#endif

#endif

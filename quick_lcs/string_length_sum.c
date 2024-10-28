#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

#include <Python.h>
#include <numpy/arrayobject.h>
#include <string.h>
#include <omp.h>  // Include OpenMP header

// Function to calculate length sums
static PyObject* length_sum(PyObject* self, PyObject* args) {
    PyArrayObject *arr1, *arr2, *result;

    // Parse the input tuple (two NumPy arrays and one result array)
    if (!PyArg_ParseTuple(args, "O!O!O!", &PyArray_Type, &arr1, &PyArray_Type, &arr2, &PyArray_Type, &result)) {
        return NULL;
    }

    // Check if both input arrays have the same size
    if (PyArray_SIZE(arr1) != PyArray_SIZE(arr2)) {
        PyErr_SetString(PyExc_ValueError, "Input arrays must have the same size.");
        return NULL;
    }

    // Check if the result array is a double array of the correct size
    if (PyArray_SIZE(result) != PyArray_SIZE(arr1) || PyArray_TYPE(result) != NPY_DOUBLE) {
        PyErr_SetString(PyExc_ValueError, "Result array must be a double array of the same size.");
        return NULL;
    }

    // Flag to indicate if an error occurs
    int error_flag = 0;

    // Iterate over both input arrays in parallel
    #pragma omp parallel for
    for (npy_intp i = 0; i < PyArray_SIZE(arr1); i++) {
        // Get Python string objects
        PyObject *str_obj1 = PyArray_GETITEM(arr1, PyArray_GETPTR1(arr1, i));
        PyObject *str_obj2 = PyArray_GETITEM(arr2, PyArray_GETPTR1(arr2, i));

        // Check that both are strings
        if (!PyUnicode_Check(str_obj1) || !PyUnicode_Check(str_obj2)) {
            error_flag = 1;  // Set error flag
            continue;        // Skip this iteration
        }

        // Get UTF-8 encoded strings
        const char *str1 = PyUnicode_AsUTF8(str_obj1);
        const char *str2 = PyUnicode_AsUTF8(str_obj2);

        // Check for NULL (encoding error)
        if (str1 == NULL || str2 == NULL) {
            error_flag = 1;  // Set error flag
            continue;        // Skip this iteration
        }

        // Calculate the lengths of each string
        int len1 = strlen(str1);
        int len2 = strlen(str2);
        double length_sum = (double)len1 + (double)len2;  // Explicitly casting lengths to double

        // Set the result in place
        *(double *) PyArray_GETPTR1(result, i) = length_sum;  // Correctly assign to result
    }

    // Check for errors after the parallel region
    if (error_flag) {
        PyErr_SetString(PyExc_TypeError, "Both arrays must contain only strings, and all strings must be convertible to UTF-8.");
        return NULL;
    }

    // Return None
    Py_RETURN_NONE;
}

// Method definitions
static PyMethodDef module_methods[] = {
    {"length_sum", (PyCFunction) length_sum, METH_VARARGS, "Sum the lengths of corresponding strings in two arrays"},
    {NULL, NULL, 0, NULL}
};

// Module definition
static struct PyModuleDef quicklcsmodule = {
    PyModuleDef_HEAD_INIT,
    "quick_lcs.string_length_sum",
    NULL,
    -1,
    module_methods
};

// Module initialization function
PyMODINIT_FUNC PyInit_string_length_sum(void) {
    import_array();  // Initialize NumPy C API
    return PyModule_Create(&quicklcsmodule);
}


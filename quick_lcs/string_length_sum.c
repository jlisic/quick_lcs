#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

#include <Python.h>
#include <numpy/arrayobject.h>
#include <string.h>
#include <omp.h>  // Include OpenMP header







double string_soft_compare(const char * a, const_char * b, size_t n_a, size_t n_b) {

    printf("Starting comparison\n");

    size_t i,j;
    size_t init_score - 0;
    double score = 0.0;

    // handle edge cases
    if n_a == 0 || n_b == 0 {
        return score;
    }

    // create the matrix
    size_t * x = (size_t *) calloc((n_a+1)*(n_b+1), sizeof(size_t));
 
    if (x == NULL) {
        // memory allocation failed
        return score;
    }

    // fill the matrix
    for (i = 0; i < n_a; i++) {
        for (j = 0; j < n_b; j++) {
            if (a[i] == b[j]) {
                x[ (n_b+1)*(i+1) + (j+1)] = x[ (n_b+1)*i + j] + 1;
            } else {
                if (x[ (n_b+1)*i + (j+1)] > x[ (n_b+1)*(i+1) + j]) {
                    x[ (n_b+1)*(i+1) + (j+1)] = x[ (n_b+1)*i + (j+1)];
                } else {
                    x[ (n_b+1)*(i+1) + (j+1)] = x[ (n_b+1)*(i+1) + j];
                }
            }
        }
    }

    // get the score
    init_score = x[(n_b+1)*(n_a+1) -1];
    free(x);
    x=NULL;

    // normalize the score
    if n_a > n_b {
        score = (double) init_score / (double) n_a;
    } else {
        score = (double) init_score / (double) n_b;
    }

    // return score
    return score;

}





/*
void C_string_compare( const char ** x, const char ** y, double * out, size_t n_x, size_t n_y) {

    size_t i, j;
    
    size_t n_char_x, n_char_y;
    double score, max_score;

    const char * x_string;
    const char * y_string;

    for (i = 0; i < n_x; i++) {

        x_string = x[i];
        n_char_x = strlen(x_string);

        //reset the scores
        max_score = 0.0;

        if (n_char_x == 0) {
            continue;
        }

        for (j = 0; j < n_y; j++) {
            score = 0.0;

            y_string
            n_char_y = strlen(y_string);

            if (n_char_y == 0) {
                continue;
            }

            // compute the score
            score = string_soft_compare(x_string, n_char_x, y_string, n_char_y);

            // save the max score
            if (score > max_score) {
                max_score = score;
            }

            // if it is a perfect match, break early
            if (score = 1.0) {
                break;
            }
            
        }
    }
}
*/






// Function to calculate length sums
static PyObject* length_sum(PyObject* self, PyObject* args) {
    PyArrayObject *arr1, *arr2, *result *result_index;
    size_t i;
    const char *str1, *str2;
    size_t len1, len2;

    double score, max_score;
    PyObject *str_obj1, *str_obj2;

    // Parse the input tuple (two NumPy arrays and one result array)
    if (!PyArg_ParseTuple(args, "O!O!O!O!", &PyArray_Type, &arr1, &PyArray_Type, &arr2, &PyArray_Type, &result, &PyArray_Type, &result_index)) {
        return NULL;
    }

    // Check if both input arrays have the same size
    if (PyArray_SIZE(arr1) != PyArray_SIZE(result)) {
        PyErr_SetString(PyExc_ValueError, "Input arrays must have the same size.");
        return NULL;
    }

    // Check if both input arrays have the same size
    if (PyArray_SIZE(arr1) != PyArray_SIZE(result_index)) {
        PyErr_SetString(PyExc_ValueError, "Input arrays must have the same size.");
        return NULL;
    }

    // Check types
    if (PyArray_TYPE(result) != NPY_DOUBLE) {
        PyErr_SetString(PyExc_ValueError, "Result array must be a double array of the same size.");
        return NULL;
    }

    // create strings to work with
    int error_flag = 0;

    // Iterate over both input arrays in parallel   
    for (npy_intp i = 0; i < PyArray_SIZE(arr1); i++) {


        // Get Python string objects
        str_obj1 = PyArray_GETITEM(arr1, PyArray_GETPTR1(arr1, i));

        // Check that both are strings
        if (!PyUnicode_Check(str_obj1)) {
            error_flag = 1;  // Set error flag
            continue;        // Skip this iteration
        }

        // Get UTF-8 encoded strings
        str1 = PyUnicode_AsUTF8(str_obj1);


        // Check for NULL (encoding error)
        if (str1 == NULL) {
            error_flag = 1;  // Set error flag
            continue;        // Skip this iteration
        }

        // et string length
        len1 = strlen(str1);
        max_score = 0

        
        for (npy_intp j = 0; j < PyArray_SIZE(arr2); j++) {
            str_obj2 = PyArray_GETITEM(arr2, PyArray_GETPTR1(arr2, j));

            // Check that both are strings
            if (!PyUnicode_Check(str_obj2)) {
                error_flag = 1;  // Set error flag
                continue;        // Skip this iteration
            }

            // Get UTF-8 encoded strings
            str2 = PyUnicode_AsUTF8(str_obj2);

            // Check for NULL (encoding error)
            if (str2 == NULL) {
                error_flag = 1;  // Set error flag
                continue;        // Skip this iteration
            }
            // get string length
            len2 = strlen(str2);

            // compute the score
            score = string_soft_compare(str1, str2, len1, len2);
            if (score > max_score) {
                max_score = score;
            }
        }

        // Set the result in place
        *(double *) PyArray_GETPTR1(result, i) = max_score;  
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


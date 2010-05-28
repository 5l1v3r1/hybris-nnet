/*
 * This file is part of the Hybris programming language interpreter.
 *
 * Copyleft of Simone Margaritelli aka evilsocket <evilsocket@gmail.com>
 *
 * Hybris is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * Hybris is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with Hybris.  If not, see <http://www.gnu.org/licenses/>.
*/
#include <hybris/hybris.h>
#include <math.h>

typedef struct {
    /*! Output value of each neuron . */
	double ** out;
    /*! Delta error value for each neuron . */
    double ** delta;
    /*! Vector of weights for each neuron . */
    double *** weights;
    /*! Storage for weight-change made in previous epoch . */
    double *** last_state;
    /*! Learning rate . */
    double learning_rate;
    /*! Momentum parameter . */
    double momentum;
	/*! Vector that contains each layer size . */
	size_t *layers;
    /*! Number of layers in net including input layer . */
	size_t numlayers;
    /*! Input layer index (usually is 0) . */
    size_t in_idx;
    /*! Output layer index (usually m_numlayers - 1) . */
    size_t out_idx;
}
nnet_t;

/**
 * @brief Sigmoid high-pass function .
 *
 * The sigmoid function is used as a filter for output values, and it is defined as :
 * \f$ P(t) = \frac{1}{1 + e^{-t}} \f$
 *
 * @param value The value to compute the sigmoid of .
 * @return Sigmoid result .
 */
INLINE double sigmoid(double value){
    return (double)(1 / (1 + exp(-value)));
}

static Float __nnet_float_value(0.0);

HYBRIS_DEFINE_FUNCTION(nnet_create);
HYBRIS_DEFINE_FUNCTION(nnet_get_lrate);
HYBRIS_DEFINE_FUNCTION(nnet_set_lrate);
HYBRIS_DEFINE_FUNCTION(nnet_get_momentum);
HYBRIS_DEFINE_FUNCTION(nnet_set_momentum);
HYBRIS_DEFINE_FUNCTION(nnet_get_nlayers);
HYBRIS_DEFINE_FUNCTION(nnet_get_layers_size);
HYBRIS_DEFINE_FUNCTION(nnet_propagate_input);
HYBRIS_DEFINE_FUNCTION(nnet_get_output);
HYBRIS_DEFINE_FUNCTION(nnet_get_error);
HYBRIS_DEFINE_FUNCTION(nnet_train);
HYBRIS_DEFINE_FUNCTION(nnet_save);
HYBRIS_DEFINE_FUNCTION(nnet_load);
HYBRIS_DEFINE_FUNCTION(nnet_destroy);

HYBRIS_EXPORTED_FUNCTIONS() {
    { "nnet_create",   		  nnet_create,  		H_REQ_ARGC(3),   { H_REQ_TYPES(otVector), H_REQ_TYPES(otFloat), H_REQ_TYPES(otFloat) } },
    { "nnet_get_lrate",		  nnet_get_lrate,		H_REQ_ARGC(1),   { H_REQ_TYPES(otHandle) } },
    { "nnet_set_lrate",		  nnet_set_lrate,		H_REQ_ARGC(2),   { H_REQ_TYPES(otHandle), H_REQ_TYPES(otFloat) } },
    { "nnet_get_momentum",	  nnet_get_momentum,	H_REQ_ARGC(1),   { H_REQ_TYPES(otHandle) } },
    { "nnet_set_momentum",	  nnet_set_momentum,	H_REQ_ARGC(2),   { H_REQ_TYPES(otHandle), H_REQ_TYPES(otFloat) } },
    { "nnet_get_nlayers",	  nnet_get_nlayers,		H_REQ_ARGC(1),   { H_REQ_TYPES(otHandle) } },
    { "nnet_get_layers_size", nnet_get_layers_size,	H_REQ_ARGC(1),   { H_REQ_TYPES(otHandle) } },
    { "nnet_propagate_input", nnet_propagate_input, H_REQ_ARGC(2),   { H_REQ_TYPES(otHandle), H_REQ_TYPES(otVector) } },
    { "nnet_get_output",	  nnet_get_output,		H_REQ_ARGC(1),   { H_REQ_TYPES(otHandle) } },
    { "nnet_get_error",	  	  nnet_get_error,		H_REQ_ARGC(2),   { H_REQ_TYPES(otHandle), H_REQ_TYPES(otVector) } },
    { "nnet_train",	  	  	  nnet_train,			H_REQ_ARGC(5,6), { H_REQ_TYPES(otHandle), H_REQ_TYPES(otVector),
																	   H_REQ_TYPES(otVector), H_REQ_TYPES(otInteger),
																	   H_REQ_TYPES(otFloat),  H_REQ_TYPES(otAlias) } },
    { "nnet_save",	  	  	  nnet_save,			H_REQ_ARGC(2),   { H_REQ_TYPES(otHandle), H_REQ_TYPES(otString) } },
    { "nnet_load",	  	  	  nnet_load,			H_REQ_ARGC(1),   { H_REQ_TYPES(otString) } },
    { "nnet_destroy",  		  nnet_destroy, 		H_REQ_ARGC(1),   { H_REQ_TYPES(otHandle) } },
    { "", NULL }
};

HYBRIS_DEFINE_FUNCTION(nnet_create){
	Vector *lsizes;
	double  lrate,
			momentum;

	vm_parse_argv( "Vdd", &lsizes, &lrate, &momentum );

	size_t i = 0, j = 0, k = 0;

	nnet_t *net 	   = new nnet_t;
	net->learning_rate = lrate;
	net->momentum	   = momentum;
	net->numlayers     = ob_get_size((Object *)lsizes);
	net->layers        = new size_t[net->numlayers];
	net->in_idx        = 0;
	net->out_idx       = net->numlayers - 1;

	for( i = 0; i < net->numlayers; ++i ){
		net->layers[i] = ob_ivalue( lsizes->value[i] );
	}

	net->out = new double * [net->numlayers];
	for( i = 0; i < net->numlayers; ++i ){
		net->out[i] = new double[ net->layers[i] ];
	}

	net->delta = new double * [net->numlayers];
	for( i = 1; i < net->numlayers; ++i ){
		net->delta[i] = new double[ net->layers[i] ];
	}

	net->weights = new double ** [net->numlayers];
	for( i = 1; i < net->numlayers; ++i ){
		net->weights[i] = new double * [ net->layers[i] ];
	}
	for( i = 1; i < net->numlayers; ++i ){
		for( j = 0; j < net->layers[i]; j++ ){
			net->weights[i][j] = new double[ net->layers[ i - 1 ] + 1 ];
		}
	}

	net->last_state = new double ** [net->numlayers];
	for( i = 1; i < net->numlayers; ++i ){
		net->last_state[i] = new double * [ net->layers[i] ];
	}
	for( i = 1; i < net->numlayers; ++i ){
		for( j = 0; j < net->layers[i]; j++ ){
			net->last_state[i][j] = new double[ net->layers[ i - 1 ] + 1 ];
		}
	}

	srand((unsigned)(time(NULL)));
	for( i = 1; i < net->numlayers; ++i ){
		for( j = 0; j < net->layers[i]; j++ ){
			for( k = 0; k < net->layers[ i - 1 ] + 1; k++ ){
				net->weights[i][j][k]    = (double)(rand())/(RAND_MAX/2) - 1;
				net->last_state[i][j][k] = (double)0.0;
			}
		}
	}

	return (Object *)gc_new_handle(net);
}

HYBRIS_DEFINE_FUNCTION(nnet_get_lrate){
	Handle *handle;

	vm_parse_argv( "H", &handle );

	return (Object *)gc_new_float( ((nnet_t *)handle->value)->learning_rate );
}

HYBRIS_DEFINE_FUNCTION(nnet_set_lrate){
	Handle *handle;
	double 	value;

	vm_parse_argv( "Hd", &handle, &value );

	((nnet_t *)handle->value)->learning_rate = value;

	return H_DEFAULT_RETURN;
}

HYBRIS_DEFINE_FUNCTION(nnet_get_momentum){
	Handle *handle;

	vm_parse_argv( "H", &handle );

	return (Object *)gc_new_float( ((nnet_t *)handle->value)->momentum );
}

HYBRIS_DEFINE_FUNCTION(nnet_set_momentum){
	Handle *handle;
	double 	value;

	vm_parse_argv( "Hd", &handle, &value );

	((nnet_t *)handle->value)->momentum = value;

	return H_DEFAULT_RETURN;
}

HYBRIS_DEFINE_FUNCTION(nnet_get_nlayers){
	Handle *handle;

	vm_parse_argv( "H", &handle );

	return (Object *)gc_new_integer( ((nnet_t *)handle->value)->numlayers );
}

HYBRIS_DEFINE_FUNCTION(nnet_get_layers_size){
	Handle *handle;
	Object *sizes = (Object *)gc_new_vector();

	vm_parse_argv( "H", &handle );

	nnet_t *net = (nnet_t *)handle->value;
	size_t i;

	for( i = 0; i < net->numlayers; ++i ){
		ob_cl_push_reference( sizes, (Object *)gc_new_integer( net->layers[i] ) );
	}

	return sizes;
}

INLINE Object *__nnet_propagate_input_safe( nnet_t *net, vector<Object *>& input ){
    double sum;
    size_t i = 0, j = 0, k = 0,
    	   n_input = net->layers[net->in_idx];

    if( input.size() != n_input ){
    	return vm_raise_exception( "Given an input vector of %d elements, but %d needed.", input.size(), n_input );
    }

	for( i = 0; i < n_input; ++i ){
		net->out[net->in_idx][i] = ob_fvalue( input[i] );
	}

	/* for each layer */
	for( i = 1; i < net->numlayers; ++i ){
		/* for each neuron in the layer */
		for( j = 0; j < net->layers[i]; ++j ){
			sum 		= 0.0;
			size_t last = net->layers[i - 1];
			/* for input from each neuron in preceeding layer */
			for( k = 0; k < last; ++k ){
				sum += net->out[i - 1][k] * net->weights[i][j][k];
			}
			/* sum + bias value */
			net->out[i][j] = sigmoid( sum + net->weights[i][j][last] );
		}
	}

	return H_DEFAULT_RETURN;
}

INLINE void __nnet_propagate_input( nnet_t *net, vector<Object *>& input ){
    double sum;
    size_t i = 0, j = 0, k = 0,
    	   n_input = net->layers[net->in_idx];

    for( i = 0; i < n_input; ++i ){
		net->out[net->in_idx][i] = ob_fvalue( input[i] );
	}

	/* for each layer */
	for( i = 1; i < net->numlayers; ++i ){
		/* for each neuron in the layer */
		for( j = 0; j < net->layers[i]; ++j ){
			sum 		= 0.0;
			size_t last = net->layers[i - 1];
			/* for input from each neuron in preceeding layer */
			for( k = 0; k < last; ++k ){
				sum += net->out[i - 1][k] * net->weights[i][j][k];
			}
			/* sum + bias value */
			net->out[i][j] = sigmoid( sum + net->weights[i][j][last] );
		}
	}
}

HYBRIS_DEFINE_FUNCTION(nnet_propagate_input){
	Handle *handle;
	Vector *input;

	vm_parse_argv( "HV", &handle, &input );

	nnet_t *net = (nnet_t *)handle->value;

	return __nnet_propagate_input_safe( net, input->value );
}

HYBRIS_DEFINE_FUNCTION(nnet_get_output){
	Handle *handle;

	vm_parse_argv( "H", &handle );

	nnet_t *net = (nnet_t *)handle->value;

	if( net->layers[net->out_idx] == 1 ){
		return (Object *)gc_new_float(net->out[net->out_idx][0]);
	}
	else{
		Object *output = (Object *)gc_new_vector();
		size_t i;

		for( i = 0; i < net->layers[net->out_idx]; ++i ){
			ob_cl_push_reference( output, (Object *)gc_new_float(net->out[net->out_idx][i]) );
		}

		return output;
	}
}

INLINE double __nnet_get_error( nnet_t *net, vector<Object *>& expected ){
	size_t i = 0, lsize = net->layers[net->out_idx];
	double mse = 0.0;

	for( i = 0; i < lsize; ++i ){
		mse += pow( ob_fvalue( expected[i] ) - net->out[ net->out_idx ][i], 2.0 );
	}

    return mse / 2.0;
}

INLINE Object *__nnet_get_error_safe( nnet_t *net, vector<Object *>& expected ){
	size_t i = 0, lsize = net->layers[net->out_idx];
	double mse = 0.0;

	if( expected.size() != lsize ){
		return vm_raise_exception( "Given an output vector of %d elements, but %d needed.", expected.size(), lsize );
	}

	for( i = 0; i < lsize; ++i ){
		mse += pow( ob_fvalue( expected[i] ) - net->out[ net->out_idx ][i], 2.0 );
	}

	__nnet_float_value.value = mse / 2.0;

    return (Object *)&__nnet_float_value;
}

HYBRIS_DEFINE_FUNCTION(nnet_get_error){
	Handle *handle;
	Vector *expected;

	vm_parse_argv( "HV", &handle, &expected );

	nnet_t *net = (nnet_t *)handle->value;
	Object *ret;

	if( (ret = __nnet_get_error_safe( net, expected->value )) != (Object *)&__nnet_float_value ){
		return ret;
	}

	return (Object *)gc_new_float(__nnet_float_value.value);
}

INLINE void __nnet_train_epoch( nnet_t *net, Vector *input, Vector *expected ){
    double sum;
    size_t i = 0,
           j = 0,
           k = 0,
           n_outputs = net->layers[net->out_idx];
    Object *ret;

    __nnet_propagate_input( net, input->value );

    for( i = 0; i < n_outputs; ++i ){
        net->delta[net->out_idx][i] = net->out[net->out_idx][i] *
									  (1 - net->out[net->out_idx][i]) *
									  ( ob_fvalue( expected->value[i] ) - net->out[net->out_idx][i] );
    }

    for( i = net->numlayers - 2; i > 0; --i ){
        for( j = 0; j < net->layers[i]; ++j ){
            sum         = 0.0;
            size_t last = net->layers[i + 1];
            for( k = 0; k < last; --k ){
                sum += net->delta[ i + 1 ][k] * net->weights[ i + 1 ][k][j];
            }
            net->delta[i][j] = net->out[i][j] * (1 - net->out[i][j]) * sum;
        }
    }

    for( i = 1; i < net->numlayers; ++i ){
        for( j = 0; j < net->layers[i]; ++j ){
        	size_t last = net->layers[i - 1];
            for( k = 0; k < last; ++k ){
                net->weights[i][j][k] += net->momentum * net->last_state[i][j][k];
            }
            net->weights[i][j][last] += net->momentum * net->last_state[i][j][last];
        }
    }

    for( i = 1; i < net->numlayers; ++i ){
        for( j = 0; j < net->layers[i]; ++j ){
        	size_t last = net->layers[i - 1];
            for( k = 0; k < last; ++k ){
                net->last_state[i][j][k]  = net->learning_rate * net->delta[i][j] * net->out[i - 1][k];
                net->weights[i][j][k]    += net->last_state[i][j][k];
            }
            net->last_state[i][j][last]  =  net->learning_rate * net->delta[i][j];
            net->weights[i][j][last]    += net->last_state[i][j][last];
        }
    }
}

HYBRIS_DEFINE_FUNCTION(nnet_train){
	Handle *handle;
	Vector *inputs,
		   *outputs;
	int		epochs;
	double	threshold;
	Alias  *callback = NULL;

	vm_parse_argv( "HVVidA", &handle, &inputs, &outputs, &epochs, &threshold, &callback );

	nnet_t *net = (nnet_t *)handle->value;
	size_t  idx = 0, i = 0, j = 0,
		    n_inputs  = ob_get_size( (Object *)inputs ),
			n_outputs = ob_get_size( (Object *)outputs );
	double error = threshold + 1;
	Object *ret;

	vmem_t   stack;
	Integer *h_epoch = gc_new_integer(0);
	Float   *h_error = gc_new_float(0.0);

	/*
	 * Check that I/O values are vectors of vectors.
	 */
	for( i = 0; i < n_inputs; ++i ){
		if( ob_is_vector(inputs->value[i]) == false ){
			return vm_raise_exception( "Input value at index %d is not a vector but %s.", i, ob_typename(inputs->value[i]) );
		}
		else if( ob_get_size(inputs->value[i]) < net->layers[net->in_idx] ){
			return vm_raise_exception( "Given an input vector of %d elements, but %d needed.", ob_get_size(inputs->value[i]), net->layers[net->in_idx] );
		}
	}
	for( i = 0; i < n_outputs; ++i ){
		if( ob_is_vector(outputs->value[i]) == false ){
			return vm_raise_exception( "Output value at index %d is not a vector but %s.", i, ob_typename(outputs->value[i]) );
		}
		else if( ob_get_size(outputs->value[i]) < net->layers[net->out_idx] ){
			return vm_raise_exception( "Given an output vector of %d elements, but %d needed.", ob_get_size(outputs->value[i]), net->layers[net->out_idx] );
		}
	}

	/*
	 * Loop until max epochs are reached or we have an error value
	 * below the given threshold.
	 */
	for( i = 0; i < epochs && error > threshold; ++i ){
		idx = i % n_inputs;

		__nnet_train_epoch( net, (Vector *)inputs->value[idx], (Vector *)outputs->value[idx] );

		error = __nnet_get_error( net, ((Vector *)outputs->value[idx])->value );

		if( callback ){
			vmem_t stack;

			h_epoch->value = i;
			h_error->value = error;

			stack.push( (Object *)h_epoch );
			stack.push( (Object *)h_error );

			vm_exec_threaded_call( vm, (Node *)callback->value, data, &stack );

			stack.release();
		}
	}

	return (Object *)gc_new_integer(i);
}

HYBRIS_DEFINE_FUNCTION(nnet_save){
	Handle *handle;
	string  filename;

	vm_parse_argv( "Hs", &handle, &filename );

	nnet_t *net = (nnet_t *)handle->value;

	FILE *fp = fopen( filename.c_str(), "w+b" );
	if( !fp ){
		return vm_raise_exception( "Could not create file %s.", filename.c_str() );
	}

	size_t i, j, k;

	fwrite( &net->learning_rate, sizeof(double), 1, fp );
	fwrite( &net->momentum, 	 sizeof(double), 1, fp );
	fwrite( &net->numlayers, 	 sizeof(size_t), 1, fp );
	fwrite( net->layers,		 sizeof(size_t), net->numlayers, fp );
	fwrite( &net->in_idx,		 sizeof(size_t), 1, fp );
	fwrite( &net->out_idx,		 sizeof(size_t), 1, fp );

	for( i = 0; i < net->numlayers; ++i ){
		fwrite( net->out[i], sizeof(double), net->layers[i], fp );
	}

	for( i = 1; i < net->numlayers; ++i ){
		fwrite( net->delta[i], sizeof(double), net->layers[i], fp );
	}

	for( i = 1; i < net->numlayers; ++i ){
		for( j = 0; j < net->layers[i]; j++ ){
			fwrite( net->weights[i][j], sizeof(double), net->layers[ i - 1 ] + 1, fp );
		}
	}

	for( i = 1; i < net->numlayers; ++i ){
		for( j = 0; j < net->layers[i]; j++ ){
			fwrite( net->last_state[i][j], sizeof(double), net->layers[ i - 1 ] + 1, fp );
		}
	}

	size_t written = ftell(fp);

	fclose(fp);

	return (Object *)gc_new_integer(written);
}

HYBRIS_DEFINE_FUNCTION(nnet_load){
	string filename;

	vm_parse_argv( "s", &filename );

	FILE *fp = fopen( filename.c_str(), "rb" );
	if( !fp ){
		return vm_raise_exception( "Could not open file %s.", filename.c_str() );
	}

	size_t i = 0, j = 0, k = 0;

	nnet_t *net = new nnet_t;

	fread( &net->learning_rate, sizeof(double), 1, fp );
	fread( &net->momentum, 	 	sizeof(double), 1, fp );
	fread( &net->numlayers, 	sizeof(size_t), 1, fp );

	net->layers = new size_t[net->numlayers];

	fread( net->layers,	  sizeof(size_t), net->numlayers, fp );
	fread( &net->in_idx,  sizeof(size_t), 1, fp );
	fread( &net->out_idx, sizeof(size_t), 1, fp );

	net->out = new double * [net->numlayers];
	for( i = 0; i < net->numlayers; ++i ){
		net->out[i] = new double[ net->layers[i] ];
		fread( net->out[i], sizeof(double), net->layers[i], fp );
	}

	net->delta = new double * [net->numlayers];
	for( i = 1; i < net->numlayers; ++i ){
		net->delta[i] = new double[ net->layers[i] ];
		fread( net->delta[i], sizeof(double), net->layers[i], fp );
	}

	net->weights = new double ** [net->numlayers];
	for( i = 1; i < net->numlayers; ++i ){
		net->weights[i] = new double * [ net->layers[i] ];
	}
	for( i = 1; i < net->numlayers; ++i ){
		for( j = 0; j < net->layers[i]; j++ ){
			net->weights[i][j] = new double[ net->layers[ i - 1 ] + 1 ];
			fread( net->weights[i][j], sizeof(double), net->layers[ i - 1 ] + 1, fp );
		}
	}

	net->last_state = new double ** [net->numlayers];
	for( i = 1; i < net->numlayers; ++i ){
		net->last_state[i] = new double * [ net->layers[i] ];
	}
	for( i = 1; i < net->numlayers; ++i ){
		for( j = 0; j < net->layers[i]; j++ ){
			net->last_state[i][j] = new double[ net->layers[ i - 1 ] + 1 ];
			fread( net->last_state[i][j], sizeof(double), net->layers[ i - 1 ] + 1, fp );
		}
	}

	fclose(fp);

	return (Object *)gc_new_handle(net);
}

HYBRIS_DEFINE_FUNCTION(nnet_destroy){
	Handle *handle;

	vm_parse_argv( "H", &handle );

	nnet_t *net = (nnet_t *)handle->value;

	if( net ){
		size_t i = 0, j = 0;

		if( net->out ){
			for( i = 0; i < net->numlayers; ++i ){
				delete[] net->out[i];
			}
			delete net->out;
			net->out = NULL;
		}

		if( net->delta ){
			for( i = 1; i < net->numlayers; ++i ){
				delete[] net->delta[i];
			}
			delete net->delta;
			net->delta = NULL;
		}

		if( net->weights ){
			for( i = 1; i < net->numlayers; ++i ){
				for( j = 0; j < net->layers[i]; ++j ){
					delete[] net->weights[i][j];
				}
				delete net->weights[i];
			}
			delete net->weights;
			net->weights = NULL;
		}

		if( net->last_state ){
			for( i = 1; i < net->numlayers; ++i ){
				for( j = 0; j < net->layers[i]; ++j ){
					delete[] net->last_state[i][j];
				}
				delete net->last_state[i];
			}
			delete net->last_state;
			net->last_state = NULL;
		}

	    delete net;
		handle->value = NULL;
	}

	return H_DEFAULT_RETURN;
}

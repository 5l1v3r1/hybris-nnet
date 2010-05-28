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
import  nnet;
include std.Exception;

/**
 * @class NNetwork
 * @brief Back-propagation neural network class .
 */
class NNetwork {
	/*! Inner neural net handler to be used with the C module . */
	protected nnet;
	/**
     * @brief NNetwork default class constructor.
     * @param layers Vector that contains each layer size .
     * @param learning_rate Learning rate value for the network .
     * @param momentum Momentum value for the network .
     */
	public method NNetwork( layers, learning_rate, momentum ){
		me.nnet = nnet_create( layers, learning_rate, momentum );
	}
	/**
     * @brief NNetwork by-file class constructor.
	 * Loads a previously saved NNetwork.
     * @param filename The name of the file to load.
	 * @see NNetwork::load
     */
	public method NNetwork( filename ){
		me.load(filename);
	}
	/**
     * @brief NNetwork class destructor.
     */
	public method __expire(){	
		nnet_destroy( me.nnet );
	}
    /**
     * @brief Set the pattern on the input layer .
     * 
     * The propagation of the pattern is defined as :
     * 
     * \f$ f (x) = sigmoid \left(\sum_i w_i g_i(x)\right) \f$ 
     * 
     * where :
     * 
     * \f$ g_i (x) = \sum_i output_i w_i \f$
     *
     * @param input The input pattern to propagate on the network .
     */
	public method propagate( input ){
		return nnet_propagate_input( me.nnet, input );
	}
    /**
     * @brief Return the output layer vector .
     * @return The network output layer .
     */
	public method output(){
		return nnet_get_output( me.nnet );
	}
    /**
     * @brief Compute error value comparing output layer with `input` pattern .
     *  
     * The mean squared error function is defined as :
     * 
     * \f$ error = \left(\sum_i (target_i - output_i)^2\right) / 2 \f$
     * 
     * @param input The pattern to compare with the output layer .
     * @return The mean squared error .
     */
	public method error( input ){
		return nnet_get_error( me.nnet, input );
	}	
    /**
     * @brief Train the network against N input patterns and N target patterns for `epochs` training loops OR until
     * the `threshold` error value is reached .
     * @param inputs Input training set .
     * @param targets Target training set .
     * @param epochs Maximum number of epochs to use for network training .
     * @param threshold Desired maximum error value .
     */
	public method train( inputs, outputs, epochs, threshold ){
		return nnet_train( me.nnet, inputs, outputs, epochs, threshold );
	}
    /**
     * @brief Train the network against N input patterns and N target patterns for `epochs` training loops OR until
     * the `threshold` error value is reached, calling a callback for every epoch.
	 *
     * The callback prototype is defined as :
	 *
	 *		function epoch_callback( epoch, error )
	 *
     * @param inputs Input training set .
     * @param targets Target training set .
     * @param epochs Maximum number of epochs to use for network training .
     * @param threshold Desired maximum error value .
     * @param callback An instance to the function to be called upon each epoch.
     */
	public method train( inputs, outputs, epochs, threshold, callback ){
		return nnet_train( me.nnet, inputs, outputs, epochs, threshold, callback );
	}
    /**
     * @brief Attempt to recognize the input pattern .
     * @param pattern The input pattern to recognize .
     * @return The output layer values .
     */
	public method recognize( pattern ){
		me.propagate(pattern);
		return me.output();
	}
    /**
     * @brief Save the network to a file in binary format .
     * @param filename The filename to write the network to.
     * @return The number of bytes written.
     */
	public method save( filename ){
		return nnet_save( me.nnet, filename );
	}
	/**
     * @brief Load the network from a previously created file .
     * @param filename The filename to load the network from.
     * @return The number of bytes read.
	 * @see NNetwork::save
     */
	public method load( filename ){
		me.nnet = nnet_load(filename);
	}
	/**
	 * @brief Class attribute getter descriptor, for inner use only.
	 * @param name For inner use only.
     */
	public method __attribute( name ){
		switch(name){
			case "lrate"       : return nnet_get_lrate( me.nnet );    	 break;
			case "momentum"    : return nnet_get_momentum( me.nnet ); 	 break;
			case "nlayers"     : return nnet_get_nlayers( me.nnet );  	 break;
			case "layers_size" : return nnet_get_layers_size( me.nnet ); break;

			default	:
				throw new Exception( name + " invalid attribute name." );
		}
	}
	/**
	 * @brief Class attribute setter descriptor, for inner use only.
	 * @param name For inner use only.
	 * @param value For inner use only.
     */
	public method __attribute( name, value ){
		switch(name){
			case "lrate"       : return nnet_set_lrate( me.nnet, value );    break;
			case "momentum"    : return nnet_set_momentum( me.nnet, value ); break;

			default	:
				throw new Exception( name + " invalid attribute name." );
		}
	}
	/**
	 * @brief Class string representation descriptor.
	 * @return A string representation of the network.
     */
	public method __to_string(){
		str    = "";
		lsizes = me.layers_size;
		max	   = lsizes.max();
		last   = lsizes.size() - 1;
		foreach( i of 0..last ){
			sym  = ( i == 0 ? "I" : ( i == last ? "O" : "H" ) );
			size = lsizes[i];
			pad  = max - size;
			str += " ".repeat(pad);

			foreach( item of 1..size ){
				str += ( (item == 1 ? "    " + sym + " " : sym + " ") );
			}	
			str += '\n';
		}
		return str;
	}
}





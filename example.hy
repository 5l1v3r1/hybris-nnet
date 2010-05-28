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
import  std.*;
include nnetwork;

function epoch_callback( epoch, error ){
	if( epoch % 100000 == 0 ){
		println( "EPOCH[" + epoch + "] : " + error );
	}
}

net = new NNetwork( [ 2, 5, 1 ], 0.6, 0.1 );

inputs = [
 [ 0.0, 0.0 ],
 [ 0.0, 1.0 ],
 [ 1.0, 0.0 ],
 [ 1.0, 1.0 ]
];

outputs = [
 [0.0], 
 [1.0], 
 [1.0],
 [0.0] 
];

epochs = net.train( inputs, outputs, 100000000, 0.00001, epoch_callback );

println( "TRAINING DONE IN " + epochs + " EPOCHS :\n" );

println( "      A         B        EXPECTED      RESULT\n" +
		 "----------------------------------------------");

foreach( i of 0..inputs.size() - 1 ){
	output = net.recognize(inputs[i]);
	println("( " + inputs[i].join(", ") + " ) ( " + outputs[i][0] + " ) : " + output );
}

println( "----------------------------------------------\n");


filename = "xor.nn";

println( "SAVING TO " + filename + " ... " + net.save(filename) + " BYTES WRITTEN.\n" );

println( "RELOADING FROM FILE {" );

xor = new NNetwork(filename);

println( "  L-Rate   : " + xor.lrate,
		 "  Momentum : " + xor.momentum,
		 "  Layers   : " + xor.layers_size.join(" - "),
		 "\n" + xor + "}\n" );

println( "      A         B        EXPECTED      RESULT\n" +
		 "----------------------------------------------");

foreach( i of 0..inputs.size() - 1 ){
	output = xor.recognize(inputs[i]);
	println("( " + inputs[i].join(", ") + " ) ( " + outputs[i][0] + " ) : " + output );
}

println( "----------------------------------------------\n");



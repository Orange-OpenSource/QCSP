"""
Software Name : QCSP Orange
SPDX-FileCopyrightText: Copyright (c) Orange SA
SPDX-License-Identifier: MIT

This software is distributed under the MIT License,
see the "LICENSE.txt" file for more details or https://spdx.org/licenses/MIT.html

Authors: 
    Louis-Adrien DUFRÈNE    louisadrien.dufrene@orange.com
    Guillaume LARUE         guillaume.larue@orange.com
    Quentin LAMPIN          quentin.lampin@orange.com

Software description: Orange study on the combination of CCSK and OFDM modulation. Part of the QCSP ANR project. See Deliverable D2.5b_OFDM-CCSK.pdf
"""

import tensorflow as tf
import numpy as np

from ccsk import cross_correlation_r


class DFTLinkDemodulator(tf.keras.layers.Layer):

    def __init__(self, root_sequence, resource_grid, reference_index_list=[0], submode='Standard', window_size=-1, **kwargs):
        super().__init__() # **kwargs
        
        self._root_sequence = root_sequence
        self._resource_grid = resource_grid
        self._num_ofdm_symb = self._resource_grid.num_ofdm_symbols
        self._ccsk_length = tf.size(self._root_sequence)
        self._ccsk_length_f = tf.cast(self._ccsk_length, dtype=tf.float32)
        self._reference_index_list = reference_index_list
        self._nb_reference = len(reference_index_list)
        
        # Demodulation Submode
        ## Standard
        ## GLAD
        self._submode = submode


        self._window_mode = False
        if (window_size >= 0) and (window_size < self._num_ofdm_symb):
            self._window_mode = True
            self._window_size = window_size
            # Provide the True and False vector to mask the relative shift matrix
            base_mask = tf.ones([self._num_ofdm_symb, self._num_ofdm_symb], dtype=tf.bool)
            self._windowing_mask = tf.linalg.band_part(base_mask, num_lower=self._window_size, num_upper=self._window_size)[tf.newaxis, :, :, tf.newaxis]


        if self._submode == 'Standard': 
            # Compute the location of the OFDM symbol containing data.
            ## Ex. References are located in OFDM symbols 0 and 3, and OFDM symbols go from 0 to 5.
            ## Data symbols are located at [1,2,4,5].
            self._data_indices_list = np.delete(np.arange(self._num_ofdm_symb), self._reference_index_list).tolist()

        elif self._submode == 'GLAD':
            self._iterations = kwargs.get('iterations', 5)
            self._data_indices_list = np.delete(np.arange(self._num_ofdm_symb), self._reference_index_list).tolist()

            self._initial_state_value = tf.constant(value=(1/self._ccsk_length_f),shape=(self._num_ofdm_symb*self._num_ofdm_symb,self._ccsk_length))
            cell = DFTLinkCell2(num_ofdm_symb=self._num_ofdm_symb, ccsk_length=self._ccsk_length, reference_index_list=self._reference_index_list)
            self.rnn = tf.keras.layers.RNN(cell, return_sequences=False, time_major=False)

            self.a = np.delete(np.arange(self._num_ofdm_symb, dtype=np.int32), self._reference_index_list)

    
    def _couples_index_(self):
        index_list = []
        for a in np.arange(self._num_ofdm_symb):
            for b in np.arange(self._num_ofdm_symb):
                if a >= b:
                    continue
                else:
                    index_list.append(tf.constant([a,b], dtype=tf.int32))
        out = tf.concat(index_list, axis=0)
        return out


    def _compute_shift_matrix_indices_(self):
        index_list = []
        index_std = 0
        autocorr_index = self._num_ofdm_symb**2 - self._num_ofdm_symb
        l, c = np.triu_indices(n=self._num_ofdm_symb-1, m=self._num_ofdm_symb-1, k=0)
        e = np.zeros(shape=[self._num_ofdm_symb-1, self._num_ofdm_symb-1], dtype=np.int32)
        e[l,c] = np.arange(self._num_couples)
        e = np.transpose(e)
        for a in np.arange(self._num_ofdm_symb):
            for b in np.arange(self._num_ofdm_symb):
                if a == b:
                    index_list.append(autocorr_index)
                elif a < b:
                    index_list.append(index_std)
                    index_std += 1
                else:
                    index_list.append(e[a-1,b] + self._num_couples)
        return index_list


    def _compute_update_shift_indices_(self):
        index_list = []
        for a in np.arange(self._num_ofdm_symb):
            for b in np.arange(self._num_ofdm_symb):
                if a == b:
                    continue
                else:
                    index_list.append(b + a*(self._num_ofdm_symb))
        return index_list
 

    def _compute_scatter_(self):
        a = tf.concat([tf.expand_dims(tf.range(self._num_ofdm_symb, dtype=tf.int32), axis=1), tf.expand_dims(tf.range(self._num_ofdm_symb, dtype=tf.int32), axis=1)], axis=1)
        a = tf.tile(a, multiples=[self.batch_size,1])
        b = tf.repeat(tf.expand_dims(tf.range(self.batch_size, dtype=tf.int32), axis=1), self._num_ofdm_symb, axis=0)
        c = tf.concat([b,a], axis=1)
        c = tf.reshape(c, shape=[1, self.batch_size, self._num_ofdm_symb, 3])
        return c


    def _couples_selection_(self):
        global_couples_selection = []
        for current_ofdm_symbol_index in np.arange(self._num_ofdm_symb):
            current_ones_index = []
            current_couples_selection = tf.zeros(shape=[int((self._num_ofdm_symb * (self._num_ofdm_symb-1)) / 2)], dtype=tf.float32)

            current_couple_index = 0
            for row_index in np.arange(self._num_ofdm_symb - 1):
                if np.abs(current_ofdm_symbol_index - row_index) > self._window_size:
                        current_couple_index += self._num_ofdm_symb - row_index - 1
                        continue
                for column_index in np.arange(self._num_ofdm_symb - 1 - row_index):
                    if np.abs(current_ofdm_symbol_index - (column_index + row_index + 1)) > self._window_size:
                        current_couple_index += self._num_ofdm_symb - row_index - 1 - column_index
                        break
                    current_ones_index.append(current_couple_index)
                    current_couple_index += 1
            
            nb_ones = len(current_ones_index)
            current_ones_index = tf.expand_dims(tf.constant(current_ones_index, dtype=tf.int32), axis=1)
            current_couples_selection = tf.tensor_scatter_nd_update(current_couples_selection, indices=current_ones_index, updates=tf.ones(shape=[nb_ones], dtype=tf.float32))
            current_couples_selection = tf.expand_dims(tf.expand_dims(tf.expand_dims(current_couples_selection, axis=0), axis=0), axis=-1)

            global_couples_selection.append(current_couples_selection)
        
        global_couples_selection = tf.concat(global_couples_selection, axis=1)

        return global_couples_selection


    def _windowing_selection_(self):
        global_ofdmsymbol_selection = []
        for current_ofdm_symbol_index in np.arange(self._num_ofdm_symb):
            current_ones_index = []
            current_ofdmsymol_selection = tf.zeros(shape=[int(self._num_ofdm_symb)], dtype=tf.float32)

            for row_index in np.arange(self._num_ofdm_symb):
                if np.abs(current_ofdm_symbol_index - row_index) <= self._window_size:
                    current_ones_index.append(row_index)

            nb_ones = len(current_ones_index)
            current_ones_index = tf.expand_dims(tf.constant(current_ones_index, dtype=tf.int32), axis=1)
            current_ofdmsymol_selection = tf.tensor_scatter_nd_update(current_ofdmsymol_selection, indices=current_ones_index, updates=tf.ones(shape=[nb_ones], dtype=tf.float32))
            current_ofdmsymol_selection = tf.expand_dims(tf.expand_dims(tf.expand_dims(current_ofdmsymol_selection, axis=0), axis=0), axis=-1)
            
            global_ofdmsymbol_selection.append(current_ofdmsymol_selection)

        global_ofdmsymbol_selection = tf.concat(global_ofdmsymbol_selection, axis=1)

        return global_ofdmsymbol_selection


    def _reference_index_gather_(self):
        output_gather_list = []

        for index in np.arange(len(self._reference_index_list)): 
            copy_index_list = self._reference_index_list.copy()
            current_reference_index = copy_index_list.pop(index)
            current_gather_array = np.arange(current_reference_index*(self._num_ofdm_symb-1),(current_reference_index+1)*(self._num_ofdm_symb-1))

            if len(copy_index_list) > 0:
                local_list = []
                for other_reference_index in copy_index_list:
                    if other_reference_index < current_reference_index:
                        local_list.append(other_reference_index)
                    if other_reference_index > current_reference_index:
                        local_list.append(other_reference_index-1)
                current_gather_array = np.delete(current_gather_array, local_list)

            output_gather_list.append(current_gather_array)
        
        output_gather_array = np.concatenate(output_gather_list, axis=0)

        return output_gather_array
    

    def _compute_data_indices_(self):
        data_indices_list = []
        for data_index in np.arange(self._num_ofdm_symb):
            if data_index not in self._reference_index_list:
                data_indices_list.append(data_index)
        return data_indices_list


    @staticmethod
    def _set_a_priori_absolute_shift_proba(num_ofdm_symb,ccsk_length,pilots_positions,pilots_shifts):
        """
        Define the a priori distributions for the absolute shift of each ofdm symbols in a frame of 'num_odfm_symb' symbols of length 'ccsk_length'
        given the position of the pilots in the frame and their known shift 

        Args:
            num_ofdm_symb (int32): The number of ofdm symbols in a frame
            ccsk_length (int32): The length of each ofdm symbol
            pilots_positions ([int32]): The positions of the pilot sequences
            pilots_shifts ([int32]): The shift value of the pilot sequences

        Returns:
            _type_: The a priori absolute shift probability of the ofdm symbols in the frame
        """
        ccsk_length_f = tf.cast(ccsk_length,dtype=tf.float32)

        # Convert the indices to a list if it's a single int32 value
        if isinstance(pilots_positions, int):
            pilots_positions = [pilots_positions]
        if isinstance(pilots_shifts, int):
            pilots_shifts = [pilots_shifts]

        pilots_positions = tf.convert_to_tensor(pilots_positions, dtype=tf.int32)
        pilots_shifts = tf.convert_to_tensor(pilots_shifts, dtype=tf.int32)

        assert tf.reduce_all(tf.equal(pilots_positions.shape,pilots_shifts.shape))

        # Create a base tensor with uniform distributions everywhere
        init = tf.constant(value=(1/ccsk_length_f),shape=[num_ofdm_symb,ccsk_length],dtype=tf.float32)

        # Create an updates tensor with ones at the specified indices
        updates = tf.one_hot(indices=pilots_shifts, depth=ccsk_length, on_value=1, off_value=0,dtype=tf.float32)

        # Use tf.tensor_scatter_nd_update to set the values at the specified indices
        scatter_indices = tf.expand_dims(pilots_positions, axis=1)

        updated_a_priori_absolute_shift_proba = tf.tensor_scatter_nd_update(init, scatter_indices, updates)

        return updated_a_priori_absolute_shift_proba


    def build(self, input_shape):
        self.batch_size = input_shape[0]


    def call(self, inputs):
        relative_shift_probability_matrix = inputs # input shape = [batch_size, num_ofdm_symb, num_ofdm_symb, fft_size]

        if self._window_mode == True:
            # We only consider the equalized symbols next to the current one, in a window of configurable size.
            # Basically the relative shift values that are out of the window (value False in the windowing matrix) are replaced with uniform probability values 1/N.
            ## Ex. if the window size is 1 and OFDM symbols are [0,1,2,3], the windowing matrix would be equal to:
            ## [True, True, False, False]
            ## [True, True, True, False]
            ## [False, True, True, True]
            ## [False, False, True, True]
            relative_shift_probability_matrix = tf.where(self._windowing_mask, relative_shift_probability_matrix, 1/self._ccsk_length_f) # output shape = [batch_size, num_ofdm_symb, num_ofdm_symb, fft_size]
        

        if self._submode == 'Standard':
            # Get the matrix lines at references indices.
            ## Ex. If references OFDM symboles indices are [0,19], then get [:,0,:,:] and [0,19,:,:].
            reference_relative_shift = tf.gather(relative_shift_probability_matrix, indices=self._reference_index_list, axis=1) # output shape = [batch_size, nb_reference, num_ofdm_symb, fft_size]
            
            # Compute the dot product of the relatice shift probabilities. Then normalize as a probability.
            ## Ex. Update P_X as the dot product of P_X/0 and P_X/19, and normalize the result so the sum = 1.
            output_shift_probability_with_ref = tf.reduce_prod(reference_relative_shift, axis=1) # output shape = [batch_size, num_ofdm_symb, fft_size]
            output_shift_probability_with_ref = output_shift_probability_with_ref / tf.reduce_sum(output_shift_probability_with_ref, axis=-1, keepdims=True) # output shape = [batch_size, num_ofdm_symb, fft_size]
            
            # Get only the estimated shift for the OFDM symbols carrying data.
            ## Ex. Data OFDM symbols indices are [1,2,...,17,18].
            output_shift_probability = tf.gather(output_shift_probability_with_ref, indices=self._data_indices_list, axis=1) # output shape = [batch_size, num_ofdm_symb - nb_reference, fft_size]

        elif self._submode == 'GLAD':   
            # This scaling is used to make the cross-correlation looks like a probability
            relative_shift_probability_matrix = relative_shift_probability_matrix / tf.reduce_sum(relative_shift_probability_matrix, axis=-1, keepdims=True) # output shape = [batch_size, num_ofdm_symb, num_ofdm_symb, fft_size]

            # Duplicate input matrix to the number of RNN iterations to be executed. 
            # Currently the same inputs are provided to each RNN iteration.
            relative_shift_probability_matrix_iterations = tf.repeat(tf.expand_dims(relative_shift_probability_matrix, axis=1), repeats=self._iterations, axis=1) # output shape = [batch_size, iterations, num_ofdm_symb, num_ofdm_symb, fft_size]
            #tf.print(relative_shift_probability_matrix_iterations,summarize=-1)

            # # Test - Inject data only at first iteration
            # iteration_mask = tf.reshape(tf.one_hot(indices=[0],depth=self._iterations), shape=[1,self._iterations,1,1,1])
            # #tf.print(iteration_mask)
            # relative_shift_probability_matrix_iterations = iteration_mask*relative_shift_probability_matrix_iterations + (1-iteration_mask)*(1/self._ccsk_length_f)
            # #tf.print(relative_shift_probability_matrix_iterations,summarize=-1)

            # Define the initial RNN states value.
            # In the current implementation, the states represents the (extrinsic) conditional probability distribution of each of the variables w.r.t. the others 
            # At the begining of the demodulation, we assume no a priori knowledge, hence the distribution of initialized to uniform distributions.
            batch_size = tf.shape(relative_shift_probability_matrix)[0]
            initial_state = tf.repeat(tf.expand_dims(self._initial_state_value, axis=0), repeats=batch_size, axis=0)

            # Start RNN demodulator
            # Return a matrix of probability of shape [batch_size, num_ofdm_symb, fft_size]
            # Simply take the highest probability (argmax axis=-1) for each OFDM symbol to get the most probable shift and hence the demodulation of the OFDM frame.
            output_shift_probability = self.rnn(inputs=relative_shift_probability_matrix_iterations, initial_state=initial_state)
            # output_shift_probability = tf.gather(output_shift_probability, indices=self.a, axis=1) # output shape = [batch_size, num_ofdm_symb - nb_references, fft_size]


            reference_relative_shift = tf.gather(output_shift_probability, indices=self._reference_index_list, axis=1) # output shape = [batch_size, nb_reference, num_ofdm_symb, fft_size]
            
            # Compute the dot product of the relatice shift probabilities. Then normalize again as a probability.
            ## Ex. Update P_X as the dot product of P_X/0 and P_X/3, and normalize the result so the sum = 1.
            output_shift_probability_with_ref = tf.reduce_prod(reference_relative_shift, axis=1) # output shape = [batch_size, num_ofdm_symb, fft_size]
            output_shift_probability_with_ref = output_shift_probability_with_ref # output shape = [batch_size, num_ofdm_symb, fft_size]
            
            # Get only the estimated shift for the OFDM symbols carrying data.
            ## Ex. Data OFDM symbols indices are [1,2].
            output_shift_probability = tf.gather(output_shift_probability_with_ref, indices=self._data_indices_list, axis=1) 

    
        return output_shift_probability # output shape = [batch_size, num_ofdm_symb - nb_references, fft_size]


# For logging purpose:
def print_colored_matrix(data_matrix, color_matrix, color_lookup_dict):

    for i in range(data_matrix.shape[0]):
        for j in range(data_matrix.shape[1]):
            print(color_lookup_dict[color_matrix[i,j].numpy()] + f"{data_matrix[i, j]:.0f}" + '\033[0m', end='\t')

        print()  # Move to the next row

def are_congruent_modulo(a, b, N):
    a = tf.cast(a,dtype=tf.int32)
    b = tf.cast(b,dtype=tf.int32)
    N = tf.cast(N,dtype=tf.int32)
    if (a % N) == (b % N):
        return 1
    else:
        return 0

def print_relative_shift_matrix(matrix, num_ofdm_symb, ccsk_length, name, compare_to=None):
    # Evaluate shifts
    matrix = tf.reshape(matrix, shape=[num_ofdm_symb, num_ofdm_symb, ccsk_length])
    estimated_shifts = tf.argmax(matrix, axis = -1)

    color_matrix = []
    # Check for inconsistency
    for i in range(num_ofdm_symb):
        tmp = []
        for j in range(num_ofdm_symb):
            tmp.append(are_congruent_modulo(estimated_shifts[i][j],-estimated_shifts[j][i],ccsk_length)) 
        color_matrix.append(tmp)

    color_matrix = tf.convert_to_tensor(color_matrix)

    if compare_to is not None:
        compare_to = tf.reshape(compare_to, shape=[num_ofdm_symb, num_ofdm_symb, ccsk_length])
        compare_to_estimated_shifts = tf.argmax(compare_to, axis = -1)
        differences = 2*tf.cast(tf.not_equal(estimated_shifts,compare_to_estimated_shifts), dtype=tf.int32)
        color_matrix = color_matrix + differences
    # Print results with colors :)
    color_lookup_dict = {
        0: '\033[91m',  # INCONSISTENT/NOT DIFFERENT    RED
        1: '\033[92m',  # CONSISTENT/NOT DIFFERENT      GREEN
        2: '\033[93m',  # INCONSISTENT/DIFFERENT        ORANGE
        3: '\033[94m',  # CONSISTENT/DIFFERENT          BLUE
    }
    print(name)
    print_colored_matrix(estimated_shifts, color_matrix, color_lookup_dict)

def print_relative_shift_matrix_certainty(matrix, num_ofdm_symb, ccsk_length, name):
    # Evaluate shifts
    matrix = tf.reshape(matrix, shape=[num_ofdm_symb, num_ofdm_symb, ccsk_length])
    estimated_shifts = tf.argmax(matrix, axis = -1)
    peak_proba = tf.reduce_max(matrix, axis=-1)

    color_matrix = []
    # Extract the peak probability of each vector to estimate certainty 
    for i in range(num_ofdm_symb):
        tmp = []
        for j in range(num_ofdm_symb):
            if peak_proba[i][j] < 0.05:
                tmp.append(0) 
            elif peak_proba[i][j] < 0.10:
                tmp.append(1)
            elif peak_proba[i][j] < 0.25:
                tmp.append(2)
            elif peak_proba[i][j] < 0.5:
                tmp.append(3)
            elif peak_proba[i][j] < 0.75:
                tmp.append(4) 
            elif peak_proba[i][j] < 0.9:
                tmp.append(5) 
            else:
                tmp.append(6) 
        color_matrix.append(tmp)

    color_matrix = tf.convert_to_tensor(color_matrix)

    # Print results with colors :)
    color_lookup_dict = {
        0: '\033[0;35m',  # PINK - Very Very Uncertain  p < 0.05            
        1: '\033[0;31m',  # RED - Very uncertain        p < 0.10        
        2: '\033[0;33m',  # YELLOW - Uncertain          p < 0.25    
        3: '\033[0;34m',  # BLUE - Neutral              p < 0.5
        4: '\033[0;36m',  # CYAN - Confident            p < 0.75    
        5: '\033[0;32m',  # GREEN - Very confident      p < 0.9        
        6: '\033[0;37m',  # WHITE - Very Certain        p > 0.9        
    }
    print(name)
    print_colored_matrix(estimated_shifts,color_matrix,color_lookup_dict)
       

class DFTLinkCell2(tf.keras.layers.Layer):

    def __init__(self, num_ofdm_symb, ccsk_length, reference_index_list=[0], **kwargs):
        # The DFT link cell defines a message-passing like demodulation approach which iteratively combines the measured relative shifts 
        # between observed symbols to recover their absolute shifts (hence, the transmitted symbols)
        super().__init__(**kwargs)
        self._num_ofdm_symb = num_ofdm_symb
        self._ccsk_length = ccsk_length
        self._reference_index_list = reference_index_list

        # state contains all the extrinsic conditional probabilities computed during previous iteration [num_ofdm_symb*(num_ofdm_symb-1),fft_size]
        self.state_size = tf.TensorShape((self._num_ofdm_symb*self._num_ofdm_symb,self._ccsk_length))

        # output contains the a posteriori absolute shift probabilities after demodulation [num_ofdm_symb,fft_size]
        self.output_size = tf.TensorShape((self._num_ofdm_symb,self._ccsk_length))

        self._equality_constraints_list = self._generate_equality_constraints_list_from_pilot_idx(self._reference_index_list, self._num_ofdm_symb)
        self._equality_constraints_list_vector = DFTLinkCell2._matrix_indices_to_vector_indices(self._equality_constraints_list, self._num_ofdm_symb) #shape: [num_of_equality_constraints,num_of_variable_in_each_constraint]

    @staticmethod
    def _matrix_indices_to_vector_indices(matrix_indices_tuple_list, num_ofdm_symbols):
        input_shape = tf.shape(matrix_indices_tuple_list)
        matrix_indices_tuple_list = tf.reshape(matrix_indices_tuple_list,[-1,(2)])
        vector_indices_list = []
        for (col,row) in matrix_indices_tuple_list:
            vector_indices_list.append(col + num_ofdm_symbols * row)
        vector_indices_list = tf.reshape(vector_indices_list,input_shape[:-1])
        return vector_indices_list
    
    @staticmethod
    def _generate_equality_constraints_list_from_pilot_idx(pilot_idx_list,num_ofdm_symb):
        out = []
        for p_i in range(num_ofdm_symb):
            tmp = []
            tmp_reversed = []
            for p_j in pilot_idx_list:
                tmp.append((p_i,p_j))
                tmp_reversed.append((p_j,p_i))
            out.append(tmp)
            out.append(tmp_reversed)
        return out
    

    def call(self, inputs, states):
        # Start of iterative decoding
        inputs_shape = tf.shape(inputs)
        batch_size = inputs_shape[0]

        # Flatten inputs matrix
        inputs = tf.reshape(inputs, shape=[batch_size, self._num_ofdm_symb*self._num_ofdm_symb, -1])
        inputs = self._replace_diagonal_elt_by_dirac(inputs, self._num_ofdm_symb)
        inputs = self._normalize_distribution(inputs)
        #print_relative_shift_matrix(inputs[0], self._num_ofdm_symb, self._ccsk_length, "inputs")
        #print_relative_shift_matrix_certainty(inputs[0], self._num_ofdm_symb, self._ccsk_length, "inputs certainty")

        # Recover states from previous iteration - Extrinsic conditional probabilities 
        prev_state = states[0]   #shape: [batch_size,num_ofdm_symb*num_ofdm_symb,fft_size]

        # Combine extrinsic with a priori  
        x = tf.multiply(prev_state, inputs) #shape: [batch_size,num_ofdm_symb*num_ofdm_symb,fft_size]
        x = self._normalize_distribution(x)
        #print_relative_shift_matrix(x[0], self._num_ofdm_symb, self._ccsk_length, "x (compared to inputs)", compare_to=inputs[0])
        #print_relative_shift_matrix_certainty(x[0], self._num_ofdm_symb, self._ccsk_length, "x certainty")

        # Replace pilots related and diagonal positions element by dirac distributions
        x = self._update_relative_shift_equality_constraints(x, self._equality_constraints_list_vector)
        x = self._normalize_distribution(x)
        #print_relative_shift_matrix(x[0], self._num_ofdm_symb, self._ccsk_length, "x")

        ###################### 1 - Extrinsic Computation - State Update ##############################      
        # Update extrinsic information (Pa/b = Pa/c*Pb/c)
        new_state = self._update_messages_simplified(x, self._num_ofdm_symb, apply_mask=True, keep_symetric=False, normalize=False)
        new_state = self._normalize_distribution(new_state)  
        
        #print_relative_shift_matrix(prev_state[0], self._num_ofdm_symb, self._ccsk_length, "prev_state(compared to new_state)", compare_to=new_state[0])
        #print_relative_shift_matrix(new_state[0], self._num_ofdm_symb, self._ccsk_length, "new_state (compared to inputs)", compare_to=inputs[0])
        #print_relative_shift_matrix_certainty(new_state[0], self._num_ofdm_symb, self._ccsk_length, name="x_1*x_2")

        ###################### 2 - A Posteriori Computation - Output Update ############################## 
        # Update extrinsic information (Pa/b = Pa/c*Pb/c) - No Masking (keep all information)
        # Replace pilots related and diagonal positions element by dirac distributions
        new_output = self._update_messages_simplified(x, self._num_ofdm_symb, apply_mask=False, keep_symetric=False, normalize=False)
        new_output = self._normalize_distribution(new_output)
        new_output = self._update_relative_shift_equality_constraints(new_output, self._equality_constraints_list_vector)
        new_output = self._normalize_distribution(new_output)
        #print_relative_shift_matrix(new_output[0], self._num_ofdm_symb, self._ccsk_length, "new_output (compared to inputs)", compare_to=inputs[0])
        #print_relative_shift_matrix_certainty(new_output[0], self._num_ofdm_symb, self._ccsk_length, "new_output certainty")

        # Outputs value of the updated relative shift matrix
        new_output = tf.reshape(new_output, shape=[batch_size,self._num_ofdm_symb,self._num_ofdm_symb,-1])[:,:,:,:]   #shape: [batch_size,num_ofdm_symb,num_ofdm_symb,fft_size]

        return new_output, [new_state]


    @staticmethod
    def _update_relative_shift_equality_constraints(flattened_relative_shift_matrices, equality_constraints_list_vector):
        batch_size = tf.shape(flattened_relative_shift_matrices)[0]

        # Gather entries corresponding to the vector indices
        updates = tf.gather(params=flattened_relative_shift_matrices, indices=equality_constraints_list_vector, axis=1) #shape: [batch_size,num_of_equality_constraints, num_of_variable_in_each_constraint, ccsk_length]

        # Reduce the distribution of all variables participating in an equality constraint
        updates = tf.reduce_prod(updates, axis=-2) #shape: [batch_size,num_of_equality_constraints,ccsk_length]

        # Repeat the result to match the number of variable participating in a constraint - to replace all the variables participating to the constraint by the result
        updates = tf.repeat(tf.expand_dims(updates, axis=-2),repeats=tf.shape(equality_constraints_list_vector)[1],axis=-2)

        # Flatten updates to work with tf.scatter_nd_update function
        flattened_updates = tf.reshape(updates, shape=[-1,tf.shape(updates)[-1]])#tf.shape(equality_constraints_list_vector)[1]])

        # Similarly, flatten the update vector indices
        flattened_vector_indices = tf.reshape(equality_constraints_list_vector, shape=[-1,1])

        # Create a range of batch indices
        batch_indices = tf.range(batch_size)

        # Repeat the batch_indices to match the number of update vector indices
        num_of_update_indices = tf.shape(flattened_vector_indices)[0]
        repeated_batch_indices = tf.reshape(tf.repeat(batch_indices, repeats=num_of_update_indices), shape=[-1,1])

        # Tile the indices to match the batch size
        tiled_vector_indices = tf.tile(flattened_vector_indices, [batch_size, 1])

        # Concatenate the batch_indices and indices along the second dimension to create batched scatter_nd_update indices
        batched_update_indices = tf.concat([repeated_batch_indices, tiled_vector_indices], axis=1)

        # Scatter and update
        updated_flattened_relative_shift_matrix = tf.tensor_scatter_nd_update(flattened_relative_shift_matrices,batched_update_indices,flattened_updates)

        return updated_flattened_relative_shift_matrix
     

    @staticmethod
    def _select_messages_1(inputs,num_ofdm_symb):
        # Expected input_shape: [-1, num_ofdm_symb*num_ofdm_symb,fft_size]
        batch_size = tf.shape(inputs)[0]
        outputs =  tf.reshape(tf.repeat(inputs,repeats=num_ofdm_symb,axis=-3), shape = [batch_size,num_ofdm_symb*num_ofdm_symb*num_ofdm_symb,-1])

        # Expected output_shape: [-1, num_ofdm_symb*num_ofdm_symb*num_ofdm_symb,fft_size]
        return outputs


    @staticmethod
    def _select_messages_2(inputs,num_ofdm_symb):
        # Expected input_shape: [-1, num_ofdm_symb*num_ofdm_symb,fft_size]
        batch_size = tf.shape(inputs)[0]
        x = tf.reshape(inputs, shape = [batch_size,num_ofdm_symb,num_ofdm_symb,-1])
        x = tf.transpose(x, perm = [0,2,1,3])
        x = tf.reshape(x, shape = [batch_size,num_ofdm_symb*num_ofdm_symb,-1])
        outputs =  tf.repeat(x,repeats=num_ofdm_symb,axis=-2)

        # Expected output_shape: [-1, num_ofdm_symb*num_ofdm_symb*num_ofdm_symb,fft_size]
        return outputs
    

    @staticmethod
    def _select_updated_messages_simplified(inputs,num_ofdm_symb):
        # Expected input_shape: [-1, num_ofdm_symb*num_ofdm_symb*num_ofdm_symb,fft_size]
        batch_size = tf.shape(inputs)[0]
        outputs = tf.reshape(inputs, shape = [batch_size,num_ofdm_symb,num_ofdm_symb*num_ofdm_symb,-1])

        # Expected output_shape: [-1, num_ofdm_symb, num_ofdm_symb*num_ofdm_symb,fft_size]
        return outputs


    @staticmethod
    def _select_updated_messages_simplified_mask(num_ofdm_symb,keep_symetric,invert_masks=False):
        # Expected input_shape: None
        mask = tf.linalg.eye(num_ofdm_symb,dtype=tf.bool)
        mask = tf.repeat(mask,repeats=num_ofdm_symb,axis=-1)

        symetric_mask = tf.reshape(mask,[num_ofdm_symb,num_ofdm_symb,num_ofdm_symb])
        symetric_mask = tf.transpose(symetric_mask, perm=[2,1,0])
        symetric_mask = tf.reshape(symetric_mask, [num_ofdm_symb,num_ofdm_symb*num_ofdm_symb])

        if invert_masks:
            tmp_mask = mask
            mask = symetric_mask
            symetric_mask = tmp_mask

        if not keep_symetric:
            mask = tf.logical_or(mask,symetric_mask)
        
        mask = tf.reshape(mask, [num_ofdm_symb,num_ofdm_symb*num_ofdm_symb,1])
        return mask   
        # Expected output_shape: [num_ofdm_symb, num_ofdm_symb*num_ofdm_symb,1]
    

    @staticmethod
    def _select_updated_messages_simplified_masked(inputs, num_ofdm_symb, apply_mask=True, keep_symetric=False, normalize=False, invert_masks=False):
        """
        Args:
            inputs (float tensor): Vector containing all of the num_ofdm_symb*num_ofdm_symb relative shift signals (of size fft_size)
            num_ofdm_symb (int): num of transmitted ofdm symbole
            apply_mask (bool): whether to mask the outputs to remove intrinsic and symetric information
            keep_symetric (bool): wheter to consider symetric relative shift as intrinsic information or not, and thus discard it (eg if we consider symetric shift as intrinsic info - keep_symetric=True - 0/1 will not be taken into accound in the computation of 1/0)
            normalize (bool): wheter to replace the masked intrinsic information by a uniform distribution of value 1/fft_size (normalize = True) or simply by the value 1. can have an impact in terms of weighting of the other distributions

        Returns:
            float tensor:
        """
        # Expected input_shape: [batch_size,num_ofdm_symb*num_ofdm_symb*num_ofdm_symb,fft_size]
        outputs = DFTLinkCell2._select_updated_messages_simplified(inputs, num_ofdm_symb)
        if apply_mask:
            mask = DFTLinkCell2._select_updated_messages_simplified_mask(num_ofdm_symb,keep_symetric,invert_masks)
            shape = tf.shape(inputs)

            if normalize:
                value = 1/float(shape[-1])
            else:
                value = 1.
            
            masked_outputs = tf.where(mask,value,outputs)

            return masked_outputs
        else:
            return outputs        
        # Expected output_shape: [batch_size, num_ofdm_symb,num_ofdm_symb*num_ofdm_symb,fft_size]


    @staticmethod
    def _update_messages_1(inputs,num_ofdm_symb):
        # Expected input_shape: [batch_size, num_ofdm_symb*num_ofdm_symb,fft_size]
        x = DFTLinkCell2._select_messages_1(inputs, num_ofdm_symb)
        y = DFTLinkCell2._select_messages_2(inputs, num_ofdm_symb)
        
        outputs = cross_correlation_r(x,y)         
        outputs = DFTLinkCell2._normalize_distribution(outputs)  

        # Expected output_shape: [batch_size, num_ofdm_symb*num_ofdm_symb*num_ofdm_symb,fft_size]
        return outputs
    

    @staticmethod
    def _update_messages_simplified(inputs, num_ofdm_symb, apply_mask, keep_symetric=False, normalize=False):
        # Expected input_shape: [batch_size, num_ofdm_symb*num_ofdm_symb*num_ofdm_symb,fft_size]
        inputs_shape = tf.shape(inputs)
        batch_size = inputs_shape[0]
        
        updated_messages = DFTLinkCell2._update_messages_1(inputs,num_ofdm_symb)

        outputs = DFTLinkCell2._select_updated_messages_simplified_masked(updated_messages, num_ofdm_symb, apply_mask, keep_symetric, normalize, invert_masks=False)
        outputs = tf.reshape(outputs, shape=[batch_size,num_ofdm_symb,num_ofdm_symb,num_ofdm_symb,-1])
        next_power_of_2_exponent = tf.cast(tf.math.ceil(tf.math.log(tf.cast(num_ofdm_symb, tf.float32))/tf.math.log(2.)), tf.int32)
        outputs = DFTLinkCell2._recursive_reduce_prod(outputs,depth=next_power_of_2_exponent,axis=-3,normalization_axis=-1)
        outputs = tf.reshape(outputs, shape=[batch_size,num_ofdm_symb*num_ofdm_symb,-1])
        outputs = DFTLinkCell2._normalize_distribution(outputs)  
        # Expected output_shape: [batch_size, num_ofdm_symb*num_ofdm_symb,fft_size]
        return outputs
    

    @staticmethod
    def _replace_diagonal_elt_by_dirac(inputs, num_ofdm_symb):
        shape = tf.shape(inputs)
        batch_size = shape[0]
        fft_size = shape[-1]

        x = tf.reshape(inputs, shape=[batch_size,num_ofdm_symb,num_ofdm_symb,-1])

        mask = tf.reshape(tf.linalg.eye(num_ofdm_symb), shape=[1,num_ofdm_symb,num_ofdm_symb,1])
        
        x = (1-mask)*x + mask*tf.one_hot(0,on_value=1.,off_value=0.,depth=fft_size)

        outputs = tf.reshape(x, shape=[batch_size,num_ofdm_symb*num_ofdm_symb,fft_size])

        return outputs

    
    @staticmethod
    def _normalize_distribution(inputs, axis=-1):
        inputs_shape = tf.shape(inputs)
        N = tf.cast(inputs_shape[axis], dtype=tf.float32)
        inputs_sum = tf.reduce_sum(inputs,axis=axis,keepdims=True)        
        normalized = tf.where(tf.equal(inputs_sum,0),tf.fill(inputs_shape, 1/N),tf.math.divide_no_nan(inputs, inputs_sum))
        return normalized
    

    @staticmethod
    def _pad_to_power_of_2(inputs, axis):
        # Compute the size of the input tensor along the specified axis
        shape = tf.shape(inputs)
        padding_axis_size = shape[axis]
        
        # Compute the next power of 2 greater than or equal to the current size
        next_power_of_2 = 2 ** tf.cast(tf.math.ceil(tf.math.log(tf.cast(padding_axis_size, tf.float32))/tf.math.log(2.)), tf.int32)
        
        # Compute the amount of padding required on each side
        padding_required = next_power_of_2 - padding_axis_size
        pad_before = tf.where(tf.greater(padding_required, 0), padding_required // 2, 0)
        pad_after = tf.where(tf.greater(padding_required, 0), padding_required - pad_before, 0)
        
        # Create a padding tensor def
        axis_padding = [pad_before, pad_after]
        all_axis_padding = tf.concat([tf.zeros(shape=[axis%len(shape),2], dtype=tf.int32), [axis_padding], tf.zeros(shape=[(len(shape)-1-axis)%len(shape),2], dtype=tf.int32)], axis=0)
        
        # Pad the input tensor
        padded_tensor = tf.pad(inputs, paddings=all_axis_padding, constant_values=1.)
        
        return padded_tensor
    

    #@tf.function
    @staticmethod
    def _split_axis(inputs, axis_to_split):
        # Get the shape of the input tensor
        input_shape = tf.shape(inputs)
        axis_to_split = axis_to_split%len(input_shape)

        # Split the 'axis_to_split' dimension into two parts
        new_shape = tf.concat([input_shape[:axis_to_split], [-1, 2], input_shape[axis_to_split+1:]], axis=0)

        # Reshape the input tensor to the new shape
        outputs = tf.reshape(inputs, new_shape)

        return outputs


    @staticmethod  
    @tf.function
    def _recursive_reduce_prod(inputs, depth, axis=None, normalization_axis=None):
        # Product reduction ensuring numerical stability using product tree reduction
        def cond(depth, x):
            return tf.greater(depth, 0)
        
        def body(depth, x):
            # Pad to closest 2ˆk (if necessary)
            x = DFTLinkCell2._pad_to_power_of_2(x, axis=axis)

            # Reshape - Split
            x = DFTLinkCell2._split_axis(x, axis_to_split=axis)

            # Reduce Prod
            x = tf.reduce_prod(x, axis=axis+1)

            # Normalize
            if normalization_axis is not None:
                x = DFTLinkCell2._normalize_distribution(x, normalization_axis)

            depth = tf.subtract(depth,1)

            return depth,x

        # Handling negative/positive axis designation
        shape = tf.shape(inputs)  
        axis = axis%len(shape)
        normalization_axis = normalization_axis%len(shape)
        loop_shape_invariants = [None]*len(shape)#[64,8,None,8,64]#list(shape)[:axis] + [None] + list(shape)[axis+1:]

        x = inputs
        depth,x = tf.while_loop(cond, body, [depth,x], shape_invariants=[depth.get_shape(),tf.TensorShape(loop_shape_invariants)])

        return x

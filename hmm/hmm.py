import numpy as np
class HiddenMarkovModel:
    """
    Class for Hidden Markov Model 
    """

    def __init__(self, observation_states: np.ndarray, hidden_states: np.ndarray, prior_p: np.ndarray, transition_p: np.ndarray, emission_p: np.ndarray):
        """

        Initialization of HMM object

        Args:
            observation_states (np.ndarray): observed states 
            hidden_states (np.ndarray): hidden states 
            prior_p (np.ndarray): prior probabities of hidden states 
            transition_p (np.ndarray): transition probabilites between hidden states
            emission_p (np.ndarray): emission probabilites from transition to hidden states 
        """             
        # Check that all args are ndarrays
        if not isinstance(observation_states, np.ndarray):
            raise ValueError('observation_states is not a numpy array.')
        if not isinstance(hidden_states, np.ndarray):
            raise ValueError('hidden_states is not a numpy array.')
        if not isinstance(prior_p, np.ndarray):
            raise ValueError('prior_p is not a numpy array.')
        if not isinstance(transition_p, np.ndarray):
            raise ValueError('transition_p is not a numpy array.')
        if not isinstance(emission_p, np.ndarray):
            raise ValueError('emission_p is not a numpy array.')

        # Check that transition matrix is square
        if np.shape(transition_p)[0] != np.shape(transition_p)[1]:
            raise ValueError('Transition matrix must be square.')
        # Now that we've established that the transition matrix is square, check that num hidden states == dims of transition matrix
        if transition_p.shape[0] != len(hidden_states):
            raise ValueError('Mismatch in transition matrix dimensions and number of hidden states.')

        self.observation_states = observation_states
        self.observation_states_dict = {state: index for index, state in enumerate(list(self.observation_states))}

        self.hidden_states = hidden_states
        self.hidden_states_dict = {index: state for index, state in enumerate(list(self.hidden_states))}
        
        self.prior_p= prior_p
        self.transition_p = transition_p
        self.emission_p = emission_p


    def forward(self, input_observation_states: np.ndarray) -> float:
        """
        TODO 

        This function runs the forward algorithm on an input sequence of observation states

        Args:
            input_observation_states (np.ndarray): observation sequence to run forward algorithm on 

        Returns:
            forward_probability (float): forward probability (likelihood) for the input observed sequence  
        """        
        # Step 0. Trigger ValueErrors for edge cases

        # Check that input_observation_states is an np.ndarray
        if not isinstance(input_observation_states, np.ndarray):
            raise ValueError('input_observation_states is not a numpy array.')

        # Step 1. Initialize variables
        # Create forward matrix (num hidden states by num input obs states) and initialize to base case
        # shorthand for lengths
        num_h = self.hidden_states.shape[0]
        num_in = len(input_observation_states)

        fwd = np.zeros((num_h, num_in)) 
        for i in range(num_h):
            init_index = self.observation_states_dict[input_observation_states[0]]
            fwd[i,0] = self.prior_p[i] * self.emission_p[i, init_index]
       
        # Step 2. Calculate probabilities
        for i in range(1, num_in):
            # get index for each inputted state i 
            state_index = self.observation_states_dict[input_observation_states[i]]
            for j in range(num_h):
                # for each hidden state j, 
                # recursively sum across products of fwd, self.transition_p, and self.emission_p to calculate individual probabilities
                fwd[j,i] = np.sum(fwd[:,i-1] * self.transition_p[:,j] * self.emission_p[j, state_index])

        # Step 3. Return final probability (after summation)
        return np.sum(fwd[:,-1])

    def viterbi(self, decode_observation_states: np.ndarray) -> list:
        """
        TODO

        This function runs the viterbi algorithm on an input sequence of observation states

        Args:
            decode_observation_states (np.ndarray): observation state sequence to decode 

        Returns:
            best_hidden_state_sequence(list): most likely list of hidden states that generated the sequence observed states
        """        
        
        # Step 1. Initialize variables
        # shorthand for lengths
        num_h = self.hidden_states.shape[0]
        num_in = len(decode_observation_states)  
        #store probabilities of hidden state at each step 
        viterbi_table = np.zeros((num_h, num_in))
        #store best path for traceback
        best_path = np.zeros(num_in, dtype=int)       
        
        back = np.zeros((num_h, num_in), dtype=int)
       # Step 2. Calculate Probabilities
       # initialize vals
        for i in range(num_h):
            init_index = self.observation_states_dict[decode_observation_states[0]]
            viterbi_table[i,0] = self.prior_p[i] * self.emission_p[i,init_index]
            back[i,0] = 0

        for i in range(1,num_in):
            # get index for each inputted state i 
            init_index = self.observation_states_dict[decode_observation_states[i]]
            for j in range(num_h):
                # for each hidden state j, recursively calculate values
                vals = viterbi_table[:,i-1] * self.transition_p[:,j]
                viterbi_table[j,i] = np.max(vals) * self.emission_p[j,init_index]
                back[j,i] = np.argmax(vals)
            
        # Step 3. Traceback 
        best_prob = np.max(viterbi_table[:,-1])
        best_path[-1] = np.argmax(viterbi_table[:,-1])
        # use pointer to traceback path
        for i in range(num_in-1):
             best_path[i] = back[best_path[i+1], i+1]

        # Step 4. Return best hidden state sequence 
        best_path_seq = [str(self.hidden_states[i]) for i in best_path]
        return best_path_seq, best_prob
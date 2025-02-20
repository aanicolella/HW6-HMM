import pytest
from hmm import HiddenMarkovModel
import numpy as np

def test_mini_weather():
    """
    TODO: 
    Create an instance of your HMM class using the "small_weather_hmm.npz" file. 
    Run the Forward and Viterbi algorithms on the observation sequence in the "small_weather_input_output.npz" file.

    Ensure that the output of your Forward algorithm is correct. 

    Ensure that the output of your Viterbi algorithm correct. 
    Assert that the state sequence returned is in the right order, has the right number of states, etc. 

    In addition, check for at least 2 edge cases using this toy model. 
    """

    mini_hmm=np.load('./data/mini_weather_hmm.npz')
    mini_input=np.load('./data/mini_weather_sequences.npz')

    # create HMM object
    mini = HiddenMarkovModel(observation_states=mini_hmm['observation_states'],
                            hidden_states=mini_hmm['hidden_states'],
                            prior_p=mini_hmm['prior_p'],
                            transition_p=mini_hmm['transition_p'],
                            emission_p=mini_hmm['emission_p'])
    
    # run forward and assert expected p
    fwd_p = mini.forward(mini_input['observation_state_sequence'])
    assert np.isclose(fwd_p, 0.03506441162109375, rtol=1e-05)

    # run viterbi and assert expected best path prob
    states, best_path = mini.viterbi(mini_input['observation_state_sequence'])
    assert np.isclose(best_path, 0.0066203212359375, rtol=1e-05)
    # assert same number of predicted states and ground truth states and assert equivalence of values
    assert len(states) == len(mini_input['best_hidden_state_sequence'])
    assert [states[i] == mini_input['best_hidden_state_sequence'][i] for i in range(len(mini_input['best_hidden_state_sequence']))]

    # Now, show edge case triggers

    # 1) show that ValueError raised if transition matrix is not square
    with pytest.raises(ValueError):
        notSquare = np.array([[0.55, 0.45, 0.6], [0.3, 0.7, 0.6]])
        valErr = HiddenMarkovModel(observation_states=mini_hmm['observation_states'],
                            hidden_states=mini_hmm['hidden_states'],
                            prior_p=mini_hmm['prior_p'],
                            transition_p=notSquare,
                            emission_p=mini_hmm['emission_p'])
    # 2) show ValueError if transition_p dims don't match number of hidden_states
    with pytest.raises(ValueError):
        addStates = np.append(mini_hmm['hidden_states'],['newPrior'])
        valErr = HiddenMarkovModel(observation_states=mini_hmm['observation_states'],
                            hidden_states=addStates,
                            prior_p=mini_hmm['prior_p'],
                            transition_p=mini_hmm['transition_p'],
                            emission_p=mini_hmm['emission_p'])
    
def test_full_weather():

    """
    TODO: 
    Create an instance of your HMM class using the "full_weather_hmm.npz" file. 
    Run the Forward and Viterbi algorithms on the observation sequence in the "full_weather_input_output.npz" file
        
    Ensure that the output of your Viterbi algorithm correct. 
    Assert that the state sequence returned is in the right order, has the right number of states, etc. 

    """
    full_hmm=np.load('./data/full_weather_hmm.npz')
    full_input=np.load('./data/full_weather_sequences.npz')

    # create HMM object
    full = HiddenMarkovModel(observation_states=full_hmm['observation_states'],
                            hidden_states=full_hmm['hidden_states'],
                            prior_p=full_hmm['prior_p'],
                            transition_p=full_hmm['transition_p'],
                            emission_p=full_hmm['emission_p'])
    
    # run forward and assert expected p
    full_p = full.forward(full_input['observation_state_sequence'])
    assert np.isclose(full_p, 1.6864513843961343e-11, rtol=1e-11)

    # run viterbi and assert expected best path prob
    states, best_path = full.viterbi(full_input['observation_state_sequence'])
    assert np.isclose(best_path, 2.5713241344000005e-15, rtol=1e-15)
    # assert same number of predicted states and ground truth states and assert equivalence of values
    assert len(states) == len(full_input['best_hidden_state_sequence'])
    assert [states[i] == full_input['best_hidden_state_sequence'][i] for i in range(len(full_input['best_hidden_state_sequence']))]









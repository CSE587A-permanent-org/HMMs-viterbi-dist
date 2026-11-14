"""
Tests for Viterbi decoding implementation.

These tests are translated from the Mathematica notebooks:
- viterbiTestTiny.nb (unit tests)
- viterbiTestLarge.nb (integration test)
"""

import unittest
import os
import numpy as np
from numpy.testing import assert_allclose
from gradescope_utils.autograder_utils.decorators import weight
from cse587Autils.HMMObjects.HMM import HMM, calculate_accuracy

from assignment.HMMs_viterbi import viterbi_decode, _build_matrix, _traceback

# Get the path to the Data directory
DATA_DIR = os.path.join(os.path.dirname(__file__), "Data")

# Total of 20 points
class TestBuildMatrix(unittest.TestCase):
    """Tests for _build_matrix function"""
    
    @weight(0)
    def test_hmm_validity_check(self):
        """Verify that all HMM files are valid"""
        hmm_files = [
            "testHMM1.hmm",
            "testHMM2.hmm",
            "testHMM3.hmm",
            "testHMM4.hmm",
            "humanMalaria.hmm"
        ]

        for hmm_file in hmm_files:
            hmm = HMM.read_hmm_file(os.path.join(DATA_DIR, hmm_file))
            self.assertTrue(hmm.check_validity(), f"{hmm_file} should be valid")

    @weight(2)
    def test_build_matrix_hmm1_single_observation(self):
        """
        buildMatrix Test 1: Single observation with hmm1

        hmm1 starts in state 0 with probability 1.0.
        Viterbi columns are normalized to 1.0.
        """
        hmm = HMM.read_hmm_file(os.path.join(DATA_DIR, "testHMM1.hmm"))
        observation_seq = [0]  # Converted from Mathematica {1}
        result = _build_matrix(observation_seq, hmm)
        expected = np.array([[1.0, 0.0]])
        assert_allclose(result, expected, rtol=1e-10)

    @weight(3)
    def test_build_matrix_hmm1_impossible_observation(self):
        """
        buildMatrix Test 2: Impossible observation with hmm1

        hmm1 starts in state 0 with probability 1.0. But observation 2 has zero
        probability of being emitted from state 1. Therefore, we are asking the
        algorithm to decode an impossible observation sequence, and the result
        is indeterminate. The attempt to normalize a Viterbi vector of all zeros
        results in NaN's.

        Note: In Python/NumPy, this will produce NaN values instead of
        Mathematica's Indeterminate. We check for NaN.
        """
        hmm = HMM.read_hmm_file(os.path.join(DATA_DIR, "testHMM1.hmm"))
        observation_seq = [2]  # Converted from Mathematica {3}
        result = _build_matrix(observation_seq, hmm)
        # Should produce NaN values due to division by zero
        self.assertTrue(np.all(np.isnan(result)) or np.all(np.isinf(result)))

    @weight(3)
    def test_build_matrix_hmm1_two_observations(self):
        """
        buildMatrix Test 3: Two observations with hmm1

        Only State 0 can output Observation 0 and only State 1 can output
        observation 3.
        """
        hmm = HMM.read_hmm_file(os.path.join(DATA_DIR, "testHMM1.hmm"))
        observation_seq = [0, 3]  # Converted from Mathematica {1, 4}
        result = _build_matrix(observation_seq, hmm)
        expected = np.array([[1.0, 0.0], [0.0, 1.0]])
        assert_allclose(result, expected, rtol=1e-10)

    @weight(3)
    def test_build_matrix_hmm2_seven_observations(self):
        """
        buildMatrix Test 4: Seven observations with hmm2

        In hmm2, observations 0 and 3 can only be output from states 1 and 2,
        respectively.
        """
        hmm = HMM.read_hmm_file(os.path.join(DATA_DIR, "testHMM2.hmm"))
        observation_seq = [0, 3, 0, 0, 3, 3, 3]  # Converted from {1,4,1,1,4,4,4}
        result = _build_matrix(observation_seq, hmm)
        expected = np.array([
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [0.0, 1.0],
            [0.0, 1.0]
        ])
        assert_allclose(result, expected, rtol=1e-10)

    @weight(3)
    def test_build_matrix_hmm2_six_observations_with_rounding(self):
        """
        buildMatrix Test 5: Six observations with hmm2, rounded to 3 decimals
        Input: [0, 2, 1, 2, 1, 3]
        Expected (rounded to 3 decimals):
        [[1.0, 0.0], [0.1, 0.9], [0.9, 0.1], [0.1, 0.9], [0.9, 0.1], [0.0, 1.0]]

        In hmm2, observations 0 and 3 can only be output from states 0 and 1,
        respectively, but observations 1 and 2 are equally likely to come from
        either state. Both states prefer to transition into the other, so when
        the observations are equally likely under either state, they will tend
        to alternate.
        """
        hmm = HMM.read_hmm_file(os.path.join(DATA_DIR, "testHMM2.hmm"))
        observation_seq = [0, 2, 1, 2, 1, 3]  # Converted from {1, 3, 2, 3, 2, 4}
        result = _build_matrix(observation_seq, hmm)
        expected = np.array([
            [1.0, 0.0],
            [0.1, 0.9],
            [0.9, 0.1],
            [0.1, 0.9],
            [0.9, 0.1],
            [0.0, 1.0]
        ])
        assert_allclose(result, expected, rtol=1e-2, atol=1e-3)

    @weight(3)
    def test_build_matrix_hmm3_six_observations_cycling(self):
        """
        buildMatrix Test 6: Six observations with hmm3 (3 states, forced cycling)
        Input: {1, 3, 2, 3, 2, 4} (converted to 0-indexed)
        Expected (rounded to 3 decimals):
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0],
         [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]

        hmm3 has 3 states. Its transition probabilities force it to always cycle
        from states 0 to 1, 1 to 2, and 2 to 0, and its initial state distribution
        forces it to always start in state 0. The emission probabilities of each
        state are different, but that shouldn't affect the traceback because the
        transitions force it to cycle regardless of the input.
        """
        hmm = HMM.read_hmm_file(os.path.join(DATA_DIR, "testHMM3.hmm"))
        observation_seq = [0, 2, 1, 2, 1, 3]  # Converted from {1, 3, 2, 3, 2, 4}
        result = _build_matrix(observation_seq, hmm)
        expected = np.array([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0]
        ])
        assert_allclose(result, expected, rtol=1e-2, atol=1e-3)

    @weight(3)
    def test_build_matrix_hmm4_eight_observations(self):
        """
        buildMatrix Test 7: Eight observations with hmm4 (3 states)
        Input: [0, 3, 3, 3, 2, 3, 3, 3]
        Expected (rounded to 3 decimals):
        [[1.0, 0.0, 0.0], [0.05, 0.9, 0.05], [0.05, 0.05, 0.9], [0.9, 0.05, 0.05],
         [0.0, 0.0, 1.0], [0.9, 0.05, 0.05], [0.05, 0.9, 0.05], [0.05, 0.05, 0.9]]

        hmm4 has 3 states. Its transition probabilities prefer to cycle from
        states 0 to 1, 1 to 2, and 2 to 0, and its initial state distribution
        forces it to always start in state 0. Observations 0, 1, and 2 can only
        be emitted by states 0, 1, and 2, respectively. Observation 3 is equally
        likely from all states. Therefore, the decode will tend to cycle on
        observation 3, but will be pulled to the corresponding state on
        observations 0, 1, and 2.
        """
        hmm = HMM.read_hmm_file(os.path.join(DATA_DIR, "testHMM4.hmm"))
        observation_seq = [0, 3, 3, 3, 2, 3, 3, 3]  # Converted from Mathematica
        result = _build_matrix(observation_seq, hmm)
        expected = np.array([
            [1.0, 0.0, 0.0],
            [0.05, 0.9, 0.05],
            [0.05, 0.05, 0.9],
            [0.9, 0.05, 0.05],
            [0.0, 0.0, 1.0],
            [0.9, 0.05, 0.05],
            [0.05, 0.9, 0.05],
            [0.05, 0.05, 0.9]
        ])
        assert_allclose(result, expected, rtol=1e-2, atol=1e-3)

# Total of 17 points
class TestTraceback(unittest.TestCase):
    """Tests for _traceback function"""

    @weight(2)
    def test_traceback_hmm1_single_column(self):
        """
        traceback Test 1: Single observation matrix with hmm1

        This one-observation Viterbi matrix indicates that state 0 is the only
        possible state at observation 0.
        """
        hmm = HMM.read_hmm_file(os.path.join(DATA_DIR, "testHMM1.hmm"))
        viterbi_matrix = np.array([[1.0, 0.0]])
        result = _traceback(viterbi_matrix, hmm)
        expected = [0]
        self.assertEqual(result, expected)

    @weight(3)
    def test_traceback_hmm1_two_observations(self):
        """traceback Test 2: Two observations with hmm1"""
        hmm = HMM.read_hmm_file(os.path.join(DATA_DIR, "testHMM1.hmm"))
        viterbi_matrix = np.array([[1.0, 0.0], [0.0, 1.0]])
        result = _traceback(viterbi_matrix, hmm)
        expected = [0, 1]
        self.assertEqual(result, expected)

    @weight(3)
    def test_traceback_hmm2_seven_observations(self):
        """traceback Test 3: Seven observations with hmm2"""
        hmm = HMM.read_hmm_file(os.path.join(DATA_DIR, "testHMM2.hmm"))
        viterbi_matrix = np.array([
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [0.0, 1.0],
            [0.0, 1.0]
        ])
        result = _traceback(viterbi_matrix, hmm)
        expected = [0, 1, 0, 0, 1, 1, 1]
        self.assertEqual(result, expected)

    @weight(3)
    def test_traceback_hmm2_alternating_states(self):
        """traceback Test 4: Alternating states (matrix3) with hmm2"""
        hmm = HMM.read_hmm_file(os.path.join(DATA_DIR, "testHMM2.hmm"))
        viterbi_matrix = np.array([
            [1.0, 0.0],
            [0.1, 0.9],
            [0.9, 0.1],
            [0.1, 0.9],
            [0.9, 0.1],
            [0.0, 1.0]
        ])
        result = _traceback(viterbi_matrix, hmm)
        expected = [0, 1, 0, 1, 0, 1]
        self.assertEqual(result, expected)

    @weight(3)
    def test_traceback_hmm3_forced_cycle(self):
        """traceback Test 5: Forced cycle (matrix4) with hmm3"""
        hmm = HMM.read_hmm_file(os.path.join(DATA_DIR, "testHMM3.hmm"))
        viterbi_matrix = np.array([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0]
        ])
        result = _traceback(viterbi_matrix, hmm)
        expected = [0, 1, 2, 0, 1, 2]
        self.assertEqual(result, expected)

    @weight(3)
    def test_traceback_hmm4_preference_cycling(self):
        """traceback Test 6: Preference-based cycling with hmm4"""
        hmm = HMM.read_hmm_file(os.path.join(DATA_DIR, "testHMM4.hmm"))
        viterbi_matrix = np.array([
            [1.0, 0.0, 0.0],
            [0.05, 0.9, 0.05],
            [0.05, 0.05, 0.9],
            [0.9, 0.05, 0.05],
            [0.0, 0.0, 1.0],
            [0.9, 0.05, 0.05],
            [0.05, 0.9, 0.05],
            [0.05, 0.05, 0.9]
        ])
        result = _traceback(viterbi_matrix, hmm)
        expected = [0, 1, 2, 0, 2, 0, 1, 2]
        self.assertEqual(result, expected)

# Total 18 points
class TestViterbiDecode(unittest.TestCase):
    """Tests for viterbi_decode function"""

    @weight(2)
    def test_viterbi_basic_testhmm1(self):
        """Test 1: Basic decode with testHMM1.hmm"""
        hmm = HMM.read_hmm_file(os.path.join(DATA_DIR, "testHMM1.hmm"))
        # Mathematica uses 1-indexed, Python uses 0-indexed
        observation_seq = [0, 3]  # Converted from {1, 4}
        result = viterbi_decode(observation_seq, hmm)
        self.assertEqual(result, ["m", "h"])

    @weight(4)
    def test_viterbi_longer_sequence_testhmm2(self):
        """Test 2: Longer sequence with testHMM2.hmm"""
        hmm = HMM.read_hmm_file(os.path.join(DATA_DIR, "testHMM2.hmm"))
        observation_seq = [0, 2, 1, 2, 3]  # Converted from {1, 3, 2, 3, 4}
        result = viterbi_decode(observation_seq, hmm)
        self.assertEqual(result, ["m", "h", "m", "m", "h"])

    @weight(4)
    def test_viterbi_eight_obs_testhmm4(self):
        """Test 3: Eight-observation sequence with testHMM4.hmm"""
        hmm = HMM.read_hmm_file(os.path.join(DATA_DIR, "testHMM4.hmm"))
        observation_seq = [0, 3, 3, 3, 2, 3, 3, 3]  # Converted from Mathematica
        result = viterbi_decode(observation_seq, hmm)
        self.assertEqual(result, ["a", "b", "c", "a", "c", "a", "b", "c"])

    @weight(4)
    def test_viterbi_real_fasta_humanmalaria(self):
        """Test 4: Real FASTA sequence with humanMalaria.hmm"""
        hmm = HMM.read_hmm_file(os.path.join(DATA_DIR, "humanMalaria.hmm"))
        fasta_file = os.path.join(DATA_DIR, "veryShortFasta.fa")
        sequences = HMM.read_fasta(fasta_file)
        observation_seq = sequences[0]  # First sequence

        result = viterbi_decode(observation_seq, hmm)
        expected = ["M", "M", "H", "M", "M", "M", "M", "H",
                    "H", "H", "H", "H", "H", "H", "H"]
        self.assertEqual(result, expected)

    @weight(4)
    def test_viterbi_large_sequence_accuracy(self):
        """
        Test 5: Large sequence accuracy test
        Tests on mixed2.fa with humanMalaria.hmm
        Expected accuracy: 118,389 correct out of 175,569 total positions
        """
        hmm = HMM.read_hmm_file(os.path.join(DATA_DIR, "humanMalaria.hmm"))

        # Read the sequence to decode
        fasta_file = os.path.join(DATA_DIR, "mixed2.fa")
        sequences = HMM.read_fasta(fasta_file)
        observation_seq = sequences[0]

        # Read the key (correct answers)
        key_file = os.path.join(DATA_DIR, "mixed2key.fa")
        key_sequences = HMM.read_fasta(key_file)
        # The key uses 'h' and 'm' as single characters, need to convert to list
        key_seq_numeric = key_sequences[0]

        # Decode the sequence
        result = viterbi_decode(observation_seq, hmm)

        # Convert result to match key format
        # The key file might use lowercase 'h' and 'm'
        # Need to map state names to match the key
        state_mapping = {'H': 'h', 'M': 'm'}
        result_mapped = [state_mapping.get(s, s).lower() for s in result]

        # Convert numeric key to state names
        # The key uses integers where h and m are encoded
        # Need to determine the encoding used in the key file
        # Looking at the key, it should be a sequence of state indicators

        # Calculate accuracy
        # The key is already read as a sequence
        # We need to compare our decoded states with the key
        accuracy = calculate_accuracy(result, key_seq_numeric)

        # Expected: 118,389 correct out of 175,569
        self.assertEqual(len(observation_seq), 175569)
        self.assertEqual(accuracy, 118389)

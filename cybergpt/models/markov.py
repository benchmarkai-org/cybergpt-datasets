import numpy as np
from typing import List
import logging


class StructuredHMM:
    """
    Structured Hidden Markov Model with word-class-state emission probabilities
    """

    def __init__(self, n_states: int, n_classes: int, vocab_size: int):
        self.n_states = n_states
        self.n_classes = n_classes
        self.vocab_size = vocab_size

        # Initial state probabilities: π[i]
        self.pi = np.random.dirichlet(np.ones(n_states))

        # State transition matrix: A[i,j] = P(state_j | state_i)
        self.A = np.random.dirichlet(np.ones(n_states), size=n_states)

        # State to class emission matrix: B1[i,k] = P(class_k | state_i)
        self.B1 = np.random.dirichlet(np.ones(n_classes), size=n_states)

        # Class to vocabulary emission matrix: B2[k,v] = P(word_v | class_k)
        self.B2 = np.random.dirichlet(np.ones(vocab_size), size=n_classes)

    def _forward(self, obs: np.ndarray) -> tuple[np.ndarray, float]:
        """
        Forward algorithm for computing P(O|λ) and forward probabilities
        Returns alpha matrix and log probability of observation sequence
        """
        T = len(obs)
        alpha = np.zeros((T, self.n_states))

        # Compute emission probabilities for the sequence
        emission_probs = np.dot(self.B1, self.B2[:, obs])  # shape: (n_states, T)

        alpha[0] = self.pi * emission_probs[:, 0]
        scaling_factors = np.zeros(T)

        # Scale to prevent numerical underflow
        scaling_factors[0] = np.sum(alpha[0])
        alpha[0] /= scaling_factors[0]

        for t in range(1, T):
            alpha[t] = np.dot(alpha[t - 1], self.A) * emission_probs[:, t]
            scaling_factors[t] = np.sum(alpha[t])
            alpha[t] /= scaling_factors[t]

        log_prob = np.sum(np.log(scaling_factors))
        return alpha, log_prob

    def _backward(self, obs: np.ndarray, scaling_factors: np.ndarray) -> np.ndarray:
        """
        Backward algorithm for computing backward probabilities
        Uses same scaling as forward algorithm
        """
        T = len(obs)
        beta = np.zeros((T, self.n_states))

        # Compute emission probabilities for the sequence
        emission_probs = np.dot(self.B1, self.B2[:, obs])  # shape: (n_states, T)

        beta[-1] = 1.0 / scaling_factors[-1]
        for t in range(T - 2, -1, -1):
            beta[t] = np.dot(self.A, beta[t + 1] * emission_probs[:, t + 1])
            beta[t] /= scaling_factors[t]

        return beta

    def _compute_xi(
        self, obs: np.ndarray, alpha: np.ndarray, beta: np.ndarray
    ) -> np.ndarray:
        """Compute xi[t][i][j] = P(q_t = i, q_{t+1} = j | O, λ)"""
        T = len(obs)
        xi = np.zeros((T - 1, self.n_states, self.n_states))

        emission_probs = np.dot(self.B1, self.B2[:, obs])

        for t in range(T - 1):
            denominator = np.dot(
                np.dot(alpha[t], self.A) * emission_probs[:, t + 1], beta[t + 1]
            )
            for i in range(self.n_states):
                numerator = (
                    alpha[t, i]
                    * self.A[i, :]
                    * emission_probs[:, t + 1]
                    * beta[t + 1, :]
                )
                xi[t, i, :] = numerator / denominator

        return xi

    def _compute_gamma(self, alpha: np.ndarray, beta: np.ndarray) -> np.ndarray:
        """Compute gamma[t][i] = P(q_t = i | O, λ)"""
        gamma = alpha * beta
        gamma /= np.sum(gamma, axis=1, keepdims=True)
        return gamma

    def fit(
        self, sequences: List[np.ndarray], n_iterations: int = 100, tol: float = 1e-6
    ) -> List[float]:
        """
        Fit the HMM parameters using the Baum-Welch algorithm
        Returns the log likelihood history
        """
        log_likelihood_history = []

        for iteration in range(n_iterations):
            new_pi = np.zeros_like(self.pi)
            new_A = np.zeros_like(self.A)
            new_B1 = np.zeros_like(self.B1)
            new_B2 = np.zeros_like(self.B2)

            total_log_likelihood = 0

            # E-step: Accumulate statistics over all sequences
            for obs in sequences:
                # Forward-Backward
                alpha, seq_log_likelihood = self._forward(obs)
                total_log_likelihood += seq_log_likelihood

                scaling_factors = np.sum(alpha, axis=1)
                beta = self._backward(obs, scaling_factors)

                # Compute state and transition probabilities
                gamma = self._compute_gamma(alpha, beta)
                xi = self._compute_xi(obs, alpha, beta)

                # Accumulate statistics
                new_pi += gamma[0]
                new_A += np.sum(xi, axis=0)

                # Accumulate B1 and B2 statistics
                for t, o in enumerate(obs):
                    new_B1 += np.outer(gamma[t], self.B2[:, o])
                    for k in range(self.n_classes):
                        new_B2[k, o] += np.sum(gamma[t] * self.B1[:, k])

            # M-step: Update parameters
            self.pi = new_pi / np.sum(new_pi)
            self.A = new_A / np.sum(new_A, axis=1, keepdims=True)
            self.B1 = new_B1 / np.sum(new_B1, axis=1, keepdims=True)
            self.B2 = new_B2 / np.sum(new_B2, axis=1, keepdims=True)

            log_likelihood_history.append(total_log_likelihood)

            if iteration > 0:
                improvement = total_log_likelihood - log_likelihood_history[-2]
                if abs(improvement) < tol:
                    logging.info(f"Converged after {iteration + 1} iterations")
                    break

        return log_likelihood_history

    def decode(self, obs: np.ndarray) -> np.ndarray:
        """
        Viterbi algorithm to find most likely state sequence
        Returns the most likely state sequence
        """
        T = len(obs)
        emission_probs = np.dot(self.B1, self.B2[:, obs])

        delta = np.zeros((T, self.n_states))
        psi = np.zeros((T, self.n_states), dtype=int)

        delta[0] = np.log(self.pi) + np.log(emission_probs[:, 0])

        # Recursion
        for t in range(1, T):
            for j in range(self.n_states):
                temp = delta[t - 1] + np.log(self.A[:, j])
                psi[t, j] = np.argmax(temp)
                delta[t, j] = np.max(temp) + np.log(emission_probs[j, t])

        # Backtrack
        states = np.zeros(T, dtype=int)
        states[-1] = np.argmax(delta[-1])

        for t in range(T - 2, -1, -1):
            states[t] = psi[t + 1, states[t + 1]]

        return states

    def sample(self, length: int) -> np.ndarray:
        """Generate a sample sequence of given length"""
        state = np.random.choice(self.n_states, p=self.pi)
        obs = np.zeros(length, dtype=int)

        for t in range(length):
            class_idx = np.random.choice(self.n_classes, p=self.B1[state])
            obs[t] = np.random.choice(self.vocab_size, p=self.B2[class_idx])
            state = np.random.choice(self.n_states, p=self.A[state])

        return obs

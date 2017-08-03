
from ukf_particles import Particles
# TODO: Write a unit test
# TODO: Take a structure as an input ?

class BayesianState:
  # Would probably need some kind of structure here, to be checked

  def __init__(self,
               state_vector,
               state_covariance):
    self.vec = state_vector
    self.cov = state_covariance
    self.dim = np.shape(state_vector,0)


class UKF:

  def __init__(self, initial_state, initial_cov,
               propagation_function, measurement_function,
               process_noise, measurement_noise):

    self.measure_noise = measurement_noise
    self.process_noise = process_noise

    self._state_meas = self._state_pred = self._state_corr = initial_state
    self._innovation = np.zero(self._state_meas)

    self._cov_cross = self._cov_pred = self._cov_meas = initial_cov

    self._propagation_function = propagation_function
    self._measurement_function = measurement_function

    self._particles = Particles(initial_state, initial_cov, process_noise, measurement_function, propagation_function)


  def predict(self):
    self._particles.set_state(self._state_corr, self.process_noise)
    self._particles.propagate(self.prop_function)
    self._state_pred = self._particles.get_predicted_state()

  def update(self, obs_state):
    # Project on measurement space & Get expected measure
    self._state_meas, self._cov_cross = self._particles.measure_sigma()
    self._innovation = obs_state.vec - self._state_meas.vec

    # The gain is a function those state differences
    gain = self._compute_kalman_gain()

    # Compute a posteriori estimate
    self._state_corr.vec = self._state_pred.vec + gain * self._innovation
    self._state_corr.cov = self._state_pred.cov - \
        gain * obs_state.cov * gain.transpose()

  def get_predicted_state(self):
    return self._state_pred

  def get_corrected_state(self):
    return self._state_corr

  def get_innovation(self):
    return self._innovation

  def set_states(self, predicted_state, corrected_state):
    self._state_pred = predicted_state
    self._state_corr = corrected_state

  def _compute_kalman_gain(self):
    # TODO: Check with Julier's original paper
    innov_cov_inv = (self._state_meas.cov + self.measure_noise).inverse()
    return self._cov_cross * innov_cov_inv

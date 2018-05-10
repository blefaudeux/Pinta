
class Particles():

    def __init__(self):
        # TODO: Clean this mess
        self._w_mean = None
        self._w_cov = None
        self._ext_cov = None
        self._state = None
        self._state_meas = None
        self._state_corr = None
        self._cross_cov = None
        self._noise_process = None
        self._extDim = None

        self._compute_weights()
        self._compute_sigma()

    def _compute_weights(self):
        # lmb = 3. - self._extDim

        # m_weightMean.setOnes()
        # m_weightCov.setOnes()

        # m_weightMean(0) = m_weightCov(0) = m_lambda / (stateExtDim +
        # m_lambda)

        # // Distributed points..
        # T const weight = static_cast < T > (1. / (2. * (stateExtDim +
        # m_lambda)))

        # m_weightMean.template tail < 2 * stateExtDim > ().array() *= weight
        # m_weightCov.template tail < 2 * stateExtDim > ().array() *= weight

        # m_weightMean /= m_weightMean.sum()
        # m_weightCov /= m_weightCov.sum()
        pass

    def _compute_sigma():
        # // Compute a new set of particles from the corrected state
        # m_pointCorr.setZero()
        # m_pointCorr.template block < stateDim, 1 > (0, 0) = m_meanCorr

        # m_extCov.template topLeftCorner < stateDim, stateDim > () = m_covCorr
        # m_extCov.template bottomRightCorner < stateDim, stateDim > () =
        # m_noiseProcess

        # // Compute "square root matrix" on covariance to get sigma points
        # Eigen:: LLT < MatExtState > lltOfCov(m_extCov )
        # MatExtState const L = lltOfCov.matrixL()
        # T const scaling = std: : sqrt(T(stateExtDim + m_lambda))

        # for (int i=1
        #         i <= stateExtDim
        #         + +i)
        # {
        #     auto const offset = (L.col(i).array() * scaling).matrix()
        #     m_pointCorr.col(i) = m_pointCorr.col(0) + offset
        #     m_pointCorr.col(stateExtDim + i) = m_pointCorr.col(0) - offset
        # }
        pass

    def measure_sigma(self):
        # TODO: Measure beforehand ?
        return self._state_meas, self._cross_cov

    def propagate(self, prop_function):
        # // Propagate existing set of points(supposed to be representative)
        # m_pointPred = m_pointCorr
        # for (int i=0
        #         i < NPoints
        #         + +i)
        # {
        #     m_propagateFunc(m_pointCorr.template block < stateDim, 1 > (0, i),
        #                     m_pointPred.template block < stateDim, 1 > (0, i))
        # }

        # // Update state
        # m_meanPred = (m_pointPred * m_weightMean).template head < stateDim > ()
        # m_covPred.setZero()

        # for (int i=0
        #         i < NPoints
        #         + +i)
        # {
        #     m_covPred += m_weightCov(i)
        #     * ((m_pointPred.col(i).template head < stateDim > () - m_meanPred)
        #         * (m_pointPred.col(i).template head < stateDim > () - m_meanPred).transpose())
        # }
        pass

    def measure(self, meas_function):
        # // Compute the projection of the sigma points onto the measurement space
        # for (int i=0
        #      i < NPoints
        #      + +i)
        # {
        #     m_measurementFunc(m_pointPred.template block < stateDim, 1 > (0, i),
        #                       m_pointMeas.col(i))
        # }

        # // Compute the mean of the measured points:
        # m_meanMeas = m_pointMeas * m_weightMean
        # // Computes the mean automatically

        # // Compute the intrinsic covariance of the measured points:
        # m_covMeas.setZero()

        # for (int i=0
        #      i < NPoints
        #      + +i)
        # {
        #     m_covMeas += m_weightCov(i) * ((m_pointMeas.col(i) - m_meanMeas)
        #                                    * (m_pointMeas.col(i) - m_meanMeas).transpose())
        # }

        # // Compute the crossed covariance between measurement space and intrisic space
        # m_covXPredMeas.setZero()

        # for (int i=0
        #      i < NPoints
        #      + +i)
        # {
        #     m_covXPredMeas += m_weightCov(i)
        #     * ((m_pointPred.template block < stateDim, 1 > (0, i) - m_meanPred)
        #         * (m_pointMeas.template block < measDim, 1 > (0, i) - m_meanMeas).transpose())
        # }

        pass

    def get_state(self):
        return self._state

    def get_predicted_state(self):
        return self._state_pred

    def get_measured_state(self):
        return self._state_meas, self._cross_cov

    def set_state(self, state, noise):
        self._state = state
        self._noise_process = noise
        self._compute_sigma()

#!/usr/bin/python3

# TODO: Implement an UKF filter


class UKF:
    def __init__(self, initialState, propModel, measModel):
        self.measState = initialState
        self.propState = None
    
        self.measModel = measModel
        self.propModel = propModel

    def update(self, measure):
        pass

    def getState(self):
        pass

    


INPUTS = ['wind_speed', 'wind_angle_x', 'wind_angle_y', 'rudder_angle']
OUTPUTS = ['boat_speed']
NN_FILENAME = "trained/conv.pt"
HIDDEN_SIZE = 64
SEQ_LEN = 256
INPUT_SIZE = [len(INPUTS), SEQ_LEN]

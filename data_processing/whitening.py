def whiten_angle(data):
    # Given a data frame with the fields:
    # 'wind_angle', 'boat_speed', 'wind_speed', 'rudder_angle'

    # Duplicate the data frame and inverse both angles
    # (we suppose that the boat has a symmetrical behaviour wrt port/starboard)
    data_inverse = data
    data_inverse['wind_angle'].mul(-1)
    data_inverse['rudder_angle'].mul(-1)

    return data_inverse

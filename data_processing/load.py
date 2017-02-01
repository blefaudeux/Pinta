from data_processing.nmea2pandas import load_json

def load(filename, clean_data=True):
    df = load_json(filename, skip_zeros=True)
    
    if clean_data:
        df['rudder_angle'] -= df['rudder_angle'].mean()
        
        #TODO: Add more pre-processing, check data
    
    return df
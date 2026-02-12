from geopy import distance
from arrow import get
import numpy as np
import pandas as pd
import math

def mean_bearing(bearing1, bearing2):
    # Convert bearings from degrees to radians
    
    bearing1_rad = math.radians(bearing1)
    bearing2_rad = math.radians(bearing2)
    
    # Compute the sine and cosine of each bearing
    sin1, cos1 = math.sin(bearing1_rad), math.cos(bearing1_rad)
    sin2, cos2 = math.sin(bearing2_rad), math.cos(bearing2_rad)
    
    # Average the sine and cosine values
    sin_mean = (sin1 + sin2) / 2
    cos_mean = (cos1 + cos2) / 2
    
    # Compute the arctangent of the average sine and cosine values
    mean_bearing_rad = math.atan2(sin_mean, cos_mean)
    
    # Convert the result back to degrees
    mean_bearing_deg = math.degrees(mean_bearing_rad)
    
    # Normalize the result to be within 0 to 360 degrees
    if mean_bearing_deg < 0:
        mean_bearing_deg += 360
        
    #if bearing1<bearing2:
    return mean_bearing_deg
    
    #else :
     #   return 180 - mean_bearing_deg
    
def subtract_angles(angle1, angle2):
    """
    Subtracts two angles and wraps the result to be within the range -180° to +180°.

    Parameters:
        angle1 (float): The first angle in degrees.
        angle2 (float): The second angle in degrees.

    Returns:
        float: The result of the subtraction, wrapped to the range -180° to +180°.
    """
    result = angle1 - angle2
    # Wrap the result within -180° to +180°
    while result > 180:
        result -= 360
    while result <= -180:
        result += 360
    return result

def boat_summary(data, WAND):

    tack_summary = pd.DataFrame()
    # adding absolute value of yaw rate:
    #data['abs_Yaw_rate'] = abs((data.HEADING_deg%180).diff())
    # Find all tack and gybes
    
    
   
    
    
    sign = data[f'TWA_{WAND}_SGP_deg'] > 0
    sign_change = (sign != sign.shift(1))
    mans = data[sign_change == True]
    # filtering the ones where the baot is stopped:
    mans = mans[mans.BOAT_SPEED_km_h_1 > 2]
    

    # getting tack data:
    for tman in mans.DATETIME.tolist():
        tman = get(tman)
        start = tman.shift(seconds=-10).format('YYYY-MM-DD HH:mm:ss')
        start_d = tman.shift(seconds=-20).format('YYYY-MM-DD HH:mm:ss')
        stop = tman.shift(seconds=+10).format('YYYY-MM-DD HH:mm:ss')
        stop_d = tman.shift(seconds=+20).format('YYYY-MM-DD HH:mm:ss')
        man_start = tman.shift(seconds=-3).format('YYYY-MM-DD HH:mm:ss')
        man_stop = tman.shift(seconds=+3).format('YYYY-MM-DD HH:mm:ss')
        build = tman.shift(seconds=+30).format('YYYY-MM-DD HH:mm:ss')
        tdata = data[(data.DATETIME <= stop) & (data.DATETIME >= start)]
        tdata_d = data[(data.DATETIME <= stop_d) & (data.DATETIME >= start_d)]
        tmanoeuvre = data[(data.DATETIME <= man_stop) & (data.DATETIME >= man_start)]
        build_data = data[(data.DATETIME > stop) & (data.DATETIME <= build)]
        turn_data = data[(data.DATETIME>=man_start) & (data.DATETIME<=man_stop)]
        end_stop = tman.shift(seconds=+25).format('YYYY-MM-DD HH:mm:ss')
        stop_data = data[(data.DATETIME > stop_d) & (data.DATETIME <= end_stop)]
        tdata_vmg = data[(data.DATETIME > start) & (data.DATETIME <= build)]
        accel_exit_data = data[(data.DATETIME > man_stop) & (data.DATETIME <= stop)]
        stop_entry = tman.shift(seconds=-5).format('YYYY-MM-DD HH:mm:ss')
        entry_data = tdata[(tdata.DATETIME <= stop_entry)]
        start_exit = tman.shift(seconds=+5).format('YYYY-MM-DD HH:mm:ss')
        exit_data = tdata[(tdata.DATETIME >= start_exit)]
        if np.sign(exit_data.TWA_MHU_SGP_deg).diff().sum()==0:
            if np.sign(entry_data.TWA_MHU_SGP_deg).diff().sum()==0:
        
        # Exit data
                start_exit = tman.shift(seconds=+5).format('YYYY-MM-DD HH:mm:ss')
                exit_data = tdata[(tdata.DATETIME >= start_exit)]
                entry_cog = entry_data.GPS_COG_deg.mean()
                exit_cog = exit_data.GPS_COG_deg.mean()
                # Distance via COG indicator : 
                delta_cog = np.abs(subtract_angles(entry_cog, exit_cog))
                if entry_data[f'TWA_{WAND}_SGP_deg'].abs().mean()<90:
                    twd_cog = mean_bearing(entry_cog, exit_cog)
                else:
                    twd_cog = mean_bearing(entry_cog, exit_cog) + 180

                twa_cog = twd_cog - tdata_vmg.GPS_COG_deg
                twa_cog = (twa_cog + 180) % 360 - 180
                vmg_cog = np.cos(twa_cog * np.pi / 180) * tdata_vmg.BOAT_SPEED_km_h_1 / 3.6
                distance_vmg_cog = np.abs(vmg_cog.sum())
                exit_ca1 = round(exit_data.ANGLE_CA1_deg.abs().mean(), 2)
                
                    
                
                boat = tdata.BOAT.iloc[0]
                if tmanoeuvre[f'TWA_{WAND}_SGP_deg'].abs().mean() >90:
                    type = 'gybe'
               
                else : 
                    type = 'tack'
                    
                dict = {
                            "Boat": boat,
                            "DATETIME": tman,
                            "type": type,
                            "distance_vmg_cog": distance_vmg_cog,
                            "exit_ca1": exit_ca1,
                }
                tack_summary = pd.concat([tack_summary, pd.DataFrame([dict])], ignore_index=True)
            tack_summary['man_id'] = np.arange(len(tack_summary))
            
            tack_summary=tack_summary[tack_summary.exit_ca1.abs()>12]
            #tack_summary = tack_summary[tack_summary.tws<100]
    return tack_summary
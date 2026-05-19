# Data Instructions

## Dataset

**Name:** UAH-DriveSet  
**Source:** University of Alcalá, Spain (Eduardo Romera)

## Download Instructions

1. Go to http://www.robesafe.uah.es/personal/eduardo.romera/uah-driveset/
2. Download the full dataset (ZIP file)
3. Unzip the file
4. Update `DATA_ROOT` in `config.py` to point to the `UAH-DriveSet` folder

## Dataset Description

| | |
|---|---|
| Drivers       | 6                                     |
| Vehicles      | 6 (one per driver)                    |
| Conditions    | Normal, Drowsy, Aggressive            |
| Road types    | Motorway, Secondary                   |
| Total trips   | 40                                    |
| Sampling rate | ~10 Hz                                |
| IMU channels  | acc_x, acc_y, acc_z, roll, pitch, yaw |

## Files Used

Each trip folder contains:
- `RAW_ACCELEROMETERS.csv` — raw IMU sensor data (used in this study)
- `SEMANTIC_FINAL.csv` — DriveSafe scores (loaded but not used)

## Reference

Romera, E., Bergasa, L.M., & Arroyo, R. (2016). Need data for driver behaviour 
analysis? Presenting the public UAH-DriveSet. *Proceedings of the 19th IEEE ITSC*, 
387–392. https://doi.org/10.1109/ITSC.2016.7795718
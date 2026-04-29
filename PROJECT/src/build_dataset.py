import os
import glob
import pandas as pd
import numpy as np

def build_features(data_dir):
    print("Loading datasets...")
    tt_files = glob.glob(os.path.join(data_dir, 'dynamic', 'timetables', '*.parquet'))
    tt_list = []
    for f in tt_files:
        df = pd.read_parquet(f, columns=['train_id', 'station_id', 'planned_arrival', 'planned_departure', 'actual_arrival', 'actual_departure'])
        df = df.drop_duplicates(subset=['train_id', 'station_id'], keep='last')
        tt_list.append(df)
    
    tt_df = pd.concat(tt_list, ignore_index=True)
    print(f"Timetables loaded: {len(tt_df)} records")

    # 1. Temporal bounds and null checks (data cleaning)
    tt_df['planned_arrival_dt'] = pd.to_datetime(tt_df['planned_arrival'], errors='coerce')
    tt_df['actual_arrival_dt'] = pd.to_datetime(tt_df['actual_arrival'], errors='coerce')

    # Compute raw target delay and bounded outliers (-60, 300 mins) based on EDA
    tt_df['target_delay'] = (tt_df['actual_arrival_dt'] - tt_df['planned_arrival_dt']).dt.total_seconds() / 60.0
    tt_df = tt_df[(tt_df['target_delay'] >= -60) & (tt_df['target_delay'] <= 300)].copy()

    # 2. Extract base temporal features
    tt_df['hour'] = tt_df['planned_arrival_dt'].dt.hour
    tt_df['day_of_week'] = tt_df['planned_arrival_dt'].dt.day
    tt_df['hour_trunc'] = tt_df['planned_arrival_dt'].dt.floor('h')

    # 3. Create seq/lag features for the previous station delay
    # sort chronologically to get valid sequence per train journey
    tt_df = tt_df.sort_values(by=['train_id', 'planned_arrival_dt'])
    tt_df['prev_station_delay'] = tt_df.groupby('train_id')['target_delay'].shift(1).fillna(0)  # fill first station delay with 0

    # 4. Integrate line maps -> TODO


if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(current_dir, '..', 'data')
    build_features(data_dir)

